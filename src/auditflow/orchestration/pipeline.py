"""
Connects retrieval, generation, and verification.
"""
from __future__ import annotations

import os

from src.auditflow.retrieval.retrieve import get_verification_context, get_all_chunks
from src.auditflow.verification.verify import verify_all_claims, build_chunk_lookup
from src.auditflow.orchestration.retry_layer import generate_with_bounded_retry
from src.auditflow.ingest.store import document_store
from schemas.errors import GenerationError, VerificationError
from schemas.generation import GenerationResult
from schemas.session import AskResponse
from schemas.verification import VerifiedClaim
from core.logging_config import logger, timed
from src.auditflow.orchestration.session_store import SessionState

if os.environ.get("USE_LOCAL_GENERATION", "true").lower() == "true":
    from src.auditflow.generation.generate import generate_answer
else:
    from src.auditflow.generation.generate_groq import generate_answer_groq as generate_answer


def answer_with_retry(question: str, document_id: str) -> GenerationResult | None:
    """Thin wrapper binding the retry layer to this pipeline's functions."""
    return generate_with_bounded_retry(
        question, document_id,
        generate_fn=generate_answer,
        full_context_fn=get_verification_context,
    )


def get_full_chunk_lookup() -> dict:
    return build_chunk_lookup(get_all_chunks())


def get_all_document_identities() -> dict:
    """document_id -> {company_name, counterparty_name, document_title}
    for every document in the corpus. Sourced from Postgres directly
    (document_store.list_documents()), NOT from chunk metadata"""
    identities = {}
    for doc in document_store.list_documents():
        identities[doc.document_id] = {
            "company_name": doc.company_name or "",
            "counterparty_name": doc.counterparty_name or "",
            "document_title": doc.document_title or "",
        }
    return identities


def _log_claims(claims: list[VerifiedClaim]) -> None:
    for claim in claims:
        logger.info("%s %s | verdict=%s reason=%s", claim.badge, claim.text, claim.verdict, claim.reason)


def _verify(claims: list, chunk_lookup: dict) -> list[VerifiedClaim]:
    try:
        return verify_all_claims(claims, chunk_lookup)
    except Exception as exc:  # noqa: BLE001 -- narrow boundary to verify.py
        raise VerificationError("Claim verification failed") from exc


class Sessions:
    """Stateless with respect to conversation memory -- active_contract,
    pending_question, and pending_candidates all live on the SessionState
    passed into ask(), """

    def __init__(self) -> None:
        self.chunk_lookup = get_full_chunk_lookup()
        self.all_identities = get_all_document_identities()

    def _mentions_different_contract(self, question: str, active_contract: str | None) -> str | None:
        """Checks if the question names a document (by either party's
        name) different from the active one, via real company_name /
        counterparty_name fields from identity extraction."""
        question_lower = question.lower()
        question_words = set(question_lower.replace("_", " ").replace("-", " ").split())

        best_match = None
        best_match_len = 0

        for doc_id, identity in self.all_identities.items():
            if doc_id == active_contract:
                continue
            for name in (identity["company_name"], identity["counterparty_name"]):
                if not name:
                    continue
                name_lower = name.lower()
                if name_lower in question_lower:
                    return doc_id
                for word in name_lower.replace(",", "").replace(".", "").split():
                    if len(word) >= 4 and word in question_words and len(word) > best_match_len:
                        best_match = doc_id
                        best_match_len = len(word)
        return best_match

    @staticmethod
    def _looks_like_contract_selection(text: str) -> bool:
        text_clean = text.strip().rstrip("?").lower()
        question_words = {"what", "who", "when", "where", "why", "how", "does",
                           "is", "are", "which", "can", "could", "would"}
        first_word = text_clean.split()[0] if text_clean.split() else ""
        return first_word not in question_words and not text.strip().endswith("?")

    @timed("ask")
    def ask(self, question: str, state: SessionState) -> AskResponse:
        logger.info("Question: %s", question)

        if state.pending_question and self._looks_like_contract_selection(question):
            return self._resolve_pending_selection(question, state)

        context = timed("retrieval + rerank")(get_verification_context)(question)
        consistency = context.consistency

        if consistency.is_confident:
            state.active_contract = consistency.top_contract
            state.pending_question = None
            return self._generate_and_verify(question, consistency, context.chunks)

        return self._handle_low_confidence(question, consistency, state)

    def _resolve_pending_selection(self, question: str, state: SessionState) -> AskResponse:
        resolved_question = state.pending_question
        chosen_document_id = question.strip()

        if chosen_document_id not in state.pending_candidates:
            logger.warning("Selection %r not among offered candidates %s", chosen_document_id, state.pending_candidates)
            return AskResponse.need_clarification(state.pending_candidates)  # ask again, don't guess

        state.pending_question = None
        state.pending_candidates = []

        result = answer_with_retry(resolved_question, chosen_document_id)
        if result is None:
            return AskResponse.low_relevance(chosen_document_id)

        state.active_contract = chosen_document_id
        verified_claims = _verify(result.claims, self.chunk_lookup)
        _log_claims(verified_claims)
        return AskResponse.answered(verified_claims, chosen_document_id)

    def _handle_low_confidence(self, question: str, consistency, state: SessionState) -> AskResponse:
        logger.debug(
            "Low confidence on fresh retrieval (concentration=%.2f, avg_score=%.3f)",
            consistency.concentration, consistency.avg_top_score,
        )
        mentioned = self._mentions_different_contract(question, state.active_contract)

        if mentioned and mentioned != state.active_contract:
            logger.debug("Question names a different document: %s - dropping stale memory", mentioned)
            state.active_contract = None
            result = answer_with_retry(question, mentioned)
            if result is not None:
                state.active_contract = mentioned
                verified_claims = _verify(result.claims, self.chunk_lookup)
                _log_claims(verified_claims)
                return AskResponse.answered(verified_claims, mentioned)

        if state.active_contract:
            logger.debug("Retrying scoped to active document: %s", state.active_contract)
            result = answer_with_retry(question, state.active_contract)
            if result is not None:
                verified_claims = _verify(result.claims, self.chunk_lookup)
                _log_claims(verified_claims)
                return AskResponse.answered(verified_claims, state.active_contract)
            logger.debug("Fallback also failed - falling through to clarification")

        concentration_threshold = 0.6
        if consistency.concentration < concentration_threshold:
            state.pending_question = question
            candidates = list(consistency.contract_breakdown.keys())
            state.pending_candidates = candidates
            logger.info("Multiple documents match - asking for clarification: %s", candidates)
            return AskResponse.need_clarification(candidates)

        state.active_contract = consistency.top_contract
        logger.info("Found %s, but not confident it answers the question.", consistency.top_contract)
        return AskResponse.low_relevance(consistency.top_contract)

    def _generate_and_verify(self, question: str, consistency, chunks) -> AskResponse:
        logger.info(
            "[CONFIDENT] top_document=%s concentration=%.2f avg_score=%.3f",
            consistency.top_contract, consistency.concentration, consistency.avg_top_score,
        )
        top_doc = consistency.top_contract
        scoped_chunks = [c for c in chunks if c.document_id == top_doc][:5]

        try:
            result: GenerationResult = generate_answer(question, scoped_chunks)
        except Exception as exc:  # noqa: BLE001 -- narrow boundary to generate.py
            raise GenerationError(f"Generation failed for document {top_doc}") from exc

        verified_claims = _verify(result.claims, self.chunk_lookup)
        _log_claims(verified_claims)
        return AskResponse.answered(verified_claims, top_doc)