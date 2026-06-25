"""
Step 7b: Use an LLM to draft natural-language rephrasings of your
eval candidates. This does NOT replace your judgment - you still need
to review every single one afterward, since the LLM can misread which
ground_truth_answer is actually correct. Treat this as a fast first
draft, not a final answer.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

REPHRASE_PROMPT = """You are helping build an evaluation set for a legal RAG system.

Below is a clause-extraction question (written in a stiff, formal style) and
the real, lawyer-verified answer text extracted from a contract.

Clause category: {category}
Original question: {cuad_question}
Ground truth answer text(s): {answers}
Is this clause actually present in the contract: {is_impossible_text}

Task: Rewrite this as a natural question a normal person (not a lawyer)
would type into a chatbot when asking about THIS SPECIFIC contract.
Then write a short, natural one-sentence version of the correct answer,
in your own words, based on the ground truth text.

If "Is this clause actually present" is False, the correct answer should
be a natural sentence saying this contract does not contain that clause.

Respond with ONLY valid JSON, no other text, no markdown fences:
{{"natural_question": "...", "natural_answer": "..."}}"""


def rephrase_one(candidate: dict) -> dict:
    is_impossible_text = "False (clause IS present)" if not candidate["is_impossible"] else "True (clause is NOT present)"
    answers_text = "; ".join(candidate["ground_truth_answer"][:2]) if candidate["ground_truth_answer"] else "N/A"

    prompt = REPHRASE_PROMPT.format(
        category=candidate["category"],
        cuad_question=candidate["cuad_question"],
        answers=answers_text,
        is_impossible_text=is_impossible_text,
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    parsed = json.loads(raw)

    return {
        **candidate,
        "natural_question": parsed["natural_question"],
        "natural_answer": parsed["natural_answer"],
        "reviewed_by_human": False,  # flag for you to flip to True once you've checked it
    }


if __name__ == "__main__":
    with open("eval/eval_candidates.json", "r", encoding="utf-8") as f:
        candidates = json.load(f)

    all_rephrased = []
    for category_name, items in candidates.items():
        print(f"\nRephrasing {len(items)} '{category_name}' candidates...")
        for i, item in enumerate(items):
            try:
                rephrased = rephrase_one(item)
                all_rephrased.append(rephrased)
                print(f"  [{i+1}/{len(items)}] {rephrased['natural_question']}")
            except Exception as e:
                print(f"  [{i+1}/{len(items)}] FAILED: {e}")

    with open("eval/eval_set_draft.json", "w", encoding="utf-8") as f:
        json.dump(all_rephrased, f, indent=2)

    print(f"\nSaved {len(all_rephrased)} draft eval questions to eval/eval_set_draft.json")
    print("IMPORTANT: review every entry and set reviewed_by_human=True before using this for real eval.")