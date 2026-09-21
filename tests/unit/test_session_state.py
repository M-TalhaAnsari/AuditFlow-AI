"""
tests/unit/test_session_state.py

SessionStore.__init__ calls the real src.auditflow.orchestration.
redis_client.get_primary(), which resolves through Sentinel -- not
available in this sandbox (or in CI without a real Redis+Sentinel pair).
So we monkeypatch get_primary to return a fakeredis.FakeStrictRedis
instance instead: a real, protocol-compatible in-memory Redis
implementation (not a hand-rolled mock), so SET/GET/EXPIRE/LPUSH/LTRIM
and the lock() Lua-script-based implementation all behave like real
Redis would.
"""
import time

import fakeredis
import pytest

from src.auditflow.orchestration import session_store as session_store_module
from src.auditflow.orchestration.session_store import SessionState, SessionStore


@pytest.fixture
def fake_redis():
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def store(monkeypatch, fake_redis):
    monkeypatch.setattr(session_store_module, "get_primary", lambda db=0: fake_redis)
    return SessionStore()


# ---------------------------------------------------- SessionState

def test_session_state_defaults():
    state = SessionState()
    assert state.active_contract is None
    assert state.pending_question is None
    assert state.pending_candidates == []


def test_session_state_round_trips_through_json():
    original = SessionState(
        active_contract="doc-a", pending_question="which one?",
        pending_candidates=["doc-a", "doc-b"],
    )
    restored = SessionState.from_json(original.to_json())
    assert restored == original


def test_session_state_from_json_none_returns_defaults():
    """Redis GET on a missing key returns None -- from_json(None) must
    hand back a fresh default state, not raise."""
    restored = SessionState.from_json(None)
    assert restored == SessionState()


def test_session_state_pending_candidates_default_is_not_shared():
    """dataclass mutable-default footgun check: field(default_factory=list)
    is used (correctly) rather than a bare `= []`, so two independently
    constructed SessionStates must not share the same list object."""
    a = SessionState()
    b = SessionState()
    a.pending_candidates.append("doc-x")
    assert b.pending_candidates == []


# ---------------------------------------------------- SessionStore.get/set/clear

def test_store_get_missing_session_returns_default(store):
    state = store.get("alice")
    assert state == SessionState()


def test_store_set_then_get_round_trips(store):
    state = SessionState(active_contract="doc-a", pending_question="q?")
    store.set("alice", state, role="employee")
    assert store.get("alice") == state


def test_store_clear_removes_session(store):
    store.set("alice", SessionState(active_contract="doc-a"), role="employee")
    store.clear("alice")
    assert store.get("alice") == SessionState()


def test_store_uses_role_based_ttl(store, fake_redis):
    """viewer sessions expire in 600s, everyone else in 3600s (or the
    3600s SESSION_TTL_SECONDS default for an unrecognized role) --
    confirms ROLE_TTL_SECONDS is actually wired to the Redis EX option,
    not just defined and unused."""
    store.set("bob", SessionState(), role="viewer")
    ttl = fake_redis.ttl(store._key("bob"))
    assert 0 < ttl <= 600

    store.set("carol", SessionState(), role="employee")
    ttl = fake_redis.ttl(store._key("carol"))
    assert 600 < ttl <= 3600


def test_store_unknown_role_falls_back_to_default_ttl(store, fake_redis):
    store.set("dave", SessionState(), role="some-typo-role")
    ttl = fake_redis.ttl(store._key("dave"))
    assert 0 < ttl <= 3600


def test_sessions_for_different_users_are_isolated(store):
    store.set("alice", SessionState(active_contract="doc-a"), role="employee")
    store.set("bob", SessionState(active_contract="doc-b"), role="employee")
    assert store.get("alice").active_contract == "doc-a"
    assert store.get("bob").active_contract == "doc-b"


# ---------------------------------------------------- locked()

def test_locked_allows_the_block_to_run(store):
    ran = False
    with store.locked("alice"):
        ran = True
    assert ran is True


def test_locked_releases_on_normal_exit_so_it_can_be_reacquired(store):
    with store.locked("alice"):
        pass
    with store.locked("alice"):  # would hang/timeout if the first lock leaked
        pass


def test_locked_releases_on_exception_so_it_does_not_leak(store):
    with pytest.raises(ValueError):
        with store.locked("alice"):
            raise ValueError("boom")
    # if release() didn't run in a finally block, this would time out
    with store.locked("alice"):
        pass


def test_locked_blocks_a_second_concurrent_holder(store, fake_redis):
    """Confirms the lock actually excludes a second acquirer for the same
    username while held -- not just that acquire()/release() don't
    error. Uses a second SessionStore sharing the same fake_redis backend
    (mirroring two request threads sharing one real Redis)."""
    other_store = SessionStore.__new__(SessionStore)
    other_store._r = fake_redis

    with store.locked("alice"):
        second_lock = fake_redis.lock(
            "session-lock:alice", timeout=10, blocking_timeout=0.2,
        )
        acquired = second_lock.acquire()
        assert acquired is False  # blocked by the first holder


# ---------------------------------------------------- recent turns

def test_push_and_get_recent_turns_round_trip(store):
    store.push_recent_turn("alice", "What is the term?", "answered", "doc-a")
    turns = store.get_recent_turns("alice")
    assert turns == [{"question": "What is the term?", "status": "answered", "document_id": "doc-a"}]


def test_recent_turns_most_recent_first(store):
    store.push_recent_turn("alice", "q1", "answered", "doc-a")
    store.push_recent_turn("alice", "q2", "answered", "doc-a")
    turns = store.get_recent_turns("alice")
    assert [t["question"] for t in turns] == ["q2", "q1"]  # lpush -> newest at index 0


def test_recent_turns_capped_at_limit(store):
    for i in range(session_store_module.RECENT_TURNS_LIMIT + 3):
        store.push_recent_turn("alice", f"q{i}", "answered", "doc-a")
    turns = store.get_recent_turns("alice")
    assert len(turns) == session_store_module.RECENT_TURNS_LIMIT
    # the 3 oldest should have been trimmed off
    assert turns[-1]["question"] == f"q{3}"