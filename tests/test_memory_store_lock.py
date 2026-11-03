"""memory.json read/modify/write cycles must not overwrite each other.

Every writer loads the whole store, changes it, and saves the whole store
back. Without a shared lock, two writers that overlap each save the copy they
loaded, and the later save silently drops the earlier one's change.
"""
import json
import multiprocessing
import threading

import pytest

from core.atomic_io import store_lock
from src.memory import MemoryManager


def _append(manager, text):
    with manager.transaction():
        entries = manager.load_all_for_update()
        entries.append(manager.add_entry(text, owner="alice"))
        manager.save(entries)


def test_overlapping_writers_keep_both_changes(tmp_path):
    manager = MemoryManager(str(tmp_path))
    first_loaded = threading.Event()
    release_first = threading.Event()

    def slow_writer():
        with manager.transaction():
            entries = manager.load_all_for_update()
            first_loaded.set()
            release_first.wait(5)
            entries.append(manager.add_entry("first", owner="alice"))
            manager.save(entries)

    slow = threading.Thread(target=slow_writer)
    slow.start()
    assert first_loaded.wait(5)

    fast = threading.Thread(target=_append, args=(manager, "second"))
    fast.start()
    fast.join(0.3)
    # The second writer must wait for the first one's save instead of
    # loading the same snapshot.
    assert fast.is_alive()

    release_first.set()
    slow.join(5)
    fast.join(5)

    texts = sorted(e["text"] for e in manager.load_all())
    assert texts == ["first", "second"]


def test_many_concurrent_writers_lose_nothing(tmp_path):
    manager = MemoryManager(str(tmp_path))
    threads = [threading.Thread(target=_append, args=(manager, f"m{i}")) for i in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert len(manager.load_all()) == 25
    assert not list(tmp_path.glob("memory.json.tmp*"))


def test_lock_is_reentrant_within_a_thread(tmp_path):
    manager = MemoryManager(str(tmp_path))
    done = threading.Event()

    def nested():
        with manager.transaction():
            with manager.transaction():
                manager.increment_uses(["missing"])  # locks the store again
                manager.save([manager.add_entry("x")])
        done.set()

    worker = threading.Thread(target=nested, daemon=True)
    worker.start()
    assert done.wait(5), "nested store lock deadlocked"


def _hold_lock(path, locked, release):
    with store_lock(path):
        locked.set()
        release.wait(10)


def test_lock_blocks_other_processes(tmp_path):
    """The memory MCP server is a separate process writing the same file."""
    manager = MemoryManager(str(tmp_path))
    ctx = multiprocessing.get_context("spawn")
    locked, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_lock, args=(manager.memory_file, locked, release))
    holder.start()
    try:
        assert locked.wait(30)
        writer = threading.Thread(target=_append, args=(manager, "after"))
        writer.start()
        writer.join(0.3)
        assert writer.is_alive()
        release.set()
        writer.join(10)
        assert not writer.is_alive()
    finally:
        release.set()
        holder.join(10)
    assert [e["text"] for e in manager.load_all()] == ["after"]


def test_save_keeps_non_ascii_text_readable(tmp_path):
    manager = MemoryManager(str(tmp_path))
    manager.save([manager.add_entry("Grüße aus Köln")])
    assert "Grüße aus Köln" in (tmp_path / "memory.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_audit_keeps_memories_saved_while_the_llm_ran(tmp_path, monkeypatch):
    """The audit snapshots the owner's memories, waits 30-120s on the LLM, then
    rewrites the store. A memory saved during that wait was never shown to the
    LLM and must survive the rewrite."""
    import src.llm_core
    import src.utility_effort
    from services.memory import memory_extractor

    manager = MemoryManager(str(tmp_path))
    old = [manager.add_entry(f"fact number {i}", owner="alice") for i in range(3)]
    other = manager.add_entry("bob's fact", owner="bob")
    manager.save(old + [other])

    async def fake_llm(*_args, **_kwargs):
        _append(manager, "added during audit")
        return json.dumps([{"id": e["id"], "text": e["text"]} for e in old[:2]])

    monkeypatch.setattr(src.llm_core, "llm_call_async", fake_llm)
    monkeypatch.setattr(src.utility_effort, "effort_for_call", lambda *_a, **_k: None)

    result = await memory_extractor.audit_memories(
        manager, None, "http://model.test/v1", "m", owner="alice"
    )

    assert result.get("error") is None
    texts = sorted(e["text"] for e in manager.load_all())
    assert texts == sorted([
        "fact number 0", "fact number 1", "bob's fact", "added during audit",
    ])
