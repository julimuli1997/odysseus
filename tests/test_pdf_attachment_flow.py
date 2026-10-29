"""create_pdf attachment: tool result → stream event → saved assistant metadata."""

import json
from types import SimpleNamespace

from src.attachment_refs import attachment_refs_from_metadata

ATTACHMENT = {"id": "a" * 32 + ".pdf", "name": "Boot_fix.pdf", "mime": "application/pdf", "size": 3100}


def _event_payload(chunk):
    assert chunk.startswith("data: ") and chunk.endswith("\n\n")
    return json.loads(chunk[len("data: "):])


def test_successful_create_pdf_emits_an_attachments_event():
    from src.agent_loop import _pdf_attachment_event

    chunk = _pdf_attachment_event("create_pdf", {"output": "ok", "exit_code": 0, "attachment": ATTACHMENT})

    assert _event_payload(chunk) == {"type": "attachments", "data": [ATTACHMENT]}


def test_no_event_for_failures_other_tools_or_malformed_refs():
    from src.agent_loop import _pdf_attachment_event

    assert _pdf_attachment_event("create_pdf", {"error": "x", "exit_code": 1, "attachment": ATTACHMENT}) is None
    assert _pdf_attachment_event("web_search", {"output": "x", "exit_code": 0, "attachment": ATTACHMENT}) is None
    assert _pdf_attachment_event("create_pdf", {"output": "x", "exit_code": 0}) is None
    bad = dict(ATTACHMENT, id="../../etc/passwd")
    assert _pdf_attachment_event("create_pdf", {"output": "x", "exit_code": 0, "attachment": bad}) is None


def test_both_tool_execution_paths_emit_the_attachment_event():
    """Round-loop calls and approved calls (run before round 1) both attach the file."""
    import inspect
    from src.agent_loop import stream_agent_loop

    source = inspect.getsource(stream_agent_loop)
    assert "_pdf_attachment_event(block.tool_type, result)" in source
    assert "_pdf_attachment_event(approved.tool_name, approved_result)" in source


def test_saved_assistant_message_carries_the_attachment(monkeypatch):
    import core.database
    import routes.chat_helpers as chat_helpers

    monkeypatch.setattr(core.database, "update_session_last_accessed", lambda _sid: None)
    monkeypatch.setattr(chat_helpers, "_append_sft_trace_record", lambda **_kw: None)
    sess = SimpleNamespace(model="qwen2.5", history=[], add_message=None)
    sess.add_message = sess.history.append
    session_manager = SimpleNamespace(save_sessions=lambda: None)

    chat_helpers.save_assistant_response(
        sess, session_manager, "s1", "Your PDF is ready.", {"model": "qwen2.5"},
        attachments=[ATTACHMENT],
    )

    metadata = sess.history[-1].metadata
    assert metadata["attachments"] == [ATTACHMENT]
    # The upload cleanup reads the same refs, so the PDF is kept while the message exists.
    assert [ref["attachment_id"] for ref in attachment_refs_from_metadata(metadata)] == [ATTACHMENT["id"]]


def test_messages_without_attachments_keep_metadata_unchanged(monkeypatch):
    import core.database
    import routes.chat_helpers as chat_helpers

    monkeypatch.setattr(core.database, "update_session_last_accessed", lambda _sid: None)
    monkeypatch.setattr(chat_helpers, "_append_sft_trace_record", lambda **_kw: None)
    sess = SimpleNamespace(model="qwen2.5", history=[])
    sess.add_message = sess.history.append

    chat_helpers.save_assistant_response(
        sess, SimpleNamespace(save_sessions=lambda: None), "s1", "hi", {"model": "qwen2.5"},
    )

    assert "attachments" not in sess.history[-1].metadata
