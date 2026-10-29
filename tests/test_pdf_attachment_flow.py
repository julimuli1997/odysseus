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


LONG_ANSWER = "## Fix Windows 0xc0000001\n\n" + "1. Boot from the USB stick and open Command Prompt.\n" * 12


def test_fallback_uses_a_substantive_reply_as_the_pdf_content():
    from src.agent_loop import _pdf_fallback_content

    assert _pdf_fallback_content(LONG_ANSWER) == LONG_ANSWER.strip()
    thinking = "<think>plan the answer</think>\n" + LONG_ANSWER
    assert _pdf_fallback_content(thinking) == LONG_ANSWER.strip()


def test_fallback_defers_to_the_previous_answer_for_short_or_question_replies():
    from src.agent_loop import _pdf_fallback_content

    assert _pdf_fallback_content("Sure — which text should go in the PDF?") == ""
    question = "To help you convert content to a PDF, I need to know what text you want. " * 3 + "Which one?"
    assert _pdf_fallback_content(question) == ""
    assert _pdf_fallback_content("") == ""


def test_fallback_title_comes_from_the_first_heading_or_sentence():
    from src.agent_loop import _pdf_fallback_title

    assert _pdf_fallback_title(LONG_ANSWER) == "Fix Windows 0xc0000001"
    assert _pdf_fallback_title("Rats are clever. They learn mazes fast.") == "Rats are clever"
    assert _pdf_fallback_title("") == "Document"
    assert len(_pdf_fallback_title("word " * 60)) <= 80


def test_fallback_runs_only_for_untainted_pdf_turns_without_a_created_pdf():
    from src.agent_loop import _should_run_pdf_fallback
    from types import SimpleNamespace

    pdf_turn = SimpleNamespace(capabilities=frozenset({"pdf_export"}))
    other_turn = SimpleNamespace(capabilities=frozenset({"search_browser"}))
    ok = dict(pdf_attached=False, tainted=False, awaiting_user=False, owner="o", session_id="s")

    assert _should_run_pdf_fallback(pdf_turn, **ok)
    assert not _should_run_pdf_fallback(other_turn, **ok)
    assert not _should_run_pdf_fallback(None, **ok)
    assert not _should_run_pdf_fallback(pdf_turn, **dict(ok, pdf_attached=True))
    assert not _should_run_pdf_fallback(pdf_turn, **dict(ok, tainted=True))
    assert not _should_run_pdf_fallback(pdf_turn, **dict(ok, awaiting_user=True))
    assert not _should_run_pdf_fallback(pdf_turn, **dict(ok, session_id=None))


def test_stream_agent_loop_wires_the_fallback_at_completion():
    import inspect
    from src.agent_loop import stream_agent_loop

    source = inspect.getsource(stream_agent_loop)
    assert "_should_run_pdf_fallback(" in source
    assert "_pdf_attached = True" in source


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
