"""create_pdf: turn chat content into a downloadable PDF for the requesting user.

The model only chooses the title and the Markdown; ``src.pdf_export`` owns the
layout. Small local models often call the tool with no content after "make a
pdf of that", so empty content falls back to the previous assistant answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

from src.ai_interaction import get_session_manager
from src.pdf_export import PdfExportError, render_markdown_pdf
from src.tool_utils import get_upload_handler
from src.upload_handler import secure_filename

logger = logging.getLogger(__name__)

PDF_MIME = "application/pdf"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            str(part.get("text") or "") for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    return ""


def _previous_assistant_answer(session_id: Optional[str]) -> str:
    """The last saved assistant message; the reply in progress is not saved yet."""
    if not session_id:
        return ""
    try:
        history = get_session_manager().get_session(session_id).history
    except Exception:
        return ""
    for message in reversed(list(history or ())):
        if getattr(message, "role", None) == "assistant":
            text = _message_text(getattr(message, "content", "")).strip()
            if text:
                return text
    return ""


def _parse_args(content: str) -> dict:
    raw = (content or "").strip()
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return {"content_markdown": raw}


def _pdf_filename(requested: Any, title: str) -> str:
    stem = os.path.basename(str(requested or "").strip()) or title or "document"
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    return f"{secure_filename(stem)[:80] or 'document'}.pdf"


class CreatePdfTool:
    async def execute(self, content: str, ctx: dict) -> dict:
        args = _parse_args(content)
        title = " ".join(str(args.get("title") or "").split())
        markdown = str(args.get("content_markdown") or args.get("content") or "").strip()
        session_id = ctx.get("session_id") if isinstance(ctx, dict) else None
        owner = ctx.get("owner") if isinstance(ctx, dict) else None
        if not markdown:
            markdown = _previous_assistant_answer(session_id)
        if not markdown:
            return {
                "error": "create_pdf: no content was given and there is no earlier answer to use.",
                "exit_code": 1,
            }

        handler = get_upload_handler()
        if handler is None:
            return {"error": "create_pdf: file storage is not available.", "exit_code": 1}

        filename = _pdf_filename(args.get("filename"), title)
        try:
            data = await asyncio.to_thread(render_markdown_pdf, title, markdown)
            saved = await asyncio.to_thread(handler.save_generated_file, data, filename, PDF_MIME, owner)
        except (PdfExportError, ValueError) as exc:
            return {"error": f"create_pdf: {exc}", "exit_code": 1}

        attachment = {
            "id": saved["id"],
            "name": saved["name"],
            "mime": saved["mime"],
            "size": saved["size"],
        }
        logger.info("create_pdf stored %s (%d bytes) for session %s", saved["name"], saved["size"], session_id)
        return {
            "output": (
                f"Created the PDF {saved['name']} ({saved['size'] // 1024 or 1} KB). "
                "It is attached to this reply as a download; tell the user it is ready."
            ),
            "exit_code": 0,
            "attachment": attachment,
        }
