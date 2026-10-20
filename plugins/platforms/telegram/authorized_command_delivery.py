"""One-attempt Telegram delivery for authorized-command responses."""

from __future__ import annotations

from typing import Any, Optional

from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id


async def send_authorized_command_response_once(
    adapter: Any,
    *,
    chat_id: str,
    content: str,
    reply_to: Optional[str],
    metadata: Any,
) -> None:
    """Make at most one Bot API request, with no formatting fallback or retry."""
    bot = getattr(adapter, "_bot", None)
    if bot is None:
        raise RuntimeError("authorized command delivery unavailable")
    routing = adapter._compute_single_send_routing(
        chat_id, reply_to, metadata, adapter._metadata_thread_id(metadata)
    )
    if routing is None:
        raise RuntimeError("authorized command delivery unavailable")
    reply_to_id, thread_kwargs = routing
    await bot.send_message(
        chat_id=normalize_telegram_chat_id(chat_id),
        text=content,
        parse_mode=None,
        reply_to_message_id=reply_to_id,
        **thread_kwargs,
        **adapter._link_preview_kwargs(),
        **adapter._notification_kwargs(metadata),
    )


__all__ = ["send_authorized_command_response_once"]
