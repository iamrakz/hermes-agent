"""Telegram-specific immutable evidence capture for authorized-command ingress."""

from __future__ import annotations

from typing import Any, Optional

from gateway.authorized_command_ingress import (
    CommandEntityEvidence,
    InboundTransportEvidence,
)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _bounded_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    rendered = str(value)
    return rendered if rendered and len(rendered) <= 256 else None


def _utf16_slice(text: str, offset: int, length: int) -> Optional[str]:
    if offset < 0 or length < 0:
        return None
    encoded = text.encode("utf-16-le")
    start, end = offset * 2, (offset + length) * 2
    if end > len(encoded):
        return None
    try:
        return encoded[start:end].decode("utf-16-le")
    except UnicodeDecodeError:
        return None


def snapshot_telegram_transport_evidence(
    message: Any, update_id: Optional[int]
) -> Optional[InboundTransportEvidence]:
    """Copy security-relevant PTB facts; no live PTB object escapes this function."""
    text = getattr(message, "text", None)
    entities = getattr(message, "entities", None)
    if not isinstance(text, str):
        text = getattr(message, "caption", None)
        entities = getattr(message, "caption_entities", None)
    if not isinstance(text, str):
        return None

    command_entities: list[CommandEntityEvidence] = []
    if isinstance(entities, (list, tuple)):
        for entity in entities:
            kind = _enum_value(getattr(entity, "type", None))
            offset = getattr(entity, "offset", None)
            length = getattr(entity, "length", None)
            if (
                isinstance(offset, bool)
                or isinstance(length, bool)
                or not isinstance(offset, int)
                or not isinstance(length, int)
            ):
                continue
            entity_text = _utf16_slice(text, offset, length)
            if entity_text is None:
                continue
            command_entities.append(
                CommandEntityEvidence(kind, offset, length, entity_text)
            )

    sender = getattr(message, "from_user", None)
    chat = getattr(message, "chat", None)
    forwarded = any(
        getattr(message, name, None) is not None
        for name in (
            "forward_origin",
            "forward_date",
            "forward_from",
            "forward_from_chat",
            "forward_sender_name",
        )
    )
    return InboundTransportEvidence(
        transport="telegram",
        original_text=text,
        command_entities=tuple(command_entities),
        sender_user_id=_bounded_id(getattr(sender, "id", None)),
        sender_is_bot=bool(getattr(sender, "is_bot", False)),
        chat_id=_bounded_id(getattr(chat, "id", None)),
        chat_type=_enum_value(getattr(chat, "type", None)) or None,
        message_id=_bounded_id(getattr(message, "message_id", None)),
        update_id=update_id
        if isinstance(update_id, int) and not isinstance(update_id, bool)
        else None,
        forwarded=forwarded,
        via_bot=getattr(message, "via_bot", None) is not None,
        sender_chat=getattr(message, "sender_chat", None) is not None,
    )


def attach_telegram_transport_evidence(
    message_event, message: Any, update_id: Optional[int]
):
    """Attach the adapter-owned snapshot and return the event for narrow wiring."""
    message_event._inbound_transport_evidence = snapshot_telegram_transport_evidence(
        message, update_id
    )
    return message_event


__all__ = [
    "attach_telegram_transport_evidence",
    "snapshot_telegram_transport_evidence",
]
