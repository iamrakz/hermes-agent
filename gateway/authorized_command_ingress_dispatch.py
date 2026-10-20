"""Validation and handler phases for generic authorized-command dispatch."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Optional, cast

from gateway.authorized_command_ingress import (
    AUTHORIZED_COMMAND_INGRESS_API_VERSION,
    NOT_CLAIMED,
    AuthorizedCommandDisposition,
    AuthorizedCommandIngressConfig,
    AuthorizedCommandRequest,
    AuthorizedCommandResult,
    AuthorizedInboundContext,
    InboundTransportEvidence,
    _Candidate,
    _HANDLERS,
    _MAX_RAW_TEXT_CHARS,
    _bounded_id,
    _cache_terminal,
    _candidate_preflight,
    _command_entity,
    _enum_value,
    _log_terminal,
    _require_authorized_context,
    _terminal_failure,
    _terminal_rejection,
)


logger = logging.getLogger(__name__)


class _PendingDispatch:
    def __init__(self, future: asyncio.Future) -> None:
        self.future = future


def _evidence_failure(
    context: AuthorizedInboundContext,
    evidence: InboundTransportEvidence,
    candidate: _Candidate,
    config: AuthorizedCommandIngressConfig,
) -> Optional[str]:
    if context.source_is_bot or evidence.sender_is_bot:
        return "bot_sender"
    if candidate.syntax_failure:
        return candidate.syntax_failure
    if len(evidence.original_text) > _MAX_RAW_TEXT_CHARS:
        return "raw_text_too_long"
    if context.source_user_id is None or context.source_chat_id is None:
        return "missing_source_identity"
    raw_user_id = _bounded_id(evidence.sender_user_id)
    raw_chat_id = _bounded_id(evidence.chat_id)
    if raw_user_id is None or raw_chat_id is None:
        return "missing_raw_identity"
    if raw_user_id != context.source_user_id or raw_chat_id != context.source_chat_id:
        return "identity_mismatch"
    if evidence.forwarded:
        return "forwarded_message"
    if evidence.via_bot:
        return "via_bot"
    if evidence.sender_chat:
        return "sender_chat"
    if config.dm_only and (
        context.source_chat_type not in {"dm", "private"}
        or _enum_value(evidence.chat_type) not in {"dm", "private"}
    ):
        return "non_dm_context"
    return None


def _reject(
    event,
    config: AuthorizedCommandIngressConfig,
    candidate: _Candidate,
    category: str,
) -> AuthorizedCommandResult:
    result = _cache_terminal(event, _terminal_rejection(category))
    _log_terminal(config, candidate, result)
    return result


def _prepare_request(
    context: AuthorizedInboundContext,
    config: AuthorizedCommandIngressConfig,
) -> (
    AuthorizedCommandResult
    | _PendingDispatch
    | tuple[object, _Candidate, AuthorizedCommandRequest]
):
    event = context._event
    cached = getattr(event, "_authorized_command_ingress_result", None)
    if isinstance(cached, AuthorizedCommandResult):
        return cached
    candidate, evidence, early = _candidate_preflight(
        event,
        context.transport_evidence,
        config,
        source_transport=context.source_transport,
        normalized_text=context.normalized_text,
        transport_constituent_count=context.transport_constituent_count,
    )
    if early is not None:
        return early
    if candidate is None:
        return NOT_CLAIMED
    if evidence is None:
        return _reject(event, config, candidate, "raw_context_unavailable")
    failure = _evidence_failure(context, evidence, candidate, config)
    if failure:
        return _reject(event, config, candidate, failure)
    entity_offset, entity_length, entity_failure = _command_entity(evidence, candidate)
    if entity_failure:
        return _reject(event, config, candidate, entity_failure)
    request = AuthorizedCommandRequest(
        api_version=AUTHORIZED_COMMAND_INGRESS_API_VERSION,
        transport=config.transport,
        command_name=candidate.command_name,
        raw_text=evidence.original_text,
        raw_command_token=candidate.raw_command_token,
        raw_argument_tail=candidate.raw_argument_tail,
        command_entity_offset=entity_offset,
        command_entity_length=entity_length,
        user_id=cast(str, context.source_user_id),
        chat_id=cast(str, context.source_chat_id),
        chat_type=context.source_chat_type,
        message_id=context.message_id,
    )
    future = asyncio.get_running_loop().create_future()
    claimed, pending = event._begin_authorized_command_ingress(future)
    if not claimed:
        cached = getattr(event, "_authorized_command_ingress_result", None)
        if isinstance(cached, AuthorizedCommandResult):
            return cached
        if isinstance(pending, asyncio.Future):
            return _PendingDispatch(pending)
        return _terminal_failure("duplicate_dispatch")
    return event, candidate, request


async def _invoke_handler(
    event,
    candidate: _Candidate,
    request: AuthorizedCommandRequest,
    config: AuthorizedCommandIngressConfig,
) -> AuthorizedCommandResult:
    registration = _HANDLERS.get(config.handler_id)
    if registration is None:
        result = _cache_terminal(event, _terminal_failure("handler_missing"))
        _log_terminal(config, candidate, result)
        return result
    if frozenset(registration.reserved_commands) != frozenset(config.reserved_commands):
        result = _cache_terminal(
            event, _terminal_failure("handler_reservation_mismatch")
        )
        _log_terminal(config, candidate, result)
        return result

    try:
        handler_result = registration.handler(request)
        if inspect.isawaitable(handler_result):
            handler_result = await handler_result
    except asyncio.CancelledError:
        result = _cache_terminal(event, _terminal_failure("cancelled"))
        _log_terminal(config, candidate, result)
        raise
    except Exception as exc:
        sanitized = RuntimeError(f"handler raised {type(exc).__name__}")
        logger.error(
            "authorized-command handler failed transport=%s command=%s "
            "handler=%s category=handler_exception",
            config.transport,
            candidate.command_name,
            config.handler_id,
            exc_info=(RuntimeError, sanitized, exc.__traceback__),
        )
        result = _cache_terminal(event, _terminal_failure("handler_exception"))
        _log_terminal(config, candidate, result)
        return result

    if not isinstance(handler_result, AuthorizedCommandResult):
        result = _cache_terminal(event, _terminal_failure("handler_contract"))
    elif handler_result.disposition is AuthorizedCommandDisposition.NOT_CLAIMED:
        result = _cache_terminal(event, _terminal_failure("handler_not_claimed"))
    else:
        result = _cache_terminal(event, handler_result)
    _log_terminal(config, candidate, result)
    return result


async def dispatch_authorized_context(
    context: AuthorizedInboundContext,
    config: AuthorizedCommandIngressConfig,
) -> AuthorizedCommandResult:
    context = _require_authorized_context(context)
    prepared = _prepare_request(context, config)
    if isinstance(prepared, AuthorizedCommandResult):
        return prepared
    if isinstance(prepared, _PendingDispatch):
        return await asyncio.shield(prepared.future)
    event, candidate, request = prepared
    return await _invoke_handler(event, candidate, request, config)


__all__ = ["dispatch_authorized_context"]
