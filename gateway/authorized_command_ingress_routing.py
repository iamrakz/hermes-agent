"""Narrow cold/busy wiring for the authorized-command ingress boundary."""

from __future__ import annotations

import logging
from typing import Any

from gateway.authorized_command_ingress import (
    NOT_CLAIMED,
    AuthorizedInboundContext,
    AuthorizedCommandResult,
    contain_unauthorized_reserved_command,
    dispatch_authorized_command,
)


logger = logging.getLogger("gateway.run")


def _config(runner: Any) -> Any:
    return getattr(getattr(runner, "config", None), "authorized_command_ingress", None)


async def _dispatch_authorized_context(
    runner: Any, context: AuthorizedInboundContext
) -> AuthorizedCommandResult:
    config = _config(runner)
    if config is None:
        return NOT_CLAIMED
    return await dispatch_authorized_command(context, config)


async def route_cold_authorized_command(
    runner: Any, context: AuthorizedInboundContext
) -> AuthorizedCommandResult:
    """Dispatch only a capability already minted by the post-auth runner seam."""
    result = await _dispatch_authorized_context(runner, context)
    if not result.claimed or not result.response:
        return result
    await _deliver_response_once(runner, context._event, result)
    return AuthorizedCommandResult(
        result.disposition, failure_category=result.failure_category
    )


def contain_internal_authorized_command(
    runner: Any, event: Any
) -> AuthorizedCommandResult:
    """Contain a reserved internal event without creating execution authority."""
    config = _config(runner)
    if config is None:
        return NOT_CLAIMED
    return contain_unauthorized_reserved_command(
        event, config, failure_category="internal_event"
    )


def contain_busy_internal_authorized_command(runner: Any, event: Any) -> bool:
    return event.internal and contain_internal_authorized_command(runner, event).claimed


async def route_busy_authorized_command(
    runner: Any, context: AuthorizedInboundContext
) -> bool:
    """Dispatch after busy authorization and terminally deliver any claimed response."""
    result = await _dispatch_authorized_context(runner, context)
    if not result.claimed:
        return False
    event = context._event
    event._gateway_accepted = True
    if result.response:
        await _deliver_response_once(runner, event, result)
    return True


async def _deliver_response_once(
    runner: Any, event: Any, result: AuthorizedCommandResult
) -> None:
    if result.response is None:
        return
    if not event._claim_authorized_command_response_delivery():
        return
    adapter = runner._delivery_adapter_for(event.source)
    if adapter is None:
        return
    try:
        await runner._send_authorized_command_reply(event, adapter, result.response)
    except Exception as exc:
        sanitized = RuntimeError("authorized-command delivery failed")
        logger.warning(
            "authorized-command reply delivery failed "
            "platform=%s disposition=%s category=delivery_failed exception=%s",
            event.source.platform.value if event.source.platform else "unknown",
            result.disposition.value,
            type(exc).__name__,
            exc_info=(RuntimeError, sanitized, None),
        )


__all__ = [
    "contain_busy_internal_authorized_command",
    "contain_internal_authorized_command",
    "route_busy_authorized_command",
    "route_cold_authorized_command",
]
