"""Post-authorization, fail-closed ingress for explicitly reserved commands.

This module is deliberately independent of slash commands, plugins, skills, tools, and
the agent loop.  A repository-owned integration registers a handler explicitly; the
gateway calls :func:`dispatch_authorized_command` only after its normal authorization
boundary.  The default configuration is disabled and no handlers are registered here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Optional

logger = logging.getLogger(__name__)

AUTHORIZED_COMMAND_INGRESS_API_VERSION = 1
SUPPORTED_AUTHORIZED_COMMAND_TRANSPORTS = frozenset({"telegram"})

_COMMAND_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_HANDLER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_FAILURE_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_COMMAND_NAME_CONTINUATION = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)
_MAX_RESERVED_COMMANDS = 32
_MAX_RAW_TEXT_CHARS = 8192
_MAX_ID_CHARS = 256
_MAX_RESPONSE_CHARS = 16384

_REJECTED_RESPONSE = "This protected command could not be accepted."
_FAILED_RESPONSE = "This protected command is temporarily unavailable."


@dataclass(frozen=True, slots=True)
class CommandEntityEvidence:
    """Immutable transport snapshot of one command entity."""

    kind: str
    offset_utf16: int
    length_utf16: int
    text: str


@dataclass(frozen=True, slots=True)
class InboundTransportEvidence:
    """Hermes-owned immutable facts captured before transport objects can change."""

    transport: str
    original_text: str
    command_entities: tuple[CommandEntityEvidence, ...]
    sender_user_id: Optional[str]
    sender_is_bot: bool
    chat_id: Optional[str]
    chat_type: Optional[str]
    message_id: Optional[str]
    update_id: Optional[int]
    forwarded: bool
    via_bot: bool
    sender_chat: bool


class AuthorizedCommandConfigError(ValueError):
    """The authorized-command ingress configuration is unsafe or unsupported."""


class AuthorizedCommandDisposition(str, Enum):
    """Only ``NOT_CLAIMED`` permits the caller to continue normal dispatch."""

    NOT_CLAIMED = "not_claimed"
    CLAIMED_HANDLED = "claimed_handled"
    CLAIMED_REJECTED = "claimed_rejected"
    CLAIMED_FAILED = "claimed_failed"


@dataclass(frozen=True, slots=True)
class AuthorizedCommandResult:
    """Narrow terminal response returned by the ingress and by registered handlers."""

    disposition: AuthorizedCommandDisposition
    response: Optional[str] = None
    failure_category: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, AuthorizedCommandDisposition):
            raise TypeError("authorized-command disposition is invalid")
        if self.response is not None and (
            not isinstance(self.response, str)
            or len(self.response) > _MAX_RESPONSE_CHARS
        ):
            raise ValueError("authorized-command response is invalid")
        if self.failure_category is not None and (
            not isinstance(self.failure_category, str)
            or not _FAILURE_CATEGORY_RE.fullmatch(self.failure_category)
        ):
            raise ValueError("authorized-command failure category is invalid")

    @property
    def claimed(self) -> bool:
        return self.disposition is not AuthorizedCommandDisposition.NOT_CLAIMED


NOT_CLAIMED = AuthorizedCommandResult(AuthorizedCommandDisposition.NOT_CLAIMED)


@dataclass(frozen=True, slots=True)
class AuthorizedCommandRequest:
    """Bounded immutable raw request passed to an explicitly registered handler.

    ``raw_argument_tail`` is the exact substring after ``/<command_name>``.  It
    therefore retains its leading delimiter and every original codepoint; this layer
    intentionally does not decide the R2.1 argument grammar.
    """

    api_version: int
    transport: str
    command_name: str
    raw_text: str
    raw_command_token: str
    raw_argument_tail: str
    command_entity_offset: Optional[int]
    command_entity_length: Optional[int]
    user_id: str
    chat_id: str
    chat_type: str
    message_id: Optional[str]


_AUTHORITY_SEAL = object()


class _SuccessfulInboundAdmission:
    """Opaque proof that the existing Hermes admission lifecycle succeeded.

    Normal construction is deliberately unavailable.  The two existing runner
    admission paths create this only after both user authorization and bot-loop
    admission have succeeded.  Generic ingress can consume it but cannot infer it
    from a ``MessageEvent``.
    """

    __slots__ = ("__event",)

    def __new__(cls, *args: Any, **kwargs: Any) -> "_SuccessfulInboundAdmission":
        del args, kwargs
        raise TypeError("successful admission is created by the Hermes lifecycle")

    @property
    def event(self) -> Any:
        return self.__event


@dataclass(frozen=True, slots=True)
class AuthorizedInboundContext:
    """Immutable receipt minted by a runner only after normal Hermes admission."""

    _event: Any = field(repr=False, compare=False)
    transport_evidence: tuple[InboundTransportEvidence, ...]
    transport_constituent_count: int
    source_transport: str
    source_user_id: Optional[str]
    source_chat_id: Optional[str]
    source_chat_type: str
    source_is_bot: bool
    message_id: Optional[str]
    normalized_text: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._seal is not _AUTHORITY_SEAL
            or not isinstance(self.transport_evidence, tuple)
            or not isinstance(self.transport_constituent_count, int)
            or isinstance(self.transport_constituent_count, bool)
            or self.transport_constituent_count < 1
        ):
            raise TypeError(
                "AuthorizedInboundContext may only be created after Hermes admission"
            )


def _require_authorized_context(context: Any) -> AuthorizedInboundContext:
    """Reject every caller-created shape before privileged dispatch begins."""
    if type(context) is not AuthorizedInboundContext:
        raise TypeError("authorized-command dispatch requires admitted context")
    try:
        context.__post_init__()
    except TypeError as exc:
        raise TypeError(
            "authorized-command dispatch requires admitted context"
        ) from exc
    return context


@dataclass(frozen=True, slots=True)
class AuthorizedCommandIngressConfig:
    """Validated gateway configuration for the reserved-command boundary."""

    api_version: int = AUTHORIZED_COMMAND_INGRESS_API_VERSION
    enabled: bool = False
    handler_id: str = ""
    transport: str = "telegram"
    reserved_commands: tuple[str, ...] = ()
    dm_only: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (
            isinstance(self.api_version, bool)
            or self.api_version != AUTHORIZED_COMMAND_INGRESS_API_VERSION
        ):
            raise AuthorizedCommandConfigError(
                f"authorized_command_ingress.api_version must be {AUTHORIZED_COMMAND_INGRESS_API_VERSION}"
            )
        if not isinstance(self.enabled, bool):
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress.enabled must be a boolean"
            )
        if not isinstance(self.dm_only, bool):
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress.dm_only must be a boolean"
            )
        if (
            not isinstance(self.transport, str)
            or self.transport not in SUPPORTED_AUTHORIZED_COMMAND_TRANSPORTS
        ):
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress.transport must be one of: "
                + ", ".join(sorted(SUPPORTED_AUTHORIZED_COMMAND_TRANSPORTS))
            )
        if not isinstance(self.handler_id, str) or (
            self.handler_id and not _HANDLER_ID_RE.fullmatch(self.handler_id)
        ):
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress.handler_id must use lowercase letters, digits, '.', '_' or '-'"
            )
        if self.enabled and not self.handler_id:
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress.handler_id must be non-empty when enabled"
            )
        if not isinstance(self.reserved_commands, tuple):
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress.reserved_commands must be a list"
            )
        if len(self.reserved_commands) > _MAX_RESERVED_COMMANDS:
            raise AuthorizedCommandConfigError(
                f"authorized_command_ingress.reserved_commands may contain at most {_MAX_RESERVED_COMMANDS} names"
            )
        seen: set[str] = set()
        for name in self.reserved_commands:
            if not isinstance(name, str) or not _COMMAND_NAME_RE.fullmatch(name):
                raise AuthorizedCommandConfigError(
                    "authorized_command_ingress reserved names must match [a-z][a-z0-9_]{0,31} "
                    "and must not include '/'"
                )
            if name in seen:
                raise AuthorizedCommandConfigError(
                    f"duplicate authorized_command_ingress reserved command: {name}"
                )
            seen.add(name)
        if self.enabled and not self.reserved_commands:
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress.reserved_commands must be non-empty when enabled"
            )
        if self.enabled:
            from hermes_cli.commands import resolve_command

            collisions = tuple(
                name for name in self.reserved_commands
                if resolve_command(name) is not None
            )
            if collisions:
                raise AuthorizedCommandConfigError(
                    "authorized_command_ingress reserved names already belong to Hermes commands: "
                    + ", ".join(collisions)
                )

    @classmethod
    def from_dict(cls, data: Any) -> "AuthorizedCommandIngressConfig":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress must be a mapping"
            )
        unknown = set(data) - {
            "api_version",
            "enabled",
            "handler_id",
            "transport",
            "reserved_commands",
            "dm_only",
        }
        if unknown:
            raise AuthorizedCommandConfigError(
                "unknown authorized_command_ingress keys: "
                + ", ".join(sorted(map(str, unknown)))
            )

        commands = data.get("reserved_commands", ())
        if not isinstance(commands, (list, tuple)):
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress.reserved_commands must be a list"
            )
        enabled = data.get("enabled", False)
        dm_only = data.get("dm_only", True)
        if not isinstance(enabled, bool):
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress.enabled must be a boolean"
            )
        if not isinstance(dm_only, bool):
            raise AuthorizedCommandConfigError(
                "authorized_command_ingress.dm_only must be a boolean"
            )
        return cls(
            api_version=data.get("api_version", AUTHORIZED_COMMAND_INGRESS_API_VERSION),
            enabled=enabled,
            handler_id=data.get("handler_id", ""),
            transport=data.get("transport", "telegram"),
            reserved_commands=tuple(commands),
            dm_only=dm_only,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "enabled": self.enabled,
            "handler_id": self.handler_id,
            "transport": self.transport,
            "reserved_commands": list(self.reserved_commands),
            "dm_only": self.dm_only,
        }


AuthorizedCommandHandler = Callable[
    [AuthorizedCommandRequest],
    Awaitable[AuthorizedCommandResult] | AuthorizedCommandResult,
]


@dataclass(frozen=True, slots=True)
class _HandlerRegistration:
    reserved_commands: tuple[str, ...]
    handler: AuthorizedCommandHandler


_HANDLERS: dict[str, _HandlerRegistration] = {}


def register_authorized_command_handler(
    handler_id: str,
    reserved_commands: tuple[str, ...],
    handler: AuthorizedCommandHandler,
) -> None:
    """Register exact reserved-name ownership for one repository-owned handler."""
    if not isinstance(handler_id, str) or not _HANDLER_ID_RE.fullmatch(handler_id):
        raise ValueError("invalid authorized-command handler id")
    if (
        not isinstance(reserved_commands, tuple)
        or not reserved_commands
        or len(reserved_commands) > _MAX_RESERVED_COMMANDS
        or len(set(reserved_commands)) != len(reserved_commands)
        or any(
            not isinstance(name, str) or not _COMMAND_NAME_RE.fullmatch(name)
            for name in reserved_commands
        )
    ):
        raise ValueError("invalid authorized-command reserved names")
    if not callable(handler):
        raise TypeError("authorized-command handler must be callable")
    existing = _HANDLERS.get(handler_id)
    registration = _HandlerRegistration(reserved_commands, handler)
    if existing is not None and existing != registration:
        raise ValueError(f"authorized-command handler already registered: {handler_id}")
    for existing_id, existing_registration in _HANDLERS.items():
        if existing_id == handler_id:
            continue
        overlap = set(reserved_commands).intersection(
            existing_registration.reserved_commands
        )
        if overlap:
            raise ValueError(
                "authorized-command reservation already owned: "
                + ", ".join(sorted(overlap))
            )
    _HANDLERS[handler_id] = registration


def unregister_authorized_command_handler(
    handler_id: str, handler: Optional[AuthorizedCommandHandler] = None
) -> bool:
    """Remove a handler only when it is the expected registration, if supplied."""
    existing = _HANDLERS.get(handler_id)
    if existing is None or (handler is not None and existing.handler is not handler):
        return False
    del _HANDLERS[handler_id]
    return True


@dataclass(frozen=True, slots=True)
class _Candidate:
    command_name: str
    raw_command_token: str
    raw_argument_tail: str
    syntax_failure: Optional[str] = None


def _platform_value(source: Any) -> str:
    platform = getattr(source, "platform", None)
    value = getattr(platform, "value", platform)
    return str(value or "")


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _bounded_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    rendered = str(value)
    return rendered if rendered and len(rendered) <= _MAX_ID_CHARS else None


def _match_candidate(
    text: str, reserved_commands: tuple[str, ...]
) -> Optional[_Candidate]:
    if not text.startswith("/"):
        return None
    # Longest first keeps overlapping configured names unambiguous.
    for name in sorted(reserved_commands, key=len, reverse=True):
        prefix = f"/{name}"
        if not text.startswith(prefix):
            continue
        following = text[len(prefix) : len(prefix) + 1]
        if following and following in _COMMAND_NAME_CONTINUATION:
            continue  # Similar command name, e.g. /daily_guidex.
        tail = text[len(prefix) :]
        if not following:
            return _Candidate(name, prefix, tail)
        if following == " ":
            return _Candidate(name, prefix, tail)
        # Once the exact reserved name is evident, every other delimiter remains
        # inside the protected boundary but cannot reach the handler.
        return _Candidate(name, prefix, tail, "malformed_delimiter")
    return None


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


def _command_entity(
    evidence: InboundTransportEvidence, candidate: _Candidate
) -> tuple[Optional[int], Optional[int], Optional[str]]:
    if not evidence.command_entities:
        return None, None, None
    for entity in evidence.command_entities:
        if entity.kind != "bot_command":
            continue
        offset = entity.offset_utf16
        length = entity.length_utf16
        if isinstance(offset, bool) or isinstance(length, bool):
            return None, None, "invalid_command_entity"
        if not isinstance(offset, int) or not isinstance(length, int):
            return None, None, "invalid_command_entity"
        entity_text = _utf16_slice(evidence.original_text, offset, length)
        if (
            offset == 0
            and entity_text == entity.text
            and entity_text == candidate.raw_command_token
        ):
            return offset, length, None
    return None, None, "invalid_command_entity"


def _result(
    disposition: AuthorizedCommandDisposition,
    *,
    response: Optional[str] = None,
    failure_category: Optional[str] = None,
) -> AuthorizedCommandResult:
    return AuthorizedCommandResult(disposition, response, failure_category)


def _terminal_rejection(
    category: str, *, silent: bool = False
) -> AuthorizedCommandResult:
    return _result(
        AuthorizedCommandDisposition.CLAIMED_REJECTED,
        response=None if silent else _REJECTED_RESPONSE,
        failure_category=category,
    )


def _terminal_failure(category: str) -> AuthorizedCommandResult:
    return _result(
        AuthorizedCommandDisposition.CLAIMED_FAILED,
        response=_FAILED_RESPONSE,
        failure_category=category,
    )


def _cache_terminal(
    event: Any, result: AuthorizedCommandResult
) -> AuthorizedCommandResult:
    existing = getattr(event, "_authorized_command_ingress_result", None)
    if isinstance(existing, AuthorizedCommandResult):
        return existing
    event._authorized_command_ingress_started = True
    event._authorized_command_ingress_result = result
    return result


def _log_terminal(
    config: AuthorizedCommandIngressConfig,
    candidate: _Candidate,
    result: AuthorizedCommandResult,
) -> None:
    logger.info(
        "authorized-command ingress transport=%s command=%s disposition=%s handler=%s category=%s",
        config.transport,
        candidate.command_name,
        result.disposition.value,
        config.handler_id,
        result.failure_category or "none",
    )


def _transport_evidences(event: Any) -> tuple[InboundTransportEvidence, ...]:
    candidates = getattr(event, "_inbound_transport_evidences", ())
    if not isinstance(candidates, tuple):
        return ()
    return tuple(
        evidence for evidence in candidates
        if isinstance(evidence, InboundTransportEvidence)
    )


def _context_from_successful_admission(
    admission: _SuccessfulInboundAdmission,
) -> AuthorizedInboundContext:
    """Derive dispatch authority only from an opaque successful-admission result."""
    if type(admission) is not _SuccessfulInboundAdmission:
        raise TypeError("authorized context requires successful Hermes admission")
    event = admission.event
    if bool(getattr(event, "internal", False)):
        raise ValueError("internal events cannot receive authorized inbound context")
    source = getattr(event, "source", None)
    return AuthorizedInboundContext(
        _event=event,
        transport_evidence=_transport_evidences(event),
        transport_constituent_count=getattr(
            event, "_inbound_transport_constituent_count", 1
        ),
        source_transport=_platform_value(source),
        source_user_id=_bounded_id(getattr(source, "user_id", None)),
        source_chat_id=_bounded_id(getattr(source, "chat_id", None)),
        source_chat_type=_enum_value(getattr(source, "chat_type", None)),
        source_is_bot=bool(getattr(source, "is_bot", False)),
        message_id=_bounded_id(getattr(event, "message_id", None)),
        normalized_text=(
            getattr(event, "text", "")
            if isinstance(getattr(event, "text", ""), str)
            else ""
        ),
        _seal=_AUTHORITY_SEAL,
    )


def _safe_reserved_names(config: Any) -> tuple[str, ...]:
    raw_names = getattr(config, "reserved_commands", ())
    if not isinstance(raw_names, (list, tuple)):
        return ()
    return tuple(
        name
        for name in raw_names
        if isinstance(name, str) and _COMMAND_NAME_RE.fullmatch(name)
    )


def _candidate_preflight(
    event: Any,
    evidences: tuple[InboundTransportEvidence, ...],
    config: AuthorizedCommandIngressConfig,
    *,
    source_transport: str,
    normalized_text: str,
    transport_constituent_count: int = 1,
) -> tuple[
    Optional[_Candidate],
    Optional[InboundTransportEvidence],
    Optional[AuthorizedCommandResult],
]:
    """Resolve one candidate once for both terminal containment and dispatch."""
    try:
        config.validate()
    except (AuthorizedCommandConfigError, AttributeError, TypeError):
        safe_names = _safe_reserved_names(config)
        candidate = next(
            (
                matched
                for evidence in evidences
                if (matched := _match_candidate(evidence.original_text, safe_names))
                is not None
            ),
            None,
        )
        if candidate is None:
            candidate = _match_candidate(normalized_text.lstrip(), safe_names)
        if candidate is None:
            return None, None, NOT_CLAIMED
        return candidate, None, _cache_terminal(event, _terminal_failure("invalid_config"))

    if not config.enabled:
        return None, None, NOT_CLAIMED

    if evidences and source_transport == config.transport and all(
        evidence.transport == config.transport for evidence in evidences
    ):
        matches = tuple(
            (candidate, evidence)
            for evidence in evidences
            if (candidate := _match_candidate(
                evidence.original_text, config.reserved_commands
            )) is not None
        )
        # Authentic transport evidence disproves a normalized/plugin-created
        # reserved command and is the sole source of privileged meaning.
        if not matches:
            return None, None, NOT_CLAIMED
        candidate, evidence = matches[0]
        if transport_constituent_count != 1:
            result = _cache_terminal(event, _terminal_rejection("ambiguous_batch"))
            _log_terminal(config, candidate, result)
            return candidate, evidence, result
        return candidate, evidence, None

    candidate = _match_candidate(normalized_text.lstrip(), config.reserved_commands)
    if candidate is None:
        return None, None, NOT_CLAIMED
    if transport_constituent_count != 1:
        result = _cache_terminal(event, _terminal_rejection("ambiguous_batch"))
        _log_terminal(config, candidate, result)
        return candidate, None, result
    category = (
        "unsupported_transport"
        if source_transport != config.transport
        or any(evidence.transport != config.transport for evidence in evidences)
        else "raw_context_unavailable"
    )
    result = _cache_terminal(event, _terminal_rejection(category))
    _log_terminal(config, candidate, result)
    return candidate, None, result


def contain_unauthorized_reserved_command(
    event: Any,
    config: AuthorizedCommandIngressConfig,
    *,
    failure_category: str,
    transport_evidence: Optional[InboundTransportEvidence] = None,
) -> AuthorizedCommandResult:
    """Terminally contain a reserved candidate without minting execution authority."""
    cached = getattr(event, "_authorized_command_ingress_result", None)
    if isinstance(cached, AuthorizedCommandResult):
        return cached
    evidences = (
        (transport_evidence,)
        if isinstance(transport_evidence, InboundTransportEvidence)
        else _transport_evidences(event)
    )
    source_transport = _platform_value(getattr(event, "source", None))
    normalized_text = getattr(event, "text", "")
    candidate, _, early = _candidate_preflight(
        event,
        evidences,
        config,
        source_transport=source_transport,
        normalized_text=normalized_text if isinstance(normalized_text, str) else "",
        transport_constituent_count=getattr(
            event, "_inbound_transport_constituent_count", 1
        ),
    )
    if early is not None:
        return early
    if candidate is None:
        return NOT_CLAIMED
    result = _cache_terminal(event, _terminal_rejection(failure_category, silent=True))
    _log_terminal(config, candidate, result)
    return result


async def dispatch_authorized_command(
    context: AuthorizedInboundContext,
    config: AuthorizedCommandIngressConfig,
) -> AuthorizedCommandResult:
    """Apply the boundary using an immutable post-admission authority receipt."""
    _require_authorized_context(context)
    from gateway.authorized_command_ingress_dispatch import dispatch_authorized_context

    return await dispatch_authorized_context(context, config)


__all__ = [
    "AUTHORIZED_COMMAND_INGRESS_API_VERSION",
    "CommandEntityEvidence",
    "InboundTransportEvidence",
    "AuthorizedCommandConfigError",
    "AuthorizedCommandDisposition",
    "AuthorizedCommandHandler",
    "AuthorizedCommandIngressConfig",
    "AuthorizedInboundContext",
    "AuthorizedCommandRequest",
    "AuthorizedCommandResult",
    "contain_unauthorized_reserved_command",
    "dispatch_authorized_command",
    "register_authorized_command_handler",
    "unregister_authorized_command_handler",
]
