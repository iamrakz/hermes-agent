"""Security contracts for the post-authorization reserved-command boundary."""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.authorized_command_ingress import (
    AUTHORIZED_COMMAND_INGRESS_API_VERSION,
    AuthorizedCommandConfigError,
    AuthorizedCommandDisposition,
    AuthorizedCommandIngressConfig,
    AuthorizedInboundContext,
    AuthorizedCommandResult,
    CommandEntityEvidence,
    InboundTransportEvidence,
    _SuccessfulInboundAdmission,
    _context_from_successful_admission,
    contain_unauthorized_reserved_command,
    dispatch_authorized_command as _dispatch_authorized_command,
    register_authorized_command_handler as _register_authorized_command_handler,
    unregister_authorized_command_handler,
)
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from plugins.platforms.telegram.authorized_command_evidence import (
    snapshot_telegram_transport_evidence,
)


_HANDLER_ID = "test-owner-read-only"
_RESERVED = ("daily_guide", "daily_brief", "attention")


def register_authorized_command_handler(handler_id, handler, commands=_RESERVED):
    _register_authorized_command_handler(handler_id, tuple(commands), handler)


async def dispatch_authorized_command(event, config):
    return await _dispatch_authorized_command(_authorized_context(event), config)


def _successful_admission(event):
    """Test-only construction of the opaque result normally produced by the runner."""
    admission = object.__new__(_SuccessfulInboundAdmission)
    object.__setattr__(
        admission, "_SuccessfulInboundAdmission__event", event
    )
    return admission


def _authorized_context(event):
    return _context_from_successful_admission(_successful_admission(event))


def _unsealed_context(event):
    context = object.__new__(AuthorizedInboundContext)
    values = {
        "_event": event,
        "transport_evidence": event._inbound_transport_evidences,
        "transport_constituent_count": 1,
        "source_transport": "telegram",
        "source_user_id": "owner-1",
        "source_chat_id": "owner-1",
        "source_chat_type": "dm",
        "source_is_bot": False,
        "message_id": "message-1",
        "normalized_text": event.text,
        "_seal": object(),
    }
    for name, value in values.items():
        object.__setattr__(context, name, value)
    return context


def _config(**overrides) -> AuthorizedCommandIngressConfig:
    values = {
        "api_version": AUTHORIZED_COMMAND_INGRESS_API_VERSION,
        "enabled": True,
        "handler_id": _HANDLER_ID,
        "transport": "telegram",
        "reserved_commands": _RESERVED,
        "dm_only": True,
    }
    values.update(overrides)
    return AuthorizedCommandIngressConfig(**values)


def _utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _raw_message(
    text: str,
    *,
    user_id: str = "owner-1",
    chat_id: str = "owner-1",
    chat_type: str = "private",
    is_bot: bool = False,
    forwarded: bool = False,
    entities: object = ...,
):
    token = text.split(" ", 1)[0]
    if entities is ...:
        entities = (
            [SimpleNamespace(type="bot_command", offset=0, length=_utf16_length(token))]
            if text.startswith("/")
            else []
        )
    return SimpleNamespace(
        text=text,
        caption=None,
        entities=entities,
        caption_entities=[],
        from_user=SimpleNamespace(id=user_id, is_bot=is_bot),
        chat=SimpleNamespace(id=chat_id, type=chat_type),
        forward_origin=SimpleNamespace(kind="user") if forwarded else None,
    )


def _event(
    raw_text: str | None = "/daily_guide today",
    *,
    normalized_text: str | None = None,
    platform: Platform = Platform.TELEGRAM,
    chat_type: str = "dm",
    user_id: str | None = "owner-1",
    chat_id: str | None = "owner-1",
    is_bot: bool = False,
    internal: bool = False,
    raw_message=None,
) -> MessageEvent:
    source = SessionSource(
        platform=platform,
        chat_id=cast(str, chat_id),
        chat_type=chat_type,
        user_id=user_id,
        user_name="owner",
        is_bot=is_bot,
    )
    if raw_message is None and raw_text is not None:
        raw_message = _raw_message(
            raw_text,
            user_id=user_id or "",
            chat_id=chat_id or "",
            chat_type="private" if chat_type == "dm" else chat_type,
            is_bot=is_bot,
        )
    event = MessageEvent(
        text=normalized_text if normalized_text is not None else (raw_text or ""),
        source=source,
        raw_message=raw_message,
        message_id="message-1",
        internal=internal,
    )
    if platform is Platform.TELEGRAM and raw_message is not None:
        event._inbound_transport_evidence = snapshot_telegram_transport_evidence(
            raw_message, 77
        )
    return event


@pytest.fixture(autouse=True)
def _clean_handler_registration():
    unregister_authorized_command_handler(_HANDLER_ID)
    yield
    unregister_authorized_command_handler(_HANDLER_ID)


def _handled(response: str = "synthetic response") -> AuthorizedCommandResult:
    return AuthorizedCommandResult(
        AuthorizedCommandDisposition.CLAIMED_HANDLED,
        response=response,
    )


def test_config_default_is_disabled_and_round_trips_through_gateway_contract():
    default = GatewayConfig()
    assert default.authorized_command_ingress.enabled is False
    assert (
        default.authorized_command_ingress.api_version
        == AUTHORIZED_COMMAND_INGRESS_API_VERSION
    )

    restored = GatewayConfig.from_dict({
        "gateway": {
            "authorized_command_ingress": {
                "api_version": 1,
                "enabled": True,
                "handler_id": _HANDLER_ID,
                "transport": "telegram",
                "reserved_commands": list(_RESERVED),
            }
        }
    })
    assert restored.authorized_command_ingress == _config()
    assert (
        GatewayConfig.from_dict(restored.to_dict()).authorized_command_ingress
        == _config()
    )


def test_real_yaml_loader_bridges_nested_gateway_config(tmp_path, monkeypatch):
    from gateway.config import load_gateway_config

    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        "  authorized_command_ingress:\n"
        "    api_version: 1\n"
        "    enabled: true\n"
        f"    handler_id: {_HANDLER_ID}\n"
        "    transport: telegram\n"
        "    reserved_commands: [daily_guide, daily_brief, attention]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)
    assert load_gateway_config().authorized_command_ingress == _config()


def test_real_yaml_loader_rejects_invalid_ingress_config(tmp_path, monkeypatch):
    from gateway.config import load_gateway_config

    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        "  authorized_command_ingress:\n"
        "    enabled: 'true'\n"
        f"    handler_id: {_HANDLER_ID}\n"
        "    transport: telegram\n"
        "    reserved_commands: [daily_guide]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)
    with pytest.raises(AuthorizedCommandConfigError):
        load_gateway_config()


@pytest.mark.parametrize(
    "block",
    [
        {"api_version": 2},
        {"enabled": "true"},
        {"enabled": True, "handler_id": "", "reserved_commands": ["daily_guide"]},
        {"handler_id": "Unsafe Handler"},
        {"transport": "discord"},
        {"reserved_commands": [""]},
        {"reserved_commands": ["/daily_guide"]},
        {"reserved_commands": ["daily-guide"]},
        {"reserved_commands": ["daily_guide", "daily_guide"]},
        {"reserved_commands": "daily_guide"},
        {"unknown_policy": True},
    ],
)
def test_invalid_config_is_rejected_before_privileged_execution(block):
    with pytest.raises(AuthorizedCommandConfigError):
        AuthorizedCommandIngressConfig.from_dict(block)


def test_handler_registration_rejects_implicit_replacement():
    first = AsyncMock(return_value=_handled())
    second = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, first)
    with pytest.raises(ValueError, match="already registered"):
        register_authorized_command_handler(_HANDLER_ID, second)


def test_handler_registration_rejects_overlapping_ownership():
    first = AsyncMock(return_value=_handled())
    second = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, first)
    with pytest.raises(ValueError, match="already owned"):
        _register_authorized_command_handler(
            "other-owner", ("daily_guide", "other"), second
        )


@pytest.mark.asyncio
async def test_disabled_default_does_not_change_normal_dispatch():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(
        _event(), AuthorizedCommandIngressConfig()
    )
    assert result.disposition is AuthorizedCommandDisposition.NOT_CLAIMED
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_invalid_config_cannot_invoke_handler_or_hide_raw_candidate():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    config = _config()
    object.__setattr__(config, "api_version", 2)
    event = _event("/daily_guide today", normalized_text="hello")
    result = await dispatch_authorized_command(event, config)
    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_FAILED
    assert result.failure_category == "invalid_config"
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_authorized_owner_reaches_explicit_handler_with_exact_raw_tail():
    seen = []
    raw_text = "/daily_guide alpha\tbeta\ngamma\u00a0delta"

    async def handler(request):
        seen.append(request)
        return _handled()

    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(_event(raw_text), _config())

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_HANDLED
    assert len(seen) == 1
    assert seen[0].raw_text == raw_text
    assert seen[0].raw_command_token == "/daily_guide"
    assert seen[0].raw_argument_tail == " alpha\tbeta\ngamma\u00a0delta"


@pytest.mark.asyncio
async def test_raw_transport_object_can_be_discarded_after_immutable_snapshot():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event("/daily_guide today")
    evidence = event._inbound_transport_evidence
    assert isinstance(evidence, InboundTransportEvidence)
    assert isinstance(evidence.command_entities[0], CommandEntityEvidence)
    event.raw_message = None

    result = await dispatch_authorized_command(event, _config())

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_HANDLED
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_raw_object_mutation_after_snapshot_cannot_change_authority():
    raw = _raw_message("/daily_guide today")
    event = _event(raw_message=raw)
    evidence = event._inbound_transport_evidence
    assert isinstance(evidence, InboundTransportEvidence)
    raw.text = "hello"
    raw.from_user.id = "attacker"
    with pytest.raises(FrozenInstanceError):
        evidence.original_text = "hello"  # type: ignore[misc]

    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(event, _config())

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_HANDLED
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_unauthorized_event_never_reaches_handler_and_gets_no_reply():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)

    result = contain_unauthorized_reserved_command(
        _event(), _config(), failure_category="not_authorized"
    )

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_REJECTED
    assert result.response is None
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_auth_caller_cannot_elevate_event_with_boolean():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    with pytest.raises(TypeError):
        await cast(Any, _dispatch_authorized_command)(
            _event(), _config(), authorized=True
        )
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_lowest_dispatch_rejects_raw_fake_and_unsealed_contexts():
    from gateway.authorized_command_ingress_dispatch import dispatch_authorized_context

    event = _event()
    fake = SimpleNamespace(
        _event=event,
        transport_evidence=event._inbound_transport_evidences,
        transport_constituent_count=1,
        source_transport="telegram",
        source_user_id="owner-1",
        source_chat_id="owner-1",
        source_chat_type="dm",
        source_is_bot=False,
        message_id="message-1",
        normalized_text=event.text,
    )
    invalid = _authorized_context(event)
    object.__setattr__(invalid, "_seal", object())

    for candidate in (event, fake, invalid):
        with pytest.raises(TypeError, match="requires admitted context"):
            await dispatch_authorized_context(candidate, _config())


def test_authorized_context_rejects_caller_supplied_boolean_seal():
    event = _event()
    with pytest.raises(TypeError, match="only be created"):
        AuthorizedInboundContext(
            _event=event,
            transport_evidence=event._inbound_transport_evidences,
            transport_constituent_count=1,
            source_transport="telegram",
            source_user_id="owner-1",
            source_chat_id="owner-1",
            source_chat_type="dm",
            source_is_bot=False,
            message_id="message-1",
            normalized_text=event.text,
            _seal=True,
        )


def test_internal_reserved_command_is_terminally_contained_without_context():
    event = _event(internal=True)
    result = contain_unauthorized_reserved_command(
        event, _config(), failure_category="internal_event"
    )
    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_REJECTED
    assert result.failure_category == "internal_event"
    assert result.response is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "category"),
    [
        (_event(is_bot=True), "bot_sender"),
        (_event(platform=Platform.DISCORD, raw_message=None), "unsupported_transport"),
        (_event(chat_type="group"), "non_dm_context"),
        (
            _event(raw_message=_raw_message("/daily_guide today", forwarded=True)),
            "forwarded_message",
        ),
        (_event(user_id=None), "missing_source_identity"),
        (
            _event(raw_message=SimpleNamespace(text="/daily_guide today", entities=[])),
            "missing_raw_identity",
        ),
    ],
    ids=[
        "bot",
        "transport",
        "non-dm",
        "forwarded",
        "sender",
        "raw-context",
    ],
)
async def test_unsuitable_context_is_claimed_and_rejected_without_handler(
    event, category
):
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)

    result = await dispatch_authorized_command(event, _config())

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_REJECTED
    assert result.failure_category == category
    handler.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("delimiter", ["\t", "\n", "\u00a0", "@", ":"])
async def test_malformed_reserved_delimiters_are_terminal_rejections(delimiter):
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)

    result = await dispatch_authorized_command(
        _event(f"/daily_guide{delimiter}payload"), _config()
    )

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_REJECTED
    assert result.failure_category == "malformed_delimiter"
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_similar_command_name_is_not_claimed():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(_event("/daily_guidex today"), _config())
    assert result.disposition is AuthorizedCommandDisposition.NOT_CLAIMED
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_rewrite_cannot_manufacture_privileged_command():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(
        _event("hello", normalized_text="/daily_guide today"), _config()
    )
    assert result.disposition is AuthorizedCommandDisposition.NOT_CLAIMED
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_rewrite_cannot_hide_real_reserved_raw_command():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(
        _event("/daily_guide today", normalized_text="hello"), _config()
    )
    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_HANDLED
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_raw_context_is_contained_but_never_executed():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event(
        raw_text=None, normalized_text="/daily_guide today", raw_message=None
    )
    result = await dispatch_authorized_command(event, _config())
    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_REJECTED
    assert result.failure_category == "raw_context_unavailable"
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_inconsistent_telegram_command_entity_fails_closed():
    raw = _raw_message(
        "/daily_guide today",
        entities=[SimpleNamespace(type="bold", offset=0, length=12)],
    )
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(_event(raw_message=raw), _config())
    assert result.failure_category == "invalid_command_entity"
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_handler_and_handler_exception_are_terminal_failures():
    missing = await dispatch_authorized_command(_event(), _config())
    assert missing.disposition is AuthorizedCommandDisposition.CLAIMED_FAILED
    assert missing.failure_category == "handler_missing"

    event = _event()

    async def raising_handler(_request):
        raise RuntimeError("synthetic failure")

    register_authorized_command_handler(_HANDLER_ID, raising_handler)
    failed = await dispatch_authorized_command(event, _config())
    assert failed.disposition is AuthorizedCommandDisposition.CLAIMED_FAILED
    assert failed.failure_category == "handler_exception"


@pytest.mark.asyncio
async def test_registered_name_subset_is_terminal_reservation_mismatch():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler, commands=("daily_guide",))

    result = await dispatch_authorized_command(_event(), _config())

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_FAILED
    assert result.failure_category == "handler_reservation_mismatch"
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_not_claimed_result_is_converted_to_terminal_failure():
    handler = AsyncMock(
        return_value=AuthorizedCommandResult(AuthorizedCommandDisposition.NOT_CLAIMED)
    )
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(_event(), _config())
    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_FAILED
    assert result.failure_category == "handler_not_claimed"


@pytest.mark.asyncio
async def test_handler_rejection_remains_claimed_and_terminal():
    handler = AsyncMock(
        return_value=AuthorizedCommandResult(
            AuthorizedCommandDisposition.CLAIMED_REJECTED,
            response="synthetic rejection",
            failure_category="invalid_arguments",
        )
    )
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(_event(), _config())
    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_REJECTED
    assert result.response == "synthetic rejection"
    assert result.failure_category == "invalid_arguments"


@pytest.mark.asyncio
async def test_repeated_dispatch_invokes_handler_at_most_once():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event()
    first = await dispatch_authorized_command(event, _config())
    second = await dispatch_authorized_command(event, _config())
    assert first == second
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_duplicate_dispatch_cannot_reenter_handler():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return _handled()

    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event()
    first_task = asyncio.create_task(dispatch_authorized_command(event, _config()))
    await entered.wait()
    duplicate_task = asyncio.create_task(
        dispatch_authorized_command(event, _config())
    )
    await asyncio.sleep(0)
    release.set()
    first, duplicate = await asyncio.gather(first_task, duplicate_task)

    assert first.disposition is AuthorizedCommandDisposition.CLAIMED_HANDLED
    assert duplicate == first
    assert calls == 1


@pytest.mark.asyncio
async def test_cancellation_is_terminal_and_never_retries_handler():
    calls = 0

    async def cancelled_handler(_request):
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError

    register_authorized_command_handler(_HANDLER_ID, cancelled_handler)
    event = _event()
    with pytest.raises(asyncio.CancelledError):
        await dispatch_authorized_command(event, _config())
    repeated = await dispatch_authorized_command(event, _config())
    assert repeated.disposition is AuthorizedCommandDisposition.CLAIMED_FAILED
    assert repeated.failure_category == "cancelled"
    assert calls == 1


def _bare_runner(config: AuthorizedCommandIngressConfig):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(authorized_command_ingress=config)
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("original_text", "rewritten_text", "expected"),
    [
        ("hello", "/daily_guide today", AuthorizedCommandDisposition.NOT_CLAIMED),
        (
            "/daily_guide today",
            "hello",
            AuthorizedCommandDisposition.CLAIMED_HANDLED,
        ),
    ],
)
async def test_admission_preserves_only_adapter_evidence_across_plugin_rewrite(
    original_text, rewritten_text, expected
):
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    original = _event(original_text)
    original_evidence = original._inbound_transport_evidence
    rewritten = _event(rewritten_text)
    runner = _bare_runner(_config())
    runner._scale_to_zero_note_real_inbound = MagicMock()
    runner._hm_pre_gateway_dispatch_hook = AsyncMock(return_value=rewritten)
    runner._is_user_authorized_for_source = MagicMock(return_value=True)
    runner._admit_bot_message_for_source = MagicMock(return_value=True)

    admitted = await runner._hm_admit_event(original)

    assert admitted is not None
    admitted_event, _, _, admission = admitted
    assert admitted_event._inbound_transport_evidence is original_evidence
    context = _context_from_successful_admission(admission)
    result = await _dispatch_authorized_command(context, _config())
    assert result.disposition is expected
    if expected is AuthorizedCommandDisposition.CLAIMED_HANDLED:
        handler.assert_awaited_once()
    else:
        handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_cold_path_intercepts_immediately_after_admission(monkeypatch):
    handler = AsyncMock(return_value=_handled("cold response"))
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event()
    runner = _bare_runner(_config())
    runner._hm_admit_event = AsyncMock(return_value=(
        event, event.source, False, _successful_admission(event)
    ))
    runner._hm_estop_gate = MagicMock(side_effect=AssertionError("normal path reached"))
    runner._delivery_adapter_for = MagicMock(return_value=SimpleNamespace())
    runner._send_authorized_command_reply = AsyncMock()
    metrics = MagicMock()
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_events.record_gateway_slash_command",
        metrics,
    )

    assert await runner._handle_message(event) is None
    runner._hm_estop_gate.assert_not_called()
    metrics.assert_not_called()
    handler.assert_awaited_once()
    runner._send_authorized_command_reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_cold_path_admission_drop_never_reaches_handler():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    runner = _bare_runner(_config())
    runner._hm_admit_event = AsyncMock(return_value=None)
    assert await runner._handle_message(_event()) is None
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_cold_handler_cancellation_never_reaches_normal_dispatch(monkeypatch):
    handler = AsyncMock(side_effect=asyncio.CancelledError)
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event()
    runner = _bare_runner(_config())
    runner._hm_admit_event = AsyncMock(return_value=(
        event, event.source, False, _successful_admission(event)
    ))
    runner._hm_estop_gate = MagicMock(side_effect=AssertionError("normal path reached"))
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_events.record_gateway_slash_command",
        MagicMock(side_effect=AssertionError("metrics path reached")),
    )

    with pytest.raises(asyncio.CancelledError):
        await runner._handle_message(event)

    runner._hm_estop_gate.assert_not_called()
    handler.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["hello", "/status"])
async def test_non_reserved_cold_traffic_continues_unchanged(monkeypatch, text):
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event(text)
    runner = _bare_runner(_config())
    runner._hm_admit_event = AsyncMock(return_value=(
        event, event.source, False, _successful_admission(event)
    ))
    runner._hm_estop_gate = MagicMock(return_value="normal-path-sentinel")
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_events.record_gateway_slash_command",
        MagicMock(),
    )
    assert await runner._handle_message(event) == "normal-path-sentinel"
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_runner_intercepts_before_queue_steer_replay_or_agent():
    handler = AsyncMock(return_value=_handled("busy response"))
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event()
    runner = _bare_runner(_config())
    runner._is_user_authorized_for_source = MagicMock(return_value=True)
    runner._admit_bot_message_for_source = MagicMock(return_value=True)
    runner._delivery_adapter_for = MagicMock(return_value=SimpleNamespace())
    runner._send_authorized_command_reply = AsyncMock()
    runner._effective_busy_input_mode = MagicMock(
        side_effect=AssertionError("busy queue/steer path reached")
    )

    assert await runner._handle_active_session_busy_message(event, "agent:key") is True
    assert event._gateway_accepted is True
    runner._send_authorized_command_reply.assert_awaited_once()
    runner._effective_busy_input_mode.assert_not_called()
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_busy_runner_unauthorized_sender_never_mints_context_or_reaches_handler(
    monkeypatch,
):
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    context_factory = MagicMock(side_effect=AssertionError("context minted pre-auth"))
    monkeypatch.setattr(
        "gateway.run_busy._context_from_successful_admission",
        context_factory,
    )
    event = _event()
    runner = _bare_runner(_config())
    runner._is_user_authorized_for_source = MagicMock(return_value=False)
    assert await runner._handle_active_session_busy_message(event, "agent:key") is True
    context_factory.assert_not_called()
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_reply_delivery_failure_remains_terminal():
    handler = AsyncMock(return_value=_handled("busy response"))
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event()
    runner = _bare_runner(_config())
    runner._is_user_authorized_for_source = MagicMock(return_value=True)
    runner._admit_bot_message_for_source = MagicMock(return_value=True)
    runner._delivery_adapter_for = MagicMock(return_value=SimpleNamespace())
    runner._send_authorized_command_reply = AsyncMock(
        side_effect=RuntimeError("synthetic delivery failure")
    )
    runner._effective_busy_input_mode = MagicMock(
        side_effect=AssertionError("busy fallback reached")
    )
    assert await runner._handle_active_session_busy_message(event, "agent:key") is True
    runner._effective_busy_input_mode.assert_not_called()
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_base_adapter_busy_guard_delegates_reserved_command_without_queueing():
    from tests.gateway.test_active_session_text_merge import _make_adapter

    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    runner = _bare_runner(_config())
    runner._is_user_authorized_for_source = MagicMock(return_value=True)
    runner._admit_bot_message_for_source = MagicMock(return_value=True)
    runner._delivery_adapter_for = MagicMock(return_value=SimpleNamespace())
    runner._send_authorized_command_reply = AsyncMock()
    runner._effective_busy_input_mode = MagicMock(
        side_effect=AssertionError("busy fallback reached")
    )

    adapter = _make_adapter()
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    event = _event()
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    assert session_key not in adapter._pending_messages
    assert event._gateway_accepted is True
    cast(AsyncMock, adapter._message_handler).assert_not_awaited()
    handler.assert_awaited_once()


def test_reserved_name_owned_by_real_hermes_registry_fails_config_closed():
    cancellation = MagicMock()
    with pytest.raises(AuthorizedCommandConfigError, match="already belong"):
        _config(reserved_commands=("stop",))
    cancellation.assert_not_called()


@pytest.mark.asyncio
async def test_real_pre_gateway_hook_cannot_route_before_authorization(monkeypatch):
    import gateway.authorized_command_ingress as authority
    from gateway.authorized_command_ingress_dispatch import dispatch_authorized_context
    from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

    authorization_completed = False
    handler_admission_state = []

    async def handler(_request):
        handler_admission_state.append(
            (
                runner._is_user_authorized_for_source.call_count,
                authorization_completed,
            )
        )
        return _handled("authorized response")

    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event(chat_type="group")
    runner = _bare_runner(_config(dm_only=False))
    runner._scale_to_zero_note_real_inbound = MagicMock()

    def authorize(_source):
        nonlocal authorization_completed
        authorization_completed = True
        return True

    runner._is_user_authorized_for_source = MagicMock(side_effect=authorize)
    runner._admit_bot_message_for_source = MagicMock(return_value=True)
    runner._delivery_adapter_for = MagicMock(return_value=SimpleNamespace())
    runner._send_authorized_command_reply = AsyncMock()
    manager = get_plugin_manager()
    monkeypatch.setattr(manager, "_discovered", True)
    calls_before_auth = []

    async def hostile_hook(event, gateway):
        calls_before_auth.append(
            (
                gateway._is_user_authorized_for_source.call_count,
                len(handler_admission_state),
            )
        )
        fake = SimpleNamespace(_event=event)
        invalid = _unsealed_context(event)
        for candidate in (event, fake, invalid):
            with pytest.raises(TypeError, match="requires admitted context"):
                await dispatch_authorized_context(
                    candidate, gateway.config.authorized_command_ingress
                )
        assert not hasattr(authority, "_mint_authorized_context_after_admission")
        with pytest.raises(TypeError, match="requires successful Hermes admission"):
            _context_from_successful_admission(cast(Any, event))
        return {"action": "rewrite", "text": "rewritten"}

    registration = PluginContext(
        PluginManifest(name="authorized-ingress-pre-auth-probe"), manager
    ).register_hook("pre_gateway_dispatch", hostile_hook)
    try:
        assert await runner._handle_message(event) is None
    finally:
        registration.dispose()
    assert calls_before_auth == [(0, 0)]
    runner._is_user_authorized_for_source.assert_called_once()
    assert handler_admission_state == [(1, True)]
    runner._send_authorized_command_reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_pre_gateway_hook_cannot_forge_authority_by_mutating_shared_state():
    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event()
    runner = _bare_runner(_config())
    runner._scale_to_zero_note_real_inbound = MagicMock()
    runner._is_user_authorized_for_source = MagicMock(return_value=True)

    async def hostile_hook(hook_event, _source):
        hook_event._authorized_command_ingress_result = _handled("forged")
        return hook_event

    runner._hm_pre_gateway_dispatch_hook = hostile_hook
    assert await runner._hm_admit_event(event) is None
    runner._is_user_authorized_for_source.assert_not_called()
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_telegram_builder_attribution_and_plugin_copy_preserve_evidence():
    from tests.gateway.test_telegram_group_gating import (
        _bot_command_entity,
        _group_message,
        _make_adapter,
    )

    text = "/daily_guide today"
    msg = _group_message(text, entities=[_bot_command_entity(text, "/daily_guide")])
    adapter = _make_adapter(
        allowed_chats=["-100"],
        group_allowed_chats=["-100"],
        observe_unmentioned_group_messages=True,
    )
    adapter._cache_replied_media = AsyncMock()
    event = await adapter._build_triggered_event(
        msg, SimpleNamespace(update_id=901), MessageType.COMMAND
    )
    evidence = event._inbound_transport_evidence
    rewritten = dataclasses.replace(event, text="[plugin rewrite]\nhello")
    assert rewritten._inbound_transport_evidence is evidence
    assert evidence.original_text == text

    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(rewritten, _config(dm_only=False))
    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_HANDLED
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_telegram_media_attribution_preserves_raw_reserved_candidate():
    from tests.gateway.test_telegram_group_gating import (
        _bot_command_entity,
        _group_message,
        _make_adapter,
    )

    caption = "/daily_guide today"
    msg = _group_message(
        None,
        caption=caption,
        caption_entities=[_bot_command_entity(caption, "/daily_guide")],
    )
    adapter = _make_adapter(
        allowed_chats=["-100"],
        group_allowed_chats=["-100"],
        observe_unmentioned_group_messages=True,
    )
    original = adapter._build_message_event(msg, MessageType.PHOTO, update_id=902)
    evidence = original._inbound_transport_evidence
    attributed = adapter._apply_telegram_group_observe_attribution(original)
    assert attributed._inbound_transport_evidence is evidence
    assert evidence.original_text == caption
    result = await dispatch_authorized_command(attributed, _config(dm_only=False))
    assert result.claimed
    assert result.failure_category == "missing_source_identity"


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_actual_media_merge_rejects_mixed_reserved_batch(reverse):
    from gateway.platforms.base import merge_pending_message_event
    from plugins.platforms.telegram.adapter import TelegramAdapter

    first, second = _event("hello"), _event("/daily_guide today")
    first.media_urls, second.media_urls = ["one.jpg"], ["two.jpg"]
    first.media_types = second.media_types = ["image/jpeg"]
    pending = {}
    adapter = object.__new__(TelegramAdapter)
    for event in ((second, first) if reverse else (first, second)):
        adapter._merge_into_pending(pending, "batch", event)
    merged = pending["batch"]
    assert len(merged._inbound_transport_evidences) == 2

    generic = {}
    generic_first, generic_second = _event("hello"), _event("/daily_guide today")
    generic_first.media_urls, generic_second.media_urls = ["one.jpg"], ["two.jpg"]
    generic_first.media_types = generic_second.media_types = ["image/jpeg"]
    generic_events = (
        (generic_second, generic_first)
        if reverse
        else (generic_first, generic_second)
    )
    for event in generic_events:
        merge_pending_message_event(generic, "batch", event)
    assert len(generic["batch"]._inbound_transport_evidences) == 2

    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(merged, _config())
    assert result.failure_category == "ambiguous_batch"
    generic_result = await dispatch_authorized_command(generic["batch"], _config())
    assert generic_result.failure_category == "ambiguous_batch"
    handler.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_actual_text_batching_rejects_later_reserved_constituent(reverse):
    from tests.gateway.test_telegram_text_batching import _make_adapter

    adapter = _make_adapter()
    ordinary, reserved = _event("hello"), _event("/daily_guide today")
    for event in ((reserved, ordinary) if reverse else (ordinary, reserved)):
        adapter._enqueue_text_event(event)
    merged = next(iter(adapter._pending_text_batches.values()))
    for task in adapter._pending_text_batch_tasks.values():
        task.cancel()
    await asyncio.gather(*adapter._pending_text_batch_tasks.values(), return_exceptions=True)
    assert len(merged._inbound_transport_evidences) == 2

    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(merged, _config())
    assert result.failure_category == "ambiguous_batch"
    handler.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_captionless_media_constituent_makes_reserved_batch_ambiguous(reverse):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    captionless, reserved = _event(None), _event("/daily_guide today")
    captionless.message_type = reserved.message_type = MessageType.PHOTO
    captionless.media_urls, reserved.media_urls = ["blank.jpg"], ["reserved.jpg"]
    captionless.media_types = reserved.media_types = ["image/jpeg"]
    pending = {}
    adapter = object.__new__(TelegramAdapter)
    for event in ((reserved, captionless) if reverse else (captionless, reserved)):
        adapter._merge_into_pending(pending, "batch", event)
    merged = pending["batch"]

    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(merged, _config())

    assert merged._inbound_transport_constituent_count == 2
    assert len(merged._inbound_transport_evidences) == 1
    assert result.failure_category == "ambiguous_batch"
    handler.assert_not_awaited()


def _authorized_busy_runner(config):
    runner = _bare_runner(config)
    runner._is_user_authorized_for_source = MagicMock(return_value=True)
    runner._admit_bot_message_for_source = MagicMock(return_value=True)
    runner._delivery_adapter_for = MagicMock(return_value=SimpleNamespace())
    runner._send_authorized_command_reply = AsyncMock()
    runner._effective_busy_input_mode = MagicMock(
        side_effect=AssertionError("ordinary busy path reached")
    )
    return runner


@pytest.mark.asyncio
async def test_busy_copy_then_cold_redispatch_executes_and_replies_once():
    from gateway.authorized_command_ingress_routing import route_cold_authorized_command

    handler = AsyncMock(return_value=_handled("once"))
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event()
    runner = _authorized_busy_runner(_config())
    assert await runner._handle_active_session_busy_message(event, "agent:key")
    rewritten = dataclasses.replace(event, text="buffered rewrite")
    result = await route_cold_authorized_command(
        runner, _authorized_context(rewritten)
    )
    assert result.claimed and result.response is None
    handler.assert_awaited_once()
    runner._send_authorized_command_reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_busy_cancellation_copy_redispatch_never_reenters_handler():
    from gateway.authorized_command_ingress_routing import route_cold_authorized_command

    entered = asyncio.Event()
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()

    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event()
    runner = _authorized_busy_runner(_config())
    task = asyncio.create_task(
        runner._handle_active_session_busy_message(event, "agent:key")
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    rewritten = dataclasses.replace(event, text="buffered rewrite")
    result = await route_cold_authorized_command(
        runner, _authorized_context(rewritten)
    )
    assert result.failure_category == "cancelled"
    assert calls == 1


@pytest.mark.asyncio
async def test_repeated_and_concurrent_busy_dispatch_reply_at_most_once():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return _handled("once")

    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _event()
    runner = _authorized_busy_runner(_config())
    first = asyncio.create_task(
        runner._handle_active_session_busy_message(event, "agent:key")
    )
    await entered.wait()
    duplicate = asyncio.create_task(
        runner._handle_active_session_busy_message(
            dataclasses.replace(event), "agent:key"
        )
    )
    await asyncio.sleep(0)
    release.set()
    assert all(await asyncio.gather(first, duplicate))
    assert await runner._handle_active_session_busy_message(event, "agent:key")
    assert calls == 1
    runner._send_authorized_command_reply.assert_awaited_once()


def _telegram_delivery_runner(adapter):
    runner = _bare_runner(_config())
    runner._is_user_authorized_for_source = MagicMock(return_value=True)
    runner._admit_bot_message_for_source = MagicMock(return_value=True)
    runner._delivery_adapter_for = MagicMock(return_value=adapter)
    runner._effective_busy_input_mode = MagicMock(
        side_effect=AssertionError("ordinary busy path reached")
    )
    return runner


def _telegram_delivery_event():
    return dataclasses.replace(_event(), message_id="41")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disconnect", "timeout"])
async def test_privileged_telegram_uncertain_delivery_is_attempted_once(failure):
    from telegram.error import NetworkError, TimedOut
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic"))
    error = (
        NetworkError("server disconnected after accepting private payload")
        if failure == "disconnect"
        else TimedOut("private payload read timed out")
    )
    adapter._bot = SimpleNamespace(send_message=AsyncMock(side_effect=error))
    runner = _telegram_delivery_runner(adapter)
    handler = AsyncMock(return_value=_handled("private response"))
    register_authorized_command_handler(_HANDLER_ID, handler)
    event = _telegram_delivery_event()

    assert await runner._handle_active_session_busy_message(event, "agent:key")
    assert await runner._handle_active_session_busy_message(
        dataclasses.replace(event), "agent:key"
    )

    handler.assert_awaited_once()
    adapter._bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_privileged_telegram_success_is_one_api_attempt():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic"))
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=7))
    )
    runner = _telegram_delivery_runner(adapter)
    register_authorized_command_handler(
        _HANDLER_ID, AsyncMock(return_value=_handled("private response"))
    )

    assert await runner._handle_active_session_busy_message(
        _telegram_delivery_event(), "agent:key"
    )
    adapter._bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_privileged_delivery_exception_does_not_log_private_text(caplog):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    sentinel = "PRIVATE_ARGUMENT_SENTINEL_DO_NOT_LOG"
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic"))
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(side_effect=RuntimeError(sentinel))
    )
    runner = _telegram_delivery_runner(adapter)
    register_authorized_command_handler(
        _HANDLER_ID, AsyncMock(return_value=_handled("private response"))
    )

    assert await runner._handle_active_session_busy_message(
        _telegram_delivery_event(), "agent:key"
    )

    assert sentinel not in caplog.text
    assert "category=delivery_failed" in caplog.text
    assert "exception=RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_non_bmp_argument_uses_telegram_utf16_entity_contract():
    seen = []

    async def handler(request):
        seen.append(request)
        return _handled()

    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(
        _event("/daily_guide 🧪 today"), _config()
    )
    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_HANDLED
    assert seen[0].raw_argument_tail == " 🧪 today"


@pytest.mark.asyncio
async def test_mutating_copied_entity_chat_and_forward_objects_cannot_change_snapshot():
    raw = _raw_message("/daily_guide today")
    raw.via_bot = None
    raw.sender_chat = None
    event = _event(raw_message=raw)
    evidence = event._inbound_transport_evidence
    raw.entities[0].offset = 9
    raw.entities[0].type = "bold"
    raw.chat.id = "attacker"
    raw.chat.type = "group"
    raw.forward_origin = SimpleNamespace(kind="channel")
    raw.via_bot = SimpleNamespace(id=9)
    raw.sender_chat = SimpleNamespace(id=8)

    handler = AsyncMock(return_value=_handled())
    register_authorized_command_handler(_HANDLER_ID, handler)
    result = await dispatch_authorized_command(event, _config())
    assert evidence.command_entities[0].offset_utf16 == 0
    assert evidence.chat_id == "owner-1" and evidence.forwarded is False
    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_HANDLED
    handler.assert_awaited_once()
