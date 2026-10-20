"""Contracts for the Rak Trading OS authorized read-command adapter."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.authorized_command_ingress import (
    AUTHORIZED_COMMAND_INGRESS_API_VERSION,
    AuthorizedCommandDisposition,
    AuthorizedCommandRequest,
    _HANDLERS,
    unregister_authorized_command_handler,
)
from gateway.config import GatewayConfig
from gateway.hooks import HookRegistry
from gateway import rak_trading_os_authorized_commands as adapter


_AS_OF = "2026-10-20T13:14:15+07:00"
_PYTHON = (
    "/Users/apirak/.local/share/uv/python/"
    "cpython-3.12.13-macos-aarch64-none/bin/python3.12"
)
_SCRIPTS = "/Users/apirak/rak-trading-os-runtime/portfolios/th-momentum/scripts"


def _request(text: str, *, command_name: str | None = None) -> AuthorizedCommandRequest:
    token, separator, tail = text.partition(" ")
    inferred = token[1:] if token.startswith("/") else token
    return AuthorizedCommandRequest(
        api_version=AUTHORIZED_COMMAND_INGRESS_API_VERSION,
        transport="telegram",
        command_name=command_name if command_name is not None else inferred,
        raw_text=text,
        raw_command_token=token,
        raw_argument_tail=f"{separator}{tail}",
        command_entity_offset=0,
        command_entity_length=len(token),
        user_id="owner-1",
        chat_id="owner-1",
        chat_type="dm",
        message_id="message-1",
    )


class _Process:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stderr.feed_data(stderr)
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.returncode = returncode
        self.killed = False

    async def wait(self) -> int:
        await asyncio.sleep(0)
        return self.returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(0)
        return await self.stdout.read(), await self.stderr.read()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _HangingProcess:
    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode = None
        self.killed = False
        self._finished = asyncio.Event()

    async def wait(self) -> int:
        await self._finished.wait()
        return -9

    async def communicate(self) -> tuple[bytes, bytes]:
        await self._finished.wait()
        return await self.stdout.read(), await self.stderr.read()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self._finished.set()


@pytest.fixture(autouse=True)
def _clean_registration():
    unregister_authorized_command_handler(adapter.HANDLER_ID)
    yield
    unregister_authorized_command_handler(adapter.HANDLER_ID)


def test_registration_owns_only_the_three_commands_and_does_not_enable_ingress():
    HookRegistry()._register_builtin_hooks()
    HookRegistry()._register_builtin_hooks()

    registration = _HANDLERS[adapter.HANDLER_ID]
    assert adapter.HANDLER_ID == "rak.trading_os.daily_read_v1"
    assert registration.reserved_commands == (
        "daily_guide",
        "daily_brief",
        "attention",
    )
    assert registration.handler is adapter.handle_rak_trading_os_authorized_command
    assert GatewayConfig().authorized_command_ingress.enabled is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "script", "intent", "target"),
    [
        (
            "/daily_guide today",
            "operator_hermes_daily_guide.py",
            "daily-guide",
            "today",
        ),
        (
            "/daily_brief today",
            "operator_hermes_daily_read_surface.py",
            "daily-brief",
            "today",
        ),
        (
            "/attention today",
            "operator_hermes_daily_read_surface.py",
            "attention",
            "today",
        ),
        (
            "/daily_guide 2026-10-20",
            "operator_hermes_daily_guide.py",
            "daily-guide",
            "2026-10-20",
        ),
        (
            "/daily_brief 2026-10-20",
            "operator_hermes_daily_read_surface.py",
            "daily-brief",
            "2026-10-20",
        ),
        (
            "/attention 2026-10-20",
            "operator_hermes_daily_read_surface.py",
            "attention",
            "2026-10-20",
        ),
    ],
)
async def test_exact_mapping_starts_one_fixed_child(
    monkeypatch, text, script, intent, target
):
    create = AsyncMock(
        return_value=_Process(
            b"Read-only evidence. Trading permission is not provided.\n"
        )
    )
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(adapter, "_bangkok_now_iso", lambda: _AS_OF)

    result = await adapter.handle_rak_trading_os_authorized_command(_request(text))

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_HANDLED
    assert (
        result.response == "Read-only evidence. Trading permission is not provided.\n"
    )
    assert create.await_count == 1
    args = create.await_args.args
    expected = (
        _PYTHON,
        "-E",
        "-s",
        "-B",
        f"{_SCRIPTS}/{script}",
        intent,
        target,
    )
    assert args == ((*expected, "--as-of", _AS_OF) if target == "today" else expected)
    assert create.await_args.kwargs == {
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "TMPDIR": "/private/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        },
        "start_new_session": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "/daily_guide",
        "/daily_guide 2026-02-30",
        "/DAILY_GUIDE today",
        "/daily_guide  today",
        " /daily_guide today",
        "/daily_guide today ",
        "/daily_guide\ttoday",
        "/daily_guide\ntoday",
        "/daily_guide today extra",
        "/daily_guide today --as-of 2026-10-20T12:00:00+07:00",
        "/daily-guide today",
        "/daily_guide@HermesBot today",
        "/daily_guide\x00today",
        "/daily_guide tomorrow",
        "show me the daily guide",
    ],
)
async def test_malformed_requests_fail_closed_before_child_start(monkeypatch, text):
    create = AsyncMock()
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)

    result = await adapter.handle_rak_trading_os_authorized_command(_request(text))

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_REJECTED
    assert result.failure_category == "invalid_request"
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_metadata_must_match_the_exact_raw_command(monkeypatch):
    create = AsyncMock()
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)

    result = await adapter.handle_rak_trading_os_authorized_command(
        _request("/daily_guide today", command_name="daily_brief")
    )

    assert result.failure_category == "invalid_request"
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_today_uses_an_aware_bangkok_timestamp(monkeypatch):
    create = AsyncMock(return_value=_Process(b"ok\n"))
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)

    await adapter.handle_rak_trading_os_authorized_command(_request("/attention today"))

    value = create.await_args.args[-1]
    parsed = adapter.datetime.fromisoformat(value)
    assert create.await_args.args[-2] == "--as-of"
    assert parsed.utcoffset() is not None
    assert (
        parsed.astimezone(adapter.ZoneInfo("Asia/Bangkok")).utcoffset()
        == parsed.utcoffset()
    )


@pytest.mark.asyncio
async def test_nonzero_exit_is_sanitized_and_not_retried(monkeypatch, caplog):
    sentinel = "PRIVATE-SENTINEL-runtime-stderr"
    create = AsyncMock(
        return_value=_Process(b"partial", sentinel.encode(), returncode=4)
    )
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)

    result = await adapter.handle_rak_trading_os_authorized_command(
        _request("/daily_brief 2026-10-20")
    )

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_FAILED
    assert result.failure_category == "runtime_failed"
    assert sentinel not in (result.response or "")
    assert sentinel not in caplog.text
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_start_exception_is_sanitized_and_not_retried(monkeypatch, caplog):
    sentinel = "PRIVATE-SENTINEL-start-exception"
    create = AsyncMock(side_effect=OSError(sentinel))
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)

    result = await adapter.handle_rak_trading_os_authorized_command(
        _request("/daily_guide 2026-10-20")
    )

    assert result.failure_category == "runtime_unavailable"
    assert sentinel not in (result.response or "")
    assert sentinel not in caplog.text
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_timeout_kills_child_and_does_not_retry(monkeypatch):
    process = _HangingProcess()
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(adapter, "_TIMEOUT_SECONDS", 0.001)

    result = await adapter.handle_rak_trading_os_authorized_command(
        _request("/attention 2026-10-20")
    )

    assert result.failure_category == "runtime_timeout"
    assert process.killed is True
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_cancellation_kills_child_without_another_start(monkeypatch):
    process = _HangingProcess()
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)

    task = asyncio.create_task(
        adapter.handle_rak_trading_os_authorized_command(
            _request("/daily_brief 2026-10-20")
        )
    )
    while create.await_count == 0:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed is True
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_oversized_stdout_fails_closed_without_partial_report(monkeypatch):
    create = AsyncMock(return_value=_Process(b"x" * (adapter._MAX_RESPONSE_CHARS + 1)))
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)

    result = await adapter.handle_rak_trading_os_authorized_command(
        _request("/daily_guide 2026-10-20")
    )

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_FAILED
    assert result.failure_category == "runtime_output_too_large"
    assert "x" * 100 not in (result.response or "")
    assert create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "returncode", "handled"),
    [
        ("daily_guide", 0, True),
        ("daily_guide", 1, True),
        ("daily_guide", 2, False),
        ("daily_guide", 4, False),
        ("daily_brief", 0, True),
        ("daily_brief", 1, False),
        ("attention", 0, True),
        ("attention", 1, False),
    ],
)
async def test_fixed_command_exit_contract(
    monkeypatch, caplog, command, returncode, handled
):
    stdout = (
        b"DAILY GUIDE\nStatus: INSPECTION REQUIRED\n"
        b"Read-only and non-authorizing.\nLive trading permission: NOT_PROVIDED\n"
    )
    sentinel = "PRIVATE-exit-contract-stderr"
    create = AsyncMock(return_value=_Process(stdout, sentinel.encode(), returncode))
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)

    result = await adapter.handle_rak_trading_os_authorized_command(
        _request(f"/{command} today")
    )

    if handled:
        assert result.disposition is AuthorizedCommandDisposition.CLAIMED_HANDLED
        assert result.response == stdout.decode()
        assert result.failure_category is None
    else:
        assert result.disposition is AuthorizedCommandDisposition.CLAIMED_FAILED
        assert result.failure_category == "runtime_failed"
        assert result.response == "Trading OS could not complete this request."
    assert sentinel not in (result.response or "")
    assert sentinel not in caplog.text
    assert create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdout", "category"),
    [
        (b"", "runtime_failed"),
        (b"\xff", "runtime_failed"),
        (b"x" * (adapter._MAX_STDOUT_BYTES + 1), "runtime_output_too_large"),
        (
            "ก".encode() * (adapter._MAX_RESPONSE_CHARS + 1),
            "runtime_output_too_large",
        ),
    ],
    ids=["empty", "invalid-utf8", "byte-limit", "character-limit"],
)
async def test_daily_guide_exit_one_still_validates_output(
    monkeypatch, caplog, stdout, category
):
    sentinel = "PRIVATE-invalid-guide-stderr"
    create = AsyncMock(return_value=_Process(stdout, sentinel.encode(), 1))
    monkeypatch.setattr(adapter.asyncio, "create_subprocess_exec", create)

    result = await adapter.handle_rak_trading_os_authorized_command(
        _request("/daily_guide today")
    )

    assert result.disposition is AuthorizedCommandDisposition.CLAIMED_FAILED
    assert result.failure_category == category
    assert sentinel not in (result.response or "")
    assert sentinel not in caplog.text
    assert create.await_count == 1
