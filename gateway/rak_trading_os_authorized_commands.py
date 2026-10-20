"""Rak Trading OS read-only commands behind authorized-command ingress."""

from __future__ import annotations

import asyncio
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from gateway.authorized_command_ingress import (
    AUTHORIZED_COMMAND_INGRESS_API_VERSION,
    AuthorizedCommandDisposition,
    AuthorizedCommandRequest,
    AuthorizedCommandResult,
    register_authorized_command_handler,
)


HANDLER_ID = "rak.trading_os.daily_read_v1"
RESERVED_COMMANDS = ("daily_guide", "daily_brief", "attention")

_PYTHON = (
    "/Users/apirak/.local/share/uv/python/"
    "cpython-3.12.13-macos-aarch64-none/bin/python3.12"
)
_SCRIPTS = "/Users/apirak/rak-trading-os-runtime/portfolios/th-momentum/scripts"
_FLAGS = ("-E", "-s", "-B")
_TIMEOUT_SECONDS = 360.0
_REAP_TIMEOUT_SECONDS = 5.0
_MAX_RESPONSE_CHARS = 16_384
_MAX_STDOUT_BYTES = _MAX_RESPONSE_CHARS * 4
_GRAMMAR = re.compile(
    r"/(?P<command>daily_guide|daily_brief|attention) "
    r"(?P<target>today|[0-9]{4}-[0-9]{2}-[0-9]{2})",
    re.ASCII,
)
_CHILD_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "TMPDIR": "/private/tmp",
    "PYTHONDONTWRITEBYTECODE": "1",
    "GIT_OPTIONAL_LOCKS": "0",
}

_INVALID_REQUEST = "This Trading OS command could not be accepted."
_RUNTIME_UNAVAILABLE = "Trading OS is temporarily unavailable."
_RUNTIME_TIMEOUT = "Trading OS did not respond in time."
_RUNTIME_FAILED = "Trading OS could not complete this request."
_RUNTIME_OUTPUT_TOO_LARGE = (
    "The complete Trading OS response exceeds the safe output limit."
)


@dataclass(frozen=True, slots=True)
class _RuntimeCommand:
    script: str
    intent: str
    target: str
    accepted_returncodes: tuple[int, ...]


_COMMANDS = {
    # Daily Guide uses 1 for valid inspection/refresh-required guidance.
    "daily_guide": (
        f"{_SCRIPTS}/operator_hermes_daily_guide.py",
        "daily-guide",
        (0, 1),
    ),
    "daily_brief": (
        f"{_SCRIPTS}/operator_hermes_daily_read_surface.py",
        "daily-brief",
        (0,),
    ),
    "attention": (
        f"{_SCRIPTS}/operator_hermes_daily_read_surface.py",
        "attention",
        (0,),
    ),
}


def _result(
    disposition: AuthorizedCommandDisposition,
    response: str,
    failure_category: str | None = None,
) -> AuthorizedCommandResult:
    return AuthorizedCommandResult(disposition, response, failure_category)


def _reject_invalid_request() -> AuthorizedCommandResult:
    return _result(
        AuthorizedCommandDisposition.CLAIMED_REJECTED,
        _INVALID_REQUEST,
        "invalid_request",
    )


def _fail(response: str, category: str) -> AuthorizedCommandResult:
    return _result(AuthorizedCommandDisposition.CLAIMED_FAILED, response, category)


def _parse_request(request: AuthorizedCommandRequest) -> _RuntimeCommand | None:
    if (
        type(request) is not AuthorizedCommandRequest
        or request.api_version != AUTHORIZED_COMMAND_INGRESS_API_VERSION
        or request.transport != "telegram"
    ):
        return None
    match = _GRAMMAR.fullmatch(request.raw_text)
    if match is None:
        return None
    command = match.group("command")
    target = match.group("target")
    if (
        request.command_name != command
        or request.raw_command_token != f"/{command}"
        or request.raw_argument_tail != f" {target}"
    ):
        return None
    if target != "today":
        try:
            if date.fromisoformat(target).isoformat() != target:
                return None
        except ValueError:
            return None
    script, intent, accepted_returncodes = _COMMANDS[command]
    return _RuntimeCommand(script, intent, target, accepted_returncodes)


def _bangkok_now_iso() -> str:
    """Return the internal, patchable clock value used only for ``today``."""
    return datetime.now(ZoneInfo("Asia/Bangkok")).isoformat()


def _argv(command: _RuntimeCommand) -> tuple[str, ...]:
    argv = (_PYTHON, *_FLAGS, command.script, command.intent, command.target)
    if command.target == "today":
        return (*argv, "--as-of", _bangkok_now_iso())
    return argv


async def handle_rak_trading_os_authorized_command(
    request: AuthorizedCommandRequest,
) -> AuthorizedCommandResult:
    """Validate one reserved request and invoke its fixed certified read surface."""
    command = _parse_request(request)
    if command is None:
        return _reject_invalid_request()
    argv = _argv(command)
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(_CHILD_ENVIRONMENT),
            start_new_session=True,
        )
    except OSError:
        return _fail(_RUNTIME_UNAVAILABLE, "runtime_unavailable")

    try:
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(), timeout=_TIMEOUT_SECONDS
        )
    except TimeoutError:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        with suppress(ProcessLookupError, TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_REAP_TIMEOUT_SECONDS)
        return _fail(_RUNTIME_TIMEOUT, "runtime_timeout")
    except asyncio.CancelledError:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        with suppress(ProcessLookupError, TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_REAP_TIMEOUT_SECONDS)
        raise
    except OSError:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        with suppress(ProcessLookupError, TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_REAP_TIMEOUT_SECONDS)
        return _fail(_RUNTIME_FAILED, "runtime_failed")

    if process.returncode not in command.accepted_returncodes:
        return _fail(_RUNTIME_FAILED, "runtime_failed")
    if len(stdout) > _MAX_STDOUT_BYTES:
        return _fail(_RUNTIME_OUTPUT_TOO_LARGE, "runtime_output_too_large")
    try:
        rendered = stdout.decode("utf-8")
    except UnicodeDecodeError:
        return _fail(_RUNTIME_FAILED, "runtime_failed")
    if not rendered or len(rendered) > _MAX_RESPONSE_CHARS:
        category = "runtime_output_too_large" if rendered else "runtime_failed"
        response = _RUNTIME_OUTPUT_TOO_LARGE if rendered else _RUNTIME_FAILED
        return _fail(response, category)
    return _result(AuthorizedCommandDisposition.CLAIMED_HANDLED, rendered)


def register_rak_trading_os_authorized_commands() -> None:
    """Register the fixed repository-owned R2.1 handler; this does not enable ingress."""
    register_authorized_command_handler(
        HANDLER_ID,
        RESERVED_COMMANDS,
        handle_rak_trading_os_authorized_command,
    )


__all__ = [
    "HANDLER_ID",
    "RESERVED_COMMANDS",
    "handle_rak_trading_os_authorized_command",
    "register_rak_trading_os_authorized_commands",
]
