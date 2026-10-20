"""Inbound message event types shared by every gateway platform adapter.

A leaf module: adapters, helpers and the runner import it, so it must not import from
gateway.platforms.*.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from gateway.session import SessionSource

# Desktop attachment reference tags prepended by buildContextText before the
# user's visible text (e.g. "@image:/tmp/foo.png\n\n/moa ask something").
# Strip these when detecting slash commands so a media-ref prefix does not hide
# a slash token from MessageEvent.is_command() / get_command().
# The pattern matches to end-of-line (not just whitespace-bounded) to handle
# Windows paths that may contain spaces (e.g. "C:\Users\John Doe\image.png").
_ATTACHMENT_REF_RE = re.compile(r"^(?:@(?:image|file|url):[^\n]+\n?)+", re.IGNORECASE)


class MessageType(Enum):
    """Types of incoming messages."""
    TEXT = "text"
    LOCATION = "location"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    STICKER = "sticker"
    COMMAND = "command"  # /command style


class ProcessingOutcome(Enum):
    """Result classification for message-processing lifecycle hooks."""
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass(eq=False, slots=True)
class _InboundEventState:
    """Shared process-local state retained by copies and merged event constituents."""

    _parent: Optional["_InboundEventState"] = field(default=None, init=False, repr=False)
    transport_evidence: tuple[Any, ...] = ()
    transport_constituent_count: int = 1
    command_started: bool = False
    command_result: Any = None
    command_future: Any = field(default=None, repr=False)
    response_delivery_started: bool = False

    def root(self) -> "_InboundEventState":
        state = self
        while state._parent is not None:
            state = state._parent
        cursor = self
        while cursor._parent is not None and cursor._parent is not state:
            parent = cursor._parent
            cursor._parent = state
            cursor = parent
        return state

    def absorb(self, other: "_InboundEventState") -> None:
        left, right = self.root(), other.root()
        if left is right:
            return
        if left.command_started or right.command_started:
            raise ValueError("cannot merge independently executing inbound events")
        right._parent = left
        left.transport_evidence += tuple(
            item for item in right.transport_evidence
            if all(item is not existing for existing in left.transport_evidence)
        )
        left.transport_constituent_count += right.transport_constituent_count
        left.command_started = left.command_started or right.command_started
        if left.command_result is None:
            left.command_result = right.command_result
        if left.command_future is None:
            left.command_future = right.command_future
        left.response_delivery_started |= right.response_delivery_started


@dataclass
class MessageEvent:
    """Incoming message from a platform — the normalized shape all adapters produce."""
    text: str
    message_type: MessageType = MessageType.TEXT
    # Author, mirrored from ``source`` for per-message prompt builders; None for non-IM sources.
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    # None only in isolated unit tests; production always sets it. Typing it Optional
    # exposes ~60 unguarded ``.source.<attr>`` reads, so that is a separate change.
    source: SessionSource = None
    raw_message: Any = None
    message_id: Optional[str] = None
    # Delivery-ledger identity for the final send, when it differs from ``message_id``. A queued
    # (/queue) chain answers the LAST message of the chain, so its final send has to be ledgered
    # under that message's id. Keyed on the opening event's id instead, two chained turns carrying
    # the same text collide on one obligation id and the earlier turn's row is overwritten (a
    # refused first reply then reads as delivered). Reply routing is unaffected: the reply anchor
    # still comes from this event.
    ledger_message_id: Optional[str] = None
    # Reply anchor for the final send when the answer is to a DIFFERENT message than the one that
    # opened the turn: a successful busy redirect turns the running turn onto the redirecting
    # message, so its reply must quote that message (#115001). ``_reply_anchor_for_event``
    # honours this over ``message_id``; None = derive from the event as usual.
    reply_anchor_override: Optional[str] = None
    # Platform update id (Telegram ``update_id``): ``/restart`` records it so the new gateway
    # advances past it even if PTB's shutdown ACK times out.
    platform_update_id: Optional[int] = None
    # Media attachments: local file paths (for vision tool access)
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    # Per-attachment text-inlining contract; None = legacy "text/* already inlined into ``text``".
    media_text_inlined: List[Optional[bool]] = field(default_factory=list)
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None  # Text of the replied-to message (for context injection)
    reply_to_author_id: Optional[str] = None
    reply_to_author_name: Optional[str] = None
    reply_to_is_own_message: bool = False  # True when the user replied to this bot/assistant's message
    # Structured interactive-prompt reply (relay only): {prompt_id, option_id, label?,
    # prompt_message_id?}; routed to the approval/slash-confirm/clarify resolvers BEFORE dispatch.
    prompt_response: Optional[Dict[str, Any]] = None
    # Auto-loaded skill(s) for topic/channel bindings; a single name or ordered list.
    auto_skill: Optional[str | list[str]] = None
    # Per-channel ephemeral system prompt; applied at API call time, never persisted to transcript.
    channel_prompt: Optional[str] = None
    # History-backfilled channel context (missed under require_mention); kept out of ``text`` so
    # run.py's sender-prefix logic sees only the trigger message.
    channel_context: Optional[str] = None
    # Set for synthetic events (e.g. background-process notifications) that must bypass user authorization.
    internal: bool = False
    # Free-form per-event metadata (e.g. ``whatsapp_from_owner=True``); plugins must ``.get()``.
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    # May this event resolve gateway commands / control prompts? Proactive plugin events set False
    # so untrusted payload text stays conversational. New fields append after it (positional compat).
    allow_gateway_control: bool = True
    # Was this message addressed to this bot? False lets a bare silence marker stand (the adapter
    # knows the message was meant for someone else); None means unknown and keeps the visible
    # fallback, like True.
    reply_expected: Optional[bool] = None

    # Process-local admission receipt, never routing metadata or execution acknowledgement.
    _gateway_accepted: bool = field(default=False, init=False, repr=False, compare=False)
    # Process-local state is an ordinary dataclass field so ``dataclasses.replace`` retains the
    # same identity. It is neither serialized nor an authorization capability.
    _inbound_event_state: _InboundEventState = field(
        default_factory=_InboundEventState, repr=False, compare=False
    )
    # Run-owned final presentation snapshot; never deserialized from ingress metadata.
    _notification_reply_muted: Optional[bool] = field(default=None, init=False, repr=False, compare=False)

    def absorb_reply_expected(self, other: "MessageEvent") -> None:
        """Absorb reply intent and process-local identity from a merged constituent."""
        if self.reply_expected is not True and other.reply_expected is not False:
            self.reply_expected = other.reply_expected
        self._inbound_event_state.absorb(other._inbound_event_state)

    def _inherit_inbound_event_state_from(self, original: "MessageEvent") -> None:
        """Carry transport identity through a hook replacement that did not copy the event."""
        self._inbound_event_state = original._inbound_event_state

    def _inbound_event_state_snapshot(self) -> tuple[Any, ...]:
        """Opaque hook-boundary checkpoint; state is not a plugin authority surface."""
        state = self._inbound_event_state.root()
        return (
            state,
            state.transport_evidence,
            state.transport_constituent_count,
            state.command_started,
            state.command_result,
            state.command_future,
            state.response_delivery_started,
        )

    def _inbound_event_state_matches(self, snapshot: tuple[Any, ...]) -> bool:
        state = self._inbound_event_state.root()
        return (
            len(snapshot) == 7
            and state is snapshot[0]
            and len(state.transport_evidence) == len(snapshot[1])
            and all(
                current is previous
                for current, previous in zip(state.transport_evidence, snapshot[1])
            )
            and state.transport_constituent_count == snapshot[2]
            and state.command_started is snapshot[3]
            and state.command_result is snapshot[4]
            and state.command_future is snapshot[5]
            and state.response_delivery_started is snapshot[6]
        )

    @property
    def _inbound_transport_evidence(self) -> Any:
        evidence = self._inbound_event_state.root().transport_evidence
        return evidence[0] if evidence else None

    @_inbound_transport_evidence.setter
    def _inbound_transport_evidence(self, evidence: Any) -> None:
        state = self._inbound_event_state.root()
        if evidence is None:
            return
        if state.transport_evidence and all(
            evidence is not existing for existing in state.transport_evidence
        ):
            raise ValueError("inbound transport evidence is immutable once captured")
        if not state.transport_evidence:
            state.transport_evidence = (evidence,)

    @property
    def _inbound_transport_evidences(self) -> tuple[Any, ...]:
        return self._inbound_event_state.root().transport_evidence

    @property
    def _inbound_transport_constituent_count(self) -> int:
        return self._inbound_event_state.root().transport_constituent_count

    @property
    def _authorized_command_ingress_started(self) -> bool:
        return self._inbound_event_state.root().command_started

    @_authorized_command_ingress_started.setter
    def _authorized_command_ingress_started(self, value: bool) -> None:
        self._inbound_event_state.root().command_started = bool(value)

    @property
    def _authorized_command_ingress_result(self) -> Any:
        return self._inbound_event_state.root().command_result

    @_authorized_command_ingress_result.setter
    def _authorized_command_ingress_result(self, result: Any) -> None:
        state = self._inbound_event_state.root()
        state.command_started = True
        state.command_result = result
        future = state.command_future
        if future is not None and not future.done():
            future.set_result(result)

    def _begin_authorized_command_ingress(self, future: Any) -> tuple[bool, Any]:
        state = self._inbound_event_state.root()
        if state.command_result is not None or state.command_started:
            return False, state.command_future
        state.command_started = True
        state.command_future = future
        return True, future

    def _claim_authorized_command_response_delivery(self) -> bool:
        state = self._inbound_event_state.root()
        if state.response_delivery_started:
            return False
        state.response_delivery_started = True
        return True

    def _command_text(self) -> str:
        """Return the message text with leading Desktop attachment refs stripped.

        Desktop's buildContextText prepends ``@image:<path>``, ``@file:<path>``,
        or ``@url:<url>`` tags before the user's visible text.  Stripping these
        lets is_command / get_command / get_command_args work correctly even
        when the payload is prefixed with one or more media refs.
        """
        return _ATTACHMENT_REF_RE.sub("", (self.text or "").lstrip()).lstrip()

    def is_command(self) -> bool:
        """Check if this is a command message (e.g., /new, /reset)."""
        return self.allow_gateway_control and self._command_text().startswith("/")

    def get_command(self) -> Optional[str]:
        """Extract command name if this is a command message."""
        if not self.is_command():
            return None
        raw = self._command_text().split(maxsplit=1)[0][1:].lower().split("@", 1)[0]
        # Reject file paths: valid command names never contain /
        return None if "/" in raw else raw

    def get_command_args(self) -> str:
        """Get the arguments after a command."""
        if not self.is_command():
            return self.text
        parts = self._command_text().lstrip().split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        # iOS auto-corrects -- to — (em dash) and - to – (en dash)
        return args.replace("\u2014\u2014", "--").replace("\u2014", "--").replace("\u2013", "-")
