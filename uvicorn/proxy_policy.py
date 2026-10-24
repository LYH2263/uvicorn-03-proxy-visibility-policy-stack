"""Request/connection level reverse-proxy trust and client visibility policy stack.

The process-wide ``Config.proxy_headers`` / ``Config.forwarded_allow_ips`` pair
only provides a single global trust table.  This module implements a stack that
can be narrowed or widened for the lifetime of a single HTTP request (or a
single WebSocket connection after the handshake) without mutating process
configuration:

* ``forwarded_trust`` controls who is trusted to set ``X-Forwarded-*`` headers;
* ``scope_rewrite`` controls whether (and which parts of) the ASGI ``scope`` is
  rewritten when a trusted peer is found;
* ``access_client_mode`` decouples the client field written to the access log
  from ``scope["client"]``.

The stack is stored in a :class:`contextvars.ContextVar`.  Every request (and
every HTTP/2 stream) runs in its own task and therefore gets its own copy of
the stack: two concurrent connections pushing different frames can never see
each other's frames.  The lifespan scope never participates.

The stack is deliberately independent from ASGI ``scope["state"]``, from the
ASGI ``contextvars`` reset policy and from transport flow control.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

from uvicorn._types import WWWScope
from uvicorn.middleware.proxy_headers import _parse_host_port, _parse_raw_hosts, _TrustedHosts
from uvicorn.protocols.utils import get_client_addr

ForwardedTrustMode = Literal["inherit", "extend", "force_trust", "force_distrust"]
ScopeRewriteMode = Literal["none", "client_only", "scheme_only", "both"]
AccessClientMode = Literal["scope_client", "transport_peer", "xff_leftmost", "xff_untrusted"]

_FORWARDED_TRUST_MODES = frozenset({"inherit", "extend", "force_trust", "force_distrust"})
_SCOPE_REWRITE_MODES = frozenset({"none", "client_only", "scheme_only", "both"})
_ACCESS_CLIENT_MODES = frozenset({"scope_client", "transport_peer", "xff_leftmost", "xff_untrusted"})


class ForwardedTrust(NamedTuple):
    """Trust directive carried by a single stack frame.

    * ``inherit``  - merge the process ``Config.forwarded_allow_ips`` baseline,
      adding nothing.  An ``inherit`` frame nested inside ``force_distrust``
      cannot restore trust; only popping the distrust frame can.
    * ``extend``   - trust the baseline plus the given hosts/networks.
    * ``force_trust`` - treat the directly connected peer as trusted for this
      frame, allowing ``X-Forwarded-For`` / ``X-Forwarded-Proto`` parsing even
      when it is not in the baseline table.  Extra hosts are appended to the
      table used for the reverse walk.
    * ``force_distrust`` - ignore every forwarding header in this frame.
    """

    mode: ForwardedTrustMode
    hosts: tuple[str, ...] = ()

    @staticmethod
    def inherit() -> ForwardedTrust:
        return ForwardedTrust("inherit")

    @staticmethod
    def extend(hosts: str | Sequence[str]) -> ForwardedTrust:
        return ForwardedTrust("extend", _coerce_hosts(hosts))

    @staticmethod
    def force_trust(hosts: str | Sequence[str] | None = None) -> ForwardedTrust:
        return ForwardedTrust("force_trust", () if hosts is None else _coerce_hosts(hosts))

    @staticmethod
    def force_distrust() -> ForwardedTrust:
        return ForwardedTrust("force_distrust")


def _coerce_hosts(hosts: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(hosts, str):
        return tuple(_parse_raw_hosts(hosts))
    return tuple(hosts)


@dataclass(frozen=True)
class ProxyPolicyFrame:
    """A single stack frame.

    Every field is optional; leaving a field unset means the frame inherits the
    effective value of the surrounding frame (or the process baseline when no
    outer frame exists).  Popping a frame therefore always restores the exact
    outer user policy - it never falls back to a process-wide singleton.
    """

    forwarded_trust: ForwardedTrust | None = None
    scope_rewrite: ScopeRewriteMode | None = None
    access_client_mode: AccessClientMode | None = None


def _validate_frame(frame: ProxyPolicyFrame) -> None:
    if frame.forwarded_trust is not None:
        trust = frame.forwarded_trust
        if trust.mode not in _FORWARDED_TRUST_MODES:
            raise ValueError(f"Invalid forwarded_trust mode: {trust.mode!r}")
        if trust.mode in {"extend", "force_trust"}:
            if trust.mode == "extend" and not trust.hosts:
                raise ValueError("forwarded_trust 'extend' requires at least one host or network")
        elif trust.hosts:
            raise ValueError(f"forwarded_trust {trust.mode!r} does not accept hosts")
    if frame.scope_rewrite is not None and frame.scope_rewrite not in _SCOPE_REWRITE_MODES:
        raise ValueError(f"Invalid scope_rewrite mode: {frame.scope_rewrite!r}")
    if frame.access_client_mode is not None and frame.access_client_mode not in _ACCESS_CLIENT_MODES:
        raise ValueError(f"Invalid access_client_mode: {frame.access_client_mode!r}")


#: The shared mount point.  An empty tuple means "no frame pushed"; the value
#: is copied per asyncio task, so concurrent connections / HTTP/2 streams get
#: independent stacks.
_policy_stack: contextvars.ContextVar[tuple[ProxyPolicyFrame, ...]] = contextvars.ContextVar(
    "uvicorn_proxy_policy_stack", default=()
)


def push_frame(frame: ProxyPolicyFrame) -> contextvars.Token[tuple[ProxyPolicyFrame, ...]]:
    """Push a frame onto the current request/connection stack.

    Returns a token that MUST be handed back to :func:`reset_frame`.  Prefer
    :func:`push_proxy_policy` for automatic cleanup on exception paths.
    """
    _validate_frame(frame)
    stack = _policy_stack.get()
    return _policy_stack.set(stack + (frame,))


def reset_frame(token: contextvars.Token[tuple[ProxyPolicyFrame, ...]]) -> None:
    """Pop the frame pushed with the matching token."""
    _policy_stack.reset(token)


@contextlib.contextmanager
def push_proxy_policy(
    *,
    forwarded_trust: ForwardedTrust | None = None,
    scope_rewrite: ScopeRewriteMode | None = None,
    access_client_mode: AccessClientMode | None = None,
) -> Iterator[None]:
    """Context manager that pops the frame automatically, including on errors."""
    frame = ProxyPolicyFrame(
        forwarded_trust=forwarded_trust,
        scope_rewrite=scope_rewrite,
        access_client_mode=access_client_mode,
    )
    token = push_frame(frame)
    try:
        yield
    finally:
        reset_frame(token)


def current_stack() -> tuple[ProxyPolicyFrame, ...]:
    """Return a snapshot of the frames active in the current context."""
    return _policy_stack.get()


def _normalize_baseline(baseline: str | Sequence[str] | None) -> tuple[str, ...]:
    if baseline is None:
        return ()
    if isinstance(baseline, str):
        return tuple(_parse_raw_hosts(baseline))
    return tuple(baseline)


@functools.lru_cache(maxsize=512)
def _merged_trusted_hosts(
    baseline: tuple[str, ...], extras: tuple[str, ...]
) -> _TrustedHosts:
    if "*" in baseline:
        return _TrustedHosts("*")
    hosts = list(baseline)
    hosts.extend(extras)
    return _TrustedHosts(hosts or [])


@dataclass(frozen=True)
class EffectiveProxyPolicy:
    """Merged view of the stack top and the process ``Config`` baseline."""

    #: Whether the baseline middleware is enabled (``Config.proxy_headers``).
    enabled: bool
    #: Effective trust directive after folding the whole stack.
    trust_mode: ForwardedTrustMode
    #: Trust table containing the baseline plus every ``extend``/``force_trust``
    #: extra host.  Only consulted once a peer gate passes.
    trusted_hosts: _TrustedHosts
    #: Whether ``scope["client"]`` / ``scope["scheme"]`` may be written.
    scope_rewrite: ScopeRewriteMode
    #: How the access log client field is resolved.
    access_client_mode: AccessClientMode
    #: Stack depth that produced this view (mainly for introspection/tests).
    depth: int = field(compare=False, default=0)

    @property
    def rewrite_client(self) -> bool:
        return self.scope_rewrite in ("client_only", "both")

    @property
    def rewrite_scheme(self) -> bool:
        return self.scope_rewrite in ("scheme_only", "both")

    def peer_is_trusted(self, peer_host: str | None) -> bool:
        """Trust decision for the directly connected peer at middleware entry."""
        if self.trust_mode == "force_distrust":
            return False
        if self.trust_mode == "force_trust":
            return True
        return peer_host in self.trusted_hosts


def effective_proxy_policy(
    baseline_forwarded_allow_ips: str | Sequence[str] | None,
    proxy_headers_enabled: bool,
) -> EffectiveProxyPolicy:
    """Fold the current stack on top of the process baseline.

    With an empty stack the result is observably identical to only having
    ``ProxyHeadersMiddleware`` (when ``proxy_headers_enabled`` is true) or to
    having no rewriting at all (when it is false).
    """
    frames = _policy_stack.get()

    scope_rewrite: ScopeRewriteMode = "both" if proxy_headers_enabled else "none"
    access_client_mode: AccessClientMode = "scope_client"
    trust_mode: ForwardedTrustMode = "inherit"
    extras: list[str] = []

    for frame in frames:
        trust = frame.forwarded_trust
        if trust is not None:
            if trust.mode == "force_distrust":
                # Sticky: an inner `inherit` cannot pretend trust back, and an
                # `extend` adds nothing while distrust is active.
                trust_mode = "force_distrust"
            elif trust.mode == "force_trust":
                trust_mode = "force_trust"
                extras.extend(trust.hosts)
            elif trust.mode == "extend":
                if trust_mode != "force_distrust":
                    if trust_mode == "inherit":
                        trust_mode = "extend"
                    extras.extend(trust.hosts)
            # "inherit" keeps the surrounding policy unchanged on purpose.
        if frame.scope_rewrite is not None:
            scope_rewrite = frame.scope_rewrite
        if frame.access_client_mode is not None:
            access_client_mode = frame.access_client_mode

    baseline = _normalize_baseline(baseline_forwarded_allow_ips)
    trusted_hosts = _merged_trusted_hosts(baseline, tuple(extras))

    return EffectiveProxyPolicy(
        enabled=proxy_headers_enabled,
        trust_mode=trust_mode,
        trusted_hosts=trusted_hosts,
        scope_rewrite=scope_rewrite,
        access_client_mode=access_client_mode,
        depth=len(frames),
    )


def _extract_x_forwarded_for(headers: Iterable[tuple[bytes, bytes]]) -> str:
    values: list[bytes] = []
    for name, value in headers:
        if name == b"x-forwarded-for":
            values.append(value)
    if not values:
        return ""
    return b", ".join(values).decode("latin1")


def _format_client_addr(client: tuple[str, int] | None) -> str:
    if not client:
        return ""
    return "%s:%d" % client


def access_client_addr(
    config: object,
    scope: WWWScope,
    transport_client: tuple[str, int] | None,
) -> str:
    """Resolve the client field for an access log record.

    ``transport_client`` is the peer captured when the transport was created;
    unlike ``scope["client"]`` it is never rewritten, so the
    ``transport_peer`` / ``xff_*`` modes keep working even when the scope was
    (or was not) rewritten by ``ProxyHeadersMiddleware``.
    """
    view = effective_proxy_policy(
        getattr(config, "forwarded_allow_ips", None),
        bool(getattr(config, "proxy_headers", False)),
    )
    return resolve_access_client(scope, transport_client, view)


def resolve_access_client(
    scope: WWWScope,
    transport_client: tuple[str, int] | None,
    view: EffectiveProxyPolicy,
) -> str:
    """Resolve the access-log client according to ``view.access_client_mode``."""
    mode = view.access_client_mode

    if mode == "scope_client":
        return get_client_addr(scope)

    if mode == "transport_peer":
        return _format_client_addr(transport_client)

    # The header based modes must obey an active force_distrust directive:
    # forwarding headers are not trustworthy in this frame at all.
    if view.trust_mode == "force_distrust":
        return _format_client_addr(transport_client)

    x_forwarded_for = _extract_x_forwarded_for(scope.get("headers", ()))
    if not x_forwarded_for:
        return _format_client_addr(transport_client)

    hosts = _parse_raw_hosts(x_forwarded_for)

    if mode == "xff_leftmost":
        # The left-most hop, without requiring it to pass the trust chain.
        host, port = _parse_host_port(hosts[0])
        if not host:
            return _format_client_addr(transport_client)
        return _format_client_addr((host, port))

    if mode == "xff_untrusted":
        # Reverse walk over the merged trust table to find the untrusted
        # boundary address.  Unlike middleware entry this does NOT require the
        # directly connected peer itself to be trusted.
        host, port = view.trusted_hosts.get_trusted_client_address(x_forwarded_for)
        if not host:
            return _format_client_addr(transport_client)
        return _format_client_addr((host, port))

    raise RuntimeError(f"Unknown access_client_mode: {mode!r}")  # pragma: no cover
