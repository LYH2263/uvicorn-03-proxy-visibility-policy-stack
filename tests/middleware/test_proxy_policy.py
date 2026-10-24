"""Unit tests for the request/connection level proxy policy stack.

Covers:
* push / pop / auto-cleanup context manager and nesting semantics;
* the effective merged view (inherit / extend / force_trust / force_distrust);
* ProxyHeadersMiddleware entry peer judgment, XFF reverse walk and the
  scope_rewrite fork (none / client_only / scheme_only / both);
* access client resolution modes including the force_distrust fallback;
* lifespan never subscribing and concurrent task isolation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from uvicorn._types import ASGIReceiveCallable, ASGIReceiveEvent, ASGISendCallable, ASGISendEvent, HTTPScope, Scope
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from uvicorn.proxy_policy import (
    EffectiveProxyPolicy,
    ForwardedTrust,
    ProxyPolicyFrame,
    access_client_addr,
    current_stack,
    effective_proxy_policy,
    push_frame,
    push_proxy_policy,
    reset_frame,
    resolve_access_client,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _noop_receive() -> ASGIReceiveEvent:  # pragma: no cover
    return {"type": "http.disconnect"}


async def _noop_send(message: ASGISendEvent) -> None:  # pragma: no cover
    return None


def _make_scope(
    headers: list[tuple[bytes, bytes]] | None = None,
    *,
    scope_type: str = "http",
    client: tuple[str, int] | None = ("127.0.0.1", 54321),
    scheme: str = "http",
) -> HTTPScope:
    scope: dict[str, Any] = {
        "type": scope_type,
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": scheme,
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": headers or [],
        "client": client,
        "server": ("127.0.0.1", 80),
    }
    if scope_type == "websocket":
        scope["subprotocols"] = []
    return scope  # type: ignore[return-value]


class _ConfigStub:
    def __init__(self, forwarded_allow_ips: str | list[str] = "127.0.0.1", proxy_headers: bool = True) -> None:
        self.forwarded_allow_ips = forwarded_allow_ips
        self.proxy_headers = proxy_headers


FORWARDING_HEADERS = [
    (b"x-forwarded-for", b"203.0.113.5, 10.0.0.10"),
    (b"x-forwarded-proto", b"https"),
]


# ---------------------------------------------------------------------------
# stack mechanics
# ---------------------------------------------------------------------------


async def test_empty_stack_matches_baseline_when_enabled() -> None:
    view = effective_proxy_policy("127.0.0.1", proxy_headers_enabled=True)
    assert view.depth == 0
    assert view.trust_mode == "inherit"
    assert view.scope_rewrite == "both"
    assert view.access_client_mode == "scope_client"
    assert view.peer_is_trusted("127.0.0.1") is True
    assert view.peer_is_trusted("10.0.0.10") is False


async def test_empty_stack_does_not_enable_rewrite_when_disabled() -> None:
    view = effective_proxy_policy("127.0.0.1", proxy_headers_enabled=False)
    assert view.scope_rewrite == "none"
    assert view.rewrite_client is False
    assert view.rewrite_scheme is False
    assert view.access_client_mode == "scope_client"


async def test_push_pop_with_tokens_restores_outer_policy() -> None:
    outer = push_frame(ProxyPolicyFrame(access_client_mode="transport_peer"))
    try:
        view = effective_proxy_policy("127.0.0.1", True)
        assert view.access_client_mode == "transport_peer"

        inner = push_frame(ProxyPolicyFrame(access_client_mode="xff_leftmost"))
        try:
            assert effective_proxy_policy("127.0.0.1", True).access_client_mode == "xff_leftmost"
        finally:
            reset_frame(inner)

        # Back to the outer user policy, never to a process-wide singleton.
        view = effective_proxy_policy("127.0.0.1", True)
        assert view.access_client_mode == "transport_peer"
        assert view.depth == 1
    finally:
        reset_frame(outer)

    assert effective_proxy_policy("127.0.0.1", True).access_client_mode == "scope_client"


async def test_context_manager_pops_on_exception() -> None:
    assert current_stack() == ()
    with pytest.raises(RuntimeError):
        with push_proxy_policy(scope_rewrite="none"):
            assert len(current_stack()) == 1
            raise RuntimeError("boom")
    assert current_stack() == ()
    assert effective_proxy_policy("127.0.0.1", True).scope_rewrite == "both"


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (ProxyPolicyFrame(scope_rewrite="bogus"), "scope_rewrite"),
        (ProxyPolicyFrame(access_client_mode="bogus"), "access_client_mode"),
        (ProxyPolicyFrame(forwarded_trust=ForwardedTrust("bogus")), "forwarded_trust"),
        (ProxyPolicyFrame(forwarded_trust=ForwardedTrust.extend([])), "extend"),
        (ProxyPolicyFrame(forwarded_trust=ForwardedTrust("inherit", ("1.2.3.4",))), "inherit"),
    ],
)
async def test_invalid_frames_are_rejected(frame: ProxyPolicyFrame, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        push_frame(frame)
    assert current_stack() == ()


# ---------------------------------------------------------------------------
# trust folding
# ---------------------------------------------------------------------------


async def test_extend_adds_baseline_hosts() -> None:
    with push_proxy_policy(forwarded_trust=ForwardedTrust.extend("10.0.0.0/8")):
        view = effective_proxy_policy("127.0.0.1", True)
        assert view.trust_mode == "extend"
        assert view.peer_is_trusted("10.0.0.10") is True
        assert view.peer_is_trusted("192.168.0.1") is False


async def test_force_trust_passes_peer_gate_for_untrusted_peer() -> None:
    with push_proxy_policy(forwarded_trust=ForwardedTrust.force_trust()):
        view = effective_proxy_policy("127.0.0.1", True)
        assert view.trust_mode == "force_trust"
        assert view.peer_is_trusted("203.0.113.9") is True


async def test_force_distrust_fails_peer_gate_even_for_baseline_trusted_peer() -> None:
    with push_proxy_policy(forwarded_trust=ForwardedTrust.force_distrust()):
        view = effective_proxy_policy("127.0.0.1", True)
        assert view.trust_mode == "force_distrust"
        assert view.peer_is_trusted("127.0.0.1") is False
        assert view.peer_is_trusted("*") is False


async def test_inner_inherit_cannot_escape_force_distrust() -> None:
    with push_proxy_policy(forwarded_trust=ForwardedTrust.force_distrust()):
        with push_proxy_policy(forwarded_trust=ForwardedTrust.inherit()):
            view = effective_proxy_policy("127.0.0.1", True)
            assert view.trust_mode == "force_distrust"
            assert view.peer_is_trusted("127.0.0.1") is False
        # popping the inner frame restores the outer distrust
        assert effective_proxy_policy("127.0.0.1", True).trust_mode == "force_distrust"


async def test_inner_extend_cannot_escape_force_distrust() -> None:
    with push_proxy_policy(forwarded_trust=ForwardedTrust.force_distrust()):
        with push_proxy_policy(forwarded_trust=ForwardedTrust.extend("10.0.0.0/8")):
            assert effective_proxy_policy("127.0.0.1", True).trust_mode == "force_distrust"


async def test_inner_force_trust_overrides_outer_distrust_until_pop() -> None:
    with push_proxy_policy(forwarded_trust=ForwardedTrust.force_distrust()):
        with push_proxy_policy(forwarded_trust=ForwardedTrust.force_trust()):
            view = effective_proxy_policy("127.0.0.1", True)
            assert view.trust_mode == "force_trust"
            assert view.peer_is_trusted("203.0.113.9") is True
        assert effective_proxy_policy("127.0.0.1", True).trust_mode == "force_distrust"


async def test_partial_frame_inherits_unset_fields_from_outer() -> None:
    with push_proxy_policy(forwarded_trust=ForwardedTrust.extend("10.0.0.0/8"), scope_rewrite="none"):
        with push_proxy_policy(access_client_mode="transport_peer"):
            view = effective_proxy_policy("127.0.0.1", True)
            assert view.trust_mode == "extend"
            assert view.peer_is_trusted("10.0.0.9") is True
            assert view.scope_rewrite == "none"
            assert view.access_client_mode == "transport_peer"
        view = effective_proxy_policy("127.0.0.1", True)
        assert view.access_client_mode == "scope_client"
        assert view.scope_rewrite == "none"


# ---------------------------------------------------------------------------
# middleware: entry peer judgment + scope_rewrite fork
# ---------------------------------------------------------------------------


async def _run_middleware(
    scope: Scope,
    *,
    trusted_hosts: str | list[str] = "127.0.0.1",
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        captured["scheme"] = scope["scheme"]
        captured["client"] = scope["client"]

    middleware = ProxyHeadersMiddleware(app, trusted_hosts=trusted_hosts)
    await middleware(scope, _noop_receive, _noop_send)
    return captured


async def test_middleware_empty_stack_unchanged_behavior_trusted() -> None:
    scope = _make_scope(FORWARDING_HEADERS)
    captured = await _run_middleware(scope)
    assert captured == {"scheme": "https", "client": ("10.0.0.10", 0)}


async def test_middleware_empty_stack_unchanged_behavior_untrusted() -> None:
    scope = _make_scope(FORWARDING_HEADERS, client=("192.168.0.55", 54321))
    captured = await _run_middleware(scope, trusted_hosts="127.0.0.1")
    assert captured == {"scheme": "http", "client": ("192.168.0.55", 54321)}


async def test_middleware_force_distrust_ignores_forwarding_headers() -> None:
    scope = _make_scope(FORWARDING_HEADERS)
    with push_proxy_policy(forwarded_trust=ForwardedTrust.force_distrust()):
        captured = await _run_middleware(scope)
    assert captured == {"scheme": "http", "client": ("127.0.0.1", 54321)}


async def test_middleware_force_trust_rewrites_for_untrusted_peer() -> None:
    scope = _make_scope(FORWARDING_HEADERS, client=("203.0.113.9", 4000))
    with push_proxy_policy(forwarded_trust=ForwardedTrust.force_trust()):
        captured = await _run_middleware(scope, trusted_hosts="127.0.0.1")
    # peer gate forced open; the XFF reverse walk still uses the baseline table
    assert captured == {"scheme": "https", "client": ("10.0.0.10", 0)}


async def test_middleware_extend_trusts_peer_from_added_network() -> None:
    headers = [
        (b"x-forwarded-for", b"203.0.113.5, 198.51.100.1"),
        (b"x-forwarded-proto", b"https"),
    ]
    scope = _make_scope(headers, client=("10.0.0.99", 4000))
    with push_proxy_policy(forwarded_trust=ForwardedTrust.extend("10.0.0.0/8")):
        captured = await _run_middleware(scope)
    # Peer passes the gate via the added network; 198.51.100.1 is not in the
    # merged table, so the reverse walk stops there.
    assert captured == {"scheme": "https", "client": ("198.51.100.1", 0)}


async def test_middleware_extend_merges_for_reverse_walk() -> None:
    headers = [
        (b"x-forwarded-for", b"203.0.113.5, 10.0.0.10"),
        (b"x-forwarded-proto", b"https"),
    ]
    scope = _make_scope(headers)
    with push_proxy_policy(forwarded_trust=ForwardedTrust.extend("10.0.0.0/8")):
        captured = await _run_middleware(scope)
    # 10.0.0.10 is trusted in the merged table, so the walk reaches 203.0.113.5
    assert captured == {"scheme": "https", "client": ("203.0.113.5", 0)}


async def test_middleware_scope_rewrite_none_changes_neither() -> None:
    scope = _make_scope(FORWARDING_HEADERS)
    with push_proxy_policy(scope_rewrite="none"):
        captured = await _run_middleware(scope)
    assert captured == {"scheme": "http", "client": ("127.0.0.1", 54321)}


async def test_middleware_scope_rewrite_client_only() -> None:
    scope = _make_scope(FORWARDING_HEADERS)
    with push_proxy_policy(scope_rewrite="client_only"):
        captured = await _run_middleware(scope)
    assert captured == {"scheme": "http", "client": ("10.0.0.10", 0)}


async def test_middleware_scope_rewrite_scheme_only_http() -> None:
    scope = _make_scope(FORWARDING_HEADERS)
    with push_proxy_policy(scope_rewrite="scheme_only"):
        captured = await _run_middleware(scope)
    assert captured == {"scheme": "https", "client": ("127.0.0.1", 54321)}


async def test_middleware_scope_rewrite_scheme_only_websocket_maps_http_to_ws() -> None:
    scope = _make_scope(FORWARDING_HEADERS, scope_type="websocket", scheme="ws")
    with push_proxy_policy(scope_rewrite="scheme_only"):
        captured = await _run_middleware(scope)
    assert captured == {"scheme": "wss", "client": ("127.0.0.1", 54321)}


async def test_middleware_scope_rewrite_both_websocket() -> None:
    scope = _make_scope(FORWARDING_HEADERS, scope_type="websocket", scheme="ws")
    with push_proxy_policy(scope_rewrite="both"):
        captured = await _run_middleware(scope)
    assert captured == {"scheme": "wss", "client": ("10.0.0.10", 0)}


async def test_middleware_lifespan_does_not_subscribe_to_stack() -> None:
    scope = {"type": "lifespan"}
    seen: dict[str, str] = {}

    async def app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        seen["type"] = scope["type"]

    middleware = ProxyHeadersMiddleware(app)
    # A force_distrust frame must not affect lifespan at all.
    with push_proxy_policy(forwarded_trust=ForwardedTrust.force_distrust()):
        await middleware(scope, _noop_receive, _noop_send)
    assert seen == {"type": "lifespan"}


# ---------------------------------------------------------------------------
# access client resolution
# ---------------------------------------------------------------------------


def _view(
    *,
    baseline: str | list[str] = "127.0.0.1",
    enabled: bool = True,
    trust: ForwardedTrust | None = None,
    mode: str | None = None,
) -> EffectiveProxyPolicy:
    if trust is not None:
        token = push_frame(ProxyPolicyFrame(forwarded_trust=trust, access_client_mode=mode))
    elif mode is not None:
        token = push_frame(ProxyPolicyFrame(access_client_mode=mode))
    else:  # pragma: no cover
        token = None
    try:
        return effective_proxy_policy(baseline, enabled)
    finally:
        if token is not None:
            reset_frame(token)


async def test_access_scope_client_follows_scope() -> None:
    scope = _make_scope()
    scope["client"] = ("9.9.9.9", 1111)
    view = _view(mode="scope_client")
    assert resolve_access_client(scope, ("127.0.0.1", 54321), view) == "9.9.9.9:1111"


async def test_access_transport_peer_ignores_rewritten_scope() -> None:
    scope = _make_scope()
    scope["client"] = ("203.0.113.5", 0)
    view = _view(mode="transport_peer")
    assert resolve_access_client(scope, ("127.0.0.1", 54321), view) == "127.0.0.1:54321"


async def test_access_xff_leftmost_skips_trust_chain() -> None:
    scope = _make_scope(FORWARDING_HEADERS)
    view = _view(baseline="127.0.0.1", mode="xff_leftmost")
    assert resolve_access_client(scope, ("127.0.0.1", 54321), view) == "203.0.113.5:0"


async def test_access_xff_untrusted_walks_merged_table() -> None:
    scope = _make_scope(FORWARDING_HEADERS)
    # 10.0.0.10 becomes trusted via extend, so the boundary is the leftmost.
    view = _view(trust=ForwardedTrust.extend("10.0.0.0/8"), mode="xff_untrusted")
    assert resolve_access_client(scope, ("127.0.0.1", 54321), view) == "203.0.113.5:0"

    # Without the extend the reverse walk stops at 10.0.0.10, even though the
    # direct peer itself does not have to pass the entry gate.
    scope2 = _make_scope(FORWARDING_HEADERS, client=("203.0.113.99", 1))
    view2 = _view(mode="xff_untrusted")
    assert resolve_access_client(scope2, ("203.0.113.99", 1), view2) == "10.0.0.10:0"


async def test_access_xff_modes_fall_back_to_transport_without_header() -> None:
    scope = _make_scope([])
    for mode in ("xff_leftmost", "xff_untrusted"):
        view = _view(mode=mode)
        assert resolve_access_client(scope, ("127.0.0.1", 54321), view) == "127.0.0.1:54321"


async def test_access_xff_modes_force_distrust_fall_back_to_transport() -> None:
    scope = _make_scope(FORWARDING_HEADERS)
    view = _view(trust=ForwardedTrust.force_distrust(), mode="xff_leftmost")
    assert resolve_access_client(scope, ("127.0.0.1", 54321), view) == "127.0.0.1:54321"
    view = _view(trust=ForwardedTrust.force_distrust(), mode="xff_untrusted")
    assert resolve_access_client(scope, ("127.0.0.1", 54321), view) == "127.0.0.1:54321"


async def test_access_works_when_proxy_headers_disabled_but_frame_pushed() -> None:
    # The baseline flag being off must not forbid header-based access fields
    # when a frame explicitly asks for them; it only defaults scope rewrite off.
    scope = _make_scope(FORWARDING_HEADERS)
    view = _view(enabled=False, mode="xff_leftmost")
    assert view.scope_rewrite == "none"
    assert resolve_access_client(scope, ("127.0.0.1", 54321), view) == "203.0.113.5:0"


async def test_access_client_addr_reads_config_baseline() -> None:
    config = _ConfigStub(forwarded_allow_ips="127.0.0.1", proxy_headers=False)
    scope = _make_scope()
    # Empty stack + disabled baseline: identical to the old behavior, i.e. the
    # untouched scope client.
    assert access_client_addr(config, scope, ("127.0.0.1", 54321)) == "127.0.0.1:54321"


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


async def test_concurrent_tasks_have_independent_stacks() -> None:
    seen: list[str] = []

    async def worker(mode: str, other_ready: asyncio.Event, i_am_ready: asyncio.Event) -> None:
        with push_proxy_policy(access_client_mode=mode):  # type: ignore[arg-type]
            i_am_ready.set()
            await other_ready.wait()
            # Yield repeatedly while the other task is inside its frame.
            for _ in range(3):
                await asyncio.sleep(0)
            seen.append(effective_proxy_policy("127.0.0.1", True).access_client_mode)

    a_ready = asyncio.Event()
    b_ready = asyncio.Event()
    await asyncio.gather(
        worker("transport_peer", b_ready, a_ready),
        worker("xff_leftmost", a_ready, b_ready),
        # A frame in the surrounding task is inherited as a snapshot but must
        # not leak back into the parent task.
        _outer_push_then_spawn(),
    )
    assert set(seen) == {"transport_peer", "xff_leftmost"}
    # The main task stack is untouched after the children finish.
    assert current_stack() == ()


async def _outer_push_then_spawn() -> None:
    with push_proxy_policy(access_client_mode="xff_untrusted"):
        child = asyncio.create_task(_child_pushes_and_pops())
        await child


async def test_main_task_frame_not_mutated_by_child() -> None:
    with push_proxy_policy(access_client_mode="xff_untrusted"):
        child = asyncio.create_task(_child_pushes_and_pops())
        await child
        assert effective_proxy_policy("127.0.0.1", True).access_client_mode == "xff_untrusted"


async def _child_pushes_and_pops() -> None:
    with push_proxy_policy(access_client_mode="scope_client"):
        assert effective_proxy_policy("127.0.0.1", True).access_client_mode == "scope_client"
        await asyncio.sleep(0)
    assert effective_proxy_policy("127.0.0.1", True).access_client_mode == "xff_untrusted"
