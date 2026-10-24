"""End-to-end acceptance tests for the proxy visibility policy stack.

Every *output* required by the feature spec is exercised here:

* h11 / httptools / zttp access logs honor ``access_client_mode``, including
  ``scope_rewrite`` unrelated to access (none + xff/transport modes);
* HTTP/2 streams pushing different frames never cross stacks;
* wsproto / websockets-sansio (and legacy ``websockets``) honor the modes on
  accept, 403 rejection and ``websocket.http.response.start`` rejection;
* visibility after the HTTP -> WebSocket upgrade follows the top stack frame;
* ``limit_concurrency`` 503 access logging follows the stack without changing
  the 503 condition;
* proxy_headers=False with an empty stack performs no scope rewrite, while a
  frame pushed *before* middleware entry is honored.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import ssl
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import httpx2
import pytest
from websockets.asyncio.client import connect

from tests.utils import run_server
from uvicorn._types import ASGIReceiveCallable, ASGISendCallable, Scope
from uvicorn.config import Config
from uvicorn.importer import import_from_string
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from uvicorn.proxy_policy import ForwardedTrust, push_proxy_policy

if TYPE_CHECKING:
    from typing import TypeAlias

    from uvicorn.protocols.websockets.websockets_impl import WebSocketProtocol as LegacyWSProtocol
    from uvicorn.protocols.websockets.websockets_sansio_impl import WebSocketsSansIOProtocol
    from uvicorn.protocols.websockets.wsproto_impl import WSProtocol as _WSProtocol

    WSProtocol: TypeAlias = "type[WebSocketsSansIOProtocol | _WSProtocol | LegacyWSProtocol]"

pytestmark = pytest.mark.anyio

XFF_VALUE = "203.0.113.5, 10.0.0.10"
XFF_SINGLE = "203.0.113.7"


@pytest.fixture(
    params=[
        pytest.param(
            "uvicorn.protocols.websockets.wsproto_impl:WSProtocol",
            id="wsproto",
        ),
        pytest.param(
            "uvicorn.protocols.websockets.websockets_sansio_impl:WebSocketsSansIOProtocol",
            id="websockets-sansio",
        ),
        pytest.param(
            "uvicorn.protocols.websockets.websockets_impl:WebSocketProtocol",
            id="websockets-legacy",
            marks=pytest.mark.filterwarnings("ignore"),
        ),
    ]
)
def all_ws_protocol_cls(request: pytest.FixtureRequest) -> WSProtocol:
    return import_from_string(request.param)


# ---------------------------------------------------------------------------
# logging helper (uvicorn.access / uvicorn.error do not propagate by default)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def caplog_for_logger(caplog: pytest.LogCaptureFixture, logger_name: str) -> Iterator[pytest.LogCaptureFixture]:
    logger = logging.getLogger(logger_name)
    logger.propagate, old_propagate = False, logger.propagate
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.propagate = old_propagate


def _messages(caplog: pytest.LogCaptureFixture, logger_name: str) -> list[str]:
    return [record.message for record in caplog.records if record.name == logger_name]


def _client_token(message: str) -> str:
    return message.split(" - ", 1)[0]


def _headers(scope: Scope) -> dict[bytes, bytes]:
    return {name: value for name, value in scope.get("headers", [])}


# ---------------------------------------------------------------------------
# apps
# ---------------------------------------------------------------------------


async def echo_app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
    host, port = scope["client"] or ("", 0)
    body = f"{scope['scheme']}://{host}:{port}".encode()
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def policy_http_app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
    headers = _headers(scope)
    mode = headers.get(b"x-policy-mode")
    extend = headers.get(b"x-policy-extend")
    rewrite = headers.get(b"x-policy-rewrite")
    trust = ForwardedTrust.extend(extend.decode()) if extend else None
    with push_proxy_policy(
        access_client_mode=mode.decode() if mode else None,
        forwarded_trust=trust,
        scope_rewrite=rewrite.decode() if rewrite else None,
    ):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


_barrier: asyncio.Event | None = None


async def barrier_http_app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
    headers = _headers(scope)
    mode = headers[b"x-policy-mode"].decode()
    with push_proxy_policy(access_client_mode=mode):  # type: ignore[arg-type]
        assert _barrier is not None
        await _barrier.wait()
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


async def ws_policy_app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
    headers = _headers(scope)
    mode = headers.get(b"x-policy-mode")
    action = headers.get(b"x-ws-action", b"accept").decode()
    while True:
        message = await receive()
        if message["type"] == "websocket.connect":
            with push_proxy_policy(access_client_mode=mode.decode() if mode else None):
                if action == "accept":
                    await send({"type": "websocket.accept"})
                elif action == "close":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                elif action == "http_response":
                    await send(
                        {
                            "type": "websocket.http.response.start",
                            "status": 401,
                            "headers": [(b"content-type", b"text/plain"), (b"content-length", b"4")],
                        }
                    )
                    await send({"type": "websocket.http.response.body", "body": b"nope"})
                    return
        elif message["type"] == "websocket.disconnect":
            return


# ---------------------------------------------------------------------------
# HTTP/1.1 access modes across h11 / httptools / zttp
# ---------------------------------------------------------------------------


async def test_http_access_default_scope_client_with_empty_stack(
    http_protocol_cls: type,
    caplog: pytest.LogCaptureFixture,
    logging_config: dict[str, Any],
    unused_tcp_port: int,
) -> None:
    config = Config(
        app=policy_http_app,
        http=http_protocol_cls,
        proxy_headers=True,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    with caplog_for_logger(caplog, "uvicorn.access"):
        async with run_server(config):
            async with httpx2.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{unused_tcp_port}/",
                    headers={"x-forwarded-for": XFF_VALUE, "x-forwarded-proto": "https"},
                )
        assert response.status_code == 204
        messages = _messages(caplog, "uvicorn.access")
        assert _client_token(messages[-1]) == "10.0.0.10:0"


@pytest.mark.parametrize(
    ("mode", "extend", "rewrite", "expected_client"),
    [
        ("transport_peer", None, None, "transport"),
        ("xff_leftmost", None, None, "203.0.113.5:0"),
        ("xff_untrusted", None, None, "10.0.0.10:0"),
        # scope was rewritten at middleware entry, but the merged table of the
        # inner frame lets access log walk further: access decoupled from scope.
        ("xff_untrusted", "10.0.0.0/8", None, "203.0.113.5:0"),
        # scope_rewrite=none and a header-based access mode must coexist: even
        # though such a frame (pushed inside the app) cannot undo the entry
        # rewrite, access still follows transport rather than scope.
        ("transport_peer", None, "none", "transport"),
    ],
)
async def test_http_access_client_modes(
    http_protocol_cls: type,
    caplog: pytest.LogCaptureFixture,
    logging_config: dict[str, Any],
    unused_tcp_port: int,
    mode: str,
    extend: str | None,
    rewrite: str | None,
    expected_client: str,
) -> None:
    config = Config(
        app=policy_http_app,
        http=http_protocol_cls,
        proxy_headers=True,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    headers = {"x-forwarded-for": XFF_VALUE, "x-policy-mode": mode}
    if extend:
        headers["x-policy-extend"] = extend
    if rewrite:
        headers["x-policy-rewrite"] = rewrite
    with caplog_for_logger(caplog, "uvicorn.access"):
        async with run_server(config):
            async with httpx2.AsyncClient() as client:
                response = await client.get(f"http://127.0.0.1:{unused_tcp_port}/", headers=headers)
        assert response.status_code == 204
        messages = _messages(caplog, "uvicorn.access")
        token = _client_token(messages[-1])
        if expected_client == "transport":
            host, _, port = token.partition(":")
            assert host == "127.0.0.1"
            assert int(port) > 0
        else:
            assert token == expected_client


async def test_http_concurrent_connections_do_not_cross_stacks(
    http_protocol_cls: type,
    caplog: pytest.LogCaptureFixture,
    logging_config: dict[str, Any],
    unused_tcp_port: int,
) -> None:
    global _barrier
    _barrier = asyncio.Event()
    config = Config(
        app=barrier_http_app,
        http=http_protocol_cls,
        proxy_headers=True,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    with caplog_for_logger(caplog, "uvicorn.access"):
        async with run_server(config):
            async with httpx2.AsyncClient() as client:
                t1 = asyncio.create_task(
                    client.get(
                        f"http://127.0.0.1:{unused_tcp_port}/a",
                        headers={"x-forwarded-for": XFF_VALUE, "x-policy-mode": "transport_peer"},
                    )
                )
                t2 = asyncio.create_task(
                    client.get(
                        f"http://127.0.0.1:{unused_tcp_port}/b",
                        headers={"x-forwarded-for": XFF_VALUE, "x-policy-mode": "xff_leftmost"},
                    )
                )
                await asyncio.sleep(0.3)
                assert _barrier is not None
                _barrier.set()
                r1, r2 = await asyncio.gather(t1, t2)
        assert (r1.status_code, r2.status_code) == (204, 204)

        by_path = {}
        for message in _messages(caplog, "uvicorn.access"):
            if '"GET /a ' in message:
                by_path["a"] = _client_token(message)
            elif '"GET /b ' in message:
                by_path["b"] = _client_token(message)
        assert set(by_path) == {"a", "b"}
        assert by_path["a"].startswith("127.0.0.1:")
        assert by_path["b"] == "203.0.113.5:0"
    _barrier = None


# ---------------------------------------------------------------------------
# HTTP/2 streams
# ---------------------------------------------------------------------------


async def test_http2_concurrent_streams_do_not_cross_stacks(
    caplog: pytest.LogCaptureFixture,
    logging_config: dict[str, Any],
    unused_tcp_port: int,
    tls_certificate_server_cert_path: str,
    tls_certificate_private_key_path: str,
    tls_ca_ssl_context: ssl.SSLContext,
) -> None:
    pytest.importorskip("zttp")
    pytest.importorskip("h2")
    global _barrier
    _barrier = asyncio.Event()
    config = Config(
        app=barrier_http_app,
        http="zttp",
        http2=True,
        proxy_headers=True,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
        ssl_certfile=tls_certificate_server_cert_path,
        ssl_keyfile=tls_certificate_private_key_path,
    )
    with caplog_for_logger(caplog, "uvicorn.access"):
        async with run_server(config):
            async with httpx2.AsyncClient(http2=True, verify=tls_ca_ssl_context) as client:
                t1 = asyncio.create_task(
                    client.get(
                        f"https://127.0.0.1:{unused_tcp_port}/a",
                        headers={"x-forwarded-for": XFF_VALUE, "x-policy-mode": "xff_untrusted"},
                    )
                )
                t2 = asyncio.create_task(
                    client.get(
                        f"https://127.0.0.1:{unused_tcp_port}/b",
                        headers={"x-forwarded-for": XFF_VALUE, "x-policy-mode": "xff_leftmost"},
                    )
                )
                await asyncio.sleep(0.3)
                assert _barrier is not None
                _barrier.set()
                r1, r2 = await asyncio.gather(t1, t2)
        assert (r1.status_code, r2.status_code) == (204, 204)

        by_path = {}
        for message in _messages(caplog, "uvicorn.access"):
            if '"GET /a ' in message:
                by_path["a"] = _client_token(message)
            elif '"GET /b ' in message:
                by_path["b"] = _client_token(message)
        assert set(by_path) == {"a", "b"}
        assert by_path["a"] == "10.0.0.10:0"
        assert by_path["b"] == "203.0.113.5:0"
    _barrier = None


# ---------------------------------------------------------------------------
# proxy_headers=False baseline + entry-time frames
# ---------------------------------------------------------------------------


async def test_proxy_headers_false_empty_stack_does_not_rewrite(
    http_protocol_cls: type,
    unused_tcp_port: int,
    logging_config: dict[str, Any],
) -> None:
    config = Config(
        app=echo_app,
        http=http_protocol_cls,
        proxy_headers=False,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    async with run_server(config):
        async with httpx2.AsyncClient() as client:
            response = await client.get(
                f"http://127.0.0.1:{unused_tcp_port}/",
                headers={"x-forwarded-for": XFF_VALUE, "x-forwarded-proto": "https"},
            )
    assert response.status_code == 200
    assert response.text.startswith("http://127.0.0.1:")


async def test_frame_pushed_before_middleware_entry_controls_rewrite(
    http_protocol_cls: type,
    unused_tcp_port: int,
    logging_config: dict[str, Any],
) -> None:
    # proxy_headers=False so Config does not add its own middleware; the user
    # wires ProxyHeadersMiddleware explicitly and an outer frame decides trust.
    proxied = ProxyHeadersMiddleware(echo_app, trusted_hosts="127.0.0.1")

    async def framed_app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        if scope["type"] == "http" and _headers(scope).get(b"x-policy-distrust"):
            with push_proxy_policy(forwarded_trust=ForwardedTrust.force_distrust()):
                await proxied(scope, receive, send)
        else:
            await proxied(scope, receive, send)

    config = Config(
        app=framed_app,
        http=http_protocol_cls,
        proxy_headers=False,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    forwarded = {"x-forwarded-for": XFF_VALUE, "x-forwarded-proto": "https"}
    async with run_server(config):
        async with httpx2.AsyncClient() as client:
            rewritten = await client.get(f"http://127.0.0.1:{unused_tcp_port}/", headers=forwarded)
            distrusted = await client.get(
                f"http://127.0.0.1:{unused_tcp_port}/",
                headers={**forwarded, "x-policy-distrust": "1"},
            )
    assert rewritten.text == "https://10.0.0.10:0"
    assert distrusted.text.startswith("http://127.0.0.1:")


async def test_frame_pushed_before_middleware_entry_rewrite_none_keeps_access_decoupled(
    http_protocol_cls: type,
    caplog: pytest.LogCaptureFixture,
    logging_config: dict[str, Any],
    unused_tcp_port: int,
) -> None:
    # The peer is trusted and valid X-Forwarded-* headers arrive, but an
    # outer frame with scope_rewrite="none" forbids scope mutation.  The
    # access record is still allowed to follow the XFF boundary.  This must
    # not secretly enable scope rewriting even though the frame is active.
    proxied = ProxyHeadersMiddleware(echo_app, trusted_hosts="127.0.0.1")

    async def framed_app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        with push_proxy_policy(scope_rewrite="none", access_client_mode="xff_untrusted"):
            await proxied(scope, receive, send)

    config = Config(
        app=framed_app,
        http=http_protocol_cls,
        proxy_headers=False,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    with caplog_for_logger(caplog, "uvicorn.access"):
        async with run_server(config):
            async with httpx2.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{unused_tcp_port}/",
                    headers={"x-forwarded-for": XFF_VALUE, "x-forwarded-proto": "https"},
                )
        assert response.status_code == 200
        # Scope untouched: scheme stays http and client stays the transport peer.
        assert response.text.startswith("http://127.0.0.1:")
        # Access log still walks the XFF chain against the merged trust table.
        messages = _messages(caplog, "uvicorn.access")
        assert _client_token(messages[-1]) == "10.0.0.10:0"


# ---------------------------------------------------------------------------
# limit_concurrency 503
# ---------------------------------------------------------------------------


async def test_limit_concurrency_503_access_uses_stack_default(
    http_protocol_cls: type,
    caplog: pytest.LogCaptureFixture,
    logging_config: dict[str, Any],
    unused_tcp_port: int,
) -> None:
    global _barrier
    _barrier = asyncio.Event()

    async def gated_app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        assert _barrier is not None
        await _barrier.wait()
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    config = Config(
        app=gated_app,
        http=http_protocol_cls,
        limit_concurrency=2,
        proxy_headers=True,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    with caplog_for_logger(caplog, "uvicorn.access"):
        async with run_server(config):
            async with httpx2.AsyncClient() as client:
                held = asyncio.create_task(client.get(f"http://127.0.0.1:{unused_tcp_port}/hold"))
                await asyncio.sleep(0.3)
                rejected = await client.get(
                    f"http://127.0.0.1:{unused_tcp_port}/",
                    headers={"x-forwarded-for": XFF_VALUE, "x-forwarded-proto": "https"},
                )
                assert rejected.status_code == 503
                assert _barrier is not None
                _barrier.set()
                assert (await held).status_code == 204

        messages = _messages(caplog, "uvicorn.access")
        reject_log = next(message for message in messages if '" 503' in message)
        # No frame is active on the 503 path and service_unavailable bypasses
        # the app chain: the client must be the transport peer, not the XFF.
        token = _client_token(reject_log)
        assert token.startswith("127.0.0.1:")
        assert token != "203.0.113.5:0"
    _barrier = None


# ---------------------------------------------------------------------------
# WebSockets: accept / 403 / http.response.start
# ---------------------------------------------------------------------------


def _ws_headers(mode: str | None, action: str = "accept", xff: str = XFF_SINGLE) -> dict[str, str]:
    headers = {"x-forwarded-for": xff, "x-forwarded-proto": "https", "x-ws-action": action}
    if mode is not None:
        headers["x-policy-mode"] = mode
    return headers


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, "203.0.113.7:0"),  # empty stack -> scope rewritten by middleware
        ("transport_peer", "transport"),
        ("xff_leftmost", "203.0.113.7:0"),
        ("xff_untrusted", "203.0.113.7:0"),
    ],
)
async def test_ws_accept_access_modes(
    all_ws_protocol_cls: WSProtocol,
    caplog: pytest.LogCaptureFixture,
    logging_config: dict[str, Any],
    unused_tcp_port: int,
    mode: str | None,
    expected: str,
) -> None:
    config = Config(
        app=ws_policy_app,
        ws=all_ws_protocol_cls,
        proxy_headers=True,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    with caplog_for_logger(caplog, "uvicorn.error"):
        async with run_server(config):
            async with connect(
                f"ws://127.0.0.1:{unused_tcp_port}/ws",
                additional_headers=_ws_headers(mode),
            ) as websocket:
                assert websocket.response is not None
        messages = [m for m in _messages(caplog, "uvicorn.error") if "[accepted]" in m]
        token = _client_token(messages[-1])
        if expected == "transport":
            assert token.startswith("127.0.0.1:")
        else:
            assert token == expected


async def test_ws_403_access_mode(
    all_ws_protocol_cls: WSProtocol,
    caplog: pytest.LogCaptureFixture,
    logging_config: dict[str, Any],
    unused_tcp_port: int,
) -> None:
    config = Config(
        app=ws_policy_app,
        ws=all_ws_protocol_cls,
        proxy_headers=True,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    with caplog_for_logger(caplog, "uvicorn.error"):
        async with run_server(config):
            with contextlib.suppress(Exception):
                async with connect(
                    f"ws://127.0.0.1:{unused_tcp_port}/ws",
                    additional_headers=_ws_headers("xff_leftmost", action="close"),
                ):
                    pass
        messages = [m for m in _messages(caplog, "uvicorn.error") if " 403" in m]
        assert messages, "expected a 403 access record"
        assert _client_token(messages[-1]) == "203.0.113.7:0"


async def test_ws_http_response_start_access_mode(
    all_ws_protocol_cls: WSProtocol,
    caplog: pytest.LogCaptureFixture,
    logging_config: dict[str, Any],
    unused_tcp_port: int,
) -> None:
    config = Config(
        app=ws_policy_app,
        ws=all_ws_protocol_cls,
        proxy_headers=True,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    with caplog_for_logger(caplog, "uvicorn.error"):
        async with run_server(config):
            with contextlib.suppress(Exception):
                async with connect(
                    f"ws://127.0.0.1:{unused_tcp_port}/ws",
                    additional_headers=_ws_headers("transport_peer", action="http_response"),
                ):
                    pass
        messages = [m for m in _messages(caplog, "uvicorn.error") if '" 401' in m]
        assert messages, "expected an HTTP 401 rejection access record"
        token = _client_token(messages[-1])
        assert token.startswith("127.0.0.1:")


async def test_ws_upgrade_visibility_follows_stack_top(
    all_ws_protocol_cls: WSProtocol,
    caplog: pytest.LogCaptureFixture,
    logging_config: dict[str, Any],
    unused_tcp_port: int,
) -> None:
    # The frame is pushed by the upgraded connection right before accept, so
    # after the upgrade visibility follows the current stack top.
    config = Config(
        app=ws_policy_app,
        ws=all_ws_protocol_cls,
        proxy_headers=True,
        log_config=logging_config,
        lifespan="off",
        port=unused_tcp_port,
    )
    with caplog_for_logger(caplog, "uvicorn.error"):
        async with run_server(config):
            async with connect(
                f"ws://127.0.0.1:{unused_tcp_port}/ws",
                additional_headers=_ws_headers("xff_untrusted"),
            ):
                pass
        messages = [m for m in _messages(caplog, "uvicorn.error") if "[accepted]" in m]
        assert _client_token(messages[-1]) == "203.0.113.7:0"
