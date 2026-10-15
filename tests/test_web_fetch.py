"""web_fetch SSRF protection and limits. No network: a fake DNS resolver and httpx.MockTransport."""

import ipaddress

import httpx
import pytest

from memagent.db.engine import SessionMaker
from memagent.tools.base import ToolError
from memagent.tools.web_fetch import (
    MAX_DOWNLOAD_BYTES,
    IPAddress,
    WebFetchTool,
    check_url,
    is_public_ip,
)
from tests.conftest import tool_context

PUBLIC_IP = "93.184.215.14"

FAKE_DNS = {
    "public.example": [PUBLIC_IP],
    "localhost": ["127.0.0.1"],
    "internal.example": ["10.0.0.5"],
    "mixed.example": [PUBLIC_IP, "192.168.1.10"],  # one bad address is enough to refuse
    "metadata.example": ["169.254.169.254"],
}


async def fake_resolve(host: str, port: int) -> list[IPAddress]:
    if host not in FAKE_DNS:
        raise OSError("name not known")
    return [ipaddress.ip_address(ip) for ip in FAKE_DNS[host]]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost:5432/",
        "http://10.1.2.3/",
        "http://172.16.0.1/",
        "http://192.168.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
        "http://internal.example/",
        "http://mixed.example/",
        "http://metadata.example/",
    ],
)
async def test_private_and_local_targets_are_blocked(url: str) -> None:
    with pytest.raises(ToolError, match="private or local"):
        await check_url(url, fake_resolve)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://public.example/", "gopher://x/"])
async def test_non_http_schemes_are_blocked(url: str) -> None:
    with pytest.raises(ToolError, match="only http"):
        await check_url(url, fake_resolve)


def test_is_public_ip() -> None:
    assert is_public_ip(ipaddress.ip_address(PUBLIC_IP))
    assert not is_public_ip(ipaddress.ip_address("::ffff:10.0.0.1"))


def _tool(handler: httpx.MockTransport) -> WebFetchTool:
    return WebFetchTool(resolve=fake_resolve, transport=handler)


async def test_fetches_public_page_as_text(sessionmaker: SessionMaker) -> None:
    html = (
        "<html><head><title>t</title><script>evil()</script></head>"
        "<body><h1>Hi</h1><p>there</p></body></html>"
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, html=html))
    output = await _tool(transport).run(
        tool_context(sessionmaker), {"url": "http://public.example/"}
    )
    assert output.text == "Hi\n\nthere"


async def test_redirect_to_private_address_is_blocked(sessionmaker: SessionMaker) -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://internal.example/admin"})

    with pytest.raises(ToolError, match="private or local"):
        await _tool(httpx.MockTransport(handler)).run(
            tool_context(sessionmaker), {"url": "http://public.example/"}
        )
    # The private hop was refused before any request was sent to it.
    assert requested == ["http://public.example/"]


async def test_blocked_url_sends_no_request(sessionmaker: SessionMaker) -> None:
    requested: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request)
        return httpx.Response(200, text="should not happen")

    with pytest.raises(ToolError):
        await _tool(httpx.MockTransport(handler)).run(
            tool_context(sessionmaker), {"url": "http://127.0.0.1:5432/"}
        )
    assert requested == []


async def test_download_size_is_capped(sessionmaker: SessionMaker) -> None:
    big = b"a" * (MAX_DOWNLOAD_BYTES + 1)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=big, headers={"content-type": "text/plain"})
    )
    with pytest.raises(ToolError, match="larger than"):
        await _tool(transport).run(tool_context(sessionmaker), {"url": "http://public.example/"})


async def test_binary_content_is_refused(sessionmaker: SessionMaker) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=b"\x89PNG", headers={"content-type": "image/png"}
        )
    )
    with pytest.raises(ToolError, match="unsupported content type"):
        await _tool(transport).run(tool_context(sessionmaker), {"url": "http://public.example/"})
