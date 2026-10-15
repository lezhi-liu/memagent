"""web_fetch: download a public web page and return it as text.

SSRF protection: a URL from the model could point at our own machine (localhost, the
database port) or a cloud metadata service (169.254.169.254). Before every request,
including every redirect hop, we resolve the host ourselves and refuse unless every
address is publicly routable. Redirects are followed by hand so each hop is checked.

Known gap (documented in the walkthrough): httpx resolves the name again when it
connects, so a DNS server that answers differently the second time (DNS rebinding) could
slip past the check. Closing it needs a transport pinned to the checked IP.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from memagent.llm.types import ToolSpec
from memagent.tools.base import ToolContext, ToolError, ToolOutput, require_str
from memagent.tools.html_text import html_to_text

type IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
type Resolver = Callable[[str, int], Awaitable[list[IPAddress]]]

TIMEOUT_SECONDS = 10.0
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_CHARS = 20_000
MAX_REDIRECTS = 5
_REDIRECT_CODES = {301, 302, 303, 307, 308}


async def system_resolver(host: str, port: int) -> list[IPAddress]:
    """Resolve with the OS resolver, off the event loop (getaddrinfo blocks)."""
    infos = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def is_public_ip(ip: IPAddress) -> bool:
    """True only for globally routable addresses.

    `is_global` already excludes private, loopback, link-local, multicast-ish, reserved
    and unspecified ranges. IPv4-mapped IPv6 (::ffff:127.0.0.1) is unwrapped first,
    otherwise it would be judged as an IPv6 address and could sneak a loopback through.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def check_url(url: str, resolve: Resolver) -> None:
    """Raise ToolError unless the URL is http(s) and its host resolves only to public IPs."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ToolError("only http and https URLs are allowed")
    host = parts.hostname
    if not host:
        raise ToolError("URL has no host")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        # A literal IP needs no DNS; checking it directly also covers "http://[::1]/".
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addresses = await resolve(host, port)
        except OSError as exc:
            raise ToolError(f"could not resolve host {host!r}") from exc
    if not addresses:
        raise ToolError(f"could not resolve host {host!r}")
    # Every address must be public: an attacker-controlled name may return a mix.
    if not all(is_public_ip(ip) for ip in addresses):
        raise ToolError("URL points to a private or local network address")


async def _read_capped(response: httpx.Response, limit: int) -> bytes:
    """Stream the body and stop at `limit`, so a huge page can't exhaust memory."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            raise ToolError(f"page is larger than {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


class WebFetchTool:
    def __init__(
        self,
        resolve: Resolver = system_resolver,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # `transport` and `resolve` are injectable so tests run with no network at all.
        self._resolve = resolve
        self._transport = transport

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_fetch",
            description="Fetch a public web page by URL and return its text content.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "http(s) URL"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolOutput:
        url = require_str(args, "url")
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=TIMEOUT_SECONDS,
            follow_redirects=False,  # we follow by hand so every hop is checked
            headers={"User-Agent": "memagent/0.1 (+web_fetch)"},
        ) as client:
            for _hop in range(MAX_REDIRECTS + 1):
                await check_url(url, self._resolve)
                try:
                    async with client.stream("GET", url) as response:
                        if response.status_code in _REDIRECT_CODES:
                            location = response.headers.get("location")
                            if not location:
                                raise ToolError("redirect without a Location header")
                            url = urljoin(url, location)
                            continue
                        if response.status_code >= 400:
                            raise ToolError(f"HTTP {response.status_code}")
                        body = await _read_capped(response, MAX_DOWNLOAD_BYTES)
                        content_type = response.headers.get("content-type", "")
                        encoding = response.encoding or "utf-8"
                except httpx.HTTPError as exc:
                    raise ToolError(f"request failed: {type(exc).__name__}") from exc
                return ToolOutput(_to_text(body, content_type, encoding))
        raise ToolError(f"more than {MAX_REDIRECTS} redirects")


def _to_text(body: bytes, content_type: str, encoding: str) -> str:
    if (
        not content_type.startswith("text/")
        and "html" not in content_type
        and "json" not in content_type
    ):
        raise ToolError(f"unsupported content type {content_type or 'unknown'!r}")
    text = body.decode(encoding, errors="replace")
    if "html" in content_type:
        text = html_to_text(text)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n[truncated]"
    return text
