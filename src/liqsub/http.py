from __future__ import annotations

import gzip
import ssl
from urllib.error import URLError
from urllib.request import Request, urlopen


def read_url(
    url: str,
    *,
    accept: str = "*/*",
    timeout: int = 120,
    allow_insecure_ssl_fallback: bool = True,
) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "liqsub/0.1 research",
            "Accept": accept,
            "Accept-Encoding": "gzip, identity",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            encoding = response.headers.get("Content-Encoding", "")
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        is_ssl_error = isinstance(reason, ssl.SSLCertVerificationError)
        if not (allow_insecure_ssl_fallback and is_ssl_error):
            raise
        context = ssl._create_unverified_context()
        with urlopen(request, timeout=timeout, context=context) as response:
            raw = response.read()
            encoding = response.headers.get("Content-Encoding", "")

    if "gzip" in encoding.lower() or raw.startswith(b"\x1f\x8b"):
        return gzip.decompress(raw)
    return raw
