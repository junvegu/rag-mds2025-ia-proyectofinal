"""Helpers when ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` is wrong: public models still load."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def huggingface_public_hub_session() -> Iterator[None]:
    """Unlink bad Hub auth: clear token env vars and avoid netrc/proxy env (``trust_env=False``)."""
    import os

    import httpx
    from huggingface_hub.utils import _http as hf_http

    saved: dict[str, str] = {}
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if key in os.environ:
            saved[key] = os.environ.pop(key)

    def factory():
        return httpx.Client(
            event_hooks={"request": [hf_http.hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
            trust_env=False,
        )

    hf_http.set_client_factory(factory)
    try:
        yield
    finally:
        hf_http.set_client_factory(hf_http.default_client_factory)
        for k, v in saved.items():
            os.environ[k] = v


def huggingface_invalid_env_token_error(exc: BaseException) -> bool:
    """True if failure is likely due to a bad token in the environment (401 on Hub)."""
    seen: set[int] = set()

    def walk(e: BaseException | None) -> bool:
        if e is None or id(e) in seen:
            return False
        seen.add(id(e))
        msg = str(e).lower()
        if "401" in msg or "invalid username or password" in msg:
            return True
        if walk(e.__cause__):
            return True
        if walk(e.__context__):
            return True
        return False

    return walk(exc)
