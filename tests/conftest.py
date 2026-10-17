"""Tests are offline by default; synthetic MockTransport responses still work."""
import pytest


@pytest.fixture(autouse=True)
def forbid_real_model_http(monkeypatch):
    import httpx
    def forbidden(*args, **kwargs):
        raise AssertionError('Real HTTP is forbidden in the offline Engine suite')
    async def async_forbidden(*args, **kwargs):
        forbidden()
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', async_forbidden)
