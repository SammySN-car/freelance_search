from collections.abc import Awaitable, Callable
from typing import Protocol

import httpx

from ..models import Listing


class JobSource(Protocol):
    name: str

    async def fetch(self, client: httpx.AsyncClient) -> list[Listing]:
        """Fetch and normalize listings from this source."""


def source_id(source: str, value: str) -> str:
    return f"{source}:{value}"


def request_json(
    client: httpx.AsyncClient, url: str, **params: str
) -> Awaitable[httpx.Response]:
    return client.get(url, params=params)


def require_success(response: httpx.Response) -> None:
    response.raise_for_status()


Fetcher = Callable[[httpx.AsyncClient], Awaitable[list[Listing]]]
