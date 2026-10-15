"""Job-source adapters."""

from .arbeitnow import ArbeitnowSource
from .base import JobSource
from .jobicy import JobicySource
from .remoteok import RemoteOKSource
from .serpapi import SerpApiSource
from .weworkremotely import WeWorkRemotelySource

__all__ = [
    "ArbeitnowSource",
    "JobSource",
    "JobicySource",
    "RemoteOKSource",
    "SerpApiSource",
    "WeWorkRemotelySource",
]
