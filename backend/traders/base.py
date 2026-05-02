"""Abstract trader interface. Implemented by paper / live engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Fill:
    ts_ms: int
    symbol: str
    side: str
    price: float
    qty: float
    fee_usd: float
    order_id: str
    is_maker: bool


class Trader(ABC):
    """Abstract trader. One instance per mode; switches at mode change."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def on_book_update(self, symbol: str) -> None:
        """Called after a book update is applied to the LOB for `symbol`."""

    @abstractmethod
    async def cancel_all(self, symbol: str) -> None: ...
