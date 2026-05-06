"""AdaptiveAnalyticsSDK -- real-time order-flow analytics.

Public re-exports. See module docstrings for design notes.
"""

from __future__ import annotations

from .exhaustion import (
    EWMAEWVar,
    ExhaustionDetector,
    RollingExtrema,
    RollingRealizedVolatility,
)
from .mab import NIGArm, ThompsonSamplingMAB
from .sdk import AdaptiveAnalyticsSDK, SymbolContext
from .types import (
    ArmId,
    BookSnapshot,
    BucketFillMode,
    BucketSizePolicy,
    BVCSigmaSource,
    ClassificationMode,
    ExhaustionType,
    GlobalConfig,
    MetricsSnapshot,
    Outcome,
    Signal,
    SymbolConfig,
    SymbolState,
    TradeTick,
)
from .vpin import BVCSigmaTracker, TrueVPINEngine

__version__ = "1.1.0"

__all__ = [
    # facade
    "AdaptiveAnalyticsSDK",
    "SymbolContext",
    # components
    "TrueVPINEngine",
    "BVCSigmaTracker",
    "ExhaustionDetector",
    "RollingExtrema",
    "RollingRealizedVolatility",
    "EWMAEWVar",
    "ThompsonSamplingMAB",
    "NIGArm",
    # enums
    "ClassificationMode",
    "BucketFillMode",
    "BucketSizePolicy",
    "BVCSigmaSource",
    "ExhaustionType",
    "ArmId",
    # value objects
    "TradeTick",
    "BookSnapshot",
    "MetricsSnapshot",
    "Signal",
    "Outcome",
    "SymbolState",
    # configs
    "SymbolConfig",
    "GlobalConfig",
]
