"""
Inter-member interfaces and integration adapters.
"""

from .alert_dispatcher import AlertDispatcher, AlertEvent
from .frame_provider import StreamFrameProvider

__all__ = [
    "StreamFrameProvider",
    "AlertDispatcher",
    "AlertEvent",
]
