"""Video Processing Module - Composes educational history lesson videos"""

from .composer import VideoComposer, EducationalContent, CompositionResult
from .thumbnail import ThumbnailGenerator
from .effects import VideoEffects
from .educational_effects import (
    EducationalEffects,
    KeyPointDisplay,
    FactCard,
    TopicHeader,
    ImageOverlay
)
from .presentation_slides import PresentationSlideGenerator

__all__ = [
    "VideoComposer",
    "EducationalContent",
    "CompositionResult",
    "ThumbnailGenerator",
    "VideoEffects",
    "EducationalEffects",
    "KeyPointDisplay",
    "FactCard",
    "TopicHeader",
    "ImageOverlay",
    "PresentationSlideGenerator",
]
