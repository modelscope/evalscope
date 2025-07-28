from pydantic import BaseModel, Field, JsonValue
from typing import Dict, Literal, Optional, Sequence, Union


class ContentBase(BaseModel):
    internal: Optional[JsonValue] = Field(default=None)
    """Model provider specific payload - typically used to aid transformation back to model types."""


class ContentText(ContentBase):
    """Text content."""

    type: Literal['text'] = Field(default='text')
    """Type."""

    text: str
    """Text content."""

    refusal: Optional[bool] = Field(default=None)
    """Was this a refusal message?"""


class ContentReasoning(ContentBase):
    """Reasoning content.

    See the specification for [thinking blocks](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#understanding-thinking-blocks) for Claude models.
    """  # noqa: E501

    type: Literal['reasoning'] = Field(default='reasoning')
    """Type."""

    reasoning: str
    """Reasoning content."""

    signature: Optional[str] = Field(default=None)
    """Signature for reasoning content (used by some models to ensure that reasoning content is not modified for replay)"""  # noqa: E501

    redacted: bool = Field(default=False)
    """Indicates that the explicit content of this reasoning block has been redacted."""


class ContentImage(ContentBase):
    """Image content."""

    type: Literal['image'] = Field(default='image')
    """Type."""

    image: str
    """Either a URL of the image or the base64 encoded image data."""

    detail: Literal['auto', 'low', 'high'] = Field(default='auto')
    """Specifies the detail level of the image.

    Currently only supported for OpenAI. Learn more in the    [Vision guide](https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding).
    """  # noqa: E501


class ContentAudio(ContentBase):
    """Audio content."""

    type: Literal['audio'] = Field(default='audio')
    """Type."""

    audio: str
    """Audio file path or base64 encoded data URL."""

    format: Literal['wav', 'mp3']
    """Format of audio data ('mp3' or 'wav')"""


class ContentVideo(ContentBase):
    """Video content."""

    type: Literal['video'] = Field(default='video')
    """Type."""

    video: str
    """Audio file path or base64 encoded data URL."""

    format: Literal['mp4', 'mpeg', 'mov']
    """Format of video data ('mp4', 'mpeg', or 'mov')"""


class ContentData(ContentBase):
    """Model internal."""

    type: Literal['data'] = Field(default='data')
    """Type."""

    data: Dict[str, JsonValue]
    """Model provider specific payload - required for internal content."""


Content = Union[
    ContentText,
    ContentReasoning,
    ContentImage,
    ContentAudio,
    ContentVideo,
    ContentData,
]
"""Content sent to or received from a model."""
