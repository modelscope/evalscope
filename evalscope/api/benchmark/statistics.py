# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from evalscope.api.dataset import Sample


@dataclass
class SubsetStatistics:
    """Statistics for a single subset of a benchmark dataset."""

    name: str
    """Name of the subset."""

    sample_count: int
    """Number of samples in this subset."""

    prompt_length_mean: float = 0.0
    """Mean prompt length in characters."""

    prompt_length_min: int = 0
    """Minimum prompt length in characters."""

    prompt_length_max: int = 0
    """Maximum prompt length in characters."""

    prompt_length_std: Optional[float] = None
    """Standard deviation of prompt length."""

    target_length_mean: Optional[float] = None
    """Mean target/answer length in characters (if applicable)."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'sample_count': self.sample_count,
            'prompt_length_mean': round(self.prompt_length_mean, 2) if self.prompt_length_mean else 0,
            'prompt_length_min': self.prompt_length_min,
            'prompt_length_max': self.prompt_length_max,
            'prompt_length_std': round(self.prompt_length_std, 2) if self.prompt_length_std else None,
            'target_length_mean': round(self.target_length_mean, 2) if self.target_length_mean else None,
        }


@dataclass
class DataStatistics:
    """
    Statistics about a benchmark dataset.

    This class holds computed statistics about the dataset including sample counts,
    prompt lengths, and per-subset breakdowns.
    """

    total_samples: int = 0
    """Total number of samples in the evaluation split."""

    subset_stats: List[SubsetStatistics] = field(default_factory=list)
    """Per-subset statistics."""

    prompt_length_mean: float = 0.0
    """Mean prompt length across all samples (in characters)."""

    prompt_length_min: int = 0
    """Minimum prompt length across all samples."""

    prompt_length_max: int = 0
    """Maximum prompt length across all samples."""

    prompt_length_std: Optional[float] = None
    """Standard deviation of prompt length."""

    target_length_mean: Optional[float] = None
    """Mean target/answer length (if applicable)."""

    computed_at: Optional[str] = None
    """ISO timestamp when statistics were computed."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'total_samples': self.total_samples,
            'subset_stats': [s.to_dict() for s in self.subset_stats],
            'prompt_length': {
                'mean': round(self.prompt_length_mean, 2) if self.prompt_length_mean else 0,
                'min': self.prompt_length_min,
                'max': self.prompt_length_max,
                'std': round(self.prompt_length_std, 2) if self.prompt_length_std else None,
            },
            'target_length_mean': round(self.target_length_mean, 2) if self.target_length_mean else None,
            'computed_at': self.computed_at,
        }

    def to_markdown_table(self, include_overall: bool = True) -> str:
        """Generate a markdown table representation of the statistics."""
        lines = []

        if include_overall:
            lines.append('| Metric | Value |')
            lines.append('|--------|-------|')
            lines.append(f'| Total Samples | {self.total_samples:,} |')
            lines.append(f'| Prompt Length (Mean) | {self.prompt_length_mean:.1f} chars |')
            lines.append(f'| Prompt Length (Min/Max) | {self.prompt_length_min} / {self.prompt_length_max} chars |')
            if self.prompt_length_std:
                lines.append(f'| Prompt Length (Std) | {self.prompt_length_std:.1f} |')
            if self.target_length_mean:
                lines.append(f'| Target Length (Mean) | {self.target_length_mean:.1f} chars |')

        if self.subset_stats and len(self.subset_stats) > 1:
            lines.append('')
            lines.append('**Per-Subset Statistics:**')
            lines.append('')
            lines.append('| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |')
            lines.append('|--------|---------|-------------|------------|------------|')
            for s in self.subset_stats:
                lines.append(
                    f'| `{s.name}` | {s.sample_count:,} | {s.prompt_length_mean:.1f} | '
                    f'{s.prompt_length_min} | {s.prompt_length_max} |'
                )

        return '\n'.join(lines)


# Default max lengths for truncation
DEFAULT_MAX_LENGTH = 500
DEFAULT_MAX_ITEM_LENGTH = 200


def truncate_value(value: Any, max_length: int = DEFAULT_MAX_LENGTH) -> Any:
    """
    Recursively truncate values that are too long.

    Args:
        value: The value to truncate.
        max_length: Maximum length for string values.

    Returns:
        Truncated value with same type structure.
    """
    if value is None:
        return None
    elif isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length] + '... [TRUNCATED]'
        return value
    elif isinstance(value, dict):
        return {k: truncate_value(v, max_length) for k, v in value.items()}
    elif isinstance(value, list):
        return [truncate_value(item, max_length) for item in value]
    elif isinstance(value, (int, float, bool)):
        return value
    else:
        # For other types, convert to string and truncate
        str_val = str(value)
        if len(str_val) > max_length:
            return str_val[:max_length] + '... [TRUNCATED]'
        return str_val


@dataclass
class SampleExample:
    """
    A representative sample example for documentation.

    This class stores a truncated example from the benchmark dataset
    for display in documentation and README files. Uses JSON format
    via model_dump() with recursive truncation for long values.
    """

    data: Dict[str, Any]
    """The sample data as a dictionary (from sample.model_dump())."""

    subset: Optional[str] = None
    """Which subset this sample is from."""

    truncated: bool = False
    """Whether any content was truncated."""

    @classmethod
    def from_sample(
        cls,
        sample: Sample,
        subset: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> 'SampleExample':
        """
        Create a SampleExample from a Sample object with auto-truncation.

        Uses sample.model_dump() to convert the sample to a dictionary,
        then recursively truncates any values that exceed max_length.

        Args:
            sample: The Sample object to convert.
            subset: Optional subset name override.
            max_length: Maximum length for string values before truncation.

        Returns:
            SampleExample with truncated content if necessary.
        """
        # Get raw data from sample using model_dump
        raw_data = sample.model_dump(exclude_unset=True, exclude_none=True)

        # Check if truncation is needed
        truncated_data = truncate_value(raw_data, max_length)
        truncated_str = str(truncated_data)
        was_truncated = '[TRUNCATED]' in truncated_str

        return cls(
            data=truncated_data,
            subset=subset or sample.subset_key,
            truncated=was_truncated,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'data': self.data,
            'subset': self.subset,
            'truncated': self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SampleExample':
        """Create from a dictionary (for deserialization)."""
        return cls(
            data=data.get('data', {}),
            subset=data.get('subset'),
            truncated=data.get('truncated', False),
        )

    def to_json_block(self, indent: int = 2) -> str:
        """
        Generate a JSON code block representation of the sample.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON code block string for markdown display.
        """
        import json

        json_str = json.dumps(self.data, ensure_ascii=False, indent=indent)
        lines = []
        if self.subset:
            lines.append(f'**Subset**: `{self.subset}`')
            lines.append('')
        lines.append('```json')
        lines.append(json_str)
        lines.append('```')
        if self.truncated:
            lines.append('')
            lines.append('*Note: Some content was truncated for display.*')
        return '\n'.join(lines)
