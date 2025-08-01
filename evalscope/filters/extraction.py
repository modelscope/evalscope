import re
from typing import List

from evalscope.api.filter import Filter
from evalscope.api.registry import register_filter


@register_filter('regex')
class RegexFilter(Filter):
    """A filter that extracts values from text using regex pattern matching.

    This filter applies a regex pattern to each model response and extracts matched values.
    If no match is found, returns a fallback value. Useful for extracting structured data
    (like numbers) from unstructured model outputs.
    """

    def __init__(
        self,
        regex_pattern: str = r'#### (\-?[0-9\.\,]+)',
        group_select: int = 0,
        fallback: str = '[invalid]',
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, instance: List[str]) -> List[str]:
        """Apply regex pattern to each string in the instance list."""
        filtered = []
        for resp in instance:
            match = self.regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m]
                    if match:
                        match = match[0]
                    else:
                        match = self.fallback
                match = match.strip()
            else:
                match = self.fallback
            filtered.append(match)
        return filtered


@register_filter('regex_pos')
class POSFilter(Filter):
    """ """

    def __init__(
        self,
        regex_pattern: str = r"\['(.*?)'\]",
        group_select=0,
        fallback=None,
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        if fallback is None:
            fallback = ['invalid']
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, instance: List[str]) -> List[str]:
        """Extract POS tags from each string in the instance list."""

        def extract_tagged_tokens(text):
            # Extract tagged tokens list from text input using regex
            tokens = re.findall(r"\('([^']*)', '([^']*)'\)", text)
            return [(token, pos) for token, pos in tokens]

        def extract_pos_tags(result):
            pos_tags = []
            if isinstance(result, str):
                result = extract_tagged_tokens(result)
            pos_tags.extend(pos for _, pos in result)
            return pos_tags if pos_tags else self.fallback

        filtered = []
        for resp in instance:
            match = extract_pos_tags(resp)
            filtered.append(str(match))
        return filtered


@register_filter('remove_whitespace')
class WhitespaceFilter(Filter):
    """Filters out leading whitespace from responses."""

    def apply(self, instance: List[str]) -> List[str]:
        """Remove leading whitespace from each string in the instance list."""
        filtered_resp = []
        for resp in instance:
            resp = resp.lstrip()
            filtered_resp.append(resp)
        return filtered_resp


@register_filter('remove_until')
class RemoveUntilFilter(Filter):
    """Filters out all text until a specified delimiter is found."""

    def __init__(self, delimiter: str) -> None:
        self.delimiter = delimiter

    def apply(self, instance: List[str]) -> List[str]:
        """Remove all text until the delimiter from each string in the instance list."""
        filtered_resp = []
        for resp in instance:
            resp = resp.split(self.delimiter, 1)[-1]
            filtered_resp.append(resp)
        return filtered_resp


@register_filter('extract')
class ExtractFilter(RegexFilter):
    ...
