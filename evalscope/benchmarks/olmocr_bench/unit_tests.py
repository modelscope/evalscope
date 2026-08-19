# Unit-test style scoring rules for olmOCR-Bench.
#
# Ported from the official implementation (Apache-2.0):
# https://github.com/allenai/olmocr/blob/main/olmocr/bench/tests.py
#
# The port covers the five rule types present in the released bench data
# (present / absent / order / table / baseline). The official `math` rules are intentionally not
# ported: they depend on KaTeX rendering and rendered-equation image comparison, so the two
# math-only sources (`arxiv_math`, `old_scans_math`) are excluded from this adapter. The unused
# `format` / `footnote` rule types are also omitted; the released data contains none of them.

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .table_parsing import parse_html_tables, parse_markdown_tables

SUPPORTED_TEST_TYPES = frozenset({'present', 'absent', 'order', 'table', 'baseline'})


class ValidationError(Exception):
    """Exception raised for validation errors."""


def normalize_text(md_content: Optional[str]) -> Optional[str]:
    """Normalize markdown content the same way the official bench does before matching."""
    if md_content is None:
        return None

    # Normalize <br> and <br/> to newlines
    md_content = re.sub(r'<br/?>', ' ', md_content)

    # Remove markdown bold formatting (** or __ for bold)
    md_content = re.sub(r'\*\*(.*?)\*\*', r'\1', md_content)
    md_content = re.sub(r'__(.*?)__', r'\1', md_content)
    md_content = re.sub(r'</?b>', '', md_content)  # Remove <b> tags if they exist
    md_content = re.sub(r'</?i>', '', md_content)  # Remove <i> tags if they exist

    # Remove markdown italics formatting (* or _ for italics)
    # Logic: The dot (.) in regex matches any character EXCEPT a newline.
    # This automatically prevents matching **start \n\n end**.
    # We use group \1 to ensure we match matching pairs (**...** or __...__).
    md_content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md_content)  # Bold
    md_content = re.sub(r'(\*|_)(.*?)\1', r'\2', md_content)  # Italics

    # Normalize whitespace in the md_content
    md_content = re.sub(r'\s+', ' ', md_content)

    # Convert down to a consistent unicode form, so é == e + accent, unicode forms
    md_content = unicodedata.normalize('NFC', md_content)

    # Dictionary of characters to replace: keys are fancy characters, values are ASCII
    # equivalents; unicode micro with greek mu comes up often enough too
    replacements = {
        '‘': "'",
        '’': "'",
        '‚': "'",
        '“': '"',
        '”': '"',
        '„': '"',
        '＿': '_',
        '–': '-',
        '—': '-',
        '‑': '-',
        '‒': '-',
        '−': '-',
        'µ': 'μ',
    }

    # Apply all replacements from the dictionary
    for fancy_char, ascii_char in replacements.items():
        md_content = md_content.replace(fancy_char, ascii_char)

    return md_content


class RepeatDetector:
    """Detect trailing repeated n-grams, ported from olmocr.repeatdetect.RepeatDetector (Apache-2.0)."""

    def __init__(self, max_ngram_size: int = 10) -> None:
        self.max_ngram_size = max_ngram_size
        self.data = ''

    def add_letters(self, new_str: str) -> None:
        self.data += new_str

    def ngram_repeats(self) -> List[int]:
        result = [0] * self.max_ngram_size

        if not self.data:
            return result

        # Normalize all whitespace to single spaces
        text = re.sub(r'\s+', ' ', self.data)

        # For each n-gram size
        for size in range(1, self.max_ngram_size + 1):
            if len(text) < size:
                continue

            # Get the last n-gram
            target = text[-size:]

            # Count backwards from the end to find repeats
            count = 0
            pos = len(text) - size  # Start position for previous n-gram

            while pos >= 0:
                if text[pos:pos + size] == target:
                    count += 1
                    pos -= size  # Move back by the size of the n-gram
                else:
                    break

            result[size - 1] = count

        return result


@dataclass(kw_only=True)
class BasePDFTest:
    """Base class for all PDF test types.

    Attributes:
        pdf: The PDF filename.
        page: The page number for the test.
        id: Unique identifier for the test.
        type: The type of test.
        max_diffs: Maximum number of character differences allowed for fuzzy matching.
    """

    pdf: str
    page: int
    id: str
    type: str
    max_diffs: int = 0
    checked: Optional[str] = None
    url: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.pdf:
            raise ValidationError('PDF filename cannot be empty')
        if not self.id:
            raise ValidationError('Test ID cannot be empty')
        if not isinstance(self.max_diffs, int) or self.max_diffs < 0:
            raise ValidationError('Max diffs must be positive number or 0')
        if self.type not in SUPPORTED_TEST_TYPES:
            raise ValidationError(
                f'Invalid test type: {self.type}. Supported types: {sorted(SUPPORTED_TEST_TYPES)}. The official math '
                'rules require KaTeX rendering and are not supported by this adapter.'
            )

    def run(self, md_content: str) -> Tuple[bool, str]:
        """Run the test on the provided markdown content.

        Args:
            md_content: The content of the .md file.

        Returns:
            A tuple (passed, explanation) where 'passed' is True if the test passes,
            and 'explanation' provides details when the test fails.
        """
        raise NotImplementedError('Subclasses must implement the run method')


@dataclass
class TextPresenceTest(BasePDFTest):
    """Test to verify the presence or absence of specific text in a PDF.

    Attributes:
        text: The text string to search for.
    """

    text: str
    case_sensitive: bool = True
    first_n: Optional[int] = None
    last_n: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type not in {'present', 'absent'}:
            raise ValidationError(f'Invalid type for TextPresenceTest: {self.type}')
        self.text = normalize_text(self.text)
        if not self.text.strip():
            raise ValidationError('Text field cannot be empty')

    def run(self, md_content: str) -> Tuple[bool, str]:
        from rapidfuzz import fuzz

        reference_query = self.text
        md_content = normalize_text(md_content)

        if not self.case_sensitive:
            reference_query = reference_query.lower()
            md_content = md_content.lower()

        if self.first_n and self.last_n:
            md_content = md_content[:self.first_n] + md_content[-self.last_n:]
        elif self.first_n:
            md_content = md_content[:self.first_n]
        elif self.last_n:
            md_content = md_content[-self.last_n:]

        # Threshold for fuzzy matching derived from max_diffs
        threshold = 1.0 - (self.max_diffs / (len(reference_query) if len(reference_query) > 0 else 1))
        best_ratio = fuzz.partial_ratio(reference_query, md_content) / 100.0

        if self.type == 'present':
            if best_ratio >= threshold:
                return True, ''
            msg = f"Expected '{reference_query[:40]}...' with threshold {threshold} " \
                  f'but best match ratio was {best_ratio:.3f}'
            return False, msg
        # ABSENT
        if best_ratio < threshold:
            return True, ''
        msg = f"Expected absence of '{reference_query[:40]}...' with threshold {threshold} " \
              f'but best match ratio was {best_ratio:.3f}'
        return False, msg


@dataclass
class TextOrderTest(BasePDFTest):
    """Test to verify that one text appears before another in a PDF.

    Attributes:
        before: The text expected to appear first.
        after: The text expected to appear after the 'before' text.
    """

    before: str
    after: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type != 'order':
            raise ValidationError(f'Invalid type for TextOrderTest: {self.type}')
        self.before = normalize_text(self.before)
        self.after = normalize_text(self.after)
        if not self.before.strip():
            raise ValidationError('Before field cannot be empty')
        if not self.after.strip():
            raise ValidationError('After field cannot be empty')
        if self.max_diffs > len(self.before) // 2 or self.max_diffs > len(self.after) // 2:
            raise ValidationError('Max diffs is too large for this test, greater than 50% of the search string')

    def run(self, md_content: str) -> Tuple[bool, str]:
        from fuzzysearch import find_near_matches

        md_content = normalize_text(md_content)
        before_matches = find_near_matches(self.before, md_content, max_l_dist=self.max_diffs)
        after_matches = find_near_matches(self.after, md_content, max_l_dist=self.max_diffs)

        if not before_matches:
            return False, f"'before' text '{self.before[:40]}...' not found with max_l_dist {self.max_diffs}"
        if not after_matches:
            return False, f"'after' text '{self.after[:40]}...' not found with max_l_dist {self.max_diffs}"

        for before_match in before_matches:
            for after_match in after_matches:
                if before_match.start < after_match.start:
                    return True, ''
        return False, (
            f"Could not find a location where '{self.before[:40]}...' appears before "
            f"'{self.after[:40]}...'."
        )


@dataclass
class TableTest(BasePDFTest):
    """Test that certain properties of a table hold, namely that some cells appear relative to
    other cells correctly."""

    # This is the target cell, which must exist in at least one place in the table
    cell: str

    # These properties say that the cell immediately up/down/left/right of the target cell has
    # the string specified
    up: str = ''
    down: str = ''
    left: str = ''
    right: str = ''

    # These properties say that the cell all the way up, or all the way left of the target cell
    # (ex. headings) has the string value specified
    top_heading: str = ''
    left_heading: str = ''

    ignore_markdown_tables: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type != 'table':
            raise ValidationError(f'Invalid type for TableTest: {self.type}')

        # Normalize the search text too
        self.cell = normalize_text(self.cell)
        self.up = normalize_text(self.up)
        self.down = normalize_text(self.down)
        self.left = normalize_text(self.left)
        self.right = normalize_text(self.right)
        self.top_heading = normalize_text(self.top_heading)
        self.left_heading = normalize_text(self.left_heading)

    def run(self, content: str) -> Tuple[bool, str]:
        from rapidfuzz import fuzz

        tables_to_check = []
        failed_reasons = []

        threshold = 1.0 - (self.max_diffs / (len(self.cell) if len(self.cell) > 0 else 1))
        threshold = max(0.5, threshold)

        if not self.ignore_markdown_tables:
            md_tables = parse_markdown_tables(content)
            tables_to_check.extend(md_tables)

        html_tables = parse_html_tables(content)
        tables_to_check.extend(html_tables)

        # If no tables found, return failure
        if not tables_to_check:
            return False, 'No tables found in the content'

        # Check each table
        for table_data in tables_to_check:
            # Find all cells that match the target cell using fuzzy matching
            matches = []
            for rowcol, cell_content in table_data.cell_text.items():
                similarity = fuzz.ratio(self.cell, normalize_text(cell_content)) / 100.0

                if similarity >= threshold:
                    matches.append(rowcol)

            # If no matches found in this table, continue to the next table
            if not matches:
                continue

            # Check the relationships for each matching cell
            for rowcol in matches:
                all_relationships_satisfied = True
                current_failed_reasons = []

                def _check_relationship(comparison_str: str, relation_func) -> None:
                    nonlocal all_relationships_satisfied
                    cur_relation_satisfied = False
                    best_similarity = 0
                    best_similarity_text = None

                    for rowcol_up in relation_func(rowcol):
                        test_cell = normalize_text(table_data.cell_text[rowcol_up])
                        test_similarity = fuzz.ratio(comparison_str, test_cell) / 100.0
                        if test_similarity > best_similarity:
                            best_similarity = test_similarity
                            best_similarity_text = test_cell

                        if test_similarity >= max(
                            0.5, 1.0 - (self.max_diffs / (len(comparison_str) if len(comparison_str) > 0 else 1))
                        ):
                            cur_relation_satisfied = True

                    if not cur_relation_satisfied:
                        all_relationships_satisfied = False
                        current_failed_reasons.append(
                            f"Cell compared to '{best_similarity_text}' doesn't match expected "
                            f"'{comparison_str}' (best similarity: {best_similarity:.2f})"
                        )

                # Check up relationship
                if self.up:
                    _check_relationship(self.up, lambda rowcol: table_data.up_relations[rowcol])

                if self.down:
                    _check_relationship(self.down, lambda rowcol: table_data.down_relations[rowcol])

                if self.left:
                    _check_relationship(self.left, lambda rowcol: table_data.left_relations[rowcol])

                if self.right:
                    _check_relationship(self.right, lambda rowcol: table_data.right_relations[rowcol])

                if self.left_heading:
                    _check_relationship(self.left_heading, lambda rowcol: table_data.left_heading_relations(*rowcol))

                if self.top_heading:
                    _check_relationship(self.top_heading, lambda rowcol: table_data.top_heading_relations(*rowcol))

                # If all relationships are satisfied for this cell, the test passes
                if all_relationships_satisfied:
                    return True, ''
                else:
                    failed_reasons.extend(current_failed_reasons)

        # If we've gone through all tables and all matching cells and none satisfied all relationships
        if not failed_reasons:
            return False, f"No cell matching '{self.cell}' found in any table with threshold {threshold}"
        return False, f"Found cells matching '{self.cell}' but relationships were not satisfied: " \
                      f"{'; '.join(failed_reasons)}"


@dataclass
class BaselineTest(BasePDFTest):
    """Make sure that several baseline quality checks pass for the output generation.

    Namely, the output is not blank, not endlessly repeating, and contains characters of the proper
    character sets.
    """

    max_length: Optional[int] = None  # Used to implement blank page checks
    max_length_skips_image_alt_tags: bool = False

    max_repeats: int = 30
    check_disallowed_characters: bool = True

    def run(self, content: str) -> Tuple[bool, str]:
        base_content_len = len(''.join(c for c in content if c.isalnum()).strip())

        # If this is a blank page check, then it short circuits the rest of the checks
        if self.max_length is not None:
            if self.max_length_skips_image_alt_tags:
                # Remove markdown image tags like ![alt text](image.png) from the text length count
                content_for_length_check = re.sub(r'!\[.*?\]\(.*?\)', '', content)
                base_content_len = len(''.join(c for c in content_for_length_check if c.isalnum()).strip())

            if base_content_len > self.max_length:
                return False, f'{base_content_len} characters were output for a page we expected to be blank'
            return True, ''

        if base_content_len == 0:
            return False, 'The text contains no alpha numeric characters'

        # Make sure that the content has no egregious repeated ngrams at the end, which indicate a
        # degradation of quality
        detector = RepeatDetector(max_ngram_size=5)
        detector.add_letters(content)
        repeats = detector.ngram_repeats()

        for index, count in enumerate(repeats):
            if count > self.max_repeats:
                return False, f'Text ends with {count} repeating {index + 1}-grams, invalid'

        pattern = re.compile(
            r'['
            r'\u4e00-\u9FFF'  # CJK Unified Ideographs (Chinese characters)
            r'\u3040-\u309F'  # Hiragana (Japanese)
            r'\u30A0-\u30FF'  # Katakana (Japanese)
            r'\U0001F600-\U0001F64F'  # Emoticons (Emoji)
            r'\U0001F300-\U0001F5FF'  # Miscellaneous Symbols and Pictographs (Emoji)
            r'\U0001F680-\U0001F6FF'  # Transport and Map Symbols (Emoji)
            r'\U0001F1E0-\U0001F1FF'  # Regional Indicator Symbols (flags, Emoji)
            r']',
            flags=re.UNICODE,
        )

        matches = pattern.findall(content)
        if self.check_disallowed_characters and matches:
            return False, f'Text contains disallowed characters {matches}'

        return True, ''


def load_single_test(data: Union[str, Dict[str, Any]]) -> BasePDFTest:
    """Load a single test from a JSON line string or JSON object.

    Args:
        data: Either a JSON string to parse or a dictionary containing test data.

    Returns:
        A test object of the appropriate type.

    Raises:
        ValidationError: If the test type is unknown or unsupported, or the data is invalid.
    """
    # Handle JSON string input
    if isinstance(data, str):
        data = data.strip()
        if not data:
            raise ValueError('Empty string provided')
        data = json.loads(data)

    # Process the test data
    test_type = data.get('type')
    if test_type in {'present', 'absent'}:
        test = TextPresenceTest(**data)
    elif test_type == 'order':
        test = TextOrderTest(**data)
    elif test_type == 'table':
        test = TableTest(**data)
    elif test_type == 'baseline':
        test = BaselineTest(**data)
    else:
        raise ValidationError(
            f'Unknown or unsupported test type: {test_type}. Supported types: {sorted(SUPPORTED_TEST_TYPES)}. '
            'The official math rules require KaTeX rendering and are not supported by this adapter.'
        )

    return test
