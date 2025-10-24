import re
from typing import Any, Dict, List, Set, Tuple

from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}
""".lstrip()

FEWSHOT_TEMPLATE = """
Here are some examples of named entity recognition:

{fewshot}

You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}
""".lstrip()

# Common error patterns to handle in XML predictions
DEFAULT_TAG_FIX_PATTERNS = [
    # Fix mismatched tags
    (r'<(\w+)>(.*?)</\w+>', r'<\1>\2</\1>'),
]


def create_target_text(tokens: List[str], ner_tags: List[str], entity_type_map: Dict[str, str]) -> str:
    """
    Create annotated text from tokens and NER tags.
    Handles BIO tagging scheme conversion to inline XML-style tags.

    Args:
        tokens: List of text tokens
        ner_tags: List of BIO tags corresponding to tokens
        entity_type_map: Mapping from BIO entity types to user-friendly tag names

    Returns:
        String with XML-style entity markup wrapped in <response> tags
    """
    result = []
    current_entity = None
    entity_tokens = []

    for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
        if tag.startswith('B-'):  # Beginning of entity
            # Close previous entity if exists
            if current_entity:
                entity_type = entity_type_map.get(current_entity, '')
                if entity_type:
                    result.append(f'<{entity_type.lower()}>{" ".join(entity_tokens)}</{entity_type.lower()}>')
                else:
                    result.append(' '.join(entity_tokens))
                entity_tokens = []

            current_entity = tag[2:]  # Remove B- prefix
            entity_tokens.append(token)
        elif tag.startswith('I-') and current_entity and tag[2:] == current_entity:  # Inside entity
            entity_tokens.append(token)
        else:  # Outside any entity (O tag)
            if current_entity:  # Close previous entity
                entity_type = entity_type_map.get(current_entity, '')
                if entity_type:
                    result.append(f'<{entity_type.lower()}>{" ".join(entity_tokens)}</{entity_type.lower()}>')
                else:
                    result.append(' '.join(entity_tokens))
                current_entity = None
                entity_tokens = []

            result.append(token)

    # Handle any remaining entity at end of sequence
    if current_entity:
        entity_type = entity_type_map.get(current_entity, '')
        if entity_type:
            result.append(f'<{entity_type.lower()}>{" ".join(entity_tokens)}</{entity_type.lower()}>')
        else:
            result.append(' '.join(entity_tokens))

    # Wrap the entire response in <response> tags as required by the pipeline
    return f'<response>{" ".join(result)}</response>'


def clean_prediction(text: str, tag_fix_patterns: List[Tuple[str, str]] = None) -> str:
    """
    Clean and fix common XML errors in model predictions.

    Args:
        text: The prediction text to clean
        tag_fix_patterns: List of regex patterns and replacements to fix common XML errors

    Returns:
        Cleaned text with fixed XML tags
    """
    if tag_fix_patterns is None:
        tag_fix_patterns = DEFAULT_TAG_FIX_PATTERNS

    cleaned = text

    # Extract content from response tags if present
    response_match = re.search(r'<response>(.*?)</response>', cleaned, re.DOTALL)
    if response_match:
        cleaned = response_match.group(1)

    # Apply fix patterns for common XML errors
    for pattern, replacement in tag_fix_patterns:
        cleaned = re.sub(pattern, replacement, cleaned)

    return cleaned


def extract_entities_from_text(text: str, reverse_entity_map: Dict[str, str]) -> List[Tuple]:
    """
    Extract entities from tagged text with robust error handling.

    Args:
        text: Text with XML entity tags
        reverse_entity_map: Mapping from user-friendly tag names to BIO entity types

    Returns:
        List of (entity_type, entity_text, start_idx, end_idx) tuples
    """
    entities = []

    # Define regex pattern to find XML-style entity tags - handle potential errors
    pattern = r'<(\w+)>(.*?)</\1>'

    try:
        for match in re.finditer(pattern, text):
            entity_type = match.group(1).lower()  # Normalize type to lowercase
            entity_text = match.group(2)
            start_idx = match.start()
            end_idx = match.end()

            # Map back to entity types if possible
            mapped_type = reverse_entity_map.get(entity_type)

            if mapped_type:
                entities.append((mapped_type, entity_text, start_idx, end_idx))
            else:
                # Unknown entity type but still count it for evaluation
                entities.append((entity_type, entity_text, start_idx, end_idx))

    except Exception as e:
        logger.warning(f'Error parsing entities in text: {str(e)}')

    # Handle malformed XML by trying to find additional tag patterns
    # This is a fallback for when the model produces incorrect tags
    unclosed_pattern = r'<(\w+)>(.*?)(?=<|$)'
    try:
        # Find potential unclosed tags
        for match in re.finditer(unclosed_pattern, text):
            # Skip if already part of a well-formed tag
            if any(start_idx <= match.start() < end_idx for _, _, start_idx, end_idx in entities):
                continue

            entity_type = match.group(1).lower()
            entity_text = match.group(2)
            start_idx = match.start()
            end_idx = match.end()

            # Map back to entity types
            mapped_type = reverse_entity_map.get(entity_type)
            if mapped_type:
                entities.append((mapped_type, entity_text, start_idx, end_idx))

    except Exception as e:
        logger.warning(f'Error handling malformed tags: {str(e)}')

    return entities


def xml_to_bio_tags(xml_text: str, original_tokens: List[str], reverse_entity_map: Dict[str, str]) -> List[str]:
    """
    Convert XML-annotated text back to BIO tags aligned with the original tokens.

    Args:
        xml_text: Text with XML entity annotations
        original_tokens: Original tokens to align with
        reverse_entity_map: Mapping from user-friendly tag names to BIO entity types

    Returns:
        List of BIO tags corresponding to the original tokens
    """
    # Extract entities with their character positions
    entities = extract_entities_from_text(xml_text, reverse_entity_map)

    # Initialize all tags as 'O'
    bio_tags = ['O'] * len(original_tokens)

    # Reconstruct the original text to find character positions for each token
    original_text = ' '.join(original_tokens)

    # Track token start positions in the original text
    token_positions = []
    pos = 0
    for token in original_tokens:
        token_pos = original_text.find(token, pos)
        if token_pos == -1:
            # Fallback: just use the current position if we can't find the exact match
            token_positions.append(pos)
        else:
            token_positions.append(token_pos)
            pos = token_pos + len(token)

    # Add token end positions
    token_ends = [pos + len(token) for pos, token in zip(token_positions, original_tokens)]

    # Map entities to tokens based on character positions
    for entity_type, entity_text, start_pos, end_pos in entities:
        # Extract the context from the XML text to help locate the correct entity occurrence
        # Get some context before and after the entity in the XML text
        context_start = max(0, start_pos - 20)
        context_end = min(len(xml_text), end_pos + 20)

        # Extract context without XML tags
        context_before = re.sub(r'<[^>]+>', '', xml_text[context_start:start_pos])
        context_after = re.sub(r'<[^>]+>', '', xml_text[end_pos:context_end])

        # Use context to find the correct entity position in original text
        search_pos = 0
        entity_start = -1

        while search_pos < len(original_text):
            # Find the next occurrence of the entity
            potential_start = original_text.find(entity_text, search_pos)
            if potential_start == -1:
                break

            # Check if the context matches
            potential_context_start = max(0, potential_start - len(context_before))
            potential_context_end = min(len(original_text), potential_start + len(entity_text) + len(context_after))

            before_match = context_before.strip() in original_text[potential_context_start:potential_start].strip()
            after_match = context_after.strip() in original_text[potential_start
                                                                 + len(entity_text):potential_context_end].strip()

            # If context matches or we can't find a better match, use this position
            if before_match or after_match or search_pos > len(original_text) // 2:
                entity_start = potential_start
                break

            # Move search position forward
            search_pos = potential_start + 1

        # If we couldn't find the entity with context, fall back to the first occurrence
        if entity_start == -1:
            entity_start = original_text.find(entity_text)
            if entity_start == -1:
                continue

        entity_end = entity_start + len(entity_text)

        # Find tokens that overlap with this entity
        for i, (token_start, token_end) in enumerate(zip(token_positions, token_ends)):
            if token_start <= entity_end and token_end >= entity_start:
                # This token overlaps with the entity
                if bio_tags[i] == 'O':
                    # Start of entity
                    if i == 0 or bio_tags[i - 1] == 'O' or not bio_tags[i - 1].endswith(entity_type):
                        bio_tags[i] = f'B-{entity_type}'
                    else:
                        # Continuation of entity
                        bio_tags[i] = f'I-{entity_type}'

    return bio_tags


def calculate_bio_metrics(pred_tags: List[str], gold_tags: List[str], tokens: List[str]) -> Tuple[int, int, int]:
    """
    Calculate metrics by comparing BIO tag sequences.

    Args:
        pred_tags: Predicted BIO tags
        gold_tags: Gold standard BIO tags
        tokens: Original tokens

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    # Extract entity spans from BIO tags
    pred_spans = extract_spans_from_bio(pred_tags, tokens)
    gold_spans = extract_spans_from_bio(gold_tags, tokens)

    # Calculate metrics
    true_positives = len(pred_spans.intersection(gold_spans))
    false_positives = len(pred_spans - gold_spans)
    false_negatives = len(gold_spans - pred_spans)

    return true_positives, false_positives, false_negatives


def extract_spans_from_bio(tags: List[str], tokens: List[str]) -> Set[Tuple]:
    """
    Extract entity spans from BIO tags.

    Args:
        tags: List of BIO tags
        tokens: List of tokens corresponding to the tags

    Returns:
        Set of (entity_type, start_idx, end_idx, text) tuples
    """
    spans = set()
    current_entity = None
    start_idx = None
    entity_tokens = []

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith('B-'):  # Beginning of entity
            # Close previous entity if exists
            if current_entity:
                entity_type = current_entity
                entity_text = ' '.join(entity_tokens)
                spans.add((entity_type, start_idx, i - 1, entity_text))
                entity_tokens = []

            current_entity = tag[2:]  # Remove B- prefix
            start_idx = i
            entity_tokens.append(token)
        elif tag.startswith('I-') and current_entity:  # Inside entity
            entity_tokens.append(token)
        elif tag == 'O':  # Outside any entity
            if current_entity:  # Close previous entity
                entity_type = current_entity
                entity_text = ' '.join(entity_tokens)
                spans.add((entity_type, start_idx, i - 1, entity_text))
                current_entity = None
                start_idx = None
                entity_tokens = []

    # Handle any remaining entity at end of sequence
    if current_entity:
        entity_type = current_entity
        entity_text = ' '.join(entity_tokens)
        spans.add((entity_type, start_idx, len(tokens) - 1, entity_text))

    return spans
