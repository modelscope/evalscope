import json
import sys
from typing import Any, Dict, Iterator, List, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('line_by_line')
class LineByLineDatasetPlugin(DatasetPluginBase):
    """Read dataset line by line and return prompt.

    Each line in the file can be one of the following formats:

    1. **Plain text** (original format)::

        example: 今天天气怎么样？

       Treated as a raw prompt string. ``__compose_query_from_parameter`` is
       called to merge CLI-level generation parameters (e.g. ``temperature``).

    2. **OpenAI messages** (JSON array)::

        example: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

       Treated as a list of message dicts. ``__compose_query_from_parameter`` is
       called to merge CLI-level generation parameters.

    3. **Complete request body** (JSON object)::
        # note max_tokens will be overridden by CLI-level generation parameters
        example: {"messages": [...], "temperature": 0.6, "max_tokens": 128}

       Treated as a complete request body. The parameters inside the JSON object
       (e.g. ``temperature``) take precedence and ``__compose_query_from_parameter``
       is **NOT** called, so CLI-level generation parameters are ignored.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def _try_parse_json(self, line: str) -> Union[str, List[Dict], Dict[str, Any]]:
        """Try to parse the line as JSON.

        Returns:
            - The original string if JSON parsing fails.
            - List[Dict] if the line is a JSON array (messages format).
            - Dict if the line is a JSON object (complete request body).
        """
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return line  # Not valid JSON, treat as plain text
        return parsed  # List[Dict] or Dict

    def build_messages(self) -> Iterator[Union[str, List[Dict], Dict[str, Any]]]:
        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            line = item.strip()
            if not line:
                continue

            parsed = self._try_parse_json(line)

            if isinstance(parsed, str):
                prompt = parsed
                is_valid, _ = self.check_prompt_length(prompt)
                if not is_valid:
                    continue
                if self.query_parameters.apply_chat_template:
                    message = self.create_message(prompt)
                    result = [message]
                else:
                    result = prompt
            else:
                result = parsed
            yield result
