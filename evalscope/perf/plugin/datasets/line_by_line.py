import json
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, Iterator, List, Literal, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset
from evalscope.perf.types import AnnotatedBody


class LineByLineArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    body_compose_mode: Literal['override', 'fill', 'passthrough'] = 'override'


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

        example: {"messages": [...], "temperature": 0.6, "max_tokens": 128}

       Treated as a complete request body. The interaction between CLI-level
       generation parameters and body fields is controlled by the
       ``body_compose_mode`` key in ``--dataset-args``:

       - ``override`` (default): CLI params overwrite body fields.
       - ``fill``: Body fields are preserved; CLI params only fill in missing
         fields (``setdefault`` semantics).
       - ``passthrough``: Body is sent as-is; ``__compose_query_from_parameter``
         is skipped entirely, so CLI-level generation parameters are ignored.

       Example::

           --dataset-args '{"body_compose_mode": "fill"}'

       .. note:: ``body_compose_mode`` is currently handled by ``--api openai``
          (the default). Other API plugins treat the body as a plain dict
          regardless of the mode.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        raw = query_parameters.dataset_args or {}
        self._dataset_config = LineByLineArgs(**raw)

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
                mode = self._dataset_config.body_compose_mode
                if mode == 'override' or not isinstance(parsed, dict):
                    result = parsed
                else:
                    result = AnnotatedBody(parsed, compose_mode=mode)
            yield result
