
from sys import maxsize
import sys
from typing import Any, Dict, Iterator
import json
from llmuses.perf.api_plugin_base import ApiPluginBase

from llmuses.perf.plugin_registry import register_api

@register_api("dashscope")
class DashScopeApiPlugin(ApiPluginBase):
    def __init__(self, mode_path: str):
        """Init the plugin

        Args:
            mode_path (str): The model path, we use the tokenizer 
                weight in the model to calculate the number of the
                input and output tokens.
        """
        super().__init__(model_path=mode_path)
    def build_request(self,
                      model: str,
                      prompt: str,
                      query_template: str) -> Dict:
        try:
            query = json.loads(query_template)
            ApiPluginBase.replace_values(query, model, prompt)
            return query
        except:
            print('Prompt: %s is invalidate!'%prompt)
            return None


    def parse_responses(self, responses, **kwargs) -> Dict:
        """Parser responses and return number of request and response tokens.

        Args:
            responses (List[bytes]): List of http response body, for stream output,
                there are multiple responses, for general only one. 
            kwargs: (Any): The command line --parameter content.

        Returns:
            Tuple: Return number of prompt token and number of completion tokens.
        """
        last_response = responses[-1]
        js = json.loads(last_response)
        return js['usage']['input_tokens'], js['usage']['output_tokens']
