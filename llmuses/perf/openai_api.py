from typing import Any, Dict, Iterator
import json
from llmuses.perf.api_plugin_base import ApiPluginBase
from transformers import AutoTokenizer
from llmuses.perf.plugin_registry import register_api

@register_api("openai")
class OpenaiPlugin(ApiPluginBase):
    """Base of openai interface.
    """
    def __init__(self, mode_path: str):
        """Init the plugin

        Args:
            mode_path (str): The model path, we use the tokenizer 
                weight in the model to calculate the number of the
                input and output tokens.
        """
        super().__init__(model_path=mode_path)
        self.tokenizer = AutoTokenizer.from_pretrained(mode_path)

    def build_request(self,
                      model: str,
                      prompt: str,
                      query_template: str) -> Dict:
        """Build the openai format request based on prompt, dataset

        Args:
            model (str): The model to use.
            prompt (str, optional): The user prompt. Defaults to None.
            query_template (str): The query template, the plugin will replace "%m" with model and "%p" with prompt.

        Raises:
            Exception: NotImplemented

        Returns:
            Dict: The request body. None if prompt format is error.
        """
        try:
            query = json.loads(query_template)
            ApiPluginBase.replace_values(query, model, prompt)
            return query
        except Exception as e:
            print(e)
            print('Prompt: %s invalidate!'%prompt)
            return None

    def parse_responses(self, responses, request: Any = None, **kwargs) -> Dict:
        """Parser responses and return number of request and response tokens.
           sample of the output delta:
           {"id":"4","object":"chat.completion.chunk","created":1714030870,"model":"llama3","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}


        Args:
            responses (List[bytes]): List of http response body, for stream output,
                there are multiple responses, for general only one. 
            kwargs: (Any): The command line --parameter content.
        Returns:
            Tuple: Return number of prompt token and number of completion tokens.
        """
        full_response_content = ''
        delta_contents = []
        for response in responses:
            js = json.loads(response)
            if 'choices' in js:
                delta = js['choices'][0]['delta']
                if delta and 'content' in delta:
                    delta_contents.append(delta['content'])
        full_response_content = ''.join([m for m in delta_contents])
        input_tokens = len(self.tokenizer.encode(request['messages'][0]['content']))
        output_tokens = len(self.tokenizer.encode(full_response_content))
        
        return input_tokens, output_tokens
        
        
