import sys
from typing import Any, Dict, Iterator
import json
from llmuses.perf.llm_parser_base import PerfPluginBase
from transformers import AutoTokenizer

class OpenaiPluginBase(PerfPluginBase):
    """Base of openai interface.
    """
    def __init__(self, mode_path: str):
        """Init the plugin

        Args:
            mode_path (str): The model path, we use the tokenizer 
                weight in the model to calculate the number of the
                input and output tokens.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(mode_path)
    def build_request(self,
                      model: str,
                      prompt: str = None,
                      dataset: str = None,
                      max_length: int = sys.maxsize,
                      min_length: int = 0,
                      **kwargs: Any) -> Iterator[Dict]:
        raise Exception('Not implemented!')

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
        
        
