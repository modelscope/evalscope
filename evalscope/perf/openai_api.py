from typing import Any, Dict, Iterator, List
import json
from evalscope.perf.api_plugin_base import ApiPluginBase
from transformers import AutoTokenizer
from evalscope.perf.plugin_registry import register_api
from evalscope.perf.query_parameters import QueryParameters

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
        if mode_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(mode_path)
        else:
            self.tokenizer = None

    def build_request(self, messages: List[Dict], param: QueryParameters) -> Dict:
        """Build the openai format request based on prompt, dataset

        Args:
            message (Dict): The basic message to generator query.
            param (QueryParameters): The query parameters.

        Raises:
            Exception: NotImplemented

        Returns:
            Dict: The request body. None if prompt format is error.
        """
        try:
            if param.query_template is not None:
                query = json.loads(param.query_template)
                query['messages'] = messages   # replace template messages with input messages.
                return self.__compose_query_from_parameter(query, param)
            else:
                query = {'messages': messages}
                return self.__compose_query_from_parameter(query, param)
        except Exception as e:
            print(e)
            return None
        
    def __compose_query_from_parameter(self, payload: Dict, param: QueryParameters):
        payload['model'] = param.model
        if param.max_tokens is not None:
            payload['max_tokens'] = param.max_tokens
        if param.frequency_penalty is not None:
            payload['frequency_penalty'] = param.frequency_penalty
        if param.logprobs is not None:
            payload['logprobs'] = param.logprobs
        if param.n_choices is not None:
            payload['n'] = param.n_choices
        if param.seed is not None:
            payload['seed'] = param.seed
        if param.stop is not None:
            payload['stop'] = param.stop
        if param.stream is not None and param.stream:
            payload['stream'] = param.stream
            payload['stream_options'] = {"include_usage": True}
        if param.stop_token_ids is not None:
            payload['stop_token_ids'] = param.stop_token_ids
        if param.temperature is not None:
            payload['temperature'] = param.temperature
        if param.top_p is not None:
            payload['top_p'] = param.top_p
        return payload

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
        delta_contents = {}
        input_tokens = None
        output_tokens = None
        for response in responses:
            js = json.loads(response)
            if js['object'] == 'chat.completion':
                for choice in js['choices']:
                    delta_contents[choice['index']] = [choice['message']['content']]     
                input_tokens = js['usage']['prompt_tokens']
                output_tokens = js['usage']['completion_tokens']                 
            else:  # 'object' == "chat.completion.chunk":
                if 'choices' in js:
                    for choice in js['choices']:
                        if 'delta' in choice and 'index' in choice:
                            delta = choice['delta']
                            idx = choice['index']
                            if 'content' in delta:
                                delta_content = delta['content']
                                if idx in delta_contents:
                                    delta_contents[idx].append(delta_content)
                                else:
                                    delta_contents[idx] = [delta_content]
                # usage in chunk: {"id":"","object":"chat.completion.chunk","created":1718269986,"model":"llama3",
                # "choices":[],"usage":{"prompt_tokens":32,"total_tokens":384,"completion_tokens":352}}
                if 'usage' in js and js['usage']:
                    input_tokens = js['usage']['prompt_tokens']
                    output_tokens = js['usage']['completion_tokens']     
        if input_tokens is None and output_tokens is None and self.tokenizer is not None:                
            input_tokens = 0
            output_tokens = 0
            for idx, choice_contents in delta_contents.items():
                full_response_content = ''.join([m for m in choice_contents])
                input_tokens += len(self.tokenizer.encode(request['messages'][0]['content']))
                output_tokens += len(self.tokenizer.encode(full_response_content))
        elif input_tokens is None and output_tokens is None:  # no usage info get.
            input_tokens = 0
            output_tokens = 0            
        
        return input_tokens, output_tokens
        
        
