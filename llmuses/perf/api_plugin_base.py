from abc import abstractmethod
import sys
from typing import Any, Dict, Iterator, List, Tuple
import json

class ApiPluginBase:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        
    @abstractmethod
    def build_request(self, 
                      model:str,
                      prompt: str,                    
                      query_template: str)->Dict:
        """Build the request.

        Args:
            model (str): The request model.
            prompt (str): The input prompt, if not None, use prompt generate request. Defaults to None.
            query_template (str): The query template, the plugin will replace "%m" with model and "%p" with prompt.

        Raises:
            NotImplementedError: The request is not impletion.

        Returns:
            Dict: return a request.
        """
        raise NotImplementedError
    
    @abstractmethod
    def parse_responses(self, 
                        responses: List, 
                        request: Any=None,
                        **kwargs:Any) -> Tuple[int, int]:
        """Parser responses and return number of request and response tokens.

        Args:
            responses (List[bytes]): List of http response body, for stream output,
                there are multiple responses, each is bytes, for general only one. 
            request (Any): The request body.

        Returns:
            Tuple: (Number of prompt_tokens and number of completion_tokens).
        """
        raise NotImplementedError  
    
    @staticmethod
    def replace_values(input_json: Any, model: str, prompt: str):
        if isinstance(input_json, dict):  
            for key, value in input_json.items():
                if isinstance(value, str):
                    input_json[key] = value.replace("%m", model).replace("%p", prompt)
                else:                    
                    ApiPluginBase.replace_values(value, model, prompt)  
        elif isinstance(input_json, list): 
            for idx, item in enumerate(input_json):
                if isinstance(item, str):
                    input_json[idx] = item.replace("%m", model).replace("%p", prompt)
                else:
                    ApiPluginBase.replace_values(item, model, prompt)
        elif isinstance(input_json, str):
            input_json = input_json.replace("%m", model).replace("%p", prompt)
        else:
            pass
