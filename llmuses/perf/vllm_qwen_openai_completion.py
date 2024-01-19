
from typing import Dict
import json

def get_query(model: str, prompt: str, **kwargs) -> Dict:
    """Get the query of from prompt and other parameters.

    Args:
        model (str): The model name.
        prompt (str): The input prompt.

    Returns:
        Dict: The request body.
    """    
    return {
        "model": model,
        "prompt": "<|im_start|>system\nYour are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"%prompt,
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "stream": True,
        **kwargs
        }
    
def parse_responses(responses, **kwargs) -> Dict:
    """Parser responses and return number of request and response tokens.

    Args:
        responses (List[bytes]): List of http response body, for stream output,
            there are multiple responses, for general only one. 

    Returns:
        Dict: Return the prompt token and completion tokens.
    """
    last_response = responses[-1]
    js = json.loads(last_response)
    return js['usage']['prompt_tokens'], js['usage']['completion_tokens']