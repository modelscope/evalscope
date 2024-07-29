from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryParameters:
    model: str
    prompt: Optional[str]
    dataset: Optional[str]
    query_template: Optional[str]
    dataset_path: Optional[str]
    frequency_penalty: Optional[float]
    logprobs: Optional[bool]
    max_tokens: Optional[int]
    n_choices: Optional[int]
    seed: Optional[int]
    stop: Optional[str]
    stream: Optional[bool]
    temperature: Optional[float]
    top_p: Optional[float]
    max_prompt_length: Optional[int]
    min_prompt_length: Optional[int]    
    include_usage: Optional[bool]
    
    def __init__(self, args):
        self.model = args.model
        self.prompt = args.prompt
        self.dataset = args.dataset
        self.query_template = args.query_template
        self.dataset_path = args.dataset_path
        self.frequency_penalty = args.frequency_penalty
        self.logprobs = args.logprobs
        self.max_tokens = args.max_tokens
        self.n_choices = args.n_choices
        self.seed = args.seed
        self.stop = args.stop
        self.stream = args.stream
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_prompt_length = args.max_prompt_length
        self.min_prompt_length = args.min_prompt_length
        self.stop_token_ids = args.stop_token_ids