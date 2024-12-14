# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from swift.llm import get_default_template_type, get_model_tokenizer, get_template, inference
from swift.utils import seed_everything

# TODO: Support custom model for swift infer


@dataclass
class SwiftInferArgs:
    model_id_or_path: str
    model_type: str
    max_new_tokens: int = 2048


class SwiftInfer:

    def __init__(self, args: SwiftInferArgs):
        model_type = args.model_type
        template_type = get_default_template_type(model_type)
        model, tokenizer = get_model_tokenizer(
            model_type, model_id_or_path=args.model_id_or_path, model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = args.max_new_tokens
        print(f'** Generation config: {model.generation_config}')

        template = get_template(template_type, tokenizer)
        seed_everything(42)

        self.tokenizer = tokenizer
        self.model = model
        self.template = template

    def predict(self, system: str, query: str, history: list):

        response, history = inference(self.model, self.template, query=query, system=system, history=history)

        return response
