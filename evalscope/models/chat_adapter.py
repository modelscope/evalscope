import os
import time
import torch
from typing import List, Union

from evalscope.constants import OutputType
from evalscope.models.base_adapter import BaseModelAdapter
from evalscope.models.local_model import LocalModel
from evalscope.models.register import register_model_adapter
from evalscope.utils.chat_service import ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage
from evalscope.utils.logger import get_logger
from evalscope.utils.model_utils import fix_do_sample_warning

logger = get_logger()


@register_model_adapter(OutputType.GENERATION)
class ChatGenerationModelAdapter(BaseModelAdapter):
    """
    Chat generation model adapter.
    """

    def __init__(self, model: LocalModel, **kwargs):
        super().__init__(model)

        self.generation_config = self._parse_generation_config(self.tokenizer, self.model)

        custom_generation_config = kwargs.pop('generation_config', None)
        custom_chat_template = kwargs.pop('chat_template', None)

        if custom_generation_config:
            logger.info('Updating generation config ...')
            self.generation_config.update(**custom_generation_config)

        if custom_chat_template:
            self.tokenizer.chat_template = custom_chat_template
            logger.info(f'Using custom chat template: {custom_chat_template}')

    def _parse_generation_config(self, tokenizer, model):
        from modelscope import GenerationConfig

        generation_config = getattr(model, 'generation_config', GenerationConfig(do_sample=False))

        try:
            remote_config = GenerationConfig.from_pretrained(
                self.model_id, revision=self.model_revision, trust_remote_code=True)
            generation_config.update(**remote_config.to_dict())
        except Exception:
            logger.warning(f'Failed to get generation config of {self.model_id} from model hub, use default.')

        if isinstance(self.model_id, str) and os.path.exists(self.model_id):
            logger.warning(f'Got local model dir: {self.model_id}')

        if tokenizer.eos_token_id is not None:
            generation_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        if generation_config.max_new_tokens is None:
            generation_config.max_new_tokens = 2048

        return generation_config

    def _model_generate(self, queries: List[str], system_prompts: List[str] = None, infer_cfg: dict = {}) -> List[str]:
        """
        Args:
            queries: The input queries.
            system_prompts: The system prompts.
            infer_cfg: The inference configuration.
        Returns:
            The prediction results.
        """
        # Process infer_cfg
        num_return_sequences = infer_cfg.get('num_return_sequences', 1)
        if num_return_sequences > 1:
            infer_cfg['do_sample'] = True

        # stop settings
        stop = infer_cfg.get('stop', [])
        if stop:
            eos_token_id = self.tokenizer.encode(stop, add_special_tokens=False)[0]
        else:
            eos_token_id = self.tokenizer.eos_token_id

        if eos_token_id is not None:
            infer_cfg['eos_token_id'] = eos_token_id

        self.generation_config.update(**infer_cfg)
        fix_do_sample_warning(self.generation_config)

        # For chat model, use the chat template to format the input
        if self.tokenizer.chat_template is not None:
            formatted_prompts = []
            for i, query in enumerate(queries):
                messages = [ChatMessage(role='user', content=query)]
                if i < len(system_prompts) and system_prompts[i]:
                    messages = [ChatMessage(role='system', content=system_prompts[i])] + messages
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        else:
            # For base model, use the queries as the input
            formatted_prompts = queries

        logger.debug(f'formatted_prompts: {formatted_prompts}')

        # Get input ids
        inputs = self.tokenizer(
            formatted_prompts, return_tensors='pt', padding=True, truncation=True,
            padding_side='left').to(self.model.device)  # padding_side='left' is important for chat model
        input_ids = inputs['input_ids']

        # Run inference
        output_ids = self.model.generate(**inputs, generation_config=self.generation_config)

        responses = []
        for i in range(0, len(output_ids), num_return_sequences):
            query_responses = []
            for j in range(num_return_sequences):
                output = output_ids[i + j]
                response = self.tokenizer.decode(
                    output[len(input_ids[i // num_return_sequences]):], skip_special_tokens=True)
                query_responses.append(response)
            responses.append(query_responses)

        return responses

    @torch.no_grad()
    def predict(self, inputs: List[dict], infer_cfg: dict = {}) -> List[dict]:
        """
        Args:
            inputs: The input data.
            infer_cfg: The inference configuration.
        Returns:
            The prediction results.
        """

        # Process inputs
        queries = []
        system_prompts = []

        for input_item in inputs:
            queries.append(input_item['data'][0])
            system_prompts.append(input_item.get('system_prompt', None))

        responses = self._model_generate(queries, system_prompts, infer_cfg)

        results = []
        for response in responses:
            choices_list = [
                ChatCompletionResponseChoice(
                    index=index, message=ChatMessage(content=one_response, role='assistant'), finish_reason='stop')
                for index, one_response in enumerate(response)
            ]

            res_d = ChatCompletionResponse(
                model=self.model_id,
                choices=choices_list,
                object='chat.completion',
                created=int(time.time()),
                usage=None).model_dump(exclude_unset=True)

            results.append(res_d)

        return results
