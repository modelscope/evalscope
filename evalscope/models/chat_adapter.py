import os
import time
import torch
from typing import Union

from evalscope.models.base_adapter import BaseModelAdapter
from evalscope.models.local_model import LocalModel
from evalscope.utils.chat_service import ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage
from evalscope.utils.logger import get_logger
from evalscope.utils.model_utils import fix_do_sample_warning

logger = get_logger()


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

    def _model_generate(self, query: str, system_prompt: str = None, infer_cfg: dict = {}) -> str:
        """
        Args:
            query: The input query.
            system_prompt: The system prompt.
            infer_cfg: The inference configuration.
        Returns:
            The prediction result.
        """
        # For chat model, use the chat template to format the input
        if self.tokenizer.chat_template is not None:
            messages = [ChatMessage(role='user', content=query)]
            if system_prompt:
                messages = [ChatMessage(role='system', content=system_prompt)] + messages
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # For base model, use the query as the input
            formatted_prompt = query

        inputs = self.tokenizer(formatted_prompt, return_tensors='pt', padding=True).to(self.device)
        input_ids = inputs['input_ids']

        # Process infer_cfg
        if isinstance(infer_cfg.get('num_return_sequences'), int) and infer_cfg['num_return_sequences'] > 1:
            infer_cfg['do_sample'] = True

        # stop settings
        stop = infer_cfg.get('stop', None)
        eos_token_id = self.tokenizer.encode(stop, add_special_tokens=False)[0] \
            if stop else self.tokenizer.eos_token_id

        if eos_token_id is not None:
            infer_cfg['eos_token_id'] = eos_token_id
            infer_cfg['pad_token_id'] = eos_token_id  # setting eos_token_id as pad token

        self.generation_config.update(**infer_cfg)
        fix_do_sample_warning(self.generation_config)

        # Run inference
        output_ids = self.model.generate(**inputs, generation_config=self.generation_config)

        response = self.tokenizer.decode(output_ids[0, len(input_ids[0]):], skip_special_tokens=True)
        return response

    @torch.no_grad()
    def predict(self, inputs: Union[str, dict, list], infer_cfg: dict = {}) -> dict:
        """
        Args:
            inputs: The input data.
            infer_cfg: The inference configuration.
        Returns:
            The prediction result.
        """

        # Process inputs
        if isinstance(inputs, str):
            query = inputs
            system_prompt = None
        elif isinstance(inputs, dict):
            query = inputs['data'][0]
            system_prompt = inputs.get('system_prompt', None)
        elif isinstance(inputs, list):
            query = '\n'.join(inputs)
            system_prompt = None
        else:
            raise TypeError(f'Unsupported inputs type: {type(inputs)}')

        response = self._model_generate(query, system_prompt, infer_cfg)

        choices_list = [
            ChatCompletionResponseChoice(
                index=0, message=ChatMessage(content=response, role='assistant'), finish_reason='stop')
        ]

        res_d = ChatCompletionResponse(
            model=self.model_id, choices=choices_list, object='chat.completion', created=int(time.time()),
            usage=None).model_dump(exclude_unset=True)

        return res_d
