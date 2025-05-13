import os
import time
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.utils.chat_service import ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage, Usage
from evalscope.utils.logger import get_logger
from evalscope.utils.model_utils import fix_do_sample_warning
from ..local_model import LocalModel
from .base_adapter import BaseModelAdapter

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

    def _model_generate(self,
                        formatted_prompts: List[str],
                        infer_cfg: Dict[str, Any] = None) -> Tuple[List[List[str]], List[int]]:
        """
        Args:
            formatted_prompts: The formatted prompts.
            infer_cfg: The inference configuration.
        Returns:
            The prediction results.
        """
        if infer_cfg is None:
            infer_cfg = {}

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

        # Get input ids
        inputs = self.tokenizer(
            formatted_prompts, return_tensors='pt', padding=True, truncation=True,
            padding_side='left').to(self.model.device)  # padding_side='left' is important for chat model
        input_ids = inputs['input_ids']

        # Run inference
        output_ids = self.model.generate(**inputs, generation_config=self.generation_config)

        # Decode output
        responses = []
        input_lengths = [len(self.tokenizer.encode(prompt)) for prompt in formatted_prompts]
        for i in range(0, len(output_ids), num_return_sequences):
            query_responses = []
            for j in range(num_return_sequences):
                output = output_ids[i + j]
                response = self.tokenizer.decode(
                    output[len(input_ids[i // num_return_sequences]):], skip_special_tokens=True)
                query_responses.append(response)
            responses.append(query_responses)

        return responses, input_lengths

    def _prepare_inputs(self, inputs: List[dict], infer_cfg: dict = {}) -> List[str]:
        """
        Prepare the inputs for the model.
        Args:
            inputs: The input data.
            infer_cfg: The inference configuration.
        Returns:
            The prepared inputs and system prompts.
        """
        queries = []
        system_prompts = []
        message_list = []

        for input_item in inputs:
            queries.append(input_item['data'][0])
            system_prompts.append(input_item.get('system_prompt', None))
            if input_item.get('messages', None):
                message_list.append(input_item.get('messages', None))

        # For non chat model, use the original queries as the input
        if self.tokenizer.chat_template is None:
            return queries

        # For chat model, use the messages as the input
        # if message_list is None, use the queries as the input
        if len(message_list) == 0:
            for i, query in enumerate(queries):
                messages = [ChatMessage(role='user', content=query)]
                if i < len(system_prompts) and system_prompts[i]:
                    messages = [ChatMessage(role='system', content=system_prompts[i])] + messages
                message_list.append(messages)

        # Format the messages
        formatted_prompts = []
        for messages in message_list:
            # apply chat template
            chat_template_kwargs = infer_cfg.get('chat_template_kwargs', None)
            if chat_template_kwargs is not None:
                prompts = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs)
            else:
                prompts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(prompts)

        logger.debug(f'formatted_prompts: {formatted_prompts}')
        return formatted_prompts

    @torch.no_grad()
    def predict(self, inputs: List[dict], infer_cfg: Optional[dict] = {}) -> List[dict]:
        """
        Args:
            inputs: The input data.
            infer_cfg: The inference configuration.
        Returns:
            The prediction results.
        """

        # Process inputs
        formatted_prompts = self._prepare_inputs(inputs, infer_cfg)

        # Run inference
        responses, input_lengths = self._model_generate(formatted_prompts, infer_cfg)

        # Process outputs
        results = []
        for response, input_length in zip(responses, input_lengths):
            choices_list = []
            completion_tokens = 0

            for index, one_response in enumerate(response):
                choice = ChatCompletionResponseChoice(
                    index=index, message=ChatMessage(content=one_response, role='assistant'), finish_reason='stop')
                choices_list.append(choice)

                completion_tokens += len(self.tokenizer.encode(one_response))

            usage = Usage(
                prompt_tokens=input_length,
                completion_tokens=completion_tokens,
                total_tokens=input_length + completion_tokens)

            res_d = ChatCompletionResponse(
                model=self.model_id,
                choices=choices_list,
                object='chat.completion',
                created=int(time.time()),
                usage=usage).model_dump(exclude_unset=True)

            results.append(res_d)

        return results
