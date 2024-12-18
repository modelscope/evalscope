# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI, Inc. and its affiliates.
# flake8: noqa
import numpy as np
import os
import sys
import time
import torch
from abc import ABC, abstractmethod
from copy import deepcopy
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torch import dtype
from typing import Any, Dict, List, Union

from evalscope.constants import DEFAULT_MODEL_CACHE_DIR
from evalscope.models.custom import CustomModel
from evalscope.utils.chat_service import ChatMessage
from evalscope.utils.logger import get_logger
from evalscope.utils.model_utils import fix_do_sample_warning

logger = get_logger()


class BaseModelAdapter(ABC):
    """
    Base class for model adapter.
    """

    def __init__(self, model, tokenizer, model_cfg: dict):
        """
        Args:
            model: The model instance which is compatible with
                AutoModel/AutoModelForCausalLM/AutoModelForSeq2SeqLM of transformers.
            tokenizer: The tokenizer instance which is compatible with AutoTokenizer of transformers.
            model_cfg:
                Attributes: model_id, model_revision, device_map, torch_dtype
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg

    @abstractmethod
    @torch.no_grad()
    def predict(self, *args, **kwargs) -> Any:
        """
        Model prediction func.
        """
        raise NotImplementedError


class MultiChoiceModelAdapter(BaseModelAdapter):
    """ The multi-choice model adapter. """

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(self,
                 model_id: str,
                 device_map: str = 'auto',
                 torch_dtype: dtype = torch.bfloat16,
                 model_revision: str = None,
                 max_length: int = None,
                 cache_dir: str = None,
                 **kwargs):
        """
        Args:
            model_id: The model id on ModelScope, or local model_dir.  TODO: torch.nn.module to be supported.
            device_map: The device map for model inference.
            torch_dtype: The torch dtype for model inference. Default: torch.bfloat16.
            model_revision: The model revision on ModelScope. Default: None.
            max_length: The max length of input sequence. Default: None.
            **kwargs: Other args.
        """
        model_cache_dir = cache_dir or DEFAULT_MODEL_CACHE_DIR

        self.model_id: str = model_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.warning(f'Device: {self.device}')

        torch_dtype = torch_dtype if torch_dtype is not None else 'auto'

        model_cfg: dict = dict()
        model_cfg['model_id'] = model_id
        model_cfg['device_map'] = device_map
        model_cfg['torch_dtype'] = str(torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,  # self.model_id
            revision=model_revision,
            trust_remote_code=True,
            cache_dir=model_cache_dir,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,  # self.model_id
            revision=model_revision,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            cache_dir=model_cache_dir,
        )

        super().__init__(model=model, tokenizer=tokenizer, model_cfg=model_cfg)

        self._max_length = max_length

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        seqlen_config_attrs = ('n_positions', 'max_position_embeddings', 'n_ctx')
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, 'model_max_length'):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @torch.no_grad()
    def predict(self, inputs: dict, infer_cfg: dict = None) -> dict:
        """
        Multi-choice model prediction func.

        Args:
            inputs (dict): The inputs for a doc. Format:
                {'data': [full_prompt], 'multi_choices': ['A', 'B', 'C', 'D']}

            infer_cfg (dict): inference configuration.

        Returns:
            res (dict): The model prediction results. Format:
            {
              'choices': [
                {
                  'index': 0,
                  'message': {
                    'content': [-14.9609, -13.6015, ...],  # loglikelihood values for inputs context-continuation pairs.
                    'role': 'assistant'
                  }
                }
              ],
              'created': 1677664795,
              # For models on the ModelScope or HuggingFace, concat model_id and revision with "-".
              'model': 'gpt-3.5-turbo-0613',
              'object': 'chat.completion',
              'usage': {
                'completion_tokens': 17,
                'prompt_tokens': 57,
                'total_tokens': 74
              }
            }
        """
        infer_cfg = infer_cfg or {}
        self.model.generation_config.update(**infer_cfg)

        input_data = inputs['data']
        multi_choices = inputs['multi_choices']

        output, input_info = self._get_logits(self.tokenizer, self.model, input_data)
        assert output.shape[0] == 1
        logits = output.flatten()

        choice_logits = [logits[self.tokenizer(ch)['input_ids'][-1:]] for ch in multi_choices]
        softval = torch.nn.functional.softmax(torch.tensor(choice_logits).float(), dim=0)

        if softval.dtype in {torch.bfloat16, torch.float16}:
            softval = softval.to(dtype=torch.float32)
        probs = softval.detach().cpu().numpy()
        pred: str = multi_choices[int(np.argmax(probs))]  # Format: A or B or C or D

        res_d = {
            'choices': [{
                'index': 0,
                'message': {
                    'content': pred,
                    'role': 'assistant'
                }
            }],
            'created': time.time(),
            'model': self.model_id,
            'object': 'chat.completion',
            'usage': {}
        }

        return res_d

    @staticmethod
    def _get_logits(tokenizer, model, inputs: List[str]):
        input_ids = tokenizer(inputs, padding=False)['input_ids']
        input_ids = torch.tensor(input_ids, device=model.device)
        tokens = {'input_ids': input_ids}

        outputs = model(input_ids)['logits']
        logits = outputs[:, -1, :]
        log_probs = torch.nn.functional.softmax(logits, dim=-1)
        return log_probs, {'tokens': tokens}


class ContinuationLogitsModelAdapter(MultiChoiceModelAdapter):

    def __init__(self,
                 model_id: str,
                 device_map: str = 'auto',
                 torch_dtype: dtype = torch.bfloat16,
                 model_revision: str = None,
                 cache_dir: str = None,
                 **kwargs):
        """
        Continuation-logits model adapter.

        Args:
            model_id: The model id on ModelScope, or local model_dir.
            device_map: The device map for model inference.
            torch_dtype: The torch dtype for model inference. Default: torch.bfloat16.
            model_revision: The model revision on ModelScope. Default: None.
            **kwargs: Other args.
        """

        super().__init__(
            model_id=model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            model_revision=model_revision,
            cache_dir=cache_dir,
            **kwargs)

    @torch.no_grad()
    def predict(self, inputs: dict, infer_cfg: dict = None) -> dict:
        """
        Multi-choice model prediction func.
        Args:
            inputs (dict): The inputs for a doc. Format:
                {'data': [(context, continuation), ...]}
            infer_cfg (dict): inference configuration.
        Returns:
            res (dict): The model prediction results. Format:
            {
              'choices': [
                {
                  'index': 0,
                  'message': {
                    'content': [-14.9609, -13.6015, ...],  # loglikelihood values for inputs context-continuation pairs.
                    'role': 'assistant'
                  }
                }
              ],
              'created': 1677664795,
              # For models on the ModelScope or HuggingFace, concat model_id and revision with "-".
              'model': 'gpt-3.5-turbo-0613',
              'object': 'chat.completion',
              'usage': {
                'completion_tokens': 17,
                'prompt_tokens': 57,
                'total_tokens': 74
              }
            }
        """
        infer_cfg = infer_cfg or {}

        pred_list: list = self.loglikelihood(inputs=inputs['data'], infer_cfg=infer_cfg)

        res_d = {
            'choices': [{
                'index': 0,
                'message': {
                    'content': pred_list,
                    'role': 'assistant'
                }
            }],
            'created': time.time(),
            'model': self.model_id,
            'object': 'chat.completion',
            'usage': {}
        }
        return res_d

    def loglikelihood(self, inputs: list, infer_cfg: dict = None) -> list:
        self.model.generation_config.update(**infer_cfg)
        # To predict one doc
        doc_ele_pred = []
        for ctx, continuation in inputs:

            # ctx_enc shape: [context_tok_len]  cont_enc shape: [continuation_tok_len]
            ctx_enc, cont_enc = self._encode_pair(ctx, continuation)

            inputs_tokens = torch.tensor(
                (ctx_enc.tolist() + cont_enc.tolist())[-(self.max_length + 1):][:-1],
                dtype=torch.long,
                device=self.model.device).unsqueeze(0)

            logits = self.model(inputs_tokens)[0]
            logits = torch.nn.functional.log_softmax(logits.float(), dim=-1)

            logits = logits[:, -len(cont_enc):, :]
            cont_enc = cont_enc.unsqueeze(0).unsqueeze(-1)
            logits = torch.gather(logits.cpu(), 2, cont_enc.cpu()).squeeze(-1)

            choice_score = float(logits.sum())
            doc_ele_pred.append(choice_score)

        # e.g. [-2.3, -9.2, -12.9, 1.1], length=len(choices)
        return doc_ele_pred

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation, padding=False)['input_ids']
        whole_enc = torch.tensor(whole_enc, device=self.device)

        context_enc = self.tokenizer(context, padding=False)['input_ids']
        context_enc = torch.tensor(context_enc, device=self.device)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc


class ChatGenerationModelAdapter(BaseModelAdapter):

    def __init__(self,
                 model_id: str,
                 model_revision: str = 'master',
                 device_map: str = 'auto',
                 torch_dtype: dtype = 'auto',
                 cache_dir: str = None,
                 **kwargs):
        """
        Chat completion model adapter. Tasks of chat and generation are supported.

        Args:
            model_id: The model id on ModelScope, or local model_dir.
            model_revision: The model revision on ModelScope. Default: None.
            device_map: The device map for model inference.
            torch_dtype: The torch dtype for model inference. Default: 'auto'.
            **kwargs: Other args.
        """

        custom_generation_config = kwargs.pop('generation_config', None)
        custom_chat_template = kwargs.pop('chat_template', None)
        model_cache_dir = cache_dir or DEFAULT_MODEL_CACHE_DIR

        self.model_id: str = model_id
        self.model_revision: str = model_revision
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.warning(f'Device: {self.device}')

        torch_dtype = torch_dtype if torch_dtype is not None else 'auto'

        model_cfg: dict = dict()
        model_cfg['model_id'] = model_id
        model_cfg['device_map'] = device_map
        model_cfg['torch_dtype'] = str(torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=model_revision,
            trust_remote_code=True,
            cache_dir=model_cache_dir,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=model_revision,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            cache_dir=model_cache_dir,
        )

        self.generation_config = self._parse_generation_config(tokenizer, model)

        if custom_generation_config:
            logger.info('Updating generation config ...')
            self.generation_config.update(**custom_generation_config)

        if custom_chat_template:
            tokenizer.chat_template = custom_chat_template
            logger.info(f'Using custom chat template: {custom_chat_template}')

        super().__init__(model=model, tokenizer=tokenizer, model_cfg=model_cfg)

    def _parse_generation_config(self, tokenizer, model):
        generation_config = getattr(model, 'generation_config', GenerationConfig(do_sample=False))

        try:
            remote_config = GenerationConfig.from_pretrained(
                self.model_id, revision=self.model_revision, trust_remote_code=True)
            generation_config.update(**remote_config.to_dict())
        except:
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

    def _model_generate(self, query: str, infer_cfg: dict) -> str:
        messages = [ChatMessage(role='user', content=query)]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

        # Process inputs
        if isinstance(inputs, str):
            query = inputs
        elif isinstance(inputs, dict):
            query = inputs['data'][0]
        elif isinstance(inputs, list):
            query = '\n'.join(inputs)
        else:
            raise TypeError(f'Unsupported inputs type: {type(inputs)}')

        response = self._model_generate(query, infer_cfg)

        choices_list = [{'index': 0, 'message': {'content': response, 'role': 'assistant'}}]

        res_d = {
            'choices': choices_list,
            'created': time.time(),
            'model': self.model_id,
            'object': 'chat.completion',
            'usage': {}
        }

        return res_d


class CustomModelAdapter(BaseModelAdapter):

    def __init__(self, custom_model: CustomModel, **kwargs):
        """
        Custom model adapter.

        Args:
            custom_model: The custom model instance.
            **kwargs: Other args.
        """
        self.custom_model = custom_model
        super(CustomModelAdapter, self).__init__(model=None, tokenizer=None, model_cfg=custom_model.config)

    def predict(self, inputs: Union[str, dict, list], **kwargs) -> List[Dict[str, Any]]:
        """
        Model prediction func.

        Args:
            inputs (Union[str, dict, list]): The input data. Depending on the specific model.
                str: 'xxx'
                dict: {'data': [full_prompt]}
                list: ['xxx', 'yyy', 'zzz']
            **kwargs: kwargs

        Returns:
            res (dict): The model prediction results. Format:
            {
              'choices': [
                {
                  'index': 0,
                  'message': {
                    'content': 'xxx',
                    'role': 'assistant'
                  }
                }
              ],
              'created': 1677664795,
              'model': 'gpt-3.5-turbo-0613',   # should be model_id
              'object': 'chat.completion',
              'usage': {
                'completion_tokens': 17,
                'prompt_tokens': 57,
                'total_tokens': 74
              }
            }
        """
        in_prompts = []

        # Note: here we assume the inputs are all prompts for the benchmark.
        for input_prompt in inputs:
            if isinstance(input_prompt, str):
                in_prompts.append(input_prompt)
            elif isinstance(input_prompt, dict):
                # TODO: to be supported for continuation list like truthful_qa
                in_prompts.append(input_prompt['data'][0])
            elif isinstance(input_prompt, list):
                in_prompts.append('\n'.join(input_prompt))
            else:
                raise TypeError(f'Unsupported inputs type: {type(input_prompt)}')

        return self.custom_model.predict(prompts=in_prompts, **kwargs)
