# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI, Inc. and its affiliates.
# flake8: noqa
import os
import sys
from typing import List, Any, Union
import numpy as np
import time
import gc
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
from torch import dtype

from llmuses.constants import DEFAULT_ROOT_CACHE_DIR
from llmuses.models.template import get_template, StopWordsCriteria, MODEL_TEMPLATE_MAP
from llmuses.utils.logger import get_logger
from transformers import StoppingCriteriaList

logger = get_logger()

# Notes:
# - modelscope>=1.9.5


def get_model_cache_dir(root_cache_dir: str):
    model_cache_dir = os.path.join(root_cache_dir, 'models')
    model_cache_dir = os.path.expanduser(model_cache_dir)
    os.makedirs(model_cache_dir, exist_ok=True)
    return model_cache_dir


def load_model(
    model_id: str,
    device_map: str = 'auto',
    torch_dtype: dtype = torch.bfloat16,
    model_revision: str = None,
    cache_dir: str = DEFAULT_ROOT_CACHE_DIR        
):
    model_cache_dir = get_model_cache_dir(cache_dir)

    torch_dtype = torch_dtype if torch_dtype is not None else 'auto'

    model_cfg: dict = dict()
    model_cfg['model_id'] = model_id
    model_cfg['device_map'] = device_map
    model_cfg['torch_dtype'] = str(torch_dtype)

    from modelscope.utils.hf_util import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              revision=model_revision,
                                              trust_remote_code=True,
                                              cache_dir=model_cache_dir,)

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 revision=model_revision,
                                                 device_map=device_map,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch_dtype,
                                                 cache_dir=model_cache_dir,)

    return model, tokenizer, model_cfg
    

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
    
    def del_model_cache(self):
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            del self.model.module
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        gc.collect()


class MultiChoiceModelAdapter(BaseModelAdapter):
    """ The multi-choice model adapter. """

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(self,
                 model_id: str,
                 model,
                 tokenizer,
                 model_cfg,
                 max_length: int = None,
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
        self.model_id: str = model_id

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.warning(f'**Device: {self.device}')

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
        pred: str = multi_choices[int(np.argmax(probs))]        # Format: A or B or C or D

        res_d = {
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'content': pred,
                        'role': 'assistant'
                    }
                }
            ],
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
                 model,
                 tokenizer,
                 model_cfg,
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

        super().__init__(model_id=model_id,
                         model=model,
                         tokenizer=tokenizer,
                         model_cfg=model_cfg,
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
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'content': pred_list,
                        'role': 'assistant'
                    }
                }
            ],
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
                 model,
                 tokenizer,
                 model_cfg,
                 model_revision: str = None,
                 **kwargs):
        """
        Chat completion model adapter. Tasks of chat and generation are supported.

        Args:
            model_id: The model id on ModelScope, or local model_dir.
            model_revision: The model revision on ModelScope. Default: None.
            device_map: The device map for model inference.
            torch_dtype: The torch dtype for model inference. Default: torch.float16.
            **kwargs: Other args.
        """
        self.model_id: str = model_id
        self.model_revision: str = model_revision

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.warning(f'**Device: {self.device}')

        self.origin_tokenizer = deepcopy(tokenizer)

        self.generation_config, self.generation_template = self._parse_generation_config(tokenizer, model)
        logger.info(f'**Generation config init: {self.generation_config.to_dict()}')

        super().__init__(model=model, tokenizer=self.generation_template.tokenizer, model_cfg=model_cfg)

    def _parse_generation_config(self, tokenizer, model):
        from modelscope.utils.hf_util import GenerationConfig

        generation_config = getattr(model, 'generation_config', GenerationConfig())

        try:
            remote_config = GenerationConfig.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                trust_remote_code=True)
            generation_config.update(**remote_config.to_dict())
        except:
            logger.warning(f'Failed to get generation config of {self.model_id} from model hub, use default.')

        # Parse templates for chat-completion
        if isinstance(self.model_id, str) and os.path.exists(self.model_id):
            logger.warning(f'Got local model dir: {self.model_id}')

        generation_template = get_template(template_type=self.template_type, tokenizer=tokenizer)

        return generation_config, generation_template

    def _model_generate(self, query: str, infer_cfg: dict) -> str:
        example = dict(query=query,
                       history=[],
                       system=None)

        inputs = self.generation_template.encode(example)
        input_ids = inputs['input_ids']
        input_ids = torch.tensor(input_ids)[None].to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)

        # Process infer_cfg
        infer_cfg = infer_cfg or {}
        if isinstance(infer_cfg.get('num_return_sequences'), int) and infer_cfg['num_return_sequences'] > 1:
            infer_cfg['do_sample'] = True

        # TODO: stop settings
        stop = infer_cfg.get('stop', None)
        eos_token_id = self.tokenizer.encode(stop, add_special_tokens=False)[0] \
            if stop else self.tokenizer.eos_token_id

        if eos_token_id is not None:
            infer_cfg['eos_token_id'] = eos_token_id
            infer_cfg['pad_token_id'] = eos_token_id  # setting eos_token_id as pad token


        self.generation_config.update(**infer_cfg)

        # stopping
        stop_words = [self.generation_template.suffix[-1]]
        decode_kwargs = {}
        stopping_criteria = StoppingCriteriaList(
            [StopWordsCriteria(self.tokenizer, stop_words, **decode_kwargs)])

        # Run inference
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            stopping_criteria=stopping_criteria, )

        response = self.tokenizer.decode(output_ids[0, len(input_ids[0]):], True, **decode_kwargs)
        return response

    @torch.no_grad()
    def predict(self, inputs: Union[str, dict, list], infer_cfg: dict = dict({})) -> dict:

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

        choices_list = [
            {'index': 0,
             'message': {'content': response,
                         'role': 'assistant'}
             }
        ]

        res_d = {
            'choices': choices_list,
            'created': time.time(),
            'model': self.model_id,
            'object': 'chat.completion',
            'usage': {}
        }

        return res_d


if __name__ == '__main__':

    if sys.argv and len(sys.argv) > 1:
        model_id_or_path = sys.argv[1]
        model_revision = sys.argv[2]
        query = sys.argv[3]
        infer_cfg = sys.argv[4]
    else:
        # model_id_or_path = '/to/path/.cache/modelscope/ZhipuAI/chatglm3-6b'
        model_id_or_path = 'ZhipuAI/chatglm3-6b-base'
        model_revision = 'v1.0.1'
        query = 'Question:俄罗斯的首都是哪里？ \n\nAnswer:'
        infer_cfg = None

    model_adapter = ChatGenerationModelAdapter(model_id=model_id_or_path, model_revision=model_revision)

    res_d = model_adapter.predict(inputs={'data': [query]}, infer_cfg=infer_cfg)
    print(res_d)
