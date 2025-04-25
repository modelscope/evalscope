import numpy as np
import time
import torch
from typing import List

from evalscope.utils.chat_service import ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage
from ..local_model import LocalModel
from .base_adapter import BaseModelAdapter


class MultiChoiceModelAdapter(BaseModelAdapter):
    """ The multi-choice model adapter. """

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(self, model: LocalModel, **kwargs):
        super().__init__(model)

        self._max_length = kwargs.get('max_length')

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
    def predict(self, inputs: List[dict], infer_cfg: dict = None) -> dict:
        """
        Multi-choice model prediction func.

        Args:
            inputs (List[dict]): The inputs for a doc. Format:
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

        input_data = [inp['data'][0] for inp in inputs]
        multi_choices = [inp['multi_choices'] for inp in inputs]

        outputs, input_info = self._get_logits(self.tokenizer, self.model, input_data)

        results = []
        for i, (logits, choices) in enumerate(zip(outputs, multi_choices)):
            choice_logits = [logits[self.tokenizer(ch)['input_ids'][-1:]] for ch in choices]
            softval = torch.nn.functional.softmax(torch.tensor(choice_logits).float(), dim=0)

            if softval.dtype in {torch.bfloat16, torch.float16}:
                softval = softval.to(dtype=torch.float32)
            probs = softval.detach().cpu().numpy()
            pred: str = choices[int(np.argmax(probs))]  # Format: A or B or C or D

            res_d = ChatCompletionResponse(
                model=self.model_id,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0, message=ChatMessage(content=pred, role='assistant'), finish_reason='stop')
                ],
                object='chat.completion',
                created=int(time.time()),
                usage=None).model_dump(exclude_unset=True)

            results.append(res_d)

        return results

    @staticmethod
    def _get_logits(tokenizer, model, inputs: List[str]):
        input_ids = tokenizer(
            inputs, padding=True, return_tensors='pt', padding_side='left')['input_ids'].to(model.device)
        tokens = {'input_ids': input_ids}

        outputs = model(input_ids)['logits']
        logits = outputs[:, -1, :]
        log_probs = torch.nn.functional.softmax(logits, dim=-1)
        return log_probs, {'tokens': tokens}


class ContinuationLogitsModelAdapter(MultiChoiceModelAdapter):
    """
    Continuation-logits model adapter.
    """

    def __init__(self, model: LocalModel, **kwargs):
        super().__init__(model, **kwargs)

    @torch.no_grad()
    def predict(self, inputs: List[dict], infer_cfg: dict = None) -> dict:
        """
        Multi-choice model prediction func.
        Args:
            inputs (List[dict]): The inputs for a doc. Format:
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

        pred_list: list = []
        for inp in inputs:
            pred_list.append(self.loglikelihood(inputs=inp['data'], infer_cfg=infer_cfg))

        results = []
        for pred in pred_list:
            res_d = ChatCompletionResponse(
                model=self.model_id,
                choices=[{
                    'index': 0,
                    'message': {
                        'content': pred,
                        'role': 'assistant'
                    }
                }],
                object='chat.completion',
                created=int(time.time()),
                usage=None).model_dump(exclude_unset=True)
            results.append(res_d)

        return results

    def loglikelihood(self, inputs: List[tuple], infer_cfg: dict = None) -> list:
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
