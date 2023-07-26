# Copyright (c) Alibaba, Inc. and its affiliates.

from llmuses.constants import EvalTaskConfig, PredictorMode
from llmuses.predictors.base import Predictor
from llmuses.predictors.model_adapter import load_model
from llmuses.utils.logger import get_logger

logger = get_logger()


class LocalInferencePredictor(Predictor):
    """
    Local inference predictor, including ChatGLM2-6B, Baichuan-13B-Chat, InternLM-7B...
    """

    def __init__(self, api_key: str, mode=PredictorMode.LOCAL, **kwargs):
        super(LocalInferencePredictor, self).__init__(
            api_key=api_key, mode=mode, **kwargs)

        self.model_name = kwargs.pop(EvalTaskConfig.ARGS_MODEL, '')
        if not self.model_name:
            raise ValueError(
                'Model name of predictor is not specified. '
                'Please set it in the task configuration or pass it to the constructor.'
            )
        elif self.mode != PredictorMode.LOCAL:
            raise ValueError(f'Invalid predictor mode: {self.mode}')

        self.inference_kwargs = kwargs

    def _init_local_model(self, **kwargs):
        model_path = kwargs.get(EvalTaskConfig.ARGS_MODEL_PATH)
        if not model_path:
            raise ValueError(
                'Model path is required for local inference predictor.')

        self.model_adapter = load_model(**kwargs)

    def predict(self, **input_kwargs) -> dict:
        """
        Run inference with the given input arguments.

        :param input_kwargs: input arguments for inference. Cols of kwargs:
            prompt: prompt text
            history: list of dict, each dict contains user and bot utterances

            Example:
                input_args = dict(
                    prompt='推荐一个附近的公园',
                    history=[
                        {
                            "user": "今天天气好吗？",
                            "bot": "今天天气不错，要出去玩玩嘛？"
                        },
                        {
                            "user": "那你有什么地方推荐？",
                            "bot": "我建议你去公园，春天来了，花朵开了，很美丽。"
                        }
                    ],
                )

        :return: dict, output of inference.
            Example: {'input': {}, 'output': {}}
        """

        input_kwargs.update(self.inference_kwargs)
        result = self._run_local_inference(**input_kwargs)

        final_res = dict(
            **input_kwargs, gen=['' if result is None else result['response']])
        print(
            f"""{final_res["idx"]}. Prompt: {final_res["prompt"]}, Answer: {final_res["gen"][0]}"""
        )
        return final_res

    def _run_local_inference(self, **kwargs):
        try:
            prompt = kwargs.get('prompt')
            history = kwargs.get('history', [])
            history = [(h['user'], h['bot']) for h in history]
            max_length = kwargs.get(EvalTaskConfig.ARGS_MAX_LEN, None)
            max_new_tokens = kwargs.get(EvalTaskConfig.ARGS_MAX_NEW_TOKENS,
                                        None)
            temperature = kwargs.get(EvalTaskConfig.ARGS_TEMPERATURE, None)
            top_p = kwargs.get(EvalTaskConfig.ARGS_TOP_P, None)

            params = {}
            # chatglm-6b/chatglm2-6b use max_length, not max_new_tokens
            if max_length is not None:
                params['max_length'] = max_length
            if max_new_tokens is not None:
                params['max_new_tokens'] = max_new_tokens
            if temperature is not None:
                if temperature < 1e-5:
                    params['do_sample'] = False
                else:
                    params['temperature'] = temperature
            if top_p is not None:
                params['top_p'] = top_p

            response, history = self.model_adapter.chat(
                prompt, history=history, **params)
            response = response.strip()
        except Exception as e:
            # raise e
            print(e)
            return None

        return {'response': response}

    def _run_remote_inference(self, **kwargs):
        logger.error(
            'Local inference predictor does not support remote inference.')
