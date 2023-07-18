# Copyright (c) Alibaba, Inc. and its affiliates.

from transformers import AutoModel, AutoTokenizer

from llmuses.constants import EvalTaskConfig, PredictorMode
from llmuses.predictors.base import Predictor
from llmuses.utils.logger import get_logger

logger = get_logger()


class ChatGLMPredictor(Predictor):
    """
    ChatGLM models predictor, including ChatGLM-6B, ChatGLM2-6B, ChatGLM-130B...
    """

    def __init__(self, api_key: str, mode=PredictorMode.LOCAL, **kwargs):
        super(ChatGLMPredictor, self).__init__(
            api_key=api_key, mode=mode, **kwargs)

        self.model_name = kwargs.pop(EvalTaskConfig.ARGS_MODEL, '')
        if not self.model_name:
            raise ValueError(
                'Model name of predictor is not specified. '
                'Please set it in the task configuration or pass it to the constructor.'
            )

        if mode == PredictorMode.REMOTE:
            raise ValueError(
                'Remote inference is not supported for ChatGLM Predicator, use OpenAI API predictor instead.'
            )

        self.max_length = int(kwargs.pop(EvalTaskConfig.ARGS_MAX_LEN, 2048))
        self.temperature = float(
            kwargs.pop(EvalTaskConfig.ARGS_TEMPERATURE, 0.8))
        self.top_p = float(kwargs.pop(EvalTaskConfig.ARGS_TOP_P, 0.8))

    def _init_local_model(self, **kwargs):
        model_path = kwargs.get(EvalTaskConfig.ARGS_MODEL_PATH,
                                'THUDM/chatglm2-6b')
        revision = kwargs.get('revision', 'main')
        num_gpus = kwargs.get(EvalTaskConfig.ARGS_NUM_GPUS, 1)
        quantize_bit = kwargs.get(EvalTaskConfig.ARGS_QUANTIZE_BIT, None)

        print(f'Loading model from {model_path}...')

        if num_gpus > 1:
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                num_gpus=num_gpus,
                revision=revision)
        elif quantize_bit is not None:
            model = (
                AutoModel.from_pretrained(
                    model_path, trust_remote_code=True,
                    revision=revision).quantize(int(quantize_bit)).cuda())
        else:
            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, revision=revision).cuda()

        self.model = model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, revision=revision)

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

        model_info = {
            EvalTaskConfig.ARGS_MODEL: self.model_name,
            EvalTaskConfig.ARGS_MAX_LEN: self.max_length,
            EvalTaskConfig.ARGS_TOP_P: self.top_p,
            EvalTaskConfig.ARGS_TEMPERATURE: self.temperature,
        }

        input_kwargs.update(model_info)
        if self.mode == PredictorMode.LOCAL:
            result = self._run_local_inference(**input_kwargs)
        else:
            raise ValueError(f'Invalid predictor mode: {self.mode}')

        final_res = dict(
            **input_kwargs, gen=['' if result is None else result['response']])
        print(
            f"""{final_res["idx"]}. Prompt: {final_res["prompt"]}, Answer: {final_res["gen"][0]}"""
        )
        return final_res

    def _run_local_inference(self, **kwargs):
        """
        Note: GPT predictor does not support local inference, no need to implement this method.
        """
        try:
            prompt = kwargs['prompt']
            params = {
                'history': kwargs.get('history', []),
                'max_length': kwargs.get('max_length'),
                'top_p': kwargs.get('top_p'),
            }
            if 'temperature' in kwargs:
                if kwargs['temperature'] < 1e-5:
                    params['do_sample'] = False
                else:
                    params['temperature'] = kwargs['temperature']

            response, _ = self.model.chat(self.tokenizer, prompt, **params)
            response = response.strip()
        except Exception as e:
            # raise e
            print(e)
            return None

        return {'response': response}

    def _run_remote_inference(self, **kwargs):
        logger.error(
            'ChatGLM predictor does not support remote inference, use OpenAI API predictor instead.'
        )
