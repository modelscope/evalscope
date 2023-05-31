# Copyright (c) Alibaba, Inc. and its affiliates.

from evals.constants import PredictorMode, EvalTaskConfig
from evals.predictors.base import Predictor
import requests
import json
from transformers import AutoTokenizer, AutoModel

DEFAULT_MAX_LEN = 2048
DEFAULT_TOP_P = 0.7
DEFAULT_TEMPERATURE = 0.95

class GlmPredictor(Predictor):
    """
    ChatGLM models predictor, including ChatGLM-130B, ChatGLM-6B, ...
    """

    def __init__(self, api_key: str, mode=PredictorMode.LOCAL, **kwargs):
        super(GlmPredictor, self).__init__(api_key=api_key, mode=mode, **kwargs)

        self.model_name = kwargs.pop(EvalTaskConfig.ARGS_MODEL, "")
        if not self.model_name:
            raise ValueError(
                "Model name of predictor is not specified. "
                "Please set it in the task configuration or pass it to the constructor."
            )

        if mode == PredictorMode.REMOTE:
            self.api_endpoint = kwargs.pop(EvalTaskConfig.ARGS_API_ENDPOINT, "")
            if not self.api_endpoint:
                raise ValueError(
                    "API endpoint of predictor is not specified. "
                    "Please set it in the task configuration or pass it to the constructor."
                )
        
        self.max_length = int(kwargs.pop(EvalTaskConfig.ARGS_MAX_LEN, DEFAULT_MAX_LEN))
        self.top_p = float(kwargs.pop(EvalTaskConfig.ARGS_TOP_P, DEFAULT_TOP_P))
        self.temperature = float(kwargs.pop(EvalTaskConfig.ARGS_TEMPERATURE, DEFAULT_TEMPERATURE))

    def _init_local_model(self, **kwargs):
        model_path = kwargs.get(EvalTaskConfig.ARGS_MODEL_PATH, "THUDM/chatglm-6b")
        revision = kwargs.get(EvalTaskConfig.ARGS_MODEL_REVISION, "v1.1.0")
        device = kwargs.get("device", 0)
        quantize_bit = kwargs.get(EvalTaskConfig.ARGS_QUANTIZE_BIT, None)

        print(f"Loading model from {model_path} revision {revision} quantize_bit {quantize_bit} on device {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision=revision
        )
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            # resume_download=True,
            revision=revision
        )
        if quantize_bit:
            model = model.quantize(int(quantize_bit))
        self.model = model.half().to(device)

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
        if self.mode == PredictorMode.REMOTE:
            result = self._run_remote_inference(**input_kwargs)
        elif self.mode == PredictorMode.LOCAL:
            result = self._run_local_inference(**input_kwargs)
        else:
            raise ValueError(f"Invalid predictor mode: {self.mode}")

        final_res = dict(**input_kwargs, gen=["" if result is None else result["response"]])
        print(
            f"""{final_res["idx"]}. Prompt: {final_res["prompt"]}, Answer: {final_res["gen"][0]}"""
        )
        return final_res

    def _run_local_inference(self, **kwargs):
        """
        Note: GPT predictor does not support local inference, no need to implement this method.
        """
        try:
            prompt = kwargs["prompt"]
            params = {
                "history": kwargs.get("history", []),
                "max_length": kwargs.get("max_length"),
                "top_p": kwargs.get("top_p"),
            }
            if "temperature" in kwargs:
                if kwargs["temperature"] <= 0.0:
                    params["do_sample"] = False
                else:
                    params["temperature"] = kwargs["temperature"]

            response, _ = self.model.chat(
                self.tokenizer,
                kwargs["prompt"],
                **params
            )
            response = response.strip()
        except Exception as e:
            # raise e
            print(e)
            return None

        return {"response": response}

    def _run_remote_inference(self, **kwargs):
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            responses = requests.post(self.api_endpoint, headers=headers, data=json.dumps(kwargs))
            GlmPredictor._check_response_on_error(responses)
        except Exception as e:
            # raise e
            print(e)
            return None

        return json.loads(responses.text)
    
    @staticmethod
    def _check_response_on_error(resp):
        if resp.status_code != 200:
            raise ValueError(
                f"Failed to call remote inference service: errCode: {resp.code}, errMsg: {resp.message}"
            )


