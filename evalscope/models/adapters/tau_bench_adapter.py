import json
import time
from typing import Any, Dict, List, Optional, Union

from evalscope.utils.logger import get_logger
from ..register import register_model_adapter
from .server_adapter import ServerModelAdapter

logger = get_logger()


@register_model_adapter(name='tau_bench_server')
class TauBenchAdapter(ServerModelAdapter):
    """
    TauBench model adapter to request remote API model and generate results for TauBench evaluation.
    Support multi-turn and single-turn function calling tasks.
    """

    def __init__(self, api_url: str, model_id: str, api_key: str = 'EMPTY', **kwargs):
        """
        Args:
            api_url: The URL of the remote API model.
            model_id: The ID of the remote API model.
            api_key: The API key of the remote API model.
        """
        super().__init__(api_url=api_url, model_id=model_id, api_key=api_key, **kwargs)

        self._patch_agent_solve()

    def predict(self, inputs: List[dict], infer_cfg: Optional[dict] = None) -> List[dict]:
        """
        Model prediction func. For multi-turn evals, we pass a list[list[message]] to the model
        where each list is a follow up turn in the conversation
        each turn is a List[List[Message]]

        Args:
            inputs (List[dict]): The input data.
            infer_cfg (dict): Inference configuration.

        Returns:
            res (List[dict]): The model prediction results.
        """
        infer_cfg = infer_cfg or {}
        results = []

        for input_item in inputs:
            raw_input = input_item.get('raw_input')

            res_d = self.solve(env_name=raw_input['env_name'], task_index=raw_input['task_index'], infer_cfg=infer_cfg)

            wrapper_res = {
                'choices': [{
                    'index': 0,
                    'message': {
                        'content': json.dumps(res_d, ensure_ascii=False),
                        'role': 'assistant'
                    }
                }],
                'created':
                time.time(),
                'model':
                self.model_id,
                'object':
                'chat.completion',
                'usage': {
                    'completion_tokens': 0,
                    'prompt_tokens': 0,
                    'total_tokens': 0
                }
            }

            results.append(wrapper_res)

        return results

    def _patch_agent_solve(self):
        """Patch ToolCallingAgent.solve method to use custom model configuration"""
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent, message_to_action
        from tau_bench.envs.base import Env
        from tau_bench.types import RESPOND_ACTION_NAME, SolveResult
        from typing import List, Optional

        def patched_solve(self,
                          env: Env,
                          task_index: Optional[int] = None,
                          max_num_steps: int = 30,
                          infer_cfg: Optional[dict] = {}) -> SolveResult:
            env_reset_res = env.reset(task_index=task_index)
            obs = env_reset_res.observation
            info = env_reset_res.info.model_dump()
            reward = 0.0
            messages: List[Dict[str, Any]] = [
                {
                    'role': 'system',
                    'content': self.wiki
                },
                {
                    'role': 'user',
                    'content': obs
                },
            ]

            for step_index in range(max_num_steps):
                # Use adapter's model configuration instead of agent's
                request_json = adapter_instance.make_request(
                    input_item={
                        'messages': messages,
                        'tools': self.tools_info
                    }, infer_cfg=infer_cfg)
                res = adapter_instance.send_request(request_json)

                next_message = res['choices'][0]['message']
                action = message_to_action(next_message)
                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

                if action.name != RESPOND_ACTION_NAME:
                    next_message['tool_calls'] = next_message['tool_calls'][:1]
                    messages.extend([
                        next_message,
                        {
                            'role': 'tool',
                            'tool_call_id': next_message['tool_calls'][0]['id'],
                            'name': next_message['tool_calls'][0]['function']['name'],
                            'content': env_response.observation,
                        },
                    ])
                else:
                    messages.extend([
                        next_message,
                        {
                            'role': 'user',
                            'content': env_response.observation
                        },
                    ])
                logger.debug(f'Task: {task_index} Step: {step_index} finished')

                if env_response.done:
                    break

            return SolveResult(
                reward=reward,
                info=info,
                messages=messages,
                total_cost=0,
            )

        adapter_instance = self

        ToolCallingAgent.solve = patched_solve

        return 'ToolCallingAgent.solve patched successfully'

    def solve(self, env_name, task_index, infer_cfg, **kwargs):
        """
        Solve a specific task in the TauBench environment.

        Args:
            env_name (str): The name of the TauBench environment.
            task_index (int): The index of the task to solve.
            **kwargs: Additional arguments for the task.

        Returns:
            dict: The result of the task.
        """
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent
        from tau_bench.envs import get_env

        # This method can be implemented to solve specific tasks in the TauBench environment
        isolated_env = get_env(
            env_name=env_name,
            user_strategy='llm',
            user_model='dummy',  # Use dummy model to prevent errors
            user_provider='openai',  # Use dummy provider to prevent errors
            task_split='test',
            task_index=task_index,
        )
        agent = ToolCallingAgent(
            tools_info=isolated_env.tools_info,
            wiki=isolated_env.wiki,
            model='dummy',  # Use dummy model to prevent errors
            provider='dummy',  # Use dummy provider to prevent errors
            temperature=0,  # dummy temperature to prevent errors
        )

        res = agent.solve(env=isolated_env, task_index=task_index, infer_cfg=infer_cfg)

        return res.model_dump()
