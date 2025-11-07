from typing import Any, Dict, List, Optional

from evalscope.api.dataset import Sample
from evalscope.api.messages import dict_to_chat_message
from evalscope.api.model import Model, ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.tool import ToolInfo
from evalscope.models.utils.openai import openai_chat_choices
from evalscope.utils.function_utils import run_once
from evalscope.utils.logger import get_logger

logger = get_logger()


@run_once
def _patch_agent_solve(model: Model):
    """Patch ToolCallingAgent.solve method to use custom model configuration"""
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent, message_to_action
    from tau_bench.envs.base import Env
    from tau_bench.types import RESPOND_ACTION_NAME, Action, SolveResult

    def patched_solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
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
            res = model.generate(
                input=[dict_to_chat_message(msg) for msg in messages],
                tools=[ToolInfo.model_validate(tool['function']) for tool in self.tools_info]
            )
            oai_res = openai_chat_choices(res.choices, include_reasoning=False)

            next_message = oai_res[0].message.model_dump(exclude_none=True)

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

    ToolCallingAgent.solve = patched_solve

    return 'ToolCallingAgent.solve patched successfully'


def predict(model: Model, sample: Sample) -> ModelOutput:
    """
    Generate predictions for tau_bench tasks using the model.

    Args:
        model: The model to use for prediction
        sample: The sample containing task metadata

    Returns:
        ModelOutput containing the prediction results
    """
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent
    from tau_bench.envs import get_env

    _patch_agent_solve(model)
    try:
        # Extract task information from sample metadata
        task_data = sample.metadata
        env_name = task_data['env_name']
        task_index = task_data['task_index']

        # Direct call to tau_bench_server adapter's solve method
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

        res = agent.solve(env=isolated_env, task_index=task_index)

        sample.metadata['task_result'] = res.model_dump(exclude_none=True)
        return ModelOutput(
            model=model.name,
            choices=[ChatCompletionChoice.from_content(res.model_dump_json(indent=2))],
        )

    except Exception as e:
        logger.error(f'Error in tau_bench prediction: {str(e)}')
        sample.metadata['task_result'] = {'reward': 0, 'error': str(e)}
        return ModelOutput(
            model=model.name,
            choices=[ChatCompletionChoice.from_content('')],
        )
