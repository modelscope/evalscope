import asyncio
import os
from ragas.llms import BaseRagasLLM
from ragas.prompt import PromptMixin, PydanticPrompt
from typing import List

from evalscope.utils.logger import get_logger

logger = get_logger()


async def translate_prompt(
    prompt_user: PromptMixin,
    target_lang: str,
    llm: BaseRagasLLM,
    adapt_instruction: bool = False,
):
    if not issubclass(type(prompt_user), PromptMixin):
        logger.info(f"{prompt_user} is not a PromptMixin, don't translate it")
        return

    class_name = prompt_user.__class__.__name__
    current_dir = os.path.dirname(__file__)
    prompt_dir = os.path.abspath(os.path.join(current_dir, f'../prompts/{target_lang}/{class_name}'))
    os.makedirs(prompt_dir, exist_ok=True)

    try:
        loader_prompts = prompt_user.load_prompts(prompt_dir, target_lang)
        prompt_user.set_prompts(**loader_prompts)
        logger.info(f'Load existing prompts from {prompt_dir}')
        return
    except FileNotFoundError:
        logger.info(f'Not find existing prompts {class_name}, generate new prompts.')

    logger.info(f'Translating prompts to {target_lang}')
    adapted_prompts = await prompt_user.adapt_prompts(
        language=target_lang, llm=llm, adapt_instruction=adapt_instruction)
    prompt_user.set_prompts(**adapted_prompts)
    try:
        prompt_user.save_prompts(prompt_dir)
    except FileExistsError:
        logger.info(f'Find existing prompt {class_name}, skip saving.')
    logger.info(f'Save new prompts to {prompt_dir}')

    return


async def translate_prompts(
    prompts: List[PromptMixin],
    target_lang: str,
    llm: BaseRagasLLM,
    adapt_instruction: bool = False,
):
    if target_lang and target_lang != 'english':
        await asyncio.gather(*[translate_prompt(prompt, target_lang, llm, adapt_instruction) for prompt in prompts])

        logger.info('Translate prompts finished')
