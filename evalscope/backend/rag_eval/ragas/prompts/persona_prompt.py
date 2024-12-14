import typing as t
from pydantic import BaseModel
from ragas.prompt import PydanticPrompt, StringIO
from ragas.testset.persona import Persona


class PersonaGenerationPromptZH(PydanticPrompt[StringIO, Persona]):
    instruction: str = ('使用提供的摘要，生成一个可能会与内容互动或从中受益的角色。包括一个独特的名字和一个简洁的角色描述。')
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[Persona] = Persona
    examples: t.List[t.Tuple[StringIO, Persona]] = [(
        StringIO(text='《数字营销指南》解释了在各种在线平台上吸引受众的策略。'),
        Persona(
            name='数字营销专家',
            role_description='专注于吸引受众并在线上提升品牌。',
        ),
    )]
