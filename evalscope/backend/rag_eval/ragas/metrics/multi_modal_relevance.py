import typing as t
from dataclasses import dataclass, field
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
from pydantic import BaseModel, Field
from evalscope.backend.rag_eval.ragas.prompts.multi_modal_prompt import ImageTextPrompt


class RelevanceInput(BaseModel):
    user_input: str = Field(description="user input")
    response: str = Field(description="response from AI")
    retrieved_contexts: list[str] = Field(description="contexts retrieved from the LLM")

    def to_string_list(self):
        return [
            f"Question: {self.user_input}",
            f"Response: {self.response}" "retrieved_contexts: ",
        ] + self.retrieved_contexts


class RelevanceOutput(BaseModel):
    relevance: bool = Field(description="boolean indicating if request was relevance")


class MultiModalRelevancePrompt(ImageTextPrompt[RelevanceInput, RelevanceOutput]):
    instruction = """
Your task is to evaluate if the response for the query is in line with the images and textual context information provided.
You have two options to answer. Either True / False.
Answer - True, if the response for the query is in line with context information otherwise False.
"""
    input_model = RelevanceInput
    output_model = RelevanceOutput
    examples = [
        (
            RelevanceInput(
                user_input="What is the weather like today in the image?",
                response="The weather is sunny and clear.",
                retrieved_contexts=["Today's weather is sunny with clear skies."],
            ),
            RelevanceOutput(relevance=True),
        ),
        (
            RelevanceInput(
                user_input="Describe the main subject in the image.",
                response="A cat is sitting on a chair.",
                retrieved_contexts=[
                    "A dog is running on the grass.",
                ],
            ),
            RelevanceOutput(relevance=False),
        ),
    ]


@dataclass
class MultiModalRelevance(MetricWithLLM, SingleTurnMetric):
    name: str = "relevance_rate"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "retrieved_contexts",
            }
        }
    )
    relevance_prompt: ImageTextPrompt = MultiModalRelevancePrompt()

    async def _ascore(self, row):
        pass

    async def _single_turn_ascore(self, sample, callbacks):
        prompt_input = RelevanceInput(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )
        prompt_response = await self.relevance_prompt.generate(
            data=prompt_input, llm=self.llm
        )
        return prompt_response.relevance


multimodal_relevance = MultiModalRelevance()
