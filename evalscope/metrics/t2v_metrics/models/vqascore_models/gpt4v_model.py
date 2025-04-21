import base64
import os
import tiktoken
import torch
from openai import OpenAI
from typing import List

from .vqa_model import VQAScoreModel

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = 'Yes'

GPT4V_MODELS = {
    # We recommend using 'gpt-4-turbo' for optimal performance.
    'gpt-4-turbo': {},
    'gpt-4o': {},
}


# Function to encode the image
def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_type(image_path):
    image_type = image_path.split('.')[-1]
    assert image_type in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']
    return image_type


class GPT4VModel(VQAScoreModel):
    """A wrapper for the GPT4V models"""

    def __init__(self, model_name='gpt-4-turbo', device='cuda', cache_dir=None, openai_key=None, top_logprobs=2):
        assert model_name in GPT4V_MODELS
        assert openai_key is not None, 'Please provide an OpenAI API key'
        self.openai_key = openai_key
        self.top_logprobs = top_logprobs
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.client = OpenAI(api_key=self.openai_key)
        # self.candidate_answers = GPT4V_MODELS[self.model_name]['candidate_answers']
        # assert GPT4V_MODELS[self.model_name]['answer'] in self.candidate_answers
        # self.candidate_tokens = []
        # for ans in self.candidate_answers:
        #     token = self.tokenizer.encode(ans)
        #     assert len(token) == 1, "Currently only support single token answers"
        #     self.candidate_tokens.append(token[0])

    def load_images(self, image: List[str]) -> torch.Tensor:
        """Load the image(s), and return the string
        """
        image = [{'path': img, 'type': get_image_type(img), 'base64': encode_image(img)} for img in image]
        return image

    def forward_single(self, image, question, answer):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    'role':
                    'user',
                    'content': [{
                        'type': 'text',
                        'text': question
                    }, {
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:image/{image['type']};base64,{image['base64']}"
                        }
                    }]
                }],
                logprobs=True,
                top_logprobs=self.top_logprobs,
                # logit_bias={yes_token:50, no_token:50}
            )
        except:
            print(
                f"Warning: completion not generated for image: {image['path']} and question: {question} and answer: {answer}"
            )
            print(f'Trying again with the same image')
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        'role':
                        'user',
                        'content': [{
                            'type': 'text',
                            'text': question
                        }, {
                            'type': 'image_url',
                            'image_url': {
                                'url': f"data:image/{image['type']};base64,{image['base64']}"
                            }
                        }]
                    }],
                    logprobs=True,
                    top_logprobs=self.top_logprobs,
                )
            except:
                print(f"Failed image: {image['path']} and question: {question} and answer: {answer}")
                return torch.Tensor([0.0])

        # print(completion.choices[0].message)
        # print(completion.choices[0].logprobs)
        # print(completion.choices[0].logprobs.content[0])
        is_generated = False
        for top_logprob in completion.choices[0].logprobs.content[0].top_logprobs:
            if top_logprob.token == answer:
                is_generated = True
                return torch.Tensor([top_logprob.logprob]).exp()
        if not is_generated:
            print(
                f"Warning: answer not generated for image: {image['path']} and question: {question} and answer: {answer}"
            )
            print(completion.choices[0].logprobs.content[0].top_logprobs)
            return torch.Tensor([0.0])

    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str = default_question_template,
                answer_template: str = default_answer_template) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), 'Number of images and texts must match'
        # Turn "a photo of a dog" into
        # Q: "Does this figure show "a photo of a dog"? Please answer yes or no."
        # A: "Yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        for ans in answers:
            ans_tokens = self.tokenizer.encode(ans)
            assert len(ans_tokens) == 1, 'Currently only support single token answers'

        images = self.load_images(images)

        lm_prob = torch.zeros(len(images))

        for idx, (image, question, answer) in enumerate(zip(images, questions, answers)):
            lm_prob[idx] = self.forward_single(image, question, answer)

        return lm_prob
