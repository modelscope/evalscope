import os
import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from PIL import Image
from transformers import AutoModel, AutoProcessor
from typing import List, Union

from evalscope.backend.rag_eval.utils.tools import PIL_to_base64, download_model
from evalscope.constants import HubType


class VisionModel:

    @staticmethod
    def load(**kw):
        api_base = kw.get('api_base', None)
        if api_base:

            return VLMAPI(
                model_name=kw.get('model_name', ''),
                openai_api_base=api_base,
                openai_api_key=kw.get('api_key', 'EMPTY'),
                prompt=kw.get('prompt', None),
            )
        else:
            return CLIPModel(**kw)


class VLMAPI:

    def __init__(self, model_name, openai_api_base, openai_api_key, prompt=None):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        self.model_name = model_name
        self.model = ChatOpenAI(
            model_name=model_name,
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
        )
        self.default_prompt = "Please describe this image in general. Directly provide the description, do not include prefix like 'This image depicts'"  # noqa: E501
        self.prompt = ChatPromptTemplate.from_messages([
            ('system', prompt if prompt else self.default_prompt),
            (
                'user',
                [{
                    'type': 'image_url',
                    'image_url': {
                        'url': 'data:image/jpeg;base64,{image_data}'
                    },
                }],
            ),
        ])
        self.chain = self.prompt | self.model
        self.transform = PIL_to_base64

    def encode_image(self, images):
        captions = []
        for image in images:
            response = self.chain.invoke({'image_data': image})
            captions.append(response.content)
        return captions


class CLIPModel(Embeddings):

    def __init__(
        self,
        model_name: str,
        revision: str = 'master',
        hub=HubType.MODELSCOPE,
        device='cpu',
    ):
        self.device = device
        self.model_name = model_name
        self.revision = revision

        # Download the model if it doesn't exist locally
        if not os.path.exists(model_name) and hub == HubType.MODELSCOPE:
            model_name = download_model(self.model_name, self.revision)

        # Load the model and processor
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.transform = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer

    def encode_text(self, batch_texts: Union[List[str], List[List[str]]]):
        if isinstance(batch_texts[0], list):
            batch_texts = [text for _, texts in enumerate(batch_texts) for text in texts]
        # Ensure that the input texts are within the token limit
        max_length = self.tokenizer.model_max_length
        if not max_length or max_length > 0xFFFFFF:
            max_length = 512
        encoded_inputs = self.tokenizer(
            text=batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    def encode_image(self, image):
        batch_images = torch.stack([d['pixel_values'][0] for d in image])
        batch_images = batch_images.to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(batch_images)
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    def embed_documents(self, texts):
        text_features = self.encode_text(texts)
        return text_features.cpu().numpy().tolist()

    def embed_query(self, text):
        text_features = self.encode_text([text])
        return text_features.cpu().numpy().tolist()[0]

    def embed_image(self, uris: List[str]):
        # read image and transform
        images = [Image.open(image_path) for image_path in uris]
        transformed_images = [self.transform(
            image,
            return_tensors='pt',
        ) for image in images]
        image_features = self.encode_image(transformed_images)
        return image_features.cpu().numpy().tolist()


if __name__ == '__main__':
    model = CLIPModel('AI-ModelScope/chinese-clip-vit-large-patch14-336px')
    model.embed_image([
        'custom_eval/multimodal/images/AMNH.jpg',
        'custom_eval/multimodal/images/AMNH.jpg',
    ])
    model.encode_text(['我喜欢吃饭' * 1000])
    print('done')
