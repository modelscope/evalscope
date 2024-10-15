import os
import torch
from evalscope.backend.rag_eval.utils.tools import download_model
from transformers import AutoModel, AutoProcessor
from evalscope.backend.rag_eval.utils.tools import PIL_to_base64


class VisionModel:
    @staticmethod
    def load(**kw):
        api_base = kw.get("api_base", None)
        if api_base:

            return VLMAPI(
                model_name=kw.get("model_name", ""),
                openai_api_base=api_base,
                openai_api_key=kw.get("api_key", "EMPTY"),
                prompt=kw.get("prompt", None),
            )
        else:
            return CLIPModel(**kw)


class VLMAPI:
    def __init__(self, model_name, openai_api_base, openai_api_key, prompt=None):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(
            base_url=openai_api_base,
            api_key=openai_api_key,
        )
        self.default_prompt = "Please describe this image in general. Directly provide the description, do not include prefix like 'This image depicts'"
        self.prompt = prompt if prompt else self.default_prompt

        self.transform = PIL_to_base64

    def encode_image(self, images):
        captions = []
        for image in images:
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                    ],
                }
            ]
            result = (
                self.client.chat.completions.create(
                    messages=message, model=self.model_name, temperature=0
                )
                .choices[0]
                .message.content
            )
            captions.append(result)
        return captions


class CLIPModel:
    def __init__(
        self,
        model_name: str,
        revision: str = "master",
        hub="modelscope",
        device="cpu",
    ):
        self.device = device
        self.model_name = model_name
        self.revision = revision
        if not os.path.exists(model_name) and hub == "modelscope":
            model_name = download_model(self.model_name, self.revision)

        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.transform = self.processor.image_processor

    def encode_text(self, batch_texts):
        text_list = [text for i, texts in enumerate(batch_texts) for text in texts]
        inputs = self.processor(text=text_list, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features

    def encode_image(self, image):
        batch_images = torch.stack([d["pixel_values"][0] for d in image])
        batch_images = batch_images.to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(batch_images)
        return image_features


if __name__ == "__main__":
    model = CLIPModel("AI-ModelScope/clip-vit-large-patch14-336")
    print("done")
