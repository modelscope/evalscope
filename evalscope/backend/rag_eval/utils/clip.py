import os
import torch
from PIL import Image
from evalscope.backend.rag_eval.utils.tools import download_model
from transformers import AutoModel, AutoProcessor


class CLIPModel:
    def __init__(
        self,
        model_name_or_path: str,
        revision: str = "master",
        hub="modelscope",
        device="cpu",
    ):
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        if not os.path.exists(model_name_or_path) and hub == "modelscope":
            model_name_or_path = download_model(self.model_name_or_path, self.revision)

        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

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
