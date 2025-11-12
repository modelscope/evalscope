import random
from PIL import Image, ImageDraw
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.random_dataset import RandomDatasetPlugin
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.io_utils import PIL_to_base64


@register_dataset('random_vl')
class RandomVLDatasetPlugin(RandomDatasetPlugin):
    """Random Vision-Language Dataset Plugin for multimodal model stress testing."""

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

        # Vision-language specific parameters
        self.image_width = query_parameters.image_width
        self.image_height = query_parameters.image_height
        self.image_format = query_parameters.image_format
        self.image_num = query_parameters.image_num

        assert self.image_num > 0, 'image_num must be greater than 0.'

    def build_messages(self) -> Iterator[List[Dict]]:
        # Reuse parent's message generation logic
        for messages in super().build_messages():
            prompt = messages[0]['content'] if isinstance(messages[0], dict) else messages[0]

            # Generate random images based on image_num
            images_b64 = []
            for _ in range(self.image_num):
                images_b64.append(self._generate_random_image_b64())

            message = self.create_message(text=prompt, image_urls=images_b64)
            yield [message]

    def _generate_random_image_b64(self) -> str:
        """Generate a random image and return as base64 string."""
        # Create a random colored image
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = Image.new(self.image_format, (self.image_width, self.image_height), color)

        # Add some random shapes for variety
        draw = ImageDraw.Draw(image)
        for _ in range(random.randint(1, 5)):
            shape_type = random.choice(['rectangle', 'ellipse', 'line'])

            # Generate two random points
            x1 = random.randint(0, self.image_width - 1)
            y1 = random.randint(0, self.image_height - 1)
            x2 = random.randint(0, self.image_width - 1)
            y2 = random.randint(0, self.image_height - 1)

            # Ensure proper coordinate ordering (x1 <= x2, y1 <= y2)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # Ensure we have at least a 1-pixel difference
            if x1 == x2:
                x2 = min(x1 + 1, self.image_width - 1)
            if y1 == y2:
                y2 = min(y1 + 1, self.image_height - 1)

            coords = [x1, y1, x2, y2]

            shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            if shape_type == 'rectangle':
                draw.rectangle(coords, fill=shape_color)
            elif shape_type == 'ellipse':
                draw.ellipse(coords, fill=shape_color)
            else:
                draw.line(coords, fill=shape_color, width=random.randint(1, 5))

        # Convert to base64
        return PIL_to_base64(image, format='PNG', add_header=True)
