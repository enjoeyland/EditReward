import os
from typing import List, Optional

import openai

from .model_utils.openai_utils import local_image_to_data_url


class Qwen3_5:
    support_multi_image = True
    merged_image_files = []

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize an OpenAI-compatible Qwen3.5 multimodal client.

        Environment variables:
            OPENAI_BASE_URL / OPENAI_API_BASE
            OPENAI_API_KEY
            QWEN3_5_MODEL_NAME / OPENAI_MODEL_NAME
        """
        self.model_path = (
            model_path
            or os.environ.get("QWEN3_5_MODEL_NAME")
            or os.environ.get("OPENAI_MODEL_NAME")
            or "Qwen/Qwen3.5-27B"
        )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    def __call__(self, inputs: List[dict]) -> str:
        if not self.support_multi_image:
            raise NotImplementedError

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an image-editing judge. "
                            "Carefully compare the source image, candidate images, "
                            "and editing instruction, then follow the requested "
                            "output format exactly."
                        ),
                    }
                ],
            },
            {"role": "user", "content": []},
        ]

        for message in inputs:
            message_type = message["type"]
            if message_type == "image":
                image_data_url = local_image_to_data_url(message["content"])
                messages[-1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    }
                )
            elif message_type == "text":
                messages[-1]["content"].append(
                    {
                        "type": "text",
                        "text": message["content"],
                    }
                )
            else:
                raise NotImplementedError(f"Unsupported input type: {message_type}")

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        response = client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4000,
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(part for part in parts if part)
        return str(content)
