"""xAI Grok LLM service."""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import image, user

from constants.model_name import ModelName
from llms.constants import PROMPT_SPLIT_JOINER
from llms.enums import LlmErrorPrompt
from llms.llm_service import LlmService

load_dotenv()


class XaiService(LlmService):
    """LLM service backed by the xAI SDK."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.x_client = Client(api_key=os.getenv("XAI_API_KEY"))

    def _get_response_content(self, chat) -> str:
        response = chat.sample()
        content = response.content
        if LlmErrorPrompt.QUOTA_EXCEEDED in content.lower():
            raise LlmErrorPrompt.QUOTA_EXCEEDED
        if LlmErrorPrompt.LENGTH_EXCEEDED in content.lower():
            raise LlmErrorPrompt.LENGTH_EXCEEDED
        return content

    def _get_prompt(self, prompt_splits: list[str]) -> str:
        base_prompt_splits = [
            f"Respond with {LlmErrorPrompt.QUOTA_EXCEEDED} if no more credit for usage",
            f"Respond with {LlmErrorPrompt.LENGTH_EXCEEDED} if input + output length is too long",
            "Do not include prompt in the response",
        ]
        return PROMPT_SPLIT_JOINER.join(base_prompt_splits + prompt_splits)

    def invoke(self, prompt: str) -> str:
        try:
            chat = self.x_client.chat.create(model=ModelName.GROK_3_MINI)
            full_prompt = self._get_prompt([prompt])
            chat.append(user(full_prompt))
            return self._get_response_content(chat)
        except Exception as exc:
            self.logger.error("LLM API error: %s", exc)
            raise

    def detect_image_items(
        self,
        image_url: str,
        limit: Optional[int] = None,
    ) -> list[str]:
        try:
            chat = self.x_client.chat.create(model=ModelName.GROK_4_FAST_NON_REASONING)
            prompt_splits = [
                "What are the items in this image?",
                "Return the items only",
                "Return the items in title case separated by commas",
                "Return the items in order of focus from most focused to least focused",
            ]
            full_prompt = self._get_prompt(prompt_splits)
            chat.append(user(full_prompt, image(image_url=image_url, detail="low")))
            content = self._get_response_content(chat)
            item_list = content.split(",")
            return item_list[:limit or len(item_list)]
        except Exception as exc:
            self.logger.error("LLM API error: %s", exc)
            raise
