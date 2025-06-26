import base64
import httpx
import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

from .exceptions import LLMError, ParsingError
from .utils import get_logger, get_json_from_response, tenacity_retry

logger = get_logger(__name__)


class BaseLLM(ABC):

    def __init__(self, model: str, api_key: str, **kwargs):
        if not api_key:
            raise LLMError("API key is required for any LLM interaction.")
        self.model = model
        self.api_key = api_key
        self.timeout = kwargs.get("timeout", 120)

    @abstractmethod
    def execute(self, prompt: str, **kwargs) -> Union[str, Dict, List]:
        """同步执行LLM调用。"""
        pass

    @abstractmethod
    async def execute_async(self, prompt: str, **kwargs) -> Union[str, Dict, List]:
        """异步执行LLM调用。"""
        pass

    def _prepare_messages(
            self,
            prompt: str,
            system_message: Optional[str] = None,
            history: Optional[List[Dict[str, Any]]] = None,
            images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        if history:
            messages.extend(history)

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images:
            for image_path in images:
                try:
                    with open(image_path, "rb") as f:
                        encoded_image = base64.b64encode(f.read()).decode("utf-8")
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                    })
                except IOError as e:
                    logger.error(f"Cannot read image file {image_path}: {e}")
                    raise LLMError(f"Image file not accessible: {image_path}") from e

        messages.append({"role": "user", "content": user_content})
        return messages

    def _process_response(self, response_json: Dict, return_json: bool) -> Union[str, Dict, List]:
        """从API响应中提取内容并根据需要解析JSON。"""
        try:
            response_text = response_json["choices"][0]["message"]["content"]
            token_usage = response_json.get("usage", {})
            if token_usage:
                logger.info(f"LLM Token usage: "
                            f"Prompt={token_usage.get('prompt_tokens', 0)}, "
                            f"Completion={token_usage.get('completion_tokens', 0)}, "
                            f"Total={token_usage.get('total_tokens', 0)}")
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid LLM response format: {response_json}")
            raise LLMError("Received an invalid response structure from LLM API.") from e

        if return_json:
            try:
                return get_json_from_response(response_text)
            except ParsingError as e:
                # 记录原始文本以便调试
                logger.error(f"Failed to parse JSON. Raw LLM response text:\n---\n{response_text}\n---")
                raise e
        return response_text


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, model: str, api_key: str, base_url: str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @tenacity_retry
    def execute(self, prompt: str, **kwargs) -> Union[str, Dict, List]:
        return_json = kwargs.pop("return_json", False)
        messages = self._prepare_messages(prompt, **kwargs)
        payload = {"model": self.model, "messages": messages, "usage": True}

        try:
            with httpx.Client(headers=self.headers, timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload)
                response.raise_for_status()
                return self._process_response(response.json(), return_json)
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API request failed with status {e.response.status_code}: {e.response.text}")
            raise LLMError(f"API Error: {e.response.text}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM call: {e}")
            raise LLMError("Unexpected error during API call.") from e

    @tenacity_retry
    async def execute_async(self, prompt: str, **kwargs) -> Union[str, Dict, List]:
        return_json = kwargs.pop("return_json", False)
        messages = self._prepare_messages(prompt, **kwargs)
        payload = {"model": self.model, "messages": messages, "usage": True}

        async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/chat/completions", json=payload)
                response.raise_for_status()
                return self._process_response(response.json(), return_json)
            except httpx.HTTPStatusError as e:
                logger.error(f"Async LLM API request failed: {e.response.text}")
                raise LLMError(f"API Error: {e.response.text}") from e
            except Exception as e:
                logger.error(f"An unexpected async error occurred during LLM call: {e}")
                raise LLMError("Unexpected async error during API call.") from e