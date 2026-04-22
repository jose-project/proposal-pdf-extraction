import asyncio
from typing import Any, Dict, Optional

import aiohttp


class RemoteLLM:
    """
    Async wrapper around the remote Qwen2.5-7B-Instruct endpoint.
    Optimized with session reuse and timeout handling.
    """

    def __init__(
        self,
        endpoint: str = "http://143.110.210.212/v1/chat/completions",
        timeout: int = 120,  # 120s to handle queued requests on single-GPU server
    ):
        self.endpoint = endpoint
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        headers = {"Content-Type": "application/json"}

        session = await self._get_session()
        async with session.post(self.endpoint, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        if "choices" not in data or not data["choices"]:
            raise ValueError("Invalid LLM response: missing choices")
        if "message" not in data["choices"][0]:
            raise ValueError("Invalid LLM response: missing message")

        text = data["choices"][0]["message"]["content"]
        return text.strip()

    def chat_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Convenience synchronous wrapper (runs the async call in an event loop).
        """
        return asyncio.run(
            self.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

