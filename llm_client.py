import httpx
import asyncio

class LLMClient:
    def __init__(self, provider, model, api_key=None):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key

    async def generate(self, prompt):
        if self.provider == "ollama":
            return await self._call_ollama(prompt)
        elif self.provider == "ollama-cloud":
            return await self._call_ollama_cloud(prompt)
        elif self.provider == "openai":
            return await self._call_openai(prompt)
        elif self.provider == "openrouter":
            return await self._call_openrouter(prompt)
        elif self.provider == "nvidia":
            return await self._call_nvidia(prompt)
        elif self.provider == "gemini":
            return await self._call_gemini(prompt)
        elif self.provider == "huggingface":
            return await self._call_huggingface(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _call_ollama(self, prompt):
        url = "http://localhost:11434/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def _call_ollama_cloud(self, prompt):
        # Menggunakan endpoint cloud Ollama (sesuaikan URL jika menggunakan provider spesifik)
        # Untuk Ollama Cloud umum/via proxy, biasanya menggunakan format chat completion
        url = "https://ollama.com/api/generate" 
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def _call_openai(self, prompt):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_openrouter(self, prompt):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_nvidia(self, prompt):
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _call_gemini(self, prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_huggingface(self, prompt):
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": prompt}
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=120.0)
            resp.raise_for_status()
            res = resp.json()
            if isinstance(res, list):
                return res[0].get("generated_text", "")
            return res.get("generated_text", "")
