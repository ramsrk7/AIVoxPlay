from abc import ABC, abstractmethod
import openai
import re
import numpy as np
from typing import Generator 

class BaseTTS(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def speak(self, text: str, stream: bool = False):
        """Convert text to audio"""
        pass

    @staticmethod
    def call_completions_endpoint(
        prompt: str,
        endpoint: str,
        model_name: str,
        api_key: str = "dummy_key",
        max_new_tokens: int = 1600,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        stop: list = ["<custom_token_6>"]
    ) -> str:
        """
        Call the OpenAI-compatible completions endpoint
        """
        # Configure the client to use the custom endpoint
        client = openai.OpenAI(base_url=endpoint, api_key=api_key)
        print("Model Name: ", model_name)
        
        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=repetition_penalty - 1.0,  # OpenAI API uses different scale
                stream=False,
                stop=stop  # Stop at EOS token
            )
            
            print(response)
            return response.choices[0].text
                
        except Exception as e:
            print(f"Error calling completions endpoint: {e}")
            return ""


    @staticmethod
    def call_completions_endpoint_stream(
        prompt: str,
        endpoint: str,
        model_name: str,
        api_key: str = "dummy_key",
        max_new_tokens: int = 1600,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        stop: list = ["<custom_token_6>"]
    ) -> Generator[str, None, None]:
        """
        Simplified version: yields raw streamed text chunks from OpenAI-compatible completions endpoint.
        """
        import openai

        client = openai.OpenAI(base_url=endpoint, api_key=api_key)

        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=repetition_penalty - 1.0,
                stream=True,
                stop=stop
            )

            for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    text = getattr(chunk.choices[0], "text", "")
                    if text:
                        yield text

        except Exception as e:
            print(f"Error during streaming: {e}")
            return

    
    @staticmethod
    def apply_fade(audio: np.ndarray, sample_rate: int = 24000, fade_ms: int = 3) -> np.ndarray:
        """
        Apply fade-in and fade-out to the audio array.
        """
        
        if audio.ndim != 1:
            raise ValueError("Expected mono audio (1D array)")

        fade_samples = int(sample_rate * fade_ms / 1000.0)
        fade_samples = min(fade_samples, len(audio) // 2)

        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)

        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out

        return audio