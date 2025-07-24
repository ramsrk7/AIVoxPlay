from abc import ABC, abstractmethod
import openai

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
        max_new_tokens: int = 1024,
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
