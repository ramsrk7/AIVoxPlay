from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
from .base import BaseTTS
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import time
import argparse, re, sys
import openai
from transformers import AutoTokenizer
from snac import SNAC
import torch
HF_TOKEN=os.getenv("HF_TOKEN")


class OrpheusTTS(BaseTTS):
    
    def __init__(self, endpoint: str, model_name: str = "unsloth/orpheus-3b-0.1-ft", api_key: str = "dummy_key"):
        super().__init__(model_name)
        # Load model and tokenizer (vLLM/HuggingFace etc.)
        self.tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-ft", use_fast=False, token=HF_TOKEN)
        self.device= "cpu"
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)
        self.MODEL_NAME = model_name
        self.AUDIO_SR = 24_000        
        self.START_TOKEN_ID = 128259  # Start of human
        self.END_TEXT_TOKEN_ID = 128009  # End of text
        self.END_HUMAN_TOKEN_ID = 128260  # End of human
        self.voices = ["tara","leah","jess","leo","dan","mia","zac","zoe"]
        self.endpoint = endpoint
        self.api_key = api_key
        

    def _build_prompt_input(self, text: str, voice: str = "tara") -> str:
        """
        Constructs a formatted prompt by prefixing with a voice label, encoding it into tokens,
        adding boundary tokens, and decoding it back to a string.

        Args:
            text (str): The input text to be spoken or processed.
            voice (str, optional): The voice identifier to prefix the prompt. Defaults to "tara".

        Returns:
            str: A decoded string including custom tokens, ready to be sent to the inference endpoint.
        """
        combined_text = f"{voice}: {text}"
        encoded_ids = self.tokenizer.encode(combined_text, add_special_tokens=False, return_tensors=None)
        token_sequence = [self.START_TOKEN_ID] + encoded_ids + [self.END_TEXT_TOKEN_ID, self.END_HUMAN_TOKEN_ID]
        decoded_prompt = self.tokenizer.decode(token_sequence, skip_special_tokens=False)

        return decoded_prompt
    
    def _decode_tokens(self, tokens):
        custom_tokens = re.findall(r'<custom_token_\d+>', tokens)
        custom_token_ids = [self.tokenizer.encode(token, add_special_tokens=False, return_tensors=None)[0] for token in custom_tokens]
        generated_tokens = torch.tensor([custom_token_ids])

        token_to_find = 128257
        token_to_remove = 128258

        token_indices = (generated_tokens == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_tokens[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = generated_tokens

        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t.item() - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        def redistribute_codes(code_list):
            layer_1, layer_2, layer_3 = [], [], []
            for i in range(len(code_list)//7):
                layer_1.append(code_list[7*i])
                layer_2.append(code_list[7*i+1]-4096)
                layer_3.append(code_list[7*i+2]-(2*4096))
                layer_3.append(code_list[7*i+3]-(3*4096))
                layer_2.append(code_list[7*i+4]-(4*4096))
                layer_3.append(code_list[7*i+5]-(5*4096))
                layer_3.append(code_list[7*i+6]-(6*4096))
            
            codes = [torch.tensor(layer_1).unsqueeze(0).to(self.device),
                     torch.tensor(layer_2).unsqueeze(0).to(self.device),
                     torch.tensor(layer_3).unsqueeze(0).to(self.device)]
            
            with torch.no_grad():
                audio_hat = self.snac_model.decode(codes)
            return audio_hat
        
        if code_lists:
            samples = redistribute_codes(code_lists[0])
            audio_array = samples.detach().squeeze().contiguous().cpu().numpy()
            #sf.write(args.out, audio_array, 24000)
            #print(f"\nAudio successfully written to {args.out}")
            return audio_array
        else:
            print("\nCould not generate audio, no codes to process.")
            return


    def speak(self, text: str, voice: str = "tara", stream: bool = False):

        if voice not in self.voices:
            voice = voices[0] #Select default voice if invalid
        if stream:
            return self._streaming_speak(text, voice)
        else:
            return self._non_streaming_speak(text, voice)

    def _streaming_speak(self, text: str, voice: str):
        # Stream via websocket or token generator
        # convert text to tokens
        # call openai api
        # convert back
       raise NotImplementedError()

    def _non_streaming_speak(self, text: str, voice: str):
        # One-shot audio generation
        # convert text to tokens
        # call openai api
        # convert back
        formatted_prompt = self._build_prompt_input(text, voice)
        
        response = OrpheusTTS.call_completions_endpoint(
            endpoint=self.endpoint,
            api_key=self.api_key,
            model_name=self.MODEL_NAME,
            prompt=formatted_prompt,
        )
        
        if isinstance(response, str):
            response_tokens = response
        elif hasattr(response, 'choices') and response.choices:
            response_tokens = response.choices[0].text
        else:
            response_tokens = ""

        print("Got the response.")    
        print(response)
        if response_tokens:
            return self._decode_tokens(response_tokens), 24000
        
        return

    def stream(
        self,
        text: str,
        voice: str = "default",
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        buffer_groups: int = 4,
        padding_ms: int = 250
    ):
        """Yields (sample_rate, audio_chunk) for given input text using Orpheus"""
        if not text.strip():
            logger.warning("Empty input text for TTS.")
            return

        if not self.snac_model:
            logger.error("SNAC model is not loaded.")
            raise RuntimeError("SNAC model is not initialized.")

        logger.info(f"Starting TTS streaming for text: {text[:50]}...")

        tts_start = time.time()
        audio_gen = self.snac_model.generate_speech_stream(
            text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            buffer_groups=buffer_groups,
            padding_ms=padding_ms
        )

        for sr, audio_chunk in audio_gen:
            if audio_chunk is not None and getattr(audio_chunk, 'size', 0) > 0:
                yield sr, audio_chunk

        logger.info(f"TTS stream completed in {time.time() - tts_start:.2f}s.")