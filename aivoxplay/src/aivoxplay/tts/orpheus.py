from __future__ import annotations
import sys
from dotenv import load_dotenv
load_dotenv()
from .base import BaseTTS
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import time
import argparse, re, sys
import queue
import openai
from transformers import AutoTokenizer
from snac import SNAC
import torch
import numpy as np
from typing import Sequence, Union
from .factory import TokenStreamParser, OrpheusAudioProcessor, dummy_stream_custom_tokens_without_parser
import threading
import asyncio
import random

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
        self._tok_base_id = self.tokenizer.encode("<custom_token_0>", add_special_tokens=False, return_tensors=None)[0]
        self._tok_find_id = 128257           # token_to_find
        self._tok_remove_id = 128258         # token_to_remove
        self._tok_offset = 128266            # offset removed later
        self._n_layers = 7
        self.parser = TokenStreamParser()                    # ORPHEUS_N_LAYERS
        

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
            return OrpheusAudioProcessor.tokens_decoder_sync(self._streaming_speak_v2(text, voice))
        else:
            return self._non_streaming_speak(text, voice)
    
    def _streaming_speak_v2(self, text: str, voice: str):
        """
        Stream audio tokens, decode in ~2k-char batches ending with '>', and yield (audio, sr).
        """
        import logging
        logger = logging.getLogger(__name__)

        formatted_prompt = self._build_prompt_input(text, voice)

        stream = OrpheusTTS.call_completions_endpoint_stream(
            endpoint=self.endpoint,
            api_key=self.api_key,
            model_name=self.MODEL_NAME,
            prompt=formatted_prompt,
        )
        #stream = dummy_stream_custom_tokens_without_parser(TEST_AUDIO_TOKENS)
        token_queue = queue.Queue()

        async def async_producer():
            for chunk in stream:
                # Place each token text into the queue.
                self.parser.feed(chunk)
                tokens = self.parser.get_tokens()
                if tokens:
                    #encoded_tokens = [vox_tts.tokenizer.encode(token, add_special_tokens=False, return_tensors=None)[0] for token in tokens]
                    for token in tokens:
                        token_queue.put(token)
            token_queue.put(None)  

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()
        


    def _streaming_speak(self, text: str, voice: str):
        """
        Stream audio tokens, decode in ~2k-char batches ending with '>', and yield (audio, sr).
        """
        import logging
        logger = logging.getLogger(__name__)

        formatted_prompt = self._build_prompt_input(text, voice)

        stream = OrpheusTTS.call_completions_endpoint_stream(
            endpoint=self.endpoint,
            api_key=self.api_key,
            model_name=self.MODEL_NAME,
            prompt=formatted_prompt,
        )

        THRESH = 2000
        final_response = ""  # buffer of tokens/chars we haven't decoded yet

        for chunk in stream:
            if not chunk:
                continue

            final_response += chunk

            # Decode in batches once threshold is crossed
            if len(final_response) % THRESH == 0:
                cut = final_response.rfind(">")
                if cut != -1:
                    to_decode = final_response[:cut + 1]
                    #final_response = final_response[cut + 1:]

                    try:
                        audio = self._decode_tokens(to_decode)
                        if audio is not None:
                            yield audio, 24000
                    except Exception as e:
                        logger.exception("Decode failed on intermediate batch: %s", e)

        # Flush whatever is left
        if final_response:
            try:
                audio = self._decode_tokens(final_response)
                if audio is not None:
                    yield audio, 24000
            except Exception as e:
                logger.exception("Decode failed on final batch: %s", e)



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

        #response = "<custom_token_3895><custom_token_7668><custom_token_11727><custom_token_16275><custom_token_18294><custom_token_24193><custom_token_24881><custom_token_1032><custom_token_7326><custom_token_8690><custom_token_12874><custom_token_18453><custom_token_22191><custom_token_27864><custom_token_3566><custom_token_7682><custom_token_10786><custom_token_15157><custom_token_19116><custom_token_22028><custom_token_27938><custom_token_996><custom_token_4285><custom_token_8858><custom_token_12660><custom_token_17103><custom_token_23364><custom_token_25184><custom_token_254><custom_token_5931><custom_token_9445><custom_token_13200><custom_token_16677><custom_token_24172><custom_token_28409><custom_token_3808><custom_token_5405><custom_token_9050><custom_token_13542><custom_token_18413><custom_token_23704><custom_token_26413><custom_token_3808><custom_token_7431><custom_token_10169><custom_token_13057><custom_token_18333><custom_token_21970><custom_token_28036><custom_token_2710><custom_token_5981><custom_token_9599><custom_token_15480><custom_token_18237><custom_token_20876><custom_token_25977><custom_token_3929><custom_token_5949><custom_token_9593><custom_token_12523><custom_token_18079><custom_token_22004><custom_token_27342><custom_token_378><custom_token_7543><custom_token_8427><custom_token_14698><custom_token_16604><custom_token_24254><custom_token_27948><custom_token_2744><custom_token_8064><custom_token_11507><custom_token_12623><custom_token_20373><custom_token_22203><custom_token_25996><custom_token_1736><custom_token_5911><custom_token_10797><custom_token_16319><custom_token_18174><custom_token_23406><custom_token_26661><custom_token_2363>"
        
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
