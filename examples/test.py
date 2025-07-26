from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-ft", use_fast=False)

decoded = tokenizer.decode([128257,128258], skip_special_tokens=False)
print(decoded)
