#!/usr/bin/env python3
import whisper
from whisper.tokenizer import get_tokenizer

model = whisper.load_model('tiny')
tokenizer = get_tokenizer(model.is_multilingual)

# Check what token 21829 is
token_id = 21829
token_text = tokenizer.decode([token_id])
print(f'Token {token_id}: "{token_text}"')
print(f'Token {token_id} repr: {repr(token_text)}')

# Check some nearby tokens
for i in range(21825, 21835):
    try:
        text = tokenizer.decode([i])
        print(f'Token {i}: "{text}" repr: {repr(text)}')
    except:
        print(f'Token {i}: [decode error]')

# Check special tokens
print("\nSpecial tokens:")
print(f"EOS token: {tokenizer.eot}")
print(f"SOT token: {tokenizer.sot}")