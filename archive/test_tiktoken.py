#!/usr/bin/env python3
"""Test tiktoken availability"""

try:
    import tiktoken
    print("✅ tiktoken available:", tiktoken.__version__)
    
    # Test encoding
    tokenizer = tiktoken.get_encoding("gpt2")
    test_text = "The Max Graph provides high performance"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"✅ Test text: {test_text}")
    print(f"✅ Tokens: {tokens}")
    print(f"✅ Decoded: {decoded}")
    
except ImportError as e:
    print(f"❌ tiktoken not available: {e}")
except Exception as e:
    print(f"❌ tiktoken error: {e}")