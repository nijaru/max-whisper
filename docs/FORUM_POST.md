# max-whisper: Trying to Accelerate Whisper with MAX Graph

**Modular Hackathon 2025 Submission**

I attempted to accelerate OpenAI's Whisper speech recognition using MAX Graph. It doesn't work - the MAX Graph version fails to do proper speech recognition.

## What I Built

Three implementations to compare performance:
- `whisper_cpu.py` - baseline OpenAI Whisper 
- `whisper_gpu.py` - CUDA accelerated version
- `whisper_max.py` - MAX Graph encoder version

**Repository**: https://github.com/nijaru/max-whisper

## Results

| Implementation | Status | Output |
|---------------|--------|--------|
| CPU baseline | ✅ Works | Full transcription of 161s audio |
| GPU accelerated | ✅ Works | Full transcription (faster) |
| MAX Graph | ❌ Broken | Only 1-2 words then stops |

## What I Built (But Doesn't Work)

- MAX Graph encoder that compiles and runs without errors
- Weight extraction system that gets 65 tensors from Whisper tiny
- Basic integration between MAX Graph and PyTorch
- MAX Graph operations (matmul, layer_norm, gelu) compile successfully

## The Problem

When I connect the MAX Graph encoder to Whisper's decoder, it only outputs single words like "The" or "I" before hitting the end-of-text token. The encoder produces valid tensors with reasonable statistics, but they don't contain the semantic information the decoder needs.

This suggests either:
- Subtle math differences that compound across transformer layers
- Missing implementation details in the encoder
- Incompatibility between MAX Graph tensors and PyTorch decoder expectations

## What I Learned

Building individual MAX Graph components is straightforward. The tensor operations work as expected and performance is good. The challenge is integrating them into complex hybrid pipelines with existing PyTorch models.

There are compatibility issues that go beyond just matching tensor shapes and data types. Getting the math exactly right in complex AI models is harder than it initially appears.

## Try It

```bash
git clone https://github.com/nijaru/max-whisper
cd max-whisper  
make install
make demo  # Shows working CPU/GPU versions and broken MAX Graph version
```

Built during the Modular Hackathon 2025 weekend.