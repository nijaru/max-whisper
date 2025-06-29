# max-whisper: Trying to Accelerate Whisper with MAX Graph

**Modular Hackathon 2025 Submission**

I built an experimental MAX Graph version of OpenAI's Whisper speech recognition. While I successfully created a working encoder and decoder integration, the transcription quality is currently limited compared to the baseline implementations.

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
| MAX Graph | 🔄 Partial | Single words, needs refinement |

## What I Built

- **Complete MAX Graph encoder** - 4-layer transformer with proper convolution and bias handling
- **Full weight integration** - All 65 pretrained weights from Whisper tiny model
- **Working decoder integration** - MAX Graph encoder successfully drives PyTorch decoder
- **End-to-end pipeline** - Audio → MAX Graph encoder → PyTorch decoder → text output
- **Real MAX Graph operations** - Uses ops.matmul, ops.layer_norm, ops.gelu (not fallbacks)

## Current Status

The integration is technically successful - MAX Graph encoder features now successfully drive the Whisper decoder to produce text output. However, transcription quality is limited, producing single words rather than full sentences.

This demonstrates that:
- Cross-framework integration is achievable
- Mathematical precision is critical in transformer models
- Small implementation differences can significantly impact semantic understanding

## What I Learned

**Technical Achievements:**
- MAX Graph integrates well with existing AI codebases
- Complex computation graphs compile and execute efficiently (~120ms)
- Cross-framework decoder integration is possible with proper API usage
- Weight extraction from pretrained models works reliably

**Key Insights:**
- Mathematical precision is crucial in AI model implementations
- Individual components can work while integration reveals subtle issues
- The foundation demonstrates MAX Graph's potential for AI acceleration

This project shows that while challenging, MAX Graph acceleration of speech models is technically feasible and could deliver significant performance improvements with additional refinement.

## Try It

```bash
git clone https://github.com/nijaru/max-whisper
cd max-whisper  
make install
make demo  # Compare all three implementations
```

Built during the Modular Hackathon 2025 weekend.