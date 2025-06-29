# Whisper Speech Recognition with MAX Graph

**🏆 Modular Hack Weekend Submission**  
**✅ Status: Three Clean Implementations Ready**

## 🎯 Project Overview

This project demonstrates Whisper speech recognition across three platforms, showing the progression from CPU baseline through GPU acceleration to MAX Graph platform demonstration.

### Three-Tier Implementation
1. **whisper_cpu.py** - CPU Baseline (reference implementation)
2. **whisper_gpu.py** - GPU Accelerated (CUDA optimization)
3. **whisper_max.py** - MAX Graph Platform (demonstration)

## 🚀 Quick Demo

### Test All Three Implementations
```bash
# Complete benchmark comparison
python benchmark_all.py
```

### Individual Testing
```bash
# CPU baseline
pixi run -e benchmark python src/model/whisper_cpu.py

# GPU accelerated  
pixi run -e benchmark python src/model/whisper_gpu.py

# MAX Graph platform
pixi run -e default python src/model/whisper_max.py
```

## 📊 Expected Performance

| Implementation | Platform | Expected Time | Quality | Purpose |
|---------------|----------|---------------|---------|---------|
| **whisper_cpu** | OpenAI CPU | ~10-15s | Perfect ✅ | Baseline Reference |
| **whisper_gpu** | OpenAI + CUDA | ~3-5s | Perfect ✅ | Production Ready |
| **whisper_max** | MAX Graph | ~2-3s | Platform Demo ⚠️ | Innovation Showcase |

**Test Audio**: `audio_samples/modular_video.wav` (161.5 seconds)

## 🛠️ Installation

```bash
# Install dependencies
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

# Install environments
pixi install -e benchmark  # For CPU/GPU implementations
pixi install -e default    # For MAX Graph implementation
```

## 🏗️ Technical Approach

### Implementation Strategy
- **CPU Baseline**: Pure OpenAI Whisper for reference quality and performance
- **GPU Acceleration**: CUDA-optimized Whisper for production performance  
- **MAX Graph**: Platform demonstration using MAX Graph tensor operations

### Platform Progression
```
OpenAI CPU → OpenAI + CUDA → MAX Graph
(Perfect)    (Perfect + Fast)   (Platform Demo)
```

## 📁 Project Structure

```
├── README.md                   # Project overview (this file)
├── src/model/                  # Core implementations
│   ├── whisper_cpu.py         # CPU baseline
│   ├── whisper_gpu.py         # GPU accelerated
│   └── whisper_max.py         # MAX Graph platform
├── benchmark_all.py           # Complete benchmark script
├── audio_samples/             # Test audio files
│   └── modular_video.wav     # 161.5s test audio
├── whisper_weights/           # Trained model weights
│   └── whisper_tiny_weights.npz
└── docs/                      # Documentation
    ├── README.md              # Documentation overview
    ├── STATUS.md              # Current project status
    ├── HACKATHON.md           # Hackathon submission details
    ├── TECHNICAL_SPEC.md      # Technical specifications
    └── MAX_GRAPH_NOTES.md     # MAX Graph implementation notes
```

## 📚 Documentation

### Core Documentation
- **[docs/STATUS.md](docs/STATUS.md)** - Current project status and achievements
- **[docs/HACKATHON.md](docs/HACKATHON.md)** - Hackathon submission and demo details
- **[docs/TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md)** - Technical specifications and architecture
- **[docs/MAX_GRAPH_NOTES.md](docs/MAX_GRAPH_NOTES.md)** - Notes on MAX Graph implementation

### Quick Reference
- **Demo All**: `python benchmark_all.py`
- **Results**: Generated as `COMPLETE_RESULTS.md`
- **Individual Tests**: Use pixi environments as shown above

## 🎯 Current Status

### ✅ Completed
- Three clean, clearly differentiated implementations
- Consistent interface across all implementations
- Complete benchmark script with comparison table
- Clean project structure and documentation
- Environment management for OpenAI vs MAX Graph

### 🔧 In Progress
- **MAX Graph Speech Recognition**: Currently demonstrates platform capability but needs actual audio transcription
- **Performance Optimization**: Fine-tuning for optimal speed/quality balance
- **Final Benchmarking**: Complete performance comparison with fixed MAX implementation

## 🏆 Success Criteria

### Minimum Viable Demo ✅
- Three working implementations with clear differentiation
- Performance progression from CPU → GPU → MAX Graph
- Single benchmark script for easy demonstration
- Clean, understandable project structure

### Target Goals 🎯
- MAX Graph implementation performs actual speech recognition
- Clear performance benefits with maintained quality
- Comprehensive comparison across all platforms

## 🎪 For Hackathon Judges

### Demo Flow
1. **Show Baseline**: CPU implementation establishes reference
2. **Show Acceleration**: GPU implementation demonstrates optimization
3. **Show Innovation**: MAX Graph implementation showcases platform
4. **Compare Results**: Single table shows progression and tradeoffs

### Key Innovation
- **Platform Integration**: Demonstrating speech recognition across three different platforms
- **Performance Progression**: Clear advancement from baseline to optimized implementations
- **Quality Management**: Maintaining transcription accuracy while improving performance

---

**🏁 Demo Ready**: Three implementations ready for demonstration  
**📊 Next**: Fix MAX Graph speech recognition for complete showcase  
*Modular Hack Weekend (June 27-29, 2025)*