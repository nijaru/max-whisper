# Technical Specification

**Project**: Whisper Speech Recognition with MAX Graph  
**Goal**: Three-tier speech recognition implementation demonstrating platform progression

## 🏗️ Architecture Overview

### Three-Implementation Strategy
```
CPU Baseline → GPU Acceleration → MAX Graph Platform
    ↓              ↓                    ↓
whisper_cpu    whisper_gpu         whisper_max
```

## 📋 Implementation Specifications

### 1. whisper_cpu.py - CPU Baseline
**Purpose**: Reference implementation and performance baseline

**Technical Details**:
- **Platform**: Pure OpenAI Whisper on CPU
- **Model**: Whisper-tiny (39M parameters)
- **Device**: CPU only (forced)
- **Quality**: Perfect transcription (ground truth)
- **Environment**: benchmark (requires OpenAI Whisper)

**Key Features**:
```python
class WhisperCPU:
    def __init__(self):
        self.whisper_model = whisper.load_model("tiny", device="cpu")
    
    def transcribe(self, audio_file):
        # Pure CPU processing
        # No optimizations
        # Reference quality
```

### 2. whisper_gpu.py - GPU Accelerated  
**Purpose**: Production-ready optimized implementation

**Technical Details**:
- **Platform**: OpenAI Whisper + CUDA GPU acceleration
- **Model**: Whisper-tiny with GPU optimizations
- **Device**: CUDA GPU (required)
- **Quality**: Perfect transcription (identical to CPU)
- **Optimizations**: 
  - `torch.backends.cudnn.benchmark = True`
  - GPU tensor operations
  - Optimized inference parameters

**Key Features**:
```python
class WhisperGPU:
    def __init__(self):
        if not torch.cuda.is_available():
            raise Exception("CUDA required")
        self.whisper_model = whisper.load_model("tiny", device="cuda")
        torch.backends.cudnn.benchmark = True
```

### 3. whisper_max.py - MAX Graph Platform
**Purpose**: Platform demonstration and research

**Technical Details**:
- **Platform**: MAX Graph tensor operations
- **Model**: Custom implementation using trained Whisper weights
- **Device**: GPU (MAX Graph)
- **Weights**: 47 trained tensors from Whisper-tiny
- **Environment**: default (MAX Graph support)

**Current Status**: Generates text, needs speech recognition fix

**Key Features**:
```python
class WhisperMAX:
    def __init__(self):
        self.max_device = DeviceRef.GPU()
        self.weights = np.load("whisper_tiny_weights.npz")  # 47 tensors
        self.tokenizer = tiktoken.get_encoding("gpt2")
```

## 🔧 Technical Components

### Audio Processing Pipeline
```
Audio File (WAV) → Librosa Load → Mel Spectrogram → Model Processing → Text Output
```

**Common Audio Specs**:
- **Sample Rate**: 16kHz (Whisper standard)
- **Format**: WAV
- **Mel Bins**: 80
- **Test Audio**: 161.5s Modular technical presentation

### Model Architecture

#### Whisper-Tiny Specifications
- **Parameters**: 39M
- **Encoder**: 384-dimensional, 6 layers
- **Decoder**: 384-dimensional, 6 layers  
- **Vocabulary**: 51,865 tokens
- **Context**: 1500 audio tokens, 224 text tokens

#### Trained Weights (47 tensors)
```
- encoder_conv1_weight: (384, 80, 3)
- encoder_conv2_weight: (384, 384, 3)  
- positional_embedding: (448, 384)
- token_embedding: (51865, 384)
- [... 43 more tensors]
```

## 🚀 Performance Specifications

### Expected Performance Targets
| Implementation | Target Time | Target Speedup | Quality Target |
|---------------|-------------|----------------|----------------|
| whisper_cpu | 10-15s | 1.0x (baseline) | Perfect ✅ |
| whisper_gpu | 3-5s | 3-5x speedup | Perfect ✅ |
| whisper_max | 2-3s | 5-7x speedup | Good ⚠️ |

### Quality Metrics
- **Perfect**: Identical transcription to OpenAI Whisper
- **Good**: Recognizable speech content with minor differences
- **Generated**: Plausible text that doesn't match audio content

## 🔄 Environment Management

### Environment Specifications

#### benchmark Environment
**Requirements**:
```toml
[dependencies]
openai-whisper = ">=20231117"
torch = {version = ">=2.0.0", extras = ["cuda"]}
librosa = ">=0.10.0"
numpy = ">=1.21.0"
```

**Capabilities**: 
- ✅ OpenAI Whisper models
- ✅ CUDA operations
- ❌ MAX Graph operations

#### default Environment  
**Requirements**:
```toml
[dependencies]
max = ">=24.5.0"
numpy = ">=1.21.0"
librosa = ">=0.10.0"
tiktoken = ">=0.5.0"
```

**Capabilities**:
- ✅ MAX Graph operations
- ✅ Tensor processing
- ❌ OpenAI Whisper models

### Environment Switching
```bash
# For CPU/GPU implementations
pixi run -e benchmark python src/model/whisper_cpu.py
pixi run -e benchmark python src/model/whisper_gpu.py

# For MAX Graph implementation  
pixi run -e default python src/model/whisper_max.py
```

## 📊 Benchmarking Specification

### Benchmark Script: benchmark_all.py
**Functionality**:
- Tests all three implementations
- Uses same audio input for fair comparison
- Handles environment switching
- Generates comparison table

**Output Format**:
```markdown
| Implementation | Platform | Time | Speedup | Status | Quality |
|---------------|----------|------|---------|--------|---------|
| CPU Baseline | OpenAI CPU | 10.5s | 1.0x | ✅ Success | Perfect ✅ |
| GPU Accelerated | OpenAI + CUDA | 3.2s | 3.3x | ✅ Success | Perfect ✅ |
| MAX Graph | MAX Graph | 2.1s | 5.0x | ✅ Success | Generated ⚠️ |
```

## 🎯 Implementation Standards

### Code Quality Requirements
- **Consistent Interface**: All implementations use `transcribe()` method
- **Error Handling**: Graceful failure with informative messages
- **Environment Detection**: Automatic capability detection
- **Clear Output**: Consistent logging and status messages

### Testing Requirements
```python
def test_implementation():
    model = Implementation()
    assert model.available == True
    result = model.transcribe("audio_samples/modular_video.wav")
    assert len(result) > 50  # Substantial output
    assert "error" not in result.lower()  # No error messages
```

## 🔧 Development Workflow

### Implementation Development
1. **Create Implementation**: Follow interface specification
2. **Test Individually**: Verify functionality in appropriate environment
3. **Add to Benchmark**: Include in benchmark_all.py
4. **Verify Comparison**: Ensure fair comparison with other implementations

### Quality Assurance
- Individual implementation demos work
- Benchmark script completes successfully
- Performance progression is logical
- Documentation matches implementation

---

**Technical Status**: Architecture defined, CPU/GPU implementations complete, MAX Graph needs speech recognition fix