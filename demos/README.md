# Demos Directory

Interactive demonstrations of MAX-Whisper functionality.

## Available Demos

### Production Component Demos
- `demo_trained_weights_simple.py` - Trained weight loading demonstration
- `integrate_real_tokenizer.py` - Real tokenizer integration showcase

### Comparison Demos
- `enhanced_comparison.py` - Side-by-side model comparison
- `demo_presentation.py` - Presentation-ready demonstration

## Usage

### Quick Demos
```bash
# Trained weights demo
pixi run -e benchmark python demos/demo_trained_weights_simple.py

# Real tokenizer demo  
pixi run -e benchmark python demos/integrate_real_tokenizer.py

# Enhanced comparison
pixi run -e benchmark python demos/enhanced_comparison.py
```

### Presentation Demo
```bash
# Full demonstration for presentations
pixi run -e benchmark python demos/demo_presentation.py
```

## Demo Features

### Trained Weights Demo
- Loads 47 extracted weight tensors
- Shows weight shapes and purposes
- Validates tensor compatibility

### Tokenizer Demo
- Real OpenAI tiktoken integration
- Encoding/decoding examples
- Special token handling

### Comparison Demo
- Side-by-side model performance
- Real audio processing
- Quality and speed metrics

### Presentation Demo
- Complete end-to-end workflow
- Performance visualizations
- Strategic impact summary