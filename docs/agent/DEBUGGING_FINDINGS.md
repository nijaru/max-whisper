# Debugging Findings Log

*Track issues, attempts, and solutions during semantic quality debugging*

## Current Investigation: Semantic Quality Issue

**Problem**: MAX Graph encoder produces repetitive tokens (`<|ml|>`, `<|tl|>`) instead of meaningful transcription

**Status**: Ready to begin systematic debugging

---

## Investigation Template

Use this template for each debugging session:

### Session: [Date] - [Focus Area]

**Objective**: [What you're trying to fix/investigate]

**Hypothesis**: [What you think might be causing the issue]

**Approach**: [What you'll try]

**Results**: [What happened]

**Findings**: [What you learned]

**Next Steps**: [What to try next]

**Todos Created**: [List any todos added to Claude Code]

---

## Known Facts (Baseline)

### What Works ✅
- MAX Graph environment setup and compilation
- Weight extraction (all 65 weights from Whisper tiny)
- Graph compilation with ops.matmul, ops.layer_norm, ops.gelu, etc.
- Cross-framework integration (MAX Graph → PyTorch)
- Device management (GPU/CPU)
- Performance: ~123ms encoder execution
- Pipeline executes without technical errors

### What Doesn't Work ❌
- Semantic output quality
- Produces repetitive tokens instead of speech transcription
- Features lack semantic richness for speech recognition

### Comparison Baselines
- **CPU Implementation**: Perfect transcription, ~10.6s
- **GPU Implementation**: Perfect transcription, ~1.9s  
- **MAX Graph Implementation**: Repetitive output, ~0.5s total

---

## Debugging Sessions

### Session: [DATE] - Initial Setup
*This will be filled in when debugging begins*

---

## Patterns & Learnings

*This section will be updated as patterns emerge from debugging sessions*

### Common Issues
- [To be filled in as debugging progresses]

### Successful Solutions  
- [To be filled in as fixes are found]

### MAX Graph Specific Findings
- [To be filled in as we learn MAX Graph characteristics]

---

## Tools & Commands Used

### Feature Extraction
```bash
# Commands will be added as debugging tools are developed
pixi run -e benchmark extract-features-cpu
pixi run -e benchmark extract-features-gpu  
pixi run -e benchmark extract-features-max
```

### Comparison Analysis
```bash
# Commands for numerical comparison
pixi run -e benchmark compare-features
pixi run -e benchmark analyze-divergence
```

### Validation
```bash
# Commands for testing fixes
pixi run -e benchmark test-fix
pixi run -e benchmark validate-precision
```

---

## Reference Information

### Key Files for Semantic Quality
- `max-whisper/whisper_max.py` - MAX Graph implementation
- `max-whisper/max_graph_encoder.py` - Encoder architecture
- `max-whisper/whisper_weight_extractor.py` - Weight handling
- `benchmarks/benchmark_runner.py` - Testing infrastructure

### Important Operations to Focus On
1. **Attention Mechanism** - Core semantic processing
2. **Layer Normalization** - Numerical stability
3. **Matrix Multiplication** - Basic operations
4. **GELU Activation** - Nonlinear processing
5. **Positional Encoding** - Sequence understanding

### Debugging Priorities
1. Compare attention outputs layer by layer
2. Verify numerical precision of operations
3. Check weight loading and format conversion
4. Validate tensor shapes and device placement
5. Analyze activation patterns and distributions

---

*This document will be updated throughout the debugging process to capture all findings, failed attempts, and successful solutions.*