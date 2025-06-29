# Documentation

This directory contains the core documentation for the Whisper MAX Graph project.

## 📁 Documentation Files

### Core Status & Progress
- **[STATUS.md](STATUS.md)** - Current project status and achievements
- **[HACKATHON.md](HACKATHON.md)** - Hackathon submission details and evaluation criteria

### Technical Documentation
- **[TECHNICAL_SPEC.md](TECHNICAL_SPEC.md)** - Technical specifications and implementation details
- **[MAX_GRAPH_NOTES.md](MAX_GRAPH_NOTES.md)** - Notes on MAX Graph implementation and getting it fully working

## 🎯 Quick Reference

### Current Status
The project has three working implementations:
- **whisper_cpu.py** - CPU baseline for reference
- **whisper_gpu.py** - GPU-accelerated production version
- **whisper_max.py** - MAX Graph platform demonstration

### Next Steps
1. **Fix whisper_max.py** to perform actual speech recognition (not just text generation) using MAX Graph + OpenAI tokenizer for correct results
2. **Create whisper_max_fast.py** - Copy working whisper_max.py and optimize with platform-specific tricks for maximum performance
3. **Benchmark all implementations** with fair comparison showing CPU → GPU → MAX Graph progression

### Key Files
- **[STATUS.md](STATUS.md)** - Current project status and achievements
- **[MAX_GRAPH_NOTES.md](MAX_GRAPH_NOTES.md)** - Implementation notes for getting MAX Graph fully working
- **[DEMO_GUIDE.md](DEMO_GUIDE.md)** - Hackathon demonstration guide for judges
- **[TECHNICAL_SPEC.md](TECHNICAL_SPEC.md)** - Technical specifications and architecture
