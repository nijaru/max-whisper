# Modular Hackathon: Whisper MAX Graph Implementation
# Makefile for easy demo and benchmark execution

# Default model size (small for production-relevant performance)
MODEL_SIZE ?= small

# Default audio file
AUDIO_FILE ?= audio_samples/modular_video.wav

# Environment setup
PIXI_ENV = pixi run -e benchmark

# Suppress make directory messages
MAKEFLAGS += --no-print-directory

.PHONY: help demo benchmark test clean

# Default target
all: help

help:
	@echo "🚀 Modular Hackathon - Whisper MAX Graph Implementation"
	@echo "======================================================="
	@echo ""
	@echo "🎯 MAIN COMMANDS:"
	@echo "  demo          - All 4 implementations (default: small model)"
	@echo "  benchmark     - Performance analysis with detailed results"
	@echo ""
	@echo "🔧 INDIVIDUAL TESTS:"
	@echo "  cpu           - CPU baseline only"
	@echo "  gpu           - GPU accelerated only"
	@echo "  max           - MAX Graph integration only"
	@echo "  fast          - MAX Graph fast only"
	@echo ""
	@echo "🛠️ UTILITIES:"
	@echo "  gpu-check     - Verify GPU setup"
	@echo "  clean         - Clean up files"
	@echo ""
	@echo "💡 EXAMPLES:"
	@echo "  make demo                    # All 4 tests, small model"
	@echo "  make demo MODEL_SIZE=tiny    # All 4 tests, tiny model"
	@echo "  make cpu MODEL_SIZE=base     # CPU test only, base model"
	@echo "  make gpu                     # GPU test only, small model"

# Main demo - all 4 implementations with TUI
demo:
	@$(PIXI_ENV) python scripts/tui_demo.py --model-size $(MODEL_SIZE) --audio-file $(AUDIO_FILE)

# Individual implementation tests
cpu:
	@$(PIXI_ENV) python scripts/tui_demo.py --model-size $(MODEL_SIZE) --audio-file $(AUDIO_FILE) --tests cpu

gpu:
	@$(PIXI_ENV) python scripts/tui_demo.py --model-size $(MODEL_SIZE) --audio-file $(AUDIO_FILE) --tests gpu

max:
	@$(PIXI_ENV) python scripts/tui_demo.py --model-size $(MODEL_SIZE) --audio-file $(AUDIO_FILE) --tests max

fast:
	@$(PIXI_ENV) python scripts/tui_demo.py --model-size $(MODEL_SIZE) --audio-file $(AUDIO_FILE) --tests fast

# Detailed benchmark analysis  
benchmark:
	@$(PIXI_ENV) python benchmark_all.py --model-size $(MODEL_SIZE) --audio-file $(AUDIO_FILE)

# Model size convenience targets
demo-tiny:
	@$(MAKE) demo MODEL_SIZE=tiny

demo-small:
	@$(MAKE) demo MODEL_SIZE=small

demo-base:
	@$(MAKE) demo MODEL_SIZE=base

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -f COMPLETE_RESULTS*.md
	rm -f *.pyc
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/model/__pycache__
	@echo "✅ Cleanup complete"

# GPU compatibility check
gpu-check:
	@$(PIXI_ENV) python scripts/gpu_check.py