# max-whisper: Speech Recognition with MAX Graph
# Makefile for demo and testing of all implementations

# Default model size (tiny for fastest testing)
MODEL_SIZE ?= tiny

# Default audio file
AUDIO_FILE ?= audio_samples/modular_video.wav

# Environment setup
PIXI_ENV = pixi run -e benchmark

# Environment check helper
define check_env
	@$(MAKE) env-check >/dev/null 2>&1 || { echo "❌ Environment not ready. Run 'make install' first."; exit 1; }
endef

# Suppress make directory messages
MAKEFLAGS += --no-print-directory

# Extract positional arguments from command line
ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
MODEL_ARG := $(if $(filter tiny small base,$(ARGS)),$(filter tiny small base,$(ARGS)),$(MODEL_SIZE))
AUDIO_ARG := $(if $(filter-out tiny small base,$(ARGS)),$(filter-out tiny small base,$(ARGS)),$(AUDIO_FILE))

# Define phony targets including model sizes
.PHONY: help demo benchmark test clean tiny small base cpu gpu max install build env-check gpu-check

# Default target - run recommended demo
all:
	$(call check_env)
	@echo "🚀 Starting recommended demo (tiny model)..."
	@$(PIXI_ENV) python scripts/tui_demo.py tiny $(AUDIO_FILE)

help:
	@echo "🚀 max-whisper - Speech Recognition with MAX Graph"
	@echo "======================================================="
	@echo ""
	@echo "🚀 QUICK START:"
	@echo "  tiny                   - Full demo with tiny model (fastest, recommended)"
	@echo "  small                  - Full demo with small model (slower)"
	@echo "  base                   - Full demo with base model (slowest)"
	@echo ""
	@echo "🎯 MAIN COMMANDS:"
	@echo "  demo [model] [file]    - All 3 implementations"
	@echo "  benchmark [model]      - Performance analysis with detailed results"
	@echo ""
	@echo "🔧 INDIVIDUAL TESTS:"
	@echo "  cpu [model] [file]     - CPU baseline only"
	@echo "  gpu [model] [file]     - GPU accelerated only"
	@echo "  max [model] [file]     - MAX Graph integration only"
	@echo ""
	@echo "🛠️ SETUP & UTILITIES:"
	@echo "  install                - Install pixi and setup environment"
	@echo "  build                  - Build project (install dependencies)"
	@echo "  env-check              - Check pixi environment"
	@echo "  gpu-check              - Verify GPU setup"
	@echo "  clean                  - Clean up files"
	@echo ""
	@echo "💡 EXAMPLES:"
	@echo "  make install           # First-time setup (install pixi + dependencies)"
	@echo "  make                   # Run recommended demo (after setup)"
	@echo "  make tiny              # Quick demo with tiny model"
	@echo "  make demo base         # All 3 tests, base model"
	@echo "  make cpu tiny          # CPU test only, tiny model"
	@echo "  make max small my.wav  # MAX Graph test, small model, custom file"

# Main demo - all 3 implementations with TUI
demo:
	$(call check_env)
	@echo "🚀 Starting full demo with $(MODEL_ARG) model..."
	@$(PIXI_ENV) python scripts/tui_demo.py $(MODEL_ARG) $(AUDIO_ARG)

# Individual implementation tests
cpu:
	$(call check_env)
	@echo "🔧 Running CPU test with $(MODEL_ARG) model..."
	@$(PIXI_ENV) python scripts/tui_demo.py $(MODEL_ARG) $(AUDIO_ARG) --tests cpu

gpu:
	$(call check_env)
	@echo "⚡ Running GPU test with $(MODEL_ARG) model..."
	@$(PIXI_ENV) python scripts/tui_demo.py $(MODEL_ARG) $(AUDIO_ARG) --tests gpu

max:
	$(call check_env)
	@echo "🎯 Running MAX Graph test with $(MODEL_ARG) model..."
	@$(PIXI_ENV) python scripts/tui_demo.py $(MODEL_ARG) $(AUDIO_ARG) --tests max

# fast implementation removed

# Detailed benchmark analysis  
benchmark:
	$(call check_env)
	@echo "📊 Running comprehensive benchmark with $(MODEL_ARG) model..."
	@$(PIXI_ENV) python benchmark_all.py --model-size $(MODEL_ARG) --audio-file $(AUDIO_ARG)

# Direct model size commands (run full demo with that model)
# Only run if they're the primary target, not secondary arguments
tiny:
ifeq ($(word 1,$(MAKECMDGOALS)),tiny)
	$(call check_env)
	@echo "🚀 Starting tiny model demo (fastest)..."
	@$(PIXI_ENV) python scripts/tui_demo.py tiny $(AUDIO_FILE)
else
	@true
endif

small:
ifeq ($(word 1,$(MAKECMDGOALS)),small)
	$(call check_env)
	@echo "🚀 Starting small model demo (recommended)..."
	@$(PIXI_ENV) python scripts/tui_demo.py small $(AUDIO_FILE)
else
	@true
endif

base:
ifeq ($(word 1,$(MAKECMDGOALS)),base)
	$(call check_env)
	@echo "🚀 Starting base model demo (best quality)..."
	@$(PIXI_ENV) python scripts/tui_demo.py base $(AUDIO_FILE)
else
	@true
endif

# Catch-all rule for audio files and unknown targets
%:
	@:

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -f COMPLETE_RESULTS*.md
	rm -f *.pyc
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/model/__pycache__
	@echo "✅ Cleanup complete"

# Installation and setup commands
install:
	@echo "🏗️ Setting up Modular Hackathon project..."
	@echo "📦 Step 1: Installing pixi package manager..."
	@if command -v pixi >/dev/null 2>&1; then \
		echo "✅ pixi already installed"; \
	else \
		echo "📥 Installing pixi..."; \
		curl -fsSL https://pixi.sh/install.sh | bash; \
		echo "✅ pixi installed"; \
		echo "⚠️  Please restart your shell or run: source ~/.bashrc"; \
		echo "⚠️  Then run 'make build' to continue setup"; \
		exit 0; \
	fi
	@echo "📦 Step 2: Installing project dependencies..."
	@$(MAKE) build
	@echo "🎉 Installation complete! Try: make small"

setup-weights:
	@echo "📦 Setting up MAX Graph Whisper weights..."
	@$(PIXI_ENV) python scripts/setup_weights.py

build:
	@echo "🔨 Building project dependencies..."
	@if ! command -v pixi >/dev/null 2>&1; then \
		echo "❌ pixi not found. Please run 'make install' first."; \
		exit 1; \
	fi
	@echo "📦 Installing pixi environments and dependencies..."
	@pixi install
	@echo "🔍 Verifying installation..."
	@$(MAKE) env-check
	@echo "✅ Build complete!"

# Environment and compatibility checks
env-check:
	@echo "🔍 Checking environment..."
	@command -v pixi >/dev/null 2>&1 || { echo "❌ pixi not found. Please install pixi first."; exit 1; }
	@echo "✅ pixi found"
	@pixi info >/dev/null 2>&1 || { echo "❌ pixi environment not ready"; exit 1; }
	@echo "✅ pixi environment ready"
	@echo "✅ Environment check complete"

gpu-check:
	@echo "🔍 Checking GPU setup..."
	@$(PIXI_ENV) python scripts/gpu_check.py