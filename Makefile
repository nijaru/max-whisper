# MAX Graph Whisper
# Professional speech recognition with MAX Graph acceleration

# Configuration
MODEL_SIZE ?= tiny
AUDIO_FILE ?= audio_samples/modular_video.wav
PIXI_ENV = pixi run -e benchmark

# Environment check helper
define check_env
	@$(MAKE) verify >/dev/null 2>&1 || { echo "❌ Environment not ready. Run 'make install' first."; exit 1; }
endef

# Suppress make directory messages
MAKEFLAGS += --no-print-directory

# Extract positional arguments from command line
ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
MODEL_ARG := $(if $(filter tiny small base,$(ARGS)),$(filter tiny small base,$(ARGS)),$(MODEL_SIZE))
AUDIO_ARG := $(if $(filter-out tiny small base,$(ARGS)),$(filter-out tiny small base,$(ARGS)),$(AUDIO_FILE))

# Define phony targets
.PHONY: help demo benchmark test clean install verify results
.PHONY: test-cpu test-gpu test-max benchmark-json debug-encoder debug-features

# Default target - show help
all: help

help:
	@echo "# MAX Graph Whisper"
	@echo "High-performance speech recognition with MAX Graph acceleration"
	@echo ""
	@echo "## Quick Start"
	@echo "  make install          # Setup environment"
	@echo "  make demo             # Interactive comparison demo"
	@echo ""
	@echo "## Core Commands"
	@echo "  demo                  # Side-by-side comparison with live metrics"
	@echo "  benchmark             # Structured performance analysis"
	@echo "  benchmark-json        # JSON output for analysis"
	@echo "  results               # View historical results"
	@echo ""
	@echo "## Individual Testing"
	@echo "  test-cpu              # Test CPU baseline"
	@echo "  test-gpu              # Test GPU acceleration"
	@echo "  test-max              # Test MAX Graph hybrid"
	@echo "  test                  # Run test suite"
	@echo ""
	@echo "## Development"
	@echo "  install               # Setup development environment"
	@echo "  verify                # Verify MAX Graph and CUDA setup"
	@echo "  clean                 # Clean build artifacts"
	@echo "  debug-encoder         # Debug encoder feature extraction"
	@echo "  debug-features        # Compare feature distributions"
	@echo ""
	@echo "## Examples"
	@echo "  make install && make demo    # Complete setup and demo"
	@echo "  make test-max               # Test MAX Graph implementation only"
	@echo "  make benchmark-json         # Get benchmark data in JSON format"

# Interactive demo
demo:
	$(call check_env)
	@echo "🚀 Starting interactive comparison demo..."
	@$(PIXI_ENV) python scripts/tui_demo.py

# Individual implementation tests
test-cpu:
	$(call check_env)
	@echo "🔧 Testing CPU baseline..."
	@$(PIXI_ENV) python max-whisper/whisper_cpu.py --model-size $(MODEL_SIZE)

test-gpu:
	$(call check_env)
	@echo "⚡ Testing GPU acceleration..."
	@$(PIXI_ENV) python max-whisper/whisper_gpu.py --model-size $(MODEL_SIZE)

test-max:
	$(call check_env)
	@echo "🎯 Testing MAX Graph hybrid..."
	@$(PIXI_ENV) python max-whisper/whisper_max.py --model-size $(MODEL_SIZE)

# Test suite
test:
	$(call check_env)
	@echo "🧪 Running test suite..."
	@$(PIXI_ENV) python -m pytest test/ -v

# Benchmarking
benchmark:
	$(call check_env)
	@echo "📊 Running structured performance analysis..."
	@$(PIXI_ENV) python benchmarks/benchmark_runner.py

benchmark-json:
	$(call check_env)
	@echo "📊 Running benchmark with JSON output..."
	@$(PIXI_ENV) python benchmarks/benchmark_runner.py --json-output

results:
	$(call check_env)
	@echo "📈 Viewing historical results..."
	@$(PIXI_ENV) python benchmarks/results_tracker.py

# Development and debugging
debug-encoder:
	$(call check_env)
	@echo "🔍 Debugging encoder feature extraction..."
	@$(PIXI_ENV) python benchmarks/research/encoder_feature_debug.py

debug-features:
	$(call check_env)
	@echo "🔍 Comparing feature distributions..."
	@$(PIXI_ENV) python benchmarks/research/simple_feature_comparison.py

# Clean up build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f *.pyc *.json
	rm -rf __pycache__
	rm -rf max-whisper/__pycache__
	rm -rf benchmarks/__pycache__
	rm -rf scripts/__pycache__
	rm -rf test/__pycache__
	rm -rf .pytest_cache
	rm -rf test/output/*
	rm -f *_debug.json *_results.json
	@echo "✅ Cleanup complete"

# Installation and setup
install:
	@echo "🚀 MAX Graph Whisper - Setting up development environment..."
	@echo ""
	@echo "📦 Installing pixi package manager..."
	@if command -v pixi >/dev/null 2>&1; then \
		echo "✅ pixi already installed"; \
	else \
		echo "📥 Installing pixi..."; \
		curl -fsSL https://pixi.sh/install.sh | bash; \
		echo "✅ pixi installed"; \
		echo "⚠️  Please restart your shell or run: source ~/.bashrc"; \
		echo "⚠️  Then run 'make install' again to continue setup"; \
		exit 0; \
	fi
	@echo ""
	@echo "📦 Installing project dependencies..."
	@pixi install
	@echo ""
	@echo "🔍 Verifying installation..."
	@$(MAKE) verify
	@echo ""
	@echo "🎉 Installation complete!"
	@echo "Try: make demo"

verify:
	@echo "🔍 Verifying MAX Graph and CUDA setup..."
	@command -v pixi >/dev/null 2>&1 || { echo "❌ pixi not found"; exit 1; }
	@echo "✅ pixi found"
	@pixi info >/dev/null 2>&1 || { echo "❌ pixi environment not ready"; exit 1; }
	@echo "✅ pixi environment ready"
	@$(PIXI_ENV) python -c "import max.graph; print('✅ MAX Graph available')" 2>/dev/null || echo "⚠️ MAX Graph not available"
	@$(PIXI_ENV) python -c "import torch; print(f'✅ PyTorch {torch.__version__} available')" 2>/dev/null || echo "❌ PyTorch not available"
	@$(PIXI_ENV) python -c "import torch; print('✅ CUDA available' if torch.cuda.is_available() else '⚠️ CUDA not available')" 2>/dev/null || echo "❌ CUDA check failed"