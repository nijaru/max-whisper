# Benchmarks Directory

Purpose-driven benchmark scripts for testing and demonstrating MAX-Whisper.

## 🎯 Main Benchmark Scripts

### `benchmark_all_models.py` - **Complete Model Comparison**
- **Purpose**: Compare all speech recognition models with real audio
- **Tests**: OpenAI Whisper, Faster-Whisper, MAX-Whisper (hybrid + trained weights)
- **Output**: Performance table + comprehensive JSON results
- **Usage**: `pixi run -e benchmark python benchmarks/benchmark_all_models.py`
- **Audience**: Judges, stakeholders, performance evaluation

### `quality_assessment.py` - **Output Quality Verification**
- **Purpose**: Verify transcription quality and assess production readiness
- **Tests**: Honest quality comparison with captured text outputs
- **Output**: Quality scores, keyword matching, production readiness assessment
- **Usage**: `pixi run -e benchmark python benchmarks/quality_assessment.py`
- **Audience**: Technical reviewers, quality assurance teams

## 🚀 Specialized Demonstrations

### `max_whisper_showcase.py` - **MAX-Whisper Innovation Demo**
- **Purpose**: Showcase MAX-Whisper's dual approach (hybrid + trained weights)
- **Tests**: Technical breakthrough demonstration with performance narrative
- **Output**: Innovation story with technical achievements
- **Usage**: `pixi run -e benchmark python benchmarks/max_whisper_showcase.py`
- **Audience**: Technical stakeholders, innovation showcase

## 🔧 Development Scripts

### `_dev_quality_tracking.py` - **Development Quality Tracking**
- **Purpose**: Track quality improvements during development (dev use only)
- **Tests**: Before/after comparisons of quality fixes
- **Output**: Progress assessment and recommendations
- **Usage**: `pixi run -e benchmark python benchmarks/_dev_quality_tracking.py`
- **Audience**: Development team (temporary/internal use)

## 📊 Generated Results

All benchmarks save results to `results/benchmarks/`:
- `complete_demo_results.json` - Main demo data
- `benchmark_results_table.txt` - Judge-friendly table
- `verification_results.json` - Technical verification data

## 🎪 Quick Demo Commands

```bash
# 🏆 Complete model comparison (primary demo)
pixi run -e benchmark python benchmarks/benchmark_all_models.py

# 🔍 Quality assessment and verification
pixi run -e benchmark python benchmarks/quality_assessment.py

# 🚀 MAX-Whisper innovation showcase
pixi run -e benchmark python benchmarks/max_whisper_showcase.py
```

## 📁 File Organization

- **benchmark_all_models.py**: Complete model comparison with real audio
- **quality_assessment.py**: Output quality verification and scoring
- **max_whisper_showcase.py**: MAX-Whisper innovation demonstration
- **_dev_quality_tracking.py**: Development quality tracking (internal)
- **README.md**: Benchmark directory documentation