# Benchmarks Directory

Organized benchmark scripts for different purposes and audiences.

## 🎯 Main Demo Scripts

### `complete_demo.py` - **Primary Judge Demo**
- **Purpose**: Complete demonstration for hackathon judges
- **Tests**: All 3 pipelines (OpenAI, Faster-Whisper, MAX-Whisper) with real audio
- **Output**: Judge-friendly results table + comprehensive JSON
- **Usage**: `pixi run -e benchmark python benchmarks/complete_demo.py`
- **Audience**: Judges, stakeholders, demo presentations

### `verification_benchmark.py` - **Technical Verification**
- **Purpose**: Honest performance and quality verification
- **Tests**: Detailed analysis with actual outputs captured
- **Output**: Verification results with quality assessments
- **Usage**: `pixi run -e benchmark python benchmarks/verification_benchmark.py`
- **Audience**: Technical reviewers, performance validation

## 🚀 Specialized Demos

### `final_phase4_complete.py` - **Phase 4 Story**
- **Purpose**: Shows Phase 4A + 4B development story
- **Tests**: Trained weights breakthrough + hybrid production approach
- **Output**: Achievement narrative with technical details
- **Usage**: `pixi run -e benchmark python benchmarks/final_phase4_complete.py`
- **Audience**: Technical stakeholders interested in development journey

### `phase4_quality_progress.py` - **Quality Tracking**
- **Purpose**: Track quality improvements during Phase 4 development
- **Tests**: Before/after comparisons of quality fixes
- **Output**: Progress assessment and recommendations
- **Usage**: `pixi run -e benchmark python benchmarks/phase4_quality_progress.py`
- **Audience**: Development team, quality assurance

## 📊 Generated Results

All benchmarks save results to `results/benchmarks/`:
- `complete_demo_results.json` - Main demo data
- `benchmark_results_table.txt` - Judge-friendly table
- `verification_results.json` - Technical verification data

## 🎪 Quick Demo Commands

```bash
# 🏆 Main judge demo (5 minutes)
pixi run -e benchmark python benchmarks/complete_demo.py

# 🔍 Technical verification (detailed)
pixi run -e benchmark python benchmarks/verification_benchmark.py

# 🚀 Phase 4 development story
pixi run -e benchmark python benchmarks/final_phase4_complete.py
```

## 📁 File Organization

- **complete_demo.py**: Comprehensive, judge-friendly
- **verification_benchmark.py**: Technical, honest assessment
- **final_phase4_complete.py**: Development narrative
- **phase4_quality_progress.py**: Quality tracking
- **README.md**: This organization guide