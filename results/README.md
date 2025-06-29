# Results Directory

Organized storage for all benchmark results, test outputs, and demonstration data.

## Directory Structure

```
results/
├── README.md                    # This file
├── benchmarks/                  # Benchmark comparison results
│   ├── benchmark_results.json           # Machine-readable benchmark data
│   ├── benchmark_results_table.txt      # Human-readable ASCII table
│   ├── benchmark_results_markdown.md    # Markdown table for docs
│   ├── benchmark_results_terminal.txt   # Terminal display format
│   └── max_whisper_benchmark.json       # Historical MAX-Whisper results
├── demos/                       # Demonstration outputs
│   ├── transcription_examples.txt       # Sample transcription outputs
│   ├── weight_loading_demo.log          # Weight integration demonstrations
│   └── tokenizer_demo_output.txt        # Tokenizer integration examples
└── tests/                       # Test results and validation data
    ├── component_test_results.log       # 4/4 component test outputs
    ├── baseline_comparison.json         # Baseline validation results
    └── performance_validation.txt       # Performance test summaries
```

## File Purposes

### Benchmark Results (`benchmarks/`)
- **Primary results** from comprehensive benchmark comparisons
- **Multiple formats** for different consumption (judges, documentation, analysis)
- **Historical data** for tracking performance improvements

### Demo Outputs (`demos/`)
- **Live demonstration** outputs for judge presentations
- **Example transcriptions** showing quality and accuracy
- **Component demonstrations** proving functionality

### Test Results (`tests/`)
- **Validation outputs** from component and integration tests
- **Performance data** for optimization tracking
- **Baseline comparisons** for fair evaluation

## Usage

### For Judges
```bash
# View latest benchmark results
cat results/benchmarks/benchmark_results_table.txt

# Detailed analysis
cat results/benchmarks/benchmark_results.json | python -m json.tool

# Example outputs
cat results/demos/transcription_examples.txt
```

### For Development
```bash
# Save new benchmark results
./scripts/run_comprehensive_benchmark.sh

# Archive test outputs  
cp test_output.log results/tests/latest_validation.log

# Track performance improvements
git diff results/benchmarks/benchmark_results.json
```

## File Naming Conventions

### Timestamps
- Use ISO format for dated files: `YYYY-MM-DD_HHMMSS`
- Example: `benchmark_results_2025-06-29_210000.json`

### Categories
- `benchmark_*` - Performance comparison data
- `test_*` - Validation and component test results  
- `demo_*` - Demonstration and example outputs
- `validation_*` - Verification and accuracy results

### Formats
- `.json` - Machine-readable data
- `.txt` - Human-readable text/tables
- `.md` - Markdown for documentation
- `.log` - Raw output logs

## Benefits

### Organization
- **Clear separation** of different result types
- **Easy navigation** for judges and developers
- **No clutter** in project root directory

### Judge Experience
- **Quick access** to performance results
- **Multiple formats** for different preferences
- **Historical tracking** of improvements

### Development
- **Version control** of important results
- **Performance tracking** over time
- **Easy comparison** between different runs

## Maintenance

### Automated Generation
Most files are generated automatically by:
- `./scripts/run_comprehensive_benchmark.sh`
- Individual test and demo scripts
- CI/CD pipelines (future)

### Manual Curation
- Archive significant milestone results
- Clean up outdated or redundant files
- Update README when adding new result types

### Git Tracking
- **Include significant results** in version control
- **Exclude large temporary files** via .gitignore
- **Track performance milestones** for project history