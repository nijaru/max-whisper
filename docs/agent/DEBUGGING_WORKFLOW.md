# Semantic Quality Debugging Workflow

*Step-by-step procedures for systematic MAX Graph semantic quality debugging*

## Overview

This workflow provides systematic procedures for debugging the semantic quality issue in the MAX Graph implementation. Follow these steps to maintain consistency and track progress effectively.

## Before Starting Any Session

### 1. Preparation Checklist
- [ ] Read `DEBUGGING_FINDINGS.md` for latest status
- [ ] Check `PROJECT_STATUS.md` for current phase
- [ ] Use `TodoRead` to see active tasks
- [ ] Ensure test environment is ready: `pixi run -e benchmark demo`

### 2. Session Setup
```bash
# Verify environment
pixi run graph-test
pixi run -e benchmark test-max  # Quick validation

# Set up logging
export DEBUG_SESSION="$(date +%Y%m%d_%H%M)"
echo "Starting debugging session: $DEBUG_SESSION"
```

## Phase 1: Feature Analysis Workflow

### Step 1: Extract Encoder Features
For systematic comparison, extract features from all implementations:

```bash
# Extract features from each implementation
pixi run -e benchmark extract-features --impl cpu --output features_cpu.json
pixi run -e benchmark extract-features --impl gpu --output features_gpu.json  
pixi run -e benchmark extract-features --impl max --output features_max.json
```

### Step 2: Numerical Comparison
Compare features quantitatively:

```bash
# Run comparison analysis
pixi run -e benchmark compare-features --baseline cpu --target max --output comparison.json

# Generate analysis report
pixi run -e benchmark analyze-divergence --comparison comparison.json
```

### Step 3: Identify Divergence Points
Look for:
- Layer where significant differences begin
- Operations with largest numerical differences  
- Patterns in the divergence (gradual vs sudden)

### Step 4: Document Findings
Update `DEBUGGING_FINDINGS.md` with:
- Session date and objective
- Numerical results (L2 norms, cosine similarities)
- Identified divergence layers/operations
- Hypotheses for next investigation

## Phase 2: Precision Debugging Workflow

### Step 1: Isolate Operations
Test individual operations in isolation:

```python
# Example: Test attention operation precision
def test_attention_precision():
    # Extract attention inputs from both implementations
    # Run attention operation separately
    # Compare outputs numerically
    pass
```

### Step 2: Fix and Validate Loop
For each identified issue:

1. **Create specific todo**: `TodoWrite` with precise task
2. **Implement fix**: Make targeted changes
3. **Test immediately**: Run feature comparison
4. **Document result**: Update `DEBUGGING_FINDINGS.md`
5. **Validate end-to-end**: Test full transcription

### Step 3: Cumulative Testing
After each fix:
```bash
# Test the fix doesn't break anything
pixi run test

# Check end-to-end performance  
pixi run -e benchmark test-max

# Compare features again
pixi run -e benchmark compare-features --update-baseline
```

## Phase 3: Validation Workflow

### Step 1: Comprehensive Testing
Test across multiple scenarios:

```bash
# Test different audio samples
pixi run -e benchmark test-max --audio audio_samples/sample1.wav
pixi run -e benchmark test-max --audio audio_samples/sample2.wav
pixi run -e benchmark test-max --audio audio_samples/sample3.wav

# Test different model sizes
pixi run -e benchmark test-max --model-size tiny
pixi run -e benchmark test-max --model-size base
```

### Step 2: Performance Validation
Ensure fixes don't hurt performance:

```bash
# Full benchmark comparison
pixi run -e benchmark benchmark-json --output final_results.json

# Analyze performance impact
pixi run -e benchmark analyze-performance --results final_results.json
```

## Session Management

### During Each Session

1. **Set clear objective**: What specific thing are you investigating?
2. **Create todos**: Use `TodoWrite` for specific tasks
3. **Track attempts**: Document what you try in `DEBUGGING_FINDINGS.md`
4. **Update status**: Mark todos complete as you finish them
5. **Document findings**: Even failed attempts are valuable

### Session Documentation Template

Add to `DEBUGGING_FINDINGS.md`:

```markdown
### Session: [DATE] - [OBJECTIVE]

**Hypothesis**: [What you think is wrong]
**Approach**: [What you're trying]
**Commands Run**:
- `command 1`
- `command 2`

**Results**: [What happened]
**Key Findings**: [What you learned]
**Todos Created**: [List specific todos added]
**Next Steps**: [What to investigate next]
```

### End of Session Checklist
- [ ] Update `DEBUGGING_FINDINGS.md` with session results
- [ ] Update todos with current status (`TodoWrite`)
- [ ] Commit any code changes with clear messages
- [ ] Update `PROJECT_STATUS.md` if phase changes
- [ ] Note any blockers or new issues discovered

## Tools and Commands Reference

### Feature Extraction
```bash
# Extract encoder features at each layer
pixi run -e benchmark extract-features --impl [cpu|gpu|max] --layers all

# Extract specific layer features
pixi run -e benchmark extract-features --impl max --layer 2 --operation attention
```

### Numerical Analysis
```bash
# Compare two feature sets
pixi run -e benchmark compare-features --baseline features1.json --target features2.json

# Analyze attention patterns
pixi run -e benchmark analyze-attention --features features_max.json

# Plot feature distributions
pixi run -e benchmark plot-features --features features_max.json --output plots/
```

### Validation
```bash
# Quick semantic quality test
pixi run -e benchmark test-semantic-quality --impl max

# Performance benchmark
pixi run -e benchmark benchmark --impl max --output performance.json

# Cross-validation with multiple audio files
pixi run -e benchmark validate-robustness --impl max --audio-dir audio_samples/
```

## Debugging Principles

### 1. Be Systematic
- Test one thing at a time
- Always compare against working baseline
- Document negative results (what doesn't work)

### 2. Use Quantitative Analysis
- Measure differences numerically (L2 norm, cosine similarity)
- Plot distributions and patterns
- Set clear thresholds for "close enough"

### 3. Isolate Problems
- Test operations individually before integration
- Use minimal test cases first
- Build up complexity gradually

### 4. Maintain Reproducibility
- Save all intermediate results
- Document exact commands used
- Version control all changes

### 5. Track Progress Carefully
- Update todos frequently
- Document failed attempts
- Note patterns and insights

## Common Pitfalls to Avoid

- **Changing multiple things at once**: Makes it hard to isolate what works
- **Not documenting failures**: Failed attempts provide valuable information
- **Skipping validation**: Always test that fixes don't break other things
- **Ignoring performance**: Fixes shouldn't significantly hurt speed
- **Not updating documentation**: Others (including future you) need context

This workflow ensures systematic progress toward fixing the semantic quality issue while maintaining good development practices.