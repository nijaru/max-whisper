# Phase 1: Encoder Quality Matching - Implementation Tasks

*Goal: Achieve 2035 character output matching CPU baseline quality*

## Current Status
- **Baseline**: CPU produces 2035 chars: "Music Max provides several different libraries..."
- **Hybrid**: Currently produces 422 chars: "Max provides several different libraries..."  
- **Gap**: 1613 missing characters (4.8x improvement needed)

## Root Cause Analysis

### Issue 1: Early Stopping from Repetition Detection
**Location**: `max-whisper/whisper_max.py` - repetition cleaning logic
**Current Behavior**: Generation stops at 422 chars when repetition detected
**Investigation Needed**: 
- Where exactly does repetition detection trigger?
- What patterns cause early termination?
- Can we adjust thresholds or disable temporarily?

### Issue 2: Feature Distribution Mismatch  
**Location**: `max-whisper/whisper_max.py:1293` - feature processing
**Current State**: MAX Graph std: 1.447 → 1.133 (30% normalization)
**Target State**: OpenAI std: 0.400
**Investigation Needed**:
- Test stronger normalization (50%, 70%, 90%)
- Compare raw vs processed feature distributions
- Validate semantic preservation with aggressive normalization

### Issue 3: Decoder Parameter Optimization
**Current Settings**: Temperature 0.3, beam_size 5, sample_len 1000
**Investigation Needed**:
- Test CPU baseline decoder parameters
- Compare temperature settings (0.0, 0.1, 0.3, 0.7)
- Analyze beam search vs sampling strategies

## Implementation Tasks

### Task 1: Repetition Detection Analysis 🔬
**Priority**: HIGH - Likely main blocker

**Steps**:
1. **Locate repetition cleaning code** in `max-whisper/whisper_max.py`
2. **Add debug logging** to show when/why cleaning triggers
3. **Test with repetition cleaning disabled** - see raw output length
4. **Analyze repetition patterns** - what content triggers cleaning?
5. **Adjust thresholds** or **modify detection logic** to allow longer generation

**Expected Outcome**: Raw generation length > 422 chars without cleaning

### Task 2: Feature Normalization Optimization 🎛️
**Priority**: HIGH - Core quality issue

**Steps**:
1. **Test stronger normalization**: 50%, 70%, 90% toward OpenAI distribution  
2. **Compare feature statistics**: mean, std, range across normalization levels
3. **Validate content quality**: ensure semantic accuracy preserved  
4. **Test multiple audio samples**: verify consistency across inputs
5. **Document optimal normalization strength** for production use

**Expected Outcome**: Optimal normalization achieving std ≈ 0.400 with quality preserved

### Task 3: Decoder Parameter Tuning ⚙️
**Priority**: MEDIUM - May help extend generation

**Steps**:
1. **Extract CPU baseline parameters** from whisper_cpu.py
2. **Test parameter combinations**:
   - Temperature: [0.0, 0.1, 0.3, 0.7]
   - Sample length: [1000, 2000, 5000] 
   - Beam size: [1, 3, 5]
3. **Measure output length and quality** for each combination
4. **Compare against CPU baseline** results
5. **Document optimal parameters** for hybrid approach

**Expected Outcome**: Parameter set achieving maximum quality/length

### Task 4: Content Validation Framework 📊
**Priority**: MEDIUM - Ensure quality preservation

**Steps**:
1. **Create automated comparison** between hybrid and CPU outputs
2. **Implement content similarity metrics** (BLEU, semantic similarity)
3. **Test across multiple audio samples** (short, medium, long)
4. **Track regression/improvement** as changes are made
5. **Document quality benchmarks** for validation

**Expected Outcome**: Automated quality validation pipeline

### Task 5: Raw Output Investigation 🔍
**Priority**: HIGH - Understand current limitations

**Steps**:
1. **Disable all post-processing** (repetition cleaning, truncation)
2. **Capture raw decoder output** before any modifications
3. **Analyze raw content quality and length**
4. **Compare raw vs cleaned outputs** 
5. **Identify processing steps** that improve vs hurt quality

**Expected Outcome**: Understanding of where quality is lost in pipeline

## Success Metrics

### Intermediate Milestones
- [ ] **Raw output > 422 chars** (repetition cleaning bypassed)
- [ ] **Feature std ≈ 0.400** (normalization optimized)  
- [ ] **Content similarity > 90%** (quality preserved)
- [ ] **Multiple samples tested** (consistency validated)

### Final Success Criteria
- [ ] **Output length: 2035 characters** (100% coverage)
- [ ] **Content accuracy: Exact match** with CPU baseline
- [ ] **Performance maintained: ≤ 1.4s** (no regression)
- [ ] **Robust across samples** (consistent quality)

## Implementation Order

1. **Task 5 (Raw Output)** - Understand current state without processing
2. **Task 1 (Repetition)** - Remove main blocker to length
3. **Task 2 (Normalization)** - Fix core feature quality issue  
4. **Task 3 (Parameters)** - Optimize decoder settings
5. **Task 4 (Validation)** - Ensure quality preservation

## Code Locations for Investigation

```
max-whisper/whisper_max.py:
- Line ~1293: Feature post-processing logic
- Line ~1330: Decoder parameter settings  
- Search: "repetition", "cleaning" - Repetition detection code
- Search: "normalization_strength" - Current 30% setting

max-whisper/whisper_cpu.py:
- Decoder parameters and settings for comparison
- Reference implementation behavior

Validation:
- Create test scripts comparing outputs
- Automated quality metrics
```

## Next Actions

1. **Start with Task 5** - Understand raw decoder output without processing
2. **Identify repetition detection** - Find exact code causing 422 char limit
3. **Test aggressive normalization** - Move closer to OpenAI feature distribution  
4. **Validate against CPU baseline** - Ensure content matches exactly

**Timeline**: Complete Phase 1 tasks to achieve 2035 character output matching CPU baseline quality.