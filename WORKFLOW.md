# MAX-Whisper Development Workflow

**Quick reference for maintaining organized development**

## 🔄 Before Starting Any Work

1. **Read STATUS.md** - Understand current state
2. **Check benchmarks/results.md** - Know current performance
3. **Identify priority** - Focus on STATUS.md next priorities

## 📝 After Any Major Change

### ✅ Required Updates
1. **Update STATUS.md** with:
   - What changed
   - Current working state
   - New capabilities/issues
   - Next priorities

2. **Update README.md** if:
   - User-facing instructions change
   - Project status changes significantly
   - Essential files change

3. **Run benchmark** and verify results.md is current

### 🗃️ File Organization Checklist

**❌ DON'T:**
- Create multiple benchmark scripts
- Leave old/unused files cluttering directories
- Create generic filenames like `test1.py`, `benchmark_new.py`
- Accumulate multiple results files

**✅ DO:**
- Use single `benchmarks/benchmark.py`
- Archive old files to `archive/` with date prefix
- Give files clear, descriptive names
- Keep one current `benchmarks/results.md`

## 🎯 Major Change Definition

Update STATUS.md when any of these occur:
- ✅ New feature working
- ❌ Something breaks that was working
- 🔧 Performance improvement/regression
- 📁 File organization changes
- 🎯 Status transitions (in-progress → complete, etc.)

## 📊 Quick Status Template

```markdown
### YYYY-MM-DD HH:MM - Brief Description
- **Changed**: [what was modified]
- **Now Working**: [what capabilities exist]
- **Next Priority**: [what needs to be done next]
- **Blockers**: [what's preventing progress]
```

## 🧹 End-of-Session Cleanup

1. Archive any temporary/experimental files
2. Update STATUS.md with session results
3. Clean up benchmarks/results.md if multiple runs created clutter
4. Commit with clear message referencing STATUS.md update

---

**Remember**: STATUS.md is the single source of truth. When in doubt, check STATUS.md first!