# Contract & Documentation Decisions

This document records explicit decisions made during Phase 10 refactoring to resolve contract/implementation discrepancies.

---

## Decision 1: Transaction Deduplication Option vs. Contract

**Issue**: The DATA_CONTRACT.md states "No automatic deduplication: User must preprocess if deduplication needed" and "Exact duplicates: Rows identical on all columns... are preserved". However, the implementation provides:
- `drop_exact_duplicates` parameter in `prepare()`
- CLI flag `--drop-exact-duplicates`
- `clean.unique(subset=exact_source_cols, maintain_order=True)` when enabled

**Decision**: Keep the `drop_exact_duplicates` option as an explicit opt-in feature. Update documentation to clarify:
- Default behavior (preserved) matches contract
- Option is available for users who need deduplication
- Accounting note: `valid_rows` is calculated BEFORE duplicate removal, so audit counts reflect pre-deduplication state

**Action Items**:
- [ ] Update DATA_CONTRACT.md "Duplicate Handling" section to mention the opt-in flag
- [ ] Add note about accounting timing (valid_rows calculated before dedup)

---

## Decision 2: Silent Transition-Report Failure

**Issue**: In `reporting.py` line 93, `except Exception: continue` hides all errors during transition computation, including corrupt files, schema errors, and computation failures.

**Decision**: Distinguish between expected missing inputs and actual errors:
- `FileNotFoundError` / empty results: Continue silently (expected)
- All other exceptions: Log warning and re-raise, or at minimum log the error type/message

**Action Items**:
- [ ] Replace broad `except Exception` with specific exception handling
- [ ] Add logging for unexpected errors
- [ ] Ensure dashboard worker and patronage cleanup patterns are consistent

---

## Decision 3: Timezone Handling Documentation Contradiction

**Issue**: 
- Line 28: "Timezone-aware Datetime input is supported when `--timezone` parameter is provided... the datetime is converted to the specified timezone before extracting the date component."
- Line 612: "Timezone naive: Datetimes truncated to date; no TZ conversion."

**Decision**: Line 28 correctly describes the implementation. Line 612 is outdated/wrong. Update Line 612 to reflect actual behavior.

**Action Items**:
- [ ] Update DATA_CONTRACT.md line 612 to: "Timezone handling: Timezone-aware Datetime input is converted to the specified `--timezone` before date extraction; naive Datetimes are used as-is."

---

## Decision 4: Quantile Interpolation Documentation Contradiction

**Issue**: 
- Line 597: "Quantile interpolation: Polars default (nearest)"
- Code (thresholds.py line 167): Uses `interpolation="linear"` explicitly

**Decision**: Code is correct (uses linear interpolation for distribution-free CI). Update documentation to match.

**Action Items**:
- [ ] Update DATA_CONTRACT.md line 597 to: "Quantile interpolation: Linear — used for distribution-free confidence intervals (see thresholds.py::_quantile_with_ci)"

---

## Decision 5: Quantile Metadata in Threshold Output

**Issue**: The analysis mentions "Linear threshold interpolation metadata versus 'Polars default (nearest)'". The threshold output should include the interpolation method used.

**Decision**: Add `quantile_interpolation: "linear"` to threshold output manifest.

**Action Items**:
- [ ] Add interpolation metadata to `write_thresholds()` output
- [ ] Update DATA_CONTRACT.md threshold schema to include this field

---

## Summary of Required Changes

| Decision | Files to Update |
|----------|----------------|
| 1. Duplicate handling | DATA_CONTRACT.md, prepare.py docstring |
| 2. Silent failures | reporting.py |
| 3. Timezone docs | DATA_CONTRACT.md |
| 4. Quantile interpolation docs | DATA_CONTRACT.md |
| 5. Quantile metadata | thresholds.py (write_thresholds), DATA_CONTRACT.md |

---

## Implementation Order

These are behavioral/interface decisions requiring stakeholder alignment. Implement after Phase 1-9 refactoring is complete and tested.