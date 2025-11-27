# ProperImage Baseline Performance Metrics

**Date:** November 27, 2025
**Branch:** improvements/optimize
**Hardware:** Apple Silicon (assumed M-series)
**Python:** 3.13
**Test Setup:** 256x256 images, 30 simulated sources, 15x15 stamp size

## Executive Summary

The `subtract()` function shows significant performance variation based on configuration:

- **Fastest:** `beta_only` or `iterative_beta` (~0.24-0.28s)
- **Slowest:** `beta_shift` or `iterative_full` (~3.37-3.41s)
- **Performance gap:** ~12-14x slower when shift optimization is enabled

## Configuration Performance Comparison

| Configuration | Time (s) | Relative Speed | Use Case |
|--------------|----------|----------------|----------|
| `beta_only` | 0.2802 | 1.00x (baseline) | Simple flux scaling |
| `iterative_beta` | 0.2404 | **0.86x (fastest)** | Iterative flux refinement |
| `beta_shift` | 3.4094 | 12.17x slower | Flux + alignment |
| `iterative_full` | 3.3870 | 12.09x slower | Full iterative w/ shift |

### Configuration Details

```python
beta_only = {
    "beta": True,
    "shift": False,
    "iterative": False
}

beta_shift = {
    "beta": True,
    "shift": True,
    "iterative": False
}

iterative_beta = {
    "beta": True,
    "shift": False,
    "iterative": True
}

iterative_full = {
    "beta": True,
    "shift": True,
    "iterative": True
}
```

## Detailed Profiling Results (beta_only)

Top time-consuming operations by cumulative time:

| Function | Cumulative | % of Total | Calls | Per Call |
|----------|-----------|------------|-------|----------|
| `subtract()` | 0.316s | 100% | 1 | 0.316s |
| `get_variable_psf()` | 0.190s | 60.1% | 2 | 0.095s |
| `fit_gaussian2d()` | 0.153s | 48.4% | 23 | 0.007s |
| Astropy fitting | 0.137s | 43.4% | 23 | 0.006s |
| scipy.leastsq | 0.107s | 33.9% | 23 | 0.005s |
| FFT operations | 0.066s | 20.9% | 59 | 0.001s |
| `least_squares` (beta) | 0.076s | 24.1% | 1 | 0.076s |
| Cost function `F()` | 0.072s | 22.8% | 22 | 0.003s |

### Key Bottlenecks Identified

1. **PSF Measurement (60% of time)**
   - Gaussian fitting dominates: 48.4% of total time
   - 23 Gaussian fits required (one per source)
   - Each fit uses scipy's Levenberg-Marquardt (`leastsq`)

2. **FFT Operations (21% of time)**
   - 59 FFT calls total
   - Each beta optimization iteration requires multiple FFTs
   - Called inside cost function (no caching)

3. **Beta Optimization (24% of time)**
   - Uses `scipy.optimize.least_squares` with TRF method
   - 22 function evaluations for cost function `F()`
   - Each evaluation computes FFTs from scratch

4. **Deep Copying (10% of time)**
   - 27,072 deepcopy calls
   - Likely from Astropy model copying

## Time Distribution

```
Total subtract() time: ~0.30s
├── PSF measurement: ~0.19s (63%)
│   ├── Gaussian fitting: ~0.15s (50%)
│   └── Stamp extraction: ~0.04s (13%)
├── Beta optimization: ~0.08s (27%)
│   ├── Cost function evals: ~0.07s (23%)
│   └── FFTs: ~0.07s (23%)
└── Other operations: ~0.03s (10%)
```

## Shift Parameter Impact

**Without shift:** 0.24-0.28s
**With shift:** 3.37-3.41s
**Slowdown:** ~12x

### Why is shift so expensive?

The shift optimization adds 2 additional parameters (dx, dy) to optimize:
- `beta_only`: 1D optimization (beta)
- `beta_shift`: 3D optimization (beta, dx, dy)

This causes:
1. More function evaluations (optimization in 3D space)
2. More FFT computations per iteration
3. Longer convergence time

## Optimization Opportunities

Based on profiling results, ranked by potential impact:

### 1. **Cache FFT Results** (HIGH IMPACT)
- **Current:** FFTs computed inside cost function on every iteration
- **Impact:** Could reduce FFT time by 50-80%
- **Effort:** Low
- **Files:** `operations.py` lines ~289

### 2. **Use Analytical Gradients** (HIGH IMPACT)
- **Current:** Using numerical jacobians (3-point)
- **Impact:** Reduce function evaluations by 3-4x
- **Effort:** Medium (JAX implementation)
- **Files:** `operations.py` optimization calls

### 3. **Optimize Shift Parameter Search** (HIGH IMPACT for beta_shift)
- **Current:** Full 3D optimization
- **Impact:** Could reduce shift optimization time by 5-10x
- **Options:**
  - Use coarser grid search first
  - Separate beta and shift optimization
  - Use differential evolution with parallelization
- **Effort:** Medium
- **Files:** `operations.py` lines 200-400

### 4. **Speed Up PSF Measurement** (MEDIUM IMPACT)
- **Current:** 23 separate Gaussian fits
- **Impact:** Reduce PSF time by 20-40%
- **Options:**
  - Vectorize fitting across sources
  - Use faster initial guesses
  - Cache repeated computations
- **Effort:** Medium
- **Files:** `utils.py:395`, `single_image.py:1469`

### 5. **Reduce Deep Copying** (LOW-MEDIUM IMPACT)
- **Current:** 27,072 deepcopy calls
- **Impact:** Save 5-10% of total time
- **Effort:** Low-Medium
- **Files:** Astropy model usage

### 6. **JIT Compilation with Numba** (MEDIUM IMPACT)
- **Current:** Pure Python loops
- **Impact:** 2-5x speedup for hot loops
- **Effort:** Medium
- **Files:** Cost functions, array operations

## Recommendations

### Phase 1: Quick Wins (1-2 days)
1. Cache FFT results outside cost function
2. Profile shift optimization separately to understand the 12x slowdown
3. Add timing instrumentation to beta/shift optimization loops

### Phase 2: Major Optimizations (1-2 weeks)
1. Implement JAX for automatic differentiation
2. Optimize shift parameter search strategy
3. Vectorize PSF fitting

### Phase 3: Advanced (2-4 weeks)
1. Numba JIT compilation for hot loops
2. Parallel function evaluation (`workers=-1`)
3. Investigate alternative optimization algorithms

## Next Steps

1. **Instrument shift optimization** to understand why it's 12x slower
2. **Implement FFT caching** as proof of concept
3. **Re-profile** to measure improvement
4. **Document speedup gains** using `compare_implementations()`

## Profiling Commands

```bash
# Basic timing
python examples/profile_subtract.py basic

# Detailed cProfile
python examples/profile_subtract.py detailed

# Configuration comparison
python examples/profile_subtract.py config

# Multi-run benchmark
python examples/profile_subtract.py benchmark

# All modes
python examples/profile_subtract.py
```

## Notes

- Measurements include logging overhead (INFO level)
- Times may vary ±10% between runs
- PSF measurement time depends on number of sources
- Optimization time depends on convergence criteria
