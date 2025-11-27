# ProperImage Profiling Tools

This directory contains profiling and benchmarking utilities for analyzing ProperImage performance.

## Quick Start

### Basic profiling example:

```python
from properimage.profiling import timer, profile_code

# Time a code block
with timer("My operation"):
    result = expensive_operation()

# Detailed profiling with cProfile
with profile_code("My function", sort_by="cumulative", top_n=20):
    result = expensive_operation()
```

### Run the subtract() profiling example:

```bash
# Run all profiling modes
python examples/profile_subtract.py

# Run specific mode
python examples/profile_subtract.py basic      # Basic timing
python examples/profile_subtract.py detailed   # Detailed cProfile
python examples/profile_subtract.py config     # Compare configurations
python examples/profile_subtract.py benchmark  # Multiple runs
```

## Profiling Module (`properimage.profiling`)

### Context Managers

#### `timer(name, verbose=True)`
Simple timing context manager.

```python
from properimage.profiling import timer

with timer("Image loading") as t:
    data = load_image()

print(f"Took {t['elapsed']:.4f} seconds")
```

#### `profile_code(name, sort_by='cumulative', top_n=20, verbose=True)`
Detailed profiling using cProfile.

```python
from properimage.profiling import profile_code

with profile_code("Subtract operation", sort_by="time", top_n=30):
    result = subtract(ref, new)
```

### Decorators

#### `@time_function`
Decorator to time a function.

```python
from properimage.profiling import time_function

@time_function
def my_slow_function():
    # expensive operation
    pass
```

#### `@profile_function(sort_by='cumulative', top_n=20)`
Decorator to profile a function.

```python
from properimage.profiling import profile_function

@profile_function(sort_by='time', top_n=15)
def my_function():
    # expensive operation
    pass
```

### Metrics Collection

#### `MetricsCollector`
Collect and aggregate metrics across multiple runs.

```python
from properimage.profiling import MetricsCollector

collector = MetricsCollector()

for i in range(10):
    with collector.time("operation1"):
        do_something()

    with collector.time("operation2"):
        do_something_else()

collector.print_summary()
```

### Benchmarking

#### `benchmark_function(func, *args, n_runs=10, warmup=2, **kwargs)`
Benchmark a function over multiple runs.

```python
from properimage.profiling import benchmark_function

def my_func(x, y):
    return expensive_computation(x, y)

metrics = benchmark_function(my_func, x, y, n_runs=100, warmup=5)
print(metrics)
```

#### `compare_implementations(implementations, *args, n_runs=10, **kwargs)`
Compare multiple implementations.

```python
from properimage.profiling import compare_implementations

def old_method(data):
    return process_old_way(data)

def new_method(data):
    return process_new_way(data)

results = compare_implementations(
    {"old": old_method, "new": new_method},
    test_data,
    n_runs=50
)

# Prints comparison and speedup metrics
```

## Data Classes

### `TimingMetrics`
Stores timing statistics:
- `name`: Operation name
- `total_time`: Total time across all runs
- `call_count`: Number of calls
- `mean_time`: Average time per call
- `std_time`: Standard deviation
- `min_time`: Minimum time
- `max_time`: Maximum time
- `times`: List of individual timings

### `SubtractMetrics`
Stores comprehensive metrics for `subtract()`:
- `total_time`: Total operation time
- `fft_time`: Time spent in FFT operations
- `optimization_time`: Time in optimization
- `beta_optimization_time`: Time optimizing beta
- `shift_optimization_time`: Time optimizing shift
- `iterations`: Number of iterations
- `function_evaluations`: Total function calls
- `success`: Whether optimization succeeded
- `optimization_method`: Method used
- `final_cost`: Final cost function value

## Usage Patterns

### 1. Profile a specific operation

```python
from properimage.profiling import timer

with timer("PSF measurement"):
    psf = single_image.get_psf()
```

### 2. Compare before/after optimization

```python
from properimage.profiling import compare_implementations

results = compare_implementations(
    {
        "old_subtract": old_subtract_function,
        "new_subtract": new_optimized_subtract,
    },
    ref_image, new_image,
    n_runs=20
)
```

### 3. Track metrics across workflow

```python
from properimage.profiling import MetricsCollector

collector = MetricsCollector()

for image_pair in image_pairs:
    with collector.time("load_images"):
        ref, new = load_images(image_pair)

    with collector.time("psf_measurement"):
        ref_psf = ref.get_psf()
        new_psf = new.get_psf()

    with collector.time("subtract"):
        result = subtract(ref, new)

collector.print_summary()
```

### 4. Detailed bottleneck analysis

```python
from properimage.profiling import profile_code

with profile_code("Subtract detailed", sort_by="cumulative", top_n=50):
    result = subtract(
        ref, new,
        iterative=True,
        beta=True,
        shift=True
    )
```

## Interpreting Results

### cProfile output
- **cumulative**: Total time spent in function and all subfunctions
- **time**: Time spent in function only (excluding subfunctions)
- **calls**: Number of times function was called

### Common bottlenecks to look for:
1. FFT operations (`_ifftwn`, `fftn`, `ifftn`)
2. Optimization loops (`least_squares`, `minimize_scalar`)
3. Cost function evaluations
4. Array operations and memory allocations

### Performance tips:
- Look for functions with high cumulative time and many calls
- Check for redundant FFT computations
- Identify opportunities for caching
- Find loops that could be vectorized or parallelized

## Next Steps

After profiling, use the metrics to:
1. Identify performance bottlenecks
2. Implement optimizations (JAX, Numba, caching, etc.)
3. Re-profile to measure improvements
4. Document speedups and trade-offs

## Example Output

```
========================================
Metrics Summary
========================================
load_images: 0.1234s (mean=0.0123s, std=0.0012s, calls=10)
psf_measurement: 2.3456s (mean=0.2346s, std=0.0234s, calls=10)
subtract: 45.6789s (mean=4.5679s, std=0.4568s, calls=10)

========================================
Implementation Comparison
========================================
new_subtract: 3.2145s (mean=3.2145s, std=0.0000s, calls=1)
old_subtract: 4.5679s (mean=4.5679s, std=0.0000s, calls=1)

Speedup vs fastest:
  old_subtract: 1.42x slower
```
