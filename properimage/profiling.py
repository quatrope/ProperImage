#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2025 QuatroPe
#
# This file is part of ProperImage (https://github.com/quatrope/ProperImage)
# License: BSD-3-Clause
# Full Text: https://github.com/quatrope/ProperImage/blob/master/LICENSE.txt
#

"""
Profiling utilities for ProperImage performance analysis.

This module provides tools to profile and benchmark critical operations,
particularly the subtract() function and its optimization loops.
"""

import cProfile
import functools
import io
import logging
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Metrics
# =============================================================================


@dataclass
class TimingMetrics:
    """Store timing metrics for a profiled operation."""

    name: str
    total_time: float
    call_count: int = 1
    mean_time: float = field(init=False)
    std_time: float = 0.0
    min_time: float = field(init=False)
    max_time: float = field(init=False)
    times: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.times:
            self.times = [self.total_time]
        self.mean_time = self.total_time / self.call_count
        self.min_time = min(self.times) if self.times else self.total_time
        self.max_time = max(self.times) if self.times else self.total_time
        if len(self.times) > 1:
            self.std_time = float(np.std(self.times))

    def __str__(self):
        return (
            f"{self.name}: {self.total_time:.4f}s "
            f"(mean={self.mean_time:.4f}s, "
            f"std={self.std_time:.4f}s, "
            f"calls={self.call_count})"
        )


@dataclass
class SubtractMetrics:
    """Store comprehensive metrics for subtract() operation."""

    total_time: float
    fft_time: float = 0.0
    optimization_time: float = 0.0
    beta_optimization_time: float = 0.0
    shift_optimization_time: float = 0.0
    iterations: int = 0
    function_evaluations: int = 0
    success: bool = False
    optimization_method: str = "unknown"
    final_cost: Optional[float] = None

    def __str__(self):
        return (
            f"SubtractMetrics:\n"
            f"  Total time: {self.total_time:.4f}s\n"
            f"  FFT time: {self.fft_time:.4f}s "
            f"({100*self.fft_time/self.total_time:.1f}%)\n"
            f"  Optimization time: {self.optimization_time:.4f}s "
            f"({100*self.optimization_time/self.total_time:.1f}%)\n"
            f"    Beta: {self.beta_optimization_time:.4f}s\n"
            f"    Shift: {self.shift_optimization_time:.4f}s\n"
            f"  Iterations: {self.iterations}\n"
            f"  Function evals: {self.function_evaluations}\n"
            f"  Method: {self.optimization_method}\n"
            f"  Success: {self.success}\n"
            f"  Final cost: {self.final_cost}"
        )


# =============================================================================
# Context Managers for Profiling
# =============================================================================


@contextmanager
def timer(name: str = "operation", verbose: bool = True):
    """
    Context manager to time a code block.

    Parameters
    ----------
    name : str
        Name of the operation being timed
    verbose : bool
        Whether to print timing information

    Yields
    ------
    dict
        Dictionary with timing information (elapsed time in seconds)

    Examples
    --------
    >>> with timer("My operation") as t:
    ...     # do something
    ...     pass
    >>> print(f"Took {t['elapsed']:.4f} seconds")
    """
    start = time.perf_counter()
    timing_info = {}

    try:
        yield timing_info
    finally:
        elapsed = time.perf_counter() - start
        timing_info["elapsed"] = elapsed
        timing_info["name"] = name

        if verbose:
            logger.info(f"{name}: {elapsed:.4f}s")


@contextmanager
def profile_code(
    name: str = "operation",
    sort_by: str = "cumulative",
    top_n: int = 20,
    verbose: bool = True,
):
    """
    Context manager to profile a code block using cProfile.

    Parameters
    ----------
    name : str
        Name of the operation being profiled
    sort_by : str
        How to sort profiling results ('cumulative', 'time', 'calls')
    top_n : int
        Number of top results to display
    verbose : bool
        Whether to print profiling information

    Yields
    ------
    dict
        Dictionary with profiling statistics

    Examples
    --------
    >>> with profile_code("My function") as stats:
    ...     # do something expensive
    ...     pass
    """
    profiler = cProfile.Profile()
    profiler.enable()
    profile_info = {}

    try:
        yield profile_info
    finally:
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(top_n)

        profile_info["stats"] = ps
        profile_info["output"] = s.getvalue()
        profile_info["name"] = name

        if verbose:
            logger.info(f"\n{'='*60}\nProfile: {name}\n{'='*60}")
            logger.info(s.getvalue())


# =============================================================================
# Decorators for Function Profiling
# =============================================================================


def time_function(func: Callable) -> Callable:
    """
    Decorator to time a function and log results.

    Examples
    --------
    >>> @time_function
    ... def my_slow_function():
    ...     time.sleep(1)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__}: {elapsed:.4f}s")
        return result

    return wrapper


def profile_function(
    sort_by: str = "cumulative", top_n: int = 20
) -> Callable:
    """
    Decorator to profile a function using cProfile.

    Parameters
    ----------
    sort_by : str
        How to sort profiling results
    top_n : int
        Number of top results to display

    Examples
    --------
    >>> @profile_function(sort_by='time', top_n=10)
    ... def my_function():
    ...     # expensive operation
    ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            result = func(*args, **kwargs)

            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
            ps.print_stats(top_n)

            logger.info(
                f"\n{'='*60}\nProfile: {func.__name__}\n{'='*60}"
            )
            logger.info(s.getvalue())

            return result

        return wrapper

    return decorator


# =============================================================================
# Metrics Collection
# =============================================================================


class MetricsCollector:
    """
    Collect and aggregate timing metrics across multiple runs.

    Examples
    --------
    >>> collector = MetricsCollector()
    >>> with collector.time("operation1"):
    ...     # do something
    ...     pass
    >>> with collector.time("operation1"):
    ...     # do it again
    ...     pass
    >>> collector.print_summary()
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}

    @contextmanager
    def time(self, name: str):
        """Time a code block and add to metrics."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self.metrics:
                self.metrics[name] = []
                self.counts[name] = 0
            self.metrics[name].append(elapsed)
            self.counts[name] += 1

    def get_metrics(self, name: str) -> Optional[TimingMetrics]:
        """Get timing metrics for a named operation."""
        if name not in self.metrics:
            return None

        times = self.metrics[name]
        return TimingMetrics(
            name=name,
            total_time=sum(times),
            call_count=self.counts[name],
            times=times,
        )

    def print_summary(self):
        """Print summary of all collected metrics."""
        logger.info("\n" + "=" * 60)
        logger.info("Metrics Summary")
        logger.info("=" * 60)

        for name in sorted(self.metrics.keys()):
            metrics = self.get_metrics(name)
            logger.info(str(metrics))

    def reset(self):
        """Clear all collected metrics."""
        self.metrics.clear()
        self.counts.clear()


# =============================================================================
# Benchmarking Utilities
# =============================================================================


def benchmark_function(
    func: Callable,
    *args,
    n_runs: int = 10,
    warmup: int = 2,
    **kwargs,
) -> TimingMetrics:
    """
    Benchmark a function over multiple runs.

    Parameters
    ----------
    func : callable
        Function to benchmark
    *args
        Positional arguments to pass to func
    n_runs : int
        Number of times to run the function
    warmup : int
        Number of warmup runs (not counted)
    **kwargs
        Keyword arguments to pass to func

    Returns
    -------
    TimingMetrics
        Timing statistics across all runs

    Examples
    --------
    >>> def my_func(x, y):
    ...     return x + y
    >>> metrics = benchmark_function(my_func, 1, 2, n_runs=100)
    >>> print(metrics)
    """
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)

    # Actual benchmark runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return TimingMetrics(
        name=func.__name__,
        total_time=sum(times),
        call_count=n_runs,
        times=times,
    )


def compare_implementations(
    implementations: Dict[str, Callable],
    *args,
    n_runs: int = 10,
    **kwargs,
) -> Dict[str, TimingMetrics]:
    """
    Compare multiple implementations of the same operation.

    Parameters
    ----------
    implementations : dict
        Dictionary mapping names to callable implementations
    *args
        Positional arguments to pass to each implementation
    n_runs : int
        Number of runs per implementation
    **kwargs
        Keyword arguments to pass to each implementation

    Returns
    -------
    dict
        Dictionary mapping names to TimingMetrics

    Examples
    --------
    >>> def version1(x):
    ...     return x ** 2
    >>> def version2(x):
    ...     return x * x
    >>> results = compare_implementations(
    ...     {"pow": version1, "mult": version2},
    ...     100,
    ...     n_runs=1000
    ... )
    >>> for name, metrics in results.items():
    ...     print(metrics)
    """
    results = {}

    for name, func in implementations.items():
        logger.info(f"Benchmarking {name}...")
        metrics = benchmark_function(func, *args, n_runs=n_runs, **kwargs)
        results[name] = metrics

    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("Implementation Comparison")
    logger.info("=" * 60)

    sorted_results = sorted(
        results.items(), key=lambda x: x[1].mean_time
    )

    for name, metrics in sorted_results:
        logger.info(str(metrics))

    if len(sorted_results) > 1:
        fastest = sorted_results[0][1].mean_time
        logger.info("\nSpeedup vs fastest:")
        for name, metrics in sorted_results[1:]:
            speedup = metrics.mean_time / fastest
            logger.info(f"  {name}: {speedup:.2f}x slower")

    return results
