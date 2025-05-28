import time
import statistics
import functools
import cProfile
import pstats
from memory_profiler import profile as memory_profile
from contextlib import contextmanager
import torch


class BenchmarkResults:
    def __init__(self, name):
        self.name = name
        self.times = []
        self.memory_usage = []
        self.cuda_memory = []
        self.memory_profile_output = None

    def add_time(self, execution_time):
        self.times.append(execution_time)

    def add_memory(self, mem_usage):
        self.memory_usage.append(mem_usage)

    def add_cuda_memory(self, cuda_mem):
        self.cuda_memory.append(cuda_mem)

    def set_memory_profile(self, profile_output):
        self.memory_profile_output = profile_output

    def get_stats(self):
        if not self.times:
            return {}

        time_stats = {
            'mean': statistics.mean(self.times),
            'median': statistics.median(self.times),
            'stdev': statistics.stdev(self.times) if len(self.times) > 1 else 0,
            'min': min(self.times),
            'max': max(self.times)
        }

        mem_stats = {
            'mean_memory': statistics.mean(self.memory_usage) if self.memory_usage else None,
            'peak_memory': max(self.memory_usage) if self.memory_usage else None
        }

        cuda_stats = {
            'mean_cuda_memory': statistics.mean(self.cuda_memory) if self.cuda_memory else None,
            'peak_cuda_memory': max(self.cuda_memory) if self.cuda_memory else None
        }

        return {**time_stats, **mem_stats, **cuda_stats}

    def __str__(self):
        stats = self.get_stats()
        if not stats:
            return f"{self.name}: No measurements"

        result = [f"Benchmark results for {self.name}:"]
        result.append(f"Time (seconds):")
        result.append(f"  Mean: {stats['mean']:.6f}")
        result.append(f"  Median: {stats['median']:.6f}")
        result.append(f"  Std Dev: {stats['stdev']:.6f}")
        result.append(f"  Min: {stats['min']:.6f}")
        result.append(f"  Max: {stats['max']:.6f}")

        if stats['mean_memory']:
            result.append(f"Memory (MB):")
            result.append(f"  Mean: {stats['mean_memory']:.2f}")
            result.append(f"  Peak: {stats['peak_memory']:.2f}")

        if stats['mean_cuda_memory']:
            result.append(f"CUDA Memory (MB):")
            result.append(f"  Mean: {stats['mean_cuda_memory']:.2f}")
            result.append(f"  Peak: {stats['peak_cuda_memory']:.2f}")

        if self.memory_profile_output:
            result.append("\nDetailed Memory Profile:")
            result.append(self.memory_profile_output)

        return "\n".join(result)


class Benchmarker:
    def __init__(self, name, num_runs=100, warmup_runs=10,
                 profile_memory=False, profile_cuda=False):
        self.name = name
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.profile_memory = profile_memory
        self.profile_cuda = profile_cuda
        self.results = BenchmarkResults(name)

    @contextmanager
    def _time_block(self):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self.results.add_time(end_time - start_time)

    def _get_memory_profiled_func(self, func):
        """Wrap function with memory profiler and capture output"""
        from io import StringIO
        import sys

        # Create string buffer to capture memory profiler output
        output = StringIO()

        @memory_profile(stream=output)
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped_func, output

    def run(self, func, *args, **kwargs):
        """Run the benchmark on the given function"""
        # Warmup runs
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)

        # If memory profiling is enabled, wrap the function
        if self.profile_memory:
            profiled_func, mem_output = self._get_memory_profiled_func(func)
        else:
            profiled_func = func

        # Actual benchmark runs
        for i in range(self.num_runs):
            if self.profile_cuda:
                # Record CUDA memory before run
                if torch.cuda.is_available():
                    cuda_mem = torch.cuda.memory_allocated() / 1024 / 1024
                    self.results.add_cuda_memory(cuda_mem)

            with self._time_block():
                profiled_func(*args, **kwargs)

        # If memory profiling was enabled, store the output
        if self.profile_memory:
            self.results.set_memory_profile(mem_output.getvalue())
            mem_output.close()

        return self.results


def benchmark(num_runs=100, warmup_runs=10, profile_memory=False, profile_cuda=False):
    """Decorator for benchmarking functions"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            benchmarker = Benchmarker(
                func.__name__,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
                profile_memory=profile_memory,
                profile_cuda=profile_cuda
            )
            return benchmarker.run(func, *args, **kwargs)

        return wrapper

    return decorator


def profile_function(func):
    """Decorator for detailed profiling using cProfile"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats()
        return result

    return wrapper