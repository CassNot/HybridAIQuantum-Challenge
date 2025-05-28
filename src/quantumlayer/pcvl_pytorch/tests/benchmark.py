import perceval as pcvl
import torch
import numpy as np

from benchmark_utils import benchmark, profile_function
from pcvl_pytorch import pytorch_slos_output_distribution_nobunching, pytorch_slos_output_distribution


# Example usage with your SLOS functions
def setup_test_case(m=4, n=2, batch_size=1):
    # Create random unitary matrix
    if batch_size == 1:
        unitary = (torch.randn(m, m) +
                    1j * torch.randn(m, m)) / torch.sqrt(torch.tensor(2.0))
    else:
        unitary = (torch.randn(batch_size, m, m) +
         1j * torch.randn(batch_size, m, m)) / torch.sqrt(torch.tensor(2.0))

    # Perform QR decomposition
    q, _ = torch.linalg.qr(unitary)

    # Create no-bunching input state
    input_state = [1]*n + [0]*(m-n)
    return q, input_state

@benchmark(num_runs=100, warmup_runs=10, profile_memory=False, profile_cuda=True)
def run_slos_perceval(unitary, input_state):
    U = unitary.numpy()
    Upcvl = pcvl.Unitary(pcvl.MatrixN.eye(U.shape[-1]))
    slos_backend = pcvl.BackendFactory().get_backend("SLOS")
    slos_backend.set_circuit(Upcvl)
    slos_backend.set_input_state(pcvl.BasicState(input_state))

    if len(unitary.shape) == 3:
        for i in range(unitary.size(0)):
            Upcvl._u = U[i]
            slos_backend.prob_distribution()
    else:
        Upcvl._u = U
        slos_backend.prob_distribution()

# Benchmark using decorator
@benchmark(num_runs=100, warmup_runs=10, profile_memory=False, profile_cuda=True)
def run_slos_original(unitary, input_state):
    return pytorch_slos_output_distribution(unitary, input_state)

@benchmark(num_runs=50, warmup_runs=10, profile_memory=False, profile_cuda=True)
def run_slos_nobunching(unitary, input_state):
    return pytorch_slos_output_distribution_nobunching(unitary, input_state, keep_keys=False)

# Run benchmarks
def compare_implementations():
    # Test cases of different sizes
    sizes = [10]
    n_photons = [5]
    batch_sizes = [50]
    
    results = {}
    
    for m in sizes:
        for n in n_photons:
            if m < n:
                continue
            for batch_size in batch_sizes:
                print(f"\nTesting m={m}, n={n}, batch_size={batch_size}")
                unitary, input_state = setup_test_case(m, n, batch_size)

                results_perceval = run_slos_perceval(unitary, input_state)
                print(results_perceval)

                # Run original implementation
                results_orig = run_slos_original(unitary, input_state)
                print(results_orig)

                # Run optimized implementation
                results_opt = run_slos_nobunching(unitary, input_state)
                print(results_opt)

                # Store results for comparison
                key = f"m={m}_n={n}_batch={batch_size}"
                results[key] = {
                    'perceval': results_perceval.get_stats(),
                    'original': results_orig.get_stats(),
                    'optimized': results_opt.get_stats()
                }
    
    return results

# For detailed profiling of a single run
@profile_function
def profile_single_run():
    unitary, input_state = setup_test_case(m=10, n=6, batch_size=10)
    return pytorch_slos_output_distribution(unitary, input_state)


if __name__ == "__main__":
    unitary, input_state = setup_test_case(14, 6, 64)
    results_opt = run_slos_nobunching(unitary, input_state)
    print(results_opt)
    exit(0)

    # Run comparison benchmarks
    results = compare_implementations()
    
    # Print speedup for each case
    for key, data in results.items():
        speedup = data['original']['mean'] / data['optimized']['mean']
        print(f"\n{key}:")
        print(f"Speedup: {speedup:.2f}x")
        
    # Run detailed profiling
    print("\nDetailed profiling of optimized version:")
    profile_single_run()