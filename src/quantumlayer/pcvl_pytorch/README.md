# Differentiable Quantum Photonics

This repository provides PyTorch-compatible tools for differentiable quantum photonic circuit simulation. It enables gradient-based optimization of quantum optical circuits by implementing:

1. Conversion of Perceval quantum circuits to differentiable PyTorch unitary matrices
2. Computation of output photon probability distributions as differentiable functions

## Features

- Convert Perceval quantum circuits to PyTorch tensors
- Support batch processing for efficient parallel simulation
- Maintain differentiability for gradient-based optimization
- Calculate complete output probability distributions
- Support for arbitrary n-photon input states
- Compatible with PyTorch's automatic differentiation

## Prerequisites

- Python 3.8 or higher
- PyTorch
- Perceval
- NumPy
- SymPy

## Installation

To install the package in development mode:
```bash
pip install -e .
```

For development with test dependencies:
```bash
pip install -e ".[test]"
```

## Basic Usage

### Converting a Single Circuit

```python
import perceval as pcvl
import torch
from pcvl_pytorch import pcvl_circuit_to_pytorch_unitary

# Create a Mach-Zehnder interferometer
circuit = pcvl.Circuit(2)
circuit.add(0, pcvl.BS())                    # First beam splitter
circuit.add(0, pcvl.PS(pcvl.P("phi1")))      # Phase shifter 1
circuit.add(0, pcvl.BS())                    # Second beam splitter
circuit.add(0, pcvl.PS(pcvl.P("phi2")))      # Phase shifter 2

# Convert to PyTorch unitary with parameters
params = torch.tensor([0.1, 0.2], requires_grad=True)
unitary = pcvl_circuit_to_pytorch_unitary(circuit, params)

# alternatively:
# unitary = pcvl_circuit_to_pytorch_unitary(circuit, {"phi1":params[0], "phi2":params[1]})

# Compute gradients
loss = torch.abs(unitary[0, 0]) ** 2
loss.backward()
print("Parameter gradients:", params.grad)
```

### Batch Processing

Process multiple parameter sets efficiently in parallel:

```python
# Create multiple parameter sets
batch_params = torch.tensor([
    [0.1, 0.2],  # First parameter set
    [0.3, 0.4],  # Second parameter set
    [0.5, 0.6]   # Third parameter set
], requires_grad=True)

# Convert to batch of unitaries
batch_unitary = pcvl_circuit_to_pytorch_unitary(circuit, batch_params)
# Shape: (batch_size, n_modes, n_modes)

# Compute loss over the batch
batch_loss = torch.sum(torch.abs(batch_unitary[:, 0, 0]) ** 2)
batch_loss.backward()
print("Batch gradients:", batch_params.grad)
```

### Computing Output Probabilities

Calculate photon output probabilities. Two functions are available for this purpose, one to run in non bunching mode
(`pytorch_slos_output_distribution`) and the other to run in bunching mode (`pytorch_slos_output_distribution`). 
The first one is far more efficient and corresponds to experimentally verifiable no-loss events where each input photon 
is detected exactly once in the output, making it particularly valuable for actual implementation on NISQ device.

```python
from pcvl_pytorch import pytorch_slos_output_distribution

# Single unitary case
input_state = [1, 0]  # Single photon input in first mode
keys, probabilities = pytorch_slos_output_distribution(unitary, input_state)

# Batch processing case
batch_size = 3
angles = torch.linspace(0, torch.pi/2, batch_size)
batch_unitaries = torch.stack([
    create_beamsplitter(theta) for theta in angles
])
keys, batch_probabilities = pytorch_slos_output_distribution(batch_unitaries, input_state)
# batch_probabilities shape: [batch_size, num_output_states]
```

## Advanced Features

### Output State Mapping

#### Threshold Detection
Transform photon numbers to binary detection events:

```python
from pcvl_pytorch import threshold_mapping

# Two photons in first mode
input_state = [2, 0]
keys, probabilities = pytorch_slos_output_distribution(
    unitary, 
    input_state,
    output_map_func=threshold_mapping
)
# States like (2,0) are mapped to (1,0)
```

#### Selective State Filtering
Filter specific output states:

```python
from typing import Tuple, Optional

def selective_mapping(state: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Keep states with total photon number ≤ 1"""
    return state if sum(state) <= 1 else None

# Hong-Ou-Mandel input
input_state = [1, 1]
keys, probabilities = pytorch_slos_output_distribution(
    unitary, 
    input_state,
    output_map_func=selective_mapping
)
```

### Optimization Example
Optimize a batch of circuits simulateneously:

```python
# Initialize batch of parameters
batch_size = 4
thetas = torch.zeros(batch_size, requires_grad=True)
optimizer = torch.optim.Adam([thetas], lr=0.01)

for step in range(100):
    optimizer.zero_grad()
    
    # Create batch of unitaries
    batch_U = torch.stack([
        create_beamsplitter(theta) for theta in thetas
    ])
    
    # Compute output probabilities
    input_state = [1, 1]  # HOM input
    keys, probs = pytorch_slos_output_distribution(batch_U, input_state)
    
    # Maximize bunching across all circuits
    bunching_idx = [i for i, k in enumerate(keys) if k in {(2, 0), (0, 2)}]
    loss = -sum(probs[:, i].mean() for i in bunching_idx)
    
    loss.backward()
    optimizer.step()
```

## Testing

Run the test suite:

```bash
pytest
```

For verbose output:
```bash
pytest -v
```

For specific test files:
```bash
pytest tests/test_pcvl2torch.py
pytest -k single_photon
```
