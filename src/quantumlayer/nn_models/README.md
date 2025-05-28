# Neural Network Models

A Python package providing implementations of classical and quantum neural network components. The package includes a standalone Multi-Layer Perceptron (MLP) and a differentiable Quantum Neural Network layer based on photonic circuits, allowing users to build both classical and quantum-enhanced neural architectures.

## Installation 

```bash
pip install git+ssh://git@gitlab.quandela.dev/coredev/nn_models.git
```

or from a local copy of the repository:
  
```bash
pip install -e ".[test,datasets]"
```

Requirements:
- Python >= 3.8
- PyTorch >= 2.0.0
- Perceval Quandela >= 0.11.0
- Custom PCVL PyTorch package

## Components

### Multi-Layer Perceptron (MLP)

A flexible implementation of a classical MLP with various customization options:

- Configurable hidden layer sizes
- Multiple activation functions (ReLU, tanh, sigmoid, etc.)
- Optional dropout
- Layer/batch normalization support
- Customizable weight initialization

Example usage:

```python
from nn_models import MLP, MLPConfig

config = MLPConfig(
    hidden_sizes=[64, 32],
    dropout=0.1,
    activation='relu',
    normalization='batch'
)

model = MLP(
    input_size=10,
    output_size=2,
    config=config
)
```

### Quantum Neural Network Layer

A quantum layer implementation using photonic circuits:

- Parameterized quantum photonic circuits
- Configurable trainable parameters and inputs
- Multiple output mapping strategies
- Support for different circuit architectures
- Integration with PyTorch's autograd system
- Dynamic circuit and input state updates: Allows changing the quantum circuit and input state during runtime while maintaining compatibility with the layer configuration.

Key features:
- Trainable quantum circuit parameters
- Flexible input mapping
- Three output mapping strategies:
  - Linear: Maps quantum outputs through a trainable linear layer
  - Grouping: Groups quantum outputs into equal-sized buckets
  - None: Direct quantum output (requires matching sizes)
- Support for custom Perceval circuits
- Dynamic updates: Change the quantum circuit or input state dynamically while preserving trained parameters and layer configuration.

Example usage:

```python
import perceval as pcvl
from nn_models import QuantumLayer, OutputMappingStrategy

# Create a quantum circuit
circuit = pcvl.Circuit(4)
circuit.add(0, pcvl.BS()//pcvl.PS(pcvl.P("theta1"))//pcvl.BS(), merge=True)
circuit.add(2, pcvl.BS()//pcvl.PS(pcvl.P("theta2"))//pcvl.BS(), merge=True)
circuit.add(1, pcvl.BS()//pcvl.PS(pcvl.P("x1"))//pcvl.BS(), merge=True)

# Create quantum layer
quantum_layer = QuantumLayer(
    input_size=1,  # One input (x1)
    output_size=3,
    circuit=circuit,
    trainable_parameters=["theta1", "theta2"],  # Parameters to be trained
    input_state=[1, 0, 1, 0],  # Initial quantum state
    output_mapping_strategy=OutputMappingStrategy.LINEAR
)

# Dynamically change the circuit
new_circuit = pcvl.Circuit(4)
new_circuit.add(0, pcvl.BS()//pcvl.PS(pcvl.P("theta1"))//pcvl.BS(), merge=True)
new_circuit.add(2, pcvl.BS()//pcvl.PS(pcvl.P("theta2"))//pcvl.BS(), merge=True)
quantum_layer.change_circuit(new_circuit)

# Dynamically change the input state
new_input_state = [1, 1, 0, 0]
quantum_layer.change_input_state(new_input_state)
```

#### Dynamic Circuit and Input State Updates

The `QuantumLayer` class provides methods to dynamically update the quantum circuit and input state:

- `change_circuit(circuit: pcvl.Circuit)`:  
  Dynamically change the quantum circuit while maintaining compatibility with the layer configuration. The new circuit must have the same number of modes and identical parameter names as the original circuit to ensure compatibility with the existing layer configuration and trained parameters.

```python
# Example usage:
new_circuit = pcvl.Circuit(4)  # Same number of modes as original
new_circuit.add(0, pcvl.BS()//pcvl.PS(pcvl.P("theta1"))//pcvl.BS())
quantum_layer.change_circuit(new_circuit)  # Updates circuit while preserving parameters
```

- **`change_input_state(input_state: List[int])`**:  
  Change the input state while keeping the same number of modes and photons. The new input state must match the number of modes and photons in the circuit.

```python
# Example usage:
new_input_state = [1, 0, 1, 0]  # Must match the number of modes and photons
quantum_layer.change_input_state(new_input_state)
```

## Examples

The package includes example scripts in the `examples/` directory:

### Iris Classification Example

`examples/iris_classifier.py` provides a comprehensive example script that demonstrates the usage of both classical and quantum models for the Iris classification task. The script:

- Implements and compares multiple model architectures:
  - Classical MLP with different sizes
  - Quantum models with different output mapping strategies
  - Various quantum circuit configurations
- Includes visualization tools for:
  - Training curves
  - Confusion matrices
  - Performance comparisons
- Provides a complete training and evaluation pipeline

To run the example:

```bash
python examples/iris_classifier.py
```

![iris-comparison.png](examples%2Foutput%2Firis-comparison.png)

## Configuration Options

### MLPConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_sizes` | List[int] | None | Sizes of hidden layers |
| `dropout` | float | 0.0 | Dropout probability |
| `activation` | str | 'relu' | Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu', 'selu') |
| `normalization` | str | None | Normalization type ('batch', 'layer', None) |

### Quantum Layer Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_size` | int | Number of input parameters for the circuit |
| `output_size` | int | Dimension of the final layer output |
| `circuit` | pcvl.Circuit | Perceval quantum circuit |
| `input_state` | List[int] | Initial photonic state configuration |
| `trainable_parameters` | Union[int, List[str]] | Parameters to be trained (count or names) |
| `output_mapping_strategy` | OutputMappingStrategy | Strategy for mapping quantum outputs |
| `output_map_func` | Callable[[Tuple[int, ...]], Optional[Tuple[int, ...]]] | Function to map quantum output states. For each state, it can: return the state unchanged, map it to a different state, or return None to discard it. The probability distribution is automatically normalized over surviving states. |

### OutputMappingStrategy

An enumeration defining how the quantum probability distribution is mapped to the final output:

| Strategy  | Description |
|-----------|-------------|
| `LINEAR`   | Applies a trainable linear layer to map the quantum probability distribution to the desired output size. Useful when you need to learn complex mappings between quantum states and output classes. |
| `GROUPING` | Groups the probability distribution values into equal-sized buckets and sums within each bucket. The number of buckets equals the desired output size. If the distribution size is not evenly divisible by the number of buckets, the distribution is padded with zero-probability states to make it divisible. Useful for reducing the dimensionality of the quantum output while preserving probability structure. |
| `NONE`     | Uses the quantum probability distribution directly as output. Requires that the number of possible quantum states matches the desired output size. Useful when your quantum circuit is designed to directly produce the desired number of outputs. |

Example of different strategies:

```python
# Linear mapping (learnable transformation)
model = QuantumLayer(
    input_size=4,
    output_size=3,  # Can be any size
    circuit=circuit,
    input_state=[1, 0, 1],
    output_mapping_strategy=OutputMappingStrategy.LINEAR
)

# Grouping (fixed transformation)
model = QuantumLayer(
    input_size=4,
    output_size=2,  # Will group quantum states into 2 buckets
    circuit=circuit,
    input_state=[1, 0, 1],
    output_mapping_strategy=OutputMappingStrategy.GROUPING
)

# Direct output (no transformation)
model = QuantumLayer(
    input_size=4,
    output_size=8,  # Must match number of possible quantum states
    circuit=circuit,
    input_state=[1, 0, 1],
    output_mapping_strategy=OutputMappingStrategy.NONE
)
```

### Custom Output State Mapping

The `output_map_func` parameter allows you to define custom mappings between quantum states before computing their probabilities. This is particularly useful for:
- Ignoring certain output states
- Grouping specific states together
- Implementing custom state post-selection

Example of custom output mapping:

```python
def custom_state_mapping(state: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """
    Custom mapping function that:
    - Ignores states with more than 2 photons in any mode
    - Maps symmetric states to the same output
    """
    # Ignore states with more than 2 photons in any mode
    if max(state) > 2:
        return None
        
    # Map symmetric states to the same output
    return tuple(sorted(state))

model = QuantumLayer(
    input_size=4,
    output_size=3,
    circuit=circuit,
    input_state=[1, 0, 1],
    output_map_func=custom_state_mapping,
    output_mapping_strategy=OutputMappingStrategy.LINEAR
)
```

## Notes on Quantum Layer Implementation

- Input parameters (x) should be in range [0, 1]. These values are internally scaled by 2π for phase shifters.
- Trainable parameters (theta) are initialized in range [0, π].
- The quantum layer can be used standalone or combined with classical layers for hybrid architectures.
- Output distributions are guaranteed to sum to 1.0 due to the quantum nature of the circuit.

# Datasets

This package provides simplified access to several datasets commonly used in machine learning, with a focus on quantum-inspired and quantum machine learning applications. Each dataset function returns a tuple of `(X, y, metadata)` where:
- `X`: Feature matrix
- `y`: Target labels
- `metadata`: Detailed dataset metadata including features description, normalization information, and other relevant details

## Available Datasets

### Iris Dataset
Classical dataset for machine learning, containing measurements of three different Iris flower species.

```python
from datasets import iris

# Get training data
X_train, y_train, metadata = iris.get_data_train()

# Get test data
X_test, y_test, metadata = iris.get_data_test()
```

All features are normalized to the range [0, 1] using min-max scaling. The metadata contains original and normalized statistics for each feature.

### Synthetic Spiral Dataset
A synthetic dataset featuring high-dimensional spiral patterns, designed for comparing classical and quantum approaches.

```python
from datasets import spiral

# Get data with default parameters
X, y, metadata = spiral.get_data()

# Get data with custom parameters
X, y, metadata = spiral.get_data(
    num_instances=2000,    # Default: 1500
    num_features=15,       # Default: 10
    num_classes=4,         # Default: 3
    random_seed=42        # Default: 42
)
```

The first two features form the base spiral pattern, with additional features created through nonlinear combinations using sine and cosine functions.

### MNIST Digits Dataset
Handwritten digits dataset with both original and Perceval Quest versions.

```python
from datasets import mnist_digits

# Original MNIST
X_train, y_train, metadata = mnist_digits.get_data_train_original()
X_test, y_test, metadata = mnist_digits.get_data_test_original()

# Perceval Quest subset
X_train, y_train, metadata = mnist_digits.get_data_train_percevalquest()
X_test, y_test, metadata = mnist_digits.get_data_test_percevalquest()
```

The Perceval Quest version is a carefully curated subset designed for quantum machine learning experiments, containing 6,000 training and 600 test images.

## Dataset Details

| Dataset | Instances | Features | Classes | Split               | Normalization |
|---------|-----------|----------|----------|---------------------|----------------|
| Iris | 150 | 4 | 3 | Train/Test          | Min-max [0,1] |
| Spiral | Configurable | Configurable | Configurable | Single              | None |
| MNIST Original | 70,000 | 784 | 10 | Train 60K/ Test 10K | [0,255] |
| MNIST Perceval | 6,600 | 784 | 10 | Train 6K/ Val 600   | [0,255] |

## Metadata Structure
Each dataset returns metadata containing:
- Dataset name and description
- Feature descriptions and statistics
- Normalization information (if applicable)
- Feature relationships and dependencies
- Citations and relevant papers
- Task type and characteristics

Example of accessing metadata information:
```python
from datasets import spiral
X, y, metadata = spiral.get_data()

# Print text version of the metadata
print(metadata)

# Print dataset description
print(metadata.description)

# Get feature information
for feature in metadata.features:
    print(feature.to_text())

# Check normalization details (if applicable)
if metadata.normalization:
    print(metadata.normalization.to_text())
```