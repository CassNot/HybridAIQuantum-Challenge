from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Literal, Union

import perceval as pcvl
import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import TypeAlias  # for Python <3.10

from nn_models import MLP, MLPConfig
from pcvl_pytorch import pcvl_circuit_to_pytorch_unitary, pytorch_slos_output_distribution

# Define possible circuit types
CircuitType: TypeAlias = Literal['mzi_based', 'bs_based']

@dataclass
class CircuitSpec:
    """Specification for creating a quantum circuit"""
    circuit_type: CircuitType
    n_params: int
    # Add any other parameters needed to specify the circuit
    n_modes: int
    additional_config: Optional[dict] = None

@dataclass
class QuantumConfig:
    """Configuration for the quantum part of HPQNN

        Either provide a pre-built circuit or specifications to build one.
        Circuit parameters are considered as input tensor, and the output is a probability distribution.
        If weights are provided, they are trainable part of the quantum circuit.

    Examples:
        # Using pre-built circuit:
        >>> config = QuantumConfig(
        ...     circuit_config=pcvl.Circuit(2)//pcvl.BS()//pcvl.PS(pcvl.P('phi1'))//pcvl.BS()//pcvl.PS(pcvl.P('phi2'))//pcvl.BS(),
        ...     input_state=[1, 1]
        ...     weights=["phi1"] # phi1 is a weight, phi2 is an input parameter
        ... )

        # Using circuit specifications:
        >>> config = QuantumConfig(
        ...     circuit_config=CircuitSpec(
        ...         circuit_type='mzi_based',
        ...         n_params=6,
        ...         n_modes=3
        ...     ),
        ...     weights=3 # three first params are used for weights, rest are input parameters
        ...     input_state=[1, 1, 1]
        ... )
        """
    circuit_config: Union[pcvl.Circuit, CircuitSpec]
    input_state: List[int]
    weights: Optional[Union[int, List[str]]] = None


def build_circuit_from_spec(spec: CircuitSpec) -> 'Circuit':
    """
    Build a Perceval circuit based on specifications.

    Args:
        spec: Circuit specifications including type and parameters

    Returns:
        Circuit: Built Perceval circuit with mesh pattern of components
    """
    import perceval as pcvl

    circuit = pcvl.Circuit(spec.n_modes)

    if spec.circuit_type == 'mzi_based':
        param_idx = 0
        layer = 0

        # First calculate how many inner phases we need
        n_inner_phases = (spec.n_params+1) // 2
        outer_phases_to_cover = {}
        while param_idx < n_inner_phases or len(outer_phases_to_cover) > 0:
            # Alternate starting position between layers
            start_pos = layer % 2

            # Add MZIs for this layer
            for pos in range(start_pos, spec.n_modes - 1, 2):
                add_outer_phase = False
                c = None
                # Add BS and inner phase
                if param_idx < n_inner_phases:
                    c = pcvl.BS() // pcvl.PS(pcvl.P(f"phi_{2*param_idx}")) // pcvl.BS()

                    if 2*param_idx + 1 < spec.n_params:
                        c = c // pcvl.PS(pcvl.P(f"phi_{2*param_idx + 1}"))
                        outer_phases_to_cover[pos] = 2*param_idx + 1
                        add_outer_phase = True

                    param_idx += 1

                else:
                    if pos in outer_phases_to_cover or pos+1 in outer_phases_to_cover:
                        c = pcvl.BS()

                if c is not None:
                    if not add_outer_phase and pos in outer_phases_to_cover:
                        del outer_phases_to_cover[pos]
                    if pos+1 in outer_phases_to_cover:
                        del outer_phases_to_cover[pos+1]

                    circuit.add((pos, pos + 1), c, merge=True)

            layer += 1

    elif spec.circuit_type == 'bs_based':
        # BS implementation
        layer = 0
        param_idx = 0

        while param_idx < spec.n_params:
            start_pos = layer % 2
            for pos in range(start_pos, spec.n_modes - 1, 2):
                if param_idx >= spec.n_params:
                    break

                bs = pcvl.BS() // pcvl.PS(pcvl.P(f"phi_{param_idx}")) // pcvl.BS()
                circuit.add((pos, pos + 1), bs, merge=True)
                param_idx += 1

            layer += 1

    else:
        raise ValueError(f"Unknown circuit type: {spec.circuit_type}")

    return circuit

class OutputMappingStrategy(Enum):
    LINEAR = 'linear'
    GROUPING = 'grouping'
    NONE = 'none'

class HPQNN(nn.Module):
    """
    Hybrid Photonic Quantum Neural Network (HPQNN) that combines classical and quantum layers.

    The network consists of:
    1. A classical MLP front-end with configurable hidden layers
    2. A quantum photonic circuit backend implemented using Perceval
    3. An optional output mapping layer to match quantum distribution size to desired output size

    The quantum part uses a photonic circuit where:
    - The classical output is mapped to quantum input states
    - The quantum circuit is parameterized and differentiable
    - The output is a probability distribution over possible photonic states

    The output mapping strategy determines how the quantum probability distribution
    is mapped to the final network output:
    - 'linear': Applies a trainable linear layer to map the distribution to desired output size
    - 'grouping': Groups the distribution values into equal-sized buckets (#distribution/#outputs)
    - 'none': No mapping applied (requires #distribution == #outputs)

    Args:
        input_size (int): Dimension of the input features
        output_size (int): Dimension of the final network output
        quantum_config (QuantumConfig): Configuration for the quantum circuit, either:
            - A pre-built Perceval circuit
            - Specifications to build a circuit (type, parameters)
        preprocessor_config (MLPConfig, optional): Configuration for the classical MLP
        output_mapping_strategy (str): Strategy for mapping quantum output to final output:
            - 'linear': Uses trainable linear layer
            - 'grouping': Groups distribution values
            - 'none': No mapping (requires matching sizes)

    Raises:
        ValueError: If input state size does not match number of input modes in the circuit
        ValueError: In no preprocessor and input_size != quantum_params.n_params
        ValueError: If output_mapping_strategy is 'none' and distribution size != output_size
        ValueError: If output_mapping_strategy is invalid

    Example:
        >>> quantum_config = QuantumConfig(
        ...     circuit_config=CircuitSpec(
        ...         circuit_type='bs_based',
        ...         n_params=6,
        ...         n_modes=3
        ...     ),
        ...     input_state=[1, 1, 1]
        ... )
        >>> classical_config = MLPConfig(
        ...     hidden_sizes=[64, 32],
        ...     dropout=0.1,
        ...     activation='relu'
        ... )
        >>> model = HPQNN(
        ...     input_size=10,
        ...     output_size=4,
        ...     quantum_config=quantum_config,
        ...     preprocessor_config=classical_config,
        ...     output_mapping_strategy=OutputMappingStrategy.LINEAR
        ... )
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 quantum_config: QuantumConfig,
                 preprocessor_config: Optional[dict] = None,
                 output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.NONE):
        super().__init__()

        # Store quantum circuit configuration
        # Build circuit if needed
        if isinstance(quantum_config.circuit_config, CircuitSpec):
            self.circuit = build_circuit_from_spec(quantum_config.circuit_config)
            self.circuit_type = quantum_config.circuit_config.circuit_type
        else:
            self.circuit = quantum_config.circuit_config
            self.circuit_type = "Custom Perceval"

        self.circuit_parameters = self.circuit.get_parameters()
        self.circuit_parameters_name = [p.name for p in self.circuit_parameters]

        self.input_state = quantum_config.input_state
        self.input_size = input_size
        self.output_size = output_size
        self.output_mapping_strategy = output_mapping_strategy

        if len(self.input_state) != self.circuit.m:
                raise ValueError(
                    "Input state size must match number of input modes in the circuit"
                )

        self.n_input_params = len(self.circuit_parameters)
        if quantum_config.weights is not None:
            weights = quantum_config.weights
            if isinstance(quantum_config.weights, list):
                self.n_weights = len(weights)
            else:
                self.n_weights = weights
            self.n_input_params -= self.n_weights
            self.weight_tensor = nn.Parameter(torch.randn(self.n_weights))
            self.map_weights = {}
            if isinstance(quantum_config.weights, list):
                self.map_params = {name: self.weight_tensor[idx] for idx, name in enumerate(quantum_config.weights)}
                self.input_parameters = [p.name
                                         for p in self.circuit_parameters if p.name not in quantum_config.weights]
            else:
                self.map_params = {self.circuit_parameters[idx].name: self.weight_tensor[idx]
                                    for idx in range(quantum_config.weights)}
                self.input_parameters = [self.circuit_parameters[idx].name
                                         for idx in range(quantum_config.weights, len(self.circuit_parameters))]
        else:
            self.map_params = {}
            self.weight_tensor = None
            self.n_weights = 0
            self.input_parameters = [p.name for p in self.circuit_parameters]

        if self.n_input_params == 0:
            raise ValueError("Quantum Circuit must have input parameters")

        # Initialize classical network
        if preprocessor_config is None:
            preprocessor_config = {}
            if input_size != self.n_input_params:
                raise ValueError(
                    "Input size must match quantum input parameter size if no preprocessor provided"
                )
            self.preprocessor_network = nn.Identity()
        else:
            # Create classical MLP
            self.preprocessor_network = MLP(
                input_size=input_size,
                output_size=self.n_input_params,
                config=preprocessor_config
            )

        # Initialize output mapping based on selected strategy - using a dummy probability distribution
        unitary = pcvl_circuit_to_pytorch_unitary(self.circuit, torch.tensor([0.0] * (self.n_input_params + self.n_weights)))
        _, probability_distribution = pytorch_slos_output_distribution(unitary, self.input_state)
        self.setup_output_mapping(probability_distribution)

    def setup_output_mapping(self, output_distribution):
        """Initialize output mapping based on selected strategy"""

        # Get the size of quantum output distribution
        self.distribution_size = output_distribution.shape[-1]

        # Initialize output mapping based on strategy
        if self.output_mapping_strategy == OutputMappingStrategy.LINEAR:
            self.output_mapping = nn.Linear(self.distribution_size, self.output_size)

        elif self.output_mapping_strategy == OutputMappingStrategy.GROUPING:
            self.group_size = self.distribution_size // self.output_size
            self.output_mapping = self.group_probabilities

        elif self.output_mapping_strategy == OutputMappingStrategy.NONE:
            if self.distribution_size != self.output_size:
                raise ValueError(
                    f"Distribution size ({self.distribution_size}) must equal "
                    f"output size ({self.output_size}) when using 'none' strategy"
                )
            self.output_mapping = nn.Identity()

        else:
            raise ValueError(f"Unknown output mapping strategy: {self.output_mapping_strategy}")

    def group_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Group probability distribution into equal-sized buckets, padding if necessary.

        Args:
            probabilities (torch.Tensor): [batch_size, distribution_size]

        Returns:
            torch.Tensor: [batch_size, output_size]
        """
        batch_size = probabilities.shape[0]
        # Calculate padding needed
        pad_size = (self.output_size - (self.distribution_size % self.output_size)) % self.output_size
        if pad_size > 0:
            # Pad with zeros - no need to renormalize
            padded = F.pad(probabilities, (0, pad_size))
        else:
            padded = probabilities

        # Now we can reshape safely
        return padded.view(batch_size, self.output_size, -1).sum(dim=-1)

    def get_quantum_output(self, input_parameters: torch.Tensor) -> torch.Tensor:
        """
        Process the classical output through the quantum circuit.

        Args:
            input_parameters (torch.Tensor): Output from classical layers [param_size] or [batch_size, param_size]

        Returns:
            torch.Tensor: Probability distribution over quantum output states [n_states] or [batch_size, n_states]
        """
        if input_parameters.dim() == 1:
            for idx, p in enumerate(self.input_parameters):
                self.map_params[p] = input_parameters[idx]
        else:
            for idx, p in enumerate(self.input_parameters):
                self.map_params[p] = input_parameters[:, idx]

        unitaries = pcvl_circuit_to_pytorch_unitary(self.circuit, self.map_params)
        _, probability_distribution = pytorch_slos_output_distribution(unitaries, self.input_state)
        return probability_distribution

    def __str__(self) -> str:
        """
        Returns a string representation of the HPQNN architecture.

        The representation includes:
        - Network dimensions (input/output sizes)
        - Quantum circuit configuration
        - Classical preprocessor network structure
        - Output mapping strategy

        Returns:
            str: Formatted string describing the network architecture
        """
        # Build the string representation
        sections = []

        # Main architecture section
        sections.append("HPQNN Architecture:")
        sections.append(f"  Input Size: {self.input_size}")
        sections.append(f"  Output Size: {self.output_size}")

        # Quantum circuit section
        sections.append("\nQuantum Circuit Configuration:")
        sections.append(f"  Circuit Type: {self.circuit_type}")
        sections.append(f"  Number of Modes: {self.circuit.m}")
        sections.append(f"  Number of Input Parameters: {self.n_input_params}")
        sections.append(f"  Number of Trainable Weight Parameters: {self.n_weights}")
        sections.append(f"  Input State: {self.input_state}")

        # Classical preprocessor section
        sections.append("\nPreprocessor Network:")
        if hasattr(self.preprocessor_network, 'config'):
            preproc_config = self.preprocessor_network.config
            sections.append(f"  Hidden Layers: {preproc_config.hidden_sizes}")
            sections.append(f"  Activation: {preproc_config.activation}")
            if preproc_config.dropout > 0:
                sections.append(f"  Dropout Rate: {preproc_config.dropout}")
            if preproc_config.normalization:
                sections.append(f"  Normalization: {preproc_config.normalization}")
        else:
            sections.append("  No preprocessor network")

        # Output mapping section
        sections.append("\nOutput Configuration:")
        sections.append(f"  Distribution Size: {self.distribution_size}")
        sections.append(f"  Output Mapping: {self.output_mapping_strategy.value}")
        if self.output_mapping_strategy == OutputMappingStrategy.GROUPING:
            sections.append(f"  Group Size: {self.group_size}")

        return "\n".join(sections)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.

        Args:
            x (torch.Tensor): Input tensor [input_size] or [batch_size, input_size]

        Returns:
            torch.Tensor: Final output after mapping [output_size] or [batch_size, output_size]
        """
        preprocessor_output = self.preprocessor_network(x)
        quantum_output = self.get_quantum_output(preprocessor_output)
        self.setup_output_mapping(quantum_output)
        return self.output_mapping(quantum_output)

# Example usage:
if __name__ == "__main__":

    print("creating circuit")
    c = pcvl.Circuit(4)
    c.add(0, pcvl.BS()//pcvl.PS(pcvl.P("phi1"))//pcvl.BS(), merge=True)
    c.add(2, pcvl.BS()//pcvl.PS(pcvl.P("phi2"))//pcvl.BS()//pcvl.PS(pcvl.P("phi3")), merge=True)
    c.add(1, pcvl.BS()//pcvl.PS(pcvl.P("x"))//pcvl.BS(), merge=True)

    quantum_config = QuantumConfig(
        circuit_config=c,
        weights=["phi1", "phi2", "phi3"],
        input_state=[1, 0, 1, 0]
    )
    classical_config = MLPConfig(
        hidden_sizes=[],
        dropout=0.1,
        activation='relu'
    )

    # Create model
    model = HPQNN(
        input_size=3,
        output_size=4,
        quantum_config=quantum_config,
        preprocessor_config=classical_config
    )

    t = torch.tensor([1.0, 2.0, 3.0])
    print(model(t))