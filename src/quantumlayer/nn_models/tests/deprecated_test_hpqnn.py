import pytest
import torch
import perceval as pcvl

from torch import nn

from nn_models import (
    HPQNN, 
    QuantumConfig, 
    CircuitSpec, 
    OutputMappingStrategy,
    build_circuit_from_spec,
    MLPConfig
)

@pytest.fixture
def basic_circuit_spec():
    return CircuitSpec(
        circuit_type='mzi_based',
        n_params=6,
        n_modes=3
    )

@pytest.fixture
def basic_quantum_config(basic_circuit_spec):
    return QuantumConfig(
        circuit_config=basic_circuit_spec,
        weights=3,
        input_state=[1, 1, 0]
    )

@pytest.fixture
def basic_classical_config():
    return MLPConfig(
        hidden_sizes=[32],
        dropout=0.1,
        activation='relu'
    )

@pytest.fixture
def basic_hpqnn(basic_quantum_config, basic_classical_config):
    return HPQNN(
        input_size=10,
        output_size=4,
        quantum_config=basic_quantum_config,
        preprocessor_config=basic_classical_config,
        output_mapping_strategy=OutputMappingStrategy.GROUPING
    )


def test_circuit_building():
    """Test circuit building from specifications"""
    # Test MZI-based circuit
    mzi_spec = CircuitSpec(circuit_type='mzi_based', n_params=4, n_modes=3)
    mzi_circuit = build_circuit_from_spec(mzi_spec)
    assert isinstance(mzi_circuit, pcvl.Circuit)
    assert mzi_circuit.m == 3  # Check number of modes

    # Test BS-based circuit
    bs_spec = CircuitSpec(circuit_type='bs_based', n_params=4, n_modes=3)
    bs_circuit = build_circuit_from_spec(bs_spec)
    assert isinstance(bs_circuit, pcvl.Circuit)
    assert bs_circuit.m == 3

    # Test invalid circuit type
    with pytest.raises(ValueError):
        invalid_spec = CircuitSpec(circuit_type='invalid', n_params=4, n_modes=3)
        build_circuit_from_spec(invalid_spec)


def test_hpqnn_initialization(basic_quantum_config, basic_classical_config):
    """Test HPQNN initialization with different configurations"""
    # Test with linear output mapping
    model = HPQNN(
        input_size=10,
        output_size=4,
        quantum_config=basic_quantum_config,
        preprocessor_config=basic_classical_config,
        output_mapping_strategy=OutputMappingStrategy.LINEAR
    )
    assert isinstance(model.output_mapping, nn.Linear)

    # Test with grouping output mapping
    model = HPQNN(
        input_size=10,
        output_size=2,
        quantum_config=basic_quantum_config,
        preprocessor_config=basic_classical_config,
        output_mapping_strategy=OutputMappingStrategy.GROUPING
    )
    assert hasattr(model, 'group_size')

    # Test with no output mapping
    distribution_size = model.probability_distribution_size
    model = HPQNN(
        input_size=10,
        output_size=distribution_size,
        quantum_config=basic_quantum_config,
        preprocessor_config=basic_classical_config,
        output_mapping_strategy=OutputMappingStrategy.NONE
    )
    assert isinstance(model.output_mapping, nn.Identity)


def test_invalid_configurations(basic_quantum_config, basic_classical_config):
    """Test that invalid configurations raise appropriate errors"""
    # Test invalid input state size
    invalid_quantum_config = QuantumConfig(
        circuit_config=basic_quantum_config.circuit_config,
        input_state=[1, 1, 1, 1]  # Too many modes
    )
    with pytest.raises(ValueError):
        HPQNN(
            input_size=10,
            output_size=4,
            quantum_config=invalid_quantum_config,
            preprocessor_config=basic_classical_config
        )

    # Test invalid output mapping strategy
    with pytest.raises(ValueError):
        HPQNN(
            input_size=10,
            output_size=4,
            quantum_config=basic_quantum_config,
            preprocessor_config=basic_classical_config,
            output_mapping_strategy='invalid'
        )


def test_forward_pass(basic_hpqnn):
    """Test forward pass through the entire network"""
    batch_size = 16
    input_size = 10
    x = torch.randn(batch_size, input_size)
    
    output = basic_hpqnn(x)
    
    assert output.shape == (batch_size, 4)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    # Check that probabilities sum approximately to 1
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)


def test_quantum_output(basic_hpqnn):
    """Test quantum circuit output properties"""
    batch_size = 16
    param_size = basic_hpqnn.n_input_params + basic_hpqnn.n_weights

    # Test with random parameters
    params = torch.randn(batch_size, param_size)
    quantum_output = basic_hpqnn.get_quantum_output(params)
    
    # Check that output is a valid probability distribution
    assert torch.allclose(quantum_output.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    assert (quantum_output >= 0).all()
    assert (quantum_output <= 1).all()


def test_grouping_strategy():
    """Test probability grouping output mapping"""
    quantum_config = QuantumConfig(
        circuit_config=CircuitSpec(
            circuit_type='mzi_based',
            n_params=6,
            n_modes=3
        ),
        input_state=[1, 1, 0]
    )
    
    model = HPQNN(
        input_size=10,
        output_size=2,  # Force grouping
        quantum_config=quantum_config,
        preprocessor_config=MLPConfig(hidden_sizes=[32]),
        output_mapping_strategy=OutputMappingStrategy.GROUPING
    )
    
    # Test grouping function directly
    probabilities = torch.rand(4, model.probability_distribution_size)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    grouped = model.group_probabilities(probabilities)
    
    assert grouped.shape == (4, 2)
    assert torch.allclose(grouped.sum(dim=1), torch.ones(4), atol=1e-6)


def test_end_to_end_training(basic_hpqnn):
    """Test that the model can be trained end-to-end"""
    # Create dummy dataset
    batch_size = 32
    input_size = 10
    x = torch.randn(batch_size, input_size)
    y = torch.randint(0, 2, (batch_size,)).float()

    # Setup optimizer
    optimizer = torch.optim.Adam(basic_hpqnn.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Initial loss
    initial_output = basic_hpqnn(x)
    initial_loss = criterion(initial_output[:,0], y)

    # Train for a few steps
    for _ in range(10):
        optimizer.zero_grad()
        output = basic_hpqnn(x)
        loss = criterion(output[:,0], y)
        loss.backward()
        optimizer.step()

    # Final loss
    final_output = basic_hpqnn(x)
    final_loss = criterion(final_output[:,0], y)

    # Check that loss decreased
    assert final_loss < initial_loss


def test_string_representation_minimal():
    """Test minimal string representation of HPQNN with essential fields"""
    # Create minimal configuration
    quantum_config = QuantumConfig(
        circuit_config=CircuitSpec(
            circuit_type='mzi_based',
            n_params=4,
            n_modes=2
        ),
        input_state=[1, 1]
    )

    model = HPQNN(
        input_size=4,
        output_size=3,
        quantum_config=quantum_config,
        preprocessor_config=None,  # No preprocessor
        output_mapping_strategy=OutputMappingStrategy.NONE
    )

    str_repr = str(model)

    # Verify only essential information
    assert 'Input Size: 4' in str_repr
    assert 'Output Size: 3' in str_repr
    assert 'Input State: [1, 1]' in str_repr


def test_no_preprocessor():
    """Test HPQNN without preprocessor network"""
    quantum_config = QuantumConfig(
        circuit_config=CircuitSpec(
            circuit_type='mzi_based',
            n_params=6,
            n_modes=3
        ),
        input_state=[1, 1, 0]
    )
    
    # This should work (input_size matches n_params)
    model = HPQNN(
        input_size=6,
        output_size=6,
        quantum_config=quantum_config,
        preprocessor_config=None
    )
    
    # This should fail (input_size doesn't match n_params)
    with pytest.raises(ValueError):
        HPQNN(
            input_size=10,
            output_size=6,
            quantum_config=quantum_config,
            preprocessor_config=None
        )