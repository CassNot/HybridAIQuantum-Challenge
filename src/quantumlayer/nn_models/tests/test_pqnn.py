import pytest
import torch
import perceval as pcvl
from nn_models import QuantumLayer, OutputMappingStrategy

@pytest.fixture
def basic_circuit():
    """Fixture providing a basic test circuit"""
    c = pcvl.Circuit(4)
    c.add(0, pcvl.BS()//pcvl.PS(pcvl.P("theta1"))//pcvl.BS(), merge=True)
    c.add(2, pcvl.BS()//pcvl.PS(pcvl.P("theta2"))//pcvl.BS()//pcvl.PS(pcvl.P("x1")), merge=True)
    c.add(1, pcvl.BS()//pcvl.PS(pcvl.P("x2"))//pcvl.BS(), merge=True)
    return c

class TestQuantumLayer:
    def test_initialization(self, basic_circuit):
        """Test layer initialization with various configurations"""
        # Test basic initialization
        layer = QuantumLayer(
            input_size=2,
            circuit=basic_circuit,
            input_state=[1, 0, 1, 0],
            trainable_parameters=["theta1", "theta2"],
            output_size=10
        )
        assert len(layer.theta_names) == 2
        assert len(layer.x_names) == 2
        assert len(list(layer.parameters())) == 1  # Only thetas

        # Test with linear output mapping
        layer = QuantumLayer(
            input_size=2,
            circuit=basic_circuit,
            input_state=[1, 0, 1, 0],
            trainable_parameters=["theta1", "theta2"],
            output_size=3,
            output_mapping_strategy=OutputMappingStrategy.LINEAR
        )
        assert len(list(layer.parameters())) > 2  # thetas + linear layer params

    def test_parameter_tracking(self, basic_circuit):
        """Test proper parameter registration and tracking"""
        layer = QuantumLayer(
            input_size=2,
            circuit=basic_circuit,
            input_state=[1, 0, 1, 0],
            trainable_parameters=["theta1", "theta2"],
            output_size=10
        )
        
        # Check if parameters are properly registered
        params = dict(layer.named_parameters())
        assert len(params) == 1
        assert all(isinstance(p, torch.nn.Parameter) for p in params.values())
        
        # Check if gradients can be computed
        x = torch.randn(1, 2)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Verify gradients exist
        assert all(p.grad is not None for p in params.values())

    def test_forward_pass(self, basic_circuit):
        """Test forward pass with different input shapes"""
        layer = QuantumLayer(
            input_size=2,
            circuit=basic_circuit,
            input_state=[1, 0, 1, 0],
            trainable_parameters=["theta1", "theta2"],
            output_size=10
        )
        
        # Test single input
        x_single = torch.randn(2)
        output_single = layer(x_single)
        assert output_single.shape == (10,)
        assert torch.allclose(output_single.sum(), torch.tensor(1.0), atol=1e-6)
        
        # Test batch input
        x_batch = torch.randn(3, 2)
        output_batch = layer(x_batch)
        assert output_batch.shape == (3, 10)
        assert torch.allclose(output_batch.sum(dim=1), torch.ones(3), atol=1e-6)

    def test_output_mapping_strategies(self, basic_circuit):
        """Test different output mapping strategies"""
        # Test LINEAR strategy
        layer_linear = QuantumLayer(
            input_size=2,
            circuit=basic_circuit,
            input_state=[1, 0, 1, 0],
            trainable_parameters=["theta1", "theta2"],
            output_size=5,
            output_mapping_strategy=OutputMappingStrategy.LINEAR
        )
        x = torch.randn(2)
        output = layer_linear(x)
        assert output.shape == (5,)
        
        # Test GROUPING strategy
        layer_grouping = QuantumLayer(
            input_size=2,
            circuit=basic_circuit,
            input_state=[1, 0, 1, 0],
            trainable_parameters=["theta1", "theta2"],
            output_size=2,
            output_mapping_strategy=OutputMappingStrategy.GROUPING
        )
        output = layer_grouping(x)
        assert output.shape == (2,)
        assert torch.allclose(output.sum(), torch.tensor(1.0), atol=1e-6)

    def test_output_map_func(self, basic_circuit):
        """Test custom output mapping function"""
        def custom_map(state_tuple):
            """Only keep states without bunching"""
            if any(s for s in state_tuple if s > 1):
                return None
            return state_tuple
            
        layer = QuantumLayer(
            circuit=basic_circuit,
            input_size=2,
            input_state=[1, 0, 1, 0],
            trainable_parameters=["theta1", "theta2"],
            output_size=6,  # Number of states without bunching
            output_map_func=custom_map
        )
        
        x = torch.randn(2)
        output = layer(x)
        assert output.shape == (6,)
        assert torch.allclose(output.sum(), torch.tensor(1.0), atol=1e-6)

    def test_error_handling(self, basic_circuit):
        """Test error cases"""
        # Test input state size mismatch
        with pytest.raises(ValueError):
            QuantumLayer(
                input_size=2,
                circuit=basic_circuit,
                input_state=[1, 0],  # Wrong size
                trainable_parameters=["theta1", "theta2"],
                output_size=4
            )
        
        # Test invalid parameter name
        with pytest.raises(ValueError):
            QuantumLayer(
                input_size=2,
                circuit=basic_circuit,
                input_state=[1, 0, 1, 0],
                trainable_parameters=["theta1", "invalid_param"],  # Invalid name
                output_size=4
            )
        
        # Test output size mismatch with NONE strategy
        with pytest.raises(ValueError):
            QuantumLayer(
                input_size=2,
                circuit=basic_circuit,
                input_state=[1, 0, 1, 0],
                trainable_parameters=["theta1", "theta2"],
                output_size=4,  # Wrong size for NONE strategy
                output_mapping_strategy=OutputMappingStrategy.NONE
            )

    def test_gradient_flow(self, basic_circuit):
        """Test gradient computation and optimization"""
        layer = QuantumLayer(
            input_size=2,
            circuit=basic_circuit,
            input_state=[1, 0, 1, 0],
            trainable_parameters=["theta1", "theta2"],
            output_size=10
        )
        
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        x = torch.randn(5, 2)  # Batch of 5 inputs
        y = torch.randn(5, 10)  # Random targets
        y = y / y.sum(dim=1, keepdim=True)  # Normalize targets to be probability distributions
        
        # Run a few optimization steps
        initial_params = {name: param.clone() for name, param in layer.named_parameters()}
        
        for _ in range(3):
            optimizer.zero_grad()
            output = layer(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()
        
        # Verify parameters have been updated
        for name, param in layer.named_parameters():
            assert not torch.allclose(param, initial_params[name])


def test_change_circuit(basic_circuit):
    """Test changing the circuit dynamically"""
    layer = QuantumLayer(
        input_size=2,
        circuit=basic_circuit,
        input_state=[1, 0, 1, 0],
        trainable_parameters=["theta1", "theta2"],
        output_size=10
    )

    # Create a new valid circuit with same parameters
    new_circuit = pcvl.Circuit(4)
    new_circuit.add(0, pcvl.BS() // pcvl.PS(pcvl.P("theta1")) // pcvl.BS(), merge=True)
    new_circuit.add(2, pcvl.BS() // pcvl.PS(pcvl.P("theta2")) // pcvl.BS(), merge=True)
    new_circuit.add(1, pcvl.BS() // pcvl.PS(pcvl.P("x1")) // pcvl.BS() // pcvl.PS(pcvl.P("x2")), merge=True)

    # Test successful circuit change
    layer.change_circuit(new_circuit)
    assert layer.circuit == new_circuit

    # Test invalid circuit (different number of modes)
    invalid_circuit = pcvl.Circuit(3)
    with pytest.raises(ValueError, match="New circuit must have the same number of modes"):
        layer.change_circuit(invalid_circuit)

    # Test invalid circuit (different parameters)
    invalid_circuit = pcvl.Circuit(4)
    invalid_circuit.add(0, pcvl.BS() // pcvl.PS(pcvl.P("theta1")) // pcvl.BS(), merge=True)
    invalid_circuit.add(2, pcvl.BS() // pcvl.PS(pcvl.P("invalid_param")) // pcvl.BS(), merge=True)
    invalid_circuit.add(1, pcvl.BS() // pcvl.PS(pcvl.P("x1")) // pcvl.BS() // pcvl.PS(pcvl.P("x2")), merge=True)

    with pytest.raises(ValueError, match="New circuit parameters must match the original circuit"):
        layer.change_circuit(invalid_circuit)


def test_change_input_state(basic_circuit):
    """Test changing the input state dynamically"""
    layer = QuantumLayer(
        input_size=2,
        circuit=basic_circuit,
        input_state=[1, 0, 1, 0],
        trainable_parameters=["theta1", "theta2"],
        output_size=10
    )

    # Test successful input state change
    new_state = [0, 1, 0, 1]  # Same number of modes and photons
    layer.change_input_state(new_state)
    assert layer.input_state == new_state

    # Test invalid state (wrong number of modes)
    invalid_state = [1, 0, 1]
    with pytest.raises(ValueError, match="New input state must match the number of modes"):
        layer.change_input_state(invalid_state)

    # Test invalid state (different number of photons)
    invalid_state = [1, 0, 2, 0]
    with pytest.raises(ValueError, match="New input state must have the same number of photons"):
        layer.change_input_state(invalid_state)
