import pytest
import torch
from torch import nn

from nn_models import MLP, MLPConfig

@pytest.fixture
def basic_config():
    return MLPConfig(
        hidden_sizes=[64, 32],
        dropout=0.1,
        activation='relu',
        normalization='batch'
    )

@pytest.fixture
def basic_mlp(basic_config):
    return MLP(input_size=10, output_size=2, config=basic_config)

def test_mlp_initialization(basic_config):
    """Test that MLP initializes correctly with valid parameters"""
    mlp = MLP(input_size=10, output_size=2, config=basic_config)
    
    # Check that the network has the correct number of layers
    # Count expected layers: 2 hidden layers + output layer
    # Each hidden layer has: Linear + BatchNorm + ReLU + Dropout
    # Output layer has just Linear
    expected_layers = (4 * 2) + 1
    assert len(list(mlp.network)) == expected_layers

    # Verify input and output dimensions of first and last layers
    first_layer = mlp.network[0]
    assert isinstance(first_layer, nn.Linear)
    assert first_layer.in_features == 10
    assert first_layer.out_features == 64

    last_layer = mlp.network[-1]
    assert isinstance(last_layer, nn.Linear)
    assert last_layer.in_features == 32
    assert last_layer.out_features == 2

def test_mlp_forward_pass(basic_mlp):
    """Test forward pass with sample input"""
    batch_size = 16
    input_size = 10
    x = torch.randn(batch_size, input_size)
    
    output = basic_mlp(x)
    
    assert output.shape == (batch_size, 2)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_weight_initialization(basic_mlp):
    """Test both xavier and kaiming weight initialization"""
    # Test Xavier initialization
    basic_mlp.initialize_weights(method='xavier')
    for layer in basic_mlp.network:
        if isinstance(layer, nn.Linear):
            # Check if weights are initialized (not all zeros or ones)
            assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
            assert not torch.allclose(layer.weight, torch.ones_like(layer.weight))
            # Check if biases are initialized to zero
            assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    # Test Kaiming initialization
    basic_mlp.initialize_weights(method='kaiming')
    for layer in basic_mlp.network:
        if isinstance(layer, nn.Linear):
            assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
            assert not torch.allclose(layer.weight, torch.ones_like(layer.weight))
            assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

def test_invalid_activation():
    """Test that invalid activation function raises ValueError"""
    invalid_config = MLPConfig(
        hidden_sizes=[64, 32],
        activation='invalid_activation'
    )
    
    with pytest.raises(ValueError) as exc_info:
        MLP(input_size=10, output_size=2, config=invalid_config)
    assert "Unsupported activation function" in str(exc_info.value)

def test_invalid_weight_initialization(basic_mlp):
    """Test that invalid weight initialization method raises ValueError"""
    with pytest.raises(ValueError) as exc_info:
        basic_mlp.initialize_weights(method='invalid_method')
    assert "Unsupported initialization method" in str(exc_info.value)

def test_different_normalizations():
    """Test MLP with different normalization options"""
    # Test with layer normalization
    layer_norm_config = MLPConfig(
        hidden_sizes=[64, 32],
        activation='relu',
        normalization='layer'
    )
    layer_norm_mlp = MLP(input_size=10, output_size=2, config=layer_norm_config)
    
    # Test with no normalization
    no_norm_config = MLPConfig(
        hidden_sizes=[64, 32],
        activation='relu',
        normalization=None
    )
    no_norm_mlp = MLP(input_size=10, output_size=2, config=no_norm_config)
    
    # Verify forward pass works for both
    x = torch.randn(16, 10)
    assert layer_norm_mlp(x).shape == (16, 2)
    assert no_norm_mlp(x).shape == (16, 2)

def test_different_activations():
    """Test MLP with different activation functions"""
    activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu', 'selu']
    
    for activation in activations:
        config = MLPConfig(
            hidden_sizes=[64, 32],
            activation=activation
        )
        mlp = MLP(input_size=10, output_size=2, config=config)
        
        # Test forward pass
        x = torch.randn(16, 10)
        output = mlp(x)
        assert output.shape == (16, 2)
        assert not torch.isnan(output).any()

def test_dropout():
    """Test that dropout behaves differently in train vs eval mode"""
    config = MLPConfig(
        hidden_sizes=[64, 32],
        dropout=0.5,
        activation='relu'
    )
    mlp = MLP(input_size=10, output_size=2, config=config)
    
    x = torch.randn(100, 10)
    
    # Test in training mode
    mlp.train()
    train_outputs = torch.stack([mlp(x) for _ in range(10)])
    
    # Test in eval mode
    mlp.eval()
    eval_outputs = torch.stack([mlp(x) for _ in range(10)])
    
    # In train mode, outputs should vary due to dropout
    assert not torch.allclose(train_outputs[0], train_outputs[1])
    
    # In eval mode, outputs should be consistent
    assert torch.allclose(eval_outputs[0], eval_outputs[1])