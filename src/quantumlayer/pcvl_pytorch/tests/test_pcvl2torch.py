import pytest
import torch
import perceval as pcvl
import sympy as sp
import numpy as np
from torch import nn

from pcvl_pytorch import sympy2torch, pcvl_circuit_to_pytorch_unitary

@pytest.fixture
def simple_mzi_circuit():
    """Fixture providing a simple Mach-Zehnder interferometer circuit"""
    circuit = pcvl.Circuit(2)
    circuit.add(0, pcvl.BS())
    circuit.add(0, pcvl.PS(pcvl.P("phi1")))
    circuit.add(0, pcvl.BS())
    circuit.add(0, pcvl.PS(pcvl.P("phi2")))
    return circuit

@pytest.fixture
def complex_circuit():
    """Fixture providing a more complex quantum circuit with multiple parameters"""
    circuit = pcvl.Circuit(3)
    circuit.add(0, pcvl.BS())
    circuit.add(1, pcvl.BS())
    circuit.add(0, pcvl.PS(pcvl.P("phi1")))
    circuit.add(1, pcvl.PS(pcvl.P("phi2")))
    circuit.add(0, pcvl.BS())
    circuit.add(1, pcvl.BS())
    return circuit


def test_sympy2torch_basic_operations():
    """Test basic mathematical operations conversion from Sympy to PyTorch"""
    params = {"x": torch.tensor([2.0]), "y": torch.tensor([3.0])}

    # Test addition
    expr = sp.sympify("x + y")
    result = sympy2torch(expr, params, batch_size=1)
    assert torch.allclose(result, torch.tensor([5.0]))

    # Test multiplication
    expr = sp.sympify("x * y")
    result = sympy2torch(expr, params, batch_size=1)
    assert torch.allclose(result, torch.tensor([6.0]))

    # Test power
    expr = sp.sympify("x**2")
    result = sympy2torch(expr, params, batch_size=1)
    assert torch.allclose(result, torch.tensor([4.0]))

    expr = sp.sympify("I")
    result = sympy2torch(expr, params, batch_size=1)
    assert torch.allclose(result, torch.tensor([1j]))

def test_sympy2torch_basic_operations_batched():
    """Test basic mathematical operations conversion from Sympy to PyTorch"""
    params = {"x": torch.tensor([2.0, 4]), "y": torch.tensor([3.0, 9])}

    # Test addition
    expr = sp.sympify("x + y")
    result = sympy2torch(expr, params, batch_size=2)
    assert torch.allclose(result, torch.tensor([5.0, 13]))

    # Test multiplication
    expr = sp.sympify("x * y")
    result = sympy2torch(expr, params, batch_size=2)
    assert torch.allclose(result, torch.tensor([6.0, 36]))

    # Test power
    expr = sp.sympify("x**2")
    result = sympy2torch(expr, params, batch_size=2)
    assert torch.allclose(result, torch.tensor([4.0, 16]))

    expr = sp.sympify("I")
    result = sympy2torch(expr, params, batch_size=2)
    assert torch.allclose(result, torch.tensor([1j, 1j]))

def test_sympy2torch_trigonometric():
    """Test trigonometric functions conversion"""
    expr = sp.sympify("sin(x)")
    params = {"x": torch.tensor([0, np.pi/2])}
    result = sympy2torch(expr, params, batch_size=2)
    assert torch.allclose(result, torch.tensor([0, 1.0]), atol=1e-6)

    expr = sp.sympify("cos(x)")
    result = sympy2torch(expr, params, batch_size=2)
    assert torch.allclose(result, torch.tensor([1.0, 0.0]), atol=1e-6)


def test_sympy2torch_complex_numbers():
    """Test complex number handling"""
    expr = sp.sympify("1 + I")
    result = sympy2torch(expr, {}, batch_size=1).squeeze()
    assert torch.allclose(result, torch.tensor(1 + 1j))

    expr = sp.sympify("exp(I*pi/2)")
    result = sympy2torch(expr, {}, batch_size=1).squeeze()
    assert torch.allclose(result, torch.tensor(1j), atol=1e-6)


def test_sympy2torch_with_gradients():
    """Test sympy2torch with tensors that require gradients"""
    # Create tensor with requires_grad=True
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    params = {"x": x, "y": y}

    # Test with a simple expression
    expr = sp.sympify("x**2 + 2*y")
    result = sympy2torch(expr, params, batch_size=1)

    # Check that result requires grad
    assert result.dim() == 1
    assert result.requires_grad

    # Compute gradients
    result.backward()

    # Check gradient values
    assert x.grad is not None
    assert y.grad is not None
    assert torch.allclose(x.grad, torch.tensor([4.0]))  # d(x^2)/dx = 2x
    assert torch.allclose(y.grad, torch.tensor([2.0]))  # d(2y)/dy = 2


def test_circuit_to_unitary_simple(simple_mzi_circuit):
    """Test conversion of a simple MZI circuit to unitary"""
    params = torch.zeros((2,), requires_grad=True)
    unitary = pcvl_circuit_to_pytorch_unitary(simple_mzi_circuit, params)
    
    # Check shape
    assert unitary.shape == (2, 2)
    
    # Check unitarity
    identity = torch.eye(2, dtype=torch.complex64)
    assert torch.allclose(unitary @ unitary.conj().T, identity, atol=1e-6)
    assert torch.allclose(unitary.conj().T @ unitary, identity, atol=1e-6)


def test_circuit_to_unitary_complex(complex_circuit):
    """Test conversion of a complex circuit to unitary"""
    unitary = pcvl_circuit_to_pytorch_unitary(complex_circuit, torch.zeros((len(complex_circuit.get_parameters()))))
    
    # Check shape
    assert unitary.shape == (3, 3)
    
    # Check unitarity
    identity = torch.eye(3, dtype=torch.complex64)
    assert torch.allclose(unitary @ unitary.conj().T, identity, atol=1e-6)
    assert torch.allclose(unitary.conj().T @ unitary, identity, atol=1e-6)


def test_circuit_to_unitary_batch(simple_mzi_circuit):
    """Test batch processing of circuit to unitary conversion"""
    # Create a batch of parameters (3 sets of parameters)
    params = torch.tensor([[0.1, 0.2],
                           [0.3, 0.4],
                           [0.5, 0.6]], requires_grad=True)

    # Convert circuit to unitary with batched parameters
    unitary = pcvl_circuit_to_pytorch_unitary(simple_mzi_circuit, params)

    # Check batch shape
    assert unitary.shape == (3, 2, 2)  # (batch_size, n_modes, n_modes)

    # Check unitarity for each matrix in the batch
    identity = torch.eye(2, dtype=torch.complex64)
    for i in range(3):
        assert torch.allclose(unitary[i] @ unitary[i].conj().T, identity, atol=1e-6)
        assert torch.allclose(unitary[i].conj().T @ unitary[i], identity, atol=1e-6)

    # Test gradient computation
    loss = torch.sum(torch.abs(unitary[:, 0, 0]) ** 2)  # Sum over batch
    loss.backward()

    # Check if gradients were computed
    assert params.grad is not None
    assert params.grad.shape == (3, 2)  # Gradients for each parameter in the batch
    assert not torch.allclose(params.grad, torch.zeros_like(params.grad))


def test_circuit_to_unitary_batch_manualmap(simple_mzi_circuit):
    """Test batch processing of circuit to unitary conversion"""
    # Create a batch of parameters (3 sets of parameters)
    weight = nn.Parameter(torch.tensor(1.0))
    params = {"phi1": torch.tensor([0.1, 0.1, 0.3]), "phi2": weight}

    # Convert circuit to unitary with batched parameters
    unitary = pcvl_circuit_to_pytorch_unitary(simple_mzi_circuit, params)

    # Check batch shape
    assert unitary.shape == (3, 2, 2)  # (batch_size, n_modes, n_modes)
    assert torch.allclose(unitary[0], unitary[1])
    assert not torch.allclose(unitary[1], unitary[2])


def test_parameter_gradients(simple_mzi_circuit):
    """Test that gradients can be computed through the unitary"""
    params = torch.tensor([0.1, 0.2], requires_grad=True)
    unitary = pcvl_circuit_to_pytorch_unitary(simple_mzi_circuit, params)
    print(unitary)
    # Compute loss (e.g., magnitude of first element)
    print(unitary[0, 0])
    loss = torch.abs(unitary[0, 0]) ** 2
    
    # Check if gradients can be computed
    loss.backward()
    assert params.grad is not None
    print(params.grad)
    assert not torch.allclose(params.grad, torch.zeros_like(params.grad))


def test_error_handling():
    """Test error handling for invalid inputs"""
    # Test with no parameters
    with pytest.raises(AttributeError):
        pcvl_circuit_to_pytorch_unitary(simple_mzi_circuit, None)

    # Test with invalid circuit
    with pytest.raises(AttributeError):
        pcvl_circuit_to_pytorch_unitary(None, torch.tensor(0))

    # Test with invalid number of parameters
    with pytest.raises(AttributeError):
        pcvl_circuit_to_pytorch_unitary(simple_mzi_circuit, torch.tensor([1]))

    # Test sympy2torch with invalid parameter mapping
    expr = sp.sympify("x + y")
    with pytest.raises(TypeError):
        sympy2torch(expr, torch.tensor([1.0]))
    with pytest.raises(KeyError):
        sympy2torch(expr, {"x": torch.tensor([1.0])}, batch_size=1)  # Missing 'y' parameter


def test_matrix_conversion():
    """Test conversion of Perceval matrix types"""
    # Test with static matrix
    matrix = pcvl.utils.matrix.MatrixS([[1, 0], [0, 1]])
    result = sympy2torch(matrix, {}, batch_size=1)
    expected = torch.eye(2, dtype=torch.complex64).unsqueeze(0)
    assert torch.allclose(result, expected)

    # Test with parametric matrix
    theta = sp.Symbol('theta')
    phi = sp.Symbol('phi')
    parametric_matrix = pcvl.utils.matrix.MatrixS([
        [sp.cos(theta), -sp.exp(sp.I * phi) * sp.sin(theta)],
        [sp.exp(-sp.I * phi) * sp.sin(theta), sp.cos(theta)]
    ])

    # Create parameters with gradients
    params = {
        'theta': torch.tensor([np.pi / 4], requires_grad=True),
        'phi': torch.tensor([np.pi / 2], requires_grad=True)
    }

    result = sympy2torch(parametric_matrix, params, batch_size=1)

    # Check shape and type
    assert result.shape == (1, 2, 2)
    assert result.dtype == torch.complex64
    assert result.requires_grad

    # Compute some loss (e.g., magnitude of first element)
    loss = torch.abs(result[0, 0, 0]) ** 2
    loss.backward()

    # Check that gradients were computed
    assert params['theta'].grad is not None
    assert params['phi'].grad is not None

    # Test unitarity of the resulting matrix
    identity = torch.eye(2, dtype=torch.complex64)
    assert torch.allclose(result[0] @ result[0].conj().T, identity, atol=1e-6)
    assert torch.allclose(result[0].conj().T @ result[0], identity, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])