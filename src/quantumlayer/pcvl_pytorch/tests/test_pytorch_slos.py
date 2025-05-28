import pytest
import torch
import math
from typing import Tuple, Optional

import numpy as np
import perceval as pcvl
import pytest
import torch

from pcvl_pytorch import pytorch_slos_output_distribution, pytorch_slos_output_distribution_nobunching, threshold_mapping

# Constants
rEPS = 1e-5  # Tolerance for relative floating point comparisons
EPS = 1e-7  # Tolerance for absolute floating point comparisons


@pytest.fixture
def U_beamsplitter():
    """Fixture for a 50-50 beamsplitter unitary"""
    theta = torch.tensor(math.pi / 4)
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta), torch.cos(theta)]
    ], dtype=torch.float)


@pytest.fixture
def U_identity():
    """Fixture for identity matrix"""
    return torch.eye(2)


def test_vacuum_input(U_identity):
    """Test with vacuum state input, should raise an error"""
    input_state = [0, 0]
    with pytest.raises(ValueError):
        pytorch_slos_output_distribution(U_identity, input_state)


def test_single_photon(U_beamsplitter):
    """Test with single photon input"""
    input_state = [1, 0]
    keys, probs = pytorch_slos_output_distribution(U_beamsplitter, input_state)

    expected_probs = torch.tensor([0.5, 0.5])  # 50-50 beamsplitter
    assert len(keys) == 2
    assert set(keys) == {(1, 0), (0, 1)}
    assert torch.allclose(probs, expected_probs, atol=EPS)


def test_two_photon_bunching(U_beamsplitter):
    """Test Hong-Ou-Mandel type interference with two photons"""
    input_state = [1, 1]
    keys, probs = pytorch_slos_output_distribution(U_beamsplitter, input_state)

    # For a 50-50 beamsplitter, we expect photon bunching
    expected_states = {(2, 0), (0, 2), (1, 1)}
    expected_probs = torch.tensor([0.5, 0, 0.5])  # Theoretical HOM probabilities

    assert len(keys) == 3
    assert set(keys) == expected_states
    assert torch.allclose(probs, expected_probs, atol=EPS)


def nobunching_mapping(state: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Helper function for selective mapping"""
    if any(n > 1 for n in state):
        # Discard states with photon bunching
        return None
    return state


def test_threshold_detection(U_beamsplitter):
    """Test with threshold detection mapping"""
    input_state = [2, 0]  # Two photons in first mode
    keys, probs = pytorch_slos_output_distribution(
        U_beamsplitter,
        input_state,
        output_map_func=threshold_mapping
    )
    # After threshold detection, states like (2,0) should become (1,0)
    expected_states = {(1, 0), (0, 1), (1, 1)}
    assert set(keys) == expected_states
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=EPS)


def test_selective_mapping(U_beamsplitter):
    """Test with selective mapping that discards states"""
    input_state = [2, 0]
    keys, probs = pytorch_slos_output_distribution(
        U_beamsplitter,
        input_state,
        output_map_func=nobunching_mapping
    )

    # Should only keep states with no bunching
    assert len(keys) == 1  # only one state without bunching
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=EPS)


def test_unitary_properties():
    """Test that the output respects unitary transformation properties"""
    # Create a random 3x3 unitary matrix
    m = 3
    random_matrix = torch.randn(m, m, dtype=torch.cfloat)
    # QR decomposition
    Q, R = torch.linalg.qr(random_matrix)

    # Make Q uniquely determined by making diagonal elements of R real and positive
    d = torch.diagonal(R)
    ph = d / torch.abs(d)
    U = Q * ph.unsqueeze(0)

    input_state = [1, 0, 0]
    keys, probs = pytorch_slos_output_distribution(U, input_state)

    # Check probability conservation
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=EPS)

    # Check that probabilities match |U_{ij}|^2 for single photon
    expected_probs = torch.abs(U[:, 0]) ** 2
    sorted_probs, _ = torch.sort(probs)
    sorted_expected, _ = torch.sort(expected_probs)
    assert torch.allclose(sorted_probs, sorted_expected, atol=EPS)


def test_invalid_unitary():
    """Test error handling for non-square unitary"""
    invalid_U = torch.eye(3, 2)
    with pytest.raises(ValueError, match="Unitary matrices must be square"):
        pytorch_slos_output_distribution(invalid_U, [1, 0])


def test_invalid_input_state_length(U_beamsplitter):
    """Test error handling for input state length mismatch"""
    with pytest.raises(ValueError, match="Input state length must match unitary matrix dimension"):
        pytorch_slos_output_distribution(U_beamsplitter, [1, 0, 0])


def test_negative_photon_numbers(U_beamsplitter):
    """Test error handling for negative photon numbers"""
    with pytest.raises(ValueError, match="Photon numbers cannot be negative"):
        pytorch_slos_output_distribution(U_beamsplitter, [-1, 0])


@pytest.mark.parametrize("input_state,expected_outputs", [
    ([1, 0], [(1, 0), (0, 1)]),  # Single photon cases
    ([2, 0], [(2, 0), (1, 1), (0, 2)]),  # Two photon cases
])
def test_output_states(U_beamsplitter, input_state, expected_outputs):
    """Parametrized test for various input states"""
    keys, _ = pytorch_slos_output_distribution(U_beamsplitter, input_state)
    assert set(keys) == set(expected_outputs)


def create_parameterized_beamsplitter(theta: torch.Tensor) -> torch.Tensor:
    """Helper function to create a parameterized beamsplitter without breaking the autograd chain."""
    row1 = torch.stack([torch.cos(theta), -torch.sin(theta)])
    row2 = torch.stack([torch.sin(theta),  torch.cos(theta)])
    # Stack the rows to form a 2x2 matrix
    U = torch.stack([row1, row2])

    # Convert to complex type while preserving the computation graph.
    return U.to(torch.cfloat)


def test_gradient_single_photon():
    """Test gradient flow for single photon input"""
    theta = torch.tensor(math.pi / 4, requires_grad=True)
    input_state = [1, 0]

    # Forward pass
    U = create_parameterized_beamsplitter(theta)
    U.retain_grad() # necessary for the check below since it is not a leaf node
    keys, probs = pytorch_slos_output_distribution(U, input_state)

    # Calculate loss (let's say we want to maximize probability in first mode)
    loss = -probs[0]  # Negative because we want to maximize

    # Backward pass
    loss.backward()

    # Check that gradients were computed
    assert U.grad is not None
    assert theta.grad is not None
    assert not torch.isnan(theta.grad).any()
    assert not torch.isinf(theta.grad).any()


def test_gradient_two_photon():
    """Test gradient flow for two-photon HOM interference"""
    theta = torch.tensor(math.pi / 4, requires_grad=True)
    input_state = [1, 1]

    # Forward pass
    U = create_parameterized_beamsplitter(theta)
    keys, probs = pytorch_slos_output_distribution(U, input_state)

    # Calculate loss (let's maximize bunching probability)
    bunching_idx = [i for i, k in enumerate(keys) if k in {(2, 0), (0, 2)}]
    bunching_prob = sum(probs[i] for i in bunching_idx)
    loss = -bunching_prob

    # Backward pass
    loss.backward()

    # Check that gradients were computed
    assert theta.grad is not None
    assert not torch.isnan(theta.grad).any()
    assert not torch.isinf(theta.grad).any()


def test_gradient_with_mapping():
    """Test gradient flow with output state mapping"""
    theta = torch.tensor(math.pi / 4, requires_grad=True)
    input_state = [2, 0]

    # Forward pass with threshold detection
    U = create_parameterized_beamsplitter(theta)
    keys, probs = pytorch_slos_output_distribution(
        U,
        input_state,
        output_map_func=threshold_mapping
    )

    # Calculate loss (maximize detection probability in first mode)
    first_mode_idx = [i for i, k in enumerate(keys) if k[0] == 1]
    if first_mode_idx:  # Should always be true for this input
        loss = -probs[first_mode_idx[0]]

        # Backward pass
        loss.backward()

        # Check that gradients were computed
        assert theta.grad is not None
        assert not torch.isnan(theta.grad).any()
        assert not torch.isinf(theta.grad).any()


def test_optimization_loop():
    """Test that we can run an optimization loop"""
    theta = torch.tensor(0.1, requires_grad=True)  # Start far from optimal value
    input_state = [1, 1]
    optimizer = torch.optim.Adam([theta], lr=0.1)

    initial_bunching = None
    final_bunching = None

    # Run a few optimization steps
    for _ in range(10):
        optimizer.zero_grad()

        U = create_parameterized_beamsplitter(theta)
        keys, probs = pytorch_slos_output_distribution(U, input_state)

        # Calculate bunching probability
        bunching_idx = [i for i, k in enumerate(keys) if k in {(2, 0), (0, 2)}]
        bunching_prob = sum(probs[i] for i in bunching_idx)

        # Store initial bunching probability
        if initial_bunching is None:
            initial_bunching = bunching_prob.item()

        # Loss to maximize bunching
        loss = -bunching_prob
        loss.backward()
        optimizer.step()

        final_bunching = bunching_prob.item()

    # Check that optimization improved bunching probability
    assert final_bunching > initial_bunching


def test_batched_single_photon(U_beamsplitter):
    """Test batched computation with single photon input"""
    batch_size = 3
    # Create a batch of different beamsplitters
    angles = torch.linspace(0, math.pi / 2, batch_size)
    batch_U = torch.stack([
        create_parameterized_beamsplitter(theta).squeeze()
        for theta in angles
    ])

    input_state = [1, 0]
    keys, probs = pytorch_slos_output_distribution(batch_U, input_state)

    # Check output shape
    assert probs.shape == (batch_size, 2)
    assert len(keys) == 2
    assert set(keys) == {(1, 0), (0, 1)}

    # For each beamsplitter, probabilities should sum to 1
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=EPS)

    # First beamsplitter (theta=0) should not split
    assert torch.allclose(probs[0], torch.tensor([1.0, 0.0]), atol=EPS)
    # Middle beamsplitter (theta=pi/4) should split 50-50
    assert torch.allclose(probs[1], torch.tensor([0.5, 0.5]), atol=EPS)
    # Last beamsplitter (theta=pi/2) should fully transmit
    assert torch.allclose(probs[2], torch.tensor([0.0, 1.0]), atol=EPS)


def test_batched_two_photon_bunching():
    """Test batched computation with two-photon HOM interference"""
    batch_size = 2
    # Create parameters for two different beamsplitters
    thetas = torch.tensor([math.pi / 4, math.pi / 3], requires_grad=True)
    batch_U = torch.stack([
        create_parameterized_beamsplitter(theta).squeeze()
        for theta in thetas
    ])

    input_state = [1, 1]
    keys, probs = pytorch_slos_output_distribution(batch_U, input_state)

    # Check output dimensions
    assert probs.shape == (batch_size, 3)
    assert len(keys) == 3
    assert set(keys) == {(2, 0), (1, 1), (0, 2)}

    # For 50-50 beamsplitter (first one), verify HOM interference
    assert torch.allclose(probs[0], torch.tensor([0.5, 0.0, 0.5]), atol=EPS)

    # Check probability conservation for all beamsplitters
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=EPS)


def test_batched_gradient_flow():
    """Test gradient flow in batched computation"""
    batch_size = 2
    thetas = torch.tensor([0.1, 0.8], requires_grad=True)
    input_state = [1, 1]

    # Forward pass with batched unitaries
    batch_U = torch.stack([
        create_parameterized_beamsplitter(theta).squeeze()
        for theta in thetas
    ])
    keys, probs = pytorch_slos_output_distribution(batch_U, input_state)

    # Calculate loss (maximize bunching for all beamsplitters)
    bunching_idx = [i for i, k in enumerate(keys) if k in {(2, 0), (0, 2)}]
    bunching_probs = sum(probs[:, i] for i in bunching_idx)
    loss = -bunching_probs.sum()  # Total bunching across batch

    # Backward pass
    loss.backward()

    # Check gradients
    assert thetas.grad is not None
    assert not torch.isnan(thetas.grad).any()
    assert not torch.isinf(thetas.grad).any()

    # Verify each theta got different gradients
    assert not torch.allclose(thetas.grad[0], thetas.grad[1], atol=EPS)


def test_batched_with_mapping():
    """Test batched computation with output state mapping"""
    batch_size = 2
    thetas = torch.tensor([math.pi / 4, math.pi / 3], requires_grad=True)
    batch_U = torch.stack([
        create_parameterized_beamsplitter(theta).squeeze()
        for theta in thetas
    ])

    input_state = [2, 0]
    keys, probs = pytorch_slos_output_distribution(
        batch_U,
        input_state,
        output_map_func=threshold_mapping
    )

    # Check output dimensions
    assert probs.shape[0] == batch_size
    assert set(keys) == {(1, 0), (0, 1), (1, 1)}

    # Verify probability conservation after mapping
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=EPS)


def test_slos_compare():
    """Test that the PyTorch implementation matches the Perceval SLOS backend"""
    U = pcvl.MatrixN.random_unitary(6)
    torch_U = torch.from_numpy(U)
    slos_backend = pcvl.BackendFactory().get_backend("SLOS")
    slos_backend.set_circuit(pcvl.Unitary(U))

    for input_state in [[1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1]]:
        keys, torch_probs = pytorch_slos_output_distribution(torch_U, input_state)
        slos_backend.set_input_state(pcvl.BasicState(input_state))
        slos_distribution = slos_backend.prob_distribution()

        # Check that we have same number of keys
        assert len(keys) == len(slos_distribution)

        for idx, k  in enumerate(keys):
            assert np.isclose(torch_probs[idx], slos_distribution[pcvl.BasicState(k)], rtol=rEPS)


def test_nobunching():
    """Test that the no bunching implementation works the same than general one"""
    U = pcvl.MatrixN.random_unitary(6)
    torch_U = torch.from_numpy(U)

    for input_state in [[1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1]]:
        keys_general, torch_probs_general = pytorch_slos_output_distribution(torch_U, input_state,
                                                                             output_map_func=nobunching_mapping)
        keys_no_bunching, torch_probs_no_bunching = pytorch_slos_output_distribution_nobunching(torch_U, input_state)

        # Check that we have same number of keys
        assert len(keys_general) == len(keys_no_bunching)

        for idx, k  in enumerate(keys_general):
            assert np.isclose(torch_probs_general[idx], torch_probs_no_bunching[keys_no_bunching.index(k)], rtol=rEPS)
