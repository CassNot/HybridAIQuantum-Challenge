# sanity check with tests cases to test the training on Clements 4x4

from slos_torch import pytorch_slos_output_distribution
import torch
import torch.nn as nn
import math as math
import perceval as pcvl
from pcvl2torch import pcvl_circuit_to_pytorch_unitary

if __name__ == "__main__":
    # Create a basic test case with a simple unitary (Hadamard-like) and single photon input
    print("### Test case for a Hadamard gate ###")
    m = 2  # 2-mode circuit
    unitary = torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / math.sqrt(2)
    input_state = [1, 1]  # Single photon in first mode

    # Make unitary parameters differentiable
    unitary.requires_grad = True

    # Compute output probabilities
    keys, probs = pytorch_slos_output_distribution(unitary, input_state)

    print("Test case: Two Single photons through a balanced beam splitter - HOM interference")
    print("\nUnitary matrix:")
    print(unitary)
    print("\nInput state:", input_state)
    print("\nOutput Fock states:", keys)
    print("\nOutput probabilities:", probs.detach())

    # Test differentiability
    try:
        loss = probs[0]  # probability of first output configuration
        loss.backward()
        print("\nGradients exist:", unitary.grad is not None)
        print("Gradient of unitary:")
        print(unitary.grad)
    except Exception as e:
        print("\nError testing differentiability:", e)

    print("### Test case for a Clements with 4 modes ###")
    print("### Batch of 1 ###")
    # 4 modes
    m = 4
    parameters = [pcvl.P(f"phi_{i}") for i in range(0, m * (m - 1))]
    circuit = pcvl.GenericInterferometer(m, lambda i: (pcvl.BS()
                                                       .add(1, pcvl.PS(parameters[2 * i]))
                                                       .add(0, pcvl.BS())
                                                       .add(1, pcvl.PS(parameters[2 * i + 1]))
                                                       )
                                         )
    pcvl.pdisplay(circuit)
    # target phases
    phases_target = torch.abs(2 * torch.pi * torch.randn(m * (m - 1), dtype=torch.float))
    # initialisation of the parameters phases we want to learn
    initial_phases = nn.Parameter(torch.abs(2 * torch.pi * torch.randn(m * (m - 1), dtype=torch.float)), requires_grad=True)
    input_state = [1, 0, 1, 0]
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([initial_phases], lr=0.01)
    try:
        matrix_target = pcvl_circuit_to_pytorch_unitary(circuit=circuit, circuit_parameters=phases_target)
        print("\n Matrix target: ",matrix_target)
        keys, probs = pytorch_slos_output_distribution(matrix_target, input_state)
        print(f"\n Keys = {keys} and probs = {probs}")
        print(f"\n Sum of the probs = {probs.sum()}")

        learned_matrix = pcvl_circuit_to_pytorch_unitary(circuit=circuit, circuit_parameters=initial_phases)
        learned_keys, learned_probs = pytorch_slos_output_distribution(learned_matrix, input_state)
        print(f"\n From matrix = {learned_matrix}"
              f"\nwe have learned probs = {learned_probs}")
        loss = criterion(learned_probs, probs)
        print(f"\n Loss = {loss}")
        loss.backward()
        print("\nGradients exist:", matrix_target.grad is not None, initial_phases.grad is not None)
        print("\nGradient of learned matrix: ", learned_matrix.grad)
        print("\nGradient of the phases: ", initial_phases.grad)
        print(f"\nPhases before optimization: {initial_phases}")
        optimizer.step()
        print("\nUpdated parameters: ", initial_phases)
        learned_matrix = pcvl_circuit_to_pytorch_unitary(circuit=circuit, circuit_parameters=initial_phases)
        learned_keys, learned_probs = pytorch_slos_output_distribution(learned_matrix, input_state)
        print(f"\n Updated probs = {learned_probs}")

    except Exception as e:
        print("\nError testing differentiability of 4 modes Circuit:", e)

    N = 3
    print(f"### Batch of N = {N} ###")
    input_state = [1, 0, 0, 0]
    try:
        # target parameters
        batch_phases_target = torch.abs(2 * torch.pi * torch.randn(N, m * (m - 1), dtype=torch.float))
        batch_matrix_target = pcvl_circuit_to_pytorch_unitary(circuit=circuit, circuit_parameters=batch_phases_target)
        batch_keys, batch_probs = pytorch_slos_output_distribution(batch_matrix_target, input_state)
        # initialisation
        batch_initial_phases = nn.Parameter(torch.abs(2 * torch.pi * torch.randn(N, m * (m - 1), dtype=torch.float)),
                                      requires_grad=True)
        # optimizer
        batch_optimizer = torch.optim.Adam([batch_initial_phases], lr=0.01)
        print(f"\n Batch target = {batch_matrix_target.shape}")
        print(f"\n Gives probabilities = {batch_probs.shape}")
        # first forward pass before updating
        batch_learned_matrix = pcvl_circuit_to_pytorch_unitary(circuit=circuit, circuit_parameters=batch_initial_phases)
        batch_learned_keys, batch_learned_probs = pytorch_slos_output_distribution(batch_learned_matrix, input_state)
        print(f"\n From matrix = {learned_matrix.shape}"
              f"\nwe have learned probs = {batch_learned_probs}")
        # compute loss and gradients
        batch_loss = criterion(batch_learned_probs, batch_probs)
        batch_loss.backward()
        print(f"\n Loss = {batch_loss.item()}")
        print("\nGradients exist:", batch_learned_matrix.grad is not None, batch_initial_phases.grad is not None)
        print("\nGradient of learned matrix: ", batch_learned_matrix.grad)
        print("\nGradient for the phases: ", batch_initial_phases.grad)
        print(f"\nPhases before optimization: {batch_initial_phases}")
        # update parameters
        batch_optimizer.step()
        print("\nUpdated parameters: ", batch_initial_phases)
        # observe updated parameters
        batch_learned_matrix = pcvl_circuit_to_pytorch_unitary(circuit=circuit, circuit_parameters=batch_initial_phases)
        batch_learned_keys, batch_learned_probs = pytorch_slos_output_distribution(batch_learned_matrix, input_state)
        print(f"\n Updated probs = {batch_learned_probs}")
        new_batch_loss = criterion(batch_learned_probs, batch_probs)
        print(f"\n New loss = {new_batch_loss}")
    except Exception as e:
        print("\nError testing differentiability of 4 batch modes Circuit:", e)

