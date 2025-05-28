"""
This module provides functionality to convert Perceval quantum circuits to PyTorch tensors
for differentiable quantum computing.

Author: Jean Senellart

The symbolic function mapping logic is inspired by SympyTorch: https://github.com/patrick-kidger/sympytorch
    Copyright 2021 Patrick Kidger
    Licensed under the Apache License, Version 2.0 (function mapping section)
"""

import functools as ft
import numbers
from typing import Any, Callable, TypeVar, Union, Dict

import perceval as pcvl
import sympy as sp
import torch
import torch.nn as nn

# Type variable for generic function typing
T = TypeVar("T")


# Helper function to reduce multiple arguments using a binary function
def _reduce(fn: Callable[..., T]) -> Callable[..., T]:
    """
    Creates a reduction function that applies a binary operation repeatedly.
    Useful for converting n-ary Sympy operations to binary PyTorch operations.
    """

    def fn_(*args: Any) -> T:
        return ft.reduce(fn, args)

    return fn_


# Helper function to create imaginary unit tensor
def _imaginary_fnc(*_: Any) -> torch.Tensor:
    """Returns the imaginary unit as a PyTorch tensor"""
    return torch.tensor(1j)


# Mapping between Sympy operations and their PyTorch equivalents
SYMPY_TO_TORCH_OPS = {
    # Basic arithmetic
    sp.Mul: _reduce(torch.mul),
    sp.Add: _reduce(torch.add),
    sp.div: torch.div,
    sp.Pow: torch.pow,

    # Basic mathematical functions
    sp.Abs: torch.abs,
    sp.sign: torch.sign,
    sp.ceiling: torch.ceil,
    sp.floor: torch.floor,
    sp.log: torch.log,
    sp.exp: torch.exp,
    sp.sqrt: torch.sqrt,

    # Trigonometric functions
    sp.cos: torch.cos,
    sp.sin: torch.sin,
    sp.tan: torch.tan,
    sp.acos: torch.acos,
    sp.asin: torch.asin,
    sp.atan: torch.atan,
    sp.atan2: torch.atan2,

    # Hyperbolic functions
    sp.cosh: torch.cosh,
    sp.sinh: torch.sinh,
    sp.tanh: torch.tanh,
    sp.acosh: torch.acosh,
    sp.asinh: torch.asinh,
    sp.atanh: torch.atanh,

    # Complex operations
    sp.re: torch.real,
    sp.im: torch.imag,
    sp.arg: torch.angle,
    sp.core.numbers.ImaginaryUnit: _imaginary_fnc,
    sp.conjugate: torch.conj,

    # Special functions
    sp.erf: torch.erf,
    sp.loggamma: torch.lgamma,

    # Comparison operations
    sp.Eq: torch.eq,
    sp.Ne: torch.ne,
    sp.StrictGreaterThan: torch.gt,
    sp.StrictLessThan: torch.lt,
    sp.LessThan: torch.le,
    sp.GreaterThan: torch.ge,

    # Logical operations
    sp.And: torch.logical_and,
    sp.Or: torch.logical_or,
    sp.Not: torch.logical_not,

    # Min/Max operations
    sp.Max: torch.max,
    sp.Min: torch.min,

    # Matrix operations
    sp.MatAdd: torch.add,
    sp.HadamardProduct: torch.mul,
    sp.Trace: torch.trace,
    sp.Determinant: torch.det,
}

def sympy2torch(sympy_object, map_params, batch_size):
    """
    Converts recursively a Sympy expression to a PyTorch tensor, expect a batch of parameters mapped in map_params.

    Args:
        sympy_object: A Sympy expression, matrix, or number
        map_params: Dictionary mapping parameter names to their PyTorch values
        batch_size: Number of samples in the batch

    Returns:
        torch.Tensor: The PyTorch equivalent of the input
    """
    # Handle Perceval's matrix type
    if isinstance(sympy_object, pcvl.utils.matrix.Matrix):
        t_object = torch.empty((batch_size, *sympy_object.shape), dtype=torch.complex64)
        for i in range(sympy_object.shape[0]):
            for j in range(sympy_object.shape[1]):
                t_object[:, i, j] = sympy2torch(sympy_object[i, j], map_params, batch_size)

    # Handle symbolic parameters: return the corresponding tensor from map_params
    elif isinstance(sympy_object, sp.Symbol):
        t_object = map_params[sympy_object.name]

    # Handle numerical values
    elif isinstance(sympy_object, sp.Number) or isinstance(sympy_object, numbers.Number):
        if (isinstance(sympy_object, sp.Number) and sympy_object.is_real) or not isinstance(sympy_object, complex):
            t_object = torch.full((batch_size,), float(sympy_object), dtype=torch.float32)
        else:
            t_object = torch.full((batch_size,), complex(sympy_object), dtype=torch.complex64)

    # Handle operations (functions, operators) with a recursive call on the arguments
    else:
        t_object = SYMPY_TO_TORCH_OPS[sympy_object.func](
            *[sympy2torch(arg, map_params, batch_size=1) for arg in sympy_object.args]
        )
        if t_object.dim() == 0:
            t_object = t_object.unsqueeze(0).repeat(batch_size)

    return t_object


def pcvl_circuit_to_pytorch_unitary(circuit: pcvl.Circuit, circuit_parameters: Union[torch.Tensor, Dict[str, torch.Tensor]]):
    """
    Converts a parameterized Perceval circuit to a PyTorch unitary matrix.
    Supports batch processing if torch_parameters is a 2D tensor.

    Args:
        circuit: Perceval Circuit object
        circuit_parameters:  either PyTorch parameters for the circuit. Can be a 2D tensor for batch processing.
                             or map name->tensor (again can be a 2D tensor)

    Returns:
        tuple: (parameters, unitary_matrix)
            - parameters: PyTorch parameters of the circuit (or batch of parameters)
            - unitary_matrix: PyTorch tensor representing the circuit's unitary (or batch of unitaries)
    """

    if isinstance(circuit_parameters, torch.Tensor):
        torch_parameters = circuit_parameters
        if torch_parameters.dim() > 2:
            raise AttributeError("torch_parameters must be a 1D or 2D tensor")

        # Get circuit parameters
        circuit_parameters = circuit.get_parameters()
        if (torch_parameters.dim() == 0 and len(circuit.get_parameters())
            ) or torch_parameters.shape[-1] != len(circuit.get_parameters()):
            raise AttributeError("torch_parameters must match the circuit parameters")

        # Initialize parameters if not provided
        is_batch = False
        if torch_parameters.dim() == 1:
            torch_parameters = torch_parameters.unsqueeze(0)  # Ensure it's a 2D tensor
        else:
            is_batch = True

        batch_size = torch_parameters.size(0)

        # Create parameter mapping
        map_params = {p.name: torch_parameters[:, idx]
                      for idx, p in enumerate(circuit_parameters)}
    elif isinstance(circuit_parameters, dict):
        is_batch = False
        batch_size = 1
        map_params = circuit_parameters
        for name, tensor in map_params.items():
            if tensor.dim() == 1:
                is_batch = True
                batch_size = tensor.shape[0]
                break
    else:
        raise AttributeError("torch_parameters must be a map or a PyTorch tensor")

    # Build unitary matrix by composing component unitaries
    u = None

    for r, c in circuit._components:
        # TODO: we should handle recursively the case where c is a circuit, otherwise sympy unitary will be too complex

        if c.name == "Unitary" and hasattr(c, 'torch_batch_unitaries'):
            cU_torch = c.torch_batch_unitaries
        else:
            # Get component's unitary in symbolic form
            cU = c.compute_unitary(use_symbolic=True)
            # Convert to PyTorch, returns a batch of torch unitaries
            cU_torch = sympy2torch(cU, map_params, batch_size=batch_size)

        # Handle components that don't span all modes
        if len(r) != circuit.m:
            nU = torch.eye(circuit.m, dtype=torch.complex64).unsqueeze(0).repeat(batch_size, 1, 1)
            nU[:, r[0]:(r[-1] + 1), r[0]:(r[-1] + 1)] = cU_torch
            cU_torch = nU

        # Compose unitaries
        if u is None:
            u = cU_torch
        else:
            u = cU_torch @ u

    if not is_batch:
        u = u.squeeze(0)

    return u


if __name__ == '__main__':
    # Create a simple quantum circuit: Mach-Zehnder interferometer
    circuit = pcvl.Circuit(2)
    circuit.add(0, pcvl.BS())  # First beam splitter
    circuit.add(0, pcvl.PS(pcvl.P("phi1")))  # Phase shifter with parameter phi1
    circuit.add(0, pcvl.BS())  # Second beam splitter
    circuit.add(0, pcvl.PS(pcvl.P("phi2")))  # Phase shifter with parameter phi2

    # Convert to PyTorch
    params = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
    _, unitary = pcvl_circuit_to_pytorch_unitary(circuit, params)

    print("Circuit parameters:", params)
    print("\nUnitary matrix:")
    print(unitary)

    # Test differentiability
    try:
        loss = torch.abs(unitary[0, 0]) ** 2
        loss.backward()
        print("\nGradients exist:", params.grad is not None)
        print("Gradients:", params.grad)
    except Exception as e:
        print("\nError testing differentiability:", e)