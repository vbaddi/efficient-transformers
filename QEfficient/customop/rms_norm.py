# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import onnxscript
import torch
from torch import nn

from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_EXPORT_OPSET))


@onnxscript.script(onnxscript.values.Opset(domain="com.qti.aisw.onnx", version=1))
def CustomRMSNorm(hidden_states: onnxscript.FLOAT, weight: onnxscript.FLOAT, epsilon: float):
    weight = ops.Cast(weight, to=1)
    variance = ops.ReduceMean(ops.Pow(hidden_states, 2), axes=[-1], keepdims=1)
    epsilon = ops.Expand(epsilon, ops.Shape(variance))
    hidden_states = hidden_states * ops.Reciprocal(ops.Sqrt(variance + epsilon))
    return weight * hidden_states


class CustomRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        hidden_states, weight, epsilon = inputs
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + epsilon)
        normed_hidden_states = hidden_states * inv_rms
        ctx.save_for_backward(hidden_states, weight, inv_rms, normed_hidden_states)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, weight, inv_rms, normed_hidden_states = ctx.saved_tensors
        grad_hidden_states = grad_output * weight
        dot = (grad_hidden_states * hidden_states).sum(dim=-1, keepdim=True)
        hidden_dim = hidden_states.shape[-1]
        grad_hidden_states = grad_hidden_states * inv_rms - hidden_states * (inv_rms**3) * dot / hidden_dim

        reduce_dims = tuple(range(grad_output.ndim - 1))
        grad_weight = (grad_output * normed_hidden_states).sum(dim=reduce_dims)
        return grad_hidden_states, grad_weight, None

    @staticmethod
    def symbolic(g: torch.Graph, hidden_states: torch.Value, weight: torch.Value, epsilon: torch.Value) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight, epsilon_f=epsilon).setTypeAs(hidden_states)


class CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def __init__(self, hidden_size, eps=1e-05):
        super(CustomRMSNormAIC, self).__init__()
        self.variance_epsilon = eps
        self.eps = eps  # Added to support GemmaRMSNorm
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return CustomRMSNormFunc.apply(
            hidden_states, self.weight, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        )


class GemmaCustomRMSNormAIC(CustomRMSNormAIC):
    """
    Modify the init function to add +1 to the weights
    """

    def __qeff_init__(self):
        with torch.no_grad():
            self.weight.copy_(self.weight + 1.0)
