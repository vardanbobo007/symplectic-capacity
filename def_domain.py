import torch
import math

import configurations
DEVICE = configurations.DEVICE
DTYPE = configurations.FLOAT_TYPE

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

# Domain takes a loop and evaluates the function on it

class RealDomain(torch.nn.Module):

    def __init__(self, coefficients, degrees):
        super().__init__()

        self.dimension = degrees.shape[-1]

        self.register_buffer("coefficients", coefficients)
        self.register_buffer("degrees", degrees)

    def evaluate(self, loop):

        x = torch.abs(loop.unsqueeze(-2))

        powers = x.pow(self.degrees)
        monomials = powers.prod(dim=-1)

        return (monomials * self.coefficients[:-1]).sum(dim=-1) + self.coefficients[-1]


class ComplexDomain(torch.nn.Module):

    def __init__(self, coefficients, degrees):
        super().__init__()

        self.dimension = 2 * degrees.shape[-1]

        self.register_buffer("coefficients", coefficients)
        self.register_buffer("degrees", degrees)

    def evaluate(self, loop):

        z_abs = loop[..., 0::2].square() + loop[..., 1::2].square()
        z_abs = z_abs.unsqueeze(-2)

        powers = z_abs.pow(self.degrees/2)
        monomials = powers.prod(dim=-1)

        return (monomials * self.coefficients[:-1]).sum(dim=-1) + self.coefficients[-1]


# Our domain becomes this, when we appy any matrix to our domain
class TransformedDomain(torch.nn.Module):
    def __init__(self, domain, matrix):
        super().__init__()

        self.domain = domain
        self.dimension = domain.dimension
        self.register_buffer("matrix", matrix)

    def evaluate(self, loop):
        return self.domain.evaluate(loop @ self.matrix)





