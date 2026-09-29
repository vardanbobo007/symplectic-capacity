import torch

import configurations
DEVICE = configurations.DEVICE
DTYPE = configurations.FLOAT_TYPE

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)


def complex_to_real_matrix(Q):

    n = Q.shape[0]
    real = Q.real
    imag = Q.imag

    matrix = torch.zeros(2*n, 2*n)

    matrix[0::2, 0::2] = real
    matrix[0::2, 1::2] = -imag
    matrix[1::2, 0::2] = imag
    matrix[1::2, 1::2] = real

    return matrix


def get_tangent_unitary(dimension):

    A = torch.randn(dimension // 2, dimension // 2, dtype=torch.complex128)
    A = A - A.mH

    return A/A.norm().clamp_min(1e-5)


def unitary_matrix_from_tangent(A, t):

    Q = torch.matrix_exp(t * A)

    return complex_to_real_matrix(Q)


def quaternion_matrix(q):

    w, x, y, z = q.unbind(dim=-1)

    return torch.stack([
        torch.stack([w, -x, -y, -z], dim=-1),
        torch.stack([x,  w, -z,  y], dim=-1),
        torch.stack([y,  z,  w, -x], dim=-1),
        torch.stack([z, -y,  x,  w], dim=-1),
    ], dim=-2)


def orthogonal_matrix_from_sphere_point(q):

    q = q / q.norm(dim=-1, keepdim=True)
    a, b, c = q.unbind(dim=-1)

    d = torch.sqrt(2 * (1 + a))
    quaternion = torch.stack([(1 + a)/d, torch.zeros_like(a), c/d, -b/d], dim=-1)

    return quaternion_matrix(quaternion)

def sphere_grid(num_subdivisions):

    vertices = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    vertices = vertices / vertices.norm(dim=-1, keepdim=True)

    indices = []
    for i in range(num_subdivisions + 1):
        if i % 2 == 0:
            row = range(num_subdivisions - i + 1)
        else:
            row = range(num_subdivisions - i, -1, -1)

        indices.extend((i, j) for j in row)

    indices = torch.tensor(indices)
    i, j = indices.unbind(dim=-1)

    weights = torch.stack([num_subdivisions - i - j, i, j], dim=-1).to(DTYPE)
    points = weights @ vertices

    return points / points.norm(dim=-1, keepdim=True)


def get_specific_orthogonal(dimension, t):

    Q = torch.eye(dimension, device=t.device, dtype=t.dtype)

    c = torch.cos(t)
    s = torch.sin(t)

    # Rotate (y_i, x_{i+1}) 
    for k in range(dimension // 2 - 1):
        i = 2*k + 1
        j = 2*k + 2

        Q[i, i] = c
        Q[i, j] = -s
        Q[j, i] = s
        Q[j, j] = c

    return Q

