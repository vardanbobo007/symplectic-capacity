import torch

import configurations
DEVICE = configurations.DEVICE
DTYPE = configurations.FLOAT_TYPE

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

def apply_J(u):

    out = torch.empty_like(u)
    out[..., ::2] = -u[...,1::2]
    out[...,1::2] = u[..., ::2]

    return out

def symplectic_action(loop, loop_dot):

    # shape of loop is (num_loops, num_points, dimension)
    action = (- loop * apply_J(loop_dot)).sum(dim = -1).mean(dim = -1)/2

    return action


def characteristic_loss(loop_dot, ham_grad, eps = 1e-14):

    # shape of loop_dot is (num_loops, num_points, dimension)
    projection = (loop_dot * ham_grad).sum(dim = -1) / ((ham_grad * ham_grad).sum(dim = -1) + eps)  # shape  is (num_loops, num_points)
    projection = torch.relu(projection)
    residual = loop_dot - projection.unsqueeze(-1) * ham_grad  # shape  is (num_loops, num_points, dimension)

    loss = (residual * residual).sum(dim = -1).mean(dim = -1)

    return loss  # shape  is (num_loops, )


def total_loss(loop, loop_dot, domain, params):

    values = domain.evaluate(loop)

    grad = torch.autograd.grad( outputs=values, inputs=loop, grad_outputs=torch.ones_like(values), create_graph=True)[0]
    ham_grad = apply_J(grad)

    boundary_loss = (values * values).mean(dim=-1)
    characteristic = characteristic_loss(loop_dot, ham_grad)
    action = symplectic_action(loop, loop_dot)

    w = params.weights

    metrics = {
        "boundary" : boundary_loss,
        "characteristic" : characteristic,
        "action" : action,
    }

    loss_per_loop = w["boundary"] * boundary_loss + w["characteristic"] * characteristic + w["action"] * action
    total_loss = loss_per_loop.mean()

    return total_loss, metrics