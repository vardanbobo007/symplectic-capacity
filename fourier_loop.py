
import torch

import configurations
DEVICE = configurations.DEVICE
DTYPE = configurations.FLOAT_TYPE

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

class FourierLoop(torch.nn.Module):
    def __init__(self, params, dimension, initial_coefficients = None):
        super().__init__()

        self.dimension = dimension
        self.num_points = params.num_points
        self.num_terms_fourier = params.num_terms_fourier

        if initial_coefficients is None:
            self.num_loops = params.num_loops
        else:
            self.num_loops = initial_coefficients["a0"].shape[0]

        t = torch.arange(self.num_points)     # shape (num_points, 1)
        t = t / self.num_points
        t = t.unsqueeze(-1) 

        k = torch.arange(1, self.num_terms_fourier+1, 1.0).unsqueeze(0)  # shape (1, num_fourier_terms)     
        
        angles =  2*torch.pi * t * k     # shape (num_points, num_fourier_terms)

        self.register_buffer("cos_terms", torch.cos(angles))  # shape (num_points, num_fourier_terms)
        self.register_buffer("sin_terms", torch.sin(angles))  # shape (num_points, num_fourier_terms)

        self.register_buffer("cos_terms_dot", -2*torch.pi * k * torch.sin(angles))   # shape (num_points, num_fourier_terms)
        self.register_buffer("sin_terms_dot", 2*torch.pi * k * torch.cos(angles))      # shape (num_points, num_fourier_terms)


        if initial_coefficients is None:
            a0, a_cos, b_sin = self.init_loops(params)
        else:
            a0, a_cos, b_sin = self.init_loops_from_coefficients(initial_coefficients, params)

        self.a0 = torch.nn.Parameter(a0)
        self.a_cos = torch.nn.Parameter(a_cos)
        self.b_sin = torch.nn.Parameter(b_sin)

    
    def init_loops(self, params):

        B = self.num_loops
        K = self.num_terms_fourier
        n = self.dimension

        a0 = params.center_std * torch.randn([B, 1, n])
        a_cos = torch.zeros([B, K, n])
        b_sin = torch.zeros_like(a_cos)

        # choosing random direction
        u = torch.randn([B, n])
        u = u / u.norm(dim = -1, keepdim = True).clamp_min(1e-12)
        Ju = self.apply_J(u)


        radius_cos = torch.empty([B, 1]).uniform_(params.radius_min, params.radius_max)
        radius_sin = torch.empty([B, 1]).uniform_(params.radius_min, params.radius_max)

        a_cos[:, 0, :] = radius_cos * u
        b_sin[:, 0, :] = radius_sin * Ju

        if K > 1:
            freq = torch.arange(2, K + 1).view(1, -1, 1)
            scales = params.perturbation_std / freq.pow(params.decay)

            a_cos[:, 1:, :] = scales * torch.randn([B, K-1, n])
            b_sin[:, 1:, :] = scales * torch.randn([B, K-1, n])

        return a0, a_cos, b_sin

    def init_loops_from_coefficients(self, initial_coefficients, params):

        old_a0 = initial_coefficients["a0"]
        old_a_cos = initial_coefficients["a_cos"]
        old_b_sin = initial_coefficients["b_sin"]

        old_num_terms = old_a_cos.shape[1]


        a0 = old_a0.clone()

        a_cos = torch.zeros(self.num_loops, self.num_terms_fourier, self.dimension)

        b_sin = torch.zeros_like(a_cos)

        a_cos[:, :old_num_terms, :] = old_a_cos

        b_sin[:, :old_num_terms, :] = old_b_sin

        noise = params.new_terms_noise
        decay = params.decay

        for k in range(old_num_terms, self.num_terms_fourier):

            scale = noise / ((k +1)**decay)

            a_cos[:, k] = scale * torch.randn_like(a_cos[:, k])

            b_sin[:, k] = scale * torch.randn_like(b_sin[:, k])

        return a0, a_cos, b_sin



    def evaluate(self):
    
        cos_terms = torch.einsum("tk, ...kd -> ...td", self.cos_terms, self.a_cos)  # shape (num_loops, num_points, n)
        sin_terms = torch.einsum("tk, ...kd -> ...td", self.sin_terms, self.b_sin) # shape (num_loops, num_points, n)

        return self.a0 + cos_terms + sin_terms      # shape (num_loops, num_points, n)
    
    def derivative(self):
        
        cos_terms_dot = torch.einsum("tk, ...kd -> ...td", self.cos_terms_dot, self.a_cos) # shape (num_loops, num_points, n)
        sin_terms_dot = torch.einsum("tk, ...kd -> ...td", self.sin_terms_dot, self.b_sin)  # shape (num_loops, num_points, n)

        return cos_terms_dot + sin_terms_dot
                

    def apply_J(self, u):

        out = torch.empty_like(u)
        out[..., ::2] = -u[...,1::2]
        out[...,1::2] = u[..., ::2]

        return out

