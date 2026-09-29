import torch

if torch.cuda.is_available() == True:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

FLOAT_TYPE = torch.float64

class types_tests_to_do:
    def __init__(self):

        self.do_Real_Quadrics_4D = False
        self.do_Real_Quadrics_6D = False

        self.do_Complex_Quadrics_4D = False
        self.do_Complex_Quadrics_6D = False

        self.do_Complex_Lp_domains_4D = False
        self.do_Complex_Lp_domains_6D = False

        self.do_Real_Lp_domains_4D = False
        self.do_Real_Lp_domains_6D = False

        self.do_traingle_sphere_computation = False     #  Use when 2n = 4. Choose p and apply sampled elements of SO(4)
        
        self.do_p_sweep = False                        # Choose  SO(4) and test for different p

class Parameters_stage1:

    def __init__(self):

        # Note if your domain does not have smoothed anglles, then use less number of Fourier terms

        self.num_points = 256   # recommended at least 8 * num_terms_fourier
        self.num_terms_fourier = 24   # number of terms in Fourier series
        self.num_loops = 512 * 1      # number of random loops to start with


#========================= LOSS PARAMETERS =====================================================================

        self.min_action = 1e-2    # Ignore loops with action less than ths number

        self.weights = {           # Weights for the loss. See the arxiv paper for more details
            "boundary" : 5e4,
            "characteristic" : 1e5,
            "action" : 1.0
        }

#======================= LOOPS INITIALIZATION PARAMETERS =========================================================
        # Parameters how we randomly create loops. See the arxiv paper for more details

        self.radius_min = 0.5         
        self.radius_max = 1.5
        self.center_std = 0.8
        self.perturbation_std = 0.1
        self.decay = 1.5

#===================== LEARNING PARAMETERS ========================================================================

        self.init_lr = 5e-3
        self.min_lr = 1e-6
        self.num_steps = 8000  

        self.print_every = 5000000     # Choose small nuber if you want to print intermediate steps

#====================== VALIDATION PARAMETERS =====================================================================
        # Ignore loops with boundary and characteristic errors bigger than these numbers. 
        # If your number of Fourier terms is insufficients, then this number may be big and you may end-up with 
        # error "No valid loops"

        self.valid_boundary_tolerance = 1e-4            
        self.valid_characteristic_tolerance = 1e-4



class Parameters_stage2:

    def __init__(self):

        self.num_points = 512   # recommended at least 8 * num_terms_fourier
        self.num_terms_fourier = 26

#========================= LOSS PARAMETERS =====================================================================

        self.min_action = 1e-4

        self.weights = {
            "boundary" : 5e4,
            "characteristic" : 1e5,
            "action" : 1.0
        }

#======================= LOOPS INITIALIZATION PARAMETERS =========================================================

        self.num_new_loop_from_old = 8    # number of local random loops created for each survived loop after stage 1
        self.noise_a0 = 0.08              # noise we add to survived loops
        self.noise_fourier = 0.08

        self.new_terms_noise = 1e-3
        self.decay = 1.2

#===================== LEARNING PARAMETERS ========================================================================
        
        self.init_lr = 1e-6
        self.min_lr = 1e-8
        self.num_steps = 6000

        self.print_every = 4000000

#====================== VALIDATION PARAMETERS =====================================================================

        self.valid_boundary_tolerance = 1e-5
        self.valid_characteristic_tolerance = 1e-5


class Parameters_stage3:

    def __init__(self):

        self.num_points = 512   # recommended at least 8 * num_terms_fourier
        self.num_terms_fourier = 28

        self.number_of_best = 64         # Choose 64 loops with minimal actions and some number of loops are also chosen radomly

#========================= LOSS PARAMETERS =====================================================================

        self.min_action = 1e-4

        self.weights = {
            "boundary" : 5e4,
            "characteristic" : 1e5,
            "action" : 1.0
        }

#======================= LOOPS INITIALIZATION PARAMETERS =========================================================

        self.new_terms_noise = 5e-4
        self.decay = 1.2

#===================== LEARNING PARAMETERS ========================================================================
        
        self.init_lr = 1e-8
        self.min_lr = 1e-9
        self.num_steps = 3000

        self.print_every = 3000000

#====================== VALIDATION PARAMETERS =====================================================================

        self.valid_boundary_tolerance = 1e-5
        self.valid_characteristic_tolerance = 1e-5


class Parameters_stage4:

    def __init__(self):

        self.num_points = 512   # recommended at least 8 * num_terms_fourier
        self.num_terms_fourier = 30

        self.number_of_best = 32
        self.num_random = 48

#========================= LOSS PARAMETERS =====================================================================

        self.min_action = 1e-4

        self.weights = {
            "boundary" : 6e4,
            "characteristic" : 1e5,
            "action" : 0.5
        }

#======================= LOOPS INITIALIZATION PARAMETERS =========================================================

        self.new_terms_noise = 1e-4
        self.decay = 1.0

#===================== LEARNING PARAMETERS ========================================================================
        
        self.lbfgs_lr = 1.0
        self.lbfgs_max_iter = 3000
        self.lbfgs_history_size = 100

#====================== VALIDATION PARAMETERS =====================================================================

        self.valid_boundary_tolerance = 1e-5
        self.valid_characteristic_tolerance = 1e-5



