import torch
import math

import def_domain
import transformations

import configurations
DEVICE = configurations.DEVICE
DTYPE = configurations.FLOAT_TYPE

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

#========================= QUADRICS IN 4D ==========================================================================

class Real_quadrics_4D:
    def __init__(self):

#---------------------- CHANGE ONLY THIS PART TO INTRODUCE NEW EXAMPLES HERE ----------------------------------------
        # Each row is an example of a qudric. Elements of each row are coefficients
        c = [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.5, 1.6, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1, 0.1, 0.01, 0.001]
            ]
#-------------------------------------------------------------------------------------------

        self.dimension = 4
        self.apply_path_of_unitary_rotations = True
        self.apply_path_of_orthogonal_rotations = False

        self.number_of_path_iterations = 2
        self.unitary_path_length = torch.pi

        if self.apply_path_of_orthogonal_rotations == True  and self.apply_path_of_unitary_rotations == True:
            raise ValueError("Choose orthogonal or unitary rotations, not both")

        self.name_of_all_examples = "Real Quadrics in R^4"

#-------------------------------------------------------------------------------------------

        self.data = []
        
        for i in range(len(c)):

            name = f" {c[i][0]}x1^2 + {c[i][1]}y1^2 + {c[i][2]}x2^2 + {c[i][3]}y2^2 - 1 <= 0" 
            degrees = torch.tensor([ [2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2] ])
            coefficients = torch.tensor([ c[i][0], c[i][1], c[i][2], c[i][3], -1.0])
            answer = torch.pi/max( math.sqrt( c[i][0] * c[i][1] ), math.sqrt( c[i][2] * c[i][3] ) )
            
            domain = def_domain.RealDomain(coefficients, degrees)

            if self.apply_path_of_orthogonal_rotations == True:            
                self.data.append(  { "name": name, "domain" : domain, "answer": "recompute", "coefficients" : coefficients[:-1] }  )
            else:
                self.data.append(  { "name": name, "domain" : domain, "answer": answer}  )

    

class Real_quadrics_6D:
    def __init__(self):

#---------------------- CHANGE ONLY THIS PART TO INTRODUCE NEW EXAMPLES HERE ----------------------------------------
            # Numbers in each row are coefficients
        c = [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            #[1.5, 2.0, 0.25, 1.0, 2.0, 3.5],
            #[1.0, 4.5, 1.0, 9.0, 3.0, 5.0],
            ]
#---------------------------------------------------------------------------------------------------------------------


        self.dimension = 6
        self.apply_path_of_unitary_rotations = True
        self.apply_path_of_orthogonal_rotations = False

        self.number_of_path_iterations = 2
        self.unitary_path_length = torch.pi

        if self.apply_path_of_orthogonal_rotations == True  and self.apply_path_of_unitary_rotations == True:
            raise ValueError("Choose orthogonal or unitary rotations, not both")

        self.name_of_all_examples = "Real Quadrics in R^6"
#---------------------------------------------------------------------------------------------

        self.data = []

        for i in range(len(c)):

            name = f" {c[i][0]}x1^2 + {c[i][1]}y1^2 + {c[i][2]}x2^2 + {c[i][3]}y2^2 + {c[i][4]}x3^2 + {c[i][5]}y3^2 - 1 <= 0" 
            degrees = torch.tensor([ 
                [2, 0, 0, 0, 0, 0], 
                [0, 2, 0, 0, 0, 0], 
                [0, 0, 2, 0, 0, 0], 
                [0, 0, 0, 2, 0, 0], 
                [0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 2]])
            coefficients = torch.tensor([ c[i][0], c[i][1], c[i][2], c[i][3], c[i][4], c[i][5], -1.0])
            answer = torch.pi/max(math.sqrt(c[i][0] * c[i][1]), math.sqrt( c[i][2] * c[i][3] ), math.sqrt( c[i][4] * c[i][5] ) )
        
            domain = def_domain.RealDomain(coefficients, degrees)
                        
            if self.apply_path_of_orthogonal_rotations == True:            
                self.data.append(  { "name": name, "domain" : domain, "answer": "recompute", "coefficients" : coefficients[:-1] }  )
            else:
                self.data.append(  { "name": name, "domain" : domain, "answer": answer}  )

class Complex_quadrics_4D:
    def __init__(self):

#---------------------- CHANGE ONLY THIS PART TO INTRODUCE NEW EXAMPLES HERE ----------------------------------------
       # Numbers in each row are coefficients  |z+1|^2 + |z_2|^2
        c = [
            [1, 1],
            [5, 2],
            [2, 0.01]
            ]
#------------------------------------------------------------------------------------------------------------------

        self.dimension = 4
        self.apply_path_of_unitary_rotations = True
        self.apply_path_of_orthogonal_rotations = False

        self.number_of_path_iterations = 2
        self.unitary_path_length = torch.pi

        if self.apply_path_of_orthogonal_rotations == True  and self.apply_path_of_unitary_rotations == True:
            raise ValueError("Choose orthogonal or unitary rotations, not both")

        self.name_of_all_examples = "Complex Quadrics in R^4"
#----------------------------------------------------------------------------------------------------------

        self.data = []

        for i in range(len(c)):

            name = f" {c[i][0]}|z1|^2 + {c[i][1]}|z2|^2  - 1 <= 0" 
            degrees = torch.tensor([ [2, 0], [0, 2]])
            coefficients = torch.tensor([ c[i][0], c[i][1], -1.0])
            answer = torch.pi/max(c[i][0], c[i][1])

            domain = def_domain.ComplexDomain(coefficients, degrees)
            
            if self.apply_path_of_orthogonal_rotations == True:            
                self.data.append(  { "name": name, "domain" : domain, "answer": "recompute", "coefficients" : coefficients[:-1].repeat_interleave(2, dim=0) }  )
            else:
                self.data.append(  { "name": name, "domain" : domain, "answer": answer}  )


class Complex_quadrics_6D:
    def __init__(self):

#---------------------- CHANGE ONLY THIS PART TO INTRODUCE NEW EXAMPLES HERE ----------------------------------------
        # Numbers in each row are coefficients
        c = [
            [1.0, 2.5, 3.0],
            [5.6, 2.2, 1.2],
            [2.0, 0.01, 0.1]
            ]
#------------------------------------------------------------------------------------------------------------------


        self.dimension = 6
        self.apply_path_of_unitary_rotations = True
        self.apply_path_of_orthogonal_rotations = False

        self.number_of_path_iterations = 2
        self.unitary_path_length = torch.pi

        if self.apply_path_of_orthogonal_rotations == True  and self.apply_path_of_unitary_rotations == True:
            raise ValueError("Choose orthogonal or unitary rotations, not both")

        self.name_of_all_examples = "Complex Quadrics in R^6"
#---------------------------------------------------------------------------------------------

        self.data = []

        for i in range(len(c)):

            name = f" {c[i][0]}|z1|^2 + {c[i][1]}|z2|^2 + {c[i][2]}|z3|^2 - 1 <= 0"
            degrees = torch.tensor([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
            coefficients = torch.tensor([c[i][0], c[i][1], c[i][2], -1.0])
            answer = torch.pi/max(c[i][0], c[i][1], c[i][2])

            domain = def_domain.ComplexDomain(coefficients, degrees)

            if self.apply_path_of_orthogonal_rotations == True:            
                self.data.append(  { "name": name, "domain" : domain, "answer": "recompute", "coefficients" : coefficients[:-1].repeat_interleave(2, dim=0) }  )
            else:
                self.data.append(  { "name": name, "domain" : domain, "answer": answer}  )
        
        

#  USE ANSWERS HERE ONLY WHEN ALL DEGREES ARE GREATER THAN OR EQUAL TO 2
class Complex_Lp_domains_4D:
    def __init__(self):

#---------------------- CHANGE ONLY THIS PART TO INTRODUCE NEW EXAMPLES HERE ----------------------------------------
# Each row is an example. The first list are degrees, second are coefficients

        c = [
            ([2.0, 3.0], [1.0, 1.0]),
            ([4.5, 5.5], [4.0, 6.0]),
            ([12.0, 16.0], [2.0, 8.0]),
            ([16.2, 10.2], [20.5, 2.5])
            ]

#------------------------------------------------------------------------------------------------------------------

        self.dimension = 4
        self.apply_path_of_unitary_rotations = True
        self.apply_path_of_orthogonal_rotations = False

        self.number_of_path_iterations = 2
        self.unitary_path_length = torch.pi

        if self.apply_path_of_orthogonal_rotations == True  and self.apply_path_of_unitary_rotations == True:
            raise ValueError("Choose orthogonal or unitary rotations, not both")

        self.name_of_all_examples = "Complex Lp domains in R^4"

#-------------------------------------------------------------------------------------------

        self.data = []

        for i in range(len(c)):

            name = f" {c[i][1][0]}|z1|^{c[i][0][0]} + {c[i][1][1]}|z2|^{c[i][0][1]}  - 1 <= 0" 
            degrees = torch.tensor([ [c[i][0][0], 0], [0, c[i][0][1]]  ])
            coefficients = torch.tensor([ c[i][1][0], c[i][1][1], -1.0])
            answer = torch.pi/max(math.pow(c[i][1][0], 2/c[i][0][0]), math.pow(c[i][1][1], 2/c[i][0][1]))

            domain = def_domain.ComplexDomain(coefficients, degrees)

            if self.apply_path_of_orthogonal_rotations == True:            
                self.data.append(  { "name": name, "domain" : domain, "answer": None}  )
            else:
                self.data.append(  { "name": name, "domain" : domain, "answer": answer}  )

#  USE ANSWERS HERE ONLY WHEN ALL DEGREES ARE GREATER THAN OR EQUAL TO 2
class Complex_Lp_domains_6D:
    def __init__(self):

#---------------------- CHANGE ONLY THIS PART TO INTRODUCE NEW EXAMPLES HERE ----------------------------------------
# Each row is an example. The first list are degrees, second are coefficients
        c = [
            ([2.0, 3.0, 4.0], [1.0, 1.0, 1.0]),
            ([4.5, 5.5, 6.5], [4.0, 6.0, 8.0]),
            ([12.2, 16.2, 10.0], [2.5, 8.2, 6.0]),
            ([16.0, 10.5, 12.0], [20.0, 2.5, 4.0])
            ]
#------------------------------------------------------------------------------------------------------------------


        self.dimension = 6
        self.apply_path_of_unitary_rotations = True
        self.apply_path_of_orthogonal_rotations = False

        self.number_of_path_iterations = 2
        self.unitary_path_length = torch.pi

        if self.apply_path_of_orthogonal_rotations == True  and self.apply_path_of_unitary_rotations == True:
            raise ValueError("Choose orthogonal or unitary rotations, not both")

        self.name_of_all_examples = "Complex Lp domains in R^6"

#-------------------------------------------------------------------------------------------

        self.data = []

        for i in range(len(c)):

            name = f" {c[i][1][0]}|z1|^{c[i][0][0]} + {c[i][1][1]}|z2|^{c[i][0][1]} + {c[i][1][2]}|z3|^{c[i][0][2]}  - 1 <= 0" 
            degrees = torch.tensor([ [c[i][0][0], 0, 0], [0, c[i][0][1], 0], [0, 0, c[i][0][2]]])
            coefficients = torch.tensor([ c[i][1][0], c[i][1][1], c[i][1][2], -1.0])
            answer = torch.pi/max(math.pow(c[i][1][0], 2/c[i][0][0]), math.pow(c[i][1][1], 2/c[i][0][1]), math.pow(c[i][1][2], 2/c[i][0][2]))

            domain = def_domain.ComplexDomain(coefficients, degrees)
        
            if self.apply_path_of_orthogonal_rotations == True:            
                self.data.append(  { "name": name, "domain" : domain, "answer": None}  )
            else:
                self.data.append(  { "name": name, "domain" : domain, "answer": answer}  )


# USE ANSWERS HERE ONLY WHEN ALL DEGREES ARE GREATER THAN OR EQUAL TO 2
class Real_Lp_domains_4D:
    def __init__(self):

#---------------------- CHANGE ONLY THIS PART TO INTRODUCE NEW EXAMPLES HERE ----------------------------------------
# Each row is an example. The first list are degrees, second are coefficients

        c = [
            ([1.5, 1.5, 1.5, 1.5], [1.0, 1.0, 1.0, 1.0]),
            #([4.5, 5.5, 4.0, 6.2], [4.0, 6.0, 2.0, 2.0]),
            #([16.2, 10.2, 2.5, 6.5], [20.5, 2.5, 1.0, 10.0]),
        ]
#------------------------------------------------------------------------------------------------------------------


        self.dimension = 4
        self.apply_path_of_unitary_rotations = False
        self.apply_path_of_orthogonal_rotations = True

        self.number_of_path_iterations = 2
        self.unitary_path_length = torch.pi/4

        if self.apply_path_of_orthogonal_rotations == True  and self.apply_path_of_unitary_rotations == True:
            raise ValueError("Choose orthogonal or unitary rotations, not both")

        self.name_of_all_examples = "Real Lp domain in R^4"
#---------------------------------------------------------------------------------------------

        self.data = []

        for degrees_data, coefficients_data in c:

            p1, q1, p2, q2 = degrees_data
            A, B, C, D = coefficients_data

            name = (f"{A}|x1|^{p1} + {B}|y1|^{q1} + {C}|x2|^{p2} + {D}|y2|^{q2} - 1 <= 0")

            degrees = torch.tensor(
                [
                    [p1, 0.0, 0.0, 0.0],
                    [0.0, q1, 0.0, 0.0],
                    [0.0, 0.0, p2, 0.0],
                    [0.0, 0.0, 0.0, q2]  ])

            coefficients = torch.tensor([A, B, C, D, -1.0])

            log_S1 = math.log(4.0) - math.log(A)/p1 - math.log(B)/q1 + math.lgamma(1.0 + 1.0/p1) + math.lgamma(1.0 + 1.0/q1) - math.lgamma(1.0 + 1.0/p1 + 1.0/q1)
            log_S2 = math.log(4.0) - math.log(C)/p2 - math.log(D)/q2 + math.lgamma(1.0 + 1.0/p2) + math.lgamma(1.0 + 1.0/q2) - math.lgamma(1.0 + 1.0/p2 + 1.0/q2)
            
            S1 = math.exp(log_S1)
            S2 = math.exp(log_S2)

            answer = min(S1, S2)

            domain = def_domain.RealDomain(coefficients, degrees)

            if self.apply_path_of_orthogonal_rotations == True:            
                self.data.append(  { "name": name, "domain" : domain, "answer": None}  )
            else:
                self.data.append(  { "name": name, "domain" : domain, "answer": answer}  )


# USE ANSWERS HERE ONLY WHEN ALL DEGREES ARE GREATER THAN OR EQUAL TO 2
class Real_Lp_domains_6D:
    def __init__(self):

#---------------------- CHANGE ONLY THIS PART TO INTRODUCE NEW EXAMPLES HERE ----------------------------------------
# Each row is an example. The first list are degrees, second are coefficients
        c = [
            ([2.0, 3.0, 4.9, 2.0, 6.0, 8.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            ([4.5, 5.5, 4.0, 6.2, 3.5, 7.5], [4.0, 6.0, 2.0, 2.0, 3.0, 5.0]),
            ([16.2, 10.2, 2.5, 6.5, 3.2, 9.5], [20.5, 2.5, 1.0, 10.0, 7.5, 3.0]),
        ]
#------------------------------------------------------------------------------------------------------------------

        self.dimension = 6
        self.apply_path_of_unitary_rotations = True
        self.apply_path_of_orthogonal_rotations = False

        self.number_of_path_iterations = 2
        self.unitary_path_length = torch.pi

        if self.apply_path_of_orthogonal_rotations == True  and self.apply_path_of_unitary_rotations == True:
            raise ValueError("Choose orthogonal or unitary rotations, not both")

        self.name_of_all_examples = "Real Lp domain in R^6"
#---------------------------------------------------------------------------------------------
        self.data = []

        for degrees_data, coefficients_data in c:

            p1, q1, p2, q2, p3, q3 = degrees_data
            A, B, C, D, E, F = coefficients_data

            name = (f"{A}|x1|^{p1} + {B}|y1|^{q1} + {C}|x2|^{p2} + {D}|y2|^{q2} + {E}|x3|^{p3} + {F}|y3|^{q3} - 1 <= 0")

            degrees = torch.tensor(
                [
                    [p1, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, q1, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, p2, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, q2, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, p3, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, q3]  ])

            coefficients = torch.tensor([A, B, C, D, E, F, -1.0])

            log_S1 = math.log(4.0) - math.log(A)/p1 - math.log(B)/q1 + math.lgamma(1.0 + 1.0/p1) + math.lgamma(1.0 + 1.0/q1) - math.lgamma(1.0 + 1.0/p1 + 1.0/q1)
            log_S2 = math.log(4.0) - math.log(C)/p2 - math.log(D)/q2 + math.lgamma(1.0 + 1.0/p2) + math.lgamma(1.0 + 1.0/q2) - math.lgamma(1.0 + 1.0/p2 + 1.0/q2)
            log_S3 = math.log(4.0) - math.log(E)/p3 - math.log(F)/q3 + math.lgamma(1.0 + 1.0/p3) + math.lgamma(1.0 + 1.0/q3) - math.lgamma(1.0 + 1.0/p3 + 1.0/q3)

            S1 = math.exp(log_S1)
            S2 = math.exp(log_S2)
            S3 = math.exp(log_S3)

            answer = min(S1, S2, S3)

            domain = def_domain.RealDomain(coefficients, degrees)

            if self.apply_path_of_orthogonal_rotations == True:            
                self.data.append(  { "name": name, "domain" : domain, "answer": None}  )
            else:
                self.data.append(  { "name": name, "domain" : domain, "answer": answer}  )




# !!! RUN THIS ONLY ON DOMAIN LIKE |x1|^p + |y1|^p + |x2|^p + |y2|^p - 1 <= 0
class Triangle_sphere_grid:
    def __init__(self):

#---------------------- CHANGE ONLY THIS PART TO INTRODUCE NEW EXAMPLES HERE ----------------------------------------
        # Fix p and apply sampled elements of SO(4)
        self.p = 2.5
        A = 1.0 
    
        self.num_subdivisions = 14   # number of sampled points will be (num_subdivisions + 1)*(num_subdivisions + 2)/2

        self.name_of_all_examples = "Triangle Sphere computation of all capacities"
#---------------------------------------------------------------------------------------------

        self.dimension = 4
        name = (f"{A}|x1|^{self.p} + {A}|y1|^{self.p} + {A}|x2|^{self.p} + {A}|y2|^{self.p} - 1 <= 0")

        degrees = torch.tensor(
            [
                [self.p, 0.0, 0.0, 0.0],
                [0.0, self.p, 0.0, 0.0],
                [0.0, 0.0, self.p, 0.0],
                [0.0, 0.0, 0.0, self.p]  ])

        coefficients = torch.tensor([A, A, A, A, -1.0])
        domain = def_domain.RealDomain(coefficients, degrees)

        self.data =  { "name": name, "domain" : domain}  


            


def quadratic_capacity_after_orthogonal(A, matrix):

    transformed_A = matrix @ A @ matrix.T

    J = torch.zeros_like(A)
    identity = torch.eye(A.shape[0]//2)

    J[0::2, 1::2] = -identity
    J[1::2, 0::2] = identity

    eigenvalues = torch.linalg.eigvals(J @ transformed_A)
    largest_eigenvalue = eigenvalues.abs().max()

    return (torch.pi / largest_eigenvalue).item()



    