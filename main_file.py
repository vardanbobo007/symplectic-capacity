import torch
from pathlib import Path
import matplotlib.pyplot as plt

import configurations
DEVICE = configurations.DEVICE
DTYPE = configurations.FLOAT_TYPE

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

import fourier_loop 
import losses 
import test_domains
import transformations
import def_domain


def extract_valid_loops(fourier, metrics, params):

    action = metrics["action"]
    characteristic = metrics["characteristic"]
    boundary = metrics["boundary"]
    
    valid_mask = (                                   # shape (num_loops)
    (boundary < params.valid_boundary_tolerance) & 
    (characteristic < params.valid_characteristic_tolerance) & 
    (action > params.min_action)
    )                                       

    valid_loops = {
    "a0": fourier.a0[valid_mask].detach().clone(),
    "a_cos": fourier.a_cos[valid_mask].detach().clone(),
    "b_sin": fourier.b_sin[valid_mask].detach().clone(),
    "action": action[valid_mask].detach().clone(),
    "boundary": boundary[valid_mask].detach().clone(),
    "characteristic": characteristic[valid_mask].detach().clone(),
    }

    return valid_loops

def generate_loops_from_valid(valid_loops, params):

    repeats = params.num_new_loop_from_old

    result = {}

    noise_scales = {
        "a0": params.noise_a0,
        "a_cos": params.noise_fourier,
        "b_sin": params.noise_fourier,
    }

    for key in ("a0", "a_cos", "b_sin"):

        values = valid_loops[key].repeat_interleave(repeats, dim=0)
        noise = noise_scales[key] * torch.randn_like(values)
        noise[::repeats] = 0

        result[key] = values + noise

    return result

def select_stage3_loops(valid_loops, params):

    num_valid = valid_loops["action"].shape[0]
    number_of_best = params.number_of_best


    target_num_loops = max(number_of_best, num_valid // 2)

    # Never request more loops than exist
    target_num_loops = min(target_num_loops, num_valid)
    actual_number_of_best = min(number_of_best, target_num_loops)

    sorted_indices = torch.argsort(valid_loops["action"])

    best_indices = sorted_indices[:actual_number_of_best]

    remaining_indices = sorted_indices[actual_number_of_best:]
    num_random_loops = (target_num_loops - actual_number_of_best)

    if num_random_loops > 0:

        permutation = torch.randperm(remaining_indices.shape[0])
        random_indices = remaining_indices[permutation[:num_random_loops]]

        selected_indices = torch.cat([best_indices, random_indices], dim=0)
    else:
        selected_indices = best_indices

    selected_loops = {
        "a0": valid_loops["a0"][selected_indices].detach().clone(),
        "a_cos": valid_loops["a_cos"][selected_indices].detach().clone(),
        "b_sin": valid_loops["b_sin"][selected_indices].detach().clone(),
    }

    return selected_loops

def select_stage4_loops(valid_loops, params):

    number_of_best = params.number_of_best
    num_random = params.num_random
    num_valid = valid_loops["action"].shape[0]

    total_to_choose = number_of_best + num_random
    total = min(total_to_choose, num_valid)

    actual_number_of_best = min(number_of_best, total)
    sorted_indices = torch.argsort(valid_loops["action"])

    best_indices = sorted_indices[:actual_number_of_best]

    remaining_indices = sorted_indices[actual_number_of_best:]

    actual_num_random = min(num_random, total - actual_number_of_best)

    if actual_num_random > 0:
        permutation = torch.randperm(remaining_indices.shape[0])

        random_indices = remaining_indices[permutation[:actual_num_random]]

        selected_indices = torch.cat([best_indices, random_indices], dim=0)

    else:
        selected_indices = best_indices

    selected_loops = {
        "a0": valid_loops["a0"][selected_indices].detach().clone(),
        "a_cos": valid_loops["a_cos"][selected_indices].detach().clone(),
        "b_sin": valid_loops["b_sin"][selected_indices].detach().clone(),
    }

    return selected_loops


def run_adam_stage(fourier, domain, params, stage, print_stat = True):
    
    optimizer = torch.optim.Adam(fourier.parameters(), lr=params.init_lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.num_steps, eta_min=params.min_lr)


    for step in range(params.num_steps):

        optimizer.zero_grad()
        loop = fourier.evaluate()
        loop_dot = fourier.derivative()

        loss, metrics = losses.total_loss(loop, loop_dot, domain, params)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step > 0 and step % params.print_every == 0:

            boundary = metrics["boundary"].detach()
            characteristic = metrics["characteristic"].detach()
            action = metrics["action"].detach()

            valid_mask = (
                (boundary < params.valid_boundary_tolerance)& 
                (characteristic < params.valid_characteristic_tolerance) & (action > params.min_action)
                )

            num_valid = valid_mask.sum().item()

            if print_stat == True:
                print("=" * 25, f"{stage}: step = {step}", "=" * 25)
                print(f"Total loss = {loss.item():.6e}")
                print(f"Mean boundary error = {boundary.mean().item():.1e}")
                print(f"Mean characteristic error = {characteristic.mean().item():.1e}")
                print(f"Number of valid loops = {num_valid} out of {fourier.num_loops}")

            if num_valid > 0:
                masked_action = action.masked_fill(~valid_mask, torch.inf,)
                best_index = masked_action.argmin()

                print(f"Best valid action = {action[best_index].item():.10f}")

    final_loop = fourier.evaluate()
    final_loop_dot = fourier.derivative()

    _, final_metrics = losses.total_loss(final_loop, final_loop_dot, domain, params)

    valid_loops = extract_valid_loops(fourier, final_metrics, params)

    if print_stat == True:
        print("-" * 70)
        print(f"{stage} finished with {valid_loops['action'].shape[0]} valid loops")

    if valid_loops["action"].numel() == 0:
        raise RuntimeError("No valid loops, try another initialization of initial loops or different number of terms in initial Fourier series")

    best_index = valid_loops["action"].argmin()

    if print_stat == True:
        print(f"{stage} best action = {valid_loops['action'][best_index].item():.10f}")
        print(f"Its boundary error = {valid_loops['boundary'][best_index].item():.1e}")
        print(f"Its characteristic error = {valid_loops['characteristic'][best_index].item():.1e}")

    return valid_loops


def run_lbfgs_stage(fourier,domain,params, print_stat = True):

    optimizer = torch.optim.LBFGS(
        fourier.parameters(),
        lr=params.lbfgs_lr,
        max_iter=params.lbfgs_max_iter,
        history_size=params.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )

    def closure():

        optimizer.zero_grad()

        loop = fourier.evaluate()
        loop_dot = fourier.derivative()

        loss, _ = losses.total_loss(loop, loop_dot, domain, params)
        loss.backward()

        return loss

    optimizer.step(closure)

    loop = fourier.evaluate()
    loop_dot = fourier.derivative()

    _, final_metrics = losses.total_loss(loop, loop_dot, domain, params)

    valid_loops = extract_valid_loops(fourier, final_metrics, params)

    best_index = valid_loops["action"].argmin()

    return valid_loops, best_index


def run_triangle_sphere_grid(test_class, checkpoint_path=None):

    params_1 = configurations.Parameters_stage1()
    params_2 = configurations.Parameters_stage2()
    params_3 = configurations.Parameters_stage3()
    params_4 = configurations.Parameters_stage4()

    data = test_class.data
    name = data["name"]
    base_domain = data["domain"]
    p = test_class.p

    num_subdivisions = test_class.num_subdivisions

    print(f"         COMPUTING ALL CAPACITIES OF {name}")

    
    points = transformations.sphere_grid(num_subdivisions)  
#----------------------------- Loading --------------------------------------------------------
# Since computations taking a lot of time, we are loading here previous results if computation was interrupted

    if checkpoint_path is None:
        checkpoint_path = Path("triangle_sphere_grid_results") / f"grid_{num_subdivisions}_p={p}.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if checkpoint["num_subdivisions"] != num_subdivisions:
            raise ValueError("Checkpoint uses a different subdivision count")
        
        results = checkpoint["results"]
        print(f"Loaded {len(results)} of {len(points)} completed points")
    else:
        results = []

    start = len(results)

#------------------------------------------------------------------------------------------------------------------------

    for i in range(start, len(points)):

        q = points[i]
        Q = transformations.orthogonal_matrix_from_sphere_point(q)
        domain = def_domain.TransformedDomain(base_domain, Q)

        print("")
        print("=" * 70)
        print(f"Sphere point {i} of {len(points)}")
        print(f"q = {q.detach().cpu().numpy()}")

        capacity, boundary_error, characteristic_error = compute_capacity(domain, params_1, params_2, params_3, params_4, print_stat = False)

        results.append({
            "index": i,
            "q": q.detach().cpu(),
            "capacity": capacity,
            "boundary": boundary_error,
            "characteristic": characteristic_error,
        })

        checkpoint = {
            "num_subdivisions": num_subdivisions,
            "points": points.detach().cpu(),
            "results": results,
        }

        # This is required if saving didn't work and not to corrupt the existing result
        temporary_path = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)

        print(f"sphere point {i} is computed")
        print(f"Capacity = {capacity:.8f}")
        print("=" * 70)
        print("")



def compute_capacity(domain, params_1, params_2, params_3, params_4, print_stat=True):

    dimension = domain.dimension

    fourier_stage1 = fourier_loop.FourierLoop(params_1, dimension)
    stage1_valid = run_adam_stage(fourier_stage1, domain, params_1, "Stage_1", print_stat)

    stage2_initial_loops = generate_loops_from_valid(stage1_valid, params_2)
    fourier_stage2 = fourier_loop.FourierLoop(params_2, dimension, stage2_initial_loops)
    stage2_valid = run_adam_stage(fourier_stage2, domain, params_2, "Stage_2", print_stat)

    stage3_initial_loops = select_stage3_loops(stage2_valid, params_3)
    fourier_stage3 = fourier_loop.FourierLoop(params_3, dimension, stage3_initial_loops)
    stage3_valid = run_adam_stage(fourier_stage3, domain, params_3, "Stage_3", print_stat)

    stage4_initial_loops = select_stage4_loops(stage3_valid, params_4)
    fourier_stage4 = fourier_loop.FourierLoop(params_4, dimension, stage4_initial_loops)
    valid_loops, best_index = run_lbfgs_stage(fourier_stage4, domain, params_4, print_stat)

    capacity = valid_loops["action"][best_index].item()
    boundary = valid_loops["boundary"][best_index].item()
    characteristic = valid_loops["characteristic"][best_index].item()

    return (capacity, boundary, characteristic)




def run_p_sweep(num_samples=150, p_min=2.0, p_max=15.0, checkpoint_path=None):
    
    p_values = torch.linspace( p_min, p_max, num_samples, dtype=DTYPE, device="cpu")

    q = torch.ones(3, dtype=DTYPE) / (3.0 ** 0.5)
    q_cpu = q.detach().cpu()
    Q = transformations.orthogonal_matrix_from_sphere_point(q)

    coefficients = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0])

    params_1 = configurations.Parameters_stage1()
    params_2 = configurations.Parameters_stage2()
    params_3 = configurations.Parameters_stage3()
    params_4 = configurations.Parameters_stage4()

    if checkpoint_path is None:
        checkpoint_path = ( Path("diagonal_p_sweep_results") / f"p_{p_min:g}_to_{p_max:g}_{num_samples}.pt" )
    else:
        checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu")

        if not torch.equal(saved["p_values"], p_values):
            raise ValueError("Checkpoint uses different p values")
        if not torch.allclose(saved["q"], q_cpu, rtol=0, atol=1e-12):
            raise ValueError("Checkpoint uses a different q")

        results = saved["results"]
    else:
        results = []

    for i in range(len(results), num_samples):

        p = p_values[i].item()
        degrees = p * torch.eye(4, dtype=DTYPE)
        base_domain = def_domain.RealDomain(coefficients, degrees)
        domain = def_domain.TransformedDomain(base_domain, Q)

        capacity, boundary, characteristic = compute_capacity( domain, params_1, params_2, params_3, params_4, print_stat=False)

        results.append({ "index": i, "p": p, "capacity": capacity })

        temporary_path = checkpoint_path.with_suffix(".tmp")
        torch.save( {"p_values": p_values, "q": q_cpu, "results": results}, temporary_path)

        temporary_path.replace(checkpoint_path)

        print(f"{i + 1}/{num_samples}: p={p:.6f}, capacity={capacity:.10f}")


def run_test(test_class):

    params_1 = configurations.Parameters_stage1()
    params_2 = configurations.Parameters_stage2()
    params_3 = configurations.Parameters_stage3()
    params_4 = configurations.Parameters_stage4()

    dimension = test_class.dimension

    if test_class.apply_path_of_unitary_rotations == True:
        t = torch.linspace(0, test_class.unitary_path_length, test_class.number_of_path_iterations)
        u = transformations.get_tangent_unitary(dimension)

    elif test_class.apply_path_of_orthogonal_rotations == True:
        t = torch.linspace(0, test_class.unitary_path_length, test_class.number_of_path_iterations)
    else:
        t = [-1]


    datas = test_class.data
    name_of_all_examples = test_class.name_of_all_examples

    print(f"                  TESTING DOMAIN {name_of_all_examples}")
    print("")

    for data in datas:

        name = data["name"]
        base_domain = data["domain"]
        answer = data.get("answer")
        answer_type = data.get("answer")

        if t == [-1]:
            domain = base_domain

        print(f"TESTING DOMAIN {name}")
        print("")

        for i in range(len(t)):

            if test_class.apply_path_of_unitary_rotations == True:

                Q = transformations.unitary_matrix_from_tangent(u, t[i])
                domain = def_domain.TransformedDomain(base_domain, Q)

                if i != 0:
                    print("Applied UNITARY matrix to the testing domain:")
                    print(Q.cpu().numpy())
                    print("Answer should stay the same.")

            elif test_class.apply_path_of_orthogonal_rotations == True:

                Q = transformations.get_specific_orthogonal(dimension, t[i])
                domain = def_domain.TransformedDomain(base_domain, Q)

                if answer_type == "recompute":
                    coefficients = data["coefficients"]
                    A = torch.diag(coefficients)
                    answer = test_domains.quadratic_capacity_after_orthogonal(A, Q)


                print(f"Computing capacity at t = {t[i].item()}")
                print("Applied ORTHOGONAL matrix to the testing domain:")
                print(Q.cpu().numpy())
                print("")

            capacity, boundary_error, characteristic_error = compute_capacity(domain, params_1, params_2, params_3, params_4)

            print("=" * 70)
            print("=" * 70)
            print("=" * 25, "Final Result", "="*25)

            print(f"HZ capacity estimate = {capacity:.8f}")
            print(f"Its boundary error = {boundary_error:.1e}")
            print(f"Its characteristic error = {characteristic_error:.1e}")

            print("")

            if answer is not None:
                absolute_error = abs(capacity - answer)
                relative_error = absolute_error / answer
                print(f"Absolute error = {absolute_error:.1e}")
                print(f"Relative error = {relative_error:.1e}")
                print("="*70)
                print("")
            else:
                print("Answer is not provided and there is nothing to compare with")
        

#----------------------------------------------------------------------------------------------------

    print(f"END OF TESTING {name_of_all_examples}")
    print("=" * 70)
    print("=" * 70)





#==================================================== TESTS ======================================================

if __name__ == "__main__":

    test_type = configurations.types_tests_to_do()

    if test_type.do_Real_Quadrics_4D == True:
        run_test( test_domains.Real_quadrics_4D())

    if test_type.do_Real_Quadrics_6D == True:
        run_test( test_domains.Real_quadrics_6D())

    if test_type.do_Complex_Quadrics_4D == True:
        run_test( test_domains.Complex_quadrics_4D())

    if test_type.do_Complex_Quadrics_6D == True:
        run_test( test_domains.Complex_quadrics_6D())

    if test_type.do_Real_Lp_domains_4D == True:
        run_test( test_domains.Real_Lp_domains_4D())

    if test_type.do_Real_Lp_domains_6D == True:
        run_test( test_domains.Real_Lp_domains_6D())

    if test_type.do_Complex_Lp_domains_4D == True:
        run_test( test_domains.Complex_Lp_domains_4D())

    if test_type.do_Complex_Lp_domains_6D:
        run_test(test_domains.Complex_Lp_domains_6D())

    if test_type.do_traingle_sphere_computation == True:
        run_triangle_sphere_grid( test_domains.Triangle_sphere_grid())

    if test_type.vary_p_fix_transformation:
        run_p_sweep()

        



 