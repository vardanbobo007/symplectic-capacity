# EHZ Capacity Approximation

This code numerically approximates the  EHZ capacity of convex domains.

The method uses Fourier series and gradient-based approximation.

There is a paper in arxiv, describing the method and rigorous formulas obtained using these numerical estimations.

## How to use the code

In most cases, only two files need to be changed:

- `test_domains.py`
- `configurations.py`

The remaining files contain the numerical method and  do not need to be modified.

### 1. Define the domain

Open `test_domains.py`.

This file contains the convex domains on which the EHZ capacity is computed. Existing examples include:

- real quadratic domains,
- complex quadratic domains,
- real \(L_p\) domains,
- complex \(L_p\) domains.

A new domain can be introduced by specifying its defining function and the corresponding coefficients and exponents.

### 2. Choose the numerical parameters

Open `configurations.py`.

This file contains the parameters used in the numerical approximation, including:

- number of Fourier modes,
- number of discretization points,
- number of initial curves,
- learning rates,
- weights of the loss terms,

At the beginning of the file, choose which computation should be performed by setting the corresponding option to `True`.

For example:

```python
self.do_Real_Lp_domains_4D = True
```

Note if your domain has some smoothed angles, then you need more Fourier terms to approximate. If you end-up with an error "No valid loops", then you don't have enough Fourier terms to approximate your domain. Increase the number for each stage.

### 3. Run the computation

Run

```bash
python main_file.py
```

The program performs four successive optimization stages and prints:

- the numerical approximation of the EHZ capacity,
- the boundary loss,
- the characteristic loss.

If an exact value is known, the code also prints the absolute and relative errors.

## Files

- `test_domains.py` — definitions of the domains and test examples.
- `configurations.py` — numerical parameters and choice of computation.
- `main_file.py` — runs the numerical optimization.
- `fourier_loop.py` — Fourier representation of closed curves.
- `losses.py` — action, boundary, and characteristic loss functions.
- `def_domain.py` — classes describing the domains.
- `transformations.py` — unitary and orthogonal transformations.

For standard use, **only `test_domains.py` and `configurations.py` need to be changed**.
