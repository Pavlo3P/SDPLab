# Gaussian QOT Barycenter Mathematics

This module implements a moment-level Gaussian model for an entropic quantum
optimal transport barycenter problem. It does not build dense Fock-space
matrices. States, costs, and dual potentials are represented by Gaussian
moments and quadratic phase-space operators.

## Phase-Space Convention

For an $m$-mode system, canonical variables are ordered as

$$
R = (Q_1, P_1, \ldots, Q_m, P_m) \in \mathbb{R}^{2m}.
$$

The canonical commutation relations are encoded by the symplectic matrix and
the nonnegative Planck constant parameter `hbar` stored on `GaussianPhaseSpace`.
The value `hbar=0` selects the classical Gaussian endpoint:

$$
[R_i, R_j] = i\,\hbar\,\Omega_{ij}.
$$

A Gaussian state is represented by

$$
\text{mean: } d \in \mathbb{R}^{2m}, \qquad
\text{covariance: } \Gamma \in \mathbb{R}^{2m \times 2m}, \qquad
\text{trace: } z > 0.
$$

The covariance must satisfy the uncertainty principle

$$
\Gamma + \tfrac{i\hbar}{2}\,\Omega \;\succeq\; 0.
$$

The stored covariance and mean describe the normalized Gaussian shape. The
scalar `normalization` stores the operator trace. For a density operator,
$\text{normalization} = 1$.

The second moment is

$$
M \;=\; \Gamma + d\,d^{\mathsf{T}}.
$$

## Quadratic Operators

A quadratic operator on phase space is represented as

$$
A \;=\; c + a^{\mathsf{T}} R + \tfrac{1}{2}\,R^{\mathsf{T}} G\,R,
$$

where

$$
c \in \mathbb{R}, \qquad
a \in \mathbb{R}^{2m}, \qquad
G \in \mathbb{R}^{2m \times 2m}, \quad G = G^{\mathsf{T}}.
$$

For a Gaussian state $\rho = (z, d, \Gamma)$,

$$
\operatorname{Tr}[\rho\,A]
\;=\; z\,\Big(\,c + a^{\mathsf{T}} d
       + \tfrac{1}{2}\,\operatorname{Tr}\!\big[G\,(\Gamma + d\,d^{\mathsf{T}})\big]\Big).
$$

This is the key reason the implementation can stay at the moment level.

## Entropy Convention

The module uses the scalar regularization

$$
\varphi(x) \;=\; x\,(\log x - 1).
$$

Its convex conjugate is

$$
\varphi^{*}(t) \;=\; \exp(t).
$$

Consequently, the dual Gibbs states use $\exp(t)$, not $\exp(t - 1)$.

For a Gaussian trace-class operator $X$, the entropy contribution is

$$
\operatorname{Tr}\!\big[X\,(\log X - 1)\big].
$$

The implementation also supports a trace-normalized log-partition variant by
constructing

```python
QOTGaussianBarycenterProblem(..., use_log_partition=True)
```

In that mode, the dual uses $\log \operatorname{Tr}\exp(\cdot)$ instead of
$\operatorname{Tr}\exp(\cdot)$. This is usually more numerically stable
because the code does not exponentiate large partition logs in the dual
objective. It corresponds to the conjugate of entropy over trace-normalized
states:

$$
\sup_{\,X \succeq 0,\,\operatorname{Tr} X = 1}
\Big\{ \operatorname{Tr}[A\,X] - \varepsilon\,\operatorname{Tr}[X \log X]\Big\}
\;=\; \varepsilon\,\log \operatorname{Tr}\exp\!\big(A/\varepsilon\big).
$$

The gradients of $\log \operatorname{Tr}\exp$ are normalized Gibbs states.
Therefore, when `use_log_partition=True`, `dual_state_couplings(...)` and
`dual_barycenter_state(...)` return Gaussian states with normalization one.

## Barycenter Problem

Let

$$
\sigma_s, \qquad s = 1, \ldots, N,
$$

be input Gaussian states on phase space $\mathcal{H}$, with weights

$$
\alpha_s \geq 0, \qquad \sum_{s=1}^{N} \alpha_s = 1.
$$

The barycenter lives on a phase space $\mathcal{H}_0$. In many examples
$\mathcal{H}_0 = \mathcal{H}$, but the implementation keeps them separate.

For each input state, introduce a coupling

$$
\pi_s \;\;\text{on}\;\; \mathcal{H}_0 \otimes \mathcal{H}.
$$

The coupling constraints are

$$
\operatorname{Tr}_{\mathcal{H}}[\pi_s] \;=\; \rho \quad \text{for all } s,
\qquad
\operatorname{Tr}_{\mathcal{H}_0}[\pi_s] \;=\; \sigma_s \quad \text{for all } s.
$$

Here $\rho$ is the unknown barycenter. The first constraint says every
coupling has the same left marginal. That common marginal is the barycenter.

With quadratic cost $C$ on the joint phase space, the regularized primal
objective is

$$
\min_{\rho,\,\{\pi_s\}}\;
\sum_{s} \alpha_s \operatorname{Tr}[\pi_s\,C]
\;+\; \varepsilon \sum_{s} \operatorname{Tr}\!\big[\pi_s\,(\log \pi_s - 1)\big]
\;+\; \tau \operatorname{Tr}\!\big[\rho\,(\log \rho - 1)\big]
$$

subject to the marginal constraints above.

The parameter $\varepsilon > 0$ regularizes the couplings. The parameter
$\tau > 0$ regularizes the barycenter itself.

## Dual Variables

For every input state, the dual has two quadratic potentials:

$$
U_s \text{ on } \mathcal{H}_0, \qquad V_s \text{ on } \mathcal{H}.
$$

They are represented as quadratic operators:

$$
U_s(R_0) = c_s + u_s^{\mathsf{T}} R_0 + \tfrac{1}{2}\,R_0^{\mathsf{T}} G_s\,R_0,
$$

$$
V_s(R)\;\, = d_s + v_s^{\mathsf{T}} R   + \tfrac{1}{2}\,R^{\mathsf{T}}   H_s\,R.
$$

The dual objective is

$$
\begin{aligned}
\mathcal{D}(U, V)
\;=\;& \sum_{s} \operatorname{Tr}[V_s\,\sigma_s] \\
&\;-\; \varepsilon \sum_{s} \operatorname{Tr}\exp\!\big(
        (U_s \otimes I + I \otimes V_s - \alpha_s C)/\varepsilon\big) \\
&\;-\; \tau \operatorname{Tr}\exp\!\big(-\textstyle\sum_{s} U_s/\tau\big).
\end{aligned}
$$

The two exponential terms define Gaussian trace-class operators when their
quadratic exponents are confining.

Define

$$
\rho_s \;=\; \exp\!\big((U_s \otimes I + I \otimes V_s - \alpha_s C)/\varepsilon\big),
\qquad
\eta \;=\; \exp\!\big(-\textstyle\sum_{s} U_s/\tau\big).
$$

The state $\eta$ is the dual barycenter candidate.

With `use_log_partition=True`, the dual objective is instead

$$
\begin{aligned}
\mathcal{D}_{\log}(U, V)
\;=\;& \sum_{s} \operatorname{Tr}[V_s\,\sigma_s] \\
&\;-\; \varepsilon \sum_{s} \log \operatorname{Tr}\exp\!\big(
        (U_s \otimes I + I \otimes V_s - \alpha_s C)/\varepsilon\big) \\
&\;-\; \tau \log \operatorname{Tr}\exp\!\big(-\textstyle\sum_{s} U_s/\tau\big).
\end{aligned}
$$

The associated $\rho_s$ and $\eta$ are the normalized Gibbs states obtained
by dividing each exponential by its trace.

## Gaussian Gibbs States

The calculus backend evaluates expressions of the form

$$
\exp\!\big(c + a^{\mathsf{T}} R + \tfrac{1}{2}\,R^{\mathsf{T}} G\,R\big).
$$

For this operator to be trace-class, the quadratic part must be negative
definite in the thermal sense. Writing

$$
H \;=\; -G,
$$

the implementation requires $H$ to be positive definite in checked contexts.
Completing the square,

$$
c + a^{\mathsf{T}} R - \tfrac{1}{2}\,R^{\mathsf{T}} H R
\;=\; c + \tfrac{1}{2}\,a^{\mathsf{T}} H^{-1} a
      - \tfrac{1}{2}\,(R - H^{-1} a)^{\mathsf{T}} H\,(R - H^{-1} a).
$$

So the Gaussian mean is

$$
d \;=\; H^{-1} a.
$$

The covariance and partition function are computed from the symplectic
spectrum of $H$. Specifically, with positive symplectic eigenvalues
$\{\nu_k\}_{k=1}^{m}$ of $H$,

$$
\log \operatorname{Tr}\exp\!\big(-\tfrac{1}{2} R^{\mathsf{T}} H R\big)
\;=\; -\sum_{k=1}^{m} \log\!\big(2 \sinh(\hbar\nu_k / 2)\big),
$$

and the thermal covariance is

$$
\Gamma \;=\; \tfrac{\hbar}{2}\,\coth\!\big(\tfrac{i\hbar}{2}\,\Omega H\big)\,(i\,\Omega).
$$

## Partial Trace for Gaussian States

A joint Gaussian state on $\mathcal{H}_0 \otimes \mathcal{H}$ stores a joint
mean and covariance:

$$
d_{\mathrm{joint}} = \begin{pmatrix} d_0 \\ d_1 \end{pmatrix},
\qquad
\Gamma_{\mathrm{joint}} =
\begin{pmatrix} \Gamma_{00} & \Gamma_{01} \\ \Gamma_{10} & \Gamma_{11} \end{pmatrix}.
$$

Partial trace is block selection:

$$
\operatorname{Tr}_{\mathcal{H}}[\pi]\;\text{ has mean } d_0,\; \text{covariance } \Gamma_{00},
$$

$$
\operatorname{Tr}_{\mathcal{H}_0}[\pi]\;\text{ has mean } d_1,\; \text{covariance } \Gamma_{11}.
$$

The trace/normalization is unchanged by partial trace.

## Dual Gradients

The module returns the true ascent gradient of the dual objective.

Using

$$
\rho_s = \exp\!\big((U_s \otimes I + I \otimes V_s - \alpha_s C)/\varepsilon\big),
\qquad
\eta = \exp\!\big(-\textstyle\sum_{r} U_r/\tau\big),
$$

the operator gradients are

$$
\nabla_{V_s} \mathcal{D} \;=\; \sigma_s - \operatorname{Tr}_{\mathcal{H}_0}[\rho_s],
\qquad
\nabla_{U_s} \mathcal{D} \;=\; \eta - \operatorname{Tr}_{\mathcal{H}}[\rho_s].
$$

These gradients are marginal mismatches.

At a dual optimum:

$$
\operatorname{Tr}_{\mathcal{H}_0}[\rho_s] = \sigma_s,
\qquad
\operatorname{Tr}_{\mathcal{H}}[\rho_s] = \eta \quad \text{for every } s.
$$

Thus the optimized $\eta$ is the common left marginal of all optimal
couplings. That is why $\eta$ is the barycenter.

## Moment Coordinates for Gradients

A signed Gaussian difference

$$
\Delta \;=\; \mu_1 - \mu_2
$$

acts on a quadratic operator

$$
A \;=\; c + a^{\mathsf{T}} R + \tfrac{1}{2}\,R^{\mathsf{T}} G\,R
$$

through moment differences:

$$
\partial_c \mathcal{D} = \operatorname{Tr}[\Delta],
\qquad
\partial_a \mathcal{D} = \operatorname{Tr}[\Delta\,R],
\qquad
\partial_G \mathcal{D} = \tfrac{1}{2}\,\operatorname{Tr}[\Delta\,R\,R^{\mathsf{T}}].
$$

If

$$
\mu_i = (z_i, d_i, \Gamma_i),
\qquad
M_i = \Gamma_i + d_i\,d_i^{\mathsf{T}},
$$

then the parameter gradients are

$$
\nabla_c \;=\; z_1 - z_2,
$$

$$
\nabla_a \;=\; z_1 d_1 - z_2 d_2,
$$

$$
\nabla_G \;=\; \tfrac{1}{2}\,\big(z_1 M_1 - z_2 M_2\big).
$$

The implementation returns these arrays stacked over $s$.

## Why Residuals Explain the Barycenter

During optimization, the gradients measure how far the current Gibbs
couplings are from satisfying the barycenter constraints:

$$
V\text{-residual:}\quad \sigma_s - \operatorname{Tr}_{\mathcal{H}_0}[\rho_s],
\qquad
U\text{-residual:}\quad \eta - \operatorname{Tr}_{\mathcal{H}}[\rho_s].
$$

If both residuals are small, then:

1. each coupling $\rho_s$ has approximately the prescribed input marginal
   $\sigma_s$;
2. all couplings have approximately the same barycenter marginal $\eta$;
3. $\eta$ is generated by the optimized barycenter dual potential.

Therefore, $\eta$ is the regularized Gaussian barycenter up to optimization
tolerance.

The Wigner plots in the example notebook visualize this final $\eta$ against
the input states. The residual plots show the optimization evidence that the
visualized state is not just a heuristic average, but the state satisfying
the regularized transport optimality equations.

## Important Practical Notes

- The implementation is moment-level and Gaussian-specific.
- Dense Fock-space matrices are not constructed.
- Quadratic Gibbs states require confining quadratic exponents.
- For JAX/JIT workflows, value-dependent validation should usually be
  disabled with `Context(..., enable_checks=False)`.
- The optimizer in the notebook is intentionally simple. It is useful for
  explanation and experimentation, not a production-grade solver.
