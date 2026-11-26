# Theory: optimisation

Problem: :math:`\min f(x)` s.t. :math:`g_i(x)\le0`, :math:`h_j(x)=0`, bounds. **KKT** (under LICQ):
:math:`\nabla f+\sum\mu_i\nabla g_i+\sum\lambda_j\nabla h_j=0`, :math:`\mu_i\ge0`, :math:`\mu_ig_i=0`, feasibility. The second-order
sufficient condition requires the Hessian of the Lagrangian to be positive definite on the null space of the
active-constraint Jacobian. For non-convex problems KKT points are only *local* candidates.

**Robust objectives.** :math:`\mathbb E_\xi f`, :math:`\mathrm{CVaR}_\beta=\mathrm{VaR}_\beta+\mathbb E[(f-\mathrm{VaR}_\beta)^+]/(1-\beta)`,
chance constraints :math:`P(g\le0)\ge1-\epsilon` checked with a Wilson lower confidence bound.

**Bayesian optimisation.** GP surrogate, expected improvement :math:`EI=(f^*-\mu)\Phi(z)+\sigma\phi(z)`; constrained BO
multiplies EI by the probability of feasibility; EHVI maximises the expected hypervolume gain.

**Protocol (spec 8G).** Baseline, space, objective, constraints, budget and stopping rule are mandatory; every
evaluation is recorded with feasibility and uncertainty; a baseline optimiser is run; the final design is
re-evaluated with the trusted solver and accepted only if it is feasible there.
