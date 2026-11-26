# Theory: numerical methods

**θ-method.** For :math:`u_t = Au + b`, :math:`(I-\theta\Delta tA)u^{n+1}=(I+(1-\theta)\Delta tA)u^n+\Delta t(\theta b^{n+1}+(1-\theta)b^n)`.
CN (θ = ½) is second order and A-stable; explicit Euler needs :math:`\alpha\Delta t/h^2\le 1/2` for the heat equation.

**von Neumann analysis.** For :math:`u_t+vu_x=Du_{xx}`, the spatial symbol is
:math:`\lambda(kh)\Delta t=-4r\sin^2(kh/2)-ic\sin kh` (central) with :math:`r=D\Delta t/h^2`, :math:`c=v\Delta t/h`, and the
amplification factor is :math:`G=(1+(1-\theta)\lambda)/(1-\theta\lambda)`. Stability requires :math:`|G|\le1` for all :math:`kh`.

**Finite volumes.** Conservative update :math:`d\bar u_i/dt=-(F_{i+1/2}-F_{i-1/2})/h`; with periodic boundaries the
total mass :math:`h\sum\bar u_i` is invariant to round-off for any face reconstruction.

**Verification.** The observed order is :math:`p=\log(e_1/e_2)/\log(h_1/h_2)`. Richardson:
:math:`f_\infty\approx f_1+(f_1-f_2)/(r^p-1)`. GCI (Roache): :math:`\mathrm{GCI}=F_s|\varepsilon|/(r^p-1)`, :math:`F_s=1.25`.

**Projection method.** :math:`u^*=u^n+\Delta t[-\nabla\cdot(uu)+\nu\nabla^2u]`, :math:`\nabla^2p=\nabla\cdot u^*/\Delta t`,
:math:`u^{n+1}=u^*-\Delta t\nabla p` on a MAC grid; the discrete divergence of :math:`u^{n+1}` is zero to solver precision.

**Hess–Smith panels.** Constant sources :math:`q_j` per panel plus one vortex strength :math:`\gamma`; flow tangency at
N mid-points and the Kutta condition close the N+1 system; :math:`C_L=2\gamma\sum_j L_j` (unit chord, unit speed).
