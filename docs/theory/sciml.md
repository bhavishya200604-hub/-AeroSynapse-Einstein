# Theory: scientific ML

**PINN loss.** :math:`\mathcal L=\lambda_r\,\overline{r^2}+\sum_k\lambda_k\mathcal L_k`, :math:`r=\mathcal N[u_\theta]-f` computed by
automatic differentiation. Hard constraints replace a penalty by a trial function, e.g. :math:`u=g+\phi N_\theta`
with :math:`\phi=0` on the boundary (exact satisfaction, no loss imbalance).

**Residual-based adaptive sampling.** RAR adds the top-k residual points from a candidate pool; RAD resamples
from :math:`p\propto|r|^k/\mathbb E|r|^k+c`; R3 keeps points above the mean residual and resamples the rest.

**Causal weighting.** Temporal residual chunks :math:`L_i` are weighted by :math:`w_i=\exp(-\epsilon\sum_{k<i}L_k)`.

**FNO.** :math:`v_{l+1}=\sigma(Wv_l+\mathcal F^{-1}(R\cdot\mathcal Fv_l))` with truncated modes; weights live in Fourier
space and can be applied on other grids (resolution transfer must still be *tested*).

**DeepONet.** :math:`G(a)(y)=\sum_k b_k(a(x_1..x_m))t_k(y)`; fixed sensors, arbitrary query points.

**Break-even queries.** A surrogate pays off after :math:`N^*=(C_{train}+C_{data})/(t_{solver}-t_{model})` queries.
