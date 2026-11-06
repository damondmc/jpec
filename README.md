# jpec

**J**AX-accelerated com**P**lex twosp**E**ct harmoni**C**: matched filtering for continuous
gravitational waves from neutron stars in binary systems.

A spinning neutron star in a binary emits a nearly monochromatic gravitational wave whose
frequency the orbit modulates: over each orbital period the signal sweeps back and forth
across many frequency bins of the detector's short Fourier transforms (SFTs). Like TwoSpect,
jpec finds the periodicity this leaves in the SFT power with a second Fourier transform, but
it keeps the complex harmonics in orbital phase and matches them against templates of the
expected modulation, instead of matching their power. The match is linear in those
harmonics, so one inverse FFT scores every orbital phase at once, and the whole search runs
as JAX on CPU or GPU.

For a template at $(f_0, a_p, P, \alpha, \delta)$ the statistic at orbital phase $\phi_b$ is

$$
S(f_0, a_p, P, \alpha, \delta \,;\, \phi_b)
= \mathrm{Re} \sum_j c_j \, e^{\, i j \phi_b},
\qquad
c_j = \frac{1}{\lVert \tau \rVert} \sum_k \tau^{\ast}_{jk}(\Delta k) \; F_{jk}(f_0, \alpha, \delta, P)
$$

with

- $f_0$: signal frequency at the Solar System Barycenter
- $a_p$: projected semi-major axis, $a \sin i / c$, in light-seconds
- $P$: orbital period, $\Omega = 2\pi / P$
- $\alpha$, $\delta$: right ascension and declination
- $\phi_b$: orbital phase, $\Omega T_{\mathrm{asc}}$
- $\Delta k = f_0 a_p \Omega\, T_{\mathrm{SFT}}$: modulation depth, in bins of width
  $1 / T_{\mathrm{SFT}}$

## The computation

A source in a circular orbit arrives at the detector with frequency

$$
f(t) = f_0 \big(1 + \vec v(t)\cdot\hat n\big) - \Delta f \cos(\Omega t - \phi_b),
\qquad \Delta f = f_0 a_p \Omega,
$$

with $\vec v$ the detector's velocity over $c$ and $\hat n$ the direction to the source. In
the time-frequency plane of the SFTs the
signal is the window's power response along that arc, and the statistic is built in four
steps, each a stage of the pipeline.

**1. Doppler correction.** The complex amplitudes of SFT $n$, at midpoint $t_n$, are
shifted by $f_0 T_{\mathrm{SFT}}\, \vec v(t_n)\cdot\hat n$ bins, so the arc depends on time
through the orbital phase $\theta = \Omega t - \phi_b$ alone. The power $p_k(t_n)$ at bin
offset $k$ from $f_0$ is then normalized by a running-median floor.

**2. The data's harmonics.** The power is decomposed in orbital phase over the $N$ SFTs,

$$
F_{jk} = \big\langle p_k(t_n)\, e^{-i j \Omega t_n} \big\rangle_n,
\qquad 1 \le j \le j_{\max} = \lceil 2\pi\,\Delta k \rceil,
$$

a direct average over the SFTs at their own timestamps, so gaps and uneven sampling cost
nothing; $j = 0$, the mean power, is dropped. It depends on $(f_0, \alpha, \delta, P)$ and
is shared by every $a_p$ and $\phi_b$.

**3. The template's harmonics.** The arc's power response
$T_k(\theta) = W(k + \Delta k \cos\theta)$, with $W$ the window's power kernel, has the
Fourier series

$$
\tau_{jk} = \frac{1}{2\pi}\int_0^{2\pi} W(k + \Delta k\cos\theta)\, e^{-i j \theta}\, d\theta ,
$$

which depends on $\Delta k$ alone, so one grid of templates serves every frequency, period
and sky position.

**4. The matched filter.** The match is

$$
c_j = \frac{1}{\lVert \tau \rVert} \sum_k \tau^{\ast}_{jk} F_{jk},
\qquad \lVert \tau \rVert^2 = \sum_{jk} \lvert \tau_{jk} \rvert^2,
$$

and the unknown orbital phase is a linear phase on the harmonics, so one inverse FFT over
$j$ gives $S$ at every $\phi_b$.

The template bank grids the five axes other than $\phi_b$: a body-centred cubic lattice on
$(f_0, \Delta k, \Omega^2)$ and a hexagonal lattice on the sky, laid in the eigenbasis of
the detector velocity's covariance. Both are spaced by the mismatch metric

$$
g_{\mu\nu} = g_{\mathrm{bin}} \, T_{\mathrm{SFT}}^2 \,
\big\langle \partial_\mu f(t_n) \, \partial_\nu f(t_n) \big\rangle_n ,
$$

the window's curvature per squared frequency bin times the SFT average of the frequency
track's derivatives with respect to the parameters, in bins.

## Layout

| package | what it holds |
|---|---|
| `jpec/waveform/` | the signal templates |
| `jpec/detector/` | reading SFTs, the detector's motion, and the Doppler correction |
| `jpec/stat/` | matched filtering on the TwoSpect harmonics, and the efficiency fit |
| `jpec/bank/` | the metric, the lattices and the tiling of the parameter space |
| `jpec/engine/` | the search loop over a template bank, and the toplist it keeps |
| `jpec/injection/` | simulated sources, injected into SFTs |
| `jpec/noise/` | the running-median noise floor, and Gaussian noise SFTs |
| `jpec/pipelines/` | the `jpec` command: SFT checks, search and injection runs |

## Install

To install the environment, run
```bash
uv sync
```

## Example

Match one template against a simulated signal in noise, over every orbital phase:

```python
import jax.numpy as jnp
import numpy as np

from jpec.constant import ARC_PAD_BINS, DAY, PHASE_OVERSAMPLE
from jpec.params import Parameters
from jpec.stat.matched_filter import (
    data_harmonics,
    max_over_phase,
    orbital_phase_mod_2pi,
)
from jpec.waveform.harmonics import (
    harmonic_count,
    phase_sample_count,
    template_harmonics,
)
from jpec.waveform.plane import arc_bin_offsets, plane_from_arc
from jpec.waveform.track import frequency_offset
from jpec.waveform.window import Window

tsft = 1800.0
window = Window(kind="tukey", beta=0.001)
signal = Parameters(f0=150.0, ap=6.6, period=9.0 * DAY, phi_b=2.3, alpha=1.2, delta=0.3)

# prepare the injection: the signal's power over four orbits of gap-free SFTs, in noise
modulation_depth_bins = float(signal.modulation_depth_bins(tsft))
bin_offsets = arc_bin_offsets(modulation_depth_bins, pad_bins=ARC_PAD_BINS)
times = jnp.arange(0.0, 4 * signal.period, tsft) + tsft / 2.0
arc = frequency_offset(
    times,
    modulation_depth=signal.modulation_depth,
    omega=signal.omega,
    phi_b=signal.phi_b,
)
signal_power = plane_from_arc(arc * tsft, bin_offsets, window=window)
noise = np.random.default_rng(0).standard_normal(signal_power.shape)
sft_power = signal_power + 0.35 * signal_power.std() * noise

# the template: its harmonics at the signal's modulation depth
j_max = harmonic_count(modulation_depth_bins)
n_phi = phase_sample_count(j_max, oversample=PHASE_OVERSAMPLE)
tau = template_harmonics(
    bin_offsets,
    modulation_depth_bins=modulation_depth_bins,
    n_theta=n_phi,
    j_max=j_max,
    window=window,
)

# the data's harmonics: a second Fourier transform, of the power in orbital phase
phase = orbital_phase_mod_2pi(times, signal.omega)
F = data_harmonics(sft_power, jnp.asarray(phase), j_max=j_max)

# the matched filter, over every orbital phase
S, phase_idx = max_over_phase(F, tau, n_phi=n_phi)
phi_b = 2.0 * np.pi * int(phase_idx) / n_phi
print(f"loudest at phi_b = {phi_b:.2f} rad; injected {signal.phi_b:.2f} rad")
```

It prints

```
loudest at phi_b = 2.29 rad; injected 2.30 rad
```

[`examples/`](examples/) has a search over $(f_0, a_p, P)$ at a fixed sky position, on
Gaussian noise with and without an injected source, and a search and an injection config
for the `jpec` commands.

## References

jpec builds on the TwoSpect search:

E. Goetz and K. Riles, "An all-sky search algorithm for continuous gravitational waves
from spinning neutron stars in binary systems", Class. Quantum Grav. 28, 215006 (2011),
[doi:10.1088/0264-9381/28/21/215006](https://doi.org/10.1088/0264-9381/28/21/215006),
[arXiv:1103.1301](https://arxiv.org/abs/1103.1301).
