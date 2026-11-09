# Notes on the Chebyshev time stepper in emu-sv

A Q&A on the Chebyshev algorithm added in commit `5ea4752` (emu-sv: Chebyshev time
steps for noiseless state-vector evolution on GPU, #279). Code under discussion:
`emu_base/math/chebyshev.py` and the `use_chebyshev` branch of
`DifferentiableStateVectorEvolution.evolve` in `emu_sv/time_evolution/torch.py`.

---

## 1. Why are the coefficients in `exp_chebyshev_coefficients` Bessel functions?

The Chebyshev expansion of a plane wave is the Jacobi-Anger identity in disguise.

**Chebyshev coefficients are Fourier coefficients.** The Chebyshev polynomials are
defined by `T_k(cos θ) = cos(kθ)`. Substituting `x = cos θ`, expanding `f(x)` on
`[-1, 1]` in `T_k` is the same as expanding `f(cos θ)` as a cosine series in `θ`. The
orthogonality of the `T_k` with weight `1/sqrt(1 - x²)` is the orthogonality of cosines
under that substitution. The coefficient formula is

```
c_0 = (1/π) ∫_0^π f(cos θ) dθ
c_k = (2/π) ∫_0^π f(cos θ) cos(kθ) dθ     (k ≥ 1)
```

**For the plane wave these integrals are Bessel's integral.** With
`f(x) = exp(-iαx)`,

```
c_k = (2/π) ∫_0^π exp(-iα cos θ) cos(kθ) dθ
```

Bessel's integral representation is

```
J_k(α) = (1/π) ∫_0^π cos(kθ - α sin θ) dθ
       = (i^{-k}/π) ∫_0^π exp(iα cos θ) cos(kθ) dθ
```

(the second form follows from the first by `θ → π/2 - θ`). Conjugating, or equivalently
using `J_k(-α) = (-1)^k J_k(α)`,

```
(1/π) ∫_0^π exp(-iα cos θ) cos(kθ) dθ = (-i)^k J_k(α)
```

which gives the docstring formula `c_0 = J_0(α)`, `c_k = 2 (-i)^k J_k(α)`.

**Equivalently, the Jacobi-Anger identity** `exp(iz cos θ) = Σ_k i^k J_k(z) exp(ikθ)`
over all integers `k`, folded with `J_{-k} = (-1)^k J_k`, is
`exp(iz cos θ) = J_0(z) + 2 Σ_{k≥1} i^k J_k(z) cos(kθ)`; replace `cos(kθ)` by `T_k(x)`
and `z` by `-α`.

**Why this matters for truncation.**

- `|J_k(α)|` is roughly constant for `k < α` and decays super-exponentially once
  `k > α`. This is why the code starts Miller's recurrence at `α + 12 α^{1/3} + 60`,
  well into the decaying regime, and why the term count is close to `α` plus a small
  overhead, nearly independent of the tolerance.
- The coefficients are known in closed form, so the error bound `Σ_{k>K} |c_k|` is
  computed exactly from the dropped terms. With `|T_k(x)| ≤ 1` on `[-1, 1]` the bound
  is rigorous for any Hermitian operator with spectrum scaled into `[-1, 1]`.

Implementation details that follow from the derivation:

- The normalisation `J_0 + 2 Σ_k J_{2k} = 1` is the Jacobi-Anger identity at
  `θ = π/2` (`x = 0`), where only the even cosines survive. Miller's recurrence gives
  the `J_k` up to an overall scale; this identity fixes it.
- The conjugation for `α < 0` comes from `exp(-iαx)` being the conjugate of
  `exp(-i|α|x)` for real `x`, with the `T_k` real.

---

## 2. How does the number of terms scale with α?

Linearly in `α`, plus a sublinear correction:

```
K(α, ε) ≈ |α| + c(ε) · |α|^(1/3) + O(1)
```

where `c(ε)` grows only slowly with the tolerance `ε`. Three regimes of `J_k(α)` in `k`:

- **`k < α` (oscillatory).** `J_k(α)` oscillates with amplitude of order `α^(-1/2)`.
  No term is small, so you cannot stop here. These are the modes that resolve a phase
  winding of `α` radians across the spectrum: `exp(-iαx)` has `α/π` oscillations on
  `[-1, 1]`, and a degree-`K` polynomial can only resolve about `K/π`.
- **`k ≈ α` (turning point).** The transition happens over a window of width
  proportional to `α^(1/3)`, the Airy region:
  `J_k(α) ≈ (2/α)^(1/3) Ai((k - α)(2/α)^(1/3))`.
- **`k > α` (decaying).** Decay like `exp(-(2/3)(k - α)^(3/2) (2/α)^(1/2))` near the
  turning point, eventually super-exponential as `(eα/2k)^k`. Reaching tolerance `ε`
  costs extra terms scaling like `α^(1/3) · log(1/ε)^(2/3)`.

For double-precision tolerance the overhead beyond `|α|` is a few dozen terms for `α`
of order 1 to 10, and roughly 100 extra terms for `α` of order 1000.

**Consequence for time stepping.** Since `α = ‖H‖ · Δt` after rescaling the spectrum to
`[-1, 1]`, the cost of a step is linear in the step size. The cost per unit simulated
time is almost independent of `Δt`, and the fixed overhead is paid once per step, so the
best strategy is the largest step the observable grid allows.

---

## 3. Could repeated squaring reduce the number of terms, given A is only a callable?

No. With only matrix-vector products available, repeated squaring cannot help, and the
linear scaling in `α` is a hard lower bound.

**Why squaring does not transfer.** Scaling and squaring works for explicit matrices
because `exp(-iαA) = (exp(-iαA/2^s))^(2^s)` and the right-hand side costs `s` matrix
multiplications once the small propagator is formed. Those multiplications square an
operator, not a vector. With `A` only a callable you must apply the small step to `v`
`2^s` times in sequence:

```
2^s · (α / 2^s + overhead)  =  α + 2^s · overhead
```

strictly worse than a single step. Forming the propagator explicitly is out of reach
anyway: the state for 25 qubits is already half a gigabyte and the dense propagator
would be its square.

**Why the linear bound is fundamental.** After `m` applications of `A` to `v`, any method
produces a vector in the Krylov space spanned by `v, Av, ..., A^(m-1) v`, i.e. `p(A) v`
with `deg p < m`. So the best you can do with `m` applications is the best polynomial
approximation of `exp(-iαx)` on the spectrum of `A`, which on an interval of half-width
1 needs degree close to `α`. The truncated Chebyshev series is within a logarithmic
factor of this optimum and spends exactly one operator application per term.

**What can actually reduce the count** (each changes the premise):

- **Krylov adapts to the spectral content of `v`** (see §5). If the state lives in a
  narrow energy window it converges in fewer than `α` terms. The price is
  orthogonalisation and storing the basis.
- **Exploiting structure**: a Trotter splitting applies the interaction diagonal exactly
  as a phase and the driving term as local rotations, at a cost independent of `α`, but
  with a step-size error of order `Δt²` or `Δt³` instead of a controllable tail.
- **Rational approximations** need linear solves with `A`, each an iterative process, so
  for a Hermitian matrix-free operator the polynomial route stays preferable.

---

## 4. How is the Chebyshev series within a logarithmic factor of the optimum?

Through the Lebesgue constant of the Chebyshev projection.

**Projection error versus best error.** Let `S_K f = Σ_{k≤K} c_k T_k` and let `p*` be
the minimax polynomial of degree `K`, with `E_K(f) = ‖f - p*‖_∞`. `S_K` is linear and
reproduces polynomials of degree `≤ K`, so `S_K p* = p*` and

```
f - S_K f = (f - p*) - S_K (f - p*)
‖f - S_K f‖_∞ ≤ (1 + Λ_K) · E_K(f),    Λ_K = ‖S_K‖_∞
```

**The Lebesgue constant is logarithmic.** Under `x = cos θ` the projection is
convolution with the Dirichlet kernel

```
D_K(u) = 1 + 2 Σ_{k=1}^K cos(ku) = sin((K + ½)u) / sin(u/2)
Λ_K = (1/π) ∫_0^π |D_K(u)| du = (4/π²) log K + O(1)
```

The `log K` comes from the `1/sin(u/2)` envelope: `K` lobes with heights decaying like
`1/u` sum to a harmonic series. (Chebyshev-point interpolation has
`(2/π) log K + O(1)`; equispaced interpolation grows exponentially, Runge's phenomenon.)

**For the exponential the series is much closer.** With super-exponentially decaying
coefficients, the error is essentially `c_{K+1} T_{K+1}`, which alternates with
magnitude `|c_{K+1}|` at `K+2` points. De la Vallée Poussin's theorem then gives

```
|c_{K+1}| - Σ_{k>K+1} |c_k|  ≤  E_K(f)  ≤  ‖f - S_K f‖_∞  ≤  |c_{K+1}| + Σ_{k>K+1} |c_k|
```

so the minimax error and the truncation error agree up to the tail-to-leading-term ratio.
In term count the gap is zero or one term. This is also why the stopping rule
`Σ_{k>K} |c_k| ≤ tolerance` in the code is not wasteful.

---

## 5. Lanczos: exponential of the projected operator vs. polynomial in A

Question: Lanczos computes `T_m = V_m^† A V_m` and exponentiates on that subspace; isn't
it better to say it approximates the exponential by the exponential on the spectrum of
the restricted operator, rather than "finding the optimal polynomial"?

Yes, that is the mechanism, and the two views are the same statement. The approximation is

```
x_m = ‖v‖ V_m exp(-iα T_m) e_1
```

Two exact facts:

- `f(T_m) = p(T_m)` with `p` the degree `m-1` polynomial interpolating `f` at the
  eigenvalues of `T_m` (the Ritz values).
- The Lanczos relation `A V_m = V_m T_m + β_m v_{m+1} e_m^†` gives
  `V_m T_m^k e_1 = A^k v / ‖v‖` for `k ≤ m-1`, hence
  `‖v‖ V_m p(T_m) e_1 = p(A) v` for `deg p ≤ m-1`.

So `x_m = p_{m-1}(A) v` with `p_{m-1}` the interpolant of `exp(-iαz)` at the Ritz
values. The projected-exponential view says how it is computed; the polynomial view says
what it is and yields the error bound, since the exact answer is also a function of `A`
applied to `v`.

**Where "adapts to v" comes from.** For any polynomial `p` of degree `≤ m-1`,

```
x - x_m = (f - p)(A) v  -  ‖v‖ V_m (f - p)(T_m) e_1
‖x - x_m‖ ≤ 2 ‖v‖ · min_p  max_{z ∈ supp(v) ∪ Ritz values} |exp(-iαz) - p(z)|
```

(Saad). In exact arithmetic the Lanczos process for `(A, v)` is identical to that for `A`
restricted to the eigenspaces where `v` has weight, so the Ritz values lie in the convex
hull of `supp(v)`, and the Ritz values are the Gauss nodes of the spectral measure of
`v`, concentrating where `v` has weight. Accurate statement: Lanczos is within a factor
two of the best polynomial approximation on the spectral support of `v`; "optimal" should
be "near-optimal for the spectral measure of `v`".

The projected exponential alone does not explain convergence: `exp(-iα T_m)` is exact for
`P_m A P_m`, but the true evolution leaves the Krylov space. The interpolation view
supplies the count: `m` of order `α w` plus overhead, with `w` the half-width of the
energy window of `v`.

Caveat: this holds in exact arithmetic. In floating point the Lanczos vectors lose
orthogonality, spurious Ritz values appear, and the process starts to see eigenvalues
outside the support of `v`.

---

## 6. What are the Ritz values?

The eigenvalues of the projected operator `T_m = V_m^† A V_m` (after Walther Ritz).

**Definition.** With `V_m` an orthonormal basis of the Krylov space
`K_m = span{v, Av, ..., A^(m-1) v}`, diagonalise `T_m = Σ_j θ_j s_j s_j^†`. The `θ_j`
are the Ritz values and `y_j = V_m s_j` the Ritz vectors. `(θ_j, y_j)` is the best
approximate eigenpair available inside `K_m`: the residual `A y_j - θ_j y_j` is
orthogonal to `K_m` (Galerkin condition); equivalently, the Ritz values are the
stationary values of the Rayleigh quotient over `K_m`.

**Where they sit.** They interlace with the eigenvalues of Hermitian `A`, lie in
`[λ_min, λ_max]`, and more precisely in the convex hull of the eigenvalues on which `v`
has weight. Extreme Ritz values converge fast to extreme eigenvalues; interior ones
converge slowly.

**Role in exponentiation.** Individual accuracy is irrelevant. The Ritz values are the
nodes of the `m`-point Gauss quadrature rule for the spectral measure
`μ_v(λ) = Σ_i |⟨φ_i|v⟩|² δ(λ - λ_i)`, with the squared first components of the `s_j`
as weights. Gauss quadrature with `m` nodes integrates polynomials of degree `2m-1`
exactly against `μ_v`, the same statement as `V_m T_m^k e_1 = A^k v / ‖v‖`. So the
interpolant at the Ritz values is a near-best polynomial for `v` even when no Ritz value
is close to any actual eigenvalue.

In this codebase, `krylov_exp` builds exactly this `T_m` and exponentiates it; Chebyshev
uses no spectral information from `v`, only the bound on the spectrum of `A`.

---

## 7. Chebyshev vs. truncated Taylor series of the exponential

Measured on `[-1, 1]`, degree `n` needed for uniform error below `ε`:

| α | ε | Taylor degree | Chebyshev degree | ratio | largest Taylor term |
|---|---|---|---|---|---|
| 1 | 1e-6 | 9 | 7 | 1.3 | 1 |
| 1 | 1e-12 | 14 | 12 | 1.2 | 1 |
| 10 | 1e-6 | 36 | 22 | 1.6 | 2.8e3 |
| 10 | 1e-12 | 46 | 30 | 1.5 | 2.8e3 |
| 30 | 1e-12 | 103 | 58 | 1.8 | 7.8e11 |
| 100 | 1e-12 | 295 | 142 | 2.1 | 1.1e42 |
| 300 | 1e-12 | 838 | 360 | 2.3 | 4.5e128 |

**Degree count: a factor approaching `e`.** The Taylor remainder is
`α^(n+1)/(n+1)! ≈ (eα/n)^n`, which only shrinks once `n > eα`, so Taylor needs about
`2.7α` terms versus `α` plus overhead for Chebyshev.

**Error at equal degree: a factor of about `2^n`.** Beyond the transition,
`J_{n+1}(α) ≈ (eα/2(n+1))^(n+1) / sqrt(2π(n+1))`, the Taylor remainder being the same
without the 2. Origin: `2^(1-n) T_n` is the monic polynomial of minimal sup norm on
`[-1, 1]`, and `x^(n+1) = 2^(-n) T_{n+1} + lower-degree terms`; Chebyshev keeps only the
`2^(-n) T_{n+1}` part of the leading Taylor error and spreads it evenly instead of piling
it at the endpoints.

**Stability: the decisive difference.** Taylor partial sums grow to `α^k/k!` near
`k ≈ α` before cancelling to a result of norm 1, losing about `0.43 α` decimal digits;
beyond `α ≈ 30` nothing survives in double precision. Chebyshev: every `T_k(A) v` has
norm `≤ ‖v‖`, every coefficient has modulus `≤ 2`, and the three-term recurrence is
stable on `[-1, 1]` with error growth at most linear in `k`. Taylor needs scaling and
squaring, which is unavailable here (§3). For this backend the comparison is "works at
all", not "2.3× fewer terms".

---

## 8. Expected ratio of Lanczos to Chebyshev term counts

In exact arithmetic Lanczos never needs more terms than Chebyshev apart from one or two,
and for a generic state with a tight spectral bound the ratio is 1. In this codebase it
is lower, mostly because Chebyshev pays for a Gershgorin estimate of the spectrum.

**Theory.** Both counts are `α_eff + c(ε) α_eff^(1/3) + O(1)` with

- Chebyshev: `α = dt · R`, `R` the Gershgorin half-width from
  `RydbergHamiltonian.spectral_bounds` (`(max d - min d)/2 + Σ|Ω|/2`), never tight;
- Lanczos: `α_eff = dt · w`, `w` the half-width of the energy window carrying the
  state's spectral weight down to the tolerance.

Expected ratio about `w / R` at large `α`, tending to 1 at small `α`.

**Measurement.** 12-atom chain, 7 µm spacing, tolerance 1e-10, with the Chebyshev
tolerance scaled by emu-sv's `CHEBYSHEV_TOLERANCE_FACTOR = 1e-2` (as the code paths
actually run):

| Ω | δ | dt | state | Lanczos | Chebyshev | ratio |
|---|---|---|---|---|---|---|
| 10 | 0 | 0.1 | ground | 40 | 61 | 0.66 |
| 10 | 0 | 0.1 | random | 47 | 61 | 0.77 |
| 0.5 | 0 | 0.1 | ground | 17 | 53 | 0.32 |
| 10 | 0 | 1.0 | ground | 248 | 379 | 0.65 |
| 10 | 0 | 1.0 | random | 265 | 379 | 0.70 |
| 0.5 | 0 | 1.0 | ground | 81 | 318 | 0.25 |
| 0.5 | 0 | 1.0 | random | 157 | 318 | 0.49 |

- **Gershgorin slack.** Even a random state, which fills the spectrum, gives 0.7 to
  0.8: the true spectral half-width is about 80 % of `R` here. Recoverable for Chebyshev
  with a tighter bound (e.g. a few Lanczos steps to estimate extreme eigenvalues).
- **Adaptivity to the state.** The ground state with small Ω converges in a quarter to a
  third of the Chebyshev count, but still needs 81 terms at `dt = 1`: the relevant window
  is where the spectral weight drops below tolerance, not the energy standard deviation.

**What the ratio does not capture.** A Lanczos term costs a matvec plus two inner
products, two axpys and a norm, versus a matvec plus two axpys for Chebyshev. Lanczos
stores every basis vector (250 iterations at 25 qubits would need 250 copies of a 512 MB
state, hence `DEFAULT_MAX_KRYLOV_DIM = 100`); Chebyshev keeps three vectors regardless of
`α`. On GPU, Chebyshev can win on wall time and certainly on memory while losing on term
count by a factor of 0.65 to 0.75.

---

## 9. Is near-optimality because Chebyshev polynomials oscillate like the target?

Partly, but the resemblance is not the mechanism. Near-optimality is a property of the
basis and holds for any smooth function: the expansion of `exp(-αx)` (imaginary time),
with modified Bessel coefficients `I_k(α)`, is just as close to minimax, with no
oscillation at all.

**What actually makes it near-optimal.** Under `x = cos θ` the Chebyshev series is a
Fourier series and the interval becomes a circle with no endpoints:

- **The error equioscillates.** The error is dominated by `c_{K+1} T_{K+1}`, which
  attains `±1` at `K+2` points, the alternation-theorem criterion for minimax. `T_n` is
  the polynomial that oscillates most evenly, independent of the target.
- **Small Lebesgue constant** (the `log K` bound), valid for every continuous function.
- **Convergence rate set by analyticity in Bernstein ellipses.** Coefficients of a
  function analytic inside the ellipse with foci `±1` and parameter `ρ` decay like
  `ρ^(-k)`. These ellipses hug the interval, whereas Taylor needs a whole disk. An
  ellipse of parameter `ρ` has semi-minor axis `(ρ - 1/ρ)/2`, about half the radius of
  the corresponding disk, which halves the growth exponent of an entire function of
  exponential type: that is the `2^n` factor of §7.

**Where the oscillation picture is right.** It explains why the count is `α`. The target
has constant local frequency `α` in `x`; `T_k(x) = cos(k arccos x)` has local frequency
`k / sqrt(1 - x²)`, slowest at `x = 0` where it equals `k`. The degree-`K` Chebyshev
polynomials oscillate at least as fast as the target everywhere exactly when `K ≥ α`,
with the binding constraint at the centre. That is the Bessel turning point `k = α` seen
from the other side, and it is sharp.

It also explains why zero counting only gives `K ≥ 2α/π`: zeros of a well-behaved
polynomial of degree `K` follow the arcsine density `K / (π sqrt(1 - x²))`, denser at the
endpoints, so a polynomial cannot spend its zeros uniformly the way the plane wave does
(uniform zeros is the Runge situation). The sparse spot at `x = 0` costs the factor
`π/2`.

Summary: the count `α` is a matching of local frequencies at the centre of the interval,
paid by any polynomial method. The near-optimality on top, paying `α` plus a small
overhead and not more, is the Fourier-series-in-disguise structure of the Chebyshev
basis, unrelated to the target being oscillatory.
