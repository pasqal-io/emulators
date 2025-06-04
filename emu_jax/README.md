# emu-jax

Experimental Jax version of emu-sv.

- `pip install -U "jax[cuda12]"`, might conflict with `torch` in terms of CUDA libraries
- no noise
- no support of phi
- only the energy observable is guaranteed to work
- jit compilation takes a long time. It is not possible to use a loop jax primitive in krylov_exp, because
the list of lanczos vectors grows and jax needs a constant pytree structure. It is possible to use a `jax.lax.scan` instead?