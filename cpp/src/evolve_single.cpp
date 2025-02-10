#include "evolve_single.hpp"

#include <torch/library.h>

namespace emulators_cpp {

namespace {

constexpr double _TIME_CONVERSION_COEFF = 0.001;

}

at::Tensor apply_effective_hamiltonian(at::Tensor state_factor, at::Tensor ham_factor, at::Tensor left_bath, at::Tensor right_bath) {
    assert(left_bath.ndim == 3 && left_bath.shape[0] == left_bath.shape[2]);
    assert(right_bath.ndim == 3 && right_bath.shape[0] == right_bath.shape[2]);
    assert(left_bath.shape[2] == state.shape[0] && right_bath.shape[2] == state.shape[2]);
    assert(left_bath.shape[1] == ham.shape[0] && right_bath.shape[1] == ham.shape[3]);

    auto result = at::tensordot(left_bath, state_factor, {2}, {0});
    result = at::tensordot(result, ham_factor, {1, 2}, {0, 2});
    result = at::tensordot(result, right_bath, {3, 1}, {1, 2});
    return result;
}

at::Tensor evolve_single(at::Tensor state_factor, at::Tensor left_bath, at::Tensor right_bath, at::Tensor ham_factor, double dt, bool is_hermitian) {
    auto op = [&left_bath, &right_bath, &ham_factor, dt](at::Tensor x) -> at::Tensor {
        auto result = apply_effective_hamiltonian(
            x,
            ham_factor,
            left_bath,
            right_bath
        );
        result *= -_TIME_CONVERSION_COEFF * c10::complex<double>(0.0, 1.0) * dt;
        return result;
    };

    return krylov_exp(
        op,
        std::move(state_factor),
        1e-8, // exp_tolerance
        1e-8, // norm_tolerance
        50L, // max_krylov_dim
        is_hermitian
    );
}

}
