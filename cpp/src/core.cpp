#include "core.hpp"

namespace emulators_cpp {

namespace {

constexpr double _TIME_CONVERSION_COEFF = 0.001; // FIXME: export from Python?


std::int64_t determine_cutoff_index(at::Tensor d, double max_error) {
    auto squared_max_error = max_error * max_error;
    double acc = 0.0;
    for (std::int64_t i = 0; i < d.size(0); ++i) {
        // FIXME this loop
        acc += d.index({i}).template item<double>();
        if (acc > squared_max_error) {
            return i;
        }
    }
    return 0;
}


std::pair<at::Tensor, at::Tensor> split_tensor(
            at::Tensor m,
            double max_error,
            std::int64_t max_rank,
            bool orth_center_right) {
    // FIXME: this can be improved I think
    if (orth_center_right) {
        auto mmT = at::mm(m, m.transpose(0, 1).conj());
        auto [d, q] = at::linalg_eigh(mmT);

        auto max_bond = std::max(
            determine_cutoff_index(d, max_error),
            d.size(0) - max_rank
        );

        auto right = at::mm(q.transpose(0, 1).conj(), m);

        return { q.index({Idx::Slice(), Idx::Slice(max_bond, Idx::None)}), right.index({Idx::Slice(max_bond, Idx::None), Idx::Slice()})};
    }

    auto mTm = at::mm(m.transpose(0, 1).conj(), m);
    auto [d, q] = at::linalg_eigh(mTm);

    auto max_bond = std::max(
        determine_cutoff_index(d, max_error),
        d.size(0) - max_rank
    );

    auto left = at::mm(m, q);

    return { left.index({Idx::Slice(), Idx::Slice(max_bond, Idx::None)}), q.transpose(0, 1).conj().index({Idx::Slice(max_bond, Idx::None), Idx::Slice()})};
}

}

at::Tensor apply_effective_hamiltonian(at::Tensor const& state_factor, at::Tensor const& ham_factor, at::Tensor const& left_bath, at::Tensor const& right_bath) {
    assert(left_bath.ndim == 3 && left_bath.shape[0] == left_bath.shape[2]);
    assert(right_bath.ndim == 3 && right_bath.shape[0] == right_bath.shape[2]);
    assert(left_bath.shape[2] == state.shape[0] && right_bath.shape[2] == state.shape[2]);
    assert(left_bath.shape[1] == ham.shape[0] && right_bath.shape[1] == ham.shape[3]);

    auto result = at::tensordot(left_bath, state_factor, {2}, {0});
    result = at::tensordot(result, ham_factor, {1, 2}, {0, 2});
    result = at::tensordot(result, right_bath, {3, 1}, {1, 2});
    return result;
}

at::Tensor evolve_single(at::Tensor const& state_factor, at::Tensor const& left_bath, at::Tensor const& right_bath, at::Tensor const& ham_factor, double dt, bool is_hermitian, Config const& config) {
    auto op = [&left_bath, &right_bath, &ham_factor, dt] (at::Tensor const& x) -> at::Tensor {
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
        state_factor,
        config.precision * config.extra_krylov_tolerance, // exp_tolerance
        config.precision * config.extra_krylov_tolerance, // norm_tolerance
        config.max_krylov_dim, // max_krylov_dim
        is_hermitian
    );
}


std::pair<at::Tensor, at::Tensor> evolve_pair(at::Tensor const& left_state_factor, at::Tensor const& right_state_factor, at::Tensor const& left_bath, at::Tensor const& right_bath,
        at::Tensor const& left_ham_factor, at::Tensor const& right_ham_factor,
        double dt, bool orth_center_right, bool is_hermitian, Config const& config) {
    auto const left_device = left_state_factor.device();
    auto const right_device = right_state_factor.device();

    auto combined_state_factors = at::tensordot(
        left_state_factor, right_state_factor.to(left_device), {2}, {0}
    ).reshape({left_state_factor.size(0), 4, right_state_factor.size(2)});


    auto combined_hamiltonian_factors =
         at::tensordot(left_ham_factor.to(left_device), right_ham_factor.to(left_device), {3}, {0})
        .transpose(2, 3)
        .reshape({left_ham_factor.size(0), 4, 4, -1});

    auto op = [&left_bath, &right_bath, &combined_hamiltonian_factors, dt] (at::Tensor const& x) -> at::Tensor {
        auto result = apply_effective_hamiltonian(
            x,
            combined_hamiltonian_factors,
            left_bath,
            right_bath
        );
        result *= -_TIME_CONVERSION_COEFF * c10::complex<double>(0.0, 1.0) * dt;
        return result;
    };

    auto evol = krylov_exp(
        op,
        combined_state_factors,
        config.precision * config.extra_krylov_tolerance, // exp_tolerance
        config.precision * config.extra_krylov_tolerance, // norm_tolerance
        config.max_krylov_dim, // max_krylov_dim
        is_hermitian
    ).reshape({left_state_factor.size(0) * 2, 2 * right_state_factor.size(2)});

    auto [l, r] = split_tensor(
        evol,
        config.precision,
        config.max_bond_dim,
        orth_center_right
    );

    return {l.reshape({left_state_factor.size(0), 2, -1}), r.reshape({-1, 2, right_state_factor.size(2)}).to(right_device)};
}

at::Tensor apply_rydberg_sv(HamParameters const& ham_params, at::Tensor state, std::complex<double> coeff) {


    throw std::runtime_error("not implemented");
}

at::Tensor evolve_sv_rydberg(double dt, HamParameters const& ham_params, at::Tensor state, double krylov_tolerance) {
    throw std::runtime_error("not implemented");
}


}
