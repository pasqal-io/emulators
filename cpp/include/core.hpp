#include "ATen/ATen.h"

namespace emulators_cpp {

namespace Idx = at::indexing;

struct Config {
    double precision = 0.;
    double extra_krylov_tolerance = 0.;
    std::int64_t max_krylov_dim = 0;
    std::int64_t max_bond_dim = 0;
};

struct HamParameters {
    at::Tensor omegas;
    at::Tensor deltas;
    at::Tensor interaction_matrix;
};

at::Tensor apply_effective_hamiltonian(at::Tensor const& state_factor, at::Tensor const& ham_factor, at::Tensor const& left_bath, at::Tensor const& right_bath);

template <typename F>
at::Tensor krylov_exp(F op, at::Tensor const& v, double exp_tolerance, double norm_tolerance, std::int64_t max_krylov_dim, bool is_hermitian) {
    auto initial_norm = v.norm().template item<double>();

    std::vector<at::Tensor> lanczos_vectors = { v / initial_norm };

    auto T = at::zeros({max_krylov_dim + 2, max_krylov_dim + 2}, at::kComplexDouble);

    // FIXME: is there another way to do this???
    std::vector<long int> dims(v.sizes().size());
    std::iota(std::begin(dims), std::end(dims), 0);

    for (std::int64_t j = 0; j < max_krylov_dim; ++j) {
        auto w = op(*(lanczos_vectors.rbegin()));

        auto n = w.norm().template item<double>();

        std::int64_t k_start = is_hermitian ? std::max(0L, j - 1) : 0;
        for (std::int64_t k = k_start; k < j + 1; ++k) {
            // FIXME: https://pytorch.org/cppdocs/api/function_namespaceat_1a149d7fbc52104997c5f5b735b6312a57.html
            auto overlap = at::tensordot(lanczos_vectors[k].conj(), w, dims, dims); // FIXME: dims is hacky?
            T.index_put_({k, j}, overlap);
            w -= overlap * lanczos_vectors[k];
        }

        auto n2 = w.norm().template item<double>();
        T.index_put_({j+1, j}, n2);

        if (n2 < norm_tolerance) {
            auto T_sliced = T.index({Idx::Slice(Idx::None, j+1), Idx::Slice(Idx::None, j+1)});
            auto expd = at::linalg_matrix_exp(T_sliced);

            auto result = at::zeros_like(v);

            for (std::int64_t k = 0; k < std::int64_t(lanczos_vectors.size()); ++k) {
                result += initial_norm * expd.index({k, 0}) * lanczos_vectors[k];
            }

            return result;
        }

        w /= n2;
        lanczos_vectors.push_back(w);

        // Compute exponential of extended T matrix
        T.index_put_({j+2, j+1}, 1);
        auto T_sliced = T.index({Idx::Slice(Idx::None, j+3), Idx::Slice(Idx::None, j+3)});
        auto expd = at::linalg_matrix_exp(T_sliced);

        // Local truncation error estimation
        auto err1 = at::abs(expd.index({j + 1, 0})).template item<double>();
        auto err2 = at::abs(expd.index({j + 2, 0}) * n).template item<double>();

        auto err = err1 < err2 ? err1 : (err1 * err2 / (err1 - err2));

        if (err < exp_tolerance) {
            auto result = at::zeros_like(v);

            for (std::int64_t k = 0; k < std::int64_t(lanczos_vectors.size()); ++k) {
                result += initial_norm * expd.index({k, 0}) * lanczos_vectors[k];
            }

            return result;
        }
    }

    throw std::runtime_error("krylov_exp failed to converge");
}

at::Tensor evolve_single(at::Tensor const& state_factor, at::Tensor const& left_bath, at::Tensor const& right_bath, at::Tensor const& ham_factor,
        double dt, bool is_hermitian, Config const& config);

std::pair<at::Tensor, at::Tensor> evolve_pair(at::Tensor const& left_state_factor, at::Tensor const& right_state_factor, at::Tensor const& left_bath, at::Tensor const& right_bath,
        at::Tensor const& left_ham_factor, at::Tensor const& right_ham_factor,
        double dt, bool orth_center_right, bool is_hermitian, Config const& config);

at::Tensor apply_rydberg_sv(HamParameters const& ham_params, at::Tensor state, std::complex<double> coeff = 1.);

at::Tensor evolve_sv_rydberg(double dt, HamParameters const& ham_params, at::Tensor state, double krylov_tolerance);

}
