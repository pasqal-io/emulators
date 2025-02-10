#include "ATen/ATen.h"

namespace emulators_cpp {

at::Tensor apply_effective_hamiltonian(at::Tensor state_factor, at::Tensor ham_factor, at::Tensor left_bath, at::Tensor right_bath);

template <typename F>
at::Tensor krylov_exp(F op, at::Tensor&& v, double exp_tolerance, double norm_tolerance, std::int64_t max_krylov_dim, bool is_hermitian) {
    auto initial_norm = v.norm().template item<double>();

    v /= initial_norm;

    std::vector<at::Tensor> lanczos_vectors = { v };
    auto T = at::zeros({max_krylov_dim + 2, max_krylov_dim + 2}, at::kComplexDouble);

    for (std::int64_t j = 0; j < max_krylov_dim; ++j) {
        auto w = op(*lanczos_vectors.rend());

        auto n = w.norm().template item<double>();

        std::int64_t k_start = is_hermitian ? std::max(0L, j - 1) : 0;
        for (std::int64_t k = k_start; k < j + 1; ++k) {
            // FIXME: https://pytorch.org/cppdocs/api/function_namespaceat_1a149d7fbc52104997c5f5b735b6312a57.html
            auto overlap = at::tensordot(lanczos_vectors[k].conj(), w, {0, 1, 2}, {0, 1, 2});
            T[k][j] = overlap;
            w -= overlap * lanczos_vectors[k];
        }

        auto n2 = w.norm().template item<double>();
        T[j + 1][j] = n2; // FIXME: use Tensor::index_put_ ?

        if (n2 < norm_tolerance) {
            auto T_sliced = T.slice(0, 0, j+1).slice(1, 0, j+1);
            auto expd = at::linalg_matrix_exp(T_sliced);

            auto result = at::zeros_like(v);

            for (std::size_t k = 0; k < lanczos_vectors.size(); ++k) {
                result += initial_norm * expd[k][0] * lanczos_vectors[k];
            }

            return result;
        }

        w /= n2;
        lanczos_vectors.push_back(w);

        // Compute exponential of extended T matrix
        T[j + 2][j + 1] = 1;
        auto T_sliced = T.slice(0, 0, j+3).slice(1, 0, j+3);
        auto expd = at::linalg_matrix_exp(T_sliced);

        // Local truncation error estimation
        auto err1 = at::abs(expd[j + 1][0]).template item<double>();
        auto err2 = at::abs(expd[j + 2][0] * n).template item<double>();

        auto err = err1 < err2 ? err1 : (err1 * err2 / (err1 - err2));

        if (err < exp_tolerance) {
            auto result = at::zeros_like(v);

            for (std::size_t k = 0; k < lanczos_vectors.size(); ++k) {
                result += initial_norm * expd[k][0] * lanczos_vectors[k];
            }

            return result;
        }
    }

    throw std::runtime_error("krylov_exp failed to converge");
}

at::Tensor evolve_single(at::Tensor state_factor, at::Tensor left_bath, at::Tensor right_bath, at::Tensor ham_factor, double dt, bool is_hermitian);

}
