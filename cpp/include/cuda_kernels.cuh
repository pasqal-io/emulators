#include <complex>

template <typename T>
struct HamDiagonalCache {
    bool fill = false;
    bool valid = false;
    T* diagonal = nullptr;
};

void call_sv_rydberg_kernel(int qubit_count, double const* deltas, double const* omegas, double const* interaction_matrix,
                std::complex<double> const* state_vector, std::complex<double>* result, std::complex<double> coeff,
                HamDiagonalCache<std::complex<double>> ham_diagonal_cache);
