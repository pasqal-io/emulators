#include <complex>

void call_sv_rydberg_kernel(int qubit_count, double const* deltas, double const* omegas, double const* interaction_matrix,
                std::complex<double> const* state_vector, std::complex<double>* result, std::complex<double> coeff);
