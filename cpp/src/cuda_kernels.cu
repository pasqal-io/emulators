#include "cuda_kernels.cuh"
#include <cuda/std/complex>

namespace {


__device__ bool is_bit_set(int value, int bit_index) {
    return (value & (1 << bit_index)) != 0;
}

__device__ double get_ham_diagonal_element(int qubit_count, double const* deltas, double const* omegas, double const* interaction_matrix, int index) {
    double result = 0.;

    for (int first_qubit = 0; first_qubit < qubit_count; ++first_qubit) {
        bool is_first_qubit_set = is_bit_set(index, qubit_count - 1 - first_qubit);

        if (is_first_qubit_set) {
            result -= deltas[first_qubit];

            for (int second_qubit = first_qubit + 1; second_qubit < qubit_count; ++second_qubit) {
                bool is_second_qubit_set = is_bit_set(index, qubit_count - 1 - second_qubit);

                if (is_second_qubit_set) {
                    result += interaction_matrix[first_qubit * qubit_count + second_qubit];
                }
            }
        }
    }

    return result;
}

}

__global__ void sv_rydberg_kernel(int qubit_count, double const* deltas, double const* omegas, double const* interaction_matrix,
                cuda::std::complex<double> const* state_vector, cuda::std::complex<double>* result, cuda::std::complex<double> coeff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (1 << qubit_count)) {
        auto diag = get_ham_diagonal_element(qubit_count, deltas, omegas, interaction_matrix, idx);

        result[idx] = diag * cuda::std::complex<double>(state_vector[idx]);

        for (int qubit = 0; qubit < qubit_count; ++qubit) {
            // bitflip qubit
            int flipped_idx = idx ^ (1 << (qubit_count - 1 - qubit));
            result[idx] += omegas[qubit] * state_vector[flipped_idx];
        }

        result[idx] *= coeff;
    }
}

void call_sv_rydberg_kernel(int qubit_count, double const* deltas, double const* omegas, double const* interaction_matrix,
                std::complex<double> const* state_vector, std::complex<double>* result, std::complex<double> coeff) {
    sv_rydberg_kernel<<<((1 << qubit_count)+511)/512, 512>>>(
        qubit_count,
        deltas,
        omegas,
        interaction_matrix,
        (cuda::std::complex<double> const*) state_vector,
        (cuda::std::complex<double> *) result,
        cuda::std::complex<double>(coeff.real(), coeff.imag())
    );
}
