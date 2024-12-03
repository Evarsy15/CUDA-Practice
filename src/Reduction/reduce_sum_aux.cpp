#include "reduce_sum_aux.hpp"

template <typename data_t>
data_t host_reduce_sum(data_t *src, int N) {
    data_t acc = (data_t) 0;
    for (int i = 0; i < N; i++) {
        acc += src[i];
    }
    return acc;
}

// Explicit instantiation
template double host_reduce_sum<double>(double *src, int N);