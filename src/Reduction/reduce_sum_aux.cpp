#include "reduce_sum_aux.hpp"

template <typename dType>
dType host_reduce_sum(dType *src, int N) {
    dType acc = (dType) 0;
    for (int i = 0; i < N; i++) {
        acc += src[i];
    }
    return acc;
}

// Explicit instantiation
template double host_reduce_sum<double>(double *src, int N);