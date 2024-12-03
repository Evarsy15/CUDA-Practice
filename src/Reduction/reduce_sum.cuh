#ifndef REDUCE_SUM_CUH
#define REDUCE_SUM_CUH

template <typename data_t>
data_t ReduceSum(data_t *src, data_t *aux, int N, int block_size);

#endif