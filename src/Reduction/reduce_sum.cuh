#ifndef REDUCE_SUM_CUH
#define REDUCE_SUM_CUH

template <typename dType>
dType ReduceSum(dType *src, dType *aux, int N, int block_size);

#endif