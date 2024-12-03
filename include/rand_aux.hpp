#include <random>

template <typename dType>
void fill_rand_uniform(dType *src, int N, dType low, dType high) {
    
}

template <typename data_t>
void fill_rand_normal(data_t *src, int N, double mean, double stdev) {
    std::default_random_engine generator;
    std::normal_distribution<data_t> distribution(mean, stdev);
    for (int i=0; i<N; i++) {
        src[i] = distribution(generator);
    }
}