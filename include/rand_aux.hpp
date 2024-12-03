#include <random>

template <typename dType>
void fill_rand_uniform(dType *src, int N, dType low, dType high) {
    
}

template <typename dType>
void fill_rand_normal(dType *src, int N, double mean, double stdev) {
    std::default_random_engine generator;
    std::normal_distribution<dType> distribution(mean, stdev);
    for (int i=0; i<N; i++) {
        src[i] = distribution(generator);
    }
}