#include <random>

void fill_rand(double *A, int size, double mean, double stdev) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stdev);
    for (int i=0; i<size; i++) {
        A[i] = distribution(generator);
    }
}

void generate_random_matrix(double *A, double *B, double *C, 
                            const int M, const int K, const int N) {
    double mean  = 0.0;
    double stdev = 1.0;
    fill_rand(A, M*K, mean, stdev);
    fill_rand(B, K*N, mean, stdev);
}