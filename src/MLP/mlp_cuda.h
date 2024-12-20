#include "mlp.h"

class MLP_CUDA : public MLP {
public:
    MLP_CUDA(int n, vector<int> layers) 
        : MLP(n, layers) { params.batch_size = 1; }

    double* forward(vector<double*> &data, int num_data);
    double* forward_single(double *data);
    
    void allocate_gpu_memory();
    void copy_mlp_into_gpu();
    void free_gpu_memory();

    float get_total_time_forward() { return total_time_forward; }

private:
    vector<double*> d_w;   // Device Weight (transposed, row-wise) 
    vector<double*> d_b;   // Device Bias
    vector<double*> d_z;   // Device Intermediates
    vector<double*> d_a;   // Device Internal Outputs

    void __forward(double *data, double *res);
    void __forward(vector<double*> &data, double *res, int start, int batch_size);

    float total_time_forward;
};