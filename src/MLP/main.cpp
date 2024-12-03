#include <string>
#include <iostream>
#include "mlp.h"
#include "mlp_cpu.h"
#include "mlp_cuda.h"

#define classes 52
#define in_dim 15
#define epochs 400
#define inputs in_dim*in_dim
#define batch 16

using namespace std;

float total_time_forward = 0;

int main(int argc, char* argv[]) {
    vector<double*> input_data;
    vector<double*> output_data;

    ifstream input("./data/input_data.txt");
    for (int i = 0; i < classes; i++) {
        input_data.push_back(new double[inputs]);
        for (int k = 0; k < inputs; k++)
	        input >> input_data[i][k];
    }

    ifstream output("./data/output_data.txt");
    for (int i = 0; i < classes; i++) {
	    output_data.push_back(new double[classes]);
	    for (int k = 0; k < classes; k++)
	        output >> output_data[i][k];
    }
    
    /* 
        Run forwarding path with CPU-Only MLP
    */
    MLP_CPU NN_CPU(inputs, {98, 65, 50, 30, 25, 40, classes});
    NN_CPU.load_weights("./weights.csv");
    int cpu_val = 0;
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < classes; i++) {
            double* x = NN_CPU.forward(input_data[i], inputs);
            total_time_forward += timer().getCpuElapsedTimeForPreviousOperation();
            MLP::match_single(x, output_data[i], classes, cpu_val);
            delete[] x;
        }
    }
    cout << ">> MLP - CPU <<" << endl;
    cout << "Inference total forward = " << total_time_forward << " ms, " \
         << "Average forward per epoch = " << total_time_forward / epochs << " ms" << endl;
    cout << "Accuracy : " << (float) cpu_val / (epochs * classes) * 100.0 << "%" << endl;
    cout << endl;

    /* 
        Run forwarding path with MLP w/ CUDA
    */
    MLP_CUDA NN_CUDA(inputs, {98, 65, 50, 30, 25, 40, classes});
    NN_CUDA.set_batch_size(batch);
    NN_CUDA.allocate_gpu_memory();
    float CUDA_malloc_overhead = timer().getCpuElapsedTimeForPreviousOperation();

    // Copy MLP layers (weights and biases) into GPU & measure data transfer time.
    NN_CUDA.load_weights("./weights.csv");
    //NN_CUDA.convert_weight_into_row_wise_format();
    NN_CUDA.copy_mlp_into_gpu();
    float MLP_transfer_overhead = timer().getCpuElapsedTimeForPreviousOperation();

    // MLP execution
    float MLP_execution_time = 0.0f;
    int num_data = input_data.size();
    int output_size = classes;
    int gpu_val = 0; // match success counter

    for (int e = 0; e < epochs; e++) {
        double *res = NN_CUDA.forward(input_data, num_data);
        MLP_execution_time += NN_CUDA.get_total_time_forward();
        MLP::match(res, output_data, num_data, classes, gpu_val);
        delete[] res;
    }
    
    /*
    for (int e = 0; e < epochs; e++) {
        MLP_execution_time += timer().getCpuElapsedTimeForPreviousOperation();
        for (int i = 0; i < classes; i++) {
            double* x = NN_CUDA.forward_single(input_data[i]);
            // MLP::match(x, output_data[i], classes, cpu_val);
            int pos1 = distance(x, max_element(x, x + classes));
            int pos2 = distance(output_data[i], max_element(output_data[i], output_data[i] + classes));
            gpu_val += pos1 == pos2;
            // if (!e) print_double_array(x, classes, "x", i);
            delete[] x;
        }
    }
    */

    cout << ">> MLP - GPU <<" << endl;
    cout << "Initial CUDA Memory allocation overhead : " << CUDA_malloc_overhead << " ms" << endl;
    cout << "MLP transfer overhead : " << MLP_transfer_overhead << " ms" << endl;
    cout << "Inference total forward = " << MLP_execution_time << " ms, " \
         << "Average forward per epoch = " << MLP_execution_time / epochs << " ms" << endl;
    cout << "Accuracy : " << (float) gpu_val / (epochs * classes) * 100.0 << "%" << endl;

    NN_CUDA.free_gpu_memory();
}

void print_double_array(double *x, int n, std::string array_name, int array_id) {
    std::cout << array_name << "[" << array_id << "] : [";
    for (int i = 0; i < n; i++) {
        std::cout << x[i] << ' ';
    }
    std::cout << "]\n";
}