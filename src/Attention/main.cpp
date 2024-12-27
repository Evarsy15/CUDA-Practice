#include <torch/extension.h>

torch::Tensor flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor naive_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v);

torch::Tensor scaled_dot_product(torch::Tensor q, torch::Tensor k);
void softmax(torch::Tensor s);
torch::Tensor matrix_multiply(torch::Tensor p, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", torch::wrap_pybind_function(flash_attention), "flash_attention");
    m.def("naive_attention", torch::wrap_pybind_function(naive_attention), "naive_attention");

    m.def("scaled_dot_product", torch::wrap_pybind_function(scaled_dot_product), "scaled_dot_product");
    m.def("softmax", torch::wrap_pybind_function(softmax), "softmax");
    m.def("matrix_multiply", torch::wrap_pybind_function(matrix_multiply), "matrix_multiply");
}
