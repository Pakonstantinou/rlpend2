#include "Net.h"
#include <torch/torch.h>

NeuralNetImpl::NeuralNetImpl(int input_size, int hidden_size, int num_classes)
        : fc1(input_size, hidden_size), fc2(hidden_size, hidden_size), fc3(hidden_size, num_classes) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor NeuralNetImpl::forward(torch::Tensor x) {
    x = torch::nn::functional::relu(fc1->forward(x));
    x = torch::nn::functional::relu(fc2->forward(x));

    return fc3->forward(x);
}