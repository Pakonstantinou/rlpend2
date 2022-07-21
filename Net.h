#pragma once

#include <torch/torch.h>

class NeuralNetImpl : public torch::nn::Module {
public:
    NeuralNetImpl(int input_size, int hidden_size, int num_classes);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};

TORCH_MODULE(NeuralNet);
