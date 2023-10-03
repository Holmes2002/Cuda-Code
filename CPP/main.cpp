#include "inference.h"
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <cuda.h>
#include <cstdlib>

at::Tensor Encoder_Infer(std::vector<torch::jit::IValue> inputs) {
    // Path to the TorchScript model file
    std::string modelPath_encoder = "model_1_encoder.pt";
    // Load the model
    torch::jit::script::Module encoder;
    try {
        encoder = torch::jit::load(modelPath_encoder);
        encoder.to(torch::kCUDA);
        printf("  - Load Model Complete");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
    };
    try {
        at::Tensor output = encoder.forward(inputs).toTensor();
        return output;

    } catch (const c10::Error& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
    }

}

int main(){
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 384, 384}).to(torch::kCUDA));
    at::Tensor output_encoder = Encoder_Infer(inputs);

    torch::Tensor tgt_inp = torch::full({1, 1}, 1, torch::kInt64).to(torch::kCUDA);
    std::cout << "Shape of TGT+: " << tgt_inp.sizes() << std::endl;
    std::cout << "Shape of OUT_ENCODER+: " << output_encoder.sizes() << std::endl;
    inference_cpp(output_encoder);

}