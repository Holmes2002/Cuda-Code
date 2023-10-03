#include <cuda_runtime.h>
#include <torch/script.h>
__global__ void translateKernel(int* translated_sentence, int* translated_sentence_device) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

}
extern "C" void inference_cpp(at::Tensor output_encoder) {
    std::string modelPath_decoder = "model_decoder.pt";
    torch::jit::script::Module decoder = torch::jit::load(modelPath_decoder);
    decoder.to(torch::kCUDA);
    int sos_token = 1;  // Define the value of sos_token
    int eos_token = 2;  // Define the value of eos_token
    int max_seq_length = 256;  // Define the maximum sequence length
    int block_size = 256;  // Define the CUDA block size
    int grid_size = (max_seq_length + block_size - 1) / block_size;  // Calculate the CUDA grid size

    // Allocate memory on the host
    int* translated_sentence_host = (int*)malloc(max_seq_length * sizeof(int));
    // int* indices_host = (int*)malloc(max_seq_length * sizeof(int));
    // float* values_host = (float*)malloc(max_seq_length * sizeof(float));
    int max_len = 1;

    // Allocate memory on the device
    int* translated_sentence_device;
    cudaMalloc((void**)&translated_sentence_device, max_seq_length * sizeof(int));   
    cudaMemcpy(translated_sentence_device, translated_sentence_host, max_seq_length * sizeof(int), cudaMemcpyHostToDevice);
    // torch::Tensor tgt_inp = torch::full({1, 1}, 1, torch::kInt64).to(torch::kCUDA);
    torch::Tensor tgt_inp = torch::tensor({{1}}, torch::kInt32).to(torch::kCUDA);
   
    
    std::vector<c10::IValue> decoder_output;
    int check = 0 ;

    check = max_len;
    torch::Tensor indices, values;
    torch::Tensor softmaxOutput;
    while (max_len<max_seq_length){
    max_len++;
    std::vector<torch::jit::IValue> decoder_inputs;
    decoder_inputs.push_back(tgt_inp);
    decoder_inputs.push_back(output_encoder);
    decoder_output = decoder.forward(decoder_inputs).toTuple()->elements();
    torch::Tensor output_decoder = decoder_output[0].toTensor();
    softmaxOutput = torch::softmax(output_decoder, 1);

    std::tie(values, indices) = torch::topk(softmaxOutput, 5);
    int value_indi = indices[0][-1][0].item<int>();

    torch::Tensor tgt_inp_2 = torch::tensor({{value_indi}}, torch::kInt32).to(torch::kCUDA);
    tgt_inp = torch::cat({tgt_inp, tgt_inp_2 }, 0);
    std::cout <<"LIST " << tgt_inp << std::endl;
	}
}