#ifndef mykernel
#define mykernel
#include <torch/script.h>
extern "C" void inference_cpp(at::Tensor output_encoder);
#endif

