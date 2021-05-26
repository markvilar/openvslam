#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <torch/script.h>

#include <chrono>
#include <iostream>
#include <memory>

int main() 
{
    std::string modelPath = 
        "/home/martin/dev/openvslam/configurations/models/UIENet-CUDA-Script.pt";

    torch::jit::script::Module module;
    try 
    {
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "error loading the model\n";
        return -1;
    }


    std::vector<torch::jit::IValue> inputs;
    auto tensorOptions = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA, 0)
        .requires_grad(false);
    try
    {
        inputs.push_back(torch::ones({1, 3, 1080, 1920}, tensorOptions));
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "error allocating tensor\n";
        std::cerr << e.what();
        return -1;
    }
    torch::Tensor output = module.forward(inputs).toTensor();
    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    std::cout << output.dtype() << std::endl;
}
