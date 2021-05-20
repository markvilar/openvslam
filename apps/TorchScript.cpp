#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <torch/script.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) 
{
    std::string modelPath = 
        "/home/martin/dev/openvslam/configurations/models/UIENet-Script.pt";

    torch::jit::script::Module module;
    try 
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 1080, 1920}));

    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    std::cout << "ok\n";
}
