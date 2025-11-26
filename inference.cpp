#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>

int main() {
    std::cout << "PER-DOME Inference Example (C++ with LibTorch)" << std::endl;
    std::flush(std::cout);
    try {
        // Configuration
        const std::string model_path = "./PER_DOME/model_8bands.pth";
        const int n_bands = 8;
        const int chunk_size = 2048;
        const int cond_dim = 128;
        
        // Set device
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            std::cout << "Using CUDA" << std::endl;
        } else {
            std::cout << "Using CPU" << std::endl;
        }
        
        // Load the model checkpoint
        std::cout << "Loading model from: " << model_path << std::endl;
        torch::jit::script::Module model;
        try {
            // For TorchScript models
            model = torch::jit::load(model_path);
            model.to(device);
            model.eval();
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model as TorchScript." << std::endl;
            std::cerr << "Note: The .pth file needs to be converted to TorchScript format." << std::endl;
            std::cerr << "Use torch.jit.script() or torch.jit.trace() in Python." << std::endl;
            return -1;
        }
        
        std::cout << "Model loaded successfully" << std::endl;
        
        // Create dummy input tensors
        // Input shape: [batch_size, chunk_size, 1] based on Python code
        auto input_audio = torch::randn({1, chunk_size, 1}, device);
        
        // Conditioning shape: [batch_size, n_bands, cond_dim]
        auto conditioning = torch::randn({1, n_bands, cond_dim}, device);
        
        std::cout << "Input audio shape: " << input_audio.sizes() << std::endl;
        std::cout << "Conditioning shape: " << conditioning.sizes() << std::endl;
        
        // Run inference
        std::cout << "\nRunning inference..." << std::endl;
        
        torch::NoGradGuard no_grad;
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_audio);
        inputs.push_back(conditioning);
        
        auto output = model.forward(inputs);
        
        // Handle output (could be a tuple or single tensor)
        torch::Tensor output_tensor;
        if (output.isTuple()) {
            auto output_tuple = output.toTuple();
            output_tensor = output_tuple->elements()[0].toTensor();
            std::cout << "Output shape: " << output_tensor.sizes() << std::endl;
            if (output_tuple->elements().size() > 1) {
                std::cout << "Additional outputs (hidden states) also returned" << std::endl;
            }
        } else {
            output_tensor = output.toTensor();
            std::cout << "Output shape: " << output_tensor.sizes() << std::endl;
        }
        
        std::cout << "\nInference successful!" << std::endl;
        std::cout << "Output min: " << output_tensor.min().item<float>() << std::endl;
        std::cout << "Output max: " << output_tensor.max().item<float>() << std::endl;
        std::cout << "Output mean: " << output_tensor.mean().item<float>() << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}


// ============================================================================
// COMPILATION INSTRUCTIONS
// ============================================================================

// WINDOWS with MSVC (Visual Studio):
// ----------------------------------
// cl.exe /std:c++17 /EHsc /MD prism_inference.cpp ^
//     /I"C:\path\to\libtorch\include" ^
//     /I"C:\path\to\libtorch\include\torch\csrc\api\include" ^
//     /link /LIBPATH:"C:\path\to\libtorch\lib" ^
//     torch.lib torch_cpu.lib c10.lib
//
// For CUDA support, also add: torch_cuda.lib c10_cuda.lib
//
// Make sure libtorch\lib is in your PATH or copy the DLLs to your exe directory

// WINDOWS with Clang:
// ------------------
// clang++ -std=c++17 prism_inference.cpp -o prism_inference.exe ^
//     -IC:\path\to\libtorch\include ^
//     -IC:\path\to\libtorch\include\torch\csrc\api\include ^
//     -LC:\path\to\libtorch\lib ^
//     -ltorch -ltorch_cpu -lc10 ^
//     -Wl,-rpath,C:\path\to\libtorch\lib
//
// Or with MSVC-compatible flags:
// clang-cl /std:c++17 /EHsc /MD prism_inference.cpp ^
//     /I"C:\path\to\libtorch\include" ^
//     /I"C:\path\to\libtorch\include\torch\csrc\api\include" ^
//     /link /LIBPATH:"C:\path\to\libtorch\lib" ^
//     torch.lib torch_cpu.lib c10.lib

// LINUX/Mac with g++:
// ------------------
// g++ -std=c++17 prism_inference.cpp -o prism_inference \
//     -I/path/to/libtorch/include \
//     -I/path/to/libtorch/include/torch/csrc/api/include \
//     -L/path/to/libtorch/lib \
//     -ltorch -ltorch_cpu -lc10 \
//     -Wl,-rpath,/path/to/libtorch/lib