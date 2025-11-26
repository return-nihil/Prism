/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

#include <torch/torch.h>
#include <torch/script.h>

//==============================================================================
/**
*/
class TestTorchAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    TestTorchAudioProcessorEditor (TestTorchAudioProcessor&);
    ~TestTorchAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

    int main() {
        std::cout << "Inference Example (C++ with LibTorch)" << std::endl;
        std::flush(std::cout);
        try {
            // Configuration
            const std::string model_path = "C:/Users/cimil/Develop/paper-ideas/Ardan-JAES-25/Prism/PER_DOME/traced_model_8bands.pth";
            const int n_bands = 8;
            const int chunk_size = 1024;
            
            // Set device
            torch::Device device(torch::kCPU);
            if (torch::cuda::is_available()) {
                device = torch::Device(torch::kCUDA);
                std::cout << "Using CUDA" << std::endl;
            } else {
                std::cout << "Using CPU" << std::endl;
            }

            // Check if file exists
            std::ifstream f(model_path.c_str());
            if (!f.good()) {
                std::cerr << "Model file not found at: " << model_path << std::endl;
                return -1;
            } else {
                f.close();
                std::cout << "Model file found at: " << model_path << std::endl;
            }
            
            // Load the model checkpoint
            std::cout << "Loading model from: " << model_path << std::endl;
            torch::jit::script::Module model;
            try {
                // For TorchScript models
                model = torch::jit::load(model_path);
            } catch (const c10::Error& e) {
                std::cerr << "Error loading model as TorchScript." << std::endl;
                std::cerr << "Note: The .pth file needs to be converted to TorchScript format." << std::endl;
                std::cerr << "Use torch.jit.script() or torch.jit.trace() in Python." << std::endl;
                std::cerr << e.what() << std::endl;
                return -1;
            }

            try {
                model.to(device);
            } catch (const c10::Error& e) {
                std::cerr << "Error moving model to device." << std::endl;
                std::cerr << e.what() << std::endl;
                return -1;
            }

            try {
                model.eval();
            } catch (const c10::Error& e) {
                std::cerr << "Error setting model to eval mode." << std::endl;
                std::cerr << e.what() << std::endl;
                return -1;
            }
            
            std::cout << "Model loaded successfully" << std::endl;
            
            // Create dummy input tensors
            // Input shape: [batch_size, chunk_size, 1] based on Python code
            auto input_audio = torch::randn({1, chunk_size, 1}, device);

            auto state = torch::randn({1, 1, 1022}, device);
            
            // Conditioning shape: [batch_size, n_bands, cond_dim]
            auto conditioning = torch::randn({1, n_bands, 8}, device);
            
            std::cout << "Input audio shape: " << input_audio.sizes() << std::endl;
            std::cout << "Conditioning shape: " << conditioning.sizes() << std::endl;
            std::cout << "State shape: " << state.sizes() << std::endl;
            
            // Run inference
            std::cout << "\nRunning inference..." << std::endl;
            
            torch::NoGradGuard no_grad;
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_audio);
            inputs.push_back(conditioning);
            inputs.push_back(state);
            
            auto output = model.forward(inputs);
            
            // Handle output (could be a tuple or single tensor)
            torch::Tensor output_tensor;
            if (output.isTuple()) {
                auto output_tuple = output.toTuple();
                output_tensor = output_tuple->elements()[0].toTensor();
                std::cout << "Output shape: " << output_tensor.sizes() << std::endl;
                if (output_tuple->elements().size() > 1) {
                    std::cout << "Additional outputs (hidden states) also returned" << std::endl;
                    // save to state variable
                    auto returned_state = output_tuple->elements()[1].toTensor();
                    std::cout << "Returned state shape: " << returned_state.sizes() << std::endl;
                    state = returned_state;
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

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    TestTorchAudioProcessor& audioProcessor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (TestTorchAudioProcessorEditor)
};
