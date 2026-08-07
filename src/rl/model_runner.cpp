#include "rl/model_runner.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <c10/core/InferenceMode.h>
#include <torch/script.h>
#include <torch/cuda.h>

namespace rlbranch {

struct ModelRunner::Impl {
    torch::jit::script::Module module;
    torch::Device device = torch::kCPU;
};

ModelRunner::ModelRunner(
    const std::string& model_path,
    const std::string& device_name)
    : device_name_(device_name) {
    try {
        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error("model file does not exist: " + model_path);
        }
        implementation_ = std::make_unique<Impl>();
        if (device_name == "cuda") {
            if (!torch::cuda::is_available()) {
                throw std::runtime_error("CUDA requested but unavailable in LibTorch");
            }
            implementation_->device = torch::Device(torch::kCUDA, 0);
        } else if (device_name == "cpu") {
            implementation_->device = torch::Device(torch::kCPU);
        } else {
            throw std::runtime_error("unsupported RL device: " + device_name);
        }
        implementation_->module = torch::jit::load(model_path, implementation_->device);
        implementation_->module.eval();
        c10::InferenceMode inference_guard;
        const auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(implementation_->device);
        auto variables = torch::zeros({64, kCandidateVariableFeatureCount}, options);
        auto globals = torch::zeros({64, kGlobalFeatureCount}, options);
        auto categories = torch::zeros({64, kVariableCategoryCount}, options);
        for (int iteration = 0; iteration < 8; ++iteration) {
            implementation_->module.forward({variables, globals, categories})
                .toTensor().to(torch::kCPU);
        }
    } catch (const std::exception& exception) {
        error_ = exception.what();
        implementation_.reset();
    }
}

ModelRunner::~ModelRunner() = default;
ModelRunner::ModelRunner(ModelRunner&&) noexcept = default;
ModelRunner& ModelRunner::operator=(ModelRunner&&) noexcept = default;

bool ModelRunner::ready() const {
    return implementation_ != nullptr;
}

const std::string& ModelRunner::error() const {
    return error_;
}

const std::string& ModelRunner::deviceName() const {
    return device_name_;
}

std::vector<float> ModelRunner::predict(
    const std::vector<float>& variable_features,
    const std::array<float, kGlobalFeatureCount>& global_features,
    const std::vector<float>& category_features,
    std::size_t candidate_count) {
    if (!ready()) {
        throw std::runtime_error("model is unavailable: " + error_);
    }
    if (candidate_count == 0
        || variable_features.size() != candidate_count * kCandidateVariableFeatureCount
        || category_features.size() != candidate_count * kVariableCategoryCount) {
        throw std::invalid_argument("model input dimensions do not match the feature schema");
    }

    c10::InferenceMode inference_guard;
    const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto variables = torch::from_blob(
        const_cast<float*>(variable_features.data()),
        {static_cast<long>(candidate_count), kCandidateVariableFeatureCount},
        cpu_options).to(implementation_->device);
    auto globals = torch::from_blob(
        const_cast<float*>(global_features.data()),
        {1, kGlobalFeatureCount},
        cpu_options).to(implementation_->device).expand(
            {static_cast<long>(candidate_count), kGlobalFeatureCount});
    auto categories = torch::from_blob(
        const_cast<float*>(category_features.data()),
        {static_cast<long>(candidate_count), kVariableCategoryCount},
        cpu_options).to(implementation_->device);

    auto output = implementation_->module.forward({variables, globals, categories})
        .toTensor().to(torch::kCPU).contiguous();
    if (output.dim() != 1 || output.numel() != static_cast<long>(candidate_count)) {
        throw std::runtime_error("model output does not contain one Q value per candidate");
    }

    std::vector<float> q_values(candidate_count);
    std::memcpy(q_values.data(), output.data_ptr<float>(), candidate_count * sizeof(float));
    for (float value : q_values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("model output contains NaN or Inf");
        }
    }
    return q_values;
}

}  // namespace rlbranch
