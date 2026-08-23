#include "rl/gcnn_model_runner.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <c10/core/InferenceMode.h>
#include <torch/cuda.h>
#include <torch/script.h>

namespace rlbranch {

struct GcnnModelRunner::Impl {
    torch::jit::script::Module module;
    torch::Device device = torch::kCPU;
};

namespace {

std::vector<torch::jit::IValue> graphInputs(
    const GraphObservation& observation,
    const torch::Device& device) {
    const auto float_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto long_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const long rows = static_cast<long>(observation.row_count);
    const long variables = static_cast<long>(observation.variable_count);
    const long edges = static_cast<long>(observation.edge_row_indices.size());
    const long candidates = static_cast<long>(observation.candidate_indices.size());

    auto row_indices = torch::from_blob(
        const_cast<std::int64_t*>(observation.edge_row_indices.data()), {edges}, long_cpu);
    auto variable_indices = torch::from_blob(
        const_cast<std::int64_t*>(observation.edge_variable_indices.data()), {edges}, long_cpu);
    auto edge_indices = torch::stack({row_indices, variable_indices}, 0).to(device);
    return {
        torch::from_blob(
            const_cast<float*>(observation.row_features.data()),
            {rows, kConstraintFeatureCount}, float_cpu).to(device),
        torch::from_blob(
            const_cast<float*>(observation.variable_features.data()),
            {variables, static_cast<long>(observation.variable_feature_dim)}, float_cpu).to(device),
        edge_indices,
        torch::from_blob(
            const_cast<float*>(observation.edge_features.data()),
            {edges, kEdgeFeatureCount}, float_cpu).to(device),
        torch::from_blob(
            const_cast<float*>(observation.global_features.data()),
            {kGlobalFeatureCount}, float_cpu).to(device),
        torch::from_blob(
            const_cast<float*>(observation.variable_categories.data()),
            {variables, kVariableCategoryCount}, float_cpu).to(device),
        torch::from_blob(
            const_cast<float*>(observation.row_categories.data()),
            {rows, kConstraintCategoryCount}, float_cpu).to(device),
        torch::from_blob(
            const_cast<std::int64_t*>(observation.candidate_indices.data()),
            {candidates}, long_cpu).to(device),
    };
}

}  // namespace

GcnnModelRunner::GcnnModelRunner(
    const std::string& model_path,
    const std::string& device_name,
    int variable_feature_dim) {
    try {
        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error("model file does not exist: " + model_path);
        }
        if (variable_feature_dim != kGraphVariableFeatureCount) {
            throw std::runtime_error(
                "official RL-GCNN is fixed at 25 variable features; "
                "got " + std::to_string(variable_feature_dim));
        }
        variable_feature_dim_ = variable_feature_dim;
        implementation_ = std::make_unique<Impl>();
        if (device_name == "cuda") {
            if (!torch::cuda::is_available()) {
                throw std::runtime_error("CUDA requested but unavailable in LibTorch");
            }
            implementation_->device = torch::Device(torch::kCUDA, 0);
        } else if (device_name != "cpu") {
            throw std::runtime_error("unsupported GCNN device: " + device_name);
        }
        implementation_->module = torch::jit::load(model_path, implementation_->device);
        implementation_->module.eval();

        GraphObservation warmup;
        warmup.row_count = 2;
        warmup.variable_count = 2;
        warmup.row_features.assign(2 * kConstraintFeatureCount, 0.0F);
        warmup.variable_feature_dim = variable_feature_dim;
        warmup.variable_features.assign(
            2 * static_cast<std::size_t>(variable_feature_dim), 0.0F);
        warmup.edge_row_indices = {0, 1};
        warmup.edge_variable_indices = {0, 1};
        warmup.edge_features.assign(2 * kEdgeFeatureCount, 0.0F);
        warmup.variable_categories.assign(2 * kVariableCategoryCount, 0.0F);
        warmup.row_categories.assign(2 * kConstraintCategoryCount, 0.0F);
        warmup.candidate_indices = {0, 1};
        c10::InferenceMode inference_guard;
        for (int iteration = 0; iteration < 3; ++iteration) {
            implementation_->module.forward(
                graphInputs(warmup, implementation_->device)).toTensor().to(torch::kCPU);
        }
    } catch (const std::exception& exception) {
        error_ = exception.what();
        implementation_.reset();
    }
}

GcnnModelRunner::~GcnnModelRunner() = default;
GcnnModelRunner::GcnnModelRunner(GcnnModelRunner&&) noexcept = default;
GcnnModelRunner& GcnnModelRunner::operator=(GcnnModelRunner&&) noexcept = default;

bool GcnnModelRunner::ready() const {
    return implementation_ != nullptr;
}

const std::string& GcnnModelRunner::error() const {
    return error_;
}

std::vector<float> GcnnModelRunner::predict(const GraphObservation& observation) {
    if (!ready()) {
        throw std::runtime_error("GCNN model is unavailable: " + error_);
    }
    if (observation.variable_feature_dim != kGraphVariableFeatureCount
        || observation.variable_feature_dim != variable_feature_dim_) {
        throw std::runtime_error("GCNN observation is not the official 25-dim variable layout");
    }
    if (observation.row_count == 0 || observation.variable_count == 0
        || observation.edge_row_indices.empty() || observation.candidate_indices.empty()) {
        throw std::invalid_argument("GCNN observation is empty");
    }
    c10::InferenceMode inference_guard;
    auto output = implementation_->module.forward(
        graphInputs(observation, implementation_->device)).toTensor().to(torch::kCPU).contiguous();
    if (output.dim() != 1
        || output.numel() != static_cast<long>(observation.candidate_indices.size())) {
        throw std::runtime_error("GCNN output does not contain one Q value per candidate");
    }
    std::vector<float> values(observation.candidate_indices.size());
    std::memcpy(values.data(), output.data_ptr<float>(), values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("GCNN output contains NaN or Inf");
        }
    }
    return values;
}

}  // namespace rlbranch
