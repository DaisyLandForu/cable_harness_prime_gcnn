#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rl/scip_feature_extractor.hpp"

namespace rlbranch {

class ModelRunner {
public:
    ModelRunner(const std::string& model_path, const std::string& device_name);
    ~ModelRunner();

    ModelRunner(ModelRunner&&) noexcept;
    ModelRunner& operator=(ModelRunner&&) noexcept;
    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    bool ready() const;
    const std::string& error() const;
    const std::string& deviceName() const;

    std::vector<float> predict(
        const std::vector<float>& variable_features,
        const std::array<float, kGlobalFeatureCount>& global_features,
        const std::vector<float>& category_features,
        std::size_t candidate_count);

private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
    std::string error_;
    std::string device_name_;
};

}  // namespace rlbranch
