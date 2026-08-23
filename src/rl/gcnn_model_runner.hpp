#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rl/scip_graph_feature_extractor.hpp"

namespace rlbranch {

class GcnnModelRunner {
public:
    GcnnModelRunner(
        const std::string& model_path,
        const std::string& device_name,
        int variable_feature_dim = kCandidateVariableFeatureCount);
    ~GcnnModelRunner();

    GcnnModelRunner(GcnnModelRunner&&) noexcept;
    GcnnModelRunner& operator=(GcnnModelRunner&&) noexcept;
    GcnnModelRunner(const GcnnModelRunner&) = delete;
    GcnnModelRunner& operator=(const GcnnModelRunner&) = delete;

    bool ready() const;
    const std::string& error() const;

    std::vector<float> predict(const GraphObservation& observation);

private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
    std::string error_;
};

}  // namespace rlbranch
