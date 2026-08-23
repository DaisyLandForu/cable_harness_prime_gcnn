#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <scip/scip.h>

#include "rl/scip_feature_extractor.hpp"

namespace rlbranch {

constexpr int kConstraintFeatureCount = 14;
constexpr int kEdgeFeatureCount = 3;
constexpr int kConstraintCategoryCount = 6;
// Phase B: ECOLE(19) + Prim neighborhood flags(6).
constexpr int kPrimVariableFeatureCount = 6;
constexpr int kGraphVariableFeatureCount =
    kCandidateVariableFeatureCount + kPrimVariableFeatureCount;

struct GraphObservation {
    std::vector<BranchCandidate> candidates;
    std::vector<float> row_features;
    std::vector<float> variable_features;
    std::vector<std::int64_t> edge_row_indices;
    std::vector<std::int64_t> edge_variable_indices;
    std::vector<float> edge_features;
    std::array<float, kGlobalFeatureCount> global_features{};
    std::vector<float> variable_categories;
    std::vector<float> row_categories;
    std::vector<std::int64_t> candidate_indices;
    std::vector<std::string> candidate_names;
    std::vector<std::string> variable_names;
    std::vector<float> local_lower_bounds;
    std::size_t row_count = 0;
    std::size_t variable_count = 0;
    int variable_feature_dim = kGraphVariableFeatureCount;
};

int aviationConstraintCategory(const std::string& row_name);

SCIP_RETCODE extractGraphObservation(
    SCIP* scip,
    GraphObservation& observation,
    bool twohop = true);

}  // namespace rlbranch
