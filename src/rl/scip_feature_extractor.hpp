#pragma once

#include <array>
#include <string>
#include <vector>

#include <scip/scip.h>

namespace rlbranch {

constexpr int kCandidateVariableFeatureCount = 19;
constexpr int kGlobalFeatureCount = 14;
constexpr int kVariableCategoryCount = 6;

struct BranchCandidate {
    SCIP_VAR* variable = nullptr;
    SCIP_Real lp_value = 0.0;
    SCIP_Real fractionality = 0.0;
    int candidate_index = -1;
    int variable_index = -1;
};

struct CandidateObservation {
    std::vector<BranchCandidate> candidates;
    std::vector<float> variable_features;
    std::array<float, kGlobalFeatureCount> global_features{};
    std::vector<float> category_features;
    std::vector<std::string> variable_names;
};

struct VariableFeatureContext {
    SCIP_Real objective_norm = 1.0;
    float number_of_lps = 0.0F;
    SCIP_SOL* best_solution = nullptr;
};

SCIP_RETCODE extractLpBranchCandidates(
    SCIP* scip,
    std::vector<BranchCandidate>& candidates);

int aviationVariableCategory(const std::string& variable_name);

VariableFeatureContext makeVariableFeatureContext(SCIP* scip);

SCIP_RETCODE extractVariableFeatures(
    SCIP* scip,
    SCIP_VAR* variable,
    const VariableFeatureContext& context,
    float* features);

void extractGlobalFeatures(
    SCIP* scip,
    std::array<float, kGlobalFeatureCount>& features);

SCIP_RETCODE extractCandidateObservation(
    SCIP* scip,
    CandidateObservation& observation);

}  // namespace rlbranch
