#include "rl/scip_feature_extractor.hpp"

#include <algorithm>
#include <cmath>

#include <scip/pub_lp.h>
#include <scip/pub_var.h>
#include <scip/scip_branch.h>
#include <scip/scip_solvingstats.h>
#include <scip/scip_tree.h>

namespace rlbranch {
namespace {

constexpr float kAgeOffset = 5.0F;

float finiteOrZero(double value) {
    return std::isfinite(value) && std::abs(value) < 1e20
        ? static_cast<float>(value)
        : 0.0F;
}

void setBoundFeatures(SCIP* scip, SCIP_COL* column, float* features) {
    const SCIP_Real lower_bound = SCIPcolGetLb(column);
    const SCIP_Real upper_bound = SCIPcolGetUb(column);
    const bool has_lower_bound = !SCIPisInfinity(scip, std::abs(lower_bound));
    const bool has_upper_bound = !SCIPisInfinity(scip, std::abs(upper_bound));
    features[5] = has_lower_bound ? 1.0F : 0.0F;
    features[6] = has_upper_bound ? 1.0F : 0.0F;
    features[10] = has_lower_bound && SCIPisEQ(scip, SCIPcolGetPrimsol(column), lower_bound)
        ? 1.0F
        : 0.0F;
    features[11] = has_upper_bound && SCIPisEQ(scip, SCIPcolGetPrimsol(column), upper_bound)
        ? 1.0F
        : 0.0F;
}

void setVariableTypeFeatures(SCIP_VAR* variable, float* features) {
    switch (SCIPvarGetType(variable)) {
    case SCIP_VARTYPE_BINARY: features[1] = 1.0F; break;
    case SCIP_VARTYPE_INTEGER: features[2] = 1.0F; break;
    case SCIP_VARTYPE_IMPLINT: features[3] = 1.0F; break;
    case SCIP_VARTYPE_CONTINUOUS: features[4] = 1.0F; break;
    }
}

void setBasisFeatures(SCIP_COL* column, float* features) {
    switch (SCIPcolGetBasisStatus(column)) {
    case SCIP_BASESTAT_LOWER: features[15] = 1.0F; break;
    case SCIP_BASESTAT_BASIC: features[16] = 1.0F; break;
    case SCIP_BASESTAT_UPPER: features[17] = 1.0F; break;
    case SCIP_BASESTAT_ZERO: features[18] = 1.0F; break;
    }
}

void setGlobalBound(float* features, int value_index, int finite_index, double value) {
    const bool finite = std::isfinite(value) && std::abs(value) < 1e20;
    features[value_index] = finite ? static_cast<float>(value) : 0.0F;
    features[finite_index] = finite ? 1.0F : 0.0F;
}

}  // namespace

SCIP_RETCODE extractLpBranchCandidates(
    SCIP* scip,
    std::vector<BranchCandidate>& candidates) {
    candidates.clear();

    SCIP_VAR** lp_candidates = nullptr;
    SCIP_Real* lp_values = nullptr;
    SCIP_Real* fractionalities = nullptr;
    int candidate_count = 0;
    int priority_candidate_count = 0;

    SCIP_CALL(SCIPgetLPBranchCands(
        scip,
        &lp_candidates,
        &lp_values,
        &fractionalities,
        &candidate_count,
        &priority_candidate_count,
        nullptr));

    if (lp_candidates == nullptr || priority_candidate_count <= 0) {
        return SCIP_OKAY;
    }

    candidates.reserve(static_cast<std::size_t>(priority_candidate_count));
    for (int index = 0; index < priority_candidate_count; ++index) {
        BranchCandidate candidate;
        candidate.variable = lp_candidates[index];
        candidate.lp_value = lp_values[index];
        candidate.fractionality = fractionalities[index];
        candidate.candidate_index = index;
        candidate.variable_index = SCIPvarGetProbindex(lp_candidates[index]);
        candidates.push_back(candidate);
    }

    return SCIP_OKAY;
}

int aviationVariableCategory(const std::string& variable_name) {
    std::string name = variable_name;
    while (name.rfind("t_", 0) == 0) {
        name.erase(0, 2);
    }
    static const std::array<std::string, 5> categories = {"m", "z", "y", "absf", "f"};
    for (int index = 0; index < static_cast<int>(categories.size()); ++index) {
        if (name == categories[index] || name.rfind(categories[index] + "_", 0) == 0) {
            return index;
        }
    }
    return kVariableCategoryCount - 1;
}

VariableFeatureContext makeVariableFeatureContext(SCIP* scip) {
    VariableFeatureContext context;
    const SCIP_Real raw_objective_norm = SCIPgetObjNorm(scip);
    context.objective_norm = raw_objective_norm > 0.0 ? raw_objective_norm : 1.0;
    context.number_of_lps = static_cast<float>(SCIPgetNLPs(scip));
    context.best_solution = SCIPgetBestSol(scip);
    return context;
}

SCIP_RETCODE extractVariableFeatures(
    SCIP* scip,
    SCIP_VAR* variable,
    const VariableFeatureContext& context,
    float* features) {
    if (variable == nullptr || features == nullptr) {
        return SCIP_INVALIDDATA;
    }
    SCIP_COL* column = SCIPvarGetCol(variable);
    if (column == nullptr) {
        return SCIP_INVALIDDATA;
    }
    std::fill(features, features + kCandidateVariableFeatureCount, 0.0F);
    features[0] = finiteOrZero(SCIPvarGetObj(variable) / context.objective_norm);
    setVariableTypeFeatures(variable, features);
    setBoundFeatures(scip, column, features);
    features[7] = finiteOrZero(SCIPgetVarRedcost(scip, variable) / context.objective_norm);
    features[8] = finiteOrZero(SCIPvarGetLPSol(variable));
    features[9] = SCIPvarGetType(variable) == SCIP_VARTYPE_CONTINUOUS
        ? 0.0F
        : finiteOrZero(SCIPfeasFrac(scip, SCIPvarGetLPSol(variable)));
    features[12] = static_cast<float>(SCIPcolGetAge(column))
        / (context.number_of_lps + kAgeOffset);
    features[13] = context.best_solution == nullptr
        ? 0.0F
        : finiteOrZero(SCIPgetSolVal(scip, context.best_solution, variable));
    features[14] = context.best_solution == nullptr
        ? 0.0F
        : finiteOrZero(SCIPvarGetAvgSol(variable));
    setBasisFeatures(column, features);
    return SCIP_OKAY;
}

void extractGlobalFeatures(
    SCIP* scip,
    std::array<float, kGlobalFeatureCount>& global) {
    global.fill(0.0F);
    global[0] = static_cast<float>(SCIPgetDepth(scip));
    global[1] = static_cast<float>(SCIPgetNNodes(scip));
    global[2] = static_cast<float>(SCIPgetNTotalNodes(scip));
    global[3] = static_cast<float>(SCIPgetNLeaves(scip));
    global[4] = static_cast<float>(SCIPgetNFeasibleLeaves(scip));
    global[5] = static_cast<float>(SCIPgetNInfeasibleLeaves(scip));
    global[6] = static_cast<float>(SCIPgetNLPIterations(scip));
    setGlobalBound(global.data(), 7, 8, SCIPgetPrimalbound(scip));
    setGlobalBound(global.data(), 9, 10, SCIPgetDualbound(scip));
    setGlobalBound(global.data(), 11, 12, SCIPgetGap(scip));
    global[13] = static_cast<float>(SCIPgetNSols(scip));
}

SCIP_RETCODE extractCandidateObservation(
    SCIP* scip,
    CandidateObservation& observation) {
    observation = CandidateObservation{};
    SCIP_CALL(extractLpBranchCandidates(scip, observation.candidates));
    if (observation.candidates.empty()) {
        return SCIP_OKAY;
    }

    const std::size_t candidate_count = observation.candidates.size();
    observation.variable_features.assign(
        candidate_count * kCandidateVariableFeatureCount,
        0.0F);
    observation.category_features.assign(candidate_count * kVariableCategoryCount, 0.0F);
    observation.variable_names.reserve(candidate_count);

    const VariableFeatureContext context = makeVariableFeatureContext(scip);

    for (std::size_t index = 0; index < candidate_count; ++index) {
        SCIP_VAR* variable = observation.candidates[index].variable;
        if (variable == nullptr) {
            return SCIP_INVALIDDATA;
        }
        float* features = observation.variable_features.data()
            + index * kCandidateVariableFeatureCount;
        SCIP_CALL(extractVariableFeatures(scip, variable, context, features));

        const std::string name = SCIPvarGetName(variable);
        observation.variable_names.push_back(name);
        const int category = aviationVariableCategory(name);
        observation.category_features[index * kVariableCategoryCount + category] = 1.0F;
    }

    extractGlobalFeatures(scip, observation.global_features);
    return SCIP_OKAY;
}

}  // namespace rlbranch
