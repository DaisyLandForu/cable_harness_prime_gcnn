#include "rl/scip_graph_feature_extractor.hpp"

#include <algorithm>
#include <cmath>

#include <scip/pub_lp.h>
#include <scip/pub_var.h>
#include <scip/scip_lp.h>
#include <scip/scip_solvingstats.h>

namespace rlbranch {
namespace {

constexpr float kAgeOffset = 5.0F;

float finiteFloat(double value) {
    return std::isfinite(value) && std::abs(value) < 1e20
        ? static_cast<float>(value)
        : 0.0F;
}

void appendRowCategory(std::vector<float>& categories, int category) {
    const std::size_t start = categories.size();
    categories.resize(start + kConstraintCategoryCount, 0.0F);
    categories[start + static_cast<std::size_t>(category)] = 1.0F;
}

void appendExpandedRow(
    SCIP* scip,
    SCIP_ROW* row,
    double lhs,
    bool lhs_finite,
    double rhs,
    bool rhs_finite,
    double norm,
    double activity,
    double objective_cosine,
    double side_sign,
    std::size_t row_index,
    GraphObservation& observation) {
    const double bound = side_sign < 0.0 ? lhs : rhs;
    const double normalized_bias = side_sign * bound / norm;
    const double raw_objective_norm = SCIPgetObjNorm(scip);
    const double objective_norm = raw_objective_norm > 0.0 ? raw_objective_norm : 1.0;
    const double dual = side_sign * SCIProwGetDualsol(row) / (norm * objective_norm);
    const bool tight = SCIPisFeasEQ(scip, activity, bound);
    const bool equality = lhs_finite && rhs_finite && SCIPisEQ(scip, lhs, rhs);
    const double slack = side_sign * (bound - activity);

    const std::array<float, kConstraintFeatureCount> features = {
        finiteFloat(normalized_bias),
        finiteFloat(side_sign * objective_cosine),
        tight ? 1.0F : 0.0F,
        finiteFloat(dual),
        static_cast<float>(SCIProwGetAge(row))
            / (static_cast<float>(SCIPgetNLPs(scip)) + kAgeOffset),
        finiteFloat(lhs),
        lhs_finite ? 1.0F : 0.0F,
        finiteFloat(rhs),
        rhs_finite ? 1.0F : 0.0F,
        finiteFloat(activity),
        finiteFloat(slack),
        equality ? 1.0F : 0.0F,
        side_sign < 0.0 ? 1.0F : 0.0F,
        side_sign > 0.0 ? 1.0F : 0.0F,
    };
    observation.row_features.insert(
        observation.row_features.end(), features.begin(), features.end());
    appendRowCategory(
        observation.row_categories,
        aviationConstraintCategory(SCIProwGetName(row)));

    SCIP_COL** columns = SCIProwGetCols(row);
    SCIP_Real* values = SCIProwGetVals(row);
    const int nonzeros = SCIProwGetNLPNonz(row);
    for (int index = 0; index < nonzeros; ++index) {
        SCIP_VAR* variable = SCIPcolGetVar(columns[index]);
        const int variable_index = SCIPvarGetProbindex(variable);
        const double coefficient = side_sign * values[index];
        observation.edge_row_indices.push_back(static_cast<std::int64_t>(row_index));
        observation.edge_variable_indices.push_back(variable_index);
        observation.edge_features.push_back(finiteFloat(coefficient));
        observation.edge_features.push_back(finiteFloat(coefficient / norm));
        observation.edge_features.push_back(
            coefficient > 0.0 ? 1.0F : (coefficient < 0.0 ? -1.0F : 0.0F));
    }
}

}  // namespace

int aviationConstraintCategory(const std::string& row_name) {
    std::string name = row_name;
    while (name.rfind("t_", 0) == 0) {
        name.erase(0, 2);
    }
    if (name.rfind("flow_", 0) == 0 || name.rfind("fforbid", 0) == 0) {
        return 0;
    }
    if (name.rfind("abs", 0) == 0) {
        return 1;
    }
    if (name.rfind("topo_", 0) == 0 || name.rfind("only_father", 0) == 0) {
        return 2;
    }
    if (name.rfind("zlower", 0) == 0 || name.rfind("onlym", 0) == 0) {
        return 3;
    }
    if (name.rfind("imbalance", 0) == 0) {
        return 4;
    }
    return kConstraintCategoryCount - 1;
}

SCIP_RETCODE extractGraphObservation(SCIP* scip, GraphObservation& observation) {
    observation = GraphObservation{};
    SCIP_CALL(extractLpBranchCandidates(scip, observation.candidates));
    if (observation.candidates.empty()) {
        return SCIP_OKAY;
    }

    SCIP_VAR** variables = SCIPgetVars(scip);
    const int variable_count = SCIPgetNVars(scip);
    if (variables == nullptr || variable_count <= 0) {
        return SCIP_INVALIDDATA;
    }
    observation.variable_count = static_cast<std::size_t>(variable_count);
    observation.variable_features.assign(
        observation.variable_count * kCandidateVariableFeatureCount, 0.0F);
    observation.variable_categories.assign(
        observation.variable_count * kVariableCategoryCount, 0.0F);
    const VariableFeatureContext context = makeVariableFeatureContext(scip);
    for (int position = 0; position < variable_count; ++position) {
        SCIP_VAR* variable = variables[position];
        const int variable_index = SCIPvarGetProbindex(variable);
        if (variable_index < 0 || variable_index >= variable_count) {
            return SCIP_INVALIDDATA;
        }
        SCIP_CALL(extractVariableFeatures(
            scip,
            variable,
            context,
            observation.variable_features.data()
                + static_cast<std::size_t>(variable_index) * kCandidateVariableFeatureCount));
        const int category = aviationVariableCategory(SCIPvarGetName(variable));
        observation.variable_categories[
            static_cast<std::size_t>(variable_index) * kVariableCategoryCount + category] = 1.0F;
    }

    observation.candidate_indices.reserve(observation.candidates.size());
    observation.candidate_names.reserve(observation.candidates.size());
    for (const BranchCandidate& candidate : observation.candidates) {
        if (candidate.variable_index < 0 || candidate.variable_index >= variable_count) {
            return SCIP_INVALIDDATA;
        }
        observation.candidate_indices.push_back(candidate.variable_index);
        observation.candidate_names.emplace_back(SCIPvarGetName(candidate.variable));
    }

    SCIP_ROW** rows = SCIPgetLPRows(scip);
    const int row_count = SCIPgetNLPRows(scip);
    if (rows == nullptr || row_count <= 0) {
        return SCIP_INVALIDDATA;
    }
    std::size_t expanded_row_index = 0;
    const double raw_objective_norm = SCIPgetObjNorm(scip);
    const double objective_norm = raw_objective_norm > 0.0 ? raw_objective_norm : 1.0;
    for (int row_position = 0; row_position < row_count; ++row_position) {
        SCIP_ROW* row = rows[row_position];
        const double raw_norm = SCIProwGetNorm(row);
        const double norm = raw_norm > 0.0 ? raw_norm : 1.0;
        const double constant = SCIProwGetConstant(row);
        const double lhs = SCIProwGetLhs(row) - constant;
        const double rhs = SCIProwGetRhs(row) - constant;
        const bool lhs_finite = !SCIPisInfinity(scip, std::abs(lhs));
        const bool rhs_finite = !SCIPisInfinity(scip, std::abs(rhs));
        SCIP_COL** columns = SCIProwGetCols(row);
        SCIP_Real* values = SCIProwGetVals(row);
        const int nonzeros = SCIProwGetNLPNonz(row);
        double activity = 0.0;
        double objective_product = 0.0;
        for (int index = 0; index < nonzeros; ++index) {
            SCIP_VAR* variable = SCIPcolGetVar(columns[index]);
            activity += values[index] * SCIPvarGetLPSol(variable);
            objective_product += values[index] * SCIPvarGetObj(variable);
        }
        const double objective_cosine = norm * objective_norm > 0.0
            ? objective_product / (norm * objective_norm)
            : 0.0;
        if (lhs_finite) {
            appendExpandedRow(
                scip, row, lhs, lhs_finite, rhs, rhs_finite, norm, activity,
                objective_cosine, -1.0, expanded_row_index++, observation);
        }
        if (rhs_finite) {
            appendExpandedRow(
                scip, row, lhs, lhs_finite, rhs, rhs_finite, norm, activity,
                objective_cosine, 1.0, expanded_row_index++, observation);
        }
    }
    observation.row_count = expanded_row_index;
    extractGlobalFeatures(scip, observation.global_features);
    if (observation.row_features.size()
            != observation.row_count * kConstraintFeatureCount
        || observation.row_categories.size()
            != observation.row_count * kConstraintCategoryCount
        || observation.edge_row_indices.size() != observation.edge_variable_indices.size()
        || observation.edge_features.size()
            != observation.edge_row_indices.size() * kEdgeFeatureCount) {
        return SCIP_INVALIDDATA;
    }
    return SCIP_OKAY;
}

}  // namespace rlbranch
