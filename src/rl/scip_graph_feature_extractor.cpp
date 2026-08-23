#include "rl/scip_graph_feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include <scip/pub_lp.h>
#include <scip/pub_var.h>
#include <scip/scip_lp.h>
#include <scip/scip_numerics.h>
#include <scip/scip_solvingstats.h>
#include <scip/scip_var.h>

#include "rl/prim_bias.hpp"

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

SCIP_RETCODE extractGraphObservation(
    SCIP* scip,
    GraphObservation& observation,
    bool twohop) {
    observation = GraphObservation{};
    SCIP_CALL(extractLpBranchCandidates(scip, observation.candidates));
    if (observation.candidates.empty()) {
        return SCIP_OKAY;
    }

    SCIP_VAR** variables = SCIPgetVars(scip);
    const int variable_count = SCIPgetNVars(scip);
    SCIP_ROW** rows = SCIPgetLPRows(scip);
    const int row_count = SCIPgetNLPRows(scip);
    if (variables == nullptr || variable_count <= 0 || rows == nullptr || row_count <= 0) {
        return SCIP_INVALIDDATA;
    }

    std::vector<char> keep_variable(static_cast<std::size_t>(variable_count), twohop ? 0 : 1);
    std::vector<char> keep_lp_row(static_cast<std::size_t>(row_count), twohop ? 0 : 1);
    std::unordered_set<int> candidate_set;
    for (const BranchCandidate& candidate : observation.candidates) {
        if (candidate.variable_index < 0 || candidate.variable_index >= variable_count) {
            return SCIP_INVALIDDATA;
        }
        candidate_set.insert(candidate.variable_index);
        keep_variable[static_cast<std::size_t>(candidate.variable_index)] = 1;
    }
    if (twohop) {
        for (int row_position = 0; row_position < row_count; ++row_position) {
            SCIP_COL** columns = SCIProwGetCols(rows[row_position]);
            const int nonzeros = SCIProwGetNLPNonz(rows[row_position]);
            bool keep = false;
            for (int index = 0; index < nonzeros; ++index) {
                const int variable_index = SCIPvarGetProbindex(SCIPcolGetVar(columns[index]));
                if (candidate_set.count(variable_index) > 0) {
                    keep = true;
                    break;
                }
            }
            if (!keep) {
                continue;
            }
            keep_lp_row[static_cast<std::size_t>(row_position)] = 1;
            for (int index = 0; index < nonzeros; ++index) {
                const int variable_index = SCIPvarGetProbindex(SCIPcolGetVar(columns[index]));
                if (variable_index >= 0 && variable_index < variable_count) {
                    keep_variable[static_cast<std::size_t>(variable_index)] = 1;
                }
            }
        }
    }

    std::vector<int> old_to_new(static_cast<std::size_t>(variable_count), -1);
    int compact_count = 0;
    for (int variable_index = 0; variable_index < variable_count; ++variable_index) {
        if (keep_variable[static_cast<std::size_t>(variable_index)]) {
            old_to_new[static_cast<std::size_t>(variable_index)] = compact_count++;
        }
    }
    observation.variable_count = static_cast<std::size_t>(compact_count);
    observation.variable_feature_dim = kGraphVariableFeatureCount;
    observation.variable_features.assign(
        observation.variable_count * static_cast<std::size_t>(kGraphVariableFeatureCount), 0.0F);
    observation.variable_categories.assign(
        observation.variable_count * kVariableCategoryCount, 0.0F);
    observation.variable_names.assign(observation.variable_count, "");
    observation.local_lower_bounds.assign(observation.variable_count, 0.0F);

    const VariableFeatureContext context = makeVariableFeatureContext(scip);
    for (int position = 0; position < variable_count; ++position) {
        SCIP_VAR* variable = variables[position];
        const int variable_index = SCIPvarGetProbindex(variable);
        if (variable_index < 0 || variable_index >= variable_count) {
            return SCIP_INVALIDDATA;
        }
        const int compact = old_to_new[static_cast<std::size_t>(variable_index)];
        if (compact < 0) {
            continue;
        }
        SCIP_CALL(extractVariableFeatures(
            scip,
            variable,
            context,
            observation.variable_features.data()
                + static_cast<std::size_t>(compact)
                    * static_cast<std::size_t>(kGraphVariableFeatureCount)));
        const int category = aviationVariableCategory(SCIPvarGetName(variable));
        observation.variable_categories[
            static_cast<std::size_t>(compact) * kVariableCategoryCount + category] = 1.0F;
        observation.variable_names[static_cast<std::size_t>(compact)] = SCIPvarGetName(variable);
        observation.local_lower_bounds[static_cast<std::size_t>(compact)] =
            static_cast<float>(SCIPvarGetLbLocal(variable));
    }
    appendPrimVariableFeatures(scip, observation);

    observation.candidate_indices.reserve(observation.candidates.size());
    observation.candidate_names.reserve(observation.candidates.size());
    for (const BranchCandidate& candidate : observation.candidates) {
        const int compact = old_to_new[static_cast<std::size_t>(candidate.variable_index)];
        if (compact < 0) {
            return SCIP_INVALIDDATA;
        }
        observation.candidate_indices.push_back(compact);
        observation.candidate_names.emplace_back(SCIPvarGetName(candidate.variable));
    }

    std::size_t expanded_row_index = 0;
    const double raw_objective_norm = SCIPgetObjNorm(scip);
    const double objective_norm = raw_objective_norm > 0.0 ? raw_objective_norm : 1.0;
    for (int row_position = 0; row_position < row_count; ++row_position) {
        if (!keep_lp_row[static_cast<std::size_t>(row_position)]) {
            continue;
        }
        SCIP_ROW* row = rows[row_position];
        const double raw_norm = SCIProwGetNorm(row);
        const double norm = raw_norm > 0.0 ? raw_norm : 1.0;
        const double constant = SCIProwGetConstant(row);
        const double raw_lhs = SCIProwGetLhs(row);
        const double raw_rhs = SCIProwGetRhs(row);
        // Ecole checks infinity on the unshifted bound, then subtracts the constant.
        const bool lhs_finite = !SCIPisInfinity(scip, std::abs(raw_lhs));
        const bool rhs_finite = !SCIPisInfinity(scip, std::abs(raw_rhs));
        const double lhs = raw_lhs - constant;
        const double rhs = raw_rhs - constant;
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
        const std::size_t features_before = observation.edge_variable_indices.size();
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
        for (std::size_t edge = features_before; edge < observation.edge_variable_indices.size();
             ++edge) {
            const int compact = old_to_new[static_cast<std::size_t>(
                observation.edge_variable_indices[edge])];
            if (compact < 0) {
                return SCIP_INVALIDDATA;
            }
            observation.edge_variable_indices[edge] = compact;
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
