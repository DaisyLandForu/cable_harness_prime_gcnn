#include "rl/prim_bias.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rlbranch {
namespace {

constexpr float kTieTolerance = 1e-7F;

std::string stripTransformedPrefix(std::string name) {
    while (name.size() >= 2 && name[0] == 't' && name[1] == '_') {
        name.erase(0, 2);
    }
    return name;
}

std::vector<std::string> splitUnderscore(const std::string& name) {
    std::vector<std::string> parts;
    std::stringstream stream(name);
    std::string item;
    while (std::getline(stream, item, '_')) {
        parts.push_back(item);
    }
    return parts;
}

bool parseIntToken(const std::string& token, int& value) {
    if (token.empty()) {
        return false;
    }
    for (char character : token) {
        if (!std::isdigit(static_cast<unsigned char>(character))) {
            return false;
        }
    }
    try {
        value = std::stoi(token);
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace

ParsedZVar parseZVariableName(const std::string& name) {
    ParsedZVar parsed;
    const auto parts = splitUnderscore(stripTransformedPrefix(name));
    if (parts.size() != 4 || parts[0] != "z") {
        return parsed;
    }
    if (!parseIntToken(parts[1], parsed.src)
        || !parseIntToken(parts[2], parsed.dst)
        || !parseIntToken(parts[3], parsed.prime)) {
        return ParsedZVar{};
    }
    parsed.valid = true;
    return parsed;
}

ParsedNodePrimeVar parseMVariableName(const std::string& name) {
    ParsedNodePrimeVar parsed;
    const auto parts = splitUnderscore(stripTransformedPrefix(name));
    if (parts.size() != 3 || parts[0] != "m") {
        return parsed;
    }
    if (!parseIntToken(parts[1], parsed.node) || !parseIntToken(parts[2], parsed.prime)) {
        return ParsedNodePrimeVar{};
    }
    parsed.valid = true;
    return parsed;
}

ParsedNodePrimeVar parseYVariableName(const std::string& name) {
    ParsedNodePrimeVar parsed;
    const auto parts = splitUnderscore(stripTransformedPrefix(name));
    if (parts.size() != 3 || parts[0] != "y") {
        return parsed;
    }
    if (!parseIntToken(parts[1], parsed.node) || !parseIntToken(parts[2], parsed.prime)) {
        return ParsedNodePrimeVar{};
    }
    parsed.valid = true;
    return parsed;
}

std::unordered_map<int, std::unordered_set<int>> buildGrownSetsFromScip(
    SCIP* scip,
    double active_threshold) {
    std::unordered_map<int, std::unordered_set<int>> grown;
    const int n_vars = SCIPgetNVars(scip);
    SCIP_VAR** vars = SCIPgetVars(scip);
    for (int index = 0; index < n_vars; ++index) {
        SCIP_VAR* variable = vars[index];
        if (variable == nullptr) {
            continue;
        }
        const std::string name = SCIPvarGetName(variable);
        const ParsedZVar parsed = parseZVariableName(name);
        if (!parsed.valid) {
            continue;
        }
        const SCIP_Real lb = SCIPvarGetLbLocal(variable);
        const SCIP_Real ub = SCIPvarGetUbLocal(variable);
        const SCIP_Real lp = SCIPvarGetLPSol(variable);
        const bool fixed_one = lb > active_threshold;
        const bool lp_one = lp > active_threshold && ub > active_threshold;
        if (!(fixed_one || lp_one)) {
            continue;
        }
        auto& nodes = grown[parsed.prime];
        nodes.insert(parsed.src);
        nodes.insert(parsed.dst);
    }
    return grown;
}

float primScoreForName(
    const std::string& name,
    const std::unordered_map<int, std::unordered_set<int>>& grown,
    bool empty_s_z_prior) {
    const ParsedZVar z_var = parseZVariableName(name);
    if (z_var.valid) {
        const auto iterator = grown.find(z_var.prime);
        if (iterator == grown.end() || iterator->second.empty()) {
            return empty_s_z_prior ? 0.5F : 0.0F;
        }
        const auto& nodes = iterator->second;
        const bool src_in = nodes.find(z_var.src) != nodes.end();
        const bool dst_in = nodes.find(z_var.dst) != nodes.end();
        if (src_in != dst_in) {
            return 1.0F;
        }
        if (src_in && dst_in) {
            return -0.5F;
        }
        return 0.25F;
    }

    const ParsedNodePrimeVar m_var = parseMVariableName(name);
    if (m_var.valid) {
        const auto iterator = grown.find(m_var.prime);
        if (iterator != grown.end() && iterator->second.count(m_var.node) > 0) {
            return 0.3F;
        }
        return 0.0F;
    }

    const ParsedNodePrimeVar y_var = parseYVariableName(name);
    if (y_var.valid) {
        const auto iterator = grown.find(y_var.prime);
        if (iterator != grown.end() && iterator->second.count(y_var.node) > 0) {
            return 0.15F;
        }
        return 0.0F;
    }
    return 0.0F;
}

std::vector<float> candidatePrimScores(
    const GraphObservation& observation,
    const std::unordered_map<int, std::unordered_set<int>>& grown) {
    return candidateBiasScores(observation, grown, "prim", /*depth=*/0);
}

std::vector<float> candidateBiasScores(
    const GraphObservation& observation,
    const std::unordered_map<int, std::unordered_set<int>>& grown,
    const std::string& mode,
    int depth) {
    std::vector<float> scores(observation.candidate_names.size(), 0.0F);
    if (mode == "none" || mode.empty()) {
        return scores;
    }
    const bool use_root_z = (mode == "root_z");
    const bool use_z = (mode == "z") || (use_root_z && depth == 0);
    const bool use_prim = (mode == "prim");
    const bool use_topology = (mode == "topology");
    if (!(use_z || use_prim || use_topology || (use_root_z && depth == 0))) {
        return scores;
    }
    for (std::size_t index = 0; index < observation.candidate_names.size(); ++index) {
        const std::string& name = observation.candidate_names[index];
        if (use_z || (use_root_z && depth == 0)) {
            scores[index] = parseZVariableName(name).valid ? 1.0F : 0.0F;
            continue;
        }
        if (use_prim) {
            scores[index] = primScoreForName(name, grown, /*empty_s_z_prior=*/true);
            continue;
        }
        if (use_topology) {
            scores[index] = primScoreForName(name, grown, /*empty_s_z_prior=*/false);
        }
    }
    return scores;
}

void appendPrimVariableFeatures(
    SCIP* scip,
    GraphObservation& observation,
    double active_threshold) {
    if (observation.variable_count == 0) {
        return;
    }
    const auto grown = buildGrownSetsFromScip(scip, active_threshold);
    const int width = kGraphVariableFeatureCount;
    if (observation.variable_features.size()
        != observation.variable_count * static_cast<std::size_t>(width)) {
        throw std::runtime_error("variable_features width mismatch for Prim append");
    }
    SCIP_VAR** vars = SCIPgetVars(scip);
    const int n_vars = SCIPgetNVars(scip);
    if (vars == nullptr || n_vars <= 0) {
        observation.variable_feature_dim = kGraphVariableFeatureCount;
        return;
    }
    for (int position = 0; position < n_vars; ++position) {
        SCIP_VAR* variable = vars[position];
        if (variable == nullptr) {
            continue;
        }
        const int variable_index = SCIPvarGetProbindex(variable);
        if (variable_index < 0
            || static_cast<std::size_t>(variable_index) >= observation.variable_count) {
            continue;
        }
        float* features = observation.variable_features.data()
            + static_cast<std::size_t>(variable_index) * static_cast<std::size_t>(width)
            + kCandidateVariableFeatureCount;
        for (int dim = 0; dim < kPrimVariableFeatureCount; ++dim) {
            features[dim] = 0.0F;
        }
        const std::string name = SCIPvarGetName(variable);
        const ParsedZVar z_var = parseZVariableName(name);
        if (z_var.valid) {
            const auto iterator = grown.find(z_var.prime);
            if (iterator == grown.end() || iterator->second.empty()) {
                features[3] = 1.0F;  // prim_grown_empty
            } else {
                const auto& nodes = iterator->second;
                const bool src_in = nodes.find(z_var.src) != nodes.end();
                const bool dst_in = nodes.find(z_var.dst) != nodes.end();
                if (src_in != dst_in) {
                    features[0] = 1.0F;  // prim_is_cut
                } else if (src_in && dst_in) {
                    features[1] = 1.0F;  // prim_both_in
                } else {
                    features[2] = 1.0F;  // prim_both_out
                }
            }
            continue;
        }
        const ParsedNodePrimeVar m_var = parseMVariableName(name);
        if (m_var.valid) {
            const auto iterator = grown.find(m_var.prime);
            if (iterator != grown.end() && iterator->second.count(m_var.node) > 0) {
                features[4] = 1.0F;
            }
            continue;
        }
        const ParsedNodePrimeVar y_var = parseYVariableName(name);
        if (y_var.valid) {
            const auto iterator = grown.find(y_var.prime);
            if (iterator != grown.end() && iterator->second.count(y_var.node) > 0) {
                features[5] = 1.0F;
            }
        }
    }
    observation.variable_feature_dim = kGraphVariableFeatureCount;
}

int stableArgmaxBiasedScores(
    const GraphObservation& observation,
    const std::vector<float>& scores) {
    if (scores.empty() || scores.size() != observation.candidates.size()) {
        return -1;
    }
    const float maximum = *std::max_element(scores.begin(), scores.end());
    int best = -1;
    for (int index = 0; index < static_cast<int>(scores.size()); ++index) {
        if (scores[index] < maximum - kTieTolerance) {
            continue;
        }
        if (best < 0
            || observation.candidate_names[index] < observation.candidate_names[best]
            || (observation.candidate_names[index] == observation.candidate_names[best]
                && observation.candidate_indices[index] < observation.candidate_indices[best])) {
            best = index;
        }
    }
    return best;
}

}  // namespace rlbranch
