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

class LayerDSU {
public:
    void add(int node) {
        if (parent.find(node) != parent.end()) {
            return;
        }
        parent[node] = node;
        size[node] = 1;
    }

    int find(int node) {
        int root = node;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[node] != root) {
            const int next = parent[node];
            parent[node] = root;
            node = next;
        }
        return root;
    }

    void unite(int left, int right) {
        add(left);
        add(right);
        int left_root = find(left);
        int right_root = find(right);
        if (left_root == right_root) {
            return;
        }
        if (size[left_root] < size[right_root]) {
            std::swap(left_root, right_root);
        }
        parent[right_root] = left_root;
        size[left_root] += size[right_root];
    }

    bool contains(int node) const {
        return parent.find(node) != parent.end();
    }

    int componentSize(int node) {
        return contains(node) ? size[find(node)] : 0;
    }

    int grownNodeCount() const {
        return static_cast<int>(parent.size());
    }

    std::unordered_set<int> nodes() const {
        std::unordered_set<int> values;
        for (const auto& item : parent) {
            values.insert(item.first);
        }
        return values;
    }

private:
    std::unordered_map<int, int> parent;
    std::unordered_map<int, int> size;
};

std::unordered_map<int, LayerDSU> buildDsuLayersFromScip(
    SCIP* scip,
    double active_threshold) {
    std::unordered_map<int, LayerDSU> layers;
    const int n_vars = SCIPgetNVars(scip);
    SCIP_VAR** vars = SCIPgetVars(scip);
    for (int index = 0; index < n_vars; ++index) {
        SCIP_VAR* variable = vars[index];
        if (variable == nullptr) {
            continue;
        }
        const ParsedZVar parsed = parseZVariableName(SCIPvarGetName(variable));
        if (!parsed.valid || SCIPvarGetLbLocal(variable) <= active_threshold) {
            continue;
        }
        layers[parsed.prime].unite(parsed.src, parsed.dst);
    }
    return layers;
}

std::unordered_map<int, std::unordered_set<int>> buildGrownSetsFromScip(
    SCIP* scip,
    double active_threshold) {
    std::unordered_map<int, std::unordered_set<int>> grown;
    for (auto& item : buildDsuLayersFromScip(scip, active_threshold)) {
        grown[item.first] = item.second.nodes();
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
    auto layers = buildDsuLayersFromScip(scip, active_threshold);
    const int width = kGraphVariableFeatureCount;
    if (observation.variable_features.size()
        != observation.variable_count * static_cast<std::size_t>(width)) {
        throw std::runtime_error("variable_features width mismatch for DSU append");
    }
    if (observation.variable_names.size() != observation.variable_count) {
        throw std::runtime_error("variable_names must align with DSU features");
    }
    for (std::size_t variable_index = 0; variable_index < observation.variable_count;
         ++variable_index) {
        float* features = observation.variable_features.data()
            + variable_index * static_cast<std::size_t>(width)
            + kCandidateVariableFeatureCount;
        for (int dim = 0; dim < kPrimVariableFeatureCount; ++dim) {
            features[dim] = 0.0F;
        }
        const ParsedZVar z_var = parseZVariableName(observation.variable_names[variable_index]);
        if (!z_var.valid) {
            continue;
        }
        auto iterator = layers.find(z_var.prime);
        if (iterator == layers.end() || iterator->second.grownNodeCount() == 0) {
            features[3] = 1.0F;  // prim_unseen
            continue;
        }
        LayerDSU& dsu = iterator->second;
        const bool src_in = dsu.contains(z_var.src);
        const bool dst_in = dsu.contains(z_var.dst);
        const float grown = static_cast<float>(dsu.grownNodeCount());
        features[4] = src_in ? static_cast<float>(dsu.componentSize(z_var.src)) / grown : 0.0F;
        features[5] = dst_in ? static_cast<float>(dsu.componentSize(z_var.dst)) / grown : 0.0F;
        if (src_in && dst_in) {
            features[dsu.find(z_var.src) == dsu.find(z_var.dst) ? 2 : 1] = 1.0F;
        } else if (src_in != dst_in) {
            features[0] = 1.0F;
        } else {
            features[3] = 1.0F;
            features[4] = 0.0F;
            features[5] = 0.0F;
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
