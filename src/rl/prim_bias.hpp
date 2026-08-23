#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <scip/scip.h>

#include "rl/scip_graph_feature_extractor.hpp"

namespace rlbranch {

struct ParsedZVar {
    int src = -1;
    int dst = -1;
    int prime = -1;
    bool valid = false;
};

struct ParsedNodePrimeVar {
    int node = -1;
    int prime = -1;
    bool valid = false;
};

ParsedZVar parseZVariableName(const std::string& name);
ParsedNodePrimeVar parseMVariableName(const std::string& name);
ParsedNodePrimeVar parseYVariableName(const std::string& name);

// Per-prime node sets already covered by active (≈1) z-edges.
std::unordered_map<int, std::unordered_set<int>> buildGrownSetsFromScip(
    SCIP* scip,
    double active_threshold = 0.5);

float primScoreForName(
    const std::string& name,
    const std::unordered_map<int, std::unordered_set<int>>& grown,
    bool empty_s_z_prior = true);

// C0: structural / family bias scores for candidates.
// mode: none|z|root_z|prim|topology
std::vector<float> candidateBiasScores(
    const GraphObservation& observation,
    const std::unordered_map<int, std::unordered_set<int>>& grown,
    const std::string& mode,
    int depth);

std::vector<float> candidatePrimScores(
    const GraphObservation& observation,
    const std::unordered_map<int, std::unordered_set<int>>& grown);

// Write 6 Prim neighborhood flags into each variable row after the ECOLE block.
void appendPrimVariableFeatures(
    SCIP* scip,
    GraphObservation& observation,
    double active_threshold = 0.5);

// Argmax over precomputed scores with the same stable tie-break as GCNN.
int stableArgmaxBiasedScores(
    const GraphObservation& observation,
    const std::vector<float>& scores);

}  // namespace rlbranch
