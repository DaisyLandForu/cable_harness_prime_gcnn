#pragma once

#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <objscip/objbranchrule.h>

#include "rl/scip_feature_extractor.hpp"

namespace rlbranch {

enum class CustomBranchingStrategy {
    Random,
    MostInfeasible,
};

struct BranchruleStats {
    std::int64_t lp_calls = 0;
    std::int64_t decisions = 0;
    std::int64_t candidates_seen = 0;
    std::int64_t legality_checks = 0;
    std::int64_t illegal_actions = 0;
    std::int64_t fallback_count = 0;
    double selection_time_total = 0.0;
    double selection_time_max = 0.0;
    double inference_time_total = 0.0;
    double inference_time_max = 0.0;
};

int selectRandomCandidate(
    const std::vector<BranchCandidate>& candidates,
    std::mt19937& random_engine);

int selectMostInfeasibleCandidate(
    SCIP* scip,
    const std::vector<BranchCandidate>& candidates);

const char* branchruleName(CustomBranchingStrategy strategy);

SCIP_RETCODE includeCustomBranchrule(
    SCIP* scip,
    CustomBranchingStrategy strategy,
    unsigned int seed,
    const std::string& log_path,
    BranchruleStats* stats);

class CustomBranchrule final : public scip::ObjBranchrule {
public:
    CustomBranchrule(
        SCIP* scip,
        CustomBranchingStrategy strategy,
        unsigned int seed,
        const std::string& log_path,
        BranchruleStats* stats);

    SCIP_DECL_BRANCHEXECLP(scip_execlp) override;

private:
    void writeLogRow(
        SCIP* scip,
        const std::vector<BranchCandidate>& candidates,
        int selected_index,
        double selection_time,
        const char* result,
        const std::string& fallback_reason);

    CustomBranchingStrategy strategy_;
    std::mt19937 random_engine_;
    BranchruleStats* stats_;
    std::ofstream log_stream_;
    std::int64_t event_index_ = 0;
};

}  // namespace rlbranch
