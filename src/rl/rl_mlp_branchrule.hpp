#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <objscip/objbranchrule.h>

#include "rl/model_runner.hpp"
#include "rl/rl_branchrule.hpp"

namespace rlbranch {

struct RlMlpOptions {
    std::string model_path;
    std::string device = "cpu";
    std::string fallback = "relpscost";
    std::string log_path;
    int max_depth = -1;
    int min_candidates = 1;
};

constexpr const char* kRlMlpBranchruleName = "rlmlp";

class RlMlpBranchrule final : public scip::ObjBranchrule {
public:
    RlMlpBranchrule(
        SCIP* scip,
        const RlMlpOptions& options,
        BranchruleStats* stats);

    SCIP_DECL_BRANCHEXECLP(scip_execlp) override;

private:
    void writeLogRow(
        SCIP* scip,
        const CandidateObservation& observation,
        const std::vector<float>& q_values,
        int selected_index,
        double inference_time,
        double selection_time,
        const char* result,
        const std::string& fallback_reason);

    RlMlpOptions options_;
    ModelRunner model_runner_;
    BranchruleStats* stats_;
    std::ofstream log_stream_;
    std::int64_t event_index_ = 0;
};

SCIP_RETCODE includeRlMlpBranchrule(
    SCIP* scip,
    const RlMlpOptions& options,
    BranchruleStats* stats);

}  // namespace rlbranch
