#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <objscip/objbranchrule.h>

#include "rl/gcnn_model_runner.hpp"
#include "rl/rl_branchrule.hpp"

namespace rlbranch {

struct RlGcnnOptions {
    std::string model_path;
    std::string device = "cpu";
    std::string fallback = "relpscost";
    std::string log_path;
    int max_depth = -1;
    int min_candidates = 1;
};

constexpr const char* kRlGcnnBranchruleName = "rlgcnn";

class RlGcnnBranchrule final : public scip::ObjBranchrule {
public:
    RlGcnnBranchrule(
        SCIP* scip,
        const RlGcnnOptions& options,
        BranchruleStats* stats);

    SCIP_DECL_BRANCHEXECLP(scip_execlp) override;

private:
    void writeLogRow(
        SCIP* scip,
        const GraphObservation& observation,
        const std::vector<float>& q_values,
        int selected_index,
        double inference_time,
        double selection_time,
        const char* result,
        const std::string& fallback_reason);

    RlGcnnOptions options_;
    GcnnModelRunner model_runner_;
    BranchruleStats* stats_;
    std::ofstream log_stream_;
    std::int64_t event_index_ = 0;
};

SCIP_RETCODE includeRlGcnnBranchrule(
    SCIP* scip,
    const RlGcnnOptions& options,
    BranchruleStats* stats);

}  // namespace rlbranch
