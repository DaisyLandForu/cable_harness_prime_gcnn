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
    // Phase A: score = Q + lambda_prim * PrimScore. 0 disables bias.
    float lambda_prim = 0.0F;
    // Apply Prim bias only when SCIP depth >= prim_min_depth (0 = always).
    int prim_min_depth = 0;
    // If true, skip Prim bias while all grown sets S_p are empty.
    bool prim_require_grown = false;
    // Phase B: append 6 Prim neighborhood dims to variable features (needs matching model).
    bool use_prim_features = false;
    // C0 bias modes: none | z | root_z | prim | topology
    // topology = PrimScore without empty-S uniform z prior (+0.5).
    std::string bias_mode = "prim";
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
        const std::vector<float>& bias_scores,
        const std::vector<float>& biased_scores,
        int selected_index,
        double inference_time,
        double selection_time,
        double graph_extract_time,
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
