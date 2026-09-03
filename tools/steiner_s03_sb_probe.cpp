// SCIP 8.0.4 native strong-branch signal probe for Steiner S03.

#include <scip/scip.h>
#include <scip/scip_branch.h>
#include <scip/scip_nodesel.h>
#include <scip/scip_param.h>
#include <scip/scipdefplugins.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <sys/resource.h>
#include <vector>

struct SCIP_BranchruleData {
    int max_states = 2;
    int iteration_limit = 10000;
    int candidate_limit = 0;
    bool idempotent = false;
    double tie_tolerance = 1e-9;
    int states_seen = 0;
    std::set<SCIP_Longint> node_numbers;
};

namespace {

bool isMappedEdgeName(const char* raw_name) {
    if (raw_name == nullptr) return false;
    std::string name(raw_name);
    while (name.rfind("t_", 0) == 0) name.erase(0, 2);
    constexpr const char* prefix = "stp_x_e";
    if (name.rfind(prefix, 0) != 0 || name.size() != 15) return false;
    return std::all_of(name.begin() + 7, name.end(), [](unsigned char c) {
        return c >= '0' && c <= '9';
    });
}

const char* statusName(SCIP_STATUS status) {
    switch (status) {
        case SCIP_STATUS_UNKNOWN: return "unknown";
        case SCIP_STATUS_USERINTERRUPT: return "userinterrupt";
        case SCIP_STATUS_NODELIMIT: return "nodelimit";
        case SCIP_STATUS_TOTALNODELIMIT: return "totalnodelimit";
        case SCIP_STATUS_STALLNODELIMIT: return "stallnodelimit";
        case SCIP_STATUS_TIMELIMIT: return "timelimit";
        case SCIP_STATUS_MEMLIMIT: return "memlimit";
        case SCIP_STATUS_GAPLIMIT: return "gaplimit";
        case SCIP_STATUS_SOLLIMIT: return "sollimit";
        case SCIP_STATUS_BESTSOLLIMIT: return "bestsollimit";
        case SCIP_STATUS_RESTARTLIMIT: return "restartlimit";
        case SCIP_STATUS_OPTIMAL: return "optimal";
        case SCIP_STATUS_INFEASIBLE: return "infeasible";
        case SCIP_STATUS_UNBOUNDED: return "unbounded";
        case SCIP_STATUS_INFORUNBD: return "inforunbd";
        case SCIP_STATUS_TERMINATE: return "terminate";
    }
    return "unrecognized";
}

SCIP_DECL_BRANCHEXECLP(branchExeclpS03Probe) {
    (void)allowaddcons;
    *result = SCIP_DIDNOTRUN;
    auto* data = SCIPbranchruleGetData(branchrule);
    if (data == nullptr || data->states_seen >= data->max_states) return SCIP_OKAY;

    SCIP_NODE* current = SCIPgetCurrentNode(scip);
    const SCIP_Longint node_number = current == nullptr ? -1 : SCIPnodeGetNumber(current);
    if (!data->node_numbers.insert(node_number).second) return SCIP_OKAY;

    SCIP_VAR** candidates = nullptr;
    SCIP_Real* candidate_solutions = nullptr;
    SCIP_Real* candidate_fractions = nullptr;
    int n_candidates = 0;
    int n_priority_candidates = 0;
    int n_fractional_implicit = 0;
    SCIP_CALL(SCIPgetLPBranchCands(
        scip, &candidates, &candidate_solutions, &candidate_fractions,
        &n_candidates, &n_priority_candidates, &n_fractional_implicit));
    (void)candidate_solutions;
    (void)candidate_fractions;
    (void)n_fractional_implicit;

    const int n_legal = n_priority_candidates;
    const int n_eval = data->candidate_limit > 0
        ? std::min(n_legal, data->candidate_limit) : n_legal;
    int mapped = 0;
    for (int i = 0; i < n_legal; ++i) {
        if (isMappedEdgeName(SCIPvarGetName(candidates[i]))) ++mapped;
    }

    int fully_valid = 0;
    int lp_errors = 0;
    int finite_scores = 0;
    double score_min = std::numeric_limits<double>::infinity();
    double score_max = -std::numeric_limits<double>::infinity();
    const SCIP_Longint lp0 = SCIPgetNLPIterations(scip);
    const SCIP_Longint sb_lp0 = SCIPgetNStrongbranchLPIterations(scip);
    const SCIP_Longint sb_calls0 = SCIPgetNStrongbranchs(scip);
    const SCIP_Real lp_obj = SCIPgetLPObjval(scip);

    if (n_eval > 0) SCIP_CALL(SCIPstartStrongbranch(scip, FALSE));
    for (int i = 0; i < n_eval; ++i) {
        SCIP_Real down = 0.0;
        SCIP_Real up = 0.0;
        SCIP_Bool down_valid = FALSE;
        SCIP_Bool up_valid = FALSE;
        SCIP_Bool down_infeasible = FALSE;
        SCIP_Bool up_infeasible = FALSE;
        SCIP_Bool down_conflict = FALSE;
        SCIP_Bool up_conflict = FALSE;
        SCIP_Bool lp_error = FALSE;
        SCIP_CALL(SCIPgetVarStrongbranchFrac(
            scip, candidates[i], data->iteration_limit,
            data->idempotent ? TRUE : FALSE,
            &down, &up, &down_valid, &up_valid,
            &down_infeasible, &up_infeasible,
            &down_conflict, &up_conflict, &lp_error));
        (void)down_conflict;
        (void)up_conflict;
        if (lp_error) ++lp_errors;
        const bool down_ok = down_valid || down_infeasible;
        const bool up_ok = up_valid || up_infeasible;
        if (!lp_error && down_ok && up_ok) ++fully_valid;
        const SCIP_Real down_gain = down_infeasible
            ? 1e6 : (down_valid ? std::max<SCIP_Real>(down - lp_obj, 0.0) : 0.0);
        const SCIP_Real up_gain = up_infeasible
            ? 1e6 : (up_valid ? std::max<SCIP_Real>(up - lp_obj, 0.0) : 0.0);
        const SCIP_Real score = SCIPgetBranchScore(scip, candidates[i], down_gain, up_gain);
        if (!SCIPisInfinity(scip, std::fabs(score)) && std::isfinite(score)) {
            ++finite_scores;
            score_min = std::min(score_min, static_cast<double>(score));
            score_max = std::max(score_max, static_cast<double>(score));
        }
    }
    if (n_eval > 0) SCIP_CALL(SCIPendStrongbranch(scip));

    const SCIP_Longint lp_delta = SCIPgetNLPIterations(scip) - lp0;
    const SCIP_Longint sb_lp_delta = SCIPgetNStrongbranchLPIterations(scip) - sb_lp0;
    const SCIP_Longint sb_calls_delta = SCIPgetNStrongbranchs(scip) - sb_calls0;
    const bool uncapped = data->candidate_limit == 0 || n_eval == n_legal;
    const bool valid = n_legal >= 2 && uncapped && mapped == n_legal &&
        fully_valid == n_eval && finite_scores >= 2 && lp_errors == 0 &&
        sb_calls_delta >= n_eval;
    const double scale = finite_scores > 0
        ? std::max({1.0, std::fabs(score_min), std::fabs(score_max)}) : 1.0;
    const bool all_tie = valid && (score_max - score_min <= data->tie_tolerance * scale);
    ++data->states_seen;
    std::cout << "S03_STATE"
              << " node=" << node_number
              << " depth=" << SCIPgetDepth(scip)
              << " legal=" << n_legal
              << " evaluated=" << n_eval
              << " mapped=" << mapped
              << " fully_valid=" << fully_valid
              << " finite_scores=" << finite_scores
              << " lp_errors=" << lp_errors
              << " score_min=" << (finite_scores ? score_min : 0.0)
              << " score_max=" << (finite_scores ? score_max : 0.0)
              << " lp_iterations_delta=" << lp_delta
              << " sb_lp_iterations_delta=" << sb_lp_delta
              << " sb_calls_delta=" << sb_calls_delta
              << " valid=" << (valid ? 1 : 0)
              << " all_tie=" << (all_tie ? 1 : 0)
              << std::endl;

    if (data->states_seen >= data->max_states) SCIPinterruptSolve(scip);
    return SCIP_OKAY;
}

SCIP_DECL_BRANCHFREE(branchFreeS03Probe) {
    (void)scip;
    delete SCIPbranchruleGetData(branchrule);
    SCIPbranchruleSetData(branchrule, nullptr);
    return SCIP_OKAY;
}

std::string requireValue(int argc, char** argv, int* index) {
    if (*index + 1 >= argc) {
        std::cerr << "missing option value\n";
        std::exit(2);
    }
    return argv[++(*index)];
}

}  // namespace

int main(int argc, char** argv) {
    std::string instance;
    int seed = 0;
    int max_states = 2;
    int iteration_limit = 10000;
    int candidate_limit = 0;
    bool idempotent = false;
    double tie_tolerance = 1e-9;
    double time_limit = 600.0;
    SCIP_Longint node_limit = 200000;
    double memory_limit = 8192.0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--instance") instance = requireValue(argc, argv, &i);
        else if (arg == "--seed") seed = std::stoi(requireValue(argc, argv, &i));
        else if (arg == "--max-states") max_states = std::stoi(requireValue(argc, argv, &i));
        else if (arg == "--iteration-limit") iteration_limit = std::stoi(requireValue(argc, argv, &i));
        else if (arg == "--candidate-limit") candidate_limit = std::stoi(requireValue(argc, argv, &i));
        else if (arg == "--idempotent") idempotent = std::stoi(requireValue(argc, argv, &i)) != 0;
        else if (arg == "--tie-tolerance") tie_tolerance = std::stod(requireValue(argc, argv, &i));
        else if (arg == "--time-limit") time_limit = std::stod(requireValue(argc, argv, &i));
        else if (arg == "--node-limit") node_limit = std::stoll(requireValue(argc, argv, &i));
        else if (arg == "--memory-limit") memory_limit = std::stod(requireValue(argc, argv, &i));
        else {
            std::cerr << "unknown option: " << arg << "\n";
            return 2;
        }
    }
    if (instance.empty() || max_states < 1 || iteration_limit < 1 || candidate_limit < 0) return 2;

    SCIP* scip = nullptr;
    SCIP_RETCODE code = SCIPcreate(&scip);
    if (code != SCIP_OKAY) return static_cast<int>(code);
    SCIP_CALL_ABORT(SCIPincludeDefaultPlugins(scip));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "display/verblevel", 0));
    SCIP_CALL_ABORT(SCIPsetRealParam(scip, "limits/time", time_limit));
    SCIP_CALL_ABORT(SCIPsetLongintParam(scip, "limits/nodes", node_limit));
    SCIP_CALL_ABORT(SCIPsetRealParam(scip, "limits/memory", memory_limit));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "parallel/minnthreads", 1));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "parallel/maxnthreads", 1));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "lp/threads", 1));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "randomization/randomseedshift", seed));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "randomization/permutationseed", seed));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "randomization/lpseed", seed));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "presolving/maxrounds", 0));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "separating/maxrounds", 0));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "separating/maxroundsroot", 0));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "limits/restarts", 0));
    SCIP_CALL_ABORT(SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, TRUE));
    SCIP_NODESEL* estimate = SCIPfindNodesel(scip, "estimate");
    if (estimate == nullptr) return 3;
    SCIP_CALL_ABORT(SCIPsetNodeselStdPriority(scip, estimate, 1000000));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "branching/relpscost/priority", 900000));

    auto* data = new SCIP_BRANCHRULEDATA();
    data->max_states = max_states;
    data->iteration_limit = iteration_limit;
    data->candidate_limit = candidate_limit;
    data->idempotent = idempotent;
    data->tie_tolerance = tie_tolerance;
    SCIP_BRANCHRULE* probe = nullptr;
    SCIP_CALL_ABORT(SCIPincludeBranchruleBasic(
        scip, &probe, "steiner_s03_sb_probe", "S03 native strong-branch signal probe",
        1000000, -1, 1.0, data));
    SCIP_CALL_ABORT(SCIPsetBranchruleExecLp(scip, probe, branchExeclpS03Probe));
    SCIP_CALL_ABORT(SCIPsetBranchruleFree(scip, probe, branchFreeS03Probe));

    SCIP_CALL_ABORT(SCIPreadProb(scip, instance.c_str(), nullptr));
    SCIP_CALL_ABORT(SCIPsolve(scip));
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "S03_FINAL"
              << " status=" << statusName(SCIPgetStatus(scip))
              << " nodes=" << SCIPgetNNodes(scip)
              << " states=" << data->states_seen
              << " lp_iterations=" << SCIPgetNLPIterations(scip)
              << " peak_rss_mb=" << (static_cast<double>(usage.ru_maxrss) / 1024.0)
              << std::endl;
    SCIP_CALL_ABORT(SCIPfree(&scip));
    return 0;
}
