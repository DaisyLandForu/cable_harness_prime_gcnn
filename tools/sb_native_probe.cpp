// C1.1 native SCIP strong-branch probe via a one-shot branchrule.
// Build: make sb_native_probe
// Example:
//   ./build/sb_native_probe --instance data/instances/train/syn_medium_s101.cip --seed 0 --max-cands 64

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Complete SCIP's opaque branchrule data type for this tool.
struct SCIP_BranchruleData {
    int max_cands = 64;
    int probes_done = 0;
    int max_probes = 3;
    SCIP_Longint last_delta_iters = 0;
    int last_n_eval = 0;
    int last_n_nonzero = 0;
    double last_score_min = 0.0;
    double last_score_max = 0.0;
};

namespace {

SCIP_DECL_BRANCHEXECLP(branchExeclpNativeProbe) {
    (void)allowaddcons;
    SCIP_BRANCHRULEDATA* data = SCIPbranchruleGetData(branchrule);
    *result = SCIP_DIDNOTRUN;
    if (data == nullptr || data->probes_done >= data->max_probes) {
        return SCIP_OKAY;
    }

    SCIP_VAR** lpcands = nullptr;
    SCIP_Real* lpcandssol = nullptr;
    SCIP_Real* lpcandsfrac = nullptr;
    int nlpcands = 0;
    int npriocands = 0;
    int nfrac = 0;
    SCIP_CALL(SCIPgetLPBranchCands(
        scip, &lpcands, &lpcandssol, &lpcandsfrac, &nlpcands, &npriocands, &nfrac));
    if (nlpcands <= 0) {
        std::cout << "probe_event depth=" << SCIPgetDepth(scip)
                  << " nlpcands=0 status=strong_no_lp_candidate\n";
        ++data->probes_done;
        return SCIP_OKAY;
    }

    const int n_eval = std::min(nlpcands, data->max_cands);
    const SCIP_Longint iters0 = SCIPgetNLPIterations(scip);
    const SCIP_Longint sb_iters0 = SCIPgetNStrongbranchLPIterations(scip);
    const SCIP_Longint sb_calls0 = SCIPgetNStrongbranchs(scip);
    const SCIP_Real lpobj = SCIPgetLPObjval(scip);

    SCIP_CALL(SCIPstartStrongbranch(scip, /*enablepropagation=*/FALSE));
    std::vector<double> scores;
    scores.reserve(static_cast<size_t>(n_eval));
    int n_nonzero = 0;
    int n_lperror = 0;
    int n_valid = 0;
    for (int i = 0; i < n_eval; ++i) {
        SCIP_VAR* var = lpcands[i];
        SCIP_Real down = 0.0;
        SCIP_Real up = 0.0;
        SCIP_Bool downvalid = FALSE;
        SCIP_Bool upvalid = FALSE;
        SCIP_Bool downinf = FALSE;
        SCIP_Bool upinf = FALSE;
        SCIP_Bool downconflict = FALSE;
        SCIP_Bool upconflict = FALSE;
        SCIP_Bool lperror = FALSE;
        SCIP_CALL(SCIPgetVarStrongbranchFrac(
            scip,
            var,
            /*itlim=*/10000,
            /*idempotent=*/FALSE,
            &down,
            &up,
            &downvalid,
            &upvalid,
            &downinf,
            &upinf,
            &downconflict,
            &upconflict,
            &lperror));
        if (lperror) {
            ++n_lperror;
        }
        SCIP_Real downgain = 0.0;
        SCIP_Real upgain = 0.0;
        if (downvalid || downinf) {
            downgain = downinf ? 1e6 : std::max<SCIP_Real>(down - lpobj, 0.0);
            ++n_valid;
        }
        if (upvalid || upinf) {
            upgain = upinf ? 1e6 : std::max<SCIP_Real>(up - lpobj, 0.0);
            ++n_valid;
        }
        const SCIP_Real score = SCIPgetBranchScore(scip, var, downgain, upgain);
        scores.push_back(static_cast<double>(score));
        if (downgain > 1e-12 || upgain > 1e-12 || downinf || upinf) {
            ++n_nonzero;
        }
        if (i < 5) {
            std::cout << "  cand=" << SCIPvarGetName(var)
                      << " frac=" << lpcandsfrac[i]
                      << " down=" << down << " up=" << up
                      << " dvalid=" << static_cast<int>(downvalid)
                      << " uvalid=" << static_cast<int>(upvalid)
                      << " dinf=" << static_cast<int>(downinf)
                      << " uinf=" << static_cast<int>(upinf)
                      << " downgain=" << downgain << " upgain=" << upgain
                      << " score=" << score << " lperror=" << static_cast<int>(lperror)
                      << "\n";
        }
    }
    SCIP_CALL(SCIPendStrongbranch(scip));

    const SCIP_Longint iters1 = SCIPgetNLPIterations(scip);
    const SCIP_Longint sb_iters1 = SCIPgetNStrongbranchLPIterations(scip);
    const SCIP_Longint sb_calls1 = SCIPgetNStrongbranchs(scip);
    const double smin = scores.empty()
        ? 0.0
        : *std::min_element(scores.begin(), scores.end());
    const double smax = scores.empty()
        ? 0.0
        : *std::max_element(scores.begin(), scores.end());
    data->last_delta_iters = iters1 - iters0;
    data->last_n_eval = n_eval;
    data->last_n_nonzero = n_nonzero;
    data->last_score_min = smin;
    data->last_score_max = smax;
    ++data->probes_done;

    // Ranking-informative requires score diversity, not merely nonzero raw gains.
    const bool informative = (smax - smin) > 1e-15;
    std::cout << "probe_event depth=" << SCIPgetDepth(scip)
              << " node=" << SCIPgetNNodes(scip)
              << " nlpcands=" << nlpcands
              << " n_eval=" << n_eval
              << " lp_iters_delta=" << (iters1 - iters0)
              << " sb_lp_iters_delta=" << (sb_iters1 - sb_iters0)
              << " sb_calls_delta=" << (sb_calls1 - sb_calls0)
              << " n_nonzero_gain=" << n_nonzero
              << " n_lperror=" << n_lperror
              << " n_valid_bounds=" << n_valid
              << " score_min=" << smin
              << " score_max=" << smax
              << " score_range=" << (smax - smin)
              << " native_sb_informative=" << (informative ? "yes" : "no")
              << "\n";

    // Branch on the best scored candidate so the search continues.
    int best = 0;
    for (int i = 1; i < n_eval; ++i) {
        if (scores[static_cast<size_t>(i)] > scores[static_cast<size_t>(best)]) {
            best = i;
        }
    }
    SCIP_NODE* downchild = nullptr;
    SCIP_NODE* eqchild = nullptr;
    SCIP_NODE* upchild = nullptr;
    SCIP_CALL(SCIPbranchVarVal(
        scip, lpcands[best], lpcandssol[best], &downchild, &eqchild, &upchild));
    if (downchild != nullptr || eqchild != nullptr || upchild != nullptr) {
        *result = SCIP_BRANCHED;
    }
    return SCIP_OKAY;
}

SCIP_DECL_BRANCHFREE(branchFreeNativeProbe) {
    SCIP_BRANCHRULEDATA* data = SCIPbranchruleGetData(branchrule);
    delete data;
    SCIPbranchruleSetData(branchrule, nullptr);
    return SCIP_OKAY;
}

void usage() {
    std::cerr << "Usage: sb_native_probe --instance <cip> [--seed N] "
                 "[--max-cands N] [--max-probes N] [--time-limit T] [--node-limit N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string instance;
    int seed = 0;
    int max_cands = 64;
    int max_probes = 3;
    double time_limit = 180.0;
    long long node_limit = 30;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char*) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--instance") {
            instance = need("--instance");
        } else if (arg == "--seed") {
            seed = std::stoi(need("--seed"));
        } else if (arg == "--max-cands") {
            max_cands = std::stoi(need("--max-cands"));
        } else if (arg == "--max-probes") {
            max_probes = std::stoi(need("--max-probes"));
        } else if (arg == "--time-limit") {
            time_limit = std::stod(need("--time-limit"));
        } else if (arg == "--node-limit") {
            node_limit = std::stoll(need("--node-limit"));
        } else if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        } else {
            usage();
            return 2;
        }
    }
    if (instance.empty()) {
        usage();
        return 2;
    }

    SCIP* scip = nullptr;
    SCIP_CALL_ABORT(SCIPcreate(&scip));
    SCIP_CALL_ABORT(SCIPincludeDefaultPlugins(scip));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "randomization/randomseedshift", seed));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "randomization/permutationseed", seed));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "randomization/lpseed", seed));
    SCIP_CALL_ABORT(SCIPsetRealParam(scip, "limits/time", time_limit));
    SCIP_CALL_ABORT(SCIPsetLongintParam(scip, "limits/nodes", node_limit));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "lp/threads", 1));
    SCIP_CALL_ABORT(SCIPsetIntParam(scip, "parallel/maxnthreads", 1));

    SCIP_BRANCHRULEDATA* data = new SCIP_BRANCHRULEDATA();
    data->max_cands = max_cands;
    data->max_probes = max_probes;
    SCIP_BRANCHRULE* branchrule = nullptr;
    SCIP_CALL_ABORT(SCIPincludeBranchruleBasic(
        scip,
        &branchrule,
        "native_sb_probe",
        "C1.1 native strong branching probe",
        1000000,  // priority >> relpscost
        -1,
        1.0,
        data));
    SCIP_CALL_ABORT(SCIPsetBranchruleExecLp(scip, branchrule, branchExeclpNativeProbe));
    SCIP_CALL_ABORT(SCIPsetBranchruleFree(scip, branchrule, branchFreeNativeProbe));

    std::cout << "instance=" << instance << " seed=" << seed << "\n";
    SCIP_CALL_ABORT(SCIPreadProb(scip, instance.c_str(), nullptr));
    SCIP_CALL_ABORT(SCIPsolve(scip));
    std::cout << "solve_status=" << SCIPgetStatus(scip)
              << " nodes=" << SCIPgetNNodes(scip)
              << " probes_done=" << data->probes_done
              << " last_lp_iters_delta=" << data->last_delta_iters
              << " last_score_range=" << (data->last_score_max - data->last_score_min)
              << "\n";
    SCIP_CALL_ABORT(SCIPfree(&scip));
    return 0;
}
