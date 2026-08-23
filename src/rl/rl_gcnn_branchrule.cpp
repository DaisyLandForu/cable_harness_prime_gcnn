#include "rl/rl_gcnn_branchrule.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include <scip/pub_tree.h>
#include <scip/scip_branch.h>
#include <scip/scip_numerics.h>
#include <scip/scip_sol.h>
#include <scip/scip_solvingstats.h>
#include <scip/scip_tree.h>
#include <scip/scip_var.h>

#include "rl/prim_bias.hpp"

namespace rlbranch {
namespace {

constexpr int kRlBranchrulePriority = 1000000;
constexpr float kTieTolerance = 1e-7F;

std::string csvEscape(const std::string& input) {
    if (input.find_first_of(",\"\n\r") == std::string::npos) {
        return input;
    }
    std::string escaped = "\"";
    for (char character : input) {
        escaped += character == '"' ? "\"\"" : std::string(1, character);
    }
    return escaped + '"';
}

bool isLegal(const GraphObservation& observation, int index) {
    return index >= 0
        && index < static_cast<int>(observation.candidates.size())
        && observation.candidates[index].variable != nullptr;
}

int stableArgmax(
    const GraphObservation& observation,
    const std::vector<float>& q_values) {
    if (q_values.empty() || q_values.size() != observation.candidates.size()) {
        return -1;
    }
    const float maximum = *std::max_element(q_values.begin(), q_values.end());
    int best = -1;
    for (int index = 0; index < static_cast<int>(q_values.size()); ++index) {
        if (q_values[index] < maximum - kTieTolerance) {
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

std::string variableFamily(const std::string& name) {
    std::string stripped = name;
    while (stripped.size() >= 2 && stripped[0] == 't' && stripped[1] == '_') {
        stripped.erase(0, 2);
    }
    if (!stripped.empty() && stripped[0] == 'z') {
        return "z";
    }
    if (!stripped.empty() && stripped[0] == 'm') {
        return "m";
    }
    if (!stripped.empty() && stripped[0] == 'y') {
        return "y";
    }
    return "other";
}

void qStats(
    const std::vector<float>& q_values,
    double& mean,
    double& stdev,
    double& qmin,
    double& qmax,
    double& top_margin) {
    mean = stdev = qmin = qmax = top_margin = std::numeric_limits<double>::quiet_NaN();
    if (q_values.empty()) {
        return;
    }
    qmin = *std::min_element(q_values.begin(), q_values.end());
    qmax = *std::max_element(q_values.begin(), q_values.end());
    mean = std::accumulate(q_values.begin(), q_values.end(), 0.0) / q_values.size();
    double var = 0.0;
    for (float value : q_values) {
        const double delta = value - mean;
        var += delta * delta;
    }
    stdev = std::sqrt(var / q_values.size());
    if (q_values.size() >= 2) {
        std::vector<float> sorted = q_values;
        std::nth_element(sorted.begin(), sorted.end() - 2, sorted.end());
        const float top2 = sorted[sorted.size() - 2];
        top_margin = qmax - top2;
    } else {
        top_margin = 0.0;
    }
}

}  // namespace

RlGcnnBranchrule::RlGcnnBranchrule(
    SCIP* scip,
    const RlGcnnOptions& options,
    BranchruleStats* stats)
    : ObjBranchrule(
          scip,
          kRlGcnnBranchruleName,
          "TorchScript bipartite GCNN branching rule with SCIP fallback",
          kRlBranchrulePriority,
          -1,
          1.0),
      options_(options),
      model_runner_(
          options.model_path,
          options.device,
          options.use_prim_features
              ? kGraphVariableFeatureCount
              : kCandidateVariableFeatureCount),
      stats_(stats) {
    if (!options.log_path.empty()) {
        const std::filesystem::path parent =
            std::filesystem::path(options.log_path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        log_stream_.open(options.log_path);
        if (log_stream_) {
            log_stream_
                << "event_index,node_id,depth,candidate_count,"
                << "selected_candidate_index,selected_variable_index,selected_variable_name,"
                << "variable_family,lp_value,fractionality,"
                << "selected_q,selected_bias,selected_biased,"
                << "q_mean,q_std,q_min,q_max,q_top1_top2_margin,"
                << "bias_mode,lambda_prim,"
                << "primal_bound,dual_bound,gap,lp_iterations,nodes,"
                << "graph_extract_time_seconds,inference_time_seconds,selection_time_seconds,"
                << "selected_is_candidate,result,fallback_reason\n";
        }
    }
}

SCIP_DECL_BRANCHEXECLP(RlGcnnBranchrule::scip_execlp) {
    (void)branchrule;
    (void)allowaddcons;
    *result = SCIP_DIDNOTRUN;
    ++stats_->lp_calls;
    const auto selection_start = std::chrono::steady_clock::now();
    GraphObservation observation;
    SCIP_RETCODE return_code = SCIP_OKAY;
    double graph_extract_time = 0.0;
    try {
        const auto extract_start = std::chrono::steady_clock::now();
        return_code = extractGraphObservation(
            scip, observation, options_.use_prim_features);
        graph_extract_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - extract_start).count();
    } catch (const std::exception& exception) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, {}, {}, -1, 0.0, 0.0, graph_extract_time, "did_not_run",
            "graph_extraction_exception:" + std::string(exception.what()));
        return SCIP_OKAY;
    }
    if (return_code != SCIP_OKAY) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, {}, {}, -1, 0.0, 0.0, graph_extract_time, "did_not_run",
            "graph_extraction_error");
        return SCIP_OKAY;
    }
    stats_->candidates_seen += static_cast<std::int64_t>(observation.candidates.size());
    if (observation.candidates.empty()) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, {}, {}, -1, 0.0, 0.0, graph_extract_time, "did_not_run",
            "no_priority_lp_candidates");
        return SCIP_OKAY;
    }
    if (options_.max_depth >= 0 && SCIPgetDepth(scip) > options_.max_depth) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, {}, {}, -1, 0.0, 0.0, graph_extract_time, "did_not_run",
            "depth_limit:" + options_.fallback);
        return SCIP_OKAY;
    }
    if (static_cast<int>(observation.candidates.size()) < options_.min_candidates) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, {}, {}, -1, 0.0, 0.0, graph_extract_time, "did_not_run",
            "minimum_candidates:" + options_.fallback);
        return SCIP_OKAY;
    }
    if (!model_runner_.ready()) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, {}, {}, -1, 0.0, 0.0, graph_extract_time, "did_not_run",
            "model_unavailable:" + model_runner_.error());
        return SCIP_OKAY;
    }

    std::vector<float> q_values;
    const auto inference_start = std::chrono::steady_clock::now();
    try {
        q_values = model_runner_.predict(observation);
    } catch (const std::exception& exception) {
        const double inference_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - inference_start).count();
        stats_->inference_time_total += inference_time;
        stats_->inference_time_max = std::max(stats_->inference_time_max, inference_time);
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, {}, {}, -1, inference_time, 0.0, graph_extract_time,
            "did_not_run", "inference_exception:" + std::string(exception.what()));
        return SCIP_OKAY;
    }
    const double inference_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - inference_start).count();
    stats_->inference_time_total += inference_time;
    stats_->inference_time_max = std::max(stats_->inference_time_max, inference_time);

    const int depth = SCIPgetDepth(scip);
    const std::string mode = options_.bias_mode.empty() ? "prim" : options_.bias_mode;
    bool apply_bias = options_.lambda_prim != 0.0F && mode != "none";
    if (apply_bias && mode != "z" && mode != "root_z") {
        apply_bias = depth >= options_.prim_min_depth;
    }
    if (apply_bias && mode == "root_z") {
        apply_bias = (depth == 0);
    }

    std::unordered_map<int, std::unordered_set<int>> grown;
    std::vector<float> bias_scores(observation.candidates.size(), 0.0F);
    std::vector<float> biased = q_values;
    if (apply_bias) {
        if (mode == "prim" || mode == "topology") {
            grown = buildGrownSetsFromScip(scip, 0.5);
            if (options_.prim_require_grown) {
                bool has_grown = false;
                for (const auto& entry : grown) {
                    if (!entry.second.empty()) {
                        has_grown = true;
                        break;
                    }
                }
                apply_bias = has_grown;
            }
        }
    }
    if (apply_bias) {
        bias_scores = candidateBiasScores(observation, grown, mode, depth);
        for (std::size_t index = 0; index < biased.size(); ++index) {
            biased[index] += options_.lambda_prim * bias_scores[index];
        }
    }
    const int selected_index = apply_bias
        ? stableArgmaxBiasedScores(observation, biased)
        : stableArgmax(observation, q_values);

    const double selection_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - selection_start).count();
    stats_->selection_time_total += selection_time;
    stats_->selection_time_max = std::max(stats_->selection_time_max, selection_time);
    ++stats_->legality_checks;
    if (!isLegal(observation, selected_index)) {
        ++stats_->illegal_actions;
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, q_values, bias_scores, biased, selected_index,
            inference_time, selection_time, graph_extract_time,
            "did_not_run", "illegal_selection");
        return SCIP_OKAY;
    }

    SCIP_NODE* down_child = nullptr;
    SCIP_NODE* equal_child = nullptr;
    SCIP_NODE* up_child = nullptr;
    const BranchCandidate& selected = observation.candidates[selected_index];
    return_code = SCIPbranchVarVal(
        scip, selected.variable, selected.lp_value,
        &down_child, &equal_child, &up_child);
    if (return_code != SCIP_OKAY) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, q_values, bias_scores, biased, selected_index,
            inference_time, selection_time, graph_extract_time,
            "did_not_run", "branch_api_error");
        return SCIP_OKAY;
    }
    if (down_child != nullptr || equal_child != nullptr || up_child != nullptr) {
        *result = SCIP_BRANCHED;
        ++stats_->decisions;
        writeLogRow(
            scip, observation, q_values, bias_scores, biased, selected_index,
            inference_time, selection_time, graph_extract_time, "branched", "");
    } else {
        *result = SCIP_REDUCEDDOM;
        writeLogRow(
            scip, observation, q_values, bias_scores, biased, selected_index,
            inference_time, selection_time, graph_extract_time, "reduced_domain", "");
    }
    return SCIP_OKAY;
}

void RlGcnnBranchrule::writeLogRow(
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
    const std::string& fallback_reason) {
    if (!log_stream_) {
        return;
    }
    const bool legal = isLegal(observation, selected_index);
    const BranchCandidate* selected = legal ? &observation.candidates[selected_index] : nullptr;
    SCIP_NODE* node = SCIPgetCurrentNode(scip);
    double q_mean = 0.0;
    double q_std = 0.0;
    double q_min = 0.0;
    double q_max = 0.0;
    double q_margin = 0.0;
    qStats(q_values, q_mean, q_std, q_min, q_max, q_margin);

    const double primal = SCIPgetPrimalbound(scip);
    const double dual = SCIPgetDualbound(scip);
    const double gap = SCIPgetGap(scip);
    const SCIP_Longint lp_iters = SCIPgetNLPIterations(scip);
    const SCIP_Longint nodes = SCIPgetNNodes(scip);

    log_stream_
        << event_index_++ << ','
        << (node == nullptr ? -1 : SCIPnodeGetNumber(node)) << ','
        << SCIPgetDepth(scip) << ','
        << observation.candidates.size() << ','
        << (selected == nullptr ? -1 : selected->candidate_index) << ','
        << (selected == nullptr ? -1 : selected->variable_index) << ',';
    if (selected == nullptr) {
        log_stream_ << ",,,,,,,,";
    } else {
        const std::string& name = observation.candidate_names[selected_index];
        log_stream_
            << csvEscape(name) << ','
            << variableFamily(name) << ','
            << std::setprecision(17) << selected->lp_value << ','
            << selected->fractionality << ','
            << (q_values.size() == observation.candidates.size()
                ? q_values[selected_index]
                : std::numeric_limits<float>::quiet_NaN()) << ','
            << (bias_scores.size() == observation.candidates.size()
                ? bias_scores[selected_index]
                : 0.0F) << ','
            << (biased_scores.size() == observation.candidates.size()
                ? biased_scores[selected_index]
                : std::numeric_limits<float>::quiet_NaN()) << ',';
    }
    log_stream_
        << std::setprecision(17) << q_mean << ',' << q_std << ',' << q_min << ','
        << q_max << ',' << q_margin << ','
        << csvEscape(options_.bias_mode) << ','
        << options_.lambda_prim << ','
        << primal << ',' << dual << ',' << gap << ','
        << lp_iters << ',' << nodes << ','
        << graph_extract_time << ','
        << inference_time << ','
        << selection_time << ','
        << (legal ? "true" : "false") << ','
        << result << ','
        << csvEscape(fallback_reason) << '\n';
}

SCIP_RETCODE includeRlGcnnBranchrule(
    SCIP* scip,
    const RlGcnnOptions& options,
    BranchruleStats* stats) {
    return SCIPincludeObjBranchrule(
        scip, new RlGcnnBranchrule(scip, options, stats), true);
}

}  // namespace rlbranch
