#include "rl/rl_mlp_branchrule.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <limits>

#include <scip/pub_tree.h>
#include <scip/pub_var.h>
#include <scip/scip_branch.h>
#include <scip/scip_tree.h>

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
    escaped += '"';
    return escaped;
}

bool isCandidateLegal(const CandidateObservation& observation, int selected_index) {
    if (selected_index < 0
        || selected_index >= static_cast<int>(observation.candidates.size())) {
        return false;
    }
    SCIP_VAR* variable = observation.candidates[selected_index].variable;
    return variable != nullptr
        && std::any_of(
            observation.candidates.begin(),
            observation.candidates.end(),
            [variable](const BranchCandidate& candidate) {
                return candidate.variable == variable;
            });
}

int stableArgmax(
    const CandidateObservation& observation,
    const std::vector<float>& q_values) {
    if (q_values.size() != observation.candidates.size() || q_values.empty()) {
        return -1;
    }
    const float maximum = *std::max_element(q_values.begin(), q_values.end());
    int best_index = -1;
    for (int index = 0; index < static_cast<int>(q_values.size()); ++index) {
        if (q_values[index] < maximum - kTieTolerance) {
            continue;
        }
        if (best_index < 0
            || observation.variable_names[index] < observation.variable_names[best_index]
            || (observation.variable_names[index] == observation.variable_names[best_index]
                && observation.candidates[index].variable_index
                    < observation.candidates[best_index].variable_index)) {
            best_index = index;
        }
    }
    return best_index;
}

}  // namespace

RlMlpBranchrule::RlMlpBranchrule(
    SCIP* scip,
    const RlMlpOptions& options,
    BranchruleStats* stats)
    : ObjBranchrule(
          scip,
          kRlMlpBranchruleName,
          "TorchScript Candidate MLP branching rule with SCIP fallback",
          kRlBranchrulePriority,
          -1,
          1.0),
      options_(options),
      model_runner_(options.model_path, options.device),
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
                << "event_index,node_id,depth,candidate_count,selected_candidate_index,"
                << "selected_variable_index,selected_variable_name,lp_value,fractionality,"
                << "selected_q,inference_time_seconds,selection_time_seconds,"
                << "selected_is_candidate,result,fallback_reason\n";
        }
    }
}

SCIP_DECL_BRANCHEXECLP(RlMlpBranchrule::scip_execlp) {
    (void)branchrule;
    (void)allowaddcons;
    *result = SCIP_DIDNOTRUN;
    ++stats_->lp_calls;

    const auto selection_start = std::chrono::steady_clock::now();
    CandidateObservation observation;
    SCIP_RETCODE return_code = SCIP_OKAY;
    try {
        return_code = extractCandidateObservation(scip, observation);
    } catch (const std::exception& exception) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, -1, 0.0, 0.0, "did_not_run",
            "candidate_extraction_exception:" + std::string(exception.what()));
        return SCIP_OKAY;
    }
    if (return_code != SCIP_OKAY) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, -1, 0.0, 0.0, "did_not_run",
            "candidate_extraction_error");
        return SCIP_OKAY;
    }

    stats_->candidates_seen += static_cast<std::int64_t>(observation.candidates.size());
    if (observation.candidates.empty()) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, -1, 0.0, 0.0, "did_not_run",
            "no_priority_lp_candidates");
        return SCIP_OKAY;
    }
    if (options_.max_depth >= 0 && SCIPgetDepth(scip) > options_.max_depth) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, -1, 0.0, 0.0, "did_not_run",
            "depth_limit:" + options_.fallback);
        return SCIP_OKAY;
    }
    if (static_cast<int>(observation.candidates.size()) < options_.min_candidates) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, -1, 0.0, 0.0, "did_not_run",
            "minimum_candidates:" + options_.fallback);
        return SCIP_OKAY;
    }
    if (!model_runner_.ready()) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, -1, 0.0, 0.0, "did_not_run",
            "model_unavailable:" + model_runner_.error());
        return SCIP_OKAY;
    }

    std::vector<float> q_values;
    const auto inference_start = std::chrono::steady_clock::now();
    try {
        q_values = model_runner_.predict(
            observation.variable_features,
            observation.global_features,
            observation.category_features,
            observation.candidates.size());
    } catch (const std::exception& exception) {
        const double inference_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - inference_start).count();
        stats_->inference_time_total += inference_time;
        stats_->inference_time_max = std::max(stats_->inference_time_max, inference_time);
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, {}, -1, inference_time, 0.0, "did_not_run",
            "inference_exception:" + std::string(exception.what()));
        return SCIP_OKAY;
    }
    const double inference_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - inference_start).count();
    stats_->inference_time_total += inference_time;
    stats_->inference_time_max = std::max(stats_->inference_time_max, inference_time);

    const int selected_index = stableArgmax(observation, q_values);
    const double selection_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - selection_start).count();
    stats_->selection_time_total += selection_time;
    stats_->selection_time_max = std::max(stats_->selection_time_max, selection_time);
    ++stats_->legality_checks;
    if (!isCandidateLegal(observation, selected_index)) {
        ++stats_->illegal_actions;
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, q_values, selected_index, inference_time,
            selection_time, "did_not_run", "illegal_selection");
        return SCIP_OKAY;
    }

    SCIP_NODE* down_child = nullptr;
    SCIP_NODE* equal_child = nullptr;
    SCIP_NODE* up_child = nullptr;
    const BranchCandidate& selected = observation.candidates[selected_index];
    return_code = SCIPbranchVarVal(
        scip,
        selected.variable,
        selected.lp_value,
        &down_child,
        &equal_child,
        &up_child);
    if (return_code != SCIP_OKAY) {
        ++stats_->fallback_count;
        writeLogRow(
            scip, observation, q_values, selected_index, inference_time,
            selection_time, "did_not_run", "branch_api_error");
        return SCIP_OKAY;
    }

    if (down_child != nullptr || equal_child != nullptr || up_child != nullptr) {
        *result = SCIP_BRANCHED;
        ++stats_->decisions;
        writeLogRow(
            scip, observation, q_values, selected_index, inference_time,
            selection_time, "branched", "");
    } else {
        *result = SCIP_REDUCEDDOM;
        writeLogRow(
            scip, observation, q_values, selected_index, inference_time,
            selection_time, "reduced_domain", "");
    }
    return SCIP_OKAY;
}

void RlMlpBranchrule::writeLogRow(
    SCIP* scip,
    const CandidateObservation& observation,
    const std::vector<float>& q_values,
    int selected_index,
    double inference_time,
    double selection_time,
    const char* result,
    const std::string& fallback_reason) {
    if (!log_stream_) {
        return;
    }
    const bool legal = isCandidateLegal(observation, selected_index);
    const BranchCandidate* selected = legal ? &observation.candidates[selected_index] : nullptr;
    SCIP_NODE* current_node = SCIPgetCurrentNode(scip);
    log_stream_
        << event_index_++ << ','
        << (current_node == nullptr ? -1 : SCIPnodeGetNumber(current_node)) << ','
        << SCIPgetDepth(scip) << ','
        << observation.candidates.size() << ','
        << (selected == nullptr ? -1 : selected->candidate_index) << ','
        << (selected == nullptr ? -1 : selected->variable_index) << ','
        << csvEscape(selected == nullptr ? "" : observation.variable_names[selected_index]) << ',';
    if (selected == nullptr) {
        log_stream_ << ",,,";
    } else {
        log_stream_
            << std::setprecision(17) << selected->lp_value << ','
            << selected->fractionality << ','
            << (q_values.size() == observation.candidates.size()
                ? q_values[selected_index]
                : std::numeric_limits<float>::quiet_NaN()) << ',';
    }
    log_stream_
        << std::setprecision(17) << inference_time << ','
        << selection_time << ','
        << (legal ? "true" : "false") << ','
        << result << ','
        << csvEscape(fallback_reason) << '\n';
}

SCIP_RETCODE includeRlMlpBranchrule(
    SCIP* scip,
    const RlMlpOptions& options,
    BranchruleStats* stats) {
    SCIP_CALL(SCIPincludeObjBranchrule(
        scip,
        new RlMlpBranchrule(scip, options, stats),
        TRUE));
    return SCIP_OKAY;
}

}  // namespace rlbranch
