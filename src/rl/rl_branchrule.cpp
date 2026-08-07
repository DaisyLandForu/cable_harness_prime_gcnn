#include "rl/rl_branchrule.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include <scip/pub_tree.h>
#include <scip/pub_var.h>
#include <scip/scip_branch.h>
#include <scip/scip_tree.h>

namespace rlbranch {
namespace {

constexpr int kBranchrulePriority = 1000000;

std::string csvEscape(const char* value) {
    std::string input = value == nullptr ? "" : value;
    if (input.find_first_of(",\"\n\r") == std::string::npos) {
        return input;
    }
    std::string escaped = "\"";
    for (char character : input) {
        if (character == '"') {
            escaped += "\"\"";
        } else {
            escaped += character;
        }
    }
    escaped += '"';
    return escaped;
}

bool isCandidateLegal(
    const std::vector<BranchCandidate>& candidates,
    int selected_index) {
    if (selected_index < 0 || selected_index >= static_cast<int>(candidates.size())) {
        return false;
    }
    SCIP_VAR* selected_variable = candidates[selected_index].variable;
    return selected_variable != nullptr
        && std::any_of(candidates.begin(), candidates.end(), [selected_variable](const BranchCandidate& candidate) {
            return candidate.variable == selected_variable;
        });
}

}  // namespace

int selectRandomCandidate(
    const std::vector<BranchCandidate>& candidates,
    std::mt19937& random_engine) {
    if (candidates.empty()) {
        return -1;
    }
    std::uniform_int_distribution<int> distribution(0, static_cast<int>(candidates.size()) - 1);
    return distribution(random_engine);
}

int selectMostInfeasibleCandidate(
    SCIP* scip,
    const std::vector<BranchCandidate>& candidates) {
    int best_index = -1;
    SCIP_Real best_score = SCIP_REAL_MIN;
    SCIP_Real best_objective = 0.0;

    for (int index = 0; index < static_cast<int>(candidates.size()); ++index) {
        const BranchCandidate& candidate = candidates[index];
        const SCIP_Real infeasibility = std::min(
            candidate.fractionality,
            1.0 - candidate.fractionality);
        const SCIP_Real score = infeasibility * SCIPvarGetBranchFactor(candidate.variable);
        const SCIP_Real objective = std::abs(SCIPvarGetObj(candidate.variable));
        if (SCIPisGT(scip, score, best_score)
            || (SCIPisGE(scip, score, best_score) && objective > best_objective)) {
            best_score = score;
            best_objective = objective;
            best_index = index;
        }
    }

    return best_index;
}

const char* branchruleName(CustomBranchingStrategy strategy) {
    return strategy == CustomBranchingStrategy::Random
        ? "rlcustomrandom"
        : "rlcustommostinf";
}

CustomBranchrule::CustomBranchrule(
    SCIP* scip,
    CustomBranchingStrategy strategy,
    unsigned int seed,
    const std::string& log_path,
    BranchruleStats* stats)
    : ObjBranchrule(
          scip,
          branchruleName(strategy),
          "candidate-safe custom branching rule for RL integration validation",
          kBranchrulePriority,
          -1,
          1.0),
      strategy_(strategy),
      random_engine_(seed),
      stats_(stats) {
    if (!log_path.empty()) {
        const std::filesystem::path parent = std::filesystem::path(log_path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        log_stream_.open(log_path);
        if (log_stream_) {
            log_stream_
                << "event_index,node_id,depth,candidate_count,selected_candidate_index,"
                << "selected_variable_index,selected_variable_name,lp_value,fractionality,"
                << "selection_time_seconds,selected_is_candidate,result,fallback_reason\n";
        }
    }
}

SCIP_DECL_BRANCHEXECLP(CustomBranchrule::scip_execlp) {
    (void)branchrule;
    (void)allowaddcons;
    *result = SCIP_DIDNOTRUN;
    ++stats_->lp_calls;

    const auto selection_start = std::chrono::steady_clock::now();
    std::vector<BranchCandidate> candidates;
    SCIP_RETCODE return_code = SCIP_OKAY;
    try {
        return_code = extractLpBranchCandidates(scip, candidates);
    } catch (...) {
        ++stats_->fallback_count;
        writeLogRow(scip, candidates, -1, 0.0, "did_not_run", "candidate_extraction_exception");
        return SCIP_OKAY;
    }
    if (return_code != SCIP_OKAY) {
        ++stats_->fallback_count;
        writeLogRow(scip, candidates, -1, 0.0, "did_not_run", "candidate_extraction_error");
        return SCIP_OKAY;
    }

    stats_->candidates_seen += static_cast<std::int64_t>(candidates.size());
    if (candidates.empty()) {
        ++stats_->fallback_count;
        writeLogRow(scip, candidates, -1, 0.0, "did_not_run", "no_priority_lp_candidates");
        return SCIP_OKAY;
    }

    int selected_index = -1;
    try {
        selected_index = strategy_ == CustomBranchingStrategy::Random
            ? selectRandomCandidate(candidates, random_engine_)
            : selectMostInfeasibleCandidate(scip, candidates);
    } catch (...) {
        ++stats_->fallback_count;
        writeLogRow(scip, candidates, -1, 0.0, "did_not_run", "selection_exception");
        return SCIP_OKAY;
    }

    const double selection_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - selection_start).count();
    stats_->selection_time_total += selection_time;
    stats_->selection_time_max = std::max(stats_->selection_time_max, selection_time);
    ++stats_->legality_checks;

    if (!isCandidateLegal(candidates, selected_index)) {
        ++stats_->illegal_actions;
        ++stats_->fallback_count;
        writeLogRow(scip, candidates, selected_index, selection_time, "did_not_run", "illegal_selection");
        return SCIP_OKAY;
    }

    SCIP_NODE* down_child = nullptr;
    SCIP_NODE* equal_child = nullptr;
    SCIP_NODE* up_child = nullptr;
    const BranchCandidate& selected = candidates[selected_index];
    return_code = SCIPbranchVarVal(
        scip,
        selected.variable,
        selected.lp_value,
        &down_child,
        &equal_child,
        &up_child);
    if (return_code != SCIP_OKAY) {
        ++stats_->fallback_count;
        writeLogRow(scip, candidates, selected_index, selection_time, "did_not_run", "branch_api_error");
        return SCIP_OKAY;
    }

    if (down_child != nullptr || equal_child != nullptr || up_child != nullptr) {
        *result = SCIP_BRANCHED;
        ++stats_->decisions;
        writeLogRow(scip, candidates, selected_index, selection_time, "branched", "");
    } else {
        *result = SCIP_REDUCEDDOM;
        writeLogRow(scip, candidates, selected_index, selection_time, "reduced_domain", "");
    }
    return SCIP_OKAY;
}

void CustomBranchrule::writeLogRow(
    SCIP* scip,
    const std::vector<BranchCandidate>& candidates,
    int selected_index,
    double selection_time,
    const char* result,
    const std::string& fallback_reason) {
    if (!log_stream_) {
        return;
    }

    const bool legal = isCandidateLegal(candidates, selected_index);
    const BranchCandidate* selected = legal ? &candidates[selected_index] : nullptr;
    SCIP_NODE* current_node = SCIPgetCurrentNode(scip);
    log_stream_
        << event_index_++ << ','
        << (current_node == nullptr ? -1 : SCIPnodeGetNumber(current_node)) << ','
        << SCIPgetDepth(scip) << ','
        << candidates.size() << ','
        << (selected == nullptr ? -1 : selected->candidate_index) << ','
        << (selected == nullptr ? -1 : selected->variable_index) << ','
        << csvEscape(selected == nullptr ? "" : SCIPvarGetName(selected->variable)) << ',';
    if (selected == nullptr) {
        log_stream_ << ",,";
    } else {
        log_stream_ << std::setprecision(17) << selected->lp_value << ',' << selected->fractionality << ',';
    }
    log_stream_
        << std::setprecision(17) << selection_time << ','
        << (legal ? "true" : "false") << ','
        << result << ','
        << csvEscape(fallback_reason.c_str()) << '\n';
}

SCIP_RETCODE includeCustomBranchrule(
    SCIP* scip,
    CustomBranchingStrategy strategy,
    unsigned int seed,
    const std::string& log_path,
    BranchruleStats* stats) {
    SCIP_CALL(SCIPincludeObjBranchrule(
        scip,
        new CustomBranchrule(scip, strategy, seed, log_path, stats),
        TRUE));
    return SCIP_OKAY;
}

}  // namespace rlbranch
