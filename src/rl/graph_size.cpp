#include "rl/graph_size.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include <scip/pub_lp.h>
#include <scip/pub_var.h>
#include <scip/scip_lp.h>
#include <scip/scip_mem.h>
#include <scip/scip_numerics.h>
#include <scip/scip_solvingstats.h>
#include <scip/scip_var.h>
#include <scip/scip_branch.h>

namespace rlbranch {
namespace {

constexpr int kVariableNumeric = 25;
constexpr int kVariableCategory = 6;
constexpr int kRowNumeric = 14;
constexpr int kRowCategory = 6;
constexpr int kEdgeFeature = 3;
constexpr int kGlobal = 14;

bool sideIsFinite(SCIP* scip, double bound) {
    return !SCIPisInfinity(scip, std::abs(bound));
}

int expandedSideCount(SCIP* scip, SCIP_ROW* row) {
    return (sideIsFinite(scip, SCIProwGetLhs(row)) ? 1 : 0)
        + (sideIsFinite(scip, SCIProwGetRhs(row)) ? 1 : 0);
}

std::int64_t parseMeminfoBytes(const std::string& key) {
    std::ifstream stream("/proc/meminfo");
    std::string name;
    long long value = 0;
    std::string unit;
    while (stream >> name >> value >> unit) {
        if (name == key) {
            return static_cast<std::int64_t>(value) * 1024;
        }
    }
    return 0;
}

}  // namespace

std::int64_t estimateGraphTensorBytes(
    std::int64_t variable_count,
    std::int64_t row_count,
    std::int64_t edge_count) {
    const std::int64_t variable_bytes =
        variable_count * (kVariableNumeric + kVariableCategory) * 4;
    const std::int64_t row_bytes = row_count * (kRowNumeric + kRowCategory) * 4;
    const std::int64_t edge_feature_bytes = edge_count * kEdgeFeature * 4;
    const std::int64_t edge_index_bytes = edge_count * 2 * 8;
    const std::int64_t global_bytes = kGlobal * 4;
    return variable_bytes + row_bytes + edge_feature_bytes + edge_index_bytes + global_bytes;
}

std::int64_t currentRssBytes() {
    std::ifstream stream("/proc/self/status");
    std::string line;
    while (std::getline(stream, line)) {
        if (line.rfind("VmRSS:", 0) != 0) {
            continue;
        }
        std::istringstream parser(line);
        std::string key;
        long long value = 0;
        std::string unit;
        parser >> key >> value >> unit;
        return static_cast<std::int64_t>(value) * 1024;
    }
    return 0;
}

std::int64_t availableMemoryBytes() {
    return parseMeminfoBytes("MemAvailable:");
}

SCIP_RETCODE collectLpCandidateIndices(
    SCIP* scip,
    std::vector<int>& candidate_indices) {
    candidate_indices.clear();
    SCIP_VAR** lpcands = nullptr;
    SCIP_Real* lpcandssol = nullptr;
    SCIP_Real* lpcandsfrac = nullptr;
    int nlpcands = 0;
    int npriocands = 0;
    int nfrac = 0;
    SCIP_CALL(SCIPgetLPBranchCands(
        scip, &lpcands, &lpcandssol, &lpcandsfrac, &nlpcands, &npriocands, &nfrac));
    candidate_indices.reserve(static_cast<std::size_t>(std::max(npriocands, 0)));
    const int count = npriocands > 0 ? npriocands : nlpcands;
    for (int index = 0; index < count; ++index) {
        if (lpcands[index] == nullptr) {
            continue;
        }
        const int variable_index = SCIPvarGetProbindex(lpcands[index]);
        if (variable_index >= 0) {
            candidate_indices.push_back(variable_index);
        }
    }
    return SCIP_OKAY;
}

SCIP_RETCODE countExpandedGraph(
    SCIP* scip,
    const std::vector<int>* candidate_variable_indices,
    GraphSizeStats& stats) {
    const auto started = std::chrono::steady_clock::now();
    stats = GraphSizeStats{};
    SCIP_ROW** rows = SCIPgetLPRows(scip);
    const int row_count = SCIPgetNLPRows(scip);
    const int variable_count = SCIPgetNVars(scip);
    if (rows == nullptr || row_count < 0 || variable_count < 0) {
        return SCIP_INVALIDDATA;
    }

    std::unordered_set<int> selected_variables;
    std::unordered_set<int> candidate_set;
    if (candidate_variable_indices != nullptr) {
        candidate_set.insert(
            candidate_variable_indices->begin(), candidate_variable_indices->end());
        selected_variables.insert(
            candidate_variable_indices->begin(), candidate_variable_indices->end());
    } else {
        stats.variable_count = variable_count;
    }

    std::int64_t expanded_rows = 0;
    std::int64_t expanded_edges = 0;
    for (int row_position = 0; row_position < row_count; ++row_position) {
        SCIP_ROW* row = rows[row_position];
        SCIP_COL** columns = SCIProwGetCols(row);
        const int nonzeros = SCIProwGetNLPNonz(row);
        bool keep_row = candidate_variable_indices == nullptr;
        if (!keep_row) {
            for (int index = 0; index < nonzeros; ++index) {
                const int variable_index = SCIPvarGetProbindex(SCIPcolGetVar(columns[index]));
                if (candidate_set.count(variable_index) > 0) {
                    keep_row = true;
                    break;
                }
            }
        }
        if (!keep_row) {
            continue;
        }
        const int sides = expandedSideCount(scip, row);
        expanded_rows += sides;
        expanded_edges += static_cast<std::int64_t>(sides) * nonzeros;
        if (candidate_variable_indices != nullptr) {
            for (int index = 0; index < nonzeros; ++index) {
                const int variable_index = SCIPvarGetProbindex(SCIPcolGetVar(columns[index]));
                if (variable_index >= 0) {
                    selected_variables.insert(variable_index);
                }
            }
        }
    }

    if (candidate_variable_indices != nullptr) {
        stats.variable_count = static_cast<std::int64_t>(selected_variables.size());
        stats.candidate_count = static_cast<int>(candidate_variable_indices->size());
    } else {
        stats.candidate_count = 0;
    }
    stats.row_count = expanded_rows;
    stats.edge_count = expanded_edges;
    stats.estimated_bytes = estimateGraphTensorBytes(
        stats.variable_count, stats.row_count, stats.edge_count);
    stats.estimated_cuda_bytes = stats.estimated_bytes;
    stats.extract_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    stats.peak_rss_bytes = currentRssBytes();
    return SCIP_OKAY;
}

SCIP_RETCODE buildFirstBranchGraphSizeReport(
    SCIP* scip,
    int chunk_size,
    std::int64_t full_materialize_limit_bytes,
    GraphSizeReport& report) {
    report = GraphSizeReport{};
    std::vector<int> candidates;
    SCIP_CALL(collectLpCandidateIndices(scip, candidates));
    report.full.candidate_count = static_cast<int>(candidates.size());
    SCIP_CALL(countExpandedGraph(scip, nullptr, report.full));
    report.full.candidate_count = static_cast<int>(candidates.size());

    const std::int64_t available = availableMemoryBytes();
    const std::int64_t available_cap =
        available > 0 ? available / 2 : full_materialize_limit_bytes;
    if (report.full.estimated_bytes > full_materialize_limit_bytes
        || (available > 0 && report.full.estimated_bytes > available_cap)) {
        report.full_materialization_skipped = true;
        std::ostringstream reason;
        reason << "full graph estimate " << report.full.estimated_bytes
               << " bytes exceeds materialization budget";
        report.skip_reason = reason.str();
    }

    SCIP_CALL(countExpandedGraph(scip, &candidates, report.twohop));
    if (chunk_size > 0 && !candidates.empty()) {
        for (std::size_t begin = 0; begin < candidates.size();
             begin += static_cast<std::size_t>(chunk_size)) {
            const std::size_t end = std::min(
                begin + static_cast<std::size_t>(chunk_size), candidates.size());
            const std::vector<int> chunk(candidates.begin() + static_cast<std::ptrdiff_t>(begin),
                candidates.begin() + static_cast<std::ptrdiff_t>(end));
            GraphSizeStats chunk_stats;
            SCIP_CALL(countExpandedGraph(scip, &chunk, chunk_stats));
            report.chunks.push_back(chunk_stats);
        }
    }
    return SCIP_OKAY;
}

}  // namespace rlbranch
