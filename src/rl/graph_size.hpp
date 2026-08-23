#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <scip/scip.h>

namespace rlbranch {

struct GraphSizeStats {
    int candidate_count = 0;
    std::int64_t variable_count = 0;
    std::int64_t row_count = 0;
    std::int64_t edge_count = 0;
    std::int64_t estimated_bytes = 0;
    std::int64_t estimated_cuda_bytes = 0;
    double extract_seconds = 0.0;
    std::int64_t peak_rss_bytes = 0;
};

struct GraphSizeReport {
    GraphSizeStats full;
    GraphSizeStats twohop;
    std::vector<GraphSizeStats> chunks;
    bool full_materialization_skipped = false;
    std::string skip_reason;
};

std::int64_t estimateGraphTensorBytes(
    std::int64_t variable_count,
    std::int64_t row_count,
    std::int64_t edge_count);

std::int64_t currentRssBytes();
std::int64_t availableMemoryBytes();

SCIP_RETCODE countExpandedGraph(
    SCIP* scip,
    const std::vector<int>* candidate_variable_indices,
    GraphSizeStats& stats);

SCIP_RETCODE collectLpCandidateIndices(
    SCIP* scip,
    std::vector<int>& candidate_indices);

SCIP_RETCODE buildFirstBranchGraphSizeReport(
    SCIP* scip,
    int chunk_size,
    std::int64_t full_materialize_limit_bytes,
    GraphSizeReport& report);

}  // namespace rlbranch
