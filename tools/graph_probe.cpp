#include "rl/graph_size.hpp"
#include "rl/scip_graph_feature_extractor.hpp"
#include "rl/scip_profile.hpp"

#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <scip/pub_branch.h>
#include <scip/scip_branch.h>
#include <scip/scip_numerics.h>
#include <scip/scip_param.h>
#include <scip/scip_prob.h>
#include <scip/scip_solvingstats.h>
#include <scip/scip_timing.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

struct ProbeOptions {
    std::string instance;
    std::string profile = "configs/scip/project-production-v1.set";
    std::string output = "results/probes/real04_first_branch.json";
    int seed = 0;
    int chunk_size = 64;
    double time_limit = 3600.0;
    double full_limit_gib = 8.0;
    SCIP_Longint node_limit = -1;
    int randomseedshift = 0;
    int permutationseed = 0;
    int lpseed = 0;
    bool dump_effective_only = false;
    bool dump_state = false;
};

struct SCIP_BranchruleData {
    ProbeOptions options;
    rlbranch::ScipProfile profile;
    rlbranch::GraphSizeReport report;
    std::string node_selector;
    std::string param_dump;
    std::string param_dump_sha256;
    std::string applied_profile_dump_sha256;
    std::string effective_dump;
    std::string effective_search_params_sha256;
    std::string effective_search_params_core_sha256;
    std::string binary_sha256;
    bool reached_first_branch = false;
    bool wrote_output = false;
};

namespace {

std::string jsonEscape(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (char character : text) {
        if (character == '"' || character == '\\') {
            escaped.push_back('\\');
            escaped.push_back(character);
        } else if (character == '\n') {
            escaped += "\\n";
        } else {
            escaped.push_back(character);
        }
    }
    return escaped;
}

void writeJsonNumber(std::ostream& out, double value) {
    if (std::isfinite(value)) {
        out << std::setprecision(17) << value;
    } else {
        out << "null";
    }
}

void writeSizeBlock(
    std::ostream& out,
    const std::string& prefix,
    const rlbranch::GraphSizeStats& stats,
    bool include_candidate_count) {
    if (include_candidate_count) {
        out << "  \"" << prefix << "candidate_count\": " << stats.candidate_count << ",\n";
    }
    out << "  \"" << prefix << "variable_count\": " << stats.variable_count << ",\n";
    out << "  \"" << prefix << "row_count\": " << stats.row_count << ",\n";
    out << "  \"" << prefix << "edge_count\": " << stats.edge_count << ",\n";
    out << "  \"" << prefix << "estimated_bytes\": " << stats.estimated_bytes << ",\n";
    out << "  \"" << prefix << "estimated_cuda_bytes\": " << stats.estimated_cuda_bytes << ",\n";
    out << "  \"" << prefix << "extract_seconds\": ";
    writeJsonNumber(out, stats.extract_seconds);
    out << ",\n";
    out << "  \"" << prefix << "peak_rss_bytes\": " << stats.peak_rss_bytes << ",\n";
}

void writeProbeJson(SCIP* scip, SCIP_BRANCHRULEDATA* data, const std::string& status) {
    const auto parent = std::filesystem::path(data->options.output).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(data->options.output);
    if (!out) {
        throw std::runtime_error("cannot write probe JSON: " + data->options.output);
    }

    std::int64_t chunk_p50 = 0;
    std::int64_t chunk_p95 = 0;
    std::int64_t chunk_max = 0;
    if (!data->report.chunks.empty()) {
        std::vector<std::int64_t> bytes;
        bytes.reserve(data->report.chunks.size());
        for (const auto& chunk : data->report.chunks) {
            bytes.push_back(chunk.estimated_bytes);
            chunk_max = std::max(chunk_max, chunk.estimated_bytes);
        }
        std::sort(bytes.begin(), bytes.end());
        chunk_p50 = bytes[(bytes.size() - 1) / 2];
        chunk_p95 = bytes[static_cast<std::size_t>(
            std::min(bytes.size() - 1, (bytes.size() * 95 + 99) / 100))];
    }

    const bool twohop_under_512mib = data->report.twohop.estimated_bytes <= (512LL * 1024 * 1024);
    const bool twohop_much_smaller =
        data->report.full.estimated_bytes == 0
        || data->report.twohop.estimated_bytes * 2 < data->report.full.estimated_bytes;
    const bool chunking_required =
        !twohop_under_512mib || data->report.twohop.estimated_bytes > (1024LL * 1024 * 1024);

    out << "{\n";
    out << "  \"instance\": \"" << data->options.instance << "\",\n";
    out << "  \"instance_sha256\": \""
        << (data->options.instance.empty() ? "" : rlbranch::sha256File(data->options.instance))
        << "\",\n";
    out << "  \"scip_profile\": \"" << data->options.profile << "\",\n";
    out << "  \"profile_sha256\": \"" << data->profile.file_sha256 << "\",\n";
    out << "  \"applied_profile_dump_sha256\": \"" << data->applied_profile_dump_sha256 << "\",\n";
    out << "  \"effective_search_params_sha256\": \""
        << data->effective_search_params_sha256 << "\",\n";
    out << "  \"effective_search_params_core_sha256\": \""
        << data->effective_search_params_core_sha256 << "\",\n";
    out << "  \"param_dump_sha256\": \"" << data->param_dump_sha256 << "\",\n";
    out << "  \"binary_sha256\": \"" << data->binary_sha256 << "\",\n";
    out << "  \"scip_version\": \"" << SCIPmajorVersion() << "." << SCIPminorVersion()
        << "." << SCIPtechVersion() << "\",\n";
    out << "  \"seed\": " << data->options.seed << ",\n";
    out << "  \"status\": \"" << status << "\",\n";
    out << "  \"reached_first_branch\": " << (data->reached_first_branch ? "true" : "false") << ",\n";
    out << "  \"active_node_selector\": \"" << data->node_selector << "\",\n";
    out << "  \"candidate_count\": " << data->report.full.candidate_count << ",\n";
    writeSizeBlock(out, "full_", data->report.full, false);
    writeSizeBlock(out, "twohop_", data->report.twohop, false);
    out << "  \"full_materialization_skipped\": "
        << (data->report.full_materialization_skipped ? "true" : "false") << ",\n";
    out << "  \"full_skip_reason\": \"" << data->report.skip_reason << "\",\n";
    out << "  \"chunk_size\": " << data->options.chunk_size << ",\n";
    out << "  \"chunk_count\": " << data->report.chunks.size() << ",\n";
    out << "  \"chunk_p50_bytes\": " << chunk_p50 << ",\n";
    out << "  \"chunk_p95_bytes\": " << chunk_p95 << ",\n";
    out << "  \"chunk_max_bytes\": " << chunk_max << ",\n";
    out << "  \"available_memory_bytes\": " << rlbranch::availableMemoryBytes() << ",\n";
    out << "  \"host_rss_bytes\": " << rlbranch::currentRssBytes() << ",\n";
    out << "  \"twohop_under_512mib\": " << (twohop_under_512mib ? "true" : "false") << ",\n";
    out << "  \"twohop_much_smaller_than_full\": "
        << (twohop_much_smaller ? "true" : "false") << ",\n";
    out << "  \"chunking_required\": " << (chunking_required ? "true" : "false") << ",\n";
    out << "  \"recommend_twohop_official\": "
        << ((twohop_under_512mib || twohop_much_smaller) ? "true" : "false") << ",\n";
    out << "  \"solving_time\": ";
    writeJsonNumber(out, SCIPgetSolvingTime(scip));
    out << ",\n";
    out << "  \"nodes\": " << SCIPgetNNodes(scip) << ",\n";
    out << "  \"lp_iterations\": " << SCIPgetNLPIterations(scip) << "\n";
    out << "}\n";
    const auto dump_path = std::filesystem::path(data->options.output).replace_extension(".params.txt");
    std::ofstream dump(dump_path);
    dump << data->param_dump;
    const auto effective_path =
        std::filesystem::path(data->options.output).replace_extension(".effective.txt");
    std::ofstream effective(effective_path);
    effective << data->effective_dump;
    data->wrote_output = true;
}

void writeFloatArray(std::ostream& out, const std::vector<float>& values) {
    out << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) {
            out << ", ";
        }
        writeJsonNumber(out, values[index]);
    }
    out << "]";
}

void writeIntArray(std::ostream& out, const std::vector<std::int64_t>& values) {
    out << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) {
            out << ", ";
        }
        out << values[index];
    }
    out << "]";
}

void writeStringArray(std::ostream& out, const std::vector<std::string>& values) {
    out << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) {
            out << ", ";
        }
        out << "\"" << jsonEscape(values[index]) << "\"";
    }
    out << "]";
}

void writeGraphObservation(std::ostream& out, const rlbranch::GraphObservation& observation) {
    out << "{\n";
    out << "    \"row_count\": " << observation.row_count << ",\n";
    out << "    \"variable_count\": " << observation.variable_count << ",\n";
    out << "    \"edge_count\": " << observation.edge_row_indices.size() << ",\n";
    out << "    \"candidate_count\": " << observation.candidate_indices.size() << ",\n";
    out << "    \"variable_feature_dim\": " << observation.variable_feature_dim << ",\n";
    out << "    \"candidate_names\": ";
    writeStringArray(out, observation.candidate_names);
    out << ",\n    \"variable_names\": ";
    writeStringArray(out, observation.variable_names);
    out << ",\n    \"candidate_indices\": ";
    writeIntArray(out, observation.candidate_indices);
    out << ",\n    \"local_lower_bounds\": ";
    writeFloatArray(out, observation.local_lower_bounds);
    out << ",\n    \"variable_features\": ";
    writeFloatArray(out, observation.variable_features);
    out << ",\n    \"row_features\": ";
    writeFloatArray(out, observation.row_features);
    out << ",\n    \"edge_features\": ";
    writeFloatArray(out, observation.edge_features);
    out << ",\n    \"edge_row_indices\": ";
    writeIntArray(out, observation.edge_row_indices);
    out << ",\n    \"edge_variable_indices\": ";
    writeIntArray(out, observation.edge_variable_indices);
    out << ",\n    \"variable_categories\": ";
    writeFloatArray(out, observation.variable_categories);
    out << ",\n    \"row_categories\": ";
    writeFloatArray(out, observation.row_categories);
    out << ",\n    \"global_features\": [";
    for (std::size_t index = 0; index < observation.global_features.size(); ++index) {
        if (index) {
            out << ", ";
        }
        writeJsonNumber(out, observation.global_features[index]);
    }
    out << "]\n  }";
}

void writeStateDump(SCIP* scip, SCIP_BRANCHRULEDATA* data) {
    rlbranch::GraphObservation full;
    rlbranch::GraphObservation twohop;
    SCIP_CALL_ABORT(rlbranch::extractGraphObservation(scip, full, /*twohop=*/false));
    SCIP_CALL_ABORT(rlbranch::extractGraphObservation(scip, twohop, /*twohop=*/true));
    const auto path = std::filesystem::path(data->options.output);
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out << "{\n";
    out << "  \"instance\": \"" << jsonEscape(data->options.instance) << "\",\n";
    out << "  \"applied_profile_dump_sha256\": \"" << data->applied_profile_dump_sha256 << "\",\n";
    out << "  \"effective_search_params_sha256\": \""
        << data->effective_search_params_sha256 << "\",\n";
    out << "  \"effective_search_params_core_sha256\": \""
        << data->effective_search_params_core_sha256 << "\",\n";
    out << "  \"effective_dump\": \"" << jsonEscape(data->effective_dump) << "\",\n";
    out << "  \"active_node_selector\": \"" << jsonEscape(data->node_selector) << "\",\n";
    out << "  \"randomization/randomseedshift\": " << data->options.randomseedshift << ",\n";
    out << "  \"randomization/permutationseed\": " << data->options.permutationseed << ",\n";
    out << "  \"randomization/lpseed\": " << data->options.lpseed << ",\n";
    out << "  \"limits/time\": ";
    writeJsonNumber(out, data->options.time_limit);
    out << ",\n  \"limits/nodes\": " << data->options.node_limit << ",\n";
    out << "  \"lp_iterations\": " << SCIPgetNLPIterations(scip) << ",\n";
    out << "  \"primal_bound\": ";
    writeJsonNumber(out, SCIPgetPrimalbound(scip));
    out << ",\n  \"dual_bound\": ";
    writeJsonNumber(out, SCIPgetDualbound(scip));
    out << ",\n  \"full\": ";
    writeGraphObservation(out, full);
    out << ",\n  \"twohop\": ";
    writeGraphObservation(out, twohop);
    out << "\n}\n";
    data->wrote_output = true;
}

SCIP_DECL_BRANCHEXECLP(branchExeclpGraphProbe) {
    (void)allowaddcons;
    SCIP_BRANCHRULEDATA* data = SCIPbranchruleGetData(branchrule);
    *result = SCIP_DIDNOTRUN;
    if (data == nullptr || data->reached_first_branch) {
        return SCIP_OKAY;
    }

    std::vector<int> candidates;
    SCIP_CALL(rlbranch::collectLpCandidateIndices(scip, candidates));
    if (candidates.empty()) {
        return SCIP_OKAY;
    }

    data->reached_first_branch = true;
    rlbranch::requireEstimateNodeSelector(scip);
    data->node_selector = rlbranch::activeNodeSelectorName(scip);
    const std::int64_t limit_bytes = static_cast<std::int64_t>(
        data->options.full_limit_gib * 1024.0 * 1024.0 * 1024.0);
    SCIP_CALL(rlbranch::buildFirstBranchGraphSizeReport(
        scip, data->options.chunk_size, limit_bytes, data->report));
    if (data->options.dump_state) {
        writeStateDump(scip, data);
    } else {
        writeProbeJson(scip, data, "first_branch");
    }
    SCIP_CALL(SCIPinterruptSolve(scip));
    return SCIP_OKAY;
}

SCIP_DECL_BRANCHFREE(branchFreeGraphProbe) {
    (void)scip;
    delete SCIPbranchruleGetData(branchrule);
    SCIPbranchruleSetData(branchrule, nullptr);
    return SCIP_OKAY;
}

void printUsage(const char* program) {
    std::cout
        << "Usage:\n  " << program
        << " --instance <cip> [--scip-profile <set>] [--output <json>]"
        << " [--seed 0] [--time-limit 3600] [--chunk-size 64] [--dump-effective]\n";
}

ProbeOptions parseOptions(int argc, char** argv) {
    ProbeOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto require = [&]() -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return argv[index];
        };
        if (arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--instance") {
            options.instance = require();
        } else if (arg == "--scip-profile") {
            options.profile = require();
        } else if (arg == "--output") {
            options.output = require();
        } else if (arg == "--seed") {
            options.seed = std::stoi(require());
            options.randomseedshift = options.seed;
            options.permutationseed = options.seed;
            options.lpseed = options.seed;
        } else if (arg == "--time-limit") {
            options.time_limit = std::stod(require());
        } else if (arg == "--chunk-size") {
            options.chunk_size = std::stoi(require());
        } else if (arg == "--threads") {
            if (std::stoi(require()) != 1) {
                throw std::invalid_argument("graph_probe requires --threads 1");
            }
        } else if (arg == "--dump-effective") {
            options.dump_effective_only = true;
        } else if (arg == "--dump-state") {
            options.dump_state = true;
        } else if (arg == "--node-limit") {
            options.node_limit = std::stoll(require());
        } else if (arg == "--randomseedshift") {
            options.randomseedshift = std::stoi(require());
        } else if (arg == "--permutationseed") {
            options.permutationseed = std::stoi(require());
        } else if (arg == "--lpseed") {
            options.lpseed = std::stoi(require());
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.instance.empty() && !options.dump_effective_only) {
        throw std::invalid_argument("--instance is required");
    }
    if (options.chunk_size <= 0 || options.time_limit <= 0.0 || options.seed < 0) {
        throw std::invalid_argument("numeric options are outside their valid range");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        ProbeOptions options = parseOptions(argc, argv);
        if (!options.dump_effective_only && !std::filesystem::exists(options.instance)) {
            throw std::runtime_error("instance not found: " + options.instance);
        }
        if (!std::filesystem::exists(options.profile)) {
            throw std::runtime_error("SCIP profile not found: " + options.profile);
        }

        SCIP* scip = nullptr;
        SCIP_CALL(SCIPcreate(&scip));
        SCIP_CALL(SCIPincludeDefaultPlugins(scip));

        auto* data = new SCIP_BRANCHRULEDATA();
        data->options = options;
        data->profile = rlbranch::loadScipProfile(options.profile);
        data->binary_sha256 = rlbranch::sha256File(argv[0]);

        // Ecole reads the instance first, then overlays permutation flags/seeds.
        // Setting those before SCIPreadProb permutes the original problem instead
        // of the transformed problem and diverges from Ecole's first branch.
        if (!options.dump_effective_only) {
            SCIP_CALL(SCIPreadProb(scip, options.instance.c_str(), nullptr));
        }

        SCIP_CALL(rlbranch::applyScipProfile(scip, data->profile));
        SCIP_CALL(SCIPsetRealParam(scip, "limits/time", options.time_limit));
        SCIP_CALL(SCIPsetLongintParam(scip, "limits/nodes", options.node_limit));
        SCIP_CALL(SCIPsetIntParam(scip, "randomization/randomseedshift", options.randomseedshift));
        SCIP_CALL(SCIPsetIntParam(scip, "randomization/permutationseed", options.permutationseed));
        SCIP_CALL(SCIPsetIntParam(scip, "randomization/lpseed", options.lpseed));
        rlbranch::assertProductionInvariants(scip);
        data->param_dump = rlbranch::dumpAppliedProfile(scip, data->profile);
        data->param_dump_sha256 = rlbranch::sha256Text(data->param_dump);
        data->applied_profile_dump_sha256 = data->param_dump_sha256;
        data->effective_dump = rlbranch::dumpEffectiveSearchParams(scip);
        data->effective_search_params_sha256 = rlbranch::sha256Text(data->effective_dump);
        data->effective_search_params_core_sha256 = rlbranch::sha256Text(
            rlbranch::dumpEffectiveSearchParams(scip, false));

        SCIP_BRANCHRULE* branchrule = nullptr;
        SCIP_CALL(SCIPincludeBranchruleBasic(
            scip,
            &branchrule,
            "graphprobe",
            "stop at first LP branching state and measure graph size",
            1000000,
            -1,
            1.0,
            data));
        SCIP_CALL(SCIPsetBranchruleExecLp(scip, branchrule, branchExeclpGraphProbe));
        SCIP_CALL(SCIPsetBranchruleFree(scip, branchrule, branchFreeGraphProbe));

        if (options.dump_effective_only) {
            if (data->options.output.empty()) {
                data->options.output = "results/probes/effective_search_params.json";
            }
            SCIP_CALL(SCIPcreateProbBasic(scip, "effective_dump"));
            rlbranch::requireEstimateNodeSelector(scip);
            data->node_selector = rlbranch::activeNodeSelectorName(scip);
            writeProbeJson(scip, data, "effective_dump");
            SCIP_CALL(SCIPfree(&scip));
            return 0;
        }

        SCIP_CALL(SCIPsolve(scip));

        if (!data->wrote_output) {
            data->node_selector = rlbranch::activeNodeSelectorName(scip);
            writeProbeJson(scip, data, "no_branch_before_stop");
        }

        const int exit_code = data->reached_first_branch ? 0 : 2;
        SCIP_CALL(SCIPfree(&scip));
        return exit_code;
    } catch (const std::exception& error) {
        std::cerr << "graph_probe failed: " << error.what() << std::endl;
        return 1;
    }
}
