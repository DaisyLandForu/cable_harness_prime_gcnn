#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <set>
#include <queue>
#include <ctime>
#include <iomanip>
#include <optional>
#include <unordered_set>
#include <stack>
#include <cmath> 
#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <scip/pub_branch.h>
#include <scip/struct_scip.h>
#include <scip/struct_stat.h>

#include "rl/rl_branchrule.hpp"
#include "rl/rl_gcnn_branchrule.hpp"
#include "rl/rl_mlp_branchrule.hpp"
#include "rl/scip_profile.hpp"

using namespace std;

int copy_num_dflt = 4;
int copy_num = copy_num_dflt;
string scale = "1.0";
char center_prefix = 'N';
int thread_num = 16;
int div_part = 1;
double gap = 0.0;

struct RunConfig {
    std::string instance_id = "1";
    std::string edges_path = "data/edges-1.csv";
    std::string pairs_path = "data/pairs-1.csv";
    std::string branching = "default";
    std::string protocol = "project-production-v1";
    std::string scip_profile = "configs/scip/project-production-v1.set";
    std::string output_json;
    std::string export_milp;
    std::string branch_log;
    std::string rl_model;
    std::string rl_device = "cpu";
    std::string rl_fallback = "relpscost";
    std::string rl_log;
    int rl_max_depth = -1;
    int rl_min_candidates = 1;
    int seed = 0;
    int randomseedshift = 0;
    int permutationseed = 0;
    int lpseed = 0;
    std::string seed_overlay;
    double time_limit = 1e20;
    SCIP_Longint node_limit = -1;
    int threads = 1;
    bool threads_explicit = false;
    bool build_only = false;
    bool legacy_cli = true;
};

struct SolverMetrics {
    std::string status = "not_run";
    std::string active_branchrule;
    std::string node_selector;
    std::string scip_profile_sha256;
    std::string scip_param_dump_sha256;
    std::string applied_profile_dump_sha256;
    std::string effective_search_params_sha256;
    std::string effective_search_params_core_sha256;
    double objective = std::numeric_limits<double>::quiet_NaN();
    double primal_bound = std::numeric_limits<double>::quiet_NaN();
    double dual_bound = std::numeric_limits<double>::quiet_NaN();
    double final_gap = std::numeric_limits<double>::quiet_NaN();
    double wall_clock_time = 0.0;
    double presolve_time = 0.0;
    double solving_time = 0.0;
    double solve_time_after_presolve = 0.0;
    double primal_dual_integral = std::numeric_limits<double>::quiet_NaN();
    double first_solution_time = std::numeric_limits<double>::quiet_NaN();
    SCIP_Longint nodes = 0;
    SCIP_Longint lp_iterations = 0;
    SCIP_Longint branchrule_calls = 0;
    int branchrule_priority = 0;
    rlbranch::BranchruleStats custom_branching;
    int n_vars = 0;
    int n_integer_vars = 0;
    int n_constraints = 0;
    int n_center_nodes = 0;
    int n_center_edges = 0;
    int n_commodities = 0;
    bool has_solution = false;
    bool solution_feasible = false;
};

std::string resolveInputPath(const std::string& path) {
    if (std::filesystem::exists(path)) {
        return path;
    }
    std::filesystem::path from_root = std::filesystem::path("code") / path;
    if (std::filesystem::exists(from_root)) {
        return from_root.string();
    }
    return path;
}

void ensureParentDirectory(const std::string& path) {
    if (path.empty()) {
        return;
    }
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream escaped;
    for (unsigned char ch : value) {
        switch (ch) {
        case '\\': escaped << "\\\\"; break;
        case '"': escaped << "\\\""; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default:
            if (ch < 0x20) {
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec << std::setfill(' ');
            } else {
                escaped << ch;
            }
        }
    }
    return escaped.str();
}

void writeJsonNumber(std::ostream& out, double value) {
    if (std::isfinite(value) && std::abs(value) < 1e19) {
        out << std::setprecision(15) << value;
    } else {
        out << "null";
    }
}

void writeRunJson(const std::string& path, const RunConfig& config, const SolverMetrics& metrics,
                  double business_objective) {
    if (path.empty()) {
        return;
    }
    ensureParentDirectory(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot open output JSON: " + path);
    }
    out << "{\n";
    out << "  \"instance_id\": \"" << jsonEscape(config.instance_id) << "\",\n";
    out << "  \"edges_path\": \"" << jsonEscape(config.edges_path) << "\",\n";
    out << "  \"pairs_path\": \"" << jsonEscape(config.pairs_path) << "\",\n";
    out << "  \"method\": \"" << jsonEscape(config.branching) << "\",\n";
    out << "  \"protocol\": \"" << jsonEscape(config.protocol) << "\",\n";
    out << "  \"seed\": " << config.seed << ",\n";
    out << "  \"randomization/randomseedshift\": " << config.randomseedshift << ",\n";
    out << "  \"randomization/permutationseed\": " << config.permutationseed << ",\n";
    out << "  \"randomization/lpseed\": " << config.lpseed << ",\n";
    out << "  \"seed_overlay\": \"" << jsonEscape(config.seed_overlay) << "\",\n";
    out << "  \"scip_version\": \"" << SCIPmajorVersion() << "." << SCIPminorVersion()
        << "." << SCIPtechVersion() << "\",\n";
    out << "  \"status\": \"" << jsonEscape(metrics.status) << "\",\n";
    out << "  \"objective\": "; writeJsonNumber(out, metrics.objective); out << ",\n";
    out << "  \"business_objective\": "; writeJsonNumber(out, business_objective); out << ",\n";
    out << "  \"primal_bound\": "; writeJsonNumber(out, metrics.primal_bound); out << ",\n";
    out << "  \"dual_bound\": "; writeJsonNumber(out, metrics.dual_bound); out << ",\n";
    out << "  \"final_gap\": "; writeJsonNumber(out, metrics.final_gap); out << ",\n";
    out << "  \"wall_clock_time\": "; writeJsonNumber(out, metrics.wall_clock_time); out << ",\n";
    out << "  \"presolve_time\": "; writeJsonNumber(out, metrics.presolve_time); out << ",\n";
    out << "  \"solving_time\": "; writeJsonNumber(out, metrics.solving_time); out << ",\n";
    out << "  \"solve_time_after_presolve\": "; writeJsonNumber(out, metrics.solve_time_after_presolve); out << ",\n";
    out << "  \"nodes\": " << metrics.nodes << ",\n";
    out << "  \"lp_iterations\": " << metrics.lp_iterations << ",\n";
    out << "  \"primal_dual_integral\": "; writeJsonNumber(out, metrics.primal_dual_integral); out << ",\n";
    out << "  \"first_solution_time\": "; writeJsonNumber(out, metrics.first_solution_time); out << ",\n";
    out << "  \"number_of_variables\": " << metrics.n_vars << ",\n";
    out << "  \"number_of_integer_variables\": " << metrics.n_integer_vars << ",\n";
    out << "  \"number_of_constraints\": " << metrics.n_constraints << ",\n";
    out << "  \"number_of_center_nodes\": " << metrics.n_center_nodes << ",\n";
    out << "  \"number_of_center_edges\": " << metrics.n_center_edges << ",\n";
    out << "  \"number_of_commodities\": " << metrics.n_commodities << ",\n";
    out << "  \"active_branching_rule\": \"" << jsonEscape(metrics.active_branchrule) << "\",\n";
    out << "  \"branchrule_calls\": " << metrics.branchrule_calls << ",\n";
    out << "  \"branchrule_priority\": " << metrics.branchrule_priority << ",\n";
    out << "  \"custom_branch_lp_calls\": " << metrics.custom_branching.lp_calls << ",\n";
    out << "  \"custom_branch_decisions\": " << metrics.custom_branching.decisions << ",\n";
    out << "  \"custom_candidates_seen\": " << metrics.custom_branching.candidates_seen << ",\n";
    out << "  \"custom_legality_checks\": " << metrics.custom_branching.legality_checks << ",\n";
    out << "  \"custom_illegal_actions\": " << metrics.custom_branching.illegal_actions << ",\n";
    out << "  \"custom_fallback_count\": " << metrics.custom_branching.fallback_count << ",\n";
    out << "  \"custom_selection_time_total\": ";
    writeJsonNumber(out, metrics.custom_branching.selection_time_total); out << ",\n";
    out << "  \"custom_selection_time_mean\": ";
    writeJsonNumber(out, metrics.custom_branching.decisions > 0
        ? metrics.custom_branching.selection_time_total / metrics.custom_branching.decisions
        : 0.0); out << ",\n";
    out << "  \"custom_selection_time_max\": ";
    writeJsonNumber(out, metrics.custom_branching.selection_time_max); out << ",\n";
    out << "  \"branch_decisions\": " << metrics.custom_branching.decisions << ",\n";
    out << "  \"rl_inference_total\": ";
    writeJsonNumber(out, metrics.custom_branching.inference_time_total); out << ",\n";
    out << "  \"rl_inference_mean\": ";
    writeJsonNumber(out, metrics.custom_branching.legality_checks > 0
        ? metrics.custom_branching.inference_time_total / metrics.custom_branching.legality_checks
        : 0.0); out << ",\n";
    out << "  \"rl_inference_max\": ";
    writeJsonNumber(out, metrics.custom_branching.inference_time_max); out << ",\n";
    out << "  \"fallback_count\": " << metrics.custom_branching.fallback_count << ",\n";
    out << "  \"rl_model\": \"" << jsonEscape(config.rl_model) << "\",\n";
    out << "  \"rl_device\": \"" << jsonEscape(config.rl_device) << "\",\n";
    out << "  \"rl_fallback\": \"" << jsonEscape(config.rl_fallback) << "\",\n";
    out << "  \"rl_max_depth\": " << config.rl_max_depth << ",\n";
    out << "  \"rl_min_candidates\": " << config.rl_min_candidates << ",\n";
    out << "  \"scip_profile\": \"" << jsonEscape(config.scip_profile) << "\",\n";
    out << "  \"scip_profile_sha256\": \"" << jsonEscape(metrics.scip_profile_sha256) << "\",\n";
    out << "  \"applied_profile_dump_sha256\": \""
        << jsonEscape(metrics.applied_profile_dump_sha256) << "\",\n";
    out << "  \"effective_search_params_sha256\": \""
        << jsonEscape(metrics.effective_search_params_sha256) << "\",\n";
    out << "  \"effective_search_params_core_sha256\": \""
        << jsonEscape(metrics.effective_search_params_core_sha256) << "\",\n";
    out << "  \"scip_param_dump_sha256\": \"" << jsonEscape(metrics.scip_param_dump_sha256) << "\",\n";
    out << "  \"node_selection_rule\": \"" << jsonEscape(metrics.node_selector) << "\",\n";
    out << "  \"threads\": " << (config.threads_explicit ? config.threads : 1) << ",\n";
    out << "  \"time_limit\": "; writeJsonNumber(out, config.time_limit); out << ",\n";
    out << "  \"node_limit\": " << config.node_limit << ",\n";
    out << "  \"has_solution\": " << (metrics.has_solution ? "true" : "false") << ",\n";
    out << "  \"solution_feasible\": " << (metrics.solution_feasible ? "true" : "false") << "\n";
    out << "}\n";
}

void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " [copy_num] [div_part]\n"
        << "  " << program << " [options]\n\n"
        << "Options:\n"
        << "  --instance-id <1-9>       Use code/data/edges-N.csv and pairs-N.csv\n"
        << "  --edges <path>            Edge CSV path\n"
        << "  --pairs <path>            Pair CSV path\n"
        << "  --copy-num <int>          Number of topology copies (default: 4)\n"
        << "  --div-part <int>          Keep 1/div_part center pairs (default: 1)\n"
        << "  --branching <method>      default|relpscost|random|mostinf|strong|custom-random|custom-mostinf|rl-gcnn\n"
        << "  --scip-profile <path>     Frozen SCIP set file (default: project-production-v1)\n"
        << "  --seed <int>              Default for all three SCIP randomization seeds\n"
        << "  --randomseedshift <int>   Override randomization/randomseedshift\n"
        << "  --permutationseed <int>   Override randomization/permutationseed\n"
        << "  --lpseed <int>            Override randomization/lpseed\n"
        << "  --seed-overlay <path>     Load the remapped seed triple from a .set file\n"
        << "  --time-limit <seconds>    Solver time limit\n"
        << "  --node-limit <int>        Solver node limit (-1 means unlimited)\n"
        << "  --threads <int>           Must be 1 for formal runs; defaults to profile\n"
        << "  --output-json <path>      Write structured run metrics\n"
        << "  --branch-log <path>       Write custom branching decisions as CSV\n"
        << "  --rl-model <path>         TorchScript GCNN model\n"
        << "  --rl-device <device>      cpu|cuda (default: cpu)\n"
        << "  --rl-fallback <method>    relpscost|default\n"
        << "  --rl-max-depth <int>      Use RL through this depth (-1 unlimited)\n"
        << "  --rl-min-candidates <int> Minimum candidates required for RL\n"
        << "  --rl-log <path>           Write RL branch decisions as CSV\n"
        << "  --export-milp <path>      Export original problem as CIP/MPS/LP\n"
        << "  --build-only              Build/export without solving\n"
        << "  --help                    Show this help\n";
}

RunConfig parseArguments(int argc, char* argv[]) {
    RunConfig config;
    if (argc > 1 && argv[1][0] != '-') {
        copy_num = std::stoi(argv[1]);
        if (argc > 2) {
            div_part = std::stoi(argv[2]);
        }
        if (argc > 3) {
            throw std::invalid_argument("legacy mode accepts at most copy_num and div_part");
        }
        config.edges_path = resolveInputPath(config.edges_path);
        config.pairs_path = resolveInputPath(config.pairs_path);
        return config;
    }

    config.legacy_cli = argc == 1;
    bool explicit_edges = false;
    bool explicit_pairs = false;
    bool explicit_shift = false;
    bool explicit_perm = false;
    bool explicit_lp = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&]() -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return argv[i];
        };
        if (arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--instance-id") {
            config.instance_id = requireValue();
        } else if (arg == "--edges") {
            config.edges_path = requireValue();
            explicit_edges = true;
        } else if (arg == "--pairs") {
            config.pairs_path = requireValue();
            explicit_pairs = true;
        } else if (arg == "--copy-num") {
            copy_num = std::stoi(requireValue());
        } else if (arg == "--div-part") {
            div_part = std::stoi(requireValue());
        } else if (arg == "--branching") {
            config.branching = requireValue();
        } else if (arg == "--scip-profile") {
            config.scip_profile = requireValue();
        } else if (arg == "--protocol") {
            throw std::invalid_argument(
                "--protocol has been removed; official runs use --scip-profile "
                "configs/scip/project-production-v1.set");
        } else if (arg == "--seed") {
            config.seed = std::stoi(requireValue());
        } else if (arg == "--randomseedshift") {
            config.randomseedshift = std::stoi(requireValue());
            explicit_shift = true;
        } else if (arg == "--permutationseed") {
            config.permutationseed = std::stoi(requireValue());
            explicit_perm = true;
        } else if (arg == "--lpseed") {
            config.lpseed = std::stoi(requireValue());
            explicit_lp = true;
        } else if (arg == "--seed-overlay") {
            config.seed_overlay = requireValue();
        } else if (arg == "--time-limit") {
            config.time_limit = std::stod(requireValue());
        } else if (arg == "--node-limit") {
            config.node_limit = std::stoll(requireValue());
        } else if (arg == "--threads") {
            config.threads = std::stoi(requireValue());
            config.threads_explicit = true;
        } else if (arg == "--output-json") {
            config.output_json = requireValue();
        } else if (arg == "--branch-log") {
            config.branch_log = requireValue();
        } else if (arg == "--rl-model") {
            config.rl_model = requireValue();
        } else if (arg == "--rl-device") {
            config.rl_device = requireValue();
        } else if (arg == "--rl-fallback") {
            config.rl_fallback = requireValue();
        } else if (arg == "--rl-max-depth") {
            config.rl_max_depth = std::stoi(requireValue());
        } else if (arg == "--rl-min-candidates") {
            config.rl_min_candidates = std::stoi(requireValue());
        } else if (arg == "--rl-log") {
            config.rl_log = requireValue();
        } else if (arg == "--export-milp") {
            config.export_milp = requireValue();
        } else if (arg == "--build-only") {
            config.build_only = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (config.branching == "most-infeasible") {
        config.branching = "mostinf";
    }
    static const std::set<std::string> methods = {
        "default", "relpscost", "random", "mostinf", "strong",
        "custom-random", "custom-mostinf", "rl-gcnn"};
    if (methods.find(config.branching) == methods.end()) {
        throw std::invalid_argument("unsupported branching method: " + config.branching);
    }
    config.randomseedshift = explicit_shift ? config.randomseedshift : config.seed;
    config.permutationseed = explicit_perm ? config.permutationseed : config.seed;
    config.lpseed = explicit_lp ? config.lpseed : config.seed;
    if (!config.seed_overlay.empty()) {
        const auto overlay = rlbranch::loadSeedOverlay(config.seed_overlay);
        if (!explicit_shift) {
            config.randomseedshift = overlay.randomseedshift;
        }
        if (!explicit_perm) {
            config.permutationseed = overlay.permutationseed;
        }
        if (!explicit_lp) {
            config.lpseed = overlay.lpseed;
        }
    }
    if (copy_num <= 0 || div_part <= 0 || config.seed < 0 || config.time_limit <= 0.0
        || config.randomseedshift < 0 || config.permutationseed < 0 || config.lpseed < 0
        || config.node_limit < -1 || (config.threads_explicit && config.threads != 1)
        || config.rl_max_depth < -1 || config.rl_min_candidates <= 0) {
        throw std::invalid_argument("numeric options are outside their valid range");
    }
    if (config.rl_device != "cpu" && config.rl_device != "cuda") {
        throw std::invalid_argument("--rl-device must be cpu or cuda");
    }
    if (config.rl_fallback != "relpscost" && config.rl_fallback != "default") {
        throw std::invalid_argument("--rl-fallback must be relpscost or default");
    }
    if (config.branching == "rl-gcnn" && config.rl_model.empty()) {
        throw std::invalid_argument("--rl-model is required for RL branching");
    }
    if (!explicit_edges) {
        config.edges_path = "data/edges-" + config.instance_id + ".csv";
    }
    if (!explicit_pairs) {
        config.pairs_path = "data/pairs-" + config.instance_id + ".csv";
    }
    config.edges_path = resolveInputPath(config.edges_path);
    config.pairs_path = resolveInputPath(config.pairs_path);
    return config;
}


template<typename Container>
std::map<int, std::set<int>> convertToAdjacencyList(const Container& edges) {
    std::map<int, std::set<int>> adjacencyList;
    for (const auto& edge : edges) {
        adjacencyList[edge.first].insert(edge.second);
        adjacencyList[edge.second].insert(edge.first);
    }
    return adjacencyList;
}

template <typename... Args>
std::string generateVarName(const std::string& prefix, Args... args) {
    std::ostringstream oss;
    oss << prefix;
    ((oss << "_" << args), ...);
    return oss.str();
}

int primeid(int i, int p) {
    return i * copy_num + p - 1;
}

std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty() && token.back() == '\r') {
            token.pop_back();
        }
        result.push_back(token);
    }
    if (!line.empty() && line.back() == ',') {
        result.emplace_back();
    }
    return result;
}

int read_csv_to_edges(std::string file_path1, 
                        std::map<std::string, int>& node_name_to_id, 
                        std::map<int, std::string>& id_to_node_name,
                        std::vector<std::pair<int, int>>& edges, 
                        std::map<std::pair<int, int>, int>& edges_to_weight, 
                        std::set<int>& center_nodes, 
                        std::vector<std::pair<int, int>>& center_edges, 
                        std::vector<std::pair<int, int>>& leaf_edges, 
                        std::map<int, int>& leaf_to_center, 
                        std::vector<std::pair<int, int>>& entry_edges) {
    std::ifstream file1(file_path1);
    std::string line;

    int next_node_id = 0;

    if (file1.is_open()) {
        std::getline(file1, line);
        while (std::getline(file1, line)) {
            auto tokens = parseCSVLine(line);
            if (tokens.size() >= 5) {
                std::string node1_name = tokens[2];
                std::string node2_name = tokens[3];
                int weight = static_cast<int>(std::stod(tokens[4]));

                if (node1_name.empty() || node2_name.empty()) {
                    std::cerr << "empty edge endpoint in " << file_path1 << std::endl;
                    return -1;
                }

                if (node_name_to_id.find(node1_name) == node_name_to_id.end()) {
                    node_name_to_id[node1_name] = next_node_id;
                    id_to_node_name[next_node_id] = node1_name;
                    next_node_id++;
                }
                if (node_name_to_id.find(node2_name) == node_name_to_id.end()) {
                    node_name_to_id[node2_name] = next_node_id;
                    id_to_node_name[next_node_id] = node2_name;
                    next_node_id++;
                }

                int node1_id = node_name_to_id[node1_name];
                int node2_id = node_name_to_id[node2_name];
                
                auto edge = (node1_id < node2_id) ? std::make_pair(node1_id, node2_id) : std::make_pair(node2_id, node1_id);
                // auto edge = std::make_pair(node1_id, node2_id);
                edges.push_back(edge);
                edges_to_weight[edge] = weight;

                int c_count = 0;
                int center = -1;
                int leaf = -1;
                int node1_count = 0;
                int node2_count = 0;
                auto is_pure_node = [](const std::string& name) {
                    return name.size() > 1 && std::all_of(name.begin() + 1, name.end(), [](unsigned char ch) {
                        return std::isdigit(ch) != 0;
                    });
                };
                if ((node1_name.front() == 'N' || node1_name.front() == 'M') && is_pure_node(node1_name)) {
                    node1_count = 3;
                    center = node1_id;
                    center_nodes.insert(node1_id);
                } else if (node1_name.front() == 'E') {
                    node1_count = 1;
                    center = node1_id;
                    center_nodes.insert(node1_id);
                } else {
                    node1_count = 0;
                    leaf = node1_id;
                }
                if ((node2_name.front() == 'N' || node2_name.front() == 'M') && is_pure_node(node2_name)) {
                    node2_count = 3;
                    center = node2_id;
                    center_nodes.insert(node2_id);
                } else if (node2_name.front() == 'E') {
                    node2_count = 1;
                    center = node2_id;
                    center_nodes.insert(node2_id);
                } else {
                    node2_count = 0;
                    leaf = node2_id;
                }
                c_count = node1_count + node2_count;

                if (c_count == 6) {
                    center_edges.emplace_back(edge);
                } else if (c_count == 4) {
                    center_edges.emplace_back(edge);
                    entry_edges.emplace_back(edge);
                } else if (c_count == 3 || c_count == 1) {
                    if (center < 0 || leaf < 0) {
                        std::cerr << "invalid center-leaf edge in " << file_path1 << std::endl;
                        return -1;
                    }
                    leaf_edges.emplace_back(edge);
                    leaf_to_center[leaf] = center;
                } else {
                    cout << "edge error" << endl;
                }

                // int node1_count = std::count_if(
                //     node1_name.begin(),
                //     node1_name.end(),
                //     [](char c) { return c == 'N' || c == 'E' || c == 'M'; }
                // );
                // if (node1_count > 0) {
                //     center_nodes.insert(node1_id);
                //     c_count++;
                //     center = node1_id;
                // } else {
                //     leaf = node1_id;
                // }
                // int node2_count = std::count_if(
                //     node2_name.begin(),
                //     node2_name.end(),
                //     [](char c) { return c == 'N' || c == 'E' || c == 'M'; }
                // );
                // if (node2_count > 0) {
                //     center_nodes.insert(node2_id);
                //     c_count++;
                //     center = node2_id;
                // } else {
                //     leaf = node2_id;
                // }

                // if (c_count == 2) {
                //     center_edges.push_back(edge);
                // } else if (c_count == 1) {
                //     leaf_edges.push_back(edge);
                //     leaf_to_center[leaf] = center;
                // } else {
                //     cout << "edge error" << endl;
                // }
            }
        }
        file1.close();
        return next_node_id;
    }
    else {
        std::cerr << "file open fail" << file_path1 << std::endl;
        return -1;
    }
}

int read_csv_to_pairs(std::string file_path2, 
                        std::map<std::string, int>& node_name_to_id,
                        std::vector<std::pair<int, int>>& pairs, 
                        std::vector<std::pair<int, int>>& ori_pairs,
                        std::map<std::pair<int, int>, double>& pairs_to_weight, 
                        std::map<std::pair<int, int>, std::pair<int, int>>& pairs_to_center_pairs, 
                        std::set<std::pair<int, int>>& center_pairs, 
                        std::map<int, std::pair<int, int>>& id_to_center_pairs, 
                        std::map<std::pair<int, int>, int>& center_pairs_to_id,
                        std::map<std::pair<int, int>, double>& center_pairs_to_weight, 
                        std::map<int, int>& leaf_to_center,
                        std::vector<std::pair<int, int>>& entry_edges,
                        std::map<int, std::vector<std::pair<int, int>>>& cpid_to_entry_edges) {
    std::ifstream file2(file_path2);
    std::string line;
    auto entry_adjlist = convertToAdjacencyList(entry_edges);
    int cpid = 0;
    if (file2.is_open()) {
        std::getline(file2, line);
        while (std::getline(file2, line)) {
            auto tokens = parseCSVLine(line);
            if (tokens.size() >= 6) {
                std::string node1_name;
                std::string node2_name;
                double weight = 0.0;
                if (tokens.size() >= 6 && !tokens[4].empty()) {
                    node1_name = tokens[3];
                    node2_name = tokens[4];
                    weight = std::stod(tokens[5]);
                } else if (tokens.size() >= 10 && !tokens[5].empty() && !tokens[9].empty()) {
                    node1_name = tokens[3];
                    node2_name = tokens[5];
                    weight = std::stod(tokens[9]);
                } else {
                    std::cerr << "unsupported pair schema with " << tokens.size()
                              << " columns in " << file_path2 << std::endl;
                    return -1;
                }

                if (node_name_to_id.find(node1_name) == node_name_to_id.end()) {
                    std::cerr << "unknown pair endpoint '" << node1_name << "' in " << file_path2 << std::endl;
                    return -1;
                }
                if (node_name_to_id.find(node2_name) == node_name_to_id.end()) {
                    std::cerr << "unknown pair endpoint '" << node2_name << "' in " << file_path2 << std::endl;
                    return -1;
                }

                int node1_id = node_name_to_id[node1_name];
                int node2_id = node_name_to_id[node2_name];
                ori_pairs.push_back(std::make_pair(node1_id, node2_id));
                
                auto pair = (node1_id < node2_id) ? std::make_pair(node1_id, node2_id) : std::make_pair(node2_id, node1_id);
                // auto pair = std::make_pair(node1_id, node2_id);
                pairs.push_back(pair);
                pairs_to_weight[pair] = weight;

                auto center1 = leaf_to_center.find(node1_id);
                auto center2 = leaf_to_center.find(node2_id);
                if (center1 == leaf_to_center.end() || center2 == leaf_to_center.end()) {
                    std::cerr << "pair endpoint is not attached to a center in " << file_path2
                              << ": " << node1_name << ", " << node2_name << std::endl;
                    return -1;
                }
                int cn1id = center1->second;
                int cn2id = center2->second;
                auto c_pair = (cn1id < cn2id) ? std::make_pair(cn1id, cn2id) : std::make_pair(cn2id, cn1id);
                pairs_to_center_pairs[pair] = c_pair;
                if (center_pairs.insert(c_pair).second) {
                    id_to_center_pairs[cpid] = c_pair;
                    center_pairs_to_id[c_pair] = cpid;
                    if (entry_adjlist.find(cn1id) != entry_adjlist.end()) {
                        for (const auto& nid : entry_adjlist[cn1id]) {
                            auto edge = (nid < cn1id) ? std::make_pair(nid, cn1id) : std::make_pair(cn1id, nid);
                            cpid_to_entry_edges[cpid].emplace_back(edge);
                        }
                    }
                    if (entry_adjlist.find(cn2id) != entry_adjlist.end()) {
                        for (const auto& nid : entry_adjlist[cn2id]) {
                            auto edge = (nid < cn2id) ? std::make_pair(nid, cn2id) : std::make_pair(cn2id, nid);
                            cpid_to_entry_edges[cpid].emplace_back(edge);
                        }
                    }
                    cpid++;
                    center_pairs_to_weight[c_pair] = weight;
                } else {
                    center_pairs_to_weight[c_pair] += weight;
                }
                
            }
        }
        file2.close();
        return 0;
    }
    else {
        std::cerr << "file open fail" << file_path2 << std::endl;
        return -1;
    }
}

int set_dflt_copynums(std::map<std::string, int>& node_name_to_id, 
                        std::map<int, int>& id_to_copynums) {
    const int n = node_name_to_id.size();
    for (int i = 0; i < n; i++) {
        id_to_copynums[i] = copy_num_dflt;
    }
    return 0;
}

int read_csv_to_copynums(std::string file_path1, 
                            std::map<std::string, int>& node_name_to_id, 
                            std::map<int, int>& id_to_copynums) {
    std::ifstream file1(file_path1);
    std::string line;

    if (file1.is_open()) {
        std::getline(file1, line);
        while (std::getline(file1, line)) {
            auto tokens = parseCSVLine(line);
            if (tokens.size() >= 2) {
                std::string node_name = tokens[0];
                int copynum = std::stoi(tokens[1]);

                if (node_name_to_id.find(node_name) == node_name_to_id.end()) {
                    return -1;
                }

                int node_id = node_name_to_id[node_name];
                id_to_copynums[node_id] = copynum;
            }
        }
        file1.close();
        return 0;
    }
    else {
        std::cerr << "file open fail" << file_path1 << std::endl;
        return -1;
    }
}

std::string statusName(SCIP_STATUS status) {
    switch (status) {
    case SCIP_STATUS_UNKNOWN: return "unknown";
    case SCIP_STATUS_USERINTERRUPT: return "user_interrupt";
    case SCIP_STATUS_NODELIMIT: return "node_limit";
    case SCIP_STATUS_TOTALNODELIMIT: return "total_node_limit";
    case SCIP_STATUS_STALLNODELIMIT: return "stall_node_limit";
    case SCIP_STATUS_TIMELIMIT: return "time_limit";
    case SCIP_STATUS_MEMLIMIT: return "memory_limit";
    case SCIP_STATUS_GAPLIMIT: return "gap_limit";
    case SCIP_STATUS_SOLLIMIT: return "solution_limit";
    case SCIP_STATUS_BESTSOLLIMIT: return "best_solution_limit";
    case SCIP_STATUS_RESTARTLIMIT: return "restart_limit";
    case SCIP_STATUS_OPTIMAL: return "optimal";
    case SCIP_STATUS_INFEASIBLE: return "infeasible";
    case SCIP_STATUS_UNBOUNDED: return "unbounded";
    case SCIP_STATUS_INFORUNBD: return "infeasible_or_unbounded";
    case SCIP_STATUS_TERMINATE: return "terminate";
    default: return "unrecognized";
    }
}

std::string configuredBranchruleName(const std::string& method) {
    if (method == "strong") {
        return "fullstrong";
    }
    if (method == "default") {
        return "relpscost";
    }
    if (method == "custom-random") {
        return rlbranch::branchruleName(rlbranch::CustomBranchingStrategy::Random);
    }
    if (method == "custom-mostinf") {
        return rlbranch::branchruleName(rlbranch::CustomBranchingStrategy::MostInfeasible);
    }
    if (method == "rl-mlp") {
        return rlbranch::kRlMlpBranchruleName;
    }
    if (method == "rl-gcnn") {
        return rlbranch::kRlGcnnBranchruleName;
    }
    return method;
}

bool isCustomBranching(const std::string& method) {
    return method == "custom-random" || method == "custom-mostinf";
}

bool isRlMlpBranching(const std::string& method) {
    return method == "rl-mlp";
}

bool isRlGcnnBranching(const std::string& method) {
    return method == "rl-gcnn";
}

SCIP_RETCODE configureScip(SCIP* scip, const RunConfig& config, SolverMetrics& metrics) {
    const auto profile = rlbranch::loadScipProfile(config.scip_profile);
    SCIP_CALL(rlbranch::applyScipProfile(scip, profile));
    SCIP_CALL(SCIPsetRealParam(scip, "limits/time", config.time_limit));
    SCIP_CALL(SCIPsetLongintParam(scip, "limits/nodes", config.node_limit));
    SCIP_CALL(SCIPsetIntParam(scip, "randomization/randomseedshift", config.randomseedshift));
    SCIP_CALL(SCIPsetIntParam(scip, "randomization/permutationseed", config.permutationseed));
    SCIP_CALL(SCIPsetIntParam(scip, "randomization/lpseed", config.lpseed));
    if (config.threads_explicit) {
        if (config.threads != 1) {
            throw std::invalid_argument("formal project-production-v1 runs require --threads 1");
        }
        SCIP_CALL(SCIPsetIntParam(scip, "parallel/minnthreads", 1));
        SCIP_CALL(SCIPsetIntParam(scip, "parallel/maxnthreads", 1));
        SCIP_CALL(SCIPsetIntParam(scip, "lp/threads", 1));
    }
    rlbranch::assertProductionInvariants(scip);
    rlbranch::requireEstimateNodeSelector(scip);
    metrics.node_selector = rlbranch::activeNodeSelectorName(scip);
    metrics.scip_profile_sha256 = profile.file_sha256;
    const std::string applied_dump = rlbranch::dumpAppliedProfile(scip, profile);
    metrics.applied_profile_dump_sha256 = rlbranch::sha256Text(applied_dump);
    metrics.scip_param_dump_sha256 = metrics.applied_profile_dump_sha256;
    metrics.effective_search_params_sha256 = rlbranch::sha256Text(
        rlbranch::dumpEffectiveSearchParams(scip));

    metrics.active_branchrule = configuredBranchruleName(config.branching);
    if (config.branching != "default") {
        const std::string priority_param = "branching/" + metrics.active_branchrule + "/priority";
        SCIP_CALL(SCIPsetIntParam(scip, priority_param.c_str(), 1000000));
    }
    if (config.branching == "rl-gcnn" && config.rl_fallback == "relpscost") {
        SCIP_CALL(SCIPsetIntParam(scip, "branching/relpscost/priority", 999999));
    }
    rlbranch::requireEstimateNodeSelector(scip);
    metrics.node_selector = rlbranch::activeNodeSelectorName(scip);
    metrics.effective_search_params_sha256 = rlbranch::sha256Text(
        rlbranch::dumpEffectiveSearchParams(scip));
    metrics.effective_search_params_core_sha256 = rlbranch::sha256Text(
        rlbranch::dumpEffectiveSearchParams(scip, false));
    return SCIP_OKAY;
}

SCIP_RETCODE SolveMIPProblem(std::set<int> nodes,
                        std::vector<std::pair<int, int>>& edges, 
                        std::map<std::pair<int, int>, int>& edges_to_weight,
                        std::map<int, std::pair<int, int>>& id_to_pairs, 
                        std::map<std::pair<int, int>, double>& pairs_to_weight,
                        std::map<int, std::set<int>>& adj_list, 
                        std::map<std::pair<int, int>, std::set<std::pair<int, int>>>& pairs_to_ret_edges_prime, 
                        std::set<std::pair<int, int>>& ret_edges_prime, 
                        std::map<int, int>& k2prime, 
                        std::map<int, int>& id_to_copynums,
                        std::vector<std::pair<int, int>>& entry_edges,
                        std::map<int, std::vector<std::pair<int, int>>>& cpid_to_entry_edges,
                        const RunConfig& config,
                        SolverMetrics& metrics,
                        double& obj_value) {

    const auto solver_wall_start = std::chrono::steady_clock::now();
    // Initialize SCIP
    SCIP* scip = nullptr;
    rlbranch::BranchruleStats custom_branching_stats;
    SCIP_CALL(SCIPcreate(&scip));
    SCIP_CALL(SCIPincludeDefaultPlugins(scip));
    if (isCustomBranching(config.branching)) {
        const auto strategy = config.branching == "custom-random"
            ? rlbranch::CustomBranchingStrategy::Random
            : rlbranch::CustomBranchingStrategy::MostInfeasible;
        SCIP_CALL(rlbranch::includeCustomBranchrule(
            scip,
            strategy,
            static_cast<unsigned int>(config.seed),
            config.branch_log,
            &custom_branching_stats));
    } else if (isRlGcnnBranching(config.branching)) {
        rlbranch::RlGcnnOptions options;
        options.model_path = config.rl_model;
        options.device = config.rl_device;
        options.fallback = config.rl_fallback;
        options.log_path = config.rl_log;
        options.max_depth = config.rl_max_depth;
        options.min_candidates = config.rl_min_candidates;
        options.lambda_prim = 0.0F;
        options.prim_min_depth = 0;
        options.prim_require_grown = false;
        options.use_prim_features = false;
        options.bias_mode = "none";
        SCIP_CALL(rlbranch::includeRlGcnnBranchrule(
            scip,
            options,
            &custom_branching_stats));
    }
    SCIP_CALL(SCIPcreateProbBasic(scip, "MIPProblem"));

    SCIP_CALL(configureScip(scip, config, metrics));
    
    cout << "SCIP" << endl;
    cout << "mip_gap: " << gap << endl;
    cout << "branching: " << config.branching << " (" << metrics.active_branchrule << ")" << endl;
    cout << "protocol: " << config.protocol << endl;
    cout << "seed: " << config.seed << endl;

    const int K = id_to_pairs.size();
    const int e = edges.size();
    const int n = nodes.size();
    metrics.n_center_nodes = n;
    metrics.n_center_edges = e;
    metrics.n_commodities = K;
    //const int copy_num = 3; // Assuming copy_num is consistent

    cout << "pairs size: " << K << endl;
    cout << "nodes size: " << n << endl;
    cout << "edges size: " << e << endl;
    cout << "copy   num: " << copy_num << endl;

    // Variable storage
    map<tuple<int, int, int>, SCIP_VAR*> x;
    map<tuple<int, int, int>, SCIP_VAR*> f;
    map<tuple<int, int, int>, SCIP_VAR*> absf;
    map<tuple<int, int>, SCIP_VAR*> m;
    map<tuple<int, int>, SCIP_VAR*> y; // topo sort
    map<tuple<int, int, int>, SCIP_VAR*> z;

    // Create variables
    for (const auto& i : nodes) {
        for (int prime = 0; prime < copy_num; ++prime) {
            SCIP_VAR* var;
            SCIP_CALL(SCIPcreateVarBasic(scip, &var,
                generateVarName("y", i, prime).c_str(),
                0.0, 1.0, 0.0, SCIP_VARTYPE_CONTINUOUS));
            SCIP_CALL(SCIPaddVar(scip, var));
            y.emplace(make_tuple(i, prime), var);
            SCIP_CALL(SCIPreleaseVar(scip, &var));
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int prime = 0; prime < copy_num; ++prime) {
            SCIP_VAR* var;
            SCIP_CALL(SCIPcreateVarBasic(scip, &var,
                generateVarName("m", k, prime).c_str(),
                0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY));
            SCIP_CALL(SCIPaddVar(scip, var));
            m.emplace(make_tuple(k, prime), var);
            SCIP_CALL(SCIPreleaseVar(scip, &var));
        }
    }

    for (int k = 0; k < K; ++k) {
        // auto save_edges = cpid_to_entry_edges[k];
        for (const auto& edge : edges) {
            int i = edge.first;
            int j = edge.second;

            // if (std::find(entry_edges.begin(), entry_edges.end(), edge) != entry_edges.end()) {
            //     if (std::find(save_edges.begin(), save_edges.end(), edge) == save_edges.end()) {
            //         continue;
            //     }
            // }

            // SCIP_VAR* var_x;
            // SCIP_CALL(SCIPcreateVarBasic(scip, &var_x,
            //     generateVarName("x", i, j, k).c_str(),
            //     0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY));
            // SCIP_CALL(SCIPaddVar(scip, var_x));
            // x.emplace(make_tuple(i, j, k), var_x);

            SCIP_VAR* var_absf;
            SCIP_CALL(SCIPcreateVarBasic(scip, &var_absf,
                generateVarName("absf", i, j, k).c_str(),
                0.0, 1.0, 0.0, SCIP_VARTYPE_CONTINUOUS));
            SCIP_CALL(SCIPaddVar(scip, var_absf));
            absf.emplace(make_tuple(i, j, k), var_absf);
            SCIP_CALL(SCIPreleaseVar(scip, &var_absf));

            SCIP_VAR* var_f1;
            SCIP_CALL(SCIPcreateVarBasic(scip, &var_f1,
                generateVarName("f", i, j, k).c_str(),
                -1.0, 1.0, 0.0, SCIP_VARTYPE_CONTINUOUS));
            SCIP_CALL(SCIPaddVar(scip, var_f1));
            f.emplace(make_tuple(i, j, k), var_f1);
            SCIP_CALL(SCIPreleaseVar(scip, &var_f1));

            SCIP_VAR* var_f2;
            SCIP_CALL(SCIPcreateVarBasic(scip, &var_f2,
                generateVarName("f", j, i, k).c_str(),
                -1.0, 1.0, 0.0, SCIP_VARTYPE_CONTINUOUS));
            SCIP_CALL(SCIPaddVar(scip, var_f2));
            f.emplace(make_tuple(j, i, k), var_f2);
            SCIP_CALL(SCIPreleaseVar(scip, &var_f2));
        }  
    }

    for (const auto& edge : edges) {
        int i = edge.first;
        int j = edge.second;

        for (int prime = 0; prime < copy_num; ++prime) {
            SCIP_VAR* var_z1;
            SCIP_CALL(SCIPcreateVarBasic(scip, &var_z1,
                generateVarName("z", i, j, prime).c_str(),
                0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY));
            SCIP_CALL(SCIPaddVar(scip, var_z1));
            z.emplace(make_tuple(i, j, prime), var_z1);
            SCIP_CALL(SCIPreleaseVar(scip, &var_z1));

            SCIP_VAR* var_z2;
            SCIP_CALL(SCIPcreateVarBasic(scip, &var_z2,
                generateVarName("z", j, i, prime).c_str(),
                0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY));
            SCIP_CALL(SCIPaddVar(scip, var_z2));
            z.emplace(make_tuple(j, i, prime), var_z2);
            SCIP_CALL(SCIPreleaseVar(scip, &var_z2));
        }
    }

    // priority
    // for (int k = 0; k < K; ++k) {
    //     for (int prime = 0; prime < copy_num; ++prime) {
    //         SCIP_CALL(SCIPchgVarBranchPriority(scip, m.at(make_tuple(k, prime)), 10-k));
    //     }
    // }

    for (const auto& [k, pair] : id_to_pairs) {
        int start = pair.first, end = pair.second;
        // auto save_edges = cpid_to_entry_edges[k];
        for (const auto& edge : entry_edges) {
            int i = edge.first;
            int j = edge.second;
            if (i == start || i == end || j == start || j == end) {
                continue;
            } else {
                SCIP_CONS* fforbid = nullptr;
                SCIP_CALL(SCIPcreateConsLinear(scip, &fforbid, "fforbid", 0, nullptr, nullptr,
                    0.0, 0.0, true, true, true, true, true, false, false, false, false, false));
                SCIP_CALL(SCIPaddCoefLinear(scip, fforbid, f.at(std::make_tuple(i, j, k)), 1.0));
                SCIP_CALL(SCIPaddCons(scip, fforbid));
                SCIP_CALL(SCIPreleaseCons(scip, &fforbid));
            }
            // if (std::find(save_edges.begin(), save_edges.end(), edge) == save_edges.end()) {
            //     int i = edge.first;
            //     int j = edge.second;
            //     model.AddConstr(x.at(std::make_tuple(i, j, k)) == 0);
            // }
        }
    }

    // Set objective
    for (int k = 0; k < K; ++k) {
        double pair_weight = pairs_to_weight[id_to_pairs[k]];
        for (const auto& edge : edges) {
            int i = edge.first;
            int j = edge.second;
            double coef = edges_to_weight[edge] * pair_weight;
            SCIP_CALL(SCIPchgVarObj(scip, absf.at(std::make_tuple(i, j, k)), coef)); 
        }
    }
    SCIP_CALL(SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE));

    for (const auto& edge : edges) {
        int i = edge.first;
        int j = edge.second;
        for (int k = 0; k < K; ++k) {
            SCIP_CONS* abs1 = nullptr;
            SCIP_CALL(SCIPcreateConsLinear(scip, &abs1, "abs1", 0, nullptr, nullptr,
                0.0, SCIPinfinity(scip), true, true, true, true, true, false, false, false, false, false));
            SCIP_CALL(SCIPaddCoefLinear(scip, abs1, absf.at(make_tuple(i, j, k)), 1.0));
            SCIP_CALL(SCIPaddCoefLinear(scip, abs1, f.at(make_tuple(i, j, k)), -1.0));
            SCIP_CALL(SCIPaddCons(scip, abs1));
            SCIP_CALL(SCIPreleaseCons(scip, &abs1));
            SCIP_CONS* abs2 = nullptr;
            SCIP_CALL(SCIPcreateConsLinear(scip, &abs2, "abs2", 0, nullptr, nullptr,
                0.0, SCIPinfinity(scip), true, true, true, true, true, false, false, false, false, false));
            SCIP_CALL(SCIPaddCoefLinear(scip, abs2, absf.at(make_tuple(i, j, k)), 1.0));
            SCIP_CALL(SCIPaddCoefLinear(scip, abs2, f.at(make_tuple(i, j, k)), 1.0));
            SCIP_CALL(SCIPaddCons(scip, abs2));
            SCIP_CALL(SCIPreleaseCons(scip, &abs2));
        }
    }

    // Connection constraints
    for (int k = 0; k < K; ++k) {
        int s = id_to_pairs.at(k).first;
        int t = id_to_pairs.at(k).second;

        // Flow balance constraints
        for (const auto& i : nodes) {
            double bound = (i == s) ? 1 : (i == t) ? -1 : 0;
            SCIP_CONS* cons = nullptr;
            SCIP_CALL(SCIPcreateConsLinear(scip, &cons, "flow_balance", 0, nullptr, nullptr,
                bound, bound, true, true, true, true, true, false, false, false, false, false));

            for (const auto& j : adj_list.at(i)) {
                SCIP_CALL(SCIPaddCoefLinear(scip, cons, f.at(make_tuple(i, j, k)), 1.0));
            }

            SCIP_CALL(SCIPaddCons(scip, cons));
            SCIP_CALL(SCIPreleaseCons(scip, &cons));
        }

        // Flow constraints for each edge
        for (const auto& edge : edges) {
            int i = edge.first;
            int j = edge.second;

            // Flow symmetry
            SCIP_CONS* sym_cons = nullptr;
            SCIP_CALL(SCIPcreateConsLinear(scip, &sym_cons, "flow_symmetry", 0, nullptr, nullptr,
                0.0, 0.0, true, true, true, true, true, false, false, false, false, false));
            SCIP_CALL(SCIPaddCoefLinear(scip, sym_cons, f.at(make_tuple(i, j, k)), 1.0));
            SCIP_CALL(SCIPaddCoefLinear(scip, sym_cons, f.at(make_tuple(j, i, k)), 1.0));
            SCIP_CALL(SCIPaddCons(scip, sym_cons));
            SCIP_CALL(SCIPreleaseCons(scip, &sym_cons));

            // // Capacity constraints
            // SCIP_CONS* upper1 = nullptr;
            // SCIP_CALL(SCIPcreateConsLinear(scip, &upper1, "capacity_upper1", 0, nullptr, nullptr,
            //     -SCIPinfinity(scip), 0.0, true, true, true, true, true, false, false, false, false, false));
            // SCIP_CALL(SCIPaddCoefLinear(scip, upper1, f.at(make_tuple(i, j, k)), 1.0));
            // SCIP_CALL(SCIPaddCoefLinear(scip, upper1, x.at(make_tuple(i, j, k)), -1.0));
            // SCIP_CALL(SCIPaddCons(scip, upper1));
            // // SCIP_CALL(SCIPreleaseCons(scip, &upper1));

            // SCIP_CONS* lower1 = nullptr;
            // SCIP_CALL(SCIPcreateConsLinear(scip, &lower1, "capacity_lower1", 0, nullptr, nullptr,
            //     0.0, SCIPinfinity(scip), true, true, true, true, true, false, false, false, false, false));
            // SCIP_CALL(SCIPaddCoefLinear(scip, lower1, f.at(make_tuple(i, j, k)), 1.0));
            // SCIP_CALL(SCIPaddCoefLinear(scip, lower1, x.at(make_tuple(i, j, k)), 1.0));
            // SCIP_CALL(SCIPaddCons(scip, lower1));
            // // SCIP_CALL(SCIPreleaseCons(scip, &lower1));

            // SCIP_CONS* upper2 = nullptr;
            // SCIP_CALL(SCIPcreateConsLinear(scip, &upper2, "capacity_upper2", 0, nullptr, nullptr,
            //     -SCIPinfinity(scip), 0.0, true, true, true, true, true, false, false, false, false, false));
            // SCIP_CALL(SCIPaddCoefLinear(scip, upper2, f.at(make_tuple(j, i, k)), 1.0));
            // SCIP_CALL(SCIPaddCoefLinear(scip, upper2, x.at(make_tuple(i, j, k)), -1.0));
            // SCIP_CALL(SCIPaddCons(scip, upper2));
            // // SCIP_CALL(SCIPreleaseCons(scip, &upper2));

            // SCIP_CONS* lower2 = nullptr;
            // SCIP_CALL(SCIPcreateConsLinear(scip, &lower2, "capacity_lower2", 0, nullptr, nullptr,
            //     0.0, SCIPinfinity(scip), true, true, true, true, true, false, false, false, false, false));
            // SCIP_CALL(SCIPaddCoefLinear(scip, lower2, f.at(make_tuple(j, i, k)), 1.0));
            // SCIP_CALL(SCIPaddCoefLinear(scip, lower2, x.at(make_tuple(i, j, k)), 1.0));
            // SCIP_CALL(SCIPaddCons(scip, lower2));
            // // SCIP_CALL(SCIPreleaseCons(scip, &lower2));
        }
    }

    // Edge merge constraints
    for (int k = 0; k < K; ++k) {
        SCIP_CONS* onlym = nullptr;
        SCIP_CALL(SCIPcreateConsLinear(scip, &onlym, "onlym", 0, nullptr, nullptr,
            1.0, 1.0, true, true, true, true, true, false, false, false, false, false));
        for (int prime = 0; prime < copy_num; ++prime) {
            SCIP_CALL(SCIPaddCoefLinear(scip, onlym, m.at(make_tuple(k, prime)), 1.0));
        }
        SCIP_CALL(SCIPaddCons(scip, onlym));
        SCIP_CALL(SCIPreleaseCons(scip, &onlym));
    }

    for (int prime = 0; prime < copy_num-1; ++prime) {
        SCIP_CONS* imbalance = nullptr;
        SCIP_CALL(SCIPcreateConsLinear(scip, &imbalance, "imbalance", 0, nullptr, nullptr,
            0.0, SCIPinfinity(scip), true, true, true, true, true, false, false, false, false, false));
        for (int k = 0; k < K; ++k) {
            SCIP_CALL(SCIPaddCoefLinear(scip, imbalance, m.at(make_tuple(k, prime)), 1.0));
            SCIP_CALL(SCIPaddCoefLinear(scip, imbalance, m.at(make_tuple(k, prime+1)), -1.0));
        }
        SCIP_CALL(SCIPaddCons(scip, imbalance));
        SCIP_CALL(SCIPreleaseCons(scip, &imbalance));
    }

    for (const auto& edge : edges) {
        int i = edge.first;
        int j = edge.second;
        for (int prime = 0; prime < copy_num; ++prime) {
            for (int k = 0; k < K; ++k) {
                SCIP_CONS* zlower = nullptr;
                SCIP_CALL(SCIPcreateConsLinear(scip, &zlower, "zlower", 0, nullptr, nullptr,
                    -1.0-1e-3, SCIPinfinity(scip), true, true, true, true, true, false, false, false, false, false));
                SCIP_CALL(SCIPaddCoefLinear(scip, zlower, z.at(make_tuple(i, j, prime)), 1.0));
                SCIP_CALL(SCIPaddCoefLinear(scip, zlower, z.at(make_tuple(j, i, prime)), 1.0));
                SCIP_CALL(SCIPaddCoefLinear(scip, zlower, f.at(make_tuple(i, j, k)), -1.0));
                SCIP_CALL(SCIPaddCoefLinear(scip, zlower, m.at(make_tuple(k, prime)), -1.0));
                SCIP_CALL(SCIPaddCons(scip, zlower));
                SCIP_CALL(SCIPreleaseCons(scip, &zlower));
            }
        }
    }

    // Loop constraints
    for (const auto& edge : edges) {
        int i = edge.first;
        int j = edge.second;
        for (int prime = 0; prime < copy_num; ++prime) {
            // Topological sequence constraints
            SCIP_CONS* topo_seq1 = nullptr;
            SCIP_CALL(SCIPcreateConsLinear(scip, &topo_seq1, "topo_seq1", 0, nullptr, nullptr,
                -SCIPinfinity(scip), 1.0 - 1e-3, true, true, true, true, true, false, false, false, false, false));
            SCIP_CALL(SCIPaddCoefLinear(scip, topo_seq1, y.at(make_tuple(i, prime)), 1.0));
            SCIP_CALL(SCIPaddCoefLinear(scip, topo_seq1, y.at(make_tuple(j, prime)), -1.0));
            SCIP_CALL(SCIPaddCoefLinear(scip, topo_seq1, z.at(make_tuple(i, j, prime)), 1.0));
            SCIP_CALL(SCIPaddCons(scip, topo_seq1));
            SCIP_CALL(SCIPreleaseCons(scip, &topo_seq1));

            SCIP_CONS* topo_seq2 = nullptr;
            SCIP_CALL(SCIPcreateConsLinear(scip, &topo_seq2, "topo_seq2", 0, nullptr, nullptr,
                -SCIPinfinity(scip), 1.0 - 1e-3, true, true, true, true, true, false, false, false, false, false));
            SCIP_CALL(SCIPaddCoefLinear(scip, topo_seq2, y.at(make_tuple(j, prime)), 1.0));
            SCIP_CALL(SCIPaddCoefLinear(scip, topo_seq2, y.at(make_tuple(i, prime)), -1.0));
            SCIP_CALL(SCIPaddCoefLinear(scip, topo_seq2, z.at(make_tuple(j, i, prime)), 1.0));
            SCIP_CALL(SCIPaddCons(scip, topo_seq2));
            SCIP_CALL(SCIPreleaseCons(scip, &topo_seq2));

            // Single direction constraint
            // SCIP_CONS* single_dir = nullptr;
            // SCIP_CALL(SCIPcreateConsLinear(scip, &single_dir, "single_dir", 0, nullptr, nullptr,
            //     -SCIPinfinity(scip), 1.0, true, true, true, true, true, false, false, false, false, false));
            // SCIP_CALL(SCIPaddCoefLinear(scip, single_dir, z.at(make_tuple(i, j, prime)), 1.0));
            // SCIP_CALL(SCIPaddCoefLinear(scip, single_dir, z.at(make_tuple(j, i, prime)), 1.0));
            // SCIP_CALL(SCIPaddCons(scip, single_dir));
            // SCIP_CALL(SCIPreleaseCons(scip, &single_dir));
        }
    }

    // Only father constraints
    for (const auto& i : nodes) {
        for (int prime = 0; prime < copy_num; ++prime) {
            SCIP_CONS* only_father = nullptr;
            SCIP_CALL(SCIPcreateConsLinear(scip, &only_father, "only_father", 0, nullptr, nullptr,
                0.0, 1.0, true, true, true, true, true, false, false, false, false, false));

            for (const auto& j : adj_list.at(i)) {
                SCIP_CALL(SCIPaddCoefLinear(scip, only_father, z.at(make_tuple(j, i, prime)), 1.0));
            }

            SCIP_CALL(SCIPaddCons(scip, only_father));
            SCIP_CALL(SCIPreleaseCons(scip, &only_father));
        }
    }

    metrics.n_vars = SCIPgetNOrigVars(scip);
    metrics.n_integer_vars = static_cast<int>(m.size() + z.size());
    metrics.n_constraints = SCIPgetNOrigConss(scip);

    std::string export_path = config.export_milp;
    if (config.legacy_cli && export_path.empty()) {
        export_path = "./save/model_scip_tree_data4+cp" + std::to_string(copy_num) + ".lp";
    }
    if (!export_path.empty()) {
        ensureParentDirectory(export_path);
        SCIP_CALL(SCIPwriteOrigProblem(scip, export_path.c_str(), nullptr, FALSE));
    }
    if (config.build_only) {
        metrics.status = "build_only";
        metrics.custom_branching = custom_branching_stats;
        metrics.wall_clock_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - solver_wall_start).count();
        SCIP_CALL(SCIPfree(&scip));
        return SCIP_OKAY;
    }

    cout << "Solving..." << endl;
    SCIP_CALL(SCIPsolve(scip));
    rlbranch::requireEstimateNodeSelector(scip);
    metrics.node_selector = rlbranch::activeNodeSelectorName(scip);

    SCIP_STATUS status = SCIPgetStatus(scip);
    metrics.status = statusName(status);
    metrics.primal_bound = SCIPgetPrimalbound(scip);
    metrics.dual_bound = SCIPgetDualbound(scip);
    metrics.final_gap = SCIPgetGap(scip);
    metrics.solving_time = SCIPgetSolvingTime(scip);
    metrics.presolve_time = SCIPgetPresolvingTime(scip);
    metrics.solve_time_after_presolve = std::max(0.0, metrics.solving_time - metrics.presolve_time);
    metrics.nodes = SCIPgetNNodes(scip);
    metrics.lp_iterations = SCIPgetNLPIterations(scip);
    metrics.primal_dual_integral = scip->stat->primaldualintegral;
    metrics.has_solution = SCIPgetNSols(scip) > 0;
    if (metrics.has_solution) {
        metrics.objective = metrics.primal_bound;
        metrics.first_solution_time = scip->stat->firstprimaltime;
        obj_value = metrics.objective;
    }
    SCIP_BRANCHRULE* selected_branchrule = SCIPfindBranchrule(scip, metrics.active_branchrule.c_str());
    if (selected_branchrule != nullptr) {
        metrics.branchrule_calls = SCIPbranchruleGetNLPCalls(selected_branchrule)
            + SCIPbranchruleGetNExternCalls(selected_branchrule)
            + SCIPbranchruleGetNPseudoCalls(selected_branchrule);
        metrics.branchrule_priority = SCIPbranchruleGetPriority(selected_branchrule);
    }
    metrics.custom_branching = custom_branching_stats;

    switch (status) {
    case SCIP_STATUS_OPTIMAL:
        cout << "OPTIMAL" << endl;
        // SCIP_CALL(SCIPwriteOrigProblem(scip, "model_data4+cp4.lp", "lp", FALSE));
        break;
    case SCIP_STATUS_INFEASIBLE:
        cout << "INFEASIBLE" << endl;
        // SCIP_CALL( SCIPcomputeIIS(scip) );
        // SCIP_CALL( SCIPwriteIIS(scip, "model_inf.iis") );
        break;
    default:
        cout << "OTHER_STATUS" << endl;
        break;
    }

    SCIP_SOL* sol = SCIPgetBestSol(scip);
    if (sol != nullptr) {
        SCIP_Bool feasible = FALSE;
        SCIP_CALL(SCIPcheckSol(scip, sol, FALSE, TRUE, TRUE, TRUE, TRUE, &feasible));
        metrics.solution_feasible = feasible;
        cout << "Solution:" << endl;
        cout << "Objective value = " << obj_value << endl;

        for (int k = 0; k < K; ++k) {
            for (int prime = 0; prime < copy_num; ++prime) {
                if (SCIPgetSolVal(scip, sol, m.at(make_tuple(k, prime))) > 0.5) {
                    k2prime.emplace(k, prime);
                }
            }
        }

        for (auto& [key, var] : f) {
            if (SCIPgetSolVal(scip, sol, var) > 0.5) {
                int i = get<0>(key);
                int j = get<1>(key);
                int k = get<2>(key);
                auto prime_it = k2prime.find(k);
                if (i == j || prime_it == k2prime.end()) {
                    continue;
                }
                int prime = prime_it->second;
                int u = i * copy_num + prime;
                int v = j * copy_num + prime;
                auto edge_prime = (u < v) ? make_pair(u, v) : make_pair(v, u);
                pairs_to_ret_edges_prime[id_to_pairs[k]].insert(edge_prime);
                ret_edges_prime.insert(edge_prime);
            }
        }
    }

    metrics.wall_clock_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solver_wall_start).count();
    SCIP_CALL(SCIPfree(&scip));
    return SCIP_OKAY;
}

std::optional<std::vector<int>> detectCycle(const std::map<int, std::set<int>>& graph) {
    std::unordered_set<int> visited;
    std::stack<int> pathStack;
    auto dfs = [&](int node, int parent, auto& dfsRef) -> std::optional<std::vector<int>> {
        visited.insert(node);
        pathStack.push(node);
        for (int neighbor : graph.at(node)) {
            if (neighbor == parent) {
                continue;
            }
            if (visited.find(neighbor) != visited.end()) {
                std::vector<int> cyclePath;
                while (!pathStack.empty()) {
                    int topNode = pathStack.top();
                    cyclePath.push_back(topNode);
                    pathStack.pop();

                    if (topNode == neighbor) {
                        break;
                    }
                }
                return cyclePath;
            } else {
                auto result = dfsRef(neighbor, node, dfsRef);
                if (result) {
                    return result;
                }
            }
        }
        pathStack.pop();
        return std::nullopt;
    };

    for (const auto& [node, _] : graph) {
        if (visited.find(node) == visited.end()) {
            auto result = dfs(node, -1, dfs);
            if (result) {
                return result;
            }
        }
    }

    return std::nullopt;
}

double ret_compose(double obj, std::vector<std::pair<int, int>>& edges, 
                    std::map<std::pair<int, int>, int>& edges_to_weight, 
                    std::vector<std::pair<int, int>>& pairs, 
                    std::map<std::pair<int, int>, double>& pairs_to_weight, 
                    std::map<int, int>& leaf_to_center, 
                    std::map<std::pair<int, int>, std::pair<int, int>>& pairs_to_center_pairs, 
                    std::map<std::pair<int, int>, int>& center_pairs_to_id,
                    std::map<int, std::set<std::pair<std::pair<int, int>, std::pair<int, int>>>>& id_to_ret_edges_prime, 
                    std::map<std::pair<int, int>, std::set<std::pair<int, int>>>& cpairs_to_ret_edges_prime, 
                    std::set<std::pair<int, int>>& ret_edges_prime, 
                    std::map<int, int>& k2prime) {
    const int K = pairs.size();
    // double ttl_obj = obj;
    double ttl_obj = 0;
    for (int k = 0; k < K; ++k) {
        auto pair = pairs[k];
        int s = pair.first;
        int t = pair.second;
        double pair_weight = pairs_to_weight[pair];
        auto cpair = pairs_to_center_pairs[pair];
        int prime = k2prime[center_pairs_to_id[cpair]];
        double edge_weight_sum = 0;

        auto sedge_prime = (s < leaf_to_center[s]) 
            ? std::make_pair(std::make_pair(s, 0), std::make_pair(leaf_to_center[s], prime)) 
            : std::make_pair(std::make_pair(leaf_to_center[s], prime), std::make_pair(s, 0));
        auto tedge_prime = (t < leaf_to_center[t]) 
            ? std::make_pair(std::make_pair(t, 0), std::make_pair(leaf_to_center[t], prime)) 
            : std::make_pair(std::make_pair(leaf_to_center[t], prime), std::make_pair(t, 0));
        id_to_ret_edges_prime[k].insert(sedge_prime);
        id_to_ret_edges_prime[k].insert(tedge_prime);
        // ret_edges_prime.insert(sedge_prime);
        // ret_edges_prime.insert(tedge_prime);
        for (const auto& edge : cpairs_to_ret_edges_prime[cpair]) {
            int u = edge.first;
            int v = edge.second;
            int i = u / copy_num;
            int i_prime = u % copy_num;
            int j = v / copy_num;
            int j_prime = v % copy_num;
            id_to_ret_edges_prime[k].insert(std::make_pair(std::make_pair(i, i_prime), std::make_pair(j, j_prime)));
        }

        for (const auto& edge : id_to_ret_edges_prime[k]) {
            auto u = edge.first;
            auto v = edge.second;
            int i = u.first;
            int j = v.first;
            edge_weight_sum += edges_to_weight[std::make_pair(i, j)];
        }
        
        ttl_obj += (edge_weight_sum * pair_weight);

        // auto sedge = (s < leaf_to_center[s]) ? std::make_pair(s, leaf_to_center[s]) : std::make_pair(leaf_to_center[s], s);
        // auto tedge = (t < leaf_to_center[t]) ? std::make_pair(t, leaf_to_center[t]) : std::make_pair(leaf_to_center[t], t);
        // ttl_obj += (edges_to_weight[sedge] + edges_to_weight[tedge]) * pair_weight;
    }
    return ttl_obj;
}

std::string getCurrentTimeSuffix() {
    // 获取当前时间
    std::time_t now = std::time(nullptr);
    std::tm tm = *std::localtime(&now);

    // 将时间格式化为字符串
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

int print_pairs_to_ret_edges(std::map<int, std::set<std::pair<std::pair<int, int>, std::pair<int, int>>>>& pairs_to_ret_edges, 
                             std::map<int, std::string>& id_to_node_name, 
                             const std::string& suffix) {
    std::ofstream outFile("./result/opt_result_" + suffix + "_pair.csv");
    std::stringstream outss;

    if (!outFile.is_open()) {
        std::cerr << "file open fail" << std::endl;
        return 1;
    }
    outFile << "pair_id,start,id,end,id" << std::endl;
    for (const auto& pair2edges : pairs_to_ret_edges) {
        int pair_id = pair2edges.first;
        auto ret_edges = pair2edges.second;
        for (const auto& edge : ret_edges) {
            auto u = edge.first;
            auto v = edge.second;
            int i = u.first;
            int i_prime = u.second+1;
            int j = v.first;
            int j_prime = v.second+1;
            outss << pair_id << "," << id_to_node_name[i] << "," << i_prime << "," << id_to_node_name[j] << "," << j_prime << std::endl;
        }
    }
    outFile << outss.str();
    outFile.close();
    return 0;
}

string find_path(int pair_id, pair<int, int> start, pair<int, int> end, 
                const set<pair<pair<int, int>, pair<int, int>>>& edges,
                const map<int, string>& id_to_node_name) {
    
    // 构建邻接表
    std::map<int, vector<int>> graph;
    for (const auto& edge : edges) {
        auto src = edge.first;
        auto dst = edge.second;
        int src_id = src.first*copy_num+src.second;
        int dst_id = dst.first*copy_num+dst.second;
        graph[src_id].push_back(dst_id);
        graph[dst_id].push_back(src_id);
    }
    int start_id = start.first*copy_num+start.second;
    int end_id = end.first*copy_num+end.second;
    std::string path = id_to_node_name.at(start.first) + ";";
    int now = start_id;
    std::set<int> visited;
    
    int flag = 1;
    while (!path.empty()) {
        visited.insert(now);
        if (flag == 0) {
            cout << "dead loop in " << pair_id << " " << id_to_node_name.at(now / copy_num) << endl;
            break;
        }
        
        if (now == end_id) {
            return path;
        }
        // cout << now << endl;
        flag = 0;
        for (int neighbor : graph.at(now)) {
            if (!visited.count(neighbor)) {
                int id = neighbor / copy_num;
                int prime = neighbor % copy_num;
                // cout << id_to_node_name.at(id) << endl;
                if (neighbor == end_id) {
                    path += id_to_node_name.at(id);
                } else {
                    path += (id_to_node_name.at(id) + "_" + to_string(prime+1)) + ";";
                }
                now = neighbor;
                flag = 1;
                break;
            }
        }
        
    }
    return "";
}

int print_pairs_to_path(std::vector<std::pair<int, int>>& pairs, 
                        std::map<int, std::set<std::pair<std::pair<int, int>, std::pair<int, int>>>>& pairs_to_ret_edges, 
                        std::map<int, std::string>& id_to_node_name, 
                        const std::string& suffix) {
    std::ofstream outFile("./result/opt_result_" + suffix + "_pair_path.csv");
    std::stringstream outss;

    if (!outFile.is_open()) {
        std::cerr << "file open fail" << std::endl;
        return 1;
    }
    outFile << "id,start,end,path" << std::endl;
    for (const auto& pair2edges : pairs_to_ret_edges) {
        int pair_id = pair2edges.first;
        auto edges = pair2edges.second;
        const auto& pair = pairs[pair_id];
        int start = pair.first;
        int end = pair.second;
        string path = find_path(pair_id, {start, 0}, {end, 0}, edges, id_to_node_name);
        outss << pair_id+1 << "," << id_to_node_name[start] << "," << id_to_node_name[end] << "," << path << std::endl;
        // outss << id << "," << id_to_node_name[start] << "-1" << "," << id_to_node_name[end] << "-1" << "," << path << std::endl;
    }
    outFile << outss.str();
    outFile.close();
    return 0;
}

int print_ret_edges(std::set<std::pair<int, int>>& ret_edges, 
                    std::map<int, std::string>& id_to_node_name, 
                    const std::string& suffix, bool is_prime = false) {
    std::string prime_suffix = is_prime ? "_prime" : "";
    std::ofstream outFileall1("./result/opt_result_" + suffix + prime_suffix + ".csv");
    std::ofstream outFileall2("./result/opt_result_" + suffix + prime_suffix + "_name.csv");
    std::stringstream outFile1;
    std::stringstream outFile2;

    if (!outFileall1.is_open() || !outFileall2.is_open()) {
        std::cerr << "file open fail" << std::endl;
        return 1;
    }

    outFile1 << "start,end" << std::endl;
    outFile2 << (is_prime ? "start,id,end,id" : "start,end") << std::endl;

    for (const auto& edge : ret_edges) {
        int u = edge.first;
        int v = edge.second;

        if (is_prime) {
            int i = u / copy_num;
            int i_prime = u % copy_num + 1;
            int j = v / copy_num;
            int j_prime = v % copy_num + 1;
            outFile1 << u << "," << v << std::endl;
            outFile2 << id_to_node_name[i] << "," << i_prime << "," << id_to_node_name[j] << "," << j_prime << std::endl;
        } else {
            outFile1 << u << "," << v << std::endl;
            outFile2 << id_to_node_name[u] << "," << id_to_node_name[v] << std::endl;
        }
    }

    outFileall1 << outFile1.str();
    outFileall2 << outFile2.str();
    outFileall1.close();
    outFileall2.close();
    return 0;
}

int main(int argc, char* argv[]) {
    const auto app_start = std::chrono::steady_clock::now();
    RunConfig config;
    try {
        config = parseArguments(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Argument error: " << error.what() << std::endl;
        printUsage(argv[0]);
        return 2;
    }
    const std::string file_path1 = config.edges_path;
    const std::string file_path2 = config.pairs_path;
    std::map<std::string, int> node_name_to_id;
    std::map<int, std::string> id_to_node_name;
    std::vector<std::pair<int, int>> edges;
    std::map<std::pair<int, int>, int> edges_to_weight;
    std::map<int, std::set<int>> adj_list;
    std::set<int> center_nodes;
    std::vector<std::pair<int, int>> center_edges; 
    std::vector<std::pair<int, int>> entry_edges;
    std::map<int, std::vector<std::pair<int, int>>> cpid_to_entry_edges;
    std::vector<std::pair<int, int>> leaf_edges;
    std::map<int, int> leaf_to_center;
    std::vector<std::pair<int, int>> ori_pairs;
    std::vector<std::pair<int, int>> pairs;
    std::map<std::pair<int, int>, double> pairs_to_weight;
    std::map<std::pair<int, int>, std::pair<int, int>> pairs_to_center_pairs;
    std::set<std::pair<int, int>> center_pairs;
    std::map<int, std::pair<int, int>> id_to_center_pairs;
    std::map<std::pair<int, int>, int> center_pairs_to_id;
    std::map<std::pair<int, int>, double> center_pairs_to_weight;
    std::map<int, int> id_to_copynums;
    // std::map<std::pair<int, int>, std::set<std::pair<int, int>>> pairs_to_ret_edges; // k--n
    std::map<std::pair<int, int>, std::set<std::pair<int, int>>> cpairs_to_ret_edges_prime; // k--copy_num*n center
    std::set<std::pair<int, int>> cret_edges_prime;
    std::map<int, std::set<std::pair<std::pair<int, int>, std::pair<int, int>>>> id_to_ret_edges_prime;    // std::set<std::pair<int, int>> ret_edges; // n
    std::set<std::pair<int, int>> ret_edges_prime; // copy_num*n
    std::map<int, std::set<int>> ret_edges_prime_adj_list;
    std::map<int, int> k2prime;
    std::cout << "Reading edges" << std::endl;
    int node_num = read_csv_to_edges(file_path1, node_name_to_id, id_to_node_name, edges, edges_to_weight, 
                                        center_nodes, center_edges, leaf_edges, leaf_to_center, entry_edges);
    if (node_num == -1) {
        return 1;
    }
    std::cout << "Reading pairs" << std::endl;
    if (read_csv_to_pairs(file_path2, node_name_to_id, pairs, ori_pairs, pairs_to_weight, pairs_to_center_pairs, center_pairs, 
                            id_to_center_pairs, center_pairs_to_id, center_pairs_to_weight, leaf_to_center,
                            entry_edges, cpid_to_entry_edges) == -1) {
        return 1;
    }
    if (center_nodes.empty() || center_edges.empty() || id_to_center_pairs.empty()) {
        std::cerr << "Input did not produce a non-empty center graph and pair set" << std::endl;
        return 1;
    }
    // set_dflt_copynums(node_name_to_id, id_to_copynums);
    // if (read_csv_to_copynums(file_path3, node_name_to_id, id_to_copynums) == -1) {
    //     return 1;
    // }
    int keep_count = id_to_center_pairs.size() / div_part;
    auto it = id_to_center_pairs.begin();
    std::advance(it, keep_count);
    id_to_center_pairs.erase(it, id_to_center_pairs.end());

    adj_list = convertToAdjacencyList(center_edges);
    std::cout << "Solving problem" << std::endl;
    double obj = 0;
    SolverMetrics metrics;
    SCIP_RETCODE solve_code = SolveMIPProblem(center_nodes, center_edges, edges_to_weight, id_to_center_pairs,
                                center_pairs_to_weight, adj_list, cpairs_to_ret_edges_prime, cret_edges_prime,
                                k2prime, id_to_copynums, entry_edges, cpid_to_entry_edges, config, metrics, obj);
    if (solve_code != SCIP_OKAY) {
        std::cerr << "SCIP failed with return code " << solve_code << std::endl;
        return 1;
    }
    // try {
    //     obj = SolveMIPProblem(center_nodes, center_edges, edges_to_weight, id_to_center_pairs, center_pairs_to_weight, 
    //                             adj_list, cpairs_to_ret_edges_prime, cret_edges_prime, k2prime, id_to_copynums, 
    //                             entry_edges, cpid_to_entry_edges);
    //     // SolveMIPProblemFromFile(pairs_to_ret_edges, ret_edges, ret_edges_prime, "model-opt.lp");
    // } catch (std::exception e) {
    //     cout << e.what() << endl;
    // } catch (...) {
    //   cout << "Unknown exception occurs!" << endl;
    // }
    double total_obj = std::numeric_limits<double>::quiet_NaN();
    if (metrics.status == "optimal" && metrics.has_solution) {
        ret_edges_prime_adj_list = convertToAdjacencyList(ret_edges_prime);
        std::cout << "Checking cycles" << std::endl;
        auto cyclePath = detectCycle(ret_edges_prime_adj_list);
        if (cyclePath) {
            std::cout << "Error, Cycle in result: " << std::endl;
            for (int node : *cyclePath) {
                std::cout << node << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "No cycle" << std::endl;
        }
        total_obj = ret_compose(obj, edges, edges_to_weight, pairs, pairs_to_weight, leaf_to_center,
                                pairs_to_center_pairs, center_pairs_to_id, id_to_ret_edges_prime,
                                cpairs_to_ret_edges_prime, ret_edges_prime, k2prime);
        std::cout << "Total best res: " << total_obj << std::endl;
        std::cout << "Outputing results" << std::endl;
    }
    // print_pairs_to_ret_edges(id_to_ret_edges_prime, id_to_node_name, suffix);
    // print_pairs_to_path(ori_pairs, id_to_ret_edges_prime, id_to_node_name, suffix);
    // print_pairs_to_ret_edges(pairs_to_ret_edges, id_to_node_name, suffix, false);
    // print_ret_edges(ret_edges_prime, id_to_node_name, suffix, true);
    // print_ret_edges(ret_edges, id_to_node_name, suffix, false);
    // print_ret_edges_prime_name(ret_edges, id_to_node_name, suffix);
    // print_model_cip(node_num, edges, edges_to_weight, pairs, pairs_to_weight, adj_list);
    metrics.wall_clock_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - app_start).count();
    try {
        writeRunJson(config.output_json, config, metrics, total_obj);
    } catch (const std::exception& error) {
        std::cerr << "Output error: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "Execute success" << std::endl;
    return 0;
}
