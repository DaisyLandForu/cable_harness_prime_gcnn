#include "rl/scip_profile.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <openssl/sha.h>

#include <scip/scip_numerics.h>
#include <scip/scip_param.h>
#include <scip/scip_sol.h>
#include <scip/type_paramset.h>
#include <scip/scip_nodesel.h>
#include <scip/pub_nodesel.h>

namespace rlbranch {
namespace {

std::string toHex(const unsigned char* data, std::size_t size) {
    static const char* kDigits = "0123456789abcdef";
    std::string hex(size * 2, '0');
    for (std::size_t index = 0; index < size; ++index) {
        hex[index * 2] = kDigits[data[index] >> 4];
        hex[index * 2 + 1] = kDigits[data[index] & 0x0F];
    }
    return hex;
}

std::string trim(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

std::string canonicalizeValue(SCIP* scip, SCIP_PARAM* param) {
    switch (SCIPparamGetType(param)) {
    case SCIP_PARAMTYPE_BOOL:
        return SCIPparamGetBool(param) ? "TRUE" : "FALSE";
    case SCIP_PARAMTYPE_INT:
        return std::to_string(SCIPparamGetInt(param));
    case SCIP_PARAMTYPE_LONGINT:
        return std::to_string(static_cast<long long>(SCIPparamGetLongint(param)));
    case SCIP_PARAMTYPE_REAL: {
        const double value = SCIPparamGetReal(param);
        if (SCIPisEQ(scip, value, std::round(value))) {
            return std::to_string(static_cast<long long>(std::llround(value)));
        }
        std::ostringstream out;
        out.precision(15);
        out << value;
        return out.str();
    }
    case SCIP_PARAMTYPE_CHAR:
        return std::string("'") + SCIPparamGetChar(param) + "'";
    case SCIP_PARAMTYPE_STRING:
        return SCIPparamGetString(param) == nullptr ? "" : SCIPparamGetString(param);
    default:
        throw std::runtime_error("unsupported SCIP parameter type");
    }
}

}  // namespace

std::string sha256File(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot read file for sha256: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return sha256Text(buffer.str());
}

std::string sha256Text(const std::string& text) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(
        reinterpret_cast<const unsigned char*>(text.data()),
        text.size(),
        digest);
    return toHex(digest, SHA256_DIGEST_LENGTH);
}

ScipProfile loadScipProfile(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open SCIP profile: " + path);
    }
    ScipProfile profile;
    profile.path = path;
    profile.file_sha256 = sha256File(path);
    std::string line;
    while (std::getline(stream, line)) {
        const std::string stripped = trim(line);
        if (stripped.empty() || stripped[0] == '#') {
            continue;
        }
        const auto split = stripped.find('=');
        if (split == std::string::npos) {
            throw std::runtime_error("invalid SCIP profile line: " + line);
        }
        const std::string name = trim(stripped.substr(0, split));
        const std::string value = trim(stripped.substr(split + 1));
        if (name.empty() || value.empty()) {
            throw std::runtime_error("invalid SCIP profile entry: " + line);
        }
        profile.entries.emplace_back(name, value);
    }
    if (profile.entries.empty()) {
        throw std::runtime_error("SCIP profile has no parameters: " + path);
    }
    return profile;
}

SCIP_RETCODE applyScipProfile(SCIP* scip, const ScipProfile& profile) {
    SCIP_CALL(SCIPreadParams(scip, profile.path.c_str()));
    return SCIP_OKAY;
}

namespace {

const char* const kEffectiveSearchParams[] = {
    "branching/preferbinary",
    "estimation/restarts/restartpolicy",
    "heuristics/alns/freq",
    "heuristics/alns/priority",
    "heuristics/rens/freq",
    "heuristics/rens/priority",
    "limits/gap",
    "limits/nodes",
    "limits/restarts",
    "limits/time",
    "lp/threads",
    "nodeselection/dfs/stdpriority",
    "nodeselection/estimate/stdpriority",
    "parallel/maxnthreads",
    "parallel/minnthreads",
    "presolving/maxrestarts",
    "randomization/lpseed",
    "randomization/permutationseed",
    "randomization/randomseedshift",
    "separating/maxrounds",
};

}  // namespace

std::string dumpAppliedProfile(SCIP* scip, const ScipProfile& profile) {
    auto entries = profile.entries;
    std::sort(entries.begin(), entries.end());
    std::ostringstream out;
    for (const auto& entry : entries) {
        SCIP_PARAM* param = SCIPgetParam(scip, entry.first.c_str());
        if (param == nullptr) {
            throw std::runtime_error("missing SCIP parameter after apply: " + entry.first);
        }
        out << entry.first << " = " << canonicalizeValue(scip, param) << '\n';
    }
    return out.str();
}

std::string dumpEffectiveSearchParams(SCIP* scip, bool include_seeds) {
    std::vector<std::string> names;
    for (const char* raw : kEffectiveSearchParams) {
        const std::string name(raw);
        if (!include_seeds && (
            name == "randomization/lpseed"
            || name == "randomization/permutationseed"
            || name == "randomization/randomseedshift")) {
            continue;
        }
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    std::ostringstream out;
    for (const auto& name : names) {
        SCIP_PARAM* param = SCIPgetParam(scip, name.c_str());
        if (param == nullptr) {
            throw std::runtime_error("missing effective SCIP parameter: " + name);
        }
        out << name << " = " << canonicalizeValue(scip, param) << '\n';
    }
    return out.str();
}

void assertProductionInvariants(SCIP* scip) {
    int min_threads = 0;
    int max_threads = 0;
    int lp_threads = 0;
    int maxrounds = 0;
    int dfs_priority = 0;
    int estimate_priority = 0;
    int restart_limit = 0;
    char restart_policy = 'n';
    SCIP_Bool prefer_binary = FALSE;
    SCIP_CALL_ABORT(SCIPgetIntParam(scip, "parallel/minnthreads", &min_threads));
    SCIP_CALL_ABORT(SCIPgetIntParam(scip, "parallel/maxnthreads", &max_threads));
    SCIP_CALL_ABORT(SCIPgetIntParam(scip, "lp/threads", &lp_threads));
    SCIP_CALL_ABORT(SCIPgetIntParam(scip, "separating/maxrounds", &maxrounds));
    SCIP_CALL_ABORT(SCIPgetIntParam(scip, "nodeselection/dfs/stdpriority", &dfs_priority));
    SCIP_CALL_ABORT(SCIPgetIntParam(
        scip, "nodeselection/estimate/stdpriority", &estimate_priority));
    SCIP_CALL_ABORT(SCIPgetBoolParam(scip, "branching/preferbinary", &prefer_binary));
    SCIP_CALL_ABORT(SCIPgetCharParam(
        scip, "estimation/restarts/restartpolicy", &restart_policy));
    SCIP_CALL_ABORT(SCIPgetIntParam(scip, "limits/restarts", &restart_limit));
    if (min_threads != 1 || max_threads != 1 || lp_threads != 1) {
        throw std::runtime_error("project-production-v1 requires SCIP/LP threads=1");
    }
    if (maxrounds == 0) {
        throw std::runtime_error("project-production-v1 must not disable cuts");
    }
    if (dfs_priority >= estimate_priority) {
        throw std::runtime_error("project-production-v1 must keep estimate above DFS");
    }
    if (!prefer_binary) {
        throw std::runtime_error("project-production-v1 requires branching/preferbinary");
    }
    if (restart_policy == 'n') {
        throw std::runtime_error("project-production-v1 must not disable restarts");
    }
    if (restart_limit == 0) {
        throw std::runtime_error("project-production-v1 must not set limits/restarts=0");
    }
}

void requireEstimateNodeSelector(SCIP* scip) {
    const std::string name = activeNodeSelectorName(scip);
    if (name != "estimate") {
        throw std::runtime_error(
            "project-production-v1 requires active node selector estimate, got: "
            + (name.empty() ? "<none>" : name));
    }
}

std::string activeNodeSelectorName(SCIP* scip) {
    SCIP_NODESEL* selector = SCIPgetNodesel(scip);
    if (selector == nullptr) {
        return "";
    }
    const char* name = SCIPnodeselGetName(selector);
    return name == nullptr ? "" : name;
}

}  // namespace rlbranch
