#pragma once

#include <string>
#include <utility>
#include <vector>

#include <scip/scip.h>

namespace rlbranch {

struct ScipProfile {
    std::string path;
    std::string file_sha256;
    std::vector<std::pair<std::string, std::string>> entries;
};

std::string sha256File(const std::string& path);
std::string sha256Text(const std::string& text);

ScipProfile loadScipProfile(const std::string& path);
SCIP_RETCODE applyScipProfile(SCIP* scip, const ScipProfile& profile);
std::string dumpAppliedProfile(SCIP* scip, const ScipProfile& profile);
void assertProductionInvariants(SCIP* scip);
std::string activeNodeSelectorName(SCIP* scip);

}  // namespace rlbranch
