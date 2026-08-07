#include <iostream>
#include <random>
#include <vector>

#include <scip/scip.h>

#include "rl/rl_branchrule.hpp"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool passed = true;

    passed &= check(rlbranch::aviationVariableCategory("t_m_3_0") == 0,
        "m variables must use the m category");
    passed &= check(rlbranch::aviationVariableCategory("t_t_absf_1_2_0") == 3,
        "repeated transformed prefixes must be stripped");
    passed &= check(rlbranch::aviationVariableCategory("auxiliary") == 5,
        "unknown variables must use the other category");
    std::vector<rlbranch::BranchCandidate> empty;
    std::mt19937 empty_engine(0);
    passed &= check(
        rlbranch::selectRandomCandidate(empty, empty_engine) == -1,
        "empty random candidate set must return -1");

    SCIP* scip = nullptr;
    SCIP_CALL_ABORT(SCIPcreate(&scip));
    SCIP_CALL_ABORT(SCIPcreateProbBasic(scip, "custom_branchrule_unit"));

    std::vector<SCIP_VAR*> variables(3, nullptr);
    const double objectives[] = {1.0, 2.0, 3.0};
    for (int index = 0; index < 3; ++index) {
        SCIP_CALL_ABORT(SCIPcreateVarBasic(
            scip,
            &variables[index],
            ("candidate_" + std::to_string(index)).c_str(),
            0.0,
            1.0,
            objectives[index],
            SCIP_VARTYPE_BINARY));
        SCIP_CALL_ABORT(SCIPaddVar(scip, variables[index]));
    }

    std::vector<rlbranch::BranchCandidate> candidates(3);
    const double fractionalities[] = {0.10, 0.49, 0.30};
    for (int index = 0; index < 3; ++index) {
        candidates[index].variable = variables[index];
        candidates[index].lp_value = fractionalities[index];
        candidates[index].fractionality = fractionalities[index];
        candidates[index].candidate_index = index;
        candidates[index].variable_index = SCIPvarGetProbindex(variables[index]);
    }

    passed &= check(
        rlbranch::selectMostInfeasibleCandidate(scip, candidates) == 1,
        "most-infeasible must select fractionality 0.49");

    rlbranch::BranchruleStats fallback_stats;
    {
        rlbranch::CustomBranchrule rule(
            scip,
            rlbranch::CustomBranchingStrategy::Random,
            0,
            "",
            &fallback_stats);
        SCIP_RESULT pseudo_result = SCIP_DIDNOTFIND;
        SCIP_CALL_ABORT(rule.scip_execps(scip, nullptr, FALSE, &pseudo_result));
        passed &= check(
            pseudo_result == SCIP_DIDNOTRUN,
            "unsupported pseudo branching must fall through to lower-priority rules");
    }

    std::mt19937 first_engine(17);
    std::mt19937 second_engine(17);
    for (int draw = 0; draw < 100; ++draw) {
        const int first = rlbranch::selectRandomCandidate(candidates, first_engine);
        const int second = rlbranch::selectRandomCandidate(candidates, second_engine);
        passed &= check(first == second, "same seed must reproduce random selections");
        passed &= check(first >= 0 && first < 3, "random selection must stay inside action set");
    }

    for (SCIP_VAR*& variable : variables) {
        SCIP_CALL_ABORT(SCIPreleaseVar(scip, &variable));
    }
    SCIP_CALL_ABORT(SCIPfree(&scip));
    BMScheckEmptyMemory();

    if (!passed) {
        return 1;
    }
    std::cout << "custom branchrule selection tests passed\n";
    return 0;
}
