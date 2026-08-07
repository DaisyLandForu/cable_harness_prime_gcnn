#include <array>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rl/model_runner.hpp"

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "usage: model_runner_parity MODEL DEVICE INPUT OUTPUT\n";
        return 2;
    }
    try {
        std::ifstream input(argv[3]);
        if (!input) {
            throw std::runtime_error("cannot open parity input");
        }
        std::size_t candidate_count = 0;
        input >> candidate_count;
        if (candidate_count == 0) {
            throw std::runtime_error("parity input has no candidates");
        }

        std::vector<float> variable_features(
            candidate_count * rlbranch::kCandidateVariableFeatureCount);
        std::array<float, rlbranch::kGlobalFeatureCount> global_features{};
        std::vector<float> category_features(
            candidate_count * rlbranch::kVariableCategoryCount);
        for (std::size_t row = 0; row < candidate_count; ++row) {
            for (int column = 0; column < rlbranch::kCandidateVariableFeatureCount; ++column) {
                input >> variable_features[row * rlbranch::kCandidateVariableFeatureCount + column];
            }
            for (int column = 0; column < rlbranch::kGlobalFeatureCount; ++column) {
                float value = 0.0F;
                input >> value;
                if (row == 0) {
                    global_features[column] = value;
                }
            }
            for (int column = 0; column < rlbranch::kVariableCategoryCount; ++column) {
                input >> category_features[row * rlbranch::kVariableCategoryCount + column];
            }
        }
        if (!input) {
            throw std::runtime_error("parity input ended before all features were read");
        }

        rlbranch::ModelRunner runner(argv[1], argv[2]);
        if (!runner.ready()) {
            throw std::runtime_error(runner.error());
        }
        const std::vector<float> q_values = runner.predict(
            variable_features,
            global_features,
            category_features,
            candidate_count);
        std::ofstream output(argv[4]);
        if (!output) {
            throw std::runtime_error("cannot open parity output");
        }
        output << std::setprecision(17);
        for (float q_value : q_values) {
            output << q_value << '\n';
        }
    } catch (const std::exception& exception) {
        std::cerr << "model runner parity failed: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
