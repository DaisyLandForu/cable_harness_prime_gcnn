#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rl/gcnn_model_runner.hpp"

namespace {

template <typename T>
void readValues(std::ifstream& stream, std::vector<T>& values, std::size_t count) {
    values.resize(count);
    stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    if (!stream) {
        throw std::runtime_error("truncated GCNN parity fixture");
    }
}

rlbranch::GraphObservation readFixture(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open fixture: " + path);
    }
    std::array<char, 8> magic{};
    stream.read(magic.data(), magic.size());
    if (std::string(magic.data(), magic.size()) != "GCNNP001") {
        throw std::runtime_error("invalid GCNN parity fixture magic");
    }
    std::array<std::uint64_t, 4> dimensions{};
    stream.read(
        reinterpret_cast<char*>(dimensions.data()),
        dimensions.size() * sizeof(std::uint64_t));
    if (!stream) {
        throw std::runtime_error("truncated GCNN fixture dimensions");
    }
    rlbranch::GraphObservation observation;
    observation.row_count = dimensions[0];
    observation.variable_count = dimensions[1];
    const std::size_t edge_count = dimensions[2];
    const std::size_t candidate_count = dimensions[3];
    readValues(
        stream,
        observation.row_features,
        observation.row_count * rlbranch::kConstraintFeatureCount);
    readValues(
        stream,
        observation.variable_features,
        observation.variable_count * rlbranch::kCandidateVariableFeatureCount);
    readValues(stream, observation.edge_row_indices, edge_count);
    readValues(stream, observation.edge_variable_indices, edge_count);
    readValues(
        stream,
        observation.edge_features,
        edge_count * rlbranch::kEdgeFeatureCount);
    stream.read(
        reinterpret_cast<char*>(observation.global_features.data()),
        observation.global_features.size() * sizeof(float));
    if (!stream) {
        throw std::runtime_error("truncated GCNN global features");
    }
    readValues(
        stream,
        observation.variable_categories,
        observation.variable_count * rlbranch::kVariableCategoryCount);
    readValues(
        stream,
        observation.row_categories,
        observation.row_count * rlbranch::kConstraintCategoryCount);
    readValues(stream, observation.candidate_indices, candidate_count);
    return observation;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: gcnn_model_runner_parity MODEL FIXTURE DEVICE OUTPUT\n";
        return 2;
    }
    try {
        rlbranch::GraphObservation observation = readFixture(argv[2]);
        rlbranch::GcnnModelRunner runner(argv[1], argv[3]);
        if (!runner.ready()) {
            throw std::runtime_error(runner.error());
        }
        const std::vector<float> values = runner.predict(observation);
        std::ofstream output(argv[4]);
        output.precision(17);
        for (float value : values) {
            output << value << '\n';
        }
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
    return 0;
}
