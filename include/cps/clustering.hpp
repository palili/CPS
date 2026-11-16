#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "nlohmann/json.hpp"

namespace cps::clustering {

constexpr int FPGA_MAX_CLB  = 38850;
constexpr int FPGA_MAX_BRAM = 2100;
constexpr int FPGA_MAX_DSP  = 2800;

struct ClusterTask {
    double start_earliest = 0.0;
    double start_latest   = 0.0;
    double exec_time      = 0.0;
    int clb  = 0;
    int dsp  = 0;
    int bram = 0;

    ClusterTask() = default;
    ClusterTask(double se, double sl, double et, int c = 0, int d = 0, int b = 0)
        : start_earliest(se), start_latest(sl), exec_time(et), clb(c), dsp(d), bram(b) {}
};

struct TimeWindow {
    double start = 0.0;
    double end   = 0.0;

    TimeWindow() = default;
    TimeWindow(double s, double e) : start(s), end(e) {}

    double length() const { return end - start; }

    double overlap(const TimeWindow& other) const {
        const double overlap_start = std::max(start, other.start);
        const double overlap_end   = std::min(end, other.end);
        if (overlap_end <= overlap_start) return 0.0;
        const double overlap_length    = overlap_end - overlap_start;
        const double min_window_length = std::min(length(), other.length());
        return min_window_length > 0.0 ? (overlap_length / min_window_length) : 0.0;
    }
};

struct HardwareResources {
    int clb  = 0;
    int dsp  = 0;
    int bram = 0;

    HardwareResources() = default;
    HardwareResources(int c, int d, int b) : clb(c), dsp(d), bram(b) {}

    HardwareResources operator+(const HardwareResources& other) const {
        return HardwareResources(clb + other.clb, dsp + other.dsp, bram + other.bram);
    }

    HardwareResources& operator+=(const HardwareResources& other) {
        clb  += other.clb;
        dsp  += other.dsp;
        bram += other.bram;
        return *this;
    }

    HardwareResources& operator-=(const HardwareResources& other) {
        clb  -= other.clb;
        dsp  -= other.dsp;
        bram -= other.bram;
        return *this;
    }

    bool withinFPGALimits() const {
        return clb <= FPGA_MAX_CLB && dsp <= FPGA_MAX_DSP && bram <= FPGA_MAX_BRAM;
    }
};

struct ClusteringResult {
    std::vector<int> communities;
    std::unordered_map<int, TimeWindow> community_windows;
    std::vector<std::vector<int>> node_to_original;
    std::unordered_map<int, HardwareResources> community_resources;
};

struct CommunityLoadParams {
    double wc = 1.0 / 3.0;
    double wb = 1.0 / 3.0;
    double wd = 1.0 / 3.0;
    double rho1 = 1.0 / 3.0;
    double rho2 = 1.0 / 3.0;
    double rho3 = 1.0 / 3.0;
    double quantile_level = 0.9;
    int max_split_rounds = 3;
};

struct CommunityLoadMetric {
    double R_size = 0.0;
    double T_stay = 0.0;
    double B_raw  = 0.0;
    double R_hat  = 0.0;
    double T_hat  = 0.0;
    double B_hat  = 0.0;
    double L      = 0.0;
};

struct QuotientCycleInfo {
    std::vector<int> community_ids;  // 1-based community IDs (matching exported JSON)
    HardwareResources combined_resources;
    bool exceeds_fpga_capacity = false;
};

struct QuotientCycleDiagnostics {
    int num_nontrivial_sccs = 0;
    bool any_exceeds_fpga_capacity = false;
    std::vector<QuotientCycleInfo> cycles;
};

class TaskClustering {
private:
    std::vector<ClusterTask> tasks;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<int>> dep_matrix_;

    std::unordered_map<int, TimeWindow> computeInitialCommunityWindows(
        const std::vector<int>& communities) const;

    std::unordered_map<int, HardwareResources> computeCommunityResources(
        const std::vector<int>& communities,
        const std::vector<std::vector<int>>& node_to_original,
        const std::vector<ClusterTask>& original_tasks) const;

    double calculateModularityDelta(
        int node, int community,
        const std::vector<int>& communities,
        const std::vector<std::vector<double>>& edge_weights,
        double total_weight,
        const std::vector<double>& node_degrees) const;

    double calculateModularity(
        const std::vector<int>& communities,
        const std::vector<std::vector<double>>& edge_weights,
        double total_weight) const;

    void buildNextLevelGraph(
        const std::vector<int>& communities,
        const std::vector<std::vector<double>>& current_weights,
        std::vector<ClusterTask>& next_tasks,
        std::vector<std::vector<double>>& next_weights,
        const std::vector<std::vector<int>>& current_node_mapping,
        std::vector<std::vector<int>>& next_node_mapping) const;

    std::map<int, CommunityLoadMetric> computeCommunityLoadMetrics(
        const std::map<int, std::vector<int>>& comm_to_tasks,
        const std::vector<ClusterTask>& orig_tasks,
        const CommunityLoadParams& params) const;

    double computeQuantileThreshold(
        const std::map<int, CommunityLoadMetric>& metrics,
        double quantile_level) const;

public:
    TaskClustering(const std::vector<ClusterTask>& t,
                   const std::vector<std::vector<double>>& edge_weights);

    void setDepMatrix(const std::vector<std::vector<int>>& dep) { dep_matrix_ = dep; }

    ClusteringResult performSingleRun(std::mt19937& rng) const;

    ClusteringResult performSingleLevelClustering(unsigned seed = 12345,
                                                  bool verbose = false) const;

    ClusteringResult postSplitOverloadedCommunities(
        const ClusteringResult& initial_result,
        const std::vector<ClusterTask>& orig_tasks,
        const CommunityLoadParams& params,
        bool verbose = false) const;

    std::vector<ClusteringResult> performMultiLevelClustering(
        double modularity_threshold = 0.001,
        const CommunityLoadParams& load_params = CommunityLoadParams{},
        unsigned seed = 12345,
        bool verbose = false) const;
};

bool readTaskGraph(const std::string& filename,
                   std::vector<ClusterTask>& tasks,
                   std::vector<std::vector<double>>& comm_matrix,
                   std::vector<std::vector<int>>& dep_matrix,
                   std::vector<std::vector<double>>& urgency_matrix,
                   bool dump_debug = false,
                   const std::string& debug_dir = "");

std::vector<std::vector<double>> computeIntegratedWeights(
    const std::vector<ClusterTask>& tasks,
    const std::vector<std::vector<double>>& comm_matrix,
    const std::vector<std::vector<int>>& dep_matrix,
    const std::vector<std::vector<double>>& urgency_matrix,
    double lambda1 = 0.4,
    double lambda2 = 0.3,
    double lambda3 = 0.3,
    bool dump_debug = false,
    const std::string& debug_dir = "");

QuotientCycleDiagnostics analyzeQuotientCycles(
    const ClusteringResult& final_result,
    const std::vector<ClusterTask>& original_tasks,
    const std::vector<std::vector<int>>& dep_matrix);

nlohmann::json buildClusteringJson(
    const std::vector<ClusteringResult>& results,
    const std::vector<ClusterTask>& original_tasks,
    const std::vector<std::vector<int>>& dep_matrix,
    double lambda1, double lambda2, double lambda3,
    unsigned cluster_seed,
    double clustering_elapsed_seconds,
    QuotientCycleDiagnostics* diag_out = nullptr);

nlohmann::json runClusteringPipeline(
    const std::string& dag_file,
    unsigned cluster_seed,
    const std::string& json_output_path = "",
    bool dump_debug = false,
    const std::string& debug_dir = "",
    bool verbose = true);

} // namespace cps::clustering
