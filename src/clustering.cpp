#include "cps/clustering.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace cps::clustering {

using json = nlohmann::json;

namespace {

void normalizeMatrix(std::vector<std::vector<double>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return;
    double max_val = matrix[0][0];
    for (const auto& row : matrix) {
        for (double val : row) max_val = std::max(max_val, val);
    }
    if (max_val == 0.0) return;
    for (auto& row : matrix) {
        for (double& val : row) val /= max_val;
    }
}

double computeGlobalCP(const std::vector<ClusterTask>& tasks,
                       const std::vector<std::vector<int>>& dep) {
    const int n = static_cast<int>(tasks.size());
    if (n == 0) return 0.0;
    if (dep.empty() || static_cast<int>(dep.size()) < n) {
        double mx = 0.0;
        for (const auto& t : tasks) mx = std::max(mx, t.exec_time);
        return mx;
    }
    std::vector<int> in_deg(n, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (dep[i][j]) ++in_deg[j];

    std::queue<int> q;
    for (int i = 0; i < n; ++i)
        if (in_deg[i] == 0) q.push(i);

    std::vector<double> dp(n);
    for (int i = 0; i < n; ++i) dp[i] = tasks[i].exec_time;

    while (!q.empty()) {
        const int u = q.front();
        q.pop();
        for (int v = 0; v < n; ++v) {
            if (!dep[u][v]) continue;
            dp[v] = std::max(dp[v], dp[u] + tasks[v].exec_time);
            if (--in_deg[v] == 0) q.push(v);
        }
    }
    return *std::max_element(dp.begin(), dp.end());
}

double computeSubgraphCP(const std::vector<int>& nodes,
                         const std::vector<ClusterTask>& tasks,
                         const std::vector<std::vector<int>>& dep) {
    const int m = static_cast<int>(nodes.size());
    if (m == 0) return 0.0;
    if (dep.empty() || static_cast<int>(dep.size()) < static_cast<int>(tasks.size())) {
        double mx = 0.0;
        for (int u : nodes) mx = std::max(mx, tasks[u].exec_time);
        return mx;
    }
    std::vector<int> in_deg(m, 0);
    for (int li = 0; li < m; ++li)
        for (int lj = 0; lj < m; ++lj)
            if (dep[nodes[li]][nodes[lj]]) ++in_deg[lj];

    std::queue<int> q;
    for (int li = 0; li < m; ++li)
        if (in_deg[li] == 0) q.push(li);

    std::vector<double> dp(m);
    for (int li = 0; li < m; ++li) dp[li] = tasks[nodes[li]].exec_time;

    while (!q.empty()) {
        const int lu = q.front();
        q.pop();
        for (int lv = 0; lv < m; ++lv) {
            if (!dep[nodes[lu]][nodes[lv]]) continue;
            dp[lv] = std::max(dp[lv], dp[lu] + tasks[nodes[lv]].exec_time);
            if (--in_deg[lv] == 0) q.push(lv);
        }
    }
    return *std::max_element(dp.begin(), dp.end());
}

void dumpSquareMatrixTxt(const std::string& filepath,
                         const std::string& header,
                         const std::vector<std::vector<double>>& mat) {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return;
    const size_t n = mat.size();
    ofs << header << " (" << n << "x" << n << "):\n";
    ofs << std::fixed << std::setprecision(4) << "     ";
    for (size_t j = 0; j < n; ++j)
        ofs << std::setw(8) << ("T" + std::to_string(j + 1));
    ofs << "\n";
    for (size_t i = 0; i < n; ++i) {
        ofs << "T" << std::setw(3) << (i + 1) << " ";
        for (size_t j = 0; j < n; ++j)
            ofs << std::setw(8) << mat[i][j];
        ofs << "\n";
    }
}

} // namespace

TaskClustering::TaskClustering(const std::vector<ClusterTask>& t,
                               const std::vector<std::vector<double>>& edge_weights)
    : tasks(t), weights(edge_weights) {}

std::unordered_map<int, TimeWindow> TaskClustering::computeInitialCommunityWindows(
    const std::vector<int>& communities) const {
    std::unordered_map<int, TimeWindow> community_windows;
    if (tasks.empty()) return community_windows;

    const std::set<int> unique_communities(communities.begin(), communities.end());
    for (int comm_id : unique_communities) {
        double min_start = std::numeric_limits<double>::max();
        double max_end   = std::numeric_limits<double>::lowest();
        for (size_t i = 0; i < communities.size(); ++i) {
            if (communities[i] == comm_id) {
                min_start = std::min(min_start, tasks[i].start_earliest);
                max_end   = std::max(max_end, tasks[i].start_latest + tasks[i].exec_time);
            }
        }
        if (min_start != std::numeric_limits<double>::max()) {
            community_windows[comm_id] = TimeWindow(min_start, max_end);
        }
    }
    return community_windows;
}

std::unordered_map<int, HardwareResources> TaskClustering::computeCommunityResources(
    const std::vector<int>& communities,
    const std::vector<std::vector<int>>& node_to_original,
    const std::vector<ClusterTask>& original_tasks) const {
    std::unordered_map<int, HardwareResources> community_resources;
    const std::set<int> unique_communities(communities.begin(), communities.end());
    for (int comm_id : unique_communities) {
        HardwareResources total_resources;
        for (size_t i = 0; i < communities.size(); ++i) {
            if (communities[i] == comm_id) {
                for (int original_node : node_to_original[i]) {
                    total_resources.clb  += original_tasks[original_node].clb;
                    total_resources.dsp  += original_tasks[original_node].dsp;
                    total_resources.bram += original_tasks[original_node].bram;
                }
            }
        }
        community_resources[comm_id] = total_resources;
    }
    return community_resources;
}

double TaskClustering::calculateModularityDelta(
    int node, int community,
    const std::vector<int>& communities,
    const std::vector<std::vector<double>>& edge_weights,
    double total_weight,
    const std::vector<double>& node_degrees) const {
    if (total_weight <= 0.0) return 0.0;
    double delta = 0.0;
    const double k_i = node_degrees[node];
    const int current_community = communities[node];
    if (community == current_community) return 0.0;

    for (size_t j = 0; j < communities.size(); ++j) {
        if (static_cast<int>(j) == node) continue;
        const double A_ij = edge_weights[node][j];
        const double k_j  = node_degrees[j];
        const double contrib = A_ij - (k_i * k_j) / (2.0 * total_weight);
        if (communities[j] == community) {
            delta += contrib;
        } else if (communities[j] == current_community) {
            delta -= contrib;
        }
    }
    return delta / total_weight;
}

double TaskClustering::calculateModularity(
    const std::vector<int>& communities,
    const std::vector<std::vector<double>>& edge_weights,
    double total_weight) const {
    if (total_weight <= 0.0) return 0.0;
    double modularity = 0.0;
    const size_t n = communities.size();
    std::vector<double> degrees(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            degrees[i] += edge_weights[i][j];

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (communities[i] == communities[j]) {
                modularity += edge_weights[i][j] - (degrees[i] * degrees[j]) / (2.0 * total_weight);
            }
        }
    }
    return modularity / (2.0 * total_weight);
}

void TaskClustering::buildNextLevelGraph(
    const std::vector<int>& communities,
    const std::vector<std::vector<double>>& current_weights,
    std::vector<ClusterTask>& next_tasks,
    std::vector<std::vector<double>>& next_weights,
    const std::vector<std::vector<int>>& current_node_mapping,
    std::vector<std::vector<int>>& next_node_mapping) const {
    const std::set<int> unique_communities(communities.begin(), communities.end());
    const int num_communities = static_cast<int>(unique_communities.size());

    const std::vector<int> community_list(unique_communities.begin(), unique_communities.end());
    std::map<int, int> community_to_index;
    for (int i = 0; i < num_communities; ++i)
        community_to_index[community_list[i]] = i;

    next_tasks.clear();
    next_weights.clear();
    next_node_mapping.assign(num_communities, {});

    for (int community_id : unique_communities) {
        const int super_node_index = community_to_index[community_id];
        std::vector<int> community_nodes;
        for (size_t j = 0; j < communities.size(); ++j) {
            if (communities[j] == community_id) {
                community_nodes.push_back(static_cast<int>(j));
                for (int original_node : current_node_mapping[j])
                    next_node_mapping[super_node_index].push_back(original_node);
            }
        }
        double est = std::numeric_limits<double>::max();
        double lst = std::numeric_limits<double>::max();
        double lft = std::numeric_limits<double>::lowest();
        int total_clb = 0, total_dsp = 0, total_bram = 0;
        for (int node : community_nodes) {
            est = std::min(est, tasks[node].start_earliest);
            lst = std::min(lst, tasks[node].start_latest);
            lft = std::max(lft, tasks[node].start_latest + tasks[node].exec_time);
            total_clb  += tasks[node].clb;
            total_dsp  += tasks[node].dsp;
            total_bram += tasks[node].bram;
        }
        next_tasks.emplace_back(est, lst, lft - lst, total_clb, total_dsp, total_bram);
    }

    next_weights.assign(num_communities, std::vector<double>(num_communities, 0.0));
    for (int i = 0; i < num_communities; ++i) {
        for (int j = i; j < num_communities; ++j) {
            double agg_weight = 0.0;
            const int community_i = community_list[i];
            const int community_j = community_list[j];
            if (i == j) {
                for (size_t ni = 0; ni < communities.size(); ++ni) {
                    if (communities[ni] != community_i) continue;
                    if (current_weights[ni][ni] > 0) agg_weight += current_weights[ni][ni];
                    for (size_t nj = ni + 1; nj < communities.size(); ++nj) {
                        if (communities[nj] != community_i) continue;
                        agg_weight += current_weights[ni][nj] * 2.0;
                    }
                }
                next_weights[i][j] = agg_weight;
            } else {
                for (size_t ni = 0; ni < communities.size(); ++ni) {
                    if (communities[ni] != community_i) continue;
                    for (size_t nj = 0; nj < communities.size(); ++nj) {
                        if (communities[nj] != community_j) continue;
                        agg_weight += current_weights[ni][nj];
                    }
                }
                next_weights[i][j] = next_weights[j][i] = agg_weight;
            }
        }
    }
}

ClusteringResult TaskClustering::performSingleRun(std::mt19937& rng) const {
    const size_t n = tasks.size();
    if (n == 0) return {};

    std::vector<int> communities(n);
    for (size_t i = 0; i < n; ++i) communities[i] = static_cast<int>(i);

    std::unordered_map<int, HardwareResources> community_resources;
    for (size_t i = 0; i < n; ++i) {
        community_resources[static_cast<int>(i)] =
            HardwareResources(tasks[i].clb, tasks[i].dsp, tasks[i].bram);
    }

    std::unordered_map<int, TimeWindow> community_windows =
        computeInitialCommunityWindows(communities);

    double total_weight = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            if (i == j) total_weight += weights[i][j] / 2.0;
            else        total_weight += weights[i][j];
        }
    }

    std::vector<double> node_degrees(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            node_degrees[i] += weights[i][j];

    bool improved = true;
    int iteration = 0;
    const int max_iterations = 20;
    while (improved && iteration < max_iterations) {
        improved = false;
        ++iteration;

        std::vector<size_t> node_indices(n);
        std::iota(node_indices.begin(), node_indices.end(), 0);
        std::shuffle(node_indices.begin(), node_indices.end(), rng);

        for (size_t idx = 0; idx < n; ++idx) {
            const size_t i = node_indices[idx];
            int best_community = communities[i];
            double best_delta  = 0.0;

            std::set<int> neighbor_communities;
            for (size_t j = 0; j < n; ++j) {
                if (weights[i][j] > 0.0) neighbor_communities.insert(communities[j]);
            }

            for (int comm : neighbor_communities) {
                if (comm == communities[i]) continue;
                const HardwareResources& target_res = community_resources.at(comm);
                const HardwareResources merged(
                    target_res.clb  + tasks[i].clb,
                    target_res.dsp  + tasks[i].dsp,
                    target_res.bram + tasks[i].bram);
                if (!merged.withinFPGALimits()) continue;

                const double delta_Q = calculateModularityDelta(
                    static_cast<int>(i), comm, communities, weights, total_weight, node_degrees);
                if (delta_Q > best_delta) {
                    best_delta     = delta_Q;
                    best_community = comm;
                }
            }

            if (best_community != communities[i]) {
                const int old_community = communities[i];
                const HardwareResources node_res(tasks[i].clb, tasks[i].dsp, tasks[i].bram);
                community_resources[old_community]  -= node_res;
                community_resources[best_community] += node_res;
                communities[i] = best_community;
                improved = true;
            }
        }
    }

    std::vector<std::vector<int>> node_to_original(n);
    for (size_t i = 0; i < n; ++i)
        node_to_original[i].push_back(static_cast<int>(i));

    return {communities, community_windows, node_to_original, {}};
}

ClusteringResult TaskClustering::performSingleLevelClustering(unsigned seed,
                                                              bool verbose) const {
    const int num_runs = 10;
    std::mt19937 rng(seed);

    ClusteringResult best_result;
    double best_modularity = -std::numeric_limits<double>::infinity();

    const size_t n = tasks.size();
    double total_weight = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            if (i == j) total_weight += weights[i][j] / 2.0;
            else        total_weight += weights[i][j];
        }
    }

    for (int run = 0; run < num_runs; ++run) {
        ClusteringResult current_result = performSingleRun(rng);
        const double current_modularity =
            calculateModularity(current_result.communities, weights, total_weight);
        if (current_modularity > best_modularity) {
            best_modularity = current_modularity;
            best_result     = std::move(current_result);
        }
    }

    if (verbose) {
        const std::set<int> uniq(best_result.communities.begin(), best_result.communities.end());
        std::cout << "  [Louvain] seed=" << seed
                  << " best_Q=" << std::fixed << std::setprecision(6) << best_modularity
                  << " communities=" << uniq.size() << "\n";
    }
    return best_result;
}

std::map<int, CommunityLoadMetric> TaskClustering::computeCommunityLoadMetrics(
    const std::map<int, std::vector<int>>& comm_to_tasks,
    const std::vector<ClusterTask>& orig_tasks,
    const CommunityLoadParams& params) const {
    std::map<int, CommunityLoadMetric> metrics;

    for (const auto& [cid, members] : comm_to_tasks) {
        CommunityLoadMetric m;
        double ck_clb = 0.0, ck_dsp = 0.0, ck_bram = 0.0;
        for (int t : members) {
            ck_clb  += orig_tasks[t].clb;
            ck_dsp  += orig_tasks[t].dsp;
            ck_bram += orig_tasks[t].bram;
        }
        m.R_size = params.wc * (ck_clb  / FPGA_MAX_CLB)
                 + params.wb * (ck_bram / FPGA_MAX_BRAM)
                 + params.wd * (ck_dsp  / FPGA_MAX_DSP);
        metrics[cid] = m;
    }

    const double cp_global = computeGlobalCP(orig_tasks, dep_matrix_);
    for (const auto& [cid, members] : comm_to_tasks) {
        const double cp_comm = computeSubgraphCP(members, orig_tasks, dep_matrix_);
        metrics[cid].T_stay  = (cp_global > 0.0) ? (cp_comm / cp_global) : 0.0;
    }

    std::unordered_map<int, int> task_to_comm;
    for (const auto& [cid, members] : comm_to_tasks)
        for (int t : members) task_to_comm[t] = cid;

    for (const auto& [cid, members] : comm_to_tasks) {
        int count_ext = 0;
        for (int t : members) {
            bool has_ext = false;
            if (!dep_matrix_.empty() && t < static_cast<int>(dep_matrix_.size())) {
                for (int u = 0; u < static_cast<int>(dep_matrix_.size()) && !has_ext; ++u) {
                    if (dep_matrix_[u][t]) {
                        auto it = task_to_comm.find(u);
                        if (it != task_to_comm.end() && it->second != cid) has_ext = true;
                    }
                }
            }
            if (has_ext) ++count_ext;
        }
        metrics[cid].B_raw = members.empty()
                           ? 0.0
                           : static_cast<double>(count_ext) / static_cast<double>(members.size());
    }

    double r_min = std::numeric_limits<double>::max(), r_max = std::numeric_limits<double>::lowest();
    double t_min = std::numeric_limits<double>::max(), t_max = std::numeric_limits<double>::lowest();
    double b_min = std::numeric_limits<double>::max(), b_max = std::numeric_limits<double>::lowest();
    for (const auto& [cid, m] : metrics) {
        r_min = std::min(r_min, m.R_size); r_max = std::max(r_max, m.R_size);
        t_min = std::min(t_min, m.T_stay); t_max = std::max(t_max, m.T_stay);
        b_min = std::min(b_min, m.B_raw);  b_max = std::max(b_max, m.B_raw);
    }

    auto minmax_norm = [](double v, double vmin, double vmax) -> double {
        return (vmax > vmin) ? (v - vmin) / (vmax - vmin) : 0.0;
    };

    for (auto& [cid, m] : metrics) {
        m.R_hat = minmax_norm(m.R_size, r_min, r_max);
        m.T_hat = minmax_norm(m.T_stay, t_min, t_max);
        m.B_hat = minmax_norm(m.B_raw,  b_min, b_max);
        m.L     = params.rho1 * m.R_hat + params.rho2 * m.T_hat + params.rho3 * m.B_hat;
    }
    return metrics;
}

double TaskClustering::computeQuantileThreshold(
    const std::map<int, CommunityLoadMetric>& metrics,
    double quantile_level) const {
    std::vector<double> loads;
    loads.reserve(metrics.size());
    for (const auto& [cid, m] : metrics) loads.push_back(m.L);
    if (loads.empty()) return 0.0;

    std::sort(loads.begin(), loads.end());
    const double idx = quantile_level * static_cast<double>(static_cast<int>(loads.size()) - 1);
    int lo = static_cast<int>(std::floor(idx));
    int hi = static_cast<int>(std::ceil(idx));
    if (lo >= static_cast<int>(loads.size())) lo = static_cast<int>(loads.size()) - 1;
    if (hi >= static_cast<int>(loads.size())) hi = static_cast<int>(loads.size()) - 1;
    if (lo == hi) return loads[lo];
    const double frac = idx - lo;
    return loads[lo] * (1.0 - frac) + loads[hi] * frac;
}

ClusteringResult TaskClustering::postSplitOverloadedCommunities(
    const ClusteringResult& initial_result,
    const std::vector<ClusterTask>& orig_tasks,
    const CommunityLoadParams& params,
    bool verbose) const {
    const int n = static_cast<int>(orig_tasks.size());
    std::vector<int> task_community(n, -1);
    for (size_t i = 0; i < initial_result.communities.size(); ++i) {
        const int cid = initial_result.communities[i];
        for (int orig : initial_result.node_to_original[i])
            task_community[orig] = cid;
    }

    std::map<int, std::vector<int>> comm_to_tasks;
    for (int i = 0; i < n; ++i)
        if (task_community[i] >= 0) comm_to_tasks[task_community[i]].push_back(i);

    int next_cid = 0;
    for (const auto& [cid, members] : comm_to_tasks)
        next_cid = std::max(next_cid, cid + 1);

    for (int round = 0; round < params.max_split_rounds; ++round) {
        const auto metrics = computeCommunityLoadMetrics(comm_to_tasks, orig_tasks, params);
        const double tau_L = computeQuantileThreshold(metrics, params.quantile_level);

        std::vector<int> overloaded;
        for (const auto& [cid, m] : metrics) {
            if (m.L > tau_L) overloaded.push_back(cid);
        }
        if (overloaded.empty()) break;

        for (int cid : overloaded) {
            if (comm_to_tasks.find(cid) == comm_to_tasks.end()) continue;
            const std::vector<int> local_to_orig = comm_to_tasks[cid];
            const int m = static_cast<int>(local_to_orig.size());
            if (m <= 1) continue;

            std::vector<ClusterTask> sub_tasks;
            sub_tasks.reserve(m);
            for (int lo = 0; lo < m; ++lo)
                sub_tasks.push_back(orig_tasks[local_to_orig[lo]]);

            std::vector<std::vector<double>> sub_weights(m, std::vector<double>(m, 0.0));
            for (int li = 0; li < m; ++li)
                for (int lj = 0; lj < m; ++lj)
                    sub_weights[li][lj] = weights[local_to_orig[li]][local_to_orig[lj]];

            TaskClustering sub_clust(sub_tasks, sub_weights);
            std::mt19937 rng(42u + static_cast<unsigned>(cid));
            const ClusteringResult sub_res = sub_clust.performSingleRun(rng);

            const std::set<int> sub_comms_set(sub_res.communities.begin(),
                                              sub_res.communities.end());
            if (static_cast<int>(sub_comms_set.size()) <= 1) {
                std::vector<int> sorted = local_to_orig;
                std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
                    return (orig_tasks[a].clb + orig_tasks[a].dsp + orig_tasks[a].bram)
                         > (orig_tasks[b].clb + orig_tasks[b].dsp + orig_tasks[b].bram);
                });
                const int half    = static_cast<int>(sorted.size()) / 2;
                const int new_cid = next_cid++;
                std::vector<int> kept, split_off;
                for (int ki = 0; ki < static_cast<int>(sorted.size()); ++ki) {
                    if (ki < half) split_off.push_back(sorted[ki]);
                    else           kept.push_back(sorted[ki]);
                }
                comm_to_tasks[cid]     = kept;
                comm_to_tasks[new_cid] = split_off;
                for (int t : split_off) task_community[t] = new_cid;
            } else {
                std::map<int, int> sub_to_new;
                bool first = true;
                for (int sc : sub_comms_set) {
                    if (first) { sub_to_new[sc] = cid; first = false; }
                    else         sub_to_new[sc] = next_cid++;
                }
                std::map<int, std::vector<int>> new_buckets;
                for (int li = 0; li < m; ++li) {
                    const int orig         = local_to_orig[li];
                    const int new_cid_task = sub_to_new[sub_res.communities[li]];
                    task_community[orig]   = new_cid_task;
                    new_buckets[new_cid_task].push_back(orig);
                }
                comm_to_tasks.erase(cid);
                for (const auto& [nc, bucket] : new_buckets)
                    comm_to_tasks[nc] = bucket;
            }
        }
        if (verbose) {
            std::cout << "  [Post-Split] round=" << (round + 1)
                      << " tau_L=" << std::fixed << std::setprecision(4) << tau_L
                      << " overloaded=" << overloaded.size()
                      << " communities=" << comm_to_tasks.size() << "\n";
        }
    }

    ClusteringResult result;
    result.communities.resize(n);
    result.node_to_original.resize(n);
    for (int i = 0; i < n; ++i) {
        result.communities[i]      = task_community[i];
        result.node_to_original[i] = {i};
    }
    for (const auto& [cid, members] : comm_to_tasks) {
        HardwareResources res;
        double min_s = std::numeric_limits<double>::max();
        double max_e = std::numeric_limits<double>::lowest();
        for (int t : members) {
            res.clb  += orig_tasks[t].clb;
            res.dsp  += orig_tasks[t].dsp;
            res.bram += orig_tasks[t].bram;
            min_s = std::min(min_s, orig_tasks[t].start_earliest);
            max_e = std::max(max_e, orig_tasks[t].start_latest + orig_tasks[t].exec_time);
        }
        result.community_resources[cid] = res;
        result.community_windows[cid]   = TimeWindow(min_s, max_e);
    }
    return result;
}

std::vector<ClusteringResult> TaskClustering::performMultiLevelClustering(
    double modularity_threshold,
    const CommunityLoadParams& load_params,
    unsigned seed,
    bool verbose) const {
    std::vector<ClusteringResult> results;
    std::vector<ClusterTask> current_tasks = tasks;
    std::vector<std::vector<double>> current_weights = weights;
    const std::vector<ClusterTask> original_tasks = tasks;

    std::vector<std::vector<int>> current_node_mapping(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i)
        current_node_mapping[i].push_back(static_cast<int>(i));

    double prev_modularity = -std::numeric_limits<double>::infinity();
    bool continue_clustering = true;

    while (continue_clustering) {
        TaskClustering current_clustering(current_tasks, current_weights);
        ClusteringResult result = current_clustering.performSingleLevelClustering(
            seed + static_cast<unsigned>(results.size()), verbose);
        results.push_back(result);

        double total_weight = 0.0;
        for (size_t i = 0; i < current_tasks.size(); ++i) {
            for (size_t j = i; j < current_tasks.size(); ++j) {
                if (i == j) total_weight += current_weights[i][j] / 2.0;
                else        total_weight += current_weights[i][j];
            }
        }
        const double current_modularity =
            calculateModularity(result.communities, current_weights, total_weight);

        std::vector<ClusterTask> next_tasks;
        std::vector<std::vector<double>> next_weights;
        std::vector<std::vector<int>> next_node_mapping;
        current_clustering.buildNextLevelGraph(
            result.communities, current_weights, next_tasks, next_weights,
            current_node_mapping, next_node_mapping);

        results.back().node_to_original = current_node_mapping;

        const std::set<int> unique_communities(result.communities.begin(),
                                               result.communities.end());
        const std::vector<int> community_list(unique_communities.begin(),
                                              unique_communities.end());
        results.back().community_windows.clear();
        for (size_t i = 0; i < community_list.size(); ++i) {
            const int comm_id = community_list[i];
            const ClusterTask& super_node = next_tasks[i];
            results.back().community_windows[comm_id] = TimeWindow(
                super_node.start_earliest,
                super_node.start_latest + super_node.exec_time);
        }

        if (current_modularity - prev_modularity < modularity_threshold) {
            continue_clustering = false;
        } else {
            prev_modularity      = current_modularity;
            current_tasks        = std::move(next_tasks);
            current_weights      = std::move(next_weights);
            current_node_mapping = std::move(next_node_mapping);
        }
    }

    if (!results.empty()) {
        results.back() = postSplitOverloadedCommunities(results.back(), tasks, load_params, verbose);
        results.back().community_resources = computeCommunityResources(
            results.back().communities, results.back().node_to_original, original_tasks);
    }
    return results;
}

bool readTaskGraph(const std::string& filename,
                   std::vector<ClusterTask>& tasks,
                   std::vector<std::vector<double>>& comm_matrix,
                   std::vector<std::vector<int>>& dep_matrix,
                   std::vector<std::vector<double>>& urgency_matrix,
                   bool dump_debug,
                   const std::string& debug_dir) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    json j = json::parse(file);
    const size_t num_nodes = j.at("nodes").size();

    tasks.assign(num_nodes, ClusterTask{});
    comm_matrix.assign(num_nodes, std::vector<double>(num_nodes, 0.0));
    dep_matrix.assign(num_nodes, std::vector<int>(num_nodes, 0));
    urgency_matrix.assign(num_nodes, std::vector<double>(num_nodes, 0.0));

    for (const auto& node : j.at("nodes")) {
        const int id = node.at("id").get<int>();
        if (id < 1 || static_cast<size_t>(id) > num_nodes)
            throw std::invalid_argument("Task node id must be in 1..N");
        const double est       = node.at("EST").get<double>();
        const double lft       = node.at("LFT").get<double>();
        const double exec_time = node.at("execution_time").get<double>();
        const int clb          = node.at("CLB").get<int>();
        const int dsp          = node.at("DSP").get<int>();
        const int bram         = node.at("BRAM").get<int>();
        tasks[id - 1] = ClusterTask(est, lft - exec_time, exec_time, clb, dsp, bram);
    }

    for (const auto& edge : j.at("edges")) {
        const int source    = edge.at("source").get<int>();
        const int target    = edge.at("target").get<int>();
        const double comm   = edge.at("communication").get<double>();
        if (source < 1 || static_cast<size_t>(source) > num_nodes ||
            target < 1 || static_cast<size_t>(target) > num_nodes)
            throw std::invalid_argument("Edge endpoint id out of range 1..N");

        comm_matrix[source - 1][target - 1] = comm;
        comm_matrix[target - 1][source - 1] = comm;
        dep_matrix[source - 1][target - 1]  = 1;

        const double slack = tasks[target - 1].start_latest
                           - tasks[source - 1].start_earliest
                           - tasks[source - 1].exec_time;
        const double urgency = slack > 0.0 ? 1.0 / (1.0 + slack) : 1.0;
        urgency_matrix[source - 1][target - 1] = urgency;
        urgency_matrix[target - 1][source - 1] = urgency;
    }

    normalizeMatrix(comm_matrix);
    if (dump_debug) {
        const std::filesystem::path base = debug_dir.empty()
            ? std::filesystem::current_path() : std::filesystem::path(debug_dir);
        dumpSquareMatrixTxt((base / "communication_matrix.txt").string(),
                            "Normalized communication matrix", comm_matrix);
    }
    return true;
}

std::vector<std::vector<double>> computeIntegratedWeights(
    const std::vector<ClusterTask>& tasks,
    const std::vector<std::vector<double>>& comm_matrix,
    const std::vector<std::vector<int>>& dep_matrix,
    const std::vector<std::vector<double>>& urgency_matrix,
    double lambda1,
    double lambda2,
    double lambda3,
    bool dump_debug,
    const std::string& debug_dir) {
    const size_t n = tasks.size();
    std::vector<std::vector<double>> weights(n, std::vector<double>(n, 0.0));
    if (n == 0) return weights;

    const double kappa = 0.35;
    const double theta = 0.5;

    double G_start = tasks[0].start_earliest;
    double G_end   = tasks[0].start_latest + tasks[0].exec_time;
    for (size_t i = 1; i < n; ++i) {
        G_start = std::min(G_start, tasks[i].start_earliest);
        G_end   = std::max(G_end, tasks[i].start_latest + tasks[i].exec_time);
    }

    std::vector<TimeWindow> windows;
    windows.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const double ES    = tasks[i].start_earliest;
        const double LS    = tasks[i].start_latest;
        const double d     = tasks[i].exec_time;
        const double slack = LS - ES;

        const double center    = (ES + LS + d) / 2.0;
        const double ell_eff   = d + kappa * slack;
        const double eff_start = center - ell_eff / 2.0;
        const double eff_end   = center + ell_eff / 2.0;

        const double w_start = std::max(eff_start, G_start);
        const double w_end   = std::min(eff_end, G_end);
        if (w_end <= w_start) windows.emplace_back(0.0, 0.0);
        else                  windows.emplace_back(w_start, w_end);
    }

    std::vector<std::vector<double>> overlap_matrix(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            const double W_comm = comm_matrix[i][j];
            const int edge_exists = (dep_matrix[i][j] != 0 || dep_matrix[j][i] != 0) ? 1 : 0;
            const double W_time = static_cast<double>(edge_exists) * urgency_matrix[i][j];

            double overlap = 0.0;
            if (i == j) {
                overlap = 1.0;
            } else {
                const double len_i = std::max(0.0, windows[i].end - windows[i].start);
                const double len_j = std::max(0.0, windows[j].end - windows[j].start);
                const double inter = std::max(
                    0.0,
                    std::min(windows[i].end, windows[j].end) -
                    std::max(windows[i].start, windows[j].start));
                const double uni   = len_i + len_j - inter;
                const double iou   = (uni > 0.0) ? (inter / uni) : 0.0;
                const double cover = (std::min(len_i, len_j) > 0.0)
                                   ? (inter / std::min(len_i, len_j)) : 0.0;
                overlap = theta * iou + (1.0 - theta) * cover;
            }
            overlap_matrix[i][j] = overlap;
            if (i != j) overlap_matrix[j][i] = overlap;

            const double w = lambda1 * W_comm + lambda2 * W_time + lambda3 * overlap;
            weights[i][j] = w;
            if (i != j) weights[j][i] = w;
        }
    }

    if (dump_debug) {
        const std::filesystem::path base = debug_dir.empty()
            ? std::filesystem::current_path() : std::filesystem::path(debug_dir);
        dumpSquareMatrixTxt((base / "overlap_matrix.txt").string(),
                            "Static temporal window overlap matrix", overlap_matrix);
        dumpSquareMatrixTxt((base / "weights_matrix_1.txt").string(),
                            "Integrated edge weight matrix", weights);
    }
    return weights;
}

QuotientCycleDiagnostics analyzeQuotientCycles(
    const ClusteringResult& final_result,
    const std::vector<ClusterTask>& original_tasks,
    const std::vector<std::vector<int>>& dep_matrix) {
    QuotientCycleDiagnostics diag;
    const size_t n_tasks = original_tasks.size();
    if (n_tasks == 0 || final_result.communities.empty()) return diag;

    std::vector<int> task_to_comm(n_tasks, -1);
    std::map<int, HardwareResources> comm_res;
    for (size_t i = 0; i < final_result.communities.size(); ++i) {
        const int cid = final_result.communities[i];
        for (int orig : final_result.node_to_original[i]) {
            task_to_comm[orig] = cid;
            comm_res[cid].clb  += original_tasks[orig].clb;
            comm_res[cid].dsp  += original_tasks[orig].dsp;
            comm_res[cid].bram += original_tasks[orig].bram;
        }
    }

    std::vector<int> comm_ids;
    std::map<int, int> comm_to_idx;
    for (const auto& [cid, res] : comm_res) {
        comm_to_idx[cid] = static_cast<int>(comm_ids.size());
        comm_ids.push_back(cid);
    }

    const int K = static_cast<int>(comm_ids.size());
    std::vector<std::set<int>> adj(K);
    if (dep_matrix.size() == n_tasks) {
        for (size_t u = 0; u < n_tasks; ++u) {
            for (size_t v = 0; v < n_tasks; ++v) {
                if (!dep_matrix[u][v]) continue;
                const int cu = task_to_comm[u];
                const int cv = task_to_comm[v];
                if (cu >= 0 && cv >= 0 && cu != cv) {
                    adj[comm_to_idx[cu]].insert(comm_to_idx[cv]);
                }
            }
        }
    }

    // Tarjan's SCC on the community quotient graph.
    int timer = 0;
    std::vector<int> disc(K, -1), low(K, -1), st;
    std::vector<bool> on_stack(K, false);
    std::vector<std::vector<int>> sccs;

    std::function<void(int)> dfs = [&](int u) {
        disc[u] = low[u] = timer++;
        st.push_back(u);
        on_stack[u] = true;
        for (int v : adj[u]) {
            if (disc[v] == -1) {
                dfs(v);
                low[u] = std::min(low[u], low[v]);
            } else if (on_stack[v]) {
                low[u] = std::min(low[u], disc[v]);
            }
        }
        if (low[u] == disc[u]) {
            std::vector<int> comp;
            while (true) {
                const int w = st.back();
                st.pop_back();
                on_stack[w] = false;
                comp.push_back(w);
                if (w == u) break;
            }
            if (comp.size() >= 2) sccs.push_back(std::move(comp));
        }
    };

    for (int i = 0; i < K; ++i)
        if (disc[i] == -1) dfs(i);

    for (const auto& comp : sccs) {
        QuotientCycleInfo info;
        for (int idx : comp) {
            const int cid = comm_ids[idx];
            info.community_ids.push_back(cid + 1); // 1-based JSON ID
            info.combined_resources += comm_res[cid];
        }
        std::sort(info.community_ids.begin(), info.community_ids.end());
        info.exceeds_fpga_capacity = !info.combined_resources.withinFPGALimits();
        if (info.exceeds_fpga_capacity) diag.any_exceeds_fpga_capacity = true;
        diag.cycles.push_back(std::move(info));
    }
    std::sort(diag.cycles.begin(), diag.cycles.end(),
              [](const QuotientCycleInfo& a, const QuotientCycleInfo& b) {
                  return a.community_ids < b.community_ids;
              });
    diag.num_nontrivial_sccs = static_cast<int>(diag.cycles.size());
    return diag;
}

nlohmann::json buildClusteringJson(
    const std::vector<ClusteringResult>& results,
    const std::vector<ClusterTask>& original_tasks,
    const std::vector<std::vector<int>>& dep_matrix,
    double lambda1, double lambda2, double lambda3,
    unsigned cluster_seed,
    double clustering_elapsed_seconds,
    QuotientCycleDiagnostics* diag_out) {
    if (results.empty()) return json::object();
    const ClusteringResult& final_result = results.back();
    const std::set<int> unique_communities(final_result.communities.begin(),
                                           final_result.communities.end());

    struct NodeInfo {
        int id;
        double est;
        double lst;
        double exec_time;
        double lft;
        double slack;
    };
    struct CommunityInfo {
        int comm_id;
        std::vector<NodeInfo> nodes;
        HardwareResources resources;
        TimeWindow window;
    };

    std::vector<CommunityInfo> comm_infos;
    for (int comm_id : unique_communities) {
        CommunityInfo ci;
        ci.comm_id = comm_id;
        HardwareResources res;
        for (size_t i = 0; i < final_result.communities.size(); ++i) {
            if (final_result.communities[i] != comm_id) continue;
            for (int orig : final_result.node_to_original[i]) {
                const ClusterTask& t = original_tasks[orig];
                res.clb  += t.clb;
                res.dsp  += t.dsp;
                res.bram += t.bram;
                ci.nodes.push_back({
                    orig + 1,
                    t.start_earliest,
                    t.start_latest,
                    t.exec_time,
                    t.start_latest + t.exec_time,
                    t.start_latest - t.start_earliest
                });
            }
        }
        ci.resources = res;
        auto it = final_result.community_windows.find(comm_id);
        if (it != final_result.community_windows.end()) ci.window = it->second;
        std::sort(ci.nodes.begin(), ci.nodes.end(),
                  [](const NodeInfo& a, const NodeInfo& b) { return a.id < b.id; });
        comm_infos.push_back(std::move(ci));
    }
    std::sort(comm_infos.begin(), comm_infos.end(),
              [](const CommunityInfo& a, const CommunityInfo& b) { return a.comm_id < b.comm_id; });

    const QuotientCycleDiagnostics diag =
        analyzeQuotientCycles(final_result, original_tasks, dep_matrix);
    if (diag_out) *diag_out = diag;

    json j_cycles = json::array();
    for (const auto& cyc : diag.cycles) {
        j_cycles.push_back({
            {"community_ids", cyc.community_ids},
            {"combined_resources", {
                {"CLB", cyc.combined_resources.clb},
                {"DSP", cyc.combined_resources.dsp},
                {"BRAM", cyc.combined_resources.bram}
            }},
            {"exceeds_fpga_capacity", cyc.exceeds_fpga_capacity}
        });
    }

    json j_root;
    j_root["config"] = {
        {"lambda1", lambda1},
        {"lambda2", lambda2},
        {"lambda3", lambda3},
        {"cluster_seed", cluster_seed}
    };
    j_root["total_original_nodes"]       = static_cast<int>(original_tasks.size());
    j_root["total_communities"]          = static_cast<int>(comm_infos.size());
    j_root["clustering_elapsed_seconds"] = clustering_elapsed_seconds;
    j_root["quotient_cycle_diagnostics"] = {
        {"num_nontrivial_sccs", diag.num_nontrivial_sccs},
        {"any_exceeds_fpga_capacity", diag.any_exceeds_fpga_capacity},
        {"cycles", j_cycles}
    };

    json j_communities = json::array();
    for (const auto& ci : comm_infos) {
        json j_nodes = json::array();
        for (const auto& ni : ci.nodes) {
            j_nodes.push_back({
                {"id",        ni.id},
                {"EST",       ni.est},
                {"LST",       ni.lst},
                {"exec_time", ni.exec_time},
                {"LFT",       ni.lft},
                {"slack",     ni.slack}
            });
        }
        j_communities.push_back({
            {"id", ci.comm_id + 1},
            {"original_node_count", static_cast<int>(ci.nodes.size())},
            {"total_resources", {
                {"CLB",  ci.resources.clb},
                {"DSP",  ci.resources.dsp},
                {"BRAM", ci.resources.bram}
            }},
            {"time_window", {
                {"start", ci.window.start},
                {"end",   ci.window.end}
            }},
            {"nodes", std::move(j_nodes)}
        });
    }
    j_root["communities"] = std::move(j_communities);
    return j_root;
}

nlohmann::json runClusteringPipeline(
    const std::string& dag_file,
    unsigned cluster_seed,
    const std::string& json_output_path,
    bool dump_debug,
    const std::string& debug_dir,
    bool verbose) {
    std::vector<ClusterTask> tasks;
    std::vector<std::vector<double>> comm_matrix;
    std::vector<std::vector<int>> dep_matrix;
    std::vector<std::vector<double>> urgency_matrix;

    if (!readTaskGraph(dag_file, tasks, comm_matrix, dep_matrix, urgency_matrix,
                       dump_debug, debug_dir)) {
        throw std::runtime_error("Failed to open or parse DAG file: " + dag_file);
    }

    const double lambda1 = 0.4;
    const double lambda2 = 0.3;
    const double lambda3 = 0.3;

    const auto start_t = std::chrono::high_resolution_clock::now();
    const auto integrated_weights = computeIntegratedWeights(
        tasks, comm_matrix, dep_matrix, urgency_matrix,
        lambda1, lambda2, lambda3, dump_debug, debug_dir);

    TaskClustering clustering(tasks, integrated_weights);
    clustering.setDepMatrix(dep_matrix);

    CommunityLoadParams load_params;
    load_params.quantile_level   = 0.9;
    load_params.rho1             = 1.0 / 3.0;
    load_params.rho2             = 1.0 / 3.0;
    load_params.rho3             = 1.0 / 3.0;
    load_params.wc               = 1.0 / 3.0;
    load_params.wb               = 1.0 / 3.0;
    load_params.wd               = 1.0 / 3.0;
    load_params.max_split_rounds = 3;

    const auto results = clustering.performMultiLevelClustering(
        0.001, load_params, cluster_seed, verbose);
    const auto end_t = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(end_t - start_t).count();

    QuotientCycleDiagnostics diag;
    json j_out = buildClusteringJson(
        results, tasks, dep_matrix, lambda1, lambda2, lambda3,
        cluster_seed, elapsed, &diag);

    if (diag.any_exceeds_fpga_capacity) {
        std::cerr << "[CPS WARNING] Community quotient graph contains "
                  << diag.num_nontrivial_sccs
                  << " non-convex dependency cycle(s) whose combined CLB/DSP/BRAM demand "
                     "exceeds FPGA capacity. Under strict single-residency execution "
                     "(|L_k| = 1, no idle suspension), co-residency may be infeasible.\n";
    } else if (verbose && diag.num_nontrivial_sccs > 0) {
        std::cout << "  [Quotient Check] " << diag.num_nontrivial_sccs
                  << " non-trivial inter-community SCC(s) detected (all within FPGA capacity).\n";
    }

    if (!json_output_path.empty()) {
        std::filesystem::create_directories(
            std::filesystem::path(json_output_path).parent_path());
        std::ofstream ofs(json_output_path);
        if (!ofs.is_open())
            throw std::runtime_error("Cannot write clustering JSON: " + json_output_path);
        ofs << j_out.dump(2) << "\n";
    }
    return j_out;
}

} // namespace cps::clustering
