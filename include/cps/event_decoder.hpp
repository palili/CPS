#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cps/chromosome_io.hpp"
#include "cps/clustering.hpp"
#include "cps/event_time.hpp"
#include "cps/fpga_grid.hpp"
#include "cps/nested_dfx_allocator.hpp"
#include "cps/preset_config.hpp"
#include "nlohmann/json.hpp"

namespace cps {

enum class TaskState { IDLE, WAITING, READY, RUNNING, DONE };
enum class CommunityState { LATENT, ACTIVE, DONE };

struct Task {
    int id = -1;
    int community_id = -1;
    int CLB = 0, BRAM = 0, DSP = 0;
    double exec_time = 0.0;
    double EST = 0.0, LFT = 0.0, LST = 0.0, slack = 0.0;
    std::vector<int> predecessors;
    std::vector<int> successors;
    TaskState state = TaskState::IDLE;
    double start_time  = -1.0;
    double finish_time = -1.0;
};

struct Community {
    int id = -1;
    std::vector<int> task_ids;
    int CLB = 0, BRAM = 0, DSP = 0;
    CommunityState state = CommunityState::LATENT;
    double activation_time            = -1.0;
    double completion_time            = -1.0;
    double accumulated_residence_time = 0.0;
    int activation_count              = 0;

    // Normalized static community features (s1..s7)
    double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    double s5 = 0.0, s6 = 0.0, s7 = 0.0;
    double pri = 0.0;
};

struct Edge {
    int source = -1;
    int target = -1;
    int communication = 0;
};

struct FPGA {
    int total_CLB  = clustering::FPGA_MAX_CLB;
    int total_BRAM = clustering::FPGA_MAX_BRAM;
    int total_DSP  = clustering::FPGA_MAX_DSP;
    int remain_CLB  = clustering::FPGA_MAX_CLB;
    int remain_BRAM = clustering::FPGA_MAX_BRAM;
    int remain_DSP  = clustering::FPGA_MAX_DSP;
};

// Working state and immutable templates for fast per-chromosome decoder replay
inline std::map<int, Task>      g_tasks;
inline std::map<int, Community> g_communities;
inline std::vector<Edge>        g_edges;
inline FPGA                     g_fpga;

inline std::map<int, Task>      g_tasks_template;
inline std::map<int, Community> g_communities_template;
inline std::vector<Edge>        g_edges_template;
inline FPGA                     g_fpga_template;

inline layout::FPGAModel                 g_layout_fpga;
inline layout::SearchParams              g_layout_params;
inline nested_dfx::NestedDFXAllocator    g_nested_alloc;
inline nested_dfx::PlaceDecision         g_last_decision;
inline double                            g_total_comm_cost = 0.0;
inline double                            g_total_fail_cost = 0.0;

inline std::map<std::pair<int, int>, double> g_comm_weight;
inline std::set<int>                         g_global_cp_tasks;

inline void resetAllDecoderTemplates() {
    g_tasks_template.clear();
    g_communities_template.clear();
    g_edges_template.clear();
    g_fpga_template = FPGA{};
    g_tasks.clear();
    g_communities.clear();
    g_edges.clear();
    g_fpga = FPGA{};
    g_comm_weight.clear();
    g_global_cp_tasks.clear();
}

inline void loadDAGFromJson(const nlohmann::json& data) {
    g_tasks_template.clear();
    g_edges_template.clear();
    if (!data.contains("nodes") || !data.contains("edges"))
        throw std::runtime_error("Invalid DAG JSON: missing 'nodes' or 'edges'");

    for (const auto& node : data.at("nodes")) {
        Task t;
        t.id        = node.at("id").get<int>();
        t.exec_time = node.at("execution_time").get<double>();
        t.CLB       = node.at("CLB").get<int>();
        t.BRAM      = node.at("BRAM").get<int>();
        t.DSP       = node.at("DSP").get<int>();
        t.EST       = node.at("EST").get<double>();
        t.LFT       = node.at("LFT").get<double>();
        t.slack     = node.at("Slack").get<double>();
        t.LST       = t.LFT - t.exec_time;
        g_tasks_template[t.id] = t;
    }

    for (const auto& edge : data.at("edges")) {
        Edge e;
        e.source        = edge.at("source").get<int>();
        e.target        = edge.at("target").get<int>();
        e.communication = edge.at("communication").get<int>();
        g_edges_template.push_back(e);
        g_tasks_template.at(e.source).successors.push_back(e.target);
        g_tasks_template.at(e.target).predecessors.push_back(e.source);
    }
}

inline void loadDAG(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot open DAG file: " + filename);
    loadDAGFromJson(nlohmann::json::parse(f));
}

inline void loadCommunitiesFromJson(const nlohmann::json& data) {
    g_communities_template.clear();
    if (!data.contains("communities"))
        throw std::runtime_error("Invalid clustering JSON: missing 'communities'");

    for (const auto& comm : data.at("communities")) {
        Community c;
        c.id   = comm.at("id").get<int>();
        c.CLB  = comm.at("total_resources").at("CLB").get<int>();
        c.BRAM = comm.at("total_resources").at("BRAM").get<int>();
        c.DSP  = comm.at("total_resources").at("DSP").get<int>();
        for (const auto& node : comm.at("nodes")) {
            const int tid = node.at("id").get<int>();
            c.task_ids.push_back(tid);
            g_tasks_template.at(tid).community_id = c.id;
        }
        g_communities_template[c.id] = c;
    }
}

inline void loadCommunities(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot open communities file: " + filename);
    loadCommunitiesFromJson(nlohmann::json::parse(f));
}

inline void resetWorkingData() {
    g_tasks       = g_tasks_template;
    g_communities = g_communities_template;
    g_edges       = g_edges_template;
    g_fpga        = g_fpga_template;

    g_nested_alloc.reset();
    std::vector<nested_dfx::TaskCommunicationEdge> task_edges;
    task_edges.reserve(g_edges.size());
    for (size_t i = 0; i < g_edges.size(); ++i) {
        task_edges.push_back({
            static_cast<int>(i),
            g_edges[i].source,
            g_edges[i].target,
            static_cast<double>(g_edges[i].communication)
        });
    }
    g_nested_alloc.setTaskCommunicationEdges(std::move(task_edges));
    g_last_decision   = nested_dfx::PlaceDecision();
    g_total_comm_cost = 0.0;
    g_total_fail_cost = 0.0;
}

inline double closeActiveEpisode(Community& comm, double t) {
    if (comm.activation_time < 0.0 || t < comm.activation_time - 1e-9)
        return 0.0;
    const double residence = std::max(0.0, t - comm.activation_time);
    comm.accumulated_residence_time += residence;
    return residence;
}

inline double cumulativeResidenceTime(const Community& comm) {
    if (comm.activation_count <= 0) return -1.0;
    return comm.accumulated_residence_time;
}

inline void buildCommunityCommWeight() {
    g_comm_weight.clear();
    std::map<int, int> t2c;
    for (const auto& [cid, comm] : g_communities_template)
        for (int tid : comm.task_ids) t2c[tid] = cid;

    for (const auto& e : g_edges_template) {
        auto its = t2c.find(e.source);
        auto itt = t2c.find(e.target);
        if (its == t2c.end() || itt == t2c.end()) continue;
        const int cs = its->second, ct = itt->second;
        if (cs == ct) continue;
        const int a = std::min(cs, ct), b = std::max(cs, ct);
        g_comm_weight[{a, b}] += static_cast<double>(e.communication);
    }
}

inline double commWeight(int a, int b) {
    if (a == b) return 0.0;
    const int lo = std::min(a, b), hi = std::max(a, b);
    auto it = g_comm_weight.find({lo, hi});
    return it == g_comm_weight.end() ? 0.0 : it->second;
}

// ============================================================================
// Static Community Features (s1..s7)
// ============================================================================

inline double computeS1(int ck_id) {
    const auto& tids = g_communities.at(ck_id).task_ids;
    std::set<int> members(tids.begin(), tids.end());
    double total = 0.0;
    for (const auto& e : g_edges) {
        const bool src_in = members.count(e.source) > 0;
        const bool tgt_in = members.count(e.target) > 0;
        if (src_in != tgt_in) total += e.communication;
    }
    return total;
}

inline double computeS2(int ck_id) {
    const auto& tids = g_communities.at(ck_id).task_ids;
    std::set<int> members(tids.begin(), tids.end());
    int count = 0;
    for (const auto& e : g_edges) {
        const bool src_in = members.count(e.source) > 0;
        const bool tgt_in = members.count(e.target) > 0;
        if (src_in != tgt_in) ++count;
    }
    return static_cast<double>(count);
}

inline double computeS3(int ck_id) {
    const auto& tids = g_communities.at(ck_id).task_ids;
    const int n = static_cast<int>(tids.size());
    if (n <= 0) return 0.0;
    double sum = 0.0;
    for (int tid : tids) {
        const double s = g_tasks.at(tid).slack;
        sum += std::min(1000.0, 1.0 / std::max(s, 1e-6));
    }
    return sum / n;
}

inline double computeS4(int ck_id, bool remaining_only = false) {
    const auto& tids = g_communities.at(ck_id).task_ids;
    std::set<int> members(tids.begin(), tids.end());
    std::map<int, std::vector<int>> adj;
    std::map<int, int> in_deg;
    for (int tid : tids) {
        adj[tid] = {};
        in_deg[tid] = 0;
    }
    for (const auto& e : g_edges) {
        if (members.count(e.source) && members.count(e.target)) {
            adj[e.source].push_back(e.target);
            in_deg[e.target]++;
        }
    }
    std::map<int, double> dist;
    std::queue<int> q;
    auto duration = [&](int tid) {
        return (remaining_only && g_tasks.at(tid).state == TaskState::DONE)
                   ? 0.0
                   : g_tasks.at(tid).exec_time;
    };
    for (int tid : tids) {
        dist[tid] = duration(tid);
        if (in_deg[tid] == 0) q.push(tid);
    }
    double critical_path = 0.0;
    while (!q.empty()) {
        const int u = q.front();
        q.pop();
        critical_path = std::max(critical_path, dist[u]);
        for (int v : adj[u]) {
            dist[v] = std::max(dist[v], dist[u] + duration(v));
            if (--in_deg[v] == 0) q.push(v);
        }
    }
    return critical_path;
}

inline std::set<int> computeGlobalCPTaskSet() {
    std::map<int, std::vector<int>> adj_succ, adj_pred;
    std::map<int, int> in_deg;
    for (const auto& [tid, t] : g_tasks_template) {
        (void)t;
        adj_succ[tid] = {};
        adj_pred[tid] = {};
        in_deg[tid]   = 0;
    }
    for (const auto& e : g_edges_template) {
        adj_succ[e.source].push_back(e.target);
        adj_pred[e.target].push_back(e.source);
        in_deg[e.target]++;
    }
    std::vector<int> topo;
    std::queue<int> q;
    for (const auto& [tid, deg] : in_deg)
        if (deg == 0) q.push(tid);
    while (!q.empty()) {
        const int u = q.front();
        q.pop();
        topo.push_back(u);
        for (int v : adj_succ[u])
            if (--in_deg[v] == 0) q.push(v);
    }
    std::map<int, double> est_long;
    for (int u : topo) {
        const double base = g_tasks_template.at(u).exec_time;
        double prev_max = 0.0;
        for (int p : adj_pred[u]) prev_max = std::max(prev_max, est_long[p]);
        est_long[u] = prev_max + base;
    }
    std::map<int, double> lst_long;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const int u = *it;
        const double base = g_tasks_template.at(u).exec_time;
        double succ_max = 0.0;
        for (int s : adj_succ[u]) succ_max = std::max(succ_max, lst_long[s]);
        lst_long[u] = succ_max + base;
    }
    double cp_len = 0.0;
    for (const auto& [tid, v] : est_long) {
        (void)tid;
        cp_len = std::max(cp_len, v);
    }
    std::set<int> cp_tasks;
    for (const auto& [tid, t] : g_tasks_template) {
        const double sum = est_long[tid] + lst_long[tid] - t.exec_time;
        if (std::abs(sum - cp_len) < 1e-6) cp_tasks.insert(tid);
    }
    return cp_tasks;
}

inline double computeS5(int ck_id, const std::set<int>& cp_tasks) {
    const auto& comm = g_communities.at(ck_id);
    const int n = static_cast<int>(comm.task_ids.size());
    if (n == 0) return 0.0;
    int hit = 0;
    for (int tid : comm.task_ids)
        if (cp_tasks.count(tid)) ++hit;
    return static_cast<double>(hit) / n;
}

inline double computeS6(int ck_id, const std::map<int, int>& depth_map) {
    const auto& comm = g_communities.at(ck_id);
    if (comm.task_ids.empty()) return 0.0;
    int min_d = std::numeric_limits<int>::max();
    for (int tid : comm.task_ids) {
        auto it = depth_map.find(tid);
        if (it != depth_map.end()) min_d = std::min(min_d, it->second);
    }
    if (min_d == std::numeric_limits<int>::max()) return 0.0;
    return 1.0 / (1.0 + min_d);
}

inline double computeS7(int ck_id, const std::map<int, std::set<int>>& comm_succ_closure) {
    auto it = comm_succ_closure.find(ck_id);
    return it == comm_succ_closure.end() ? 0.0 : static_cast<double>(it->second.size());
}

inline void computeAllStaticFeatures() {
    const std::set<int> cp_tasks = computeGlobalCPTaskSet();
    g_global_cp_tasks = cp_tasks;

    std::map<int, std::vector<int>> adj_succ;
    std::map<int, int> in_deg;
    for (const auto& [tid, t] : g_tasks_template) {
        (void)t;
        adj_succ[tid] = {};
        in_deg[tid]   = 0;
    }
    for (const auto& e : g_edges_template) {
        adj_succ[e.source].push_back(e.target);
        in_deg[e.target]++;
    }
    std::map<int, int> depth_map;
    std::queue<int> q;
    for (const auto& [tid, d] : in_deg) {
        if (d == 0) {
            depth_map[tid] = 0;
            q.push(tid);
        }
    }
    while (!q.empty()) {
        const int u = q.front();
        q.pop();
        for (int v : adj_succ[u]) {
            const int nd = depth_map[u] + 1;
            auto it = depth_map.find(v);
            if (it == depth_map.end() || nd > it->second) depth_map[v] = nd;
            if (--in_deg[v] == 0) q.push(v);
        }
    }

    std::map<int, int> task2comm;
    for (const auto& [cid, c] : g_communities)
        for (int tid : c.task_ids) task2comm[tid] = cid;

    std::map<int, std::set<int>> comm_adj;
    for (const auto& [cid, c] : g_communities) {
        (void)c;
        comm_adj[cid] = {};
    }
    for (const auto& e : g_edges_template) {
        auto a = task2comm.find(e.source);
        auto b = task2comm.find(e.target);
        if (a == task2comm.end() || b == task2comm.end()) continue;
        if (a->second != b->second) comm_adj[a->second].insert(b->second);
    }

    std::map<int, std::set<int>> comm_succ_closure;
    for (const auto& [cid, _adj] : comm_adj) {
        (void)_adj;
        std::set<int> reach;
        std::queue<int> bq;
        bq.push(cid);
        while (!bq.empty()) {
            const int u = bq.front();
            bq.pop();
            for (int v : comm_adj[u])
                if (reach.insert(v).second) bq.push(v);
        }
        comm_succ_closure[cid] = reach;
    }

    std::vector<std::pair<int, double>> s1v, s2v, s3v, s4v, s5v, s6v, s7v;
    for (const auto& [cid, comm] : g_communities) {
        (void)comm;
        s1v.push_back({cid, computeS1(cid)});
        s2v.push_back({cid, computeS2(cid)});
        s3v.push_back({cid, computeS3(cid)});
        s4v.push_back({cid, computeS4(cid)});
        s5v.push_back({cid, computeS5(cid, cp_tasks)});
        s6v.push_back({cid, computeS6(cid, depth_map)});
        s7v.push_back({cid, computeS7(cid, comm_succ_closure)});
    }

    auto normalize = [](std::vector<std::pair<int, double>>& vals) {
        double mn = 1e18, mx = -1e18;
        for (const auto& [id, v] : vals) {
            (void)id;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        const double range = (mx - mn < 1e-9) ? 1.0 : (mx - mn);
        for (auto& [id, v] : vals) {
            (void)id;
            v = (v - mn) / range;
        }
    };
    normalize(s1v); normalize(s2v); normalize(s3v); normalize(s4v);
    normalize(s5v); normalize(s6v); normalize(s7v);

    for (const auto& [id, v] : s1v) g_communities[id].s1 = v;
    for (const auto& [id, v] : s2v) g_communities[id].s2 = v;
    for (const auto& [id, v] : s3v) g_communities[id].s3 = v;
    for (const auto& [id, v] : s4v) g_communities[id].s4 = v;
    for (const auto& [id, v] : s5v) g_communities[id].s5 = v;
    for (const auto& [id, v] : s6v) g_communities[id].s6 = v;
    for (const auto& [id, v] : s7v) g_communities[id].s7 = v;
}

inline void computeStaticPriority(const Chromosome& chr) {
    for (auto& [cid, comm] : g_communities) {
        (void)cid;
        comm.pri = chr.alpha[0] * comm.s1 + chr.alpha[1] * comm.s2
                 + chr.alpha[2] * comm.s3 + chr.alpha[3] * comm.s4
                 + chr.alpha[4] * comm.s5 + chr.alpha[5] * comm.s6
                 + chr.alpha[6] * comm.s7;
    }
}

// ============================================================================
// Dynamic Runtime & Pool-Wide Reservation Features (d1..d11)
// ============================================================================

inline bool isHypotheticallyReady(int tid, double t) {
    const auto& task = g_tasks.at(tid);
    for (int pid : task.predecessors)
        if (g_tasks.at(pid).state != TaskState::DONE) return false;
    return t >= task.EST - 1e-9;
}

inline double computeD1(int ck_id, double t) {
    int ready_cnt = 0, total = 0;
    const auto& comm = g_communities.at(ck_id);
    for (int tid : comm.task_ids) {
        if (g_tasks.at(tid).state == TaskState::DONE) continue;
        ++total;
        if (comm.state == CommunityState::ACTIVE) {
            if (g_tasks.at(tid).state == TaskState::READY) ++ready_cnt;
        } else {
            if (isHypotheticallyReady(tid, t)) ++ready_cnt;
        }
    }
    return total > 0 ? static_cast<double>(ready_cnt) / total : 0.0;
}

inline double computeD2(int ck_id, double t) {
    double max_urgency = 0.0;
    for (int tid : g_communities.at(ck_id).task_ids) {
        if (g_tasks.at(tid).state == TaskState::DONE) continue;
        const double diff = g_tasks.at(tid).LST - t;
        const double urgency = 1.0 / (1.0 + std::max(diff, 0.0));
        max_urgency = std::max(max_urgency, urgency);
    }
    return max_urgency;
}

inline double computeD3(int ck_id) {
    std::set<int> succ_comms;
    for (int tid : g_communities.at(ck_id).task_ids) {
        for (int sid : g_tasks.at(tid).successors) {
            const int sc = g_tasks.at(sid).community_id;
            if (sc != ck_id && g_communities.at(sc).state == CommunityState::LATENT)
                succ_comms.insert(sc);
        }
    }
    int total_latent = 0;
    for (const auto& [cid, comm] : g_communities)
        if (comm.state == CommunityState::LATENT && cid != ck_id) ++total_latent;
    return total_latent > 0 ? static_cast<double>(succ_comms.size()) / total_latent : 0.0;
}

inline double computeD4(int ck_id) {
    auto pressure = [](double req, double remain) {
        const double denom = req + std::max(remain, 0.0);
        return denom > 1e-9 ? req / denom : 0.0;
    };
    const auto& comm = g_communities.at(ck_id);
    const double r1 = pressure(comm.CLB,  g_fpga.remain_CLB);
    const double r2 = pressure(comm.BRAM, g_fpga.remain_BRAM);
    const double r3 = pressure(comm.DSP,  g_fpga.remain_DSP);
    return std::max({r1, r2, r3});
}

inline double computeD5(int ck_id) {
    const auto& tids = g_communities.at(ck_id).task_ids;
    std::set<int> ck_members(tids.begin(), tids.end());
    int unlockable = 0, total_waiting = 0;
    for (const auto& [cid, comm] : g_communities) {
        (void)cid;
        if (comm.state != CommunityState::ACTIVE) continue;
        for (int tid : comm.task_ids) {
            if (g_tasks.at(tid).state != TaskState::WAITING) continue;
            ++total_waiting;
            bool blocker_in_ck = false, blocker_elsewhere = false;
            for (int pid : g_tasks.at(tid).predecessors) {
                if (g_tasks.at(pid).state != TaskState::DONE) {
                    if (ck_members.count(pid)) blocker_in_ck     = true;
                    else                       blocker_elsewhere = true;
                }
            }
            if (blocker_in_ck && !blocker_elsewhere) ++unlockable;
        }
    }
    return total_waiting > 0 ? static_cast<double>(unlockable) / total_waiting : 0.0;
}

inline double computeD6(int ck_id, double t) {
    double rCLB = 0.0, rBRAM = 0.0, rDSP = 0.0;
    const auto& c = g_communities.at(ck_id);
    for (int tid : c.task_ids) {
        if (g_tasks.at(tid).state == TaskState::DONE) continue;
        const bool is_ready = (c.state == CommunityState::ACTIVE)
                                  ? (g_tasks.at(tid).state == TaskState::READY)
                                  : isHypotheticallyReady(tid, t);
        if (is_ready) {
            rCLB  += g_tasks.at(tid).CLB;
            rBRAM += g_tasks.at(tid).BRAM;
            rDSP  += g_tasks.at(tid).DSP;
        }
    }
    const double ratio_clb  = c.CLB  > 0 ? rCLB  / c.CLB  : 0.0;
    const double ratio_bram = c.BRAM > 0 ? rBRAM / c.BRAM : 0.0;
    const double ratio_dsp  = c.DSP  > 0 ? rDSP  / c.DSP  : 0.0;
    return (ratio_clb + ratio_bram + ratio_dsp) / 3.0;
}

inline double computeD7(int ck_id) {
    const auto& tids = g_communities.at(ck_id).task_ids;
    std::set<int> members(tids.begin(), tids.end());
    int blocked = 0, incomplete = 0;
    for (int tid : tids) {
        if (g_tasks.at(tid).state == TaskState::DONE) continue;
        ++incomplete;
        for (int pid : g_tasks.at(tid).predecessors) {
            if (!members.count(pid) && g_tasks.at(pid).state != TaskState::DONE) {
                ++blocked;
                break;
            }
        }
    }
    return incomplete > 0 ? static_cast<double>(blocked) / incomplete : 0.0;
}

inline double computeDynamicScore(int ck_id, double t, const Chromosome& chr) {
    const double d1 = computeD1(ck_id, t);
    const double d2 = computeD2(ck_id, t);
    const double d3 = computeD3(ck_id);
    const double d4 = computeD4(ck_id);
    const double d5 = computeD5(ck_id);
    const double d6 = computeD6(ck_id, t);
    const double d7 = computeD7(ck_id);
    return chr.beta[0] * d1 + chr.beta[1] * d2 + chr.beta[2] * d3
         + chr.beta[4] * d5 + chr.beta[5] * d6
         - chr.beta[3] * d4 - chr.beta[6] * d7;
}

inline bool canActivate(int ck_id, double t) {
    bool has_ready = false;
    const auto& c = g_communities.at(ck_id);
    for (int tid : c.task_ids) {
        if (g_tasks.at(tid).state == TaskState::DONE) continue;
        if (isHypotheticallyReady(tid, t)) { has_ready = true; break; }
    }
    if (!has_ready) return false;
    return c.CLB  <= g_fpga.remain_CLB
        && c.BRAM <= g_fpga.remain_BRAM
        && c.DSP  <= g_fpga.remain_DSP;
}

struct ActivationCandidate {
    int cid = -1;
    double score = 0.0;
    double min_ready_lst = 1e18;
    double cp_ready_urgency = 0.0;
    double block_pressure = 0.0;
    int urgent_blocks = 0;
    double cp_ready_feature = 0.0;
    double ready_lst_feature = 0.0;
    double block_count_feature = 0.0;
    double block_pressure_feature = 0.0;
};

inline double minReadyLST(int ck_id, double t) {
    double best = 1e18;
    for (int tid : g_communities.at(ck_id).task_ids) {
        if (g_tasks.at(tid).state == TaskState::DONE) continue;
        if (isHypotheticallyReady(tid, t))
            best = std::min(best, g_tasks.at(tid).LST);
    }
    return best;
}

inline double cpReadyUrgency(int ck_id, double t) {
    double urgency = 0.0;
    for (int tid : g_communities.at(ck_id).task_ids) {
        if (!g_global_cp_tasks.count(tid)) continue;
        if (g_tasks.at(tid).state == TaskState::DONE) continue;
        if (!isHypotheticallyReady(tid, t)) continue;
        const double slack_to_lst = std::max(g_tasks.at(tid).LST - t, 0.0);
        urgency = std::max(urgency, 1.0 + 1.0 / (1.0 + slack_to_lst));
    }
    return urgency;
}

inline bool fitsAfterTaking(int take_cid, int other_cid) {
    const auto& take  = g_communities.at(take_cid);
    const auto& other = g_communities.at(other_cid);
    return other.CLB  <= g_fpga.remain_CLB  - take.CLB
        && other.BRAM <= g_fpga.remain_BRAM - take.BRAM
        && other.DSP  <= g_fpga.remain_DSP  - take.DSP;
}

inline bool moreUrgentForReservation(const ActivationCandidate& a,
                                     const ActivationCandidate& b) {
    if (a.cp_ready_urgency > b.cp_ready_urgency + 1e-9) return true;
    if (b.cp_ready_urgency > a.cp_ready_urgency + 1e-9) return false;
    return a.min_ready_lst < b.min_ready_lst - 1e-9;
}

inline void annotateReservationPressure(std::vector<ActivationCandidate>& cand) {
    for (auto& a : cand) {
        a.urgent_blocks  = 0;
        a.block_pressure = 0.0;
        for (const auto& b : cand) {
            if (a.cid == b.cid) continue;
            if (!moreUrgentForReservation(b, a)) continue;
            if (fitsAfterTaking(a.cid, b.cid)) continue;

            const auto& take  = g_communities.at(a.cid);
            const auto& other = g_communities.at(b.cid);
            const double rem_clb  = g_fpga.remain_CLB  - take.CLB;
            const double rem_bram = g_fpga.remain_BRAM - take.BRAM;
            const double rem_dsp  = g_fpga.remain_DSP  - take.DSP;
            const double shortage =
                std::max(0.0, other.CLB  - rem_clb)  / std::max(1.0, static_cast<double>(g_fpga.total_CLB)) +
                std::max(0.0, other.BRAM - rem_bram) / std::max(1.0, static_cast<double>(g_fpga.total_BRAM)) +
                std::max(0.0, other.DSP  - rem_dsp)  / std::max(1.0, static_cast<double>(g_fpga.total_DSP));
            a.urgent_blocks++;
            a.block_pressure += shortage;
        }
    }
}

inline std::vector<ActivationCandidate> buildActivationCandidates(
    double t, const Chromosome& chr, bool ignore_theta = false) {
    std::vector<ActivationCandidate> cand;
    for (const auto& [cid, comm] : g_communities) {
        if (comm.state != CommunityState::LATENT) continue;
        if (!canActivate(cid, t)) continue;
        ActivationCandidate ac;
        ac.cid              = cid;
        ac.min_ready_lst    = minReadyLST(cid, t);
        ac.cp_ready_urgency = cpReadyUrgency(cid, t);
        cand.push_back(ac);
    }

    annotateReservationPressure(cand);
    const double peers = static_cast<double>(
        std::max<size_t>(1, cand.size() > 0 ? cand.size() - 1 : 0));
    for (auto& ac : cand) {
        ac.cp_ready_feature       = ac.cp_ready_urgency / 2.0;
        ac.ready_lst_feature      = 1.0 / (1.0 + std::max(ac.min_ready_lst - t, 0.0));
        ac.block_count_feature    = ac.urgent_blocks / peers;
        ac.block_pressure_feature = ac.block_pressure / (3.0 * peers);
        const double dyn = computeDynamicScore(ac.cid, t, chr)
                         + chr.beta[7]  * ac.cp_ready_feature
                         + chr.beta[8]  * ac.ready_lst_feature
                         - chr.beta[9]  * ac.block_count_feature
                         - chr.beta[10] * ac.block_pressure_feature;
        ac.score = chr.lambda * g_communities.at(ac.cid).pri + (1.0 - chr.lambda) * dyn;
        if (!std::isfinite(ac.score))
            throw std::runtime_error("Non-finite community score");
    }

    if (!ignore_theta) {
        cand.erase(std::remove_if(cand.begin(), cand.end(), [&](const auto& ac) {
            return ac.score <= chr.theta;
        }), cand.end());
    }
    std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.cid < b.cid;
    });
    return cand;
}

inline bool hasRunningTask() {
    for (const auto& [tid, task] : g_tasks) {
        (void)tid;
        if (task.state == TaskState::RUNNING) return true;
    }
    return false;
}

inline bool hasActiveReadyTask() {
    for (const auto& [tid, task] : g_tasks) {
        (void)tid;
        if (task.state == TaskState::READY) return true;
    }
    return false;
}

inline std::vector<ActivationCandidate> buildActivationCandidatesWithFallback(
    double t, const Chromosome& chr, bool& forced_threshold_bypass) {
    forced_threshold_bypass = false;
    std::vector<ActivationCandidate> cand = buildActivationCandidates(t, chr);
    if (!cand.empty()) return cand;
    if (hasRunningTask() || hasActiveReadyTask()) return cand;

    cand = buildActivationCandidates(t, chr, /*ignore_theta=*/true);
    forced_threshold_bypass = !cand.empty();
    return cand;
}

// ============================================================================
// Event-Driven Single-Residency Decoder (LATENT -> ACTIVE -> DONE, |L_k| = 1)
// ============================================================================

inline void activateCommunity(int ck_id, double t) {
    auto& comm = g_communities.at(ck_id);
    comm.state           = CommunityState::ACTIVE;
    comm.activation_time = t;
    comm.completion_time = -1.0;
    comm.activation_count++;
    g_fpga.remain_CLB  -= comm.CLB;
    g_fpga.remain_BRAM -= comm.BRAM;
    g_fpga.remain_DSP  -= comm.DSP;
    for (int tid : comm.task_ids) {
        if (g_tasks.at(tid).state == TaskState::DONE) continue;
        g_tasks.at(tid).state = isHypotheticallyReady(tid, t)
                                    ? TaskState::READY
                                    : TaskState::WAITING;
    }
}

inline void releaseCommunity(int ck_id, double t) {
    auto& comm = g_communities.at(ck_id);
    closeActiveEpisode(comm, t);
    comm.state           = CommunityState::DONE;
    comm.completion_time = t;
    g_fpga.remain_CLB  += comm.CLB;
    g_fpga.remain_BRAM += comm.BRAM;
    g_fpga.remain_DSP  += comm.DSP;
    g_nested_alloc.release(ck_id, t);
}

inline bool communityTasksDone(int cid) {
    auto it = g_communities.find(cid);
    if (it == g_communities.end()) return true;
    for (int tid : it->second.task_ids) {
        auto tit = g_tasks.find(tid);
        if (tit != g_tasks.end() && tit->second.state != TaskState::DONE)
            return false;
    }
    return true;
}

inline bool canReleaseNestedGroupNow(int cid) {
    for (int co_cid : g_nested_alloc.activeCommsInSameChildDrr(cid)) {
        auto it = g_communities.find(co_cid);
        if (it == g_communities.end()) continue;
        if (it->second.state == CommunityState::ACTIVE && !communityTasksDone(co_cid))
            return false;
    }
    return true;
}

inline double totalTaskCommunicationWeight() {
    double total = 0.0;
    for (const auto& e : g_edges_template) {
        if (e.communication < 0)
            throw std::invalid_argument("Negative task-edge communication");
        total += e.communication;
    }
    return total;
}

inline double totalTaskWeightedWork() {
    double total = 0.0;
    for (const auto& [id, task] : g_tasks_template) {
        (void)id;
        total += task.exec_time * (task.CLB + 2.5 * task.BRAM + 2.5 * task.DSP);
    }
    if (!std::isfinite(total) || total < 0.0)
        throw std::invalid_argument("Task weighted work must be finite and nonnegative");
    return total;
}

inline bool layoutPlaceAndActivate(int ck_id, double t) {
    const auto& comm = g_communities.at(ck_id);
    layout::CommunityResource demand;
    demand.clb_req  = comm.CLB;
    demand.bram_req = comm.BRAM;
    demand.dsp_req  = comm.DSP;

    const double exec_hint = computeS4(ck_id, g_nested_alloc.cfg().reseff_weight > 0.0);

    std::vector<int> ready_tasks;
    for (int tid : comm.task_ids) {
        if (g_tasks.at(tid).state != TaskState::DONE &&
            g_tasks.at(tid).state != TaskState::RUNNING &&
            isHypotheticallyReady(tid, t)) {
            ready_tasks.push_back(tid);
        }
    }
    bool has_cofitting_ready_peer = false;
    for (const auto& [other_cid, other_comm] : g_communities) {
        if (other_cid == ck_id || other_comm.state != CommunityState::LATENT) continue;
        if (canActivate(other_cid, t) && fitsAfterTaking(ck_id, other_cid)) {
            has_cofitting_ready_peer = true;
            break;
        }
    }

    g_nested_alloc.setPendingCommunicationTasks(ready_tasks);
    g_nested_alloc.setHasCofittingReadyPeer(has_cofitting_ready_peer);
    nested_dfx::PlaceDecision d = g_nested_alloc.place(ck_id, t, demand, exec_hint);
    g_nested_alloc.setHasCofittingReadyPeer(false);
    g_nested_alloc.setPendingCommunicationTasks({});
    g_last_decision = d;

    if (!d.success) {
        g_total_fail_cost += d.fail_cost;
        return false;
    }

    g_total_comm_cost += d.comm_cost;
    activateCommunity(ck_id, t);
    return true;
}

inline double runEventDrivenDecoder(const Chromosome& chr, std::ostream* log = nullptr) {
    resetWorkingData();
    computeStaticPriority(chr);

    double t = 0.0;
    constexpr int MAX_ITER = 10000;

    if (log) {
        *log << "## Event-Driven Schedule Trace\n\n";
        *log << "| Time | Event | Entity | Details |\n";
        *log << "|-----:|:------|:------:|:--------|\n";
    }

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // 1. Task completion
        for (auto& [tid, task] : g_tasks) {
            if (task.state == TaskState::RUNNING && task.finish_time <= t + 1e-9) {
                task.state = TaskState::DONE;
                if (log) {
                    *log << "| " << std::fixed << std::setprecision(1) << t
                         << " | TaskDone | T" << tid
                         << " | Community C" << task.community_id << " |\n";
                }
            }
        }

        // 2. Atomic cohort release for completed communities
        std::vector<int> completed;
        for (const auto& [cid, comm] : g_communities) {
            if (comm.state == CommunityState::ACTIVE &&
                communityTasksDone(cid) &&
                canReleaseNestedGroupNow(cid)) {
                completed.push_back(cid);
            }
        }
        g_nested_alloc.releaseGroup(completed, t);
        for (int cid : completed) {
            releaseCommunity(cid, t);
            if (log) {
                *log << "| " << std::fixed << std::setprecision(1) << t
                     << " | UnloadComm | C" << cid
                     << " | Atomic cohort release |\n";
            }
        }

        // 3. WAITING -> READY transitions inside active communities
        for (auto& [cid, comm] : g_communities) {
            (void)cid;
            if (comm.state != CommunityState::ACTIVE) continue;
            for (int tid : comm.task_ids) {
                if (g_tasks.at(tid).state != TaskState::WAITING) continue;
                bool all_pred = true;
                for (int pid : g_tasks.at(tid).predecessors) {
                    if (g_tasks.at(pid).state != TaskState::DONE) {
                        all_pred = false;
                        break;
                    }
                }
                if (all_pred && t >= g_tasks.at(tid).EST - 1e-9)
                    g_tasks.at(tid).state = TaskState::READY;
            }
        }

        // 4. Iterative community activation & nested-DFX physical slot placement
        while (true) {
            bool forced_threshold_bypass = false;
            std::vector<ActivationCandidate> cand_v =
                buildActivationCandidatesWithFallback(t, chr, forced_threshold_bypass);
            if (cand_v.empty()) break;

            if (forced_threshold_bypass && log) {
                *log << "| " << std::fixed << std::setprecision(1) << t
                     << " | IdleFallback | theta"
                     << " | System idle: bypass theta once |\n";
            }

            bool activated = false;
            for (const auto& ac : cand_v) {
                const int cid = ac.cid;
                const double sc = ac.score;
                const bool ok = layoutPlaceAndActivate(cid, t);
                if (!ok) {
                    if (log) {
                        *log << "| " << std::fixed << std::setprecision(1) << t
                             << " | PlaceReject | C" << cid
                             << " | Score=" << std::setprecision(4) << sc
                             << ", ReserveBlocks=" << ac.urgent_blocks << " |\n";
                    }
                    continue;
                }

                const auto& d = g_last_decision;
                if (log) {
                    *log << "| " << std::fixed << std::setprecision(1) << t
                         << " | ActivateComm | C" << cid
                         << " | Score=" << std::setprecision(4) << sc
                         << ", Mode=" << nested_dfx::placementModeName(d.mode)
                         << ", Parent=" << d.parent_id
                         << ", Slot=" << d.slot_id
                         << ", Config=" << d.configuration_id
                         << ", Rect(" << d.rect.x << "," << d.rect.y << ","
                         << d.rect.w << "," << d.rect.h << ") |\n";
                    for (const auto& [pid, new_rect] : d.shifted_parents) {
                        *log << "| " << std::fixed << std::setprecision(1) << t
                             << " | ReplayShift | Parent" << pid
                             << " | -> Rect(" << new_rect.x << "," << new_rect.y
                             << "," << new_rect.w << "," << new_rect.h << ") |\n";
                    }
                }
                activated = true;
                break;
            }
            if (!activated) break;
        }

        // 5. Dispatch READY -> RUNNING tasks and bind task residences
        for (auto& [cid, comm] : g_communities) {
            if (comm.state != CommunityState::ACTIVE) continue;
            for (int tid : comm.task_ids) {
                auto& task = g_tasks.at(tid);
                if (task.state == TaskState::READY) {
                    g_nested_alloc.bindTaskToActiveResidence(tid, cid);
                    task.state       = TaskState::RUNNING;
                    task.start_time  = t;
                    task.finish_time = t + task.exec_time;
                    if (log) {
                        *log << "| " << std::fixed << std::setprecision(1) << t
                             << " | StartTask | T" << tid
                             << " | C" << cid << ", ET=" << task.exec_time
                             << ", Finish@" << task.finish_time << " |\n";
                    }
                }
            }
        }

        // 6. Check full DAG completion and validate spatio-temporal invariants
        bool all_done = true;
        for (const auto& [tid, task] : g_tasks) {
            (void)tid;
            if (task.state != TaskState::DONE) { all_done = false; break; }
        }
        if (all_done) {
            double makespan = 0.0;
            for (const auto& [tid, task] : g_tasks) {
                (void)tid;
                makespan = std::max(makespan, task.finish_time);
            }
            std::string error;
            if (!g_nested_alloc.validate(&error)) {
                if (log) *log << "\n> Final layout validation failed: " << error << "\n";
                return -1.0;
            }
            g_nested_alloc.requireCompleteTaskCommunication();
            return makespan;
        }

        // 7. Advance to next event timestamp (strict single residency: deadlock -> infeasible)
        const double next_t = scheduling::nextEventTime(g_tasks, t);
        if (!std::isfinite(next_t)) {
            if (log) *log << "\n> Deadlock under single-residency constraint: no future event.\n";
            return -1.0;
        }
        t = next_t;
    }
    return -1.0;
}

inline double runSimulationSilent(const Chromosome& chr) {
    return runEventDrivenDecoder(chr, nullptr);
}

inline double runSimulationVerbose(const Chromosome& chr, std::ostream& out) {
    return runEventDrivenDecoder(chr, &out);
}

inline double weightedTaskWork(int cid, double w_bram = 2.5, double w_dsp = 2.5) {
    double work = 0.0;
    for (int tid : g_communities.at(cid).task_ids) {
        const auto& t = g_tasks.at(tid);
        work += t.exec_time * (static_cast<double>(t.CLB)
                             + w_bram * static_cast<double>(t.BRAM)
                             + w_dsp  * static_cast<double>(t.DSP));
    }
    return work;
}

inline nested_dfx::CommDrrPlacement lookupCommDrr(int cid) {
    const double act = (g_communities.at(cid).activation_time >= 0.0)
                           ? g_communities.at(cid).activation_time
                           : -1.0;
    return g_nested_alloc.findCommDrrPlacement(cid, act);
}

inline double computeFinalResourceBox() {
    double box = 0.0;
    for (const auto& [cid, comm] : g_communities) {
        const double residence = cumulativeResidenceTime(comm);
        if (residence < 0.0) continue;
        const nested_dfx::CommDrrPlacement drr = lookupCommDrr(cid);
        if (drr.valid)
            box += residence * static_cast<double>(drr.drr_rect.w) * static_cast<double>(drr.drr_rect.h);
    }
    return box;
}

inline double recomputeFinalCommCostFromState() {
    return g_nested_alloc.recomputeTotalCommCost(g_layout_params);
}

inline std::pair<double, std::vector<nested_dfx::CommCostPair>>
recomputeFinalCommCostDetailFromState() {
    return g_nested_alloc.recomputeCommCostDetail(g_layout_params);
}

// Initializes static community features and the Nested-DFX allocator for a preset.
inline void initDecoderWorkspace(const PresetParameters& preset_params) {
    g_fpga_template.total_CLB  = g_fpga_template.remain_CLB  = clustering::FPGA_MAX_CLB;
    g_fpga_template.total_BRAM = g_fpga_template.remain_BRAM = clustering::FPGA_MAX_BRAM;
    g_fpga_template.total_DSP  = g_fpga_template.remain_DSP  = clustering::FPGA_MAX_DSP;

    resetWorkingData();
    computeAllStaticFeatures();
    g_communities_template = g_communities;

    g_layout_fpga   = layout::FPGAModel::makeDefault();
    g_layout_params = layout::SearchParams();
    buildCommunityCommWeight();

    nested_dfx::NestedDFXConfig ncfg;
    ncfg.candidate_scoring = preset_params.enable_dmc_scoring
                                 ? nested_dfx::CandidateScoring::ACTION_ROLE_DMC
                                 : nested_dfx::CandidateScoring::ACTION_ROLE;
    ncfg.new_parent_weight = preset_params.new_parent_weight;
    ncfg.dmc_weight        = preset_params.dmc_weight;
    ncfg.reseff_weight     = preset_params.reseff_weight;
    ncfg.reseff_total_work = totalTaskWeightedWork();
    ncfg.dmc_total_weight  = totalTaskCommunicationWeight();
    ncfg.collect_debug_lines = false;

    g_nested_alloc.init(g_layout_fpga, ncfg, [](int a, int b) { return commWeight(a, b); });
}

} // namespace cps
