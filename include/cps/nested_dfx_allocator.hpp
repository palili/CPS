/*
 * nested_dfx_allocator.hpp — Hierarchical Nested-DFX Physical Slot Allocator
 *
 * Three-tier physical hierarchy on heterogeneous column-based FPGAs:
 *   ParentRegion    — Disjoint top-level FPGA rectangles with epoch versioning
 *   SubParentRegion — Second-tier containers inside a ParentRegion
 *   ChildDRR        — Leaf dynamically reconfigurable slots hosting time layers
 *
 * Every candidate plan is constructed on an isolated state copy, reconciled into
 * explicit slot/configuration identities (slot_semantics.hpp), and validated
 * against all 9 spatio-temporal invariants before competing for selection.
 */
#pragma once

#include "cps/fpga_grid.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nested_dfx {

using Rect = layout::Rect;
using FPGAModel = layout::FPGAModel;
using CommunityResource = layout::CommunityResource;
using RectCap = layout::RectCap;

constexpr double NDFX_EPS = 1e-9;

inline bool rectContains(const Rect& outer, const Rect& inner) {
    if (!outer.valid() || !inner.valid()) return false;
    return outer.x <= inner.x && outer.y <= inner.y &&
           outer.xr() >= inner.xr() && outer.yt() >= inner.yt();
}

inline bool rectPartialOverlap(const Rect& a, const Rect& b) {
    if (!Rect::overlap(a, b)) return false;
    return !(rectContains(a, b) || rectContains(b, a));
}

inline Rect boundingBox(const Rect& a, const Rect& b) {
    const int x  = std::min(a.x, b.x);
    const int y  = std::min(a.y, b.y);
    const int xr = std::max(a.xr(), b.xr());
    const int yt = std::max(a.yt(), b.yt());
    return {x, y, xr - x, yt - y};
}

inline bool rectsEqual(const Rect& a, const Rect& b) {
    return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h;
}

inline RectCap rectCapacity(const FPGAModel& m, const Rect& r) {
    if (!r.valid()) return RectCap{0, 0, 0};
    return layout::rectCapacity(m, r.x, r.xr() - 1, r.h);
}

inline bool capacitySatisfies(const RectCap& c, const CommunityResource& d) {
    return c.clb  + NDFX_EPS >= d.clb_req &&
           c.bram + NDFX_EPS >= d.bram_req &&
           c.dsp  + NDFX_EPS >= d.dsp_req;
}

inline CommunityResource maxDemand(const CommunityResource& a,
                                   const CommunityResource& b) {
    CommunityResource m;
    m.clb_req  = std::max(a.clb_req,  b.clb_req);
    m.bram_req = std::max(a.bram_req, b.bram_req);
    m.dsp_req  = std::max(a.dsp_req,  b.dsp_req);
    return m;
}

inline CommunityResource addDemand(const CommunityResource& a,
                                   const CommunityResource& b) {
    CommunityResource s;
    s.clb_req  = a.clb_req  + b.clb_req;
    s.bram_req = a.bram_req + b.bram_req;
    s.dsp_req  = a.dsp_req  + b.dsp_req;
    return s;
}

struct RegionConfiguration {
    int id = -1;
    int parent_id = -1;
    int owner_sub = -1;
    int parent_configuration = -1;
    int layout_version = -1;
    bool whole = false;
};

struct PlanningChange {
    double time = 0;
    std::string operation;
    int sub = -1, slot = -1;
    Rect before, after;
};

struct ChildDRR {
    int id              = -1;
    int parent_id       = -1;
    int layout_version  = 0;
    Rect rect;
    int owner_sub = -1;
    int configuration_id = -1;
    bool full_cover = false;
    bool parent_full = false;
    bool retained = false;
    int active_comm_id   = -1;
    int active_layer_idx = -1;
    std::set<int> active_comm_ids;
    std::map<int, int> active_layer_by_comm;
    CommunityResource max_hosted_demand;
    bool busy() const { return !active_comm_ids.empty(); }
};

struct SubParentRegion {
    int split_configuration = -1;
    int full_slot = -1;
    int id             = -1;
    int parent_id      = -1;
    int layout_version = 0;
    Rect rect;
    int active_comm_id   = -1;
    int active_layer_idx = -1;
    std::vector<int> covered_child_drrs;
    CommunityResource max_hosted_demand;
    bool active() const { return active_comm_id >= 0; }
};

struct ParentRegion {
    int split_configuration = -1;
    int full_slot = -1;
    int id                    = -1;
    Rect rect;
    int current_layout_version = 0;
    bool mono_busy            = false;
    int  mono_active_comm     = -1;
    int  mono_active_layer_idx = -1;
    std::vector<int> child_drr_ids;
    std::vector<int> subparent_ids;
    std::vector<int> version_history;
};

struct TimeLayer {
    int layout_version = -1;
    int    layer_idx     = -1;
    int    comm_id       = -1;
    Rect   rect;
    double t_on          = 0;
    double t_off         = -1;
    bool   active        = false;
    CommunityResource demand;
    double placement_cost = 0;
    int    mode_tag      = 0;
    std::vector<std::pair<double, Rect>> rect_changes;
};

struct ActivePlacement {
    enum Kind : int { CHILD = 0, SUBPARENT = 1, PARENT_MONO = 2 };
    Kind kind = CHILD;
    int  parent_id    = -1;
    int  container_id = -1;
    int  layer_idx    = -1;
};

struct CommDrrPlacement {
    ActivePlacement::Kind kind          = ActivePlacement::CHILD;
    int                   parent_id     = -1;
    int                   container_id  = -1;
    Rect                  drr_rect;
    Rect                  occupied_rect;
    double                t_on          = -1;
    double                t_off         = -1;
    int                   layer_idx     = -1;
    int                   mode_tag      = 0;
    bool                  valid         = false;
};

inline const char* containerKindName(ActivePlacement::Kind k) {
    switch (k) {
        case ActivePlacement::CHILD:       return "ChildDRR";
        case ActivePlacement::SUBPARENT:   return "SubParent";
        case ActivePlacement::PARENT_MONO: return "ParentMono";
    }
    return "?";
}

struct CommCostPair {
    int edge_id = -1, source_task = -1, target_task = -1;
    int version_a = -1, version_b = -1;
    int container_a = -1, container_b = -1;
    int layer_a = -1, layer_b = -1;
    int kind_a = -1, kind_b = -1;
    int    comm_a         = -1;
    int    comm_b         = -1;
    double weight         = 0.0;
    double dx             = 0.0;
    double dy             = 0.0;
    double dt             = 0.0;
    double d_norm         = 0.0;
    double dt_norm        = 0.0;
    bool   time_overlap   = false;
    bool   same_drr_reuse = false;
    bool   final_full_cover_reuse = false;
    int    owner_sub_a = -1, owner_sub_b = -1;
    int    configuration_a = -1, configuration_b = -1;
    int    parent_a       = -1;
    int    parent_b       = -1;
    int    relation_cost  = 3;
    std::string relation  = "diff_parent";
    double cost           = 0.0;
};

struct CommCostLayerSnap {
    int layout_version = -1;
    int owner_sub = -1;
    int configuration_id = -1;
    int parent_configuration = -1;
    bool verified_full_cover = false;
    Rect owner_sub_final_rect;
    int layer_idx = -1;
    int                   comm_id = -1;
    layout::Rect          rect;
    layout::Rect          final_rect;
    double                t_on = 0, t_off = -1;
    bool                  still_active = false;
    ActivePlacement::Kind kind = ActivePlacement::CHILD;
    int                   parent_id = -1;
    int                   container_id = -1;
};

struct TaskCommunicationEdge {
    int id = -1, source = -1, target = -1;
    double weight = 0;
};

struct TaskResidenceBinding {
    ActivePlacement placement;
    int layout_version = -1;
};

enum class PlacementMode : int {
    NONE = 0,
    CHILD_REUSE,
    CHILD_EXPAND,
    CHILD_NEW_IN_PARENT,
    CHILD_EXPAND_PARENT,
    CHILD_NEW_PARENT,
    SUB_REUSE,
    SUB_EXPAND,
    SUB_NEW_IN_PARENT,
    SUB_EXPAND_PARENT,
    MODE_A_USE_PARENT,
    SUB_NEW_PARENT,
    CHILD_MERGE_ACTIVE
};

inline const char* placementModeName(PlacementMode m) {
    switch (m) {
        case PlacementMode::CHILD_REUSE:         return "CHILD_REUSE";
        case PlacementMode::CHILD_EXPAND:        return "CHILD_EXPAND";
        case PlacementMode::CHILD_NEW_IN_PARENT: return "CHILD_NEW_IN_PARENT";
        case PlacementMode::CHILD_EXPAND_PARENT: return "CHILD_EXPAND_PARENT";
        case PlacementMode::CHILD_NEW_PARENT:    return "CHILD_NEW_PARENT";
        case PlacementMode::SUB_REUSE:           return "SUB_REUSE";
        case PlacementMode::SUB_EXPAND:          return "SUB_EXPAND";
        case PlacementMode::SUB_NEW_IN_PARENT:   return "SUB_NEW_IN_PARENT";
        case PlacementMode::SUB_EXPAND_PARENT:   return "SUB_EXPAND_PARENT";
        case PlacementMode::MODE_A_USE_PARENT:   return "MODE_A_USE_PARENT";
        case PlacementMode::SUB_NEW_PARENT:      return "SUB_NEW_PARENT";
        case PlacementMode::CHILD_MERGE_ACTIVE:  return "CHILD_MERGE_ACTIVE";
        default: return "NONE";
    }
}

struct PlaceDecision {
    bool   success      = false;
    PlacementMode mode  = PlacementMode::NONE;
    int    parent_id    = -1;
    int    container_id = -1;
    int    slot_id      = -1;
    int    configuration_id = -1;
    std::string structural_operation;
    Rect   rect;
    double cost         = 0;
    double comm_cost    = 0;
    double dmc_estimate_normalized = 0;
    double waste_cost   = 0;
    double action_cost  = 0;
    int    refreshed_parent_id = -1;
    double fail_cost    = 0;
    std::vector<std::pair<int, Rect>> shifted_parents;
    std::vector<std::string> debug_lines;
};

enum class CandidateScoring { ACTION_ROLE, ACTION_ROLE_DMC };

inline constexpr CandidateScoring DEFAULT_CANDIDATE_SCORING = CandidateScoring::ACTION_ROLE;
inline constexpr double DEFAULT_NEW_PARENT_WEIGHT = 0.20;
inline constexpr double DEFAULT_DMC_WEIGHT = 1.0;
inline constexpr double DEFAULT_RESEFF_WEIGHT = 0.0;

inline const char* candidateScoringName(CandidateScoring scoring) {
    switch (scoring) {
        case CandidateScoring::ACTION_ROLE:     return "action_role";
        case CandidateScoring::ACTION_ROLE_DMC: return "action_role_dmc";
    }
    return "action_role";
}

// Physical geometry and defragmentation heuristics for nested-DFX candidate scoring.
struct AllocatorHeuristics {
    double unsplit_sub_area_ratio    = 0.85;
    double waste_tighten_start_area  = 0.75;
    double waste_tighten_span        = 0.15;
    double base_moderate_waste_limit = 0.42;
    double waste_tighten_delta       = 0.04;
    double clb_dsp_dominance_margin  = 0.05;
    double tight_fit_waste           = 0.15;
    double child_moderate_fit_waste  = 0.42;
    double giant_container_area_ratio = 0.55;
    double giant_parent_area_ratio   = 0.65;
    double dense_top_area_ratio      = 0.85;
    double oversized_reuse_area_ratio = 0.50;
    double pure_reuse_discount_std   = 0.20;
    double pure_reuse_base_pack      = 0.15;
    double pure_reuse_slope_pack     = 0.75;
    double refresh_weight_fragmented = 0.01;
    double refresh_weight_normal     = 0.35;
    double covered_child_tight_waste = 0.35;
    double retro_expand_area_factor  = 1.25;
    double retro_sub_resided_ratio   = 0.25;
    double retro_pack_duration_mult  = 2.0;
    double new_container_weight      = 0.05;
    double unrefreshed_defrag_weight = 0.25;
    double movement_weight           = 0.05;
    double role_mismatch_weight      = 0.05;
    double dmc_local_blend           = 0.25;
    double dmc_global_blend          = 0.75;
    double reseff_retro_box_weight   = 0.25;
};

struct NestedDFXConfig {
    CandidateScoring candidate_scoring = DEFAULT_CANDIDATE_SCORING;
    double new_parent_weight = DEFAULT_NEW_PARENT_WEIGHT;
    double dmc_weight        = 0.0;
    double reseff_weight     = DEFAULT_RESEFF_WEIGHT;
    double reseff_total_work = 0.0;
    double dmc_total_weight  = 0.0;

    double large_clb_ratio  = 0.40;
    double large_bram_ratio = 0.40;
    double large_dsp_ratio  = 0.40;
    double small_clb_ratio  = 0.20;
    double small_bram_ratio = 0.20;
    double small_dsp_ratio  = 0.20;

    int default_child_w  = 20;
    int default_child_h  = 60;
    int default_sub_w    = 40;
    int default_sub_h    = 100;
    int default_parent_w = 40;
    int default_parent_h = 120;

    int grow_step_w = 1;
    int grow_step_h = 1;

    bool enable_global_replay       = true;
    int  replay_max_shifts          = 4;
    bool enable_refresh             = true;
    bool aggressive_refresh_on_idle = false;
    bool enable_active_child_merge  = true;
    double reuse_min_util           = 0.0;
    double huge_fail_cost           = 10.0;
    bool collect_debug_lines        = false;

    AllocatorHeuristics heuristics{};
};

class NestedDFXAllocator {
    friend struct SlotTestAccess;
public:
    NestedDFXAllocator() = default;

    void init(const FPGAModel& fpga,
              const NestedDFXConfig& cfg,
              std::function<double(int, int)> weight_fn) {
        if (!std::isfinite(cfg.new_parent_weight) || cfg.new_parent_weight < 0)
            throw std::invalid_argument("new_parent_weight must be finite and nonnegative");
        if (!std::isfinite(cfg.dmc_weight) || cfg.dmc_weight < 0 ||
            !std::isfinite(cfg.dmc_total_weight) || cfg.dmc_total_weight < 0 ||
            !std::isfinite(3.0 * cfg.dmc_total_weight))
            throw std::invalid_argument("ICC weight and normalization scale must be finite and nonnegative");
        if (!std::isfinite(cfg.reseff_weight) || cfg.reseff_weight < 0 ||
            !std::isfinite(cfg.reseff_total_work) || cfg.reseff_total_work < 0)
            throw std::invalid_argument("Invalid ResEff weight/normalization");
        fpga_      = fpga;
        cfg_       = cfg;
        weight_fn_ = std::move(weight_fn);
        reset();
    }

    void reset() {
        configurations_.clear();
        planning_changes_.clear();
        next_configuration_ = 1;
        layout_revision_    = 0;
        next_version_       = 1;
        parents_.clear();
        child_drrs_.clear();
        subparents_.clear();
        child_layers_.clear();
        sub_layers_.clear();
        parent_mono_layers_.clear();
        active_.clear();
        latest_resource_placement_.clear();
        task_edges_.clear();
        task_bindings_.clear();
        pending_communication_tasks_.clear();
        has_cofitting_ready_peer_ = false;
        next_parent_id_ = 1;
        next_child_id_  = 1;
        next_sub_id_    = 1;
    }

    enum class CommSizeClass { HARD_SMALL, FLEXIBLE, HARD_LARGE };

    void fpgaCapacities(double& cap_clb, double& cap_bram, double& cap_dsp) const {
        cap_clb  = static_cast<double>(fpga_.pre_clb.back()) * fpga_.H;
        cap_bram = static_cast<double>(fpga_.pre_bram.back()) *
                   std::floor(static_cast<double>(fpga_.H) / std::max(NDFX_EPS, fpga_.bram_cell_h));
        cap_dsp  = static_cast<double>(fpga_.pre_dsp.back()) *
                   std::floor(static_cast<double>(fpga_.H) / std::max(NDFX_EPS, fpga_.dsp_cell_h));
    }

    CommSizeClass classifyCommunity(const CommunityResource& d) const {
        double cap_clb, cap_bram, cap_dsp;
        fpgaCapacities(cap_clb, cap_bram, cap_dsp);
        const double r_clb  = (cap_clb  > NDFX_EPS) ? d.clb_req  / cap_clb  : 0.0;
        const double r_bram = (cap_bram > NDFX_EPS) ? d.bram_req / cap_bram : 0.0;
        const double r_dsp  = (cap_dsp  > NDFX_EPS) ? d.dsp_req  / cap_dsp  : 0.0;
        if (r_clb  > cfg_.large_clb_ratio)  return CommSizeClass::HARD_LARGE;
        if (r_bram > cfg_.large_bram_ratio) return CommSizeClass::HARD_LARGE;
        if (r_dsp  > cfg_.large_dsp_ratio)  return CommSizeClass::HARD_LARGE;
        if (r_clb  <= cfg_.small_clb_ratio &&
            r_bram <= cfg_.small_bram_ratio &&
            r_dsp  <= cfg_.small_dsp_ratio) return CommSizeClass::HARD_SMALL;
        return CommSizeClass::FLEXIBLE;
    }

    bool canSubdivideUnsplitSubParent(const SubParentRegion& sub, const CommunityResource& d) const {
        const double fpga_area = std::max(1.0, double(fpga_.W) * fpga_.H);
        const double area = double(sub.rect.w) * sub.rect.h;
        return area >= cfg_.heuristics.unsplit_sub_area_ratio * fpga_area &&
               sub.covered_child_drrs.empty() &&
               classifyCommunity(d) == CommSizeClass::FLEXIBLE;
    }

    double moderateWasteLimit(double container_area) const {
        const auto& h = cfg_.heuristics;
        const double fpga_area = std::max(1.0, double(fpga_.W) * fpga_.H);
        const double r = std::clamp(
            (container_area / fpga_area - h.waste_tighten_start_area) / h.waste_tighten_span,
            0.0, 1.0);
        return h.base_moderate_waste_limit - h.waste_tighten_delta * r;
    }

    static const char* sizeClassName(CommSizeClass c) {
        switch (c) {
            case CommSizeClass::HARD_SMALL: return "HARD_SMALL";
            case CommSizeClass::FLEXIBLE:   return "FLEXIBLE";
            case CommSizeClass::HARD_LARGE: return "HARD_LARGE";
        }
        return "?";
    }

    const std::map<int, ParentRegion>&          parents()    const { return parents_; }
    const std::map<int, ChildDRR>&              childDrrs()  const { return child_drrs_; }
    const std::map<int, SubParentRegion>&       subparents() const { return subparents_; }
    const std::map<int, std::vector<TimeLayer>>& childLayers() const { return child_layers_; }
    const std::map<int, std::vector<TimeLayer>>& subLayers()   const { return sub_layers_; }
    const std::map<int, std::vector<TimeLayer>>& parentMono()  const { return parent_mono_layers_; }
    const std::map<int, ActivePlacement>&       active()     const { return active_; }
    const std::map<int, RegionConfiguration>&   configurations() const { return configurations_; }
    const std::vector<PlanningChange>&          planningChanges() const { return planning_changes_; }
    int layoutRevision() const { return layout_revision_; }
    std::map<std::string, int> configurationRequirements(int slot) const;
    Rect reservationRect(int slot) const { return child_drrs_.at(slot).rect; }
    void releaseGroup(const std::vector<int>& communities, double time);
    const FPGAModel& fpga() const { return fpga_; }
    const NestedDFXConfig& cfg() const { return cfg_; }

    double accruedResourceBox(double t_now, int override_comm = -1,
        ActivePlacement::Kind override_kind = ActivePlacement::CHILD, int override_container = -1) const;

    void setTaskCommunicationEdges(std::vector<TaskCommunicationEdge> edges);
    void setPendingCommunicationTasks(const std::vector<int>& tasks) {
        pending_communication_tasks_ = std::set<int>(tasks.begin(), tasks.end());
    }
    void setHasCofittingReadyPeer(bool v) { has_cofitting_ready_peer_ = v; }
    void bindTaskToActiveResidence(int task_id, int comm_id);
    void requireCompleteTaskCommunication() const;
    const std::map<int, TaskResidenceBinding>& taskBindings() const { return task_bindings_; }
    const std::vector<TaskCommunicationEdge>& taskEdges() const { return task_edges_; }

    PlaceDecision place(int comm_id, double t_now,
                        const CommunityResource& demand,
                        double exec_time_estimate);

    void release(int comm_id, double t_off);

    bool validate(std::string* err) const;

    std::vector<layout::PlacedComm> snapshotForCommCost() const;

    double recomputeTotalCommCost(const layout::SearchParams& params) const;
    std::pair<double, std::vector<CommCostPair>>
    recomputeCommCostDetail(const layout::SearchParams& params) const;

    CommDrrPlacement findCommDrrPlacement(int comm_id,
                                          double activation_time = -1) const;
    std::vector<int> activeCommsInSameChildDrr(int comm_id) const;

private:
    FPGAModel                      fpga_;
    NestedDFXConfig                cfg_;
    std::function<double(int, int)> weight_fn_;

    std::map<int, RegionConfiguration> configurations_;
    std::vector<PlanningChange> planning_changes_;
    int next_configuration_ = 1;
    int layout_revision_    = 0;
    int makeConfiguration(int pid, int sid, int version, bool whole);
    int ensureFullSlot(int pid, int sid);
    void reconcileSlotHierarchy(int pid);
    bool validateSlots(std::string* error) const;

    std::map<int, ParentRegion>    parents_;
    std::map<int, ChildDRR>        child_drrs_;
    std::map<int, SubParentRegion> subparents_;

    std::map<int, std::vector<TimeLayer>> child_layers_;
    std::map<int, std::vector<TimeLayer>> sub_layers_;
    std::map<int, std::vector<TimeLayer>> parent_mono_layers_;

    std::map<int, ActivePlacement> active_;
    std::map<int, ActivePlacement> latest_resource_placement_;
    std::vector<TaskCommunicationEdge> task_edges_;
    std::map<int, TaskResidenceBinding> task_bindings_;
    std::set<int> pending_communication_tasks_;
    bool has_cofitting_ready_peer_ = false;

    CommCostLayerSnap taskResidenceSnapshot(const TaskResidenceBinding& binding) const;
    Rect dmcContainerRect(ActivePlacement::Kind kind, int container_id) const;
    void fillDmcSlotContext(CommCostLayerSnap& snapshot) const;

    int next_parent_id_ = 1;
    int next_child_id_  = 1;
    int next_sub_id_    = 1;
    int next_version_   = 1;

    bool withinFPGA(const Rect& r) const { return r.withinFPGA(fpga_); }

    bool parentEntirelyIdle(int pid) const {
        auto it = parents_.find(pid);
        if (it == parents_.end()) return false;
        const auto& parent = it->second;
        if (parent.mono_busy) return false;
        for (int did : parent.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit != child_drrs_.end() && cit->second.busy()) return false;
        }
        for (int sid : parent.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit != subparents_.end() && sit->second.active()) return false;
        }
        return true;
    }

    double bottleneckWaste(const Rect& r, const CommunityResource& d) const {
        if (!r.valid()) return 1.0;
        const RectCap cap = rectCapacity(fpga_, r);
        if (!capacitySatisfies(cap, d)) return 1.0;
        const double u_clb  = cap.clb  > NDFX_EPS ? d.clb_req  / cap.clb  : 0.0;
        const double u_bram = cap.bram > NDFX_EPS ? d.bram_req / cap.bram : 0.0;
        const double u_dsp  = cap.dsp  > NDFX_EPS ? d.dsp_req  / cap.dsp  : 0.0;
        return std::max(0.0, 1.0 - std::max({u_clb, u_bram, u_dsp}));
    }

    bool isClbBottlenecked(const CommunityResource& d) const {
        const RectCap fcap = rectCapacity(fpga_, Rect{0, 0, fpga_.W, fpga_.H});
        const double u_clb = fcap.clb > NDFX_EPS ? d.clb_req / fcap.clb : 0.0;
        const double u_dsp = fcap.dsp > NDFX_EPS ? d.dsp_req / fcap.dsp : 0.0;
        return u_clb > u_dsp + cfg_.heuristics.clb_dsp_dominance_margin;
    }

    bool diagnoseParentFragmentation(int pid, const CommunityResource& demand) const {
        if (pid < 0 || !parentEntirelyIdle(pid)) return false;
        auto pit = parents_.find(pid);
        if (pit == parents_.end()) return false;
        const auto& p = pit->second;
        const auto& h = cfg_.heuristics;
        const double fpga_area = double(fpga_.W) * fpga_.H;

        int top_containers = 0;
        int sub_containers = 0;
        bool has_vertical_uncovered_split = false;
        bool has_tight_fit       = false;
        bool has_moderate_fit    = false;
        bool has_giant_container = false;
        double top_area_sum = 0.0;

        for (int sid : p.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit == subparents_.end() || sit->second.layout_version != p.current_layout_version)
                continue;
            ++top_containers;
            ++sub_containers;
            const double area = double(sit->second.rect.w) * sit->second.rect.h;
            top_area_sum += area;
            const RectCap cap = rectCapacity(fpga_, sit->second.rect);
            if (!capacitySatisfies(cap, demand)) {
                if (p.subparent_ids.size() == 1 && p.child_drr_ids.size() <= 1)
                    has_moderate_fit = true;
            } else {
                const double bw = bottleneckWaste(sit->second.rect, demand);
                if (bw <= h.tight_fit_waste) has_tight_fit = true;
                if (bw <= moderateWasteLimit(area)) has_moderate_fit = true;
            }
            if (area >= h.giant_container_area_ratio * fpga_area &&
                !canSubdivideUnsplitSubParent(sit->second, demand))
                has_giant_container = true;
        }
        for (int did : p.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit == child_drrs_.end()) continue;
            const auto& drr = cit->second;
            if (drr.layout_version != p.current_layout_version || drr.retained || drr.full_cover)
                continue;
            const double area = double(drr.rect.w) * drr.rect.h;
            if (drr.owner_sub < 0) {
                ++top_containers;
                top_area_sum += area;
                if (drr.rect.y > p.rect.y) has_vertical_uncovered_split = true;
            }
            const double bw = bottleneckWaste(drr.rect, demand);
            if (bw <= h.tight_fit_waste) has_tight_fit = true;
            if (bw <= h.child_moderate_fit_waste) has_moderate_fit = true;
            if (area >= h.giant_container_area_ratio * fpga_area)
                has_giant_container = true;
        }

        const double parent_area = double(p.rect.w) * p.rect.h;
        const bool split_defrag =
            top_containers >= 2 && has_vertical_uncovered_split && !has_tight_fit;
        const bool giant_oversized_defrag =
            parent_area >= h.giant_parent_area_ratio * fpga_area &&
            (has_giant_container ||
             ((top_containers >= 3 || sub_containers >= 2) &&
              top_area_sum >= h.dense_top_area_ratio * parent_area)) &&
            !has_moderate_fit && (isClbBottlenecked(demand) || has_cofitting_ready_peer_);
        return split_defrag || giant_oversized_defrag;
    }

    struct PlacementPlan {
        bool retain_old_slot = false;
        PlacementMode mode = PlacementMode::NONE;
        int parent_id = -1;
        int container_id = -1;
        Rect parent_rect;
        Rect rect;
        Rect occupied_rect;
        std::vector<std::pair<int, Rect>> shifted_parents;
        std::map<int, Rect> packed_rects;
    };

    struct PlacementSearch;
    void considerPlan(int comm_id, double t_now, const CommunityResource& demand,
                      const PlacementPlan& plan, PlacementSearch& search) const;
    void evaluateSlotPlan(int comm_id, double t_now, const CommunityResource& demand,
                          const PlacementPlan& plan, PlacementSearch& search) const;
    void tryExistingContainerExpansion(
        int comm_id, double t_now, const CommunityResource& demand,
        PlacementMode mode, int parent_id, int container_id, const Rect& original,
        const CommunityResource& needed, const std::vector<Rect>& forbid,
        const std::vector<Rect>& soft_containment, PlacementSearch& search) const;
    void applyParentShift(int parent_id, const Rect& rect);

    void tryActionChildReuse(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;
    void tryActionChildMergeActive(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;
    void tryActionChildExpand(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;
    void tryActionChildNewInParent(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;
    void tryActionChildExpandParent(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;
    void tryActionChildNewParent(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;

    void tryActionSubReuse(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;
    void tryActionSubExpand(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;
    void tryActionSubNewInParent(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;
    void tryActionSubExpandParent(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;
    void tryActionModeA(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;
    void tryActionSubNewParent(int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const;

    bool growRectInContainer(Rect& rect,
                             const Rect& container,
                             const std::vector<Rect>& forbid,
                             const std::vector<Rect>& soft_containment,
                             const CommunityResource& demand) const;

    bool expandSubparentToContainChildren(Rect& sub_rect,
                                          const Rect& container,
                                          const std::vector<Rect>& forbid,
                                          const std::vector<Rect>& children) const;

    struct MergePackItem {
        int comm_id = -1;
        int layer_idx = -1;
        double t_on = 0.0;
        CommunityResource demand;
    };
    bool packChildMergeGroup(const Rect& container,
                             const std::vector<MergePackItem>& items,
                             std::map<int, Rect>& rect_by_comm) const;

    int createParentRegion(const Rect& rect);
    int createChildDRR(int parent_id, const Rect& rect);
    int createSubParent(int parent_id, const Rect& rect);

    int commitChildLayer(int child_drr_id, int comm_id, double t_now,
                         const CommunityResource& demand, int mode_tag,
                         double placement_cost,
                         const Rect* occupied_rect = nullptr);
    int commitSubLayer(int sub_id, int comm_id, double t_now,
                       const CommunityResource& demand, int mode_tag,
                       double placement_cost);
    int commitParentMonoLayer(int parent_id, int comm_id, double t_now,
                              const CommunityResource& demand,
                              double placement_cost);

    void recomputeSubContainments(int parent_id);

    bool tryReplayExpandParent(int parent_id, const Rect& target_rect,
                               std::vector<std::pair<int, Rect>>& shifted_out) const;
    bool validateShiftPlan(const std::map<int, Rect>& new_parent_rects,
                           std::string* err = nullptr) const;
    bool refreshParentLayoutVersion(int parent_id);

    double estimateDmcIncrement(int comm_id, double t_now, ActivePlacement::Kind kind,
                                int parent_id, int container_id,
                                double* resolved_weight_out = nullptr) const;
    double computeWasteCost(const Rect& rect, const CommunityResource& d) const;

    inline void dbg(PlaceDecision& d, std::string&& s) const {
        if (cfg_.collect_debug_lines) d.debug_lines.push_back(std::move(s));
    }
};

struct NestedDFXAllocator::PlacementSearch {
    std::unique_ptr<NestedDFXAllocator> state;
    PlaceDecision best;
    int refreshed_parent_id = -1;
    bool refreshed_parent_fragmented = false;
    size_t generated = 0;
    size_t feasible  = 0;
    double resource_box_before        = 0;
    double resource_duration_estimate = 0;
};

inline double NestedDFXAllocator::accruedResourceBox(
    double t_now, int override_comm,
    ActivePlacement::Kind override_kind, int override_container) const {
    if (!std::isfinite(t_now) || t_now < 0)
        throw std::invalid_argument("Invalid resource-history time");
    std::map<int, double> elapsed;
    auto collect = [&](const auto& groups) {
        for (const auto& [id, layers] : groups) {
            (void)id;
            for (const auto& layer : layers) {
                const double end = layer.active ? t_now : layer.t_off;
                if (end < layer.t_on || end > t_now)
                    throw std::logic_error("Resource history contains invalid/future residence");
                elapsed[layer.comm_id] += end - layer.t_on;
            }
        }
    };
    collect(child_layers_);
    collect(sub_layers_);
    collect(parent_mono_layers_);
    double box = 0;
    for (const auto& [cid, duration] : elapsed) {
        const auto& binding = latest_resource_placement_.at(cid);
        const Rect rect = cid == override_comm
            ? dmcContainerRect(override_kind, override_container)
            : dmcContainerRect(binding.kind, binding.container_id);
        box += double(rect.w) * rect.h * duration;
    }
    return box;
}

inline void NestedDFXAllocator::applyParentShift(int parent_id, const Rect& rect) {
    auto& parent = parents_.at(parent_id);
    const int dx = rect.x - parent.rect.x;
    const int dy = rect.y - parent.rect.y;
    auto move_rect = [&](Rect& r) { r.x += dx; r.y += dy; };
    auto move_layers = [&](std::vector<TimeLayer>& layers) {
        for (auto& layer : layers) {
            move_rect(layer.rect);
            for (auto& change : layer.rect_changes) move_rect(change.second);
        }
    };
    for (auto& [did, child] : child_drrs_) {
        if (child.parent_id != parent_id) continue;
        move_rect(child.rect);
        move_layers(child_layers_[did]);
    }
    for (auto& [sid, sub] : subparents_) {
        if (sub.parent_id != parent_id) continue;
        move_rect(sub.rect);
        move_layers(sub_layers_[sid]);
    }
    move_layers(parent_mono_layers_[parent_id]);
    parent.rect = rect;
}

inline void NestedDFXAllocator::considerPlan(
    int comm_id, double t_now, const CommunityResource& demand,
    const PlacementPlan& plan, PlacementSearch& search) const {
    const size_t before_feasible = search.feasible;
    evaluateSlotPlan(comm_id, t_now, demand, plan, search);
    const bool sub_action = plan.mode == PlacementMode::SUB_REUSE ||
                            plan.mode == PlacementMode::SUB_EXPAND;
    if (search.feasible == before_feasible &&
        sub_action && plan.container_id >= 0 && subparents_.count(plan.container_id)) {
        const auto& sub = subparents_.at(plan.container_id);
        if (sub.full_slot >= 0 && !rectsEqual(sub.rect, plan.rect) &&
            !child_drrs_.at(sub.full_slot).busy()) {
            auto retained = plan;
            retained.retain_old_slot = true;
            evaluateSlotPlan(comm_id, t_now, demand, retained, search);
        }
    }
}

inline void NestedDFXAllocator::evaluateSlotPlan(
    int comm_id, double t_now, const CommunityResource& demand,
    const PlacementPlan& plan, PlacementSearch& search) const {
    if (search.refreshed_parent_id >= 0 &&
        plan.parent_id != search.refreshed_parent_id) return;
    ++search.generated;
    NestedDFXAllocator trial = *this;
    for (const auto& [pid, rect] : plan.shifted_parents)
        trial.applyParentShift(pid, rect);

    PlaceDecision candidate;
    candidate.success             = true;
    candidate.mode                = plan.mode;
    candidate.parent_id           = plan.parent_id;
    candidate.container_id        = plan.container_id;
    candidate.rect                = plan.occupied_rect.valid() ? plan.occupied_rect : plan.rect;
    candidate.shifted_parents     = plan.shifted_parents;
    candidate.refreshed_parent_id = search.refreshed_parent_id;
    if (candidate.parent_id < 0)
        candidate.parent_id = trial.createParentRegion(plan.parent_rect);
    else if (plan.parent_rect.valid())
        trial.parents_.at(candidate.parent_id).rect = plan.parent_rect;

    ActivePlacement::Kind kind = ActivePlacement::CHILD;
    switch (plan.mode) {
        case PlacementMode::SUB_REUSE:
        case PlacementMode::SUB_EXPAND:
        case PlacementMode::SUB_NEW_IN_PARENT:
        case PlacementMode::SUB_EXPAND_PARENT:
        case PlacementMode::SUB_NEW_PARENT:
            kind = ActivePlacement::SUBPARENT;
            break;
        case PlacementMode::MODE_A_USE_PARENT:
            kind = ActivePlacement::PARENT_MONO;
            break;
        default:
            break;
    }

    CommunityResource container_demand = demand;
    if (kind == ActivePlacement::CHILD) {
        if (candidate.container_id < 0)
            candidate.container_id = trial.createChildDRR(candidate.parent_id, plan.rect);
        auto& child = trial.child_drrs_.at(candidate.container_id);
        child.rect = plan.rect;
        for (const auto& [cid, rect] : plan.packed_rects) {
            if (cid == comm_id) continue;
            auto& layer = trial.child_layers_.at(child.id).at(child.active_layer_by_comm.at(cid));
            container_demand = addDemand(container_demand, layer.demand);
            if (!rectsEqual(layer.rect, rect)) {
                layer.rect = rect;
                if (!layer.rect_changes.empty() &&
                    std::abs(layer.rect_changes.back().first - t_now) <= NDFX_EPS)
                    layer.rect_changes.back().second = rect;
                else
                    layer.rect_changes.push_back({t_now, rect});
            }
        }
        child.max_hosted_demand = maxDemand(child.max_hosted_demand, container_demand);
    } else if (kind == ActivePlacement::SUBPARENT) {
        if (candidate.container_id < 0)
            candidate.container_id = trial.createSubParent(candidate.parent_id, plan.rect);
        auto& sub = trial.subparents_.at(candidate.container_id);
        const Rect before = sub.rect;
        if (plan.retain_old_slot && sub.full_slot >= 0) {
            auto& old = trial.child_drrs_.at(sub.full_slot);
            old.full_cover       = false;
            old.retained         = true;
            old.configuration_id = -1;
            trial.parents_.at(candidate.parent_id).child_drr_ids.push_back(old.id);
            trial.planning_changes_.push_back({t_now, "RETAIN_SLOT", sub.id, old.id, old.rect, old.rect});
            sub.full_slot = -1;
        }
        sub.rect = plan.rect;
        if (!rectsEqual(before, sub.rect))
            trial.planning_changes_.push_back({t_now, "EXPAND_SUB", sub.id, sub.full_slot, before, sub.rect});
    }

    trial.reconcileSlotHierarchy(candidate.parent_id);
    int effective_slot = candidate.container_id;
    if (kind == ActivePlacement::SUBPARENT)
        effective_slot = trial.ensureFullSlot(candidate.parent_id, candidate.container_id);
    else if (kind == ActivePlacement::PARENT_MONO)
        effective_slot = trial.ensureFullSlot(candidate.parent_id, -1);
    trial.reconcileSlotHierarchy(candidate.parent_id);
    ++trial.layout_revision_;

    candidate.slot_id          = effective_slot;
    candidate.configuration_id = trial.child_drrs_.at(effective_slot).configuration_id;
    candidate.structural_operation = placementModeName(plan.mode);
    if (plan.retain_old_slot) {
        candidate.structural_operation += "+RETAIN_OLD_SLOT";
    } else if (kind == ActivePlacement::SUBPARENT && plan.container_id >= 0 &&
               !rectsEqual(subparents_.at(plan.container_id).rect, plan.rect)) {
        candidate.structural_operation += "+EXPAND_FULL_COVER";
    }

    const auto& h = cfg_.heuristics;
    const RectCap full = rectCapacity(fpga_, {0, 0, fpga_.W, fpga_.H});
    auto ratio = [](double amount, double total) {
        return total > NDFX_EPS ? amount / total : 0.0;
    };
    auto fraction = [&](const RectCap& cap) {
        return (ratio(cap.clb, full.clb) + ratio(cap.bram, full.bram) +
                ratio(cap.dsp, full.dsp)) / 3.0;
    };
    const RectCap post = rectCapacity(fpga_, trial.parents_.at(candidate.parent_id).rect);
    double growth = fraction(post);
    if (plan.parent_id >= 0) {
        const RectCap pre = rectCapacity(fpga_, parents_.at(plan.parent_id).rect);
        growth = fraction({std::max(0.0, post.clb - pre.clb),
                           std::max(0.0, post.bram - pre.bram),
                           std::max(0.0, post.dsp - pre.dsp)});
    }
    auto moved = [&](const Rect& old_rect, const Rect& next_rect) {
        return fraction(rectCapacity(fpga_, old_rect)) *
            (double(std::abs(next_rect.cx() - old_rect.cx())) / fpga_.W +
             double(std::abs(next_rect.cy() - old_rect.cy())) / fpga_.H);
    };
    double movement = 0.0;
    for (const auto& [pid, rect] : plan.shifted_parents)
        movement += moved(parents_.at(pid).rect, rect);
    if (kind == ActivePlacement::CHILD && plan.container_id >= 0) {
        const auto& old_child = child_drrs_.at(plan.container_id);
        for (const auto& [cid, rect] : plan.packed_rects) {
            const auto it = old_child.active_layer_by_comm.find(cid);
            if (it != old_child.active_layer_by_comm.end()) {
                const auto& tl = child_layers_.at(old_child.id).at(it->second);
                if (std::abs(tl.t_on - t_now) > NDFX_EPS)
                    movement += moved(tl.rect, rect);
            }
        }
    }

    const double fpga_area = double(fpga_.W) * fpga_.H;
    candidate.waste_cost = trial.computeWasteCost(plan.rect, container_demand) / 3.0;
    const bool unrefreshed_defrag_needed =
        search.refreshed_parent_id < 0 && diagnoseParentFragmentation(plan.parent_id, demand);
    const bool merge_expands_child =
        plan.mode == PlacementMode::CHILD_MERGE_ACTIVE &&
        plan.container_id >= 0 && child_drrs_.count(plan.container_id) &&
        !rectsEqual(child_drrs_.at(plan.container_id).rect, plan.rect);
    const bool oversized_clb_reuse =
        plan.mode != PlacementMode::CHILD_MERGE_ACTIVE &&
        double(plan.rect.w) * plan.rect.h >= h.oversized_reuse_area_ratio * fpga_area &&
        (isClbBottlenecked(container_demand) || has_cofitting_ready_peer_) &&
        bottleneckWaste(plan.rect, container_demand) > moderateWasteLimit(double(plan.rect.w) * plan.rect.h);
    const bool pure_reuse =
        plan.parent_id >= 0 && search.refreshed_parent_id < 0 && !plan.retain_old_slot &&
        !unrefreshed_defrag_needed && !oversized_clb_reuse && !merge_expands_child &&
        (plan.mode == PlacementMode::CHILD_REUSE ||
         plan.mode == PlacementMode::SUB_REUSE ||
         plan.mode == PlacementMode::CHILD_MERGE_ACTIVE);
    const double reuse_bw = bottleneckWaste(plan.rect, container_demand);
    if (pure_reuse) {
        candidate.waste_cost *= cfg_.reseff_weight > 0
            ? (h.pure_reuse_base_pack + h.pure_reuse_slope_pack * reuse_bw)
            : h.pure_reuse_discount_std;
    }

    const bool new_parent    = plan.parent_id < 0;
    const bool new_container = kind != ActivePlacement::PARENT_MONO &&
                               (plan.container_id < 0 || plan.retain_old_slot || merge_expands_child);
    const bool refresh       = candidate.refreshed_parent_id >= 0;
    const bool refreshed_fragmented = refresh && search.refreshed_parent_fragmented;
    const double refresh_weight = refreshed_fragmented
        ? h.refresh_weight_fragmented : h.refresh_weight_normal;
    const CommSizeClass cls = classifyCommunity(demand);
    bool tight_covered_child_reuse = false;
    if (cfg_.reseff_weight > 0 && plan.mode == PlacementMode::CHILD_REUSE &&
        plan.container_id >= 0 && reuse_bw <= h.covered_child_tight_waste) {
        const auto cit = child_drrs_.find(plan.container_id);
        tight_covered_child_reuse = cit != child_drrs_.end() && cit->second.owner_sub >= 0;
    }
    const bool role_mismatch =
        (plan.mode != PlacementMode::CHILD_MERGE_ACTIVE || merge_expands_child) &&
        !tight_covered_child_reuse &&
        ((cls == CommSizeClass::HARD_LARGE && kind == ActivePlacement::CHILD) ||
         (cls == CommSizeClass::HARD_SMALL && kind != ActivePlacement::CHILD));

    double retro_expand = 0.0;
    if (kind == ActivePlacement::CHILD && plan.container_id >= 0 && t_now > NDFX_EPS) {
        const auto cit = child_drrs_.find(plan.container_id);
        if (cit != child_drrs_.end()) {
            const double old_area = double(cit->second.rect.w) * cit->second.rect.h;
            const double new_area = double(plan.rect.w) * plan.rect.h;
            if (new_area > h.retro_expand_area_factor * old_area) {
                double resided = 0.0;
                const auto lit = child_layers_.find(plan.container_id);
                if (lit != child_layers_.end()) {
                    for (const auto& layer : lit->second) {
                        const double end = layer.active ? t_now : layer.t_off;
                        if (end > layer.t_on) resided += end - layer.t_on;
                    }
                }
                const bool protect_slot =
                    (cit->second.owner_sub >= 0 && resided >= h.retro_sub_resided_ratio * t_now) ||
                    (cfg_.reseff_weight > 0 && search.resource_duration_estimate > NDFX_EPS &&
                     resided > h.retro_pack_duration_mult * search.resource_duration_estimate);
                if (protect_slot)
                    retro_expand = (new_area - old_area) * resided / (fpga_area * t_now);
            }
        }
    }

    const double retro_weight = cfg_.reseff_weight > 0 ? 2.0 : 1.0;
    candidate.action_cost = growth + cfg_.new_parent_weight * new_parent
                          + h.new_container_weight * new_container
                          + refresh_weight * refresh
                          + h.unrefreshed_defrag_weight * unrefreshed_defrag_needed
                          + h.movement_weight * movement
                          + retro_weight * retro_expand
                          + h.role_mismatch_weight * role_mismatch;
    candidate.comm_cost = 0.0;
    candidate.cost      = candidate.waste_cost + candidate.action_cost;

    if (cfg_.candidate_scoring == CandidateScoring::ACTION_ROLE_DMC && cfg_.dmc_weight > 0) {
        double resolved_weight = 0.0;
        candidate.comm_cost = trial.estimateDmcIncrement(
            comm_id, t_now, ActivePlacement::CHILD,
            candidate.parent_id, effective_slot, &resolved_weight);
        const double global_norm = cfg_.dmc_total_weight > NDFX_EPS
            ? candidate.comm_cost / (3.0 * cfg_.dmc_total_weight) : 0.0;
        const double local_norm = resolved_weight > NDFX_EPS
            ? candidate.comm_cost / (3.0 * resolved_weight) : 0.0;
        candidate.dmc_estimate_normalized =
            h.dmc_local_blend * local_norm + h.dmc_global_blend * global_norm;
        candidate.cost += cfg_.dmc_weight * candidate.dmc_estimate_normalized;
    }

    if (cfg_.reseff_weight > 0 && cfg_.reseff_total_work > 0) {
        const double after = trial.accruedResourceBox(
            t_now, comm_id, ActivePlacement::CHILD, effective_slot);
        const Rect final_rect = trial.dmcContainerRect(ActivePlacement::CHILD, effective_slot);
        const double increment =
            h.reseff_retro_box_weight * std::max(0.0, after - search.resource_box_before) +
            double(final_rect.w) * final_rect.h * search.resource_duration_estimate;
        candidate.cost += cfg_.reseff_weight * increment / cfg_.reseff_total_work;
    }
    if (!std::isfinite(candidate.cost) || !std::isfinite(candidate.action_cost)) return;

    trial.commitChildLayer(effective_slot, comm_id, t_now, demand,
                           static_cast<int>(plan.mode), candidate.cost, &candidate.rect);
    if (kind == ActivePlacement::SUBPARENT) {
        auto& sub = trial.subparents_.at(candidate.container_id);
        sub.active_comm_id    = comm_id;
        sub.max_hosted_demand = maxDemand(sub.max_hosted_demand, demand);
    } else if (kind == ActivePlacement::PARENT_MONO) {
        auto& parent = trial.parents_.at(candidate.parent_id);
        parent.mono_busy        = true;
        parent.mono_active_comm = comm_id;
    }

    std::string rejection;
    if (!trial.validate(cfg_.collect_debug_lines ? &rejection : nullptr)) {
        if (cfg_.collect_debug_lines)
            search.best.debug_lines.push_back(
                std::string("REJECT ") + placementModeName(plan.mode) + ": " + rejection);
        return;
    }
    ++search.feasible;
    const bool better = !search.state ||
        (candidate.cost < search.best.cost - NDFX_EPS ||
         (std::abs(candidate.cost - search.best.cost) <= NDFX_EPS &&
          candidate.action_cost < search.best.action_cost - NDFX_EPS));
    if (!better) return;

    trial.dbg(candidate, std::string(placementModeName(candidate.mode)) +
              " parent=" + std::to_string(candidate.parent_id) +
              " container=" + std::to_string(candidate.container_id));
    if (candidate.refreshed_parent_id >= 0)
        trial.dbg(candidate, "REFRESH parent=" + std::to_string(candidate.refreshed_parent_id));
    search.best  = std::move(candidate);
    search.state = std::make_unique<NestedDFXAllocator>(std::move(trial));
}

inline bool NestedDFXAllocator::growRectInContainer(
    Rect& rect,
    const Rect& container,
    const std::vector<Rect>& forbid,
    const std::vector<Rect>& soft_containment,
    const CommunityResource& demand) const {
    if (!rect.valid() || !container.valid()) return false;
    if (!rectContains(container, rect)) return false;

    auto satisfy = [&](const Rect& r) {
        if (!r.withinFPGA(fpga_)) return false;
        if (!rectContains(container, r)) return false;
        for (const auto& f : forbid) if (Rect::overlap(r, f)) return false;
        for (const auto& s : soft_containment) if (rectPartialOverlap(r, s)) return false;
        return capacitySatisfies(rectCapacity(fpga_, r), demand);
    };

    if (satisfy(rect)) return true;

    auto absorbPartials = [&](Rect& r) -> bool {
        for (int iter = 0; iter < 16; iter++) {
            bool changed = false;
            for (const auto& s : soft_containment) {
                if (rectPartialOverlap(r, s)) {
                    const Rect bb = boundingBox(r, s);
                    if (!rectContains(container, bb) || !bb.withinFPGA(fpga_)) return false;
                    bool clash = false;
                    for (const auto& f : forbid) if (Rect::overlap(bb, f)) { clash = true; break; }
                    if (clash) return false;
                    r = bb;
                    changed = true;
                }
            }
            if (!changed) return true;
        }
        return false;
    };

    Rect cur = rect;
    if (!absorbPartials(cur)) return false;

    const int max_iter = (container.w + container.h) * 2 + 4;
    int iter = 0;
    while (iter++ < max_iter) {
        if (satisfy(cur)) {
            rect = cur;
            return true;
        }
        struct Cand { Rect r; int penalty; int dir; };
        std::vector<Cand> cands;
        {
            Rect r = cur;
            r.w += cfg_.grow_step_w;
            if (rectContains(container, r) && r.withinFPGA(fpga_)) {
                bool ok = true;
                for (const auto& f : forbid) if (Rect::overlap(r, f)) { ok = false; break; }
                if (ok) cands.push_back({r, cfg_.grow_step_w, 0});
            }
        }
        if (cur.x - cfg_.grow_step_w >= container.x) {
            Rect r = cur;
            r.x -= cfg_.grow_step_w;
            r.w += cfg_.grow_step_w;
            if (rectContains(container, r) && r.withinFPGA(fpga_)) {
                bool ok = true;
                for (const auto& f : forbid) if (Rect::overlap(r, f)) { ok = false; break; }
                if (ok) cands.push_back({r, cfg_.grow_step_w, 1});
            }
        }
        {
            Rect r = cur;
            r.h += cfg_.grow_step_h;
            if (rectContains(container, r) && r.withinFPGA(fpga_)) {
                bool ok = true;
                for (const auto& f : forbid) if (Rect::overlap(r, f)) { ok = false; break; }
                if (ok) cands.push_back({r, cfg_.grow_step_h, 2});
            }
        }
        if (cur.y - cfg_.grow_step_h >= container.y) {
            Rect r = cur;
            r.y -= cfg_.grow_step_h;
            r.h += cfg_.grow_step_h;
            if (rectContains(container, r) && r.withinFPGA(fpga_)) {
                bool ok = true;
                for (const auto& f : forbid) if (Rect::overlap(r, f)) { ok = false; break; }
                if (ok) cands.push_back({r, cfg_.grow_step_h, 3});
            }
        }
        if (cands.empty()) return false;

        int best_i = -1;
        double best_score = -1;
        for (size_t i = 0; i < cands.size(); i++) {
            const RectCap cap = rectCapacity(fpga_, cands[i].r);
            const double score =
                std::max(0.0, demand.clb_req  - std::min(demand.clb_req,  cap.clb)) +
                std::max(0.0, demand.bram_req - std::min(demand.bram_req, cap.bram)) +
                std::max(0.0, demand.dsp_req  - std::min(demand.dsp_req,  cap.dsp));
            const double composite = score * 1000.0 + double(cands[i].r.w * cands[i].r.h);
            if (best_i < 0 || composite < best_score) {
                best_score = composite;
                best_i     = static_cast<int>(i);
            }
        }
        if (best_i < 0) return false;
        Rect next = cands[best_i].r;
        if (!absorbPartials(next)) return false;
        if (rectsEqual(cur, next)) return false;
        cur = next;
    }
    return false;
}

inline bool NestedDFXAllocator::packChildMergeGroup(
    const Rect& container,
    const std::vector<MergePackItem>& items,
    std::map<int, Rect>& rect_by_comm) const {
    rect_by_comm.clear();
    if (!container.valid() || items.empty()) return false;

    std::vector<MergePackItem> order = items;
    std::stable_sort(order.begin(), order.end(),
        [](const MergePackItem& a, const MergePackItem& b) {
            if (std::abs(a.t_on - b.t_on) > NDFX_EPS) return a.t_on < b.t_on;
            return a.comm_id < b.comm_id;
        });

    auto fits = [&](const Rect& r, const CommunityResource& d) {
        return r.valid() && rectContains(container, r) &&
               capacitySatisfies(rectCapacity(fpga_, r), d);
    };

    auto try_vertical = [&]() -> bool {
        int y = container.y;
        std::map<int, Rect> out;
        for (const auto& item : order) {
            Rect chosen;
            for (int h = 1; y + h <= container.yt(); ++h) {
                Rect cand{container.x, y, container.w, h};
                if (fits(cand, item.demand)) { chosen = cand; break; }
            }
            if (!chosen.valid()) return false;
            out[item.comm_id] = chosen;
            y = chosen.yt();
        }
        rect_by_comm.swap(out);
        return true;
    };

    auto try_horizontal = [&]() -> bool {
        int x = container.x;
        std::map<int, Rect> out;
        for (const auto& item : order) {
            Rect chosen;
            for (int w = 1; x + w <= container.xr(); ++w) {
                Rect cand{x, container.y, w, container.h};
                if (fits(cand, item.demand)) { chosen = cand; break; }
            }
            if (!chosen.valid()) return false;
            out[item.comm_id] = chosen;
            x = chosen.xr();
        }
        rect_by_comm.swap(out);
        return true;
    };

    if (try_vertical()) return true;
    if (try_horizontal()) return true;

    std::map<int, Rect> out;
    std::vector<Rect> placed;
    for (const auto& item : order) {
        std::vector<std::pair<int, int>> anchors;
        anchors.push_back({container.x, container.y});
        for (const auto& r : placed) {
            anchors.push_back({r.xr(), r.y});
            anchors.push_back({r.x, r.yt()});
            anchors.push_back({r.xr(), r.yt()});
            anchors.push_back({container.x, r.yt()});
            anchors.push_back({r.xr(), container.y});
        }
        std::sort(anchors.begin(), anchors.end());
        anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());

        Rect best;
        int best_area = std::numeric_limits<int>::max();
        for (const auto& [ax, ay] : anchors) {
            if (ax < container.x || ay < container.y ||
                ax >= container.xr() || ay >= container.yt())
                continue;
            const int w0 = std::min(cfg_.default_child_w, container.xr() - ax);
            const int h0 = std::min(cfg_.default_child_h, container.yt() - ay);
            const Rect cand{ax, ay, w0, h0};
            if (!cand.valid() || !rectContains(container, cand)) continue;
            bool clash = false;
            for (const auto& r : placed)
                if (Rect::overlap(cand, r)) { clash = true; break; }
            if (clash) continue;
            Rect grown = cand;
            if (!growRectInContainer(grown, container, placed, {}, item.demand)) continue;
            const int area = grown.w * grown.h;
            if (area < best_area) {
                best_area = area;
                best      = grown;
            }
        }
        if (!best.valid()) return false;
        out[item.comm_id] = best;
        placed.push_back(best);
    }

    rect_by_comm.swap(out);
    return true;
}

inline bool NestedDFXAllocator::expandSubparentToContainChildren(
    Rect& sub_rect,
    const Rect& container,
    const std::vector<Rect>& forbid,
    const std::vector<Rect>& children) const {
    for (int iter = 0; iter < 32; iter++) {
        bool changed = false;
        for (const auto& c : children) {
            if (rectPartialOverlap(sub_rect, c)) {
                const Rect bb = boundingBox(sub_rect, c);
                if (!rectContains(container, bb) || !bb.withinFPGA(fpga_)) return false;
                bool clash = false;
                for (const auto& f : forbid)
                    if (Rect::overlap(bb, f)) { clash = true; break; }
                if (clash) return false;
                sub_rect = bb;
                changed  = true;
            }
        }
        if (!changed) return true;
    }
    return false;
}

inline int NestedDFXAllocator::createParentRegion(const Rect& rect) {
    const int pid = next_parent_id_++;
    ParentRegion p;
    p.id   = pid;
    p.rect = rect;
    p.current_layout_version = next_version_++;
    p.version_history.push_back(p.current_layout_version);
    parents_[pid] = p;
    return pid;
}

inline int NestedDFXAllocator::createChildDRR(int parent_id, const Rect& rect) {
    const int cid = next_child_id_++;
    ChildDRR c;
    c.id             = cid;
    c.parent_id      = parent_id;
    c.layout_version = parents_[parent_id].current_layout_version;
    c.rect           = rect;
    child_drrs_[cid] = c;
    parents_[parent_id].child_drr_ids.push_back(cid);
    child_layers_[cid] = {};
    return cid;
}

inline int NestedDFXAllocator::createSubParent(int parent_id, const Rect& rect) {
    const int sid = next_sub_id_++;
    SubParentRegion s;
    s.id             = sid;
    s.parent_id      = parent_id;
    s.layout_version = parents_[parent_id].current_layout_version;
    s.rect           = rect;
    subparents_[sid] = s;
    parents_[parent_id].subparent_ids.push_back(sid);
    sub_layers_[sid] = {};
    recomputeSubContainments(parent_id);
    return sid;
}

inline void NestedDFXAllocator::recomputeSubContainments(int parent_id) {
    auto pit = parents_.find(parent_id);
    if (pit == parents_.end()) return;
    const auto& parent = pit->second;
    const int ver = parent.current_layout_version;
    for (int sid : parent.subparent_ids) {
        auto sit = subparents_.find(sid);
        if (sit == subparents_.end() || sit->second.layout_version != ver) continue;
        sit->second.covered_child_drrs.clear();
        for (int did : parent.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit == child_drrs_.end() || cit->second.layout_version != ver) continue;
            if (rectContains(sit->second.rect, cit->second.rect))
                sit->second.covered_child_drrs.push_back(did);
        }
    }
}

#include "cps/slot_semantics.hpp"

inline int NestedDFXAllocator::commitChildLayer(
    int child_drr_id, int comm_id, double t_now,
    const CommunityResource& demand, int mode_tag,
    double placement_cost, const Rect* occupied_rect) {
    auto& layers = child_layers_[child_drr_id];
    TimeLayer tl;
    tl.layer_idx = static_cast<int>(layers.size());
    tl.comm_id   = comm_id;
    tl.rect      = (occupied_rect && occupied_rect->valid())
                 ? *occupied_rect : child_drrs_[child_drr_id].rect;
    tl.t_on           = t_now;
    tl.t_off          = -1;
    tl.active         = true;
    tl.demand         = demand;
    tl.placement_cost = placement_cost;
    tl.mode_tag       = mode_tag;
    tl.layout_version = child_drrs_.at(child_drr_id).layout_version;
    tl.rect_changes.push_back({t_now, tl.rect});
    layers.push_back(tl);

    auto& drr = child_drrs_[child_drr_id];
    drr.active_comm_ids.insert(comm_id);
    drr.active_layer_by_comm[comm_id] = tl.layer_idx;
    if (drr.active_comm_id < 0) {
        drr.active_comm_id   = comm_id;
        drr.active_layer_idx = tl.layer_idx;
    }
    drr.max_hosted_demand = maxDemand(drr.max_hosted_demand, demand);

    ActivePlacement ap;
    ap.kind         = ActivePlacement::CHILD;
    ap.parent_id    = drr.parent_id;
    ap.container_id = child_drr_id;
    ap.layer_idx    = tl.layer_idx;
    active_[comm_id] = ap;
    if (cfg_.reseff_weight > 0) latest_resource_placement_[comm_id] = ap;
    return tl.layer_idx;
}

inline int NestedDFXAllocator::commitSubLayer(
    int sub_id, int comm_id, double t_now,
    const CommunityResource& demand, int mode_tag, double placement_cost) {
    const int pid = subparents_.at(sub_id).parent_id;
    reconcileSlotHierarchy(pid);
    const int slot  = ensureFullSlot(pid, sub_id);
    const int layer = commitChildLayer(slot, comm_id, t_now, demand, mode_tag, placement_cost, nullptr);
    auto& sub = subparents_.at(sub_id);
    sub.active_comm_id    = comm_id;
    sub.max_hosted_demand = maxDemand(sub.max_hosted_demand, demand);
    return layer;
}

inline int NestedDFXAllocator::commitParentMonoLayer(
    int parent_id, int comm_id, double t_now,
    const CommunityResource& demand, double placement_cost) {
    reconcileSlotHierarchy(parent_id);
    const int slot  = ensureFullSlot(parent_id, -1);
    const int layer = commitChildLayer(
        slot, comm_id, t_now, demand,
        static_cast<int>(PlacementMode::MODE_A_USE_PARENT), placement_cost, nullptr);
    auto& p = parents_.at(parent_id);
    p.mono_busy        = true;
    p.mono_active_comm = comm_id;
    return layer;
}

inline double NestedDFXAllocator::computeWasteCost(
    const Rect& rect, const CommunityResource& d) const {
    if (!rect.valid()) return 0;
    const RectCap cap = rectCapacity(fpga_, rect);
    auto term = [&](double cap_x, double req_x) {
        if (req_x <= NDFX_EPS) return 0.0;
        const double w = (cap_x - req_x) / std::max(cap_x, NDFX_EPS);
        return w < 0.0 ? 0.0 : w;
    };
    return term(cap.clb, d.clb_req) + term(cap.bram, d.bram_req) + term(cap.dsp, d.dsp_req);
}

inline void NestedDFXAllocator::tryActionChildReuse(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    for (const auto& [did, drr] : child_drrs_) {
        if (drr.full_cover || drr.busy()) continue;
        const RectCap cap = rectCapacity(fpga_, drr.rect);
        if (!capacitySatisfies(cap, demand)) continue;
        if (cfg_.reuse_min_util > 1e-9) {
            double u_min = std::numeric_limits<double>::infinity();
            auto upd = [&](double req, double c) {
                if (req <= NDFX_EPS) return;
                const double u = (c > NDFX_EPS) ? (req / c) : 1.0;
                if (u < u_min) u_min = u;
            };
            upd(demand.clb_req,  cap.clb);
            upd(demand.bram_req, cap.bram);
            upd(demand.dsp_req,  cap.dsp);
            if (std::isfinite(u_min) && u_min < cfg_.reuse_min_util) continue;
        }
        auto pit = parents_.find(drr.parent_id);
        if (pit == parents_.end()) continue;
        if (drr.layout_version != pit->second.current_layout_version) continue;
        if (pit->second.mono_busy) continue;
        bool blocked = false;
        for (int sid : pit->second.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit != subparents_.end() && sit->second.active() &&
                rectContains(sit->second.rect, drr.rect)) {
                blocked = true;
                break;
            }
        }
        if (blocked) continue;
        PlacementPlan plan;
        plan.mode         = PlacementMode::CHILD_REUSE;
        plan.parent_id    = drr.parent_id;
        plan.container_id = did;
        plan.rect         = drr.rect;
        considerPlan(comm_id, t_now, demand, plan, search);
    }
}

inline void NestedDFXAllocator::tryActionChildMergeActive(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    if (!cfg_.enable_active_child_merge) return;

    for (const auto& [did, drr] : child_drrs_) {
        if (drr.full_cover || !drr.busy() || drr.active_comm_ids.count(comm_id)) continue;
        auto pit = parents_.find(drr.parent_id);
        if (pit == parents_.end()) continue;
        const auto& parent = pit->second;
        if (drr.layout_version != parent.current_layout_version || parent.mono_busy) continue;

        CommunityResource total = demand;
        std::vector<MergePackItem> members;
        bool has_active_layer      = false;
        bool same_activation_event = true;
        double related_weight      = 0.0;
        for (int active_cid : drr.active_comm_ids) {
            auto lit = drr.active_layer_by_comm.find(active_cid);
            if (lit == drr.active_layer_by_comm.end()) continue;
            const int li = lit->second;
            if (li < 0 || li >= static_cast<int>(child_layers_.at(did).size())) continue;
            const TimeLayer& tl = child_layers_.at(did)[li];
            if (!tl.active) continue;
            if (std::abs(tl.t_on - t_now) > NDFX_EPS) same_activation_event = false;
            total = addDemand(total, tl.demand);
            members.push_back({active_cid, li, tl.t_on, tl.demand});
            if (weight_fn_) related_weight += weight_fn_(comm_id, active_cid);
            has_active_layer = true;
        }
        if (!has_active_layer || !same_activation_event || related_weight <= NDFX_EPS) continue;
        members.push_back({comm_id, -1, t_now, demand});

        std::vector<Rect> forbid, soft_sub;
        for (int odid : parent.child_drr_ids) {
            if (odid == did) continue;
            auto cit = child_drrs_.find(odid);
            if (cit != child_drrs_.end() && cit->second.layout_version == parent.current_layout_version)
                forbid.push_back(cit->second.rect);
        }
        for (int sid : parent.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit == subparents_.end() || sit->second.layout_version != parent.current_layout_version) continue;
            if (sit->second.active()) forbid.push_back(sit->second.rect);
            else                      soft_sub.push_back(sit->second.rect);
        }

        struct MergeCandidate {
            Rect rect;
            Rect parent_rect;
            std::vector<std::pair<int, Rect>> shift;
        };
        std::vector<MergeCandidate> merge_cands;

        Rect grown = drr.rect;
        if (growRectInContainer(grown, parent.rect, forbid, soft_sub, total))
            merge_cands.push_back({grown, parent.rect, {}});

        std::vector<Rect> forbid_world = forbid;
        for (const auto& [opid, op] : parents_)
            if (opid != drr.parent_id) forbid_world.push_back(op.rect);
        const Rect fpga_rect{0, 0, fpga_.W, fpga_.H};
        Rect grown_world = drr.rect;
        if (growRectInContainer(grown_world, fpga_rect, forbid_world, soft_sub, total)) {
            const Rect new_parent = boundingBox(parent.rect, grown_world);
            if (new_parent.withinFPGA(fpga_)) {
                std::vector<std::pair<int, Rect>> shift_plan;
                bool need_replay = false;
                for (const auto& [opid, op] : parents_) {
                    if (opid != drr.parent_id && Rect::overlap(new_parent, op.rect)) {
                        need_replay = true;
                        break;
                    }
                }
                if (!need_replay) {
                    merge_cands.push_back({grown_world, new_parent, {}});
                } else if (cfg_.enable_global_replay &&
                           tryReplayExpandParent(drr.parent_id, new_parent, shift_plan)) {
                    merge_cands.push_back({grown_world, new_parent, shift_plan});
                }
            }
        }

        for (const auto& mc : merge_cands) {
            std::map<int, Rect> packed;
            if (!packChildMergeGroup(mc.rect, members, packed)) continue;
            auto nr = packed.find(comm_id);
            if (nr == packed.end() || !nr->second.valid()) continue;

            PlacementPlan plan;
            plan.mode            = PlacementMode::CHILD_MERGE_ACTIVE;
            plan.parent_id       = drr.parent_id;
            plan.container_id    = did;
            plan.parent_rect     = mc.parent_rect;
            plan.rect            = mc.rect;
            plan.occupied_rect   = nr->second;
            plan.packed_rects    = packed;
            plan.shifted_parents = mc.shift;
            considerPlan(comm_id, t_now, demand, plan, search);
        }
    }
}

inline void NestedDFXAllocator::tryExistingContainerExpansion(
    int comm_id, double t_now, const CommunityResource& demand,
    PlacementMode mode, int parent_id, int container_id, const Rect& original,
    const CommunityResource& needed, const std::vector<Rect>& forbid,
    const std::vector<Rect>& soft_containment, PlacementSearch& search) const {
    const Rect& parent_rect = parents_.at(parent_id).rect;
    std::vector<Rect> seen;
    auto attempt = [&](const Rect& boundary, const std::vector<Rect>& obstacles) {
        Rect grown = original;
        if (!growRectInContainer(grown, boundary, obstacles, soft_containment, needed)) return;
        if (mode == PlacementMode::SUB_EXPAND &&
            !expandSubparentToContainChildren(grown, boundary, obstacles, soft_containment)) return;
        if (rectsEqual(grown, original)) return;
        for (const auto& previous : seen)
            if (rectsEqual(grown, previous)) return;
        seen.push_back(grown);

        const Rect new_parent = boundingBox(parent_rect, grown);
        if (!new_parent.withinFPGA(fpga_)) return;
        std::vector<std::pair<int, Rect>> shifts;
        for (const auto& [pid, parent] : parents_) {
            if (pid == parent_id || !Rect::overlap(new_parent, parent.rect)) continue;
            if (!cfg_.enable_global_replay ||
                !tryReplayExpandParent(parent_id, new_parent, shifts)) return;
            break;
        }

        PlacementPlan plan;
        plan.mode            = mode;
        plan.parent_id       = parent_id;
        plan.container_id    = container_id;
        plan.parent_rect     = new_parent;
        plan.rect            = grown;
        plan.shifted_parents = shifts;
        considerPlan(comm_id, t_now, demand, plan, search);
    };

    attempt(parent_rect, forbid);
    const Rect fpga_rect{0, 0, fpga_.W, fpga_.H};
    if (rectsEqual(parent_rect, fpga_rect)) return;
    std::vector<Rect> world_obstacles = forbid;
    for (const auto& [pid, parent] : parents_)
        if (pid != parent_id) world_obstacles.push_back(parent.rect);
    attempt(fpga_rect, world_obstacles);
    if (cfg_.enable_global_replay && parents_.size() > 1)
        attempt(fpga_rect, forbid);
}

inline void NestedDFXAllocator::tryActionChildExpand(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    for (const auto& [did, drr] : child_drrs_) {
        if (drr.full_cover || drr.busy()) continue;
        auto pit = parents_.find(drr.parent_id);
        if (pit == parents_.end()) continue;
        if (drr.layout_version != pit->second.current_layout_version || pit->second.mono_busy) continue;

        bool blocked_active = false;
        for (int sid : pit->second.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit != subparents_.end() && sit->second.active() &&
                rectContains(sit->second.rect, drr.rect)) {
                blocked_active = true;
                break;
            }
        }
        if (blocked_active) continue;

        std::vector<Rect> forbid, soft_sub;
        for (int sid : pit->second.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit == subparents_.end() || sit->second.layout_version != pit->second.current_layout_version) continue;
            if (sit->second.active()) forbid.push_back(sit->second.rect);
            else                      soft_sub.push_back(sit->second.rect);
        }
        for (int odid : pit->second.child_drr_ids) {
            if (odid == did) continue;
            auto cit = child_drrs_.find(odid);
            if (cit != child_drrs_.end() && cit->second.layout_version == pit->second.current_layout_version)
                forbid.push_back(cit->second.rect);
        }

        const CommunityResource req = maxDemand(demand, drr.max_hosted_demand);
        tryExistingContainerExpansion(
            comm_id, t_now, demand, PlacementMode::CHILD_EXPAND,
            drr.parent_id, did, drr.rect, req, forbid, soft_sub, search);
    }
}

inline void NestedDFXAllocator::tryActionChildNewInParent(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    for (const auto& [pid, parent] : parents_) {
        if (parent.mono_busy) continue;
        std::vector<Rect> forbid, soft_sub;
        for (int did : parent.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit != child_drrs_.end() && cit->second.layout_version == parent.current_layout_version)
                forbid.push_back(cit->second.rect);
        }
        for (int sid : parent.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit == subparents_.end() || sit->second.layout_version != parent.current_layout_version) continue;
            if (sit->second.active()) forbid.push_back(sit->second.rect);
            else                      soft_sub.push_back(sit->second.rect);
        }
        const Rect start{parent.rect.x, parent.rect.y,
                         std::min(cfg_.default_child_w, parent.rect.w),
                         std::min(cfg_.default_child_h, parent.rect.h)};
        if (!rectContains(parent.rect, start)) continue;

        std::vector<std::pair<int, int>> anchors;
        anchors.push_back({parent.rect.x, parent.rect.y});
        for (int did : parent.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit == child_drrs_.end()) continue;
            anchors.push_back({cit->second.rect.xr(), cit->second.rect.y});
            anchors.push_back({cit->second.rect.x,    cit->second.rect.yt()});
        }
        for (int sid : parent.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit == subparents_.end()) continue;
            anchors.push_back({sit->second.rect.xr(), sit->second.rect.y});
            anchors.push_back({sit->second.rect.x,    sit->second.rect.yt()});
        }
        for (auto [ax, ay] : anchors) {
            if (ax < parent.rect.x || ay < parent.rect.y) continue;
            const int w0 = std::min(cfg_.default_child_w, parent.rect.xr() - ax);
            const int h0 = std::min(cfg_.default_child_h, parent.rect.yt() - ay);
            if (w0 <= 0 || h0 <= 0) continue;
            const Rect cand{ax, ay, w0, h0};
            if (!rectContains(parent.rect, cand)) continue;
            bool clash = false;
            for (const auto& f : forbid) if (Rect::overlap(cand, f)) { clash = true; break; }
            if (clash) continue;
            Rect grown = cand;
            if (!growRectInContainer(grown, parent.rect, forbid, soft_sub, demand)) continue;
            PlacementPlan plan;
            plan.mode      = PlacementMode::CHILD_NEW_IN_PARENT;
            plan.parent_id = pid;
            plan.rect      = grown;
            considerPlan(comm_id, t_now, demand, plan, search);
        }
    }
}

inline void NestedDFXAllocator::tryActionChildExpandParent(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    for (const auto& [pid, parent] : parents_) {
        if (parent.mono_busy) continue;
        std::vector<Rect> forbid_world, soft_idle_sub;
        for (const auto& [opid, op] : parents_) if (opid != pid) forbid_world.push_back(op.rect);
        for (int did : parent.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit != child_drrs_.end() && cit->second.layout_version == parent.current_layout_version)
                forbid_world.push_back(cit->second.rect);
        }
        for (int sid : parent.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit == subparents_.end() || sit->second.layout_version != parent.current_layout_version) continue;
            if (sit->second.active()) forbid_world.push_back(sit->second.rect);
            else                      soft_idle_sub.push_back(sit->second.rect);
        }

        const std::vector<std::pair<int, int>> anchors = {
            {parent.rect.xr(), parent.rect.y},
            {parent.rect.x,    parent.rect.yt()}
        };
        for (auto [ax, ay] : anchors) {
            if (ax < 0 || ay < 0) continue;
            const int w0 = std::min(cfg_.default_child_w, fpga_.W - ax);
            const int h0 = std::min(cfg_.default_child_h, fpga_.H - ay);
            if (w0 <= 0 || h0 <= 0) continue;
            const Rect cstart{ax, ay, w0, h0};
            if (!cstart.withinFPGA(fpga_)) continue;
            bool clash = false;
            for (const auto& f : forbid_world) if (Rect::overlap(cstart, f)) { clash = true; break; }
            if (clash) continue;

            Rect grown_child = cstart;
            const Rect fpga_rect{0, 0, fpga_.W, fpga_.H};
            if (!growRectInContainer(grown_child, fpga_rect, forbid_world, soft_idle_sub, demand)) continue;

            const Rect new_parent = boundingBox(parent.rect, grown_child);
            if (!new_parent.withinFPGA(fpga_)) continue;

            std::vector<std::pair<int, Rect>> shift_plan;
            bool need_replay = false;
            for (const auto& [opid, op] : parents_) {
                if (opid != pid && Rect::overlap(new_parent, op.rect)) { need_replay = true; break; }
            }
            if (need_replay) {
                if (!cfg_.enable_global_replay) continue;
                if (!tryReplayExpandParent(pid, new_parent, shift_plan)) continue;
            }

            PlacementPlan plan;
            plan.mode            = PlacementMode::CHILD_EXPAND_PARENT;
            plan.parent_id       = pid;
            plan.parent_rect     = new_parent;
            plan.rect            = grown_child;
            plan.shifted_parents = shift_plan;
            considerPlan(comm_id, t_now, demand, plan, search);
        }
    }
}

inline void NestedDFXAllocator::tryActionChildNewParent(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    std::vector<Rect> forbid_world;
    for (const auto& [pid, p] : parents_) {
        (void)pid;
        forbid_world.push_back(p.rect);
    }
    const Rect fpga_rect{0, 0, fpga_.W, fpga_.H};
    std::vector<std::pair<int, int>> anchors = {{0, 0}};
    for (const auto& [pid, p] : parents_) {
        (void)pid;
        anchors.push_back({p.rect.xr(), p.rect.y});
        anchors.push_back({p.rect.x,    p.rect.yt()});
    }

    for (auto [ax, ay] : anchors) {
        if (ax < 0 || ay < 0 || ax >= fpga_.W || ay >= fpga_.H) continue;
        const int w0 = std::min(cfg_.default_child_w, fpga_.W - ax);
        const int h0 = std::min(cfg_.default_child_h, fpga_.H - ay);
        if (w0 <= 0 || h0 <= 0) continue;
        const Rect cstart{ax, ay, w0, h0};
        if (!cstart.withinFPGA(fpga_)) continue;
        bool clash = false;
        for (const auto& f : forbid_world) if (Rect::overlap(cstart, f)) { clash = true; break; }
        if (clash) continue;

        Rect grown_child = cstart;
        if (!growRectInContainer(grown_child, fpga_rect, forbid_world, {}, demand)) continue;
        const Rect new_parent = grown_child;
        bool clash2 = false;
        for (const auto& f : forbid_world) if (Rect::overlap(new_parent, f)) { clash2 = true; break; }
        if (clash2) continue;

        PlacementPlan plan;
        plan.mode        = PlacementMode::CHILD_NEW_PARENT;
        plan.parent_rect = new_parent;
        plan.rect        = grown_child;
        considerPlan(comm_id, t_now, demand, plan, search);
    }
}

inline void NestedDFXAllocator::tryActionSubReuse(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    for (const auto& [sid, sub] : subparents_) {
        if (sub.active()) continue;
        auto pit = parents_.find(sub.parent_id);
        if (pit == parents_.end() || pit->second.mono_busy) continue;
        if (sub.layout_version != pit->second.current_layout_version) continue;

        bool blocked = false;
        for (int did : sub.covered_child_drrs) {
            auto cit = child_drrs_.find(did);
            if (cit != child_drrs_.end() && cit->second.busy()) { blocked = true; break; }
        }
        if (blocked) continue;

        Rect rect = sub.rect;
        if (!capacitySatisfies(rectCapacity(fpga_, rect), demand)) {
            std::vector<Rect> forbid, soft_child;
            for (int osid : pit->second.subparent_ids) {
                if (osid == sid) continue;
                auto sit = subparents_.find(osid);
                if (sit != subparents_.end() && sit->second.layout_version == pit->second.current_layout_version)
                    forbid.push_back(sit->second.rect);
            }
            for (int did : pit->second.child_drr_ids) {
                auto cit = child_drrs_.find(did);
                if (cit == child_drrs_.end() || cit->second.layout_version != pit->second.current_layout_version) continue;
                if (cit->second.busy()) forbid.push_back(cit->second.rect);
                else                    soft_child.push_back(cit->second.rect);
            }
            Rect grown = rect;
            if (!growRectInContainer(grown, pit->second.rect, forbid, soft_child, demand)) continue;
            if (!expandSubparentToContainChildren(grown, pit->second.rect, forbid, soft_child)) continue;
            if (!capacitySatisfies(rectCapacity(fpga_, grown), demand)) continue;
            rect = grown;
        }
        if (cfg_.reuse_min_util > 1e-9) {
            const RectCap cap_check = rectCapacity(fpga_, rect);
            double u_min = std::numeric_limits<double>::infinity();
            auto upd = [&](double req, double c) {
                if (req <= NDFX_EPS) return;
                const double u = (c > NDFX_EPS) ? (req / c) : 1.0;
                if (u < u_min) u_min = u;
            };
            upd(demand.clb_req,  cap_check.clb);
            upd(demand.bram_req, cap_check.bram);
            upd(demand.dsp_req,  cap_check.dsp);
            if (std::isfinite(u_min) && u_min < cfg_.reuse_min_util) continue;
        }
        PlacementPlan plan;
        plan.mode         = PlacementMode::SUB_REUSE;
        plan.parent_id    = sub.parent_id;
        plan.container_id = sid;
        plan.rect         = rect;
        considerPlan(comm_id, t_now, demand, plan, search);
    }
}

inline void NestedDFXAllocator::tryActionSubExpand(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    for (const auto& [sid, sub] : subparents_) {
        if (sub.active()) continue;
        auto pit = parents_.find(sub.parent_id);
        if (pit == parents_.end() || pit->second.mono_busy) continue;
        if (sub.layout_version != pit->second.current_layout_version) continue;

        bool blocked = false;
        for (int did : sub.covered_child_drrs) {
            auto cit = child_drrs_.find(did);
            if (cit != child_drrs_.end() && cit->second.busy()) { blocked = true; break; }
        }
        if (blocked) continue;

        std::vector<Rect> forbid, soft_child;
        for (int osid : pit->second.subparent_ids) {
            if (osid == sid) continue;
            auto sit = subparents_.find(osid);
            if (sit != subparents_.end() && sit->second.layout_version == pit->second.current_layout_version)
                forbid.push_back(sit->second.rect);
        }
        for (int did : pit->second.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit == child_drrs_.end() || cit->second.layout_version != pit->second.current_layout_version) continue;
            if (cit->second.busy()) forbid.push_back(cit->second.rect);
            else                    soft_child.push_back(cit->second.rect);
        }
        const CommunityResource needed = maxDemand(demand, sub.max_hosted_demand);
        tryExistingContainerExpansion(
            comm_id, t_now, demand, PlacementMode::SUB_EXPAND,
            sub.parent_id, sid, sub.rect, needed, forbid, soft_child, search);
    }
}

inline void NestedDFXAllocator::tryActionSubNewInParent(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    for (const auto& [pid, parent] : parents_) {
        if (parent.mono_busy) continue;
        std::vector<Rect> forbid, soft_child;
        for (int sid : parent.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit != subparents_.end() && sit->second.layout_version == parent.current_layout_version)
                forbid.push_back(sit->second.rect);
        }
        for (int did : parent.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit == child_drrs_.end() || cit->second.layout_version != parent.current_layout_version) continue;
            if (cit->second.busy()) forbid.push_back(cit->second.rect);
            else                    soft_child.push_back(cit->second.rect);
        }

        std::vector<std::pair<int, int>> anchors = {{parent.rect.x, parent.rect.y}};
        for (int did : parent.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit == child_drrs_.end()) continue;
            anchors.push_back({cit->second.rect.xr(), cit->second.rect.y});
            anchors.push_back({cit->second.rect.x,    cit->second.rect.yt()});
        }
        for (int sid : parent.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit == subparents_.end()) continue;
            anchors.push_back({sit->second.rect.xr(), sit->second.rect.y});
            anchors.push_back({sit->second.rect.x,    sit->second.rect.yt()});
        }
        for (auto [ax, ay] : anchors) {
            if (ax < parent.rect.x || ay < parent.rect.y) continue;
            const int w0 = std::min(cfg_.default_sub_w, parent.rect.xr() - ax);
            const int h0 = std::min(cfg_.default_sub_h, parent.rect.yt() - ay);
            if (w0 <= 0 || h0 <= 0) continue;
            const Rect cstart{ax, ay, w0, h0};
            if (!rectContains(parent.rect, cstart)) continue;
            bool clash = false;
            for (const auto& f : forbid) if (Rect::overlap(cstart, f)) { clash = true; break; }
            if (clash) continue;

            Rect grown = cstart;
            if (!growRectInContainer(grown, parent.rect, forbid, soft_child, demand)) continue;
            if (!expandSubparentToContainChildren(grown, parent.rect, forbid, soft_child)) continue;
            if (!capacitySatisfies(rectCapacity(fpga_, grown), demand)) continue;

            PlacementPlan plan;
            plan.mode      = PlacementMode::SUB_NEW_IN_PARENT;
            plan.parent_id = pid;
            plan.rect      = grown;
            considerPlan(comm_id, t_now, demand, plan, search);
        }
    }
}

inline void NestedDFXAllocator::tryActionSubExpandParent(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    for (const auto& [pid, parent] : parents_) {
        if (parent.mono_busy) continue;
        std::vector<Rect> forbid_world, soft_child;
        for (const auto& [opid, op] : parents_) if (opid != pid) forbid_world.push_back(op.rect);
        for (int did : parent.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit == child_drrs_.end() || cit->second.layout_version != parent.current_layout_version) continue;
            if (cit->second.busy()) forbid_world.push_back(cit->second.rect);
            else                    soft_child.push_back(cit->second.rect);
        }
        for (int sid : parent.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit != subparents_.end() && sit->second.layout_version == parent.current_layout_version)
                forbid_world.push_back(sit->second.rect);
        }

        const std::vector<std::pair<int, int>> anchors = {
            {parent.rect.xr(), parent.rect.y},
            {parent.rect.x,    parent.rect.yt()}
        };
        for (auto [ax, ay] : anchors) {
            if (ax < 0 || ay < 0) continue;
            const int w0 = std::min(cfg_.default_sub_w, fpga_.W - ax);
            const int h0 = std::min(cfg_.default_sub_h, fpga_.H - ay);
            if (w0 <= 0 || h0 <= 0) continue;
            const Rect cstart{ax, ay, w0, h0};
            if (!cstart.withinFPGA(fpga_)) continue;
            bool clash = false;
            for (const auto& f : forbid_world) if (Rect::overlap(cstart, f)) { clash = true; break; }
            if (clash) continue;

            Rect grown_sub = cstart;
            const Rect fpga_rect{0, 0, fpga_.W, fpga_.H};
            if (!growRectInContainer(grown_sub, fpga_rect, forbid_world, soft_child, demand)) continue;
            if (!expandSubparentToContainChildren(grown_sub, fpga_rect, forbid_world, soft_child)) continue;

            const Rect new_parent = boundingBox(parent.rect, grown_sub);
            if (!new_parent.withinFPGA(fpga_)) continue;

            std::vector<std::pair<int, Rect>> shift_plan;
            bool need_replay = false;
            for (const auto& [opid, op] : parents_) {
                if (opid != pid && Rect::overlap(new_parent, op.rect)) { need_replay = true; break; }
            }
            if (need_replay) {
                if (!cfg_.enable_global_replay) continue;
                if (!tryReplayExpandParent(pid, new_parent, shift_plan)) continue;
            }

            PlacementPlan plan;
            plan.mode            = PlacementMode::SUB_EXPAND_PARENT;
            plan.parent_id       = pid;
            plan.parent_rect     = new_parent;
            plan.rect            = grown_sub;
            plan.shifted_parents = shift_plan;
            considerPlan(comm_id, t_now, demand, plan, search);
        }
    }
}

inline void NestedDFXAllocator::tryActionModeA(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    for (const auto& [pid, parent] : parents_) {
        if (!parentEntirelyIdle(pid)) continue;
        if (!capacitySatisfies(rectCapacity(fpga_, parent.rect), demand)) continue;
        PlacementPlan plan;
        plan.mode         = PlacementMode::MODE_A_USE_PARENT;
        plan.parent_id    = pid;
        plan.container_id = pid;
        plan.rect         = parent.rect;
        considerPlan(comm_id, t_now, demand, plan, search);
    }
}

inline void NestedDFXAllocator::tryActionSubNewParent(
    int comm_id, double t_now, const CommunityResource& demand, PlacementSearch& search) const {
    std::vector<Rect> forbid_world;
    for (const auto& [pid, p] : parents_) {
        (void)pid;
        forbid_world.push_back(p.rect);
    }
    const Rect fpga_rect{0, 0, fpga_.W, fpga_.H};
    std::vector<std::pair<int, int>> anchors = {{0, 0}};
    for (const auto& [pid, p] : parents_) {
        (void)pid;
        anchors.push_back({p.rect.xr(), p.rect.y});
        anchors.push_back({p.rect.x,    p.rect.yt()});
    }
    for (auto [ax, ay] : anchors) {
        if (ax < 0 || ay < 0 || ax >= fpga_.W || ay >= fpga_.H) continue;
        const int w0 = std::min(cfg_.default_sub_w, fpga_.W - ax);
        const int h0 = std::min(cfg_.default_sub_h, fpga_.H - ay);
        if (w0 <= 0 || h0 <= 0) continue;
        const Rect cstart{ax, ay, w0, h0};
        if (!cstart.withinFPGA(fpga_)) continue;
        bool clash = false;
        for (const auto& f : forbid_world) if (Rect::overlap(cstart, f)) { clash = true; break; }
        if (clash) continue;

        Rect grown = cstart;
        if (!growRectInContainer(grown, fpga_rect, forbid_world, {}, demand)) continue;
        const Rect new_parent = grown;
        bool clash2 = false;
        for (const auto& f : forbid_world) if (Rect::overlap(new_parent, f)) { clash2 = true; break; }
        if (clash2) continue;

        PlacementPlan plan;
        plan.mode        = PlacementMode::SUB_NEW_PARENT;
        plan.parent_rect = new_parent;
        plan.rect        = grown;
        considerPlan(comm_id, t_now, demand, plan, search);
    }
}

inline bool NestedDFXAllocator::tryReplayExpandParent(
    int parent_id, const Rect& target_rect,
    std::vector<std::pair<int, Rect>>& shifted_out) const {
    if (!cfg_.enable_global_replay || !target_rect.withinFPGA(fpga_)) return false;

    std::vector<int> blockers;
    for (const auto& [opid, op] : parents_) {
        if (opid != parent_id && Rect::overlap(target_rect, op.rect))
            blockers.push_back(opid);
    }
    if (static_cast<int>(blockers.size()) > cfg_.replay_max_shifts) return false;

    std::map<int, Rect> proposed;
    proposed[parent_id] = target_rect;
    for (int bid : blockers) {
        const Rect& br = parents_.at(bid).rect;
        const int dx_right = target_rect.xr() - br.x;
        const int dx_left  = br.xr() - target_rect.x;
        const int dy_up    = target_rect.yt() - br.y;
        const int dy_down  = br.yt() - target_rect.y;
        struct Try { int dx, dy; };
        std::vector<Try> tries;
        if (br.cx() >= target_rect.cx()) tries.push_back({dx_right, 0});
        else                             tries.push_back({-dx_left, 0});
        if (br.cy() >= target_rect.cy()) tries.push_back({0, dy_up});
        else                             tries.push_back({0, -dy_down});
        tries.push_back({dx_right, dy_up});
        tries.push_back({-dx_left, -dy_down});

        bool fixed = false;
        for (auto t : tries) {
            Rect new_br = br;
            new_br.x += t.dx;
            new_br.y += t.dy;
            if (!new_br.withinFPGA(fpga_) || Rect::overlap(new_br, target_rect)) continue;
            bool clash = false;
            for (const auto& [opid, op] : parents_) {
                if (opid == bid || opid == parent_id) continue;
                const Rect r = proposed.count(opid) ? proposed[opid] : op.rect;
                if (Rect::overlap(new_br, r)) { clash = true; break; }
            }
            if (clash) continue;
            proposed[bid] = new_br;
            fixed = true;
            break;
        }
        if (!fixed) return false;
    }
    if (!validateShiftPlan(proposed)) return false;

    shifted_out.clear();
    for (const auto& [opid, r] : proposed) {
        if (opid != parent_id) shifted_out.push_back({opid, r});
    }
    return true;
}

inline bool NestedDFXAllocator::validateShiftPlan(
    const std::map<int, Rect>& new_parent_rects, std::string* err) const {
    NestedDFXAllocator trial = *this;
    for (const auto& [pid, rect] : new_parent_rects) {
        const auto& old = parents_.at(pid).rect;
        if (old.w == rect.w && old.h == rect.h)
            trial.applyParentShift(pid, rect);
        else
            trial.parents_.at(pid).rect = rect;
    }
    return trial.validate(err);
}

inline bool NestedDFXAllocator::refreshParentLayoutVersion(int parent_id) {
    if (!cfg_.enable_refresh) return false;
    auto pit = parents_.find(parent_id);
    if (pit == parents_.end() || !parentEntirelyIdle(parent_id)) return false;
    const int new_ver = next_version_++;
    pit->second.current_layout_version = new_ver;
    pit->second.version_history.push_back(new_ver);
    pit->second.child_drr_ids.clear();
    pit->second.subparent_ids.clear();
    pit->second.split_configuration = -1;
    pit->second.full_slot           = -1;
    return true;
}

inline PlaceDecision NestedDFXAllocator::place(
    int comm_id, double t_now,
    const CommunityResource& demand,
    double exec_time_estimate) {
    PlacementSearch search;
    if (cfg_.reseff_weight > 0) {
        if (!std::isfinite(exec_time_estimate) || exec_time_estimate < 0)
            throw std::invalid_argument("Invalid resource duration");
        search.resource_box_before        = accruedResourceBox(t_now);
        search.resource_duration_estimate = exec_time_estimate;
    }
    const CommSizeClass cls = classifyCommunity(demand);
    if (active_.count(comm_id)) {
        PlaceDecision failed;
        failed.fail_cost = cfg_.huge_fail_cost;
        return failed;
    }

    auto small = [&]() {
        tryActionChildMergeActive(comm_id, t_now, demand, search);
        tryActionChildReuse(comm_id, t_now, demand, search);
        tryActionChildExpand(comm_id, t_now, demand, search);
        tryActionChildNewInParent(comm_id, t_now, demand, search);
        tryActionChildExpandParent(comm_id, t_now, demand, search);
        tryActionChildNewParent(comm_id, t_now, demand, search);
    };
    auto large = [&]() {
        tryActionSubReuse(comm_id, t_now, demand, search);
        tryActionSubExpand(comm_id, t_now, demand, search);
        tryActionSubNewInParent(comm_id, t_now, demand, search);
        tryActionSubExpandParent(comm_id, t_now, demand, search);
        tryActionModeA(comm_id, t_now, demand, search);
        tryActionSubNewParent(comm_id, t_now, demand, search);
    };
    auto refreshed = [&]() {
        if (!cfg_.enable_refresh) return;
        for (const auto& [pid, parent] : parents_) {
            if (!parentEntirelyIdle(pid)) continue;
            if (parent.child_drr_ids.empty() && parent.subparent_ids.empty()) continue;
            const bool fragmented = diagnoseParentFragmentation(pid, demand);
            NestedDFXAllocator trial = *this;
            if (!trial.refreshParentLayoutVersion(pid)) continue;
            search.refreshed_parent_id         = pid;
            search.refreshed_parent_fragmented = fragmented;
            if (cls == CommSizeClass::HARD_LARGE) {
                trial.tryActionSubNewInParent(comm_id, t_now, demand, search);
                trial.tryActionSubExpandParent(comm_id, t_now, demand, search);
                trial.tryActionChildNewInParent(comm_id, t_now, demand, search);
                trial.tryActionChildExpandParent(comm_id, t_now, demand, search);
            } else {
                trial.tryActionChildNewInParent(comm_id, t_now, demand, search);
                trial.tryActionChildExpandParent(comm_id, t_now, demand, search);
                trial.tryActionSubNewInParent(comm_id, t_now, demand, search);
                trial.tryActionSubExpandParent(comm_id, t_now, demand, search);
            }
        }
        search.refreshed_parent_id         = -1;
        search.refreshed_parent_fragmented = false;
    };

    if (cfg_.aggressive_refresh_on_idle) refreshed();
    if (cls == CommSizeClass::HARD_LARGE) { large(); small(); }
    else                                  { small(); large(); }
    if (!cfg_.aggressive_refresh_on_idle) refreshed();

    if (search.state) {
        if (cfg_.collect_debug_lines) {
            search.best.debug_lines.push_back(
                "UNIFIED_SELECT generated=" + std::to_string(search.generated) +
                " feasible=" + std::to_string(search.feasible) +
                " action_cost=" + std::to_string(search.best.action_cost));
        }
        *this = std::move(*search.state);
        return search.best;
    }
    PlaceDecision failed;
    failed.fail_cost = cfg_.huge_fail_cost;
    dbg(failed, "ALL ACTIONS FAILED");
    return failed;
}

inline void NestedDFXAllocator::release(int comm_id, double t_off) {
    auto it = active_.find(comm_id);
    if (it == active_.end()) return;
    const auto ap = it->second;
    active_.erase(it);
    if (ap.kind == ActivePlacement::CHILD) {
        auto& drr = child_drrs_[ap.container_id];
        drr.active_comm_ids.erase(comm_id);
        drr.active_layer_by_comm.erase(comm_id);
        if (drr.active_comm_id == comm_id || drr.active_comm_id < 0) {
            drr.active_comm_id   = -1;
            drr.active_layer_idx = -1;
            if (!drr.active_comm_ids.empty()) {
                const int next_cid = *drr.active_comm_ids.begin();
                drr.active_comm_id = next_cid;
                auto lit = drr.active_layer_by_comm.find(next_cid);
                if (lit != drr.active_layer_by_comm.end())
                    drr.active_layer_idx = lit->second;
            }
        }
        if (ap.layer_idx >= 0 && ap.layer_idx < static_cast<int>(child_layers_[ap.container_id].size())) {
            auto& tl  = child_layers_[ap.container_id][ap.layer_idx];
            tl.t_off  = t_off;
            tl.active = false;
        }
    } else if (ap.kind == ActivePlacement::SUBPARENT) {
        auto& sub = subparents_[ap.container_id];
        if (sub.active_comm_id == comm_id) {
            sub.active_comm_id   = -1;
            sub.active_layer_idx = -1;
        }
        if (ap.layer_idx >= 0 && ap.layer_idx < static_cast<int>(sub_layers_[ap.container_id].size())) {
            auto& tl  = sub_layers_[ap.container_id][ap.layer_idx];
            tl.t_off  = t_off;
            tl.active = false;
        }
    } else {
        auto& parent = parents_[ap.container_id];
        if (parent.mono_active_comm == comm_id) {
            parent.mono_busy             = false;
            parent.mono_active_comm      = -1;
            parent.mono_active_layer_idx = -1;
        }
        if (ap.layer_idx >= 0 && ap.layer_idx < static_cast<int>(parent_mono_layers_[ap.container_id].size())) {
            auto& tl  = parent_mono_layers_[ap.container_id][ap.layer_idx];
            tl.t_off  = t_off;
            tl.active = false;
        }
    }
    if (ap.kind == ActivePlacement::CHILD) {
        const auto& slot = child_drrs_.at(ap.container_id);
        if (!slot.busy() && slot.full_cover) {
            if (slot.owner_sub >= 0) {
                subparents_.at(slot.owner_sub).active_comm_id   = -1;
                subparents_.at(slot.owner_sub).active_layer_idx = -1;
            } else if (slot.parent_full) {
                parents_.at(slot.parent_id).mono_busy        = false;
                parents_.at(slot.parent_id).mono_active_comm = -1;
            }
        }
    }
}

inline bool NestedDFXAllocator::validate(std::string* err) const {
    auto fail = [&](const std::string& s) {
        if (err) *err = s;
        return false;
    };

    if (!validateSlots(err)) return false;

    std::vector<std::pair<int, Rect>> plist;
    for (const auto& [pid, p] : parents_) {
        if (!p.rect.valid())     return fail("Parent " + std::to_string(pid) + " invalid rect");
        if (!withinFPGA(p.rect)) return fail("Parent " + std::to_string(pid) + " out of FPGA");
        plist.push_back({pid, p.rect});
    }
    for (size_t i = 0; i < plist.size(); i++) {
        for (size_t j = i + 1; j < plist.size(); j++) {
            if (Rect::overlap(plist[i].second, plist[j].second))
                return fail("Parent " + std::to_string(plist[i].first) + " overlaps with " +
                            std::to_string(plist[j].first));
        }
    }

    for (const auto& [did, drr] : child_drrs_) {
        auto pit = parents_.find(drr.parent_id);
        if (pit == parents_.end()) return fail("ChildDRR " + std::to_string(did) + " orphan");
        if (!rectContains(pit->second.rect, drr.rect))
            return fail("ChildDRR " + std::to_string(did) + " not contained in Parent " +
                        std::to_string(drr.parent_id));
    }
    for (const auto& [sid, sub] : subparents_) {
        auto pit = parents_.find(sub.parent_id);
        if (pit == parents_.end()) return fail("SubParent " + std::to_string(sid) + " orphan");
        if (!rectContains(pit->second.rect, sub.rect))
            return fail("SubParent " + std::to_string(sid) + " not contained in Parent " +
                        std::to_string(sub.parent_id));
    }

    for (const auto& [pid, p] : parents_) {
        (void)pid;
        std::vector<int> subs;
        for (int sid : p.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit != subparents_.end() && sit->second.layout_version == p.current_layout_version)
                subs.push_back(sid);
        }
        for (size_t i = 0; i < subs.size(); i++) {
            for (size_t j = i + 1; j < subs.size(); j++) {
                if (Rect::overlap(subparents_.at(subs[i]).rect, subparents_.at(subs[j]).rect))
                    return fail("SubParents " + std::to_string(subs[i]) + " and " +
                                std::to_string(subs[j]) + " overlap (same layout_version)");
            }
        }
    }

    for (const auto& [pid, p] : parents_) {
        if (!p.mono_busy) continue;
        for (int did : p.child_drr_ids) {
            auto cit = child_drrs_.find(did);
            if (cit != child_drrs_.end() && cit->second.busy())
                return fail("Parent " + std::to_string(pid) + " mono_busy but ChildDRR " +
                            std::to_string(did) + " busy");
        }
        for (int sid : p.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit != subparents_.end() && sit->second.active())
                return fail("Parent " + std::to_string(pid) + " mono_busy but SubParent " +
                            std::to_string(sid) + " active");
        }
    }

    for (const auto& [sid, sub] : subparents_) {
        if (!sub.active()) continue;
        for (int did : sub.covered_child_drrs) {
            auto cit = child_drrs_.find(did);
            if (cit != child_drrs_.end() && cit->second.busy())
                return fail("SubParent " + std::to_string(sid) + " active but covered ChildDRR " +
                            std::to_string(did) + " busy");
        }
    }

    for (const auto& [pid, p] : parents_) {
        (void)pid;
        for (int sid : p.subparent_ids) {
            auto sit = subparents_.find(sid);
            if (sit == subparents_.end() || sit->second.layout_version != p.current_layout_version) continue;
            for (int did : p.child_drr_ids) {
                auto cit = child_drrs_.find(did);
                if (cit == child_drrs_.end() || cit->second.layout_version != p.current_layout_version) continue;
                if (Rect::overlap(sit->second.rect, cit->second.rect) &&
                    !rectContains(sit->second.rect, cit->second.rect))
                    return fail("SubParent " + std::to_string(sid) +
                                " reverse/partial overlap with ChildDRR " + std::to_string(did));
            }
        }
    }

    struct ResidenceRef {
        const TimeLayer* layer;
        const char* kind;
        int container_id;
    };
    std::vector<ResidenceRef> all_layers;
    auto check_layers = [&](const std::vector<TimeLayer>& layers, const Rect& container,
                            const char* kind, int container_id) {
        for (const auto& tl : layers) {
            if (!rectContains(container, tl.rect) ||
                !capacitySatisfies(rectCapacity(fpga_, tl.rect), tl.demand))
                return false;
            if (!std::isfinite(tl.t_on) || !std::isfinite(tl.t_off)) return false;
            if (tl.t_off >= 0 && tl.t_off < tl.t_on - NDFX_EPS) return false;
            all_layers.push_back({&tl, kind, container_id});
        }
        return true;
    };
    for (const auto& [did, layers] : child_layers_)
        if (!child_drrs_.count(did) || !check_layers(layers, child_drrs_.at(did).rect, "ChildDRR", did))
            return fail("invalid ChildDRR residence geometry/capacity");
    for (const auto& [sid, layers] : sub_layers_)
        if (!subparents_.count(sid) || !check_layers(layers, subparents_.at(sid).rect, "SubParent", sid))
            return fail("invalid SubParent residence geometry/capacity");
    for (const auto& [pid, layers] : parent_mono_layers_)
        if (!parents_.count(pid) || !check_layers(layers, parents_.at(pid).rect, "ParentMono", pid))
            return fail("invalid ParentMono residence geometry/capacity");

    auto off = [](const TimeLayer& tl) {
        return (tl.active || tl.t_off < 0) ? std::numeric_limits<double>::infinity() : tl.t_off;
    };
    for (size_t i = 0; i < all_layers.size(); ++i) {
        for (size_t j = i + 1; j < all_layers.size(); ++j) {
            const auto& a = *all_layers[i].layer;
            const auto& b = *all_layers[j].layer;
            const double overlap_on  = std::max(a.t_on, b.t_on);
            const double overlap_off = std::min(off(a), off(b));
            if (overlap_on < overlap_off - NDFX_EPS &&
                std::string(all_layers[i].kind) == "ChildDRR" &&
                std::string(all_layers[j].kind) == "ChildDRR" &&
                all_layers[i].container_id == all_layers[j].container_id) {
                const bool both_open = std::isinf(off(a)) && std::isinf(off(b));
                const bool both_closed_equal =
                    std::isfinite(off(a)) && std::isfinite(off(b)) &&
                    std::abs(off(a) - off(b)) <= NDFX_EPS;
                if (std::abs(a.t_on - b.t_on) > NDFX_EPS || !(both_open || both_closed_equal))
                    return fail("shared ChildDRR cohort must load and unload together");
            }
            if (overlap_on < overlap_off - NDFX_EPS && Rect::overlap(a.rect, b.rect)) {
                if (err) {
                    std::ostringstream message;
                    auto describe = [&](const ResidenceRef& ref) {
                        const auto& tl = *ref.layer;
                        message << "C" << tl.comm_id << " " << ref.kind << "#" << ref.container_id
                                << " layer=" << tl.layer_idx << " [" << tl.t_on << ",";
                        if (std::isinf(off(tl))) message << "open";
                        else message << off(tl);
                        message << ") rect=(" << tl.rect.x << "," << tl.rect.y << ","
                                << tl.rect.w << "," << tl.rect.h << ")";
                    };
                    message << "time-overlapping residences overlap in space: ";
                    describe(all_layers[i]);
                    message << " vs ";
                    describe(all_layers[j]);
                    *err = message.str();
                }
                return false;
            }
        }
    }

    for (const auto& [cid, ap] : active_) {
        if (ap.kind == ActivePlacement::CHILD) {
            auto cit = child_drrs_.find(ap.container_id);
            if (cit == child_drrs_.end()) return fail("active map references missing ChildDRR");
            if (!cit->second.active_comm_ids.count(cid))
                return fail("ChildDRR " + std::to_string(ap.container_id) +
                            " active mismatch comm " + std::to_string(cid));
            auto lit = cit->second.active_layer_by_comm.find(cid);
            if (lit == cit->second.active_layer_by_comm.end() || lit->second != ap.layer_idx)
                return fail("ChildDRR " + std::to_string(ap.container_id) +
                            " active layer mismatch comm " + std::to_string(cid));
            if (ap.layer_idx < 0 ||
                ap.layer_idx >= static_cast<int>(child_layers_.at(ap.container_id).size()) ||
                !child_layers_.at(ap.container_id)[ap.layer_idx].active)
                return fail("ChildDRR " + std::to_string(ap.container_id) +
                            " inactive active-map layer comm " + std::to_string(cid));
        } else if (ap.kind == ActivePlacement::SUBPARENT) {
            auto sit = subparents_.find(ap.container_id);
            if (sit == subparents_.end()) return fail("active map references missing SubParent");
            if (sit->second.active_comm_id != cid)
                return fail("SubParent " + std::to_string(ap.container_id) +
                            " active mismatch comm " + std::to_string(cid));
        } else {
            auto pit = parents_.find(ap.container_id);
            if (pit == parents_.end()) return fail("active map references missing Parent");
            if (!pit->second.mono_busy || pit->second.mono_active_comm != cid)
                return fail("Parent " + std::to_string(ap.container_id) +
                            " mono_busy mismatch comm " + std::to_string(cid));
        }
    }

    for (const auto& [did, drr] : child_drrs_) {
        if (!capacitySatisfies(rectCapacity(fpga_, drr.rect), drr.max_hosted_demand))
            return fail("ChildDRR " + std::to_string(did) +
                        " current rect cannot satisfy historical max demand");
    }
    for (const auto& [sid, sub] : subparents_) {
        if (!capacitySatisfies(rectCapacity(fpga_, sub.rect), sub.max_hosted_demand))
            return fail("SubParent " + std::to_string(sid) +
                        " current rect cannot satisfy historical max demand");
    }
    return true;
}

inline std::vector<int> NestedDFXAllocator::activeCommsInSameChildDrr(int comm_id) const {
    std::vector<int> out;
    auto it = active_.find(comm_id);
    if (it == active_.end()) return out;
    const auto& ap = it->second;
    if (ap.kind != ActivePlacement::CHILD) {
        out.push_back(comm_id);
        return out;
    }
    auto cit = child_drrs_.find(ap.container_id);
    if (cit == child_drrs_.end()) {
        out.push_back(comm_id);
        return out;
    }
    out.assign(cit->second.active_comm_ids.begin(),
               cit->second.active_comm_ids.end());
    if (out.empty()) out.push_back(comm_id);
    return out;
}

inline CommDrrPlacement NestedDFXAllocator::findCommDrrPlacement(
    int comm_id, double activation_time) const {
    CommDrrPlacement best;
    double best_score = -1.0;

    auto consider = [&](ActivePlacement::Kind kind, int container_id, int parent_id,
                        const TimeLayer& tl) {
        if (tl.comm_id != comm_id) return;
        CommDrrPlacement cand;
        cand.kind          = kind;
        cand.parent_id     = parent_id;
        cand.container_id  = container_id;
        cand.drr_rect      = tl.rect;
        cand.occupied_rect = tl.rect;
        cand.t_on          = tl.t_on;
        cand.t_off         = tl.t_off;
        cand.layer_idx     = tl.layer_idx;
        cand.mode_tag      = tl.mode_tag;
        cand.valid         = tl.rect.valid();

        double score = 0.0;
        if (activation_time >= 0.0) {
            score = -std::abs(tl.t_on - activation_time);
        } else {
            const double off = (tl.t_off >= 0.0 ? tl.t_off : tl.t_on);
            score = std::max(0.0, off - tl.t_on);
        }
        if (!best.valid || score > best_score) {
            best       = cand;
            best_score = score;
        }
    };

    for (const auto& [did, vec] : child_layers_) {
        int pid = -1;
        auto cit = child_drrs_.find(did);
        if (cit != child_drrs_.end()) pid = cit->second.parent_id;
        for (const auto& tl : vec) consider(ActivePlacement::CHILD, did, pid, tl);
    }
    for (const auto& [sid, vec] : sub_layers_) {
        int pid = -1;
        auto sit = subparents_.find(sid);
        if (sit != subparents_.end()) pid = sit->second.parent_id;
        for (const auto& tl : vec) consider(ActivePlacement::SUBPARENT, sid, pid, tl);
    }
    for (const auto& [pid, vec] : parent_mono_layers_) {
        for (const auto& tl : vec) consider(ActivePlacement::PARENT_MONO, pid, pid, tl);
    }

    if (best.valid) {
        switch (best.kind) {
            case ActivePlacement::CHILD: {
                auto it = child_drrs_.find(best.container_id);
                if (it != child_drrs_.end()) best.drr_rect = it->second.rect;
                break;
            }
            case ActivePlacement::SUBPARENT: {
                auto it = subparents_.find(best.container_id);
                if (it != subparents_.end()) best.drr_rect = it->second.rect;
                break;
            }
            case ActivePlacement::PARENT_MONO: {
                auto it = parents_.find(best.container_id);
                if (it != parents_.end()) best.drr_rect = it->second.rect;
                break;
            }
        }
    }
    return best;
}

inline double effOffLayerSnap(const CommCostLayerSnap& p) {
    if (p.still_active) return std::numeric_limits<double>::infinity();
    if (p.t_off < 0)    return 0.0;
    return p.t_off;
}

inline bool samePhysicalDrrLayer(const CommCostLayerSnap& a,
                                 const CommCostLayerSnap& b) {
    if (a.parent_id != b.parent_id || a.layout_version != b.layout_version) return false;
    if (a.kind == b.kind) return a.container_id == b.container_id;
    return false;
}

inline int dmcRelationCost(bool overlap, bool same_drr, bool same_parent,
                           bool same_version = true) {
    if (same_parent && !same_version) return 2;
    if (overlap) return same_drr ? 0 : 1;
    if (same_drr) return 1;
    if (same_parent) return 2;
    return 3;
}

#ifndef FULL_COVER_DMC_COEFFICIENT
#define FULL_COVER_DMC_COEFFICIENT 1
#endif
inline constexpr int FINAL_FULL_COVER_DMC_COST = FULL_COVER_DMC_COEFFICIENT;
static_assert(FINAL_FULL_COVER_DMC_COST == 1 || FINAL_FULL_COVER_DMC_COST == 2,
              "Full-cover ICC coefficient must be 1 or 2");

inline bool finalFullCoverReuse(const CommCostLayerSnap& a, const CommCostLayerSnap& b) {
    const bool overlap = a.t_on < effOffLayerSnap(b) - NDFX_EPS &&
                         b.t_on < effOffLayerSnap(a) - NDFX_EPS;
    if (overlap || a.kind != ActivePlacement::CHILD || b.kind != ActivePlacement::CHILD ||
        a.parent_id < 0 || a.parent_id != b.parent_id || a.layout_version < 0 ||
        a.layout_version != b.layout_version || a.owner_sub < 0 || a.owner_sub != b.owner_sub ||
        a.parent_configuration < 0 || a.parent_configuration != b.parent_configuration ||
        a.configuration_id < 0 || b.configuration_id < 0 ||
        a.configuration_id == b.configuration_id || a.container_id == b.container_id)
        return false;
    auto covers = [](const CommCostLayerSnap& whole, const CommCostLayerSnap& other) {
        return whole.verified_full_cover && whole.final_rect.valid() && other.final_rect.valid() &&
               whole.owner_sub_final_rect.valid() && other.owner_sub_final_rect.valid() &&
               rectsEqual(whole.final_rect, whole.owner_sub_final_rect) &&
               rectsEqual(whole.owner_sub_final_rect, other.owner_sub_final_rect) &&
               rectContains(whole.final_rect, other.final_rect);
    };
    return covers(a, b) || covers(b, a);
}

inline int dmcRelationForLayers(const CommCostLayerSnap& a, const CommCostLayerSnap& b) {
    const bool overlap = a.t_on < effOffLayerSnap(b) - NDFX_EPS &&
                         b.t_on < effOffLayerSnap(a) - NDFX_EPS;
    if (finalFullCoverReuse(a, b)) return FINAL_FULL_COVER_DMC_COST;
    return dmcRelationCost(overlap, samePhysicalDrrLayer(a, b),
                           a.parent_id >= 0 && a.parent_id == b.parent_id,
                           a.layout_version == b.layout_version);
}

inline const char* dmcRelationName(bool overlap, bool same_drr, bool same_parent,
                                   bool same_version = true) {
    if (same_parent && !same_version) return "same_parent_cross_version";
    if (overlap) return same_drr ? "same_tl" : "same_tl_diff_drr";
    if (same_drr) return "same_drr";
    if (same_parent) return "same_parent_same_version_diff_drr";
    return "diff_parent";
}

inline void NestedDFXAllocator::setTaskCommunicationEdges(std::vector<TaskCommunicationEdge> edges) {
    std::set<int> ids;
    double total = 0;
    for (const auto& edge : edges) {
        if (edge.id < 0 || edge.source < 0 || edge.target < 0 ||
            !ids.insert(edge.id).second || !std::isfinite(edge.weight) || edge.weight < 0)
            throw std::invalid_argument("Invalid task communication edge");
        total += edge.weight;
    }
    if (!std::isfinite(3.0 * total))
        throw std::invalid_argument("Task edge communication weight overflow");
    task_edges_ = std::move(edges);
    cfg_.dmc_total_weight = total;
    task_bindings_.clear();
    pending_communication_tasks_.clear();
}

inline void NestedDFXAllocator::bindTaskToActiveResidence(int task_id, int comm_id) {
    if (task_bindings_.count(task_id))
        throw std::logic_error("A task cannot be bound to more than one execution residence");
    const auto& ap = active_.at(comm_id);
    const int version = ap.kind == ActivePlacement::CHILD ? child_drrs_.at(ap.container_id).layout_version :
                        ap.kind == ActivePlacement::SUBPARENT ? subparents_.at(ap.container_id).layout_version :
                        parents_.at(ap.parent_id).current_layout_version;
    task_bindings_.emplace(task_id, TaskResidenceBinding{ap, version});
}

inline Rect NestedDFXAllocator::dmcContainerRect(ActivePlacement::Kind kind, int container_id) const {
    if (kind == ActivePlacement::CHILD) return child_drrs_.at(container_id).rect;
    if (kind == ActivePlacement::SUBPARENT) return subparents_.at(container_id).rect;
    return parents_.at(container_id).rect;
}

inline void NestedDFXAllocator::fillDmcSlotContext(CommCostLayerSnap& s) const {
    s.owner_sub             = -1;
    s.configuration_id      = -1;
    s.parent_configuration  = -1;
    s.verified_full_cover   = false;
    s.owner_sub_final_rect  = {};
    if (s.kind != ActivePlacement::CHILD) return;
    const auto& slot = child_drrs_.at(s.container_id);
    s.final_rect = slot.rect;
    auto ci = configurations_.find(slot.configuration_id);
    if (slot.owner_sub < 0 || ci == configurations_.end()) return;
    auto si = subparents_.find(slot.owner_sub);
    if (si == subparents_.end()) return;
    const auto& c   = ci->second;
    const auto& sub = si->second;
    if (slot.parent_id != s.parent_id || slot.layout_version != s.layout_version ||
        sub.parent_id != slot.parent_id || sub.layout_version != slot.layout_version ||
        c.owner_sub != slot.owner_sub || c.parent_id != slot.parent_id ||
        c.layout_version != slot.layout_version)
        return;
    auto pc = configurations_.find(c.parent_configuration);
    if (pc == configurations_.end() || pc->second.owner_sub >= 0 || pc->second.whole ||
        pc->second.parent_id != slot.parent_id || pc->second.layout_version != slot.layout_version)
        return;
    s.owner_sub            = slot.owner_sub;
    s.configuration_id     = c.id;
    s.parent_configuration = c.parent_configuration;
    s.owner_sub_final_rect = sub.rect;
    s.verified_full_cover  = slot.full_cover && !slot.parent_full && !slot.retained && c.whole &&
                             slot.rect.valid() && sub.rect.valid() && rectsEqual(slot.rect, sub.rect);
}

inline CommCostLayerSnap NestedDFXAllocator::taskResidenceSnapshot(const TaskResidenceBinding& b) const {
    const auto& ap = b.placement;
    const auto& map = ap.kind == ActivePlacement::CHILD ? child_layers_ :
                      ap.kind == ActivePlacement::SUBPARENT ? sub_layers_ : parent_mono_layers_;
    const auto& tl = map.at(ap.container_id).at(ap.layer_idx);
    CommCostLayerSnap s;
    s.comm_id        = tl.comm_id;
    s.rect           = dmcContainerRect(ap.kind, ap.container_id);
    s.t_on           = tl.t_on;
    s.t_off          = tl.t_off;
    s.still_active   = tl.active;
    s.kind           = ap.kind;
    s.parent_id      = ap.parent_id;
    s.container_id   = ap.container_id;
    s.layer_idx      = ap.layer_idx;
    s.layout_version = b.layout_version;
    s.final_rect     = dmcContainerRect(ap.kind, ap.container_id);
    fillDmcSlotContext(s);
    return s;
}

inline void NestedDFXAllocator::requireCompleteTaskCommunication() const {
    for (const auto& e : task_edges_)
        if (!task_bindings_.count(e.source) || !task_bindings_.count(e.target))
            throw std::logic_error("Completed decode is missing a task-edge endpoint residence");
}

inline double NestedDFXAllocator::estimateDmcIncrement(
    int comm_id, double t_now,
    ActivePlacement::Kind kind, int parent_id, int container_id,
    double* resolved_weight_out) const {
    CommCostLayerSnap added;
    added.comm_id        = comm_id;
    added.t_on           = t_now;
    added.still_active   = true;
    added.kind           = kind;
    added.parent_id      = parent_id;
    added.container_id   = container_id;
    added.final_rect     = dmcContainerRect(kind, container_id);
    added.layout_version = kind == ActivePlacement::CHILD ? child_drrs_.at(container_id).layout_version :
                           kind == ActivePlacement::SUBPARENT ? subparents_.at(container_id).layout_version :
                           parents_.at(parent_id).current_layout_version;
    fillDmcSlotContext(added);

    double total = 0;
    double resolved_weight = 0;
    for (const auto& e : task_edges_) {
        const auto a = task_bindings_.find(e.source);
        const auto b = task_bindings_.find(e.target);
        const bool pending_a = a == task_bindings_.end() && pending_communication_tasks_.count(e.source);
        const bool pending_b = b == task_bindings_.end() && pending_communication_tasks_.count(e.target);
        if (!pending_a && !pending_b) continue;
        if ((!pending_a && a == task_bindings_.end()) || (!pending_b && b == task_bindings_.end())) continue;
        const auto sa = pending_a ? added : taskResidenceSnapshot(a->second);
        const auto sb = pending_b ? added : taskResidenceSnapshot(b->second);
        total += e.weight * dmcRelationForLayers(sa, sb);
        resolved_weight += e.weight;
    }
    if (weight_fn_) {
        for (const auto& [peer_cid, ap] : active_) {
            if (peer_cid == comm_id) continue;
            const auto& map = ap.kind == ActivePlacement::CHILD ? child_layers_ :
                              ap.kind == ActivePlacement::SUBPARENT ? sub_layers_ : parent_mono_layers_;
            const auto& tl = map.at(ap.container_id).at(ap.layer_idx);
            if (std::abs(tl.t_on - t_now) > NDFX_EPS) continue;
            const double w_peer = weight_fn_(comm_id, peer_cid);
            if (w_peer <= NDFX_EPS) continue;
            const int version = ap.kind == ActivePlacement::CHILD ? child_drrs_.at(ap.container_id).layout_version :
                                ap.kind == ActivePlacement::SUBPARENT ? subparents_.at(ap.container_id).layout_version :
                                parents_.at(ap.parent_id).current_layout_version;
            const auto s_peer = taskResidenceSnapshot(TaskResidenceBinding{ap, version});
            total += w_peer * dmcRelationForLayers(added, s_peer);
            resolved_weight += w_peer;
        }
    }
    if (resolved_weight_out) *resolved_weight_out = resolved_weight;
    return total;
}

inline double NestedDFXAllocator::recomputeTotalCommCost(const layout::SearchParams& params) const {
    (void)params;
    double total = 0;
    for (const auto& e : task_edges_) {
        const auto a = task_bindings_.find(e.source);
        const auto b = task_bindings_.find(e.target);
        if (a == task_bindings_.end() || b == task_bindings_.end()) continue;
        total += e.weight * dmcRelationForLayers(
            taskResidenceSnapshot(a->second), taskResidenceSnapshot(b->second));
    }
    return total;
}

inline std::pair<double, std::vector<CommCostPair>>
NestedDFXAllocator::recomputeCommCostDetail(const layout::SearchParams& params) const {
    std::vector<CommCostPair> out;
    out.reserve(task_edges_.size());
    double total = 0;
    for (const auto& e : task_edges_) {
        const auto ai = task_bindings_.find(e.source);
        const auto bi = task_bindings_.find(e.target);
        if (ai == task_bindings_.end() || bi == task_bindings_.end()) continue;
        const auto a = taskResidenceSnapshot(ai->second);
        const auto b = taskResidenceSnapshot(bi->second);
        const bool overlap  = a.t_on < effOffLayerSnap(b) - NDFX_EPS &&
                              b.t_on < effOffLayerSnap(a) - NDFX_EPS;
        const bool same_drr    = samePhysicalDrrLayer(a, b);
        const bool same_parent = a.parent_id == b.parent_id;

        CommCostPair p;
        p.edge_id        = e.id;
        p.source_task    = e.source;
        p.target_task    = e.target;
        p.comm_a         = a.comm_id;
        p.comm_b         = b.comm_id;
        p.weight         = e.weight;
        p.parent_a       = a.parent_id;
        p.parent_b       = b.parent_id;
        p.version_a      = a.layout_version;
        p.version_b      = b.layout_version;
        p.container_a    = a.container_id;
        p.container_b    = b.container_id;
        p.layer_a        = a.layer_idx;
        p.layer_b        = b.layer_idx;
        p.kind_a         = static_cast<int>(a.kind);
        p.kind_b         = static_cast<int>(b.kind);
        p.time_overlap   = overlap;
        p.same_drr_reuse = same_drr;
        p.final_full_cover_reuse = finalFullCoverReuse(a, b);
        p.owner_sub_a    = a.owner_sub;
        p.owner_sub_b    = b.owner_sub;
        p.configuration_a = a.configuration_id;
        p.configuration_b = b.configuration_id;
        p.relation_cost  = dmcRelationForLayers(a, b);
        p.relation       = p.final_full_cover_reuse
            ? "same_subparent_final_full_cover_reuse"
            : dmcRelationName(overlap, same_drr, same_parent, a.layout_version == b.layout_version);
        p.cost   = e.weight * p.relation_cost;
        p.dx     = std::abs(a.rect.cx() - b.rect.cx());
        p.dy     = std::abs(a.rect.cy() - b.rect.cy());
        p.d_norm = (p.dx + p.dy) / std::max(1, fpga_.W + fpga_.H);
        p.dt     = overlap ? 0.0 : std::max(0.0, std::max(a.t_on - effOffLayerSnap(b),
                                                          b.t_on - effOffLayerSnap(a)));
        p.dt_norm = p.dt / std::max(1.0, params.T_max);
        total += p.cost;
        out.push_back(p);
    }
    return {total, std::move(out)};
}

inline std::vector<layout::PlacedComm> NestedDFXAllocator::snapshotForCommCost() const {
    std::vector<layout::PlacedComm> all;
    auto push = [&](const TimeLayer& tl) {
        layout::PlacedComm p;
        p.comm_id          = tl.comm_id;
        p.rect             = tl.rect;
        p.t_on             = tl.t_on;
        p.t_off            = tl.t_off;
        p.still_active     = tl.active;
        p.time_layer       = tl.layer_idx;
        p.drr_id           = 0;
        p.last_place_cost  = tl.placement_cost;
        p.demand           = tl.demand;
        p.has_exact_demand = true;
        all.push_back(p);
    };
    for (const auto& [k, vec] : child_layers_)       { (void)k; for (const auto& tl : vec) push(tl); }
    for (const auto& [k, vec] : sub_layers_)         { (void)k; for (const auto& tl : vec) push(tl); }
    for (const auto& [k, vec] : parent_mono_layers_) { (void)k; for (const auto& tl : vec) push(tl); }
    return all;
}

} // namespace nested_dfx
