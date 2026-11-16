#include <iostream>
#include <stdexcept>
#include <string>

#include "cps/nested_dfx_allocator.hpp"
#include "cps/slot_export.hpp"

namespace nested_dfx {

struct SlotTestAccess {
    static CommCostLayerSnap snapshot(const NestedDFXAllocator& a, int slot, double on, double off) {
        CommCostLayerSnap s;
        s.kind           = ActivePlacement::CHILD;
        s.container_id   = slot;
        s.parent_id      = a.child_drrs_.at(slot).parent_id;
        s.layout_version = a.child_drrs_.at(slot).layout_version;
        s.t_on           = on;
        s.t_off          = off;
        a.fillDmcSlotContext(s);
        return s;
    }
    static void boundLayer(NestedDFXAllocator& a, int slot, int cid, int task, double on, double off) {
        a.commitChildLayer(slot, cid, on, {4, 0, 0}, 0, 0, nullptr);
        a.bindTaskToActiveResidence(task, cid);
        a.releaseGroup({cid}, off);
    }
    static double increment(NestedDFXAllocator& a, int slot, int cid, int task, double on) {
        a.setPendingCommunicationTasks({task});
        const double value = a.estimateDmcIncrement(
            cid, on, ActivePlacement::CHILD, a.child_drrs_.at(slot).parent_id, slot);
        a.setPendingCommunicationTasks({});
        return value;
    }
    static int parent(NestedDFXAllocator& a) {
        const int p = a.createParentRegion({0, 0, 20, 10});
        a.reconcileSlotHierarchy(p);
        return p;
    }
    static int child(NestedDFXAllocator& a, int p, Rect r) {
        const int d = a.createChildDRR(p, r);
        a.reconcileSlotHierarchy(p);
        return d;
    }
    static int sub(NestedDFXAllocator& a, int p, Rect r) {
        const int s = a.createSubParent(p, r);
        a.reconcileSlotHierarchy(p);
        a.ensureFullSlot(p, s);
        return s;
    }
    static void layer(NestedDFXAllocator& a, int d, int c, double on, double off, Rect occupied = {}) {
        a.commitChildLayer(d, c, on, {4, 0, 0}, 0, 0, occupied.valid() ? &occupied : nullptr);
        if (off >= 0) a.releaseGroup({c}, off);
    }
    static bool expand(NestedDFXAllocator& a, int sid, Rect r, bool retained_only = false) {
        NestedDFXAllocator::PlacementPlan plan;
        plan.mode            = PlacementMode::SUB_EXPAND;
        plan.parent_id       = a.subparents_.at(sid).parent_id;
        plan.container_id    = sid;
        plan.rect            = r;
        plan.retain_old_slot = retained_only;
        NestedDFXAllocator::PlacementSearch search;
        if (retained_only) a.evaluateSlotPlan(99, 30, {80, 0, 0}, plan, search);
        else a.considerPlan(99, 30, {80, 0, 0}, plan, search);
        if (!search.state) return false;
        a = std::move(*search.state);
        return true;
    }
    static void resizeFollow(NestedDFXAllocator& a, int sid, Rect r) {
        auto& s = a.subparents_.at(sid);
        s.rect  = r;
        a.reconcileSlotHierarchy(s.parent_id);
    }
};

} // namespace nested_dfx

namespace {

using namespace nested_dfx;
using X = SlotTestAccess;

void require(bool condition, const char* label) {
    if (!condition) throw std::runtime_error(label);
    std::cout << "PASS " << label << '\n';
}

NestedDFXAllocator makeTestAllocator() {
    layout::FPGAModel f;
    f.W = 20;
    f.H = 10;
    f.col_type.assign(20, layout::COL_CLB);
    f.buildPrefix();

    NestedDFXConfig c;
    c.enable_refresh    = false;
    c.candidate_scoring = CandidateScoring::ACTION_ROLE;
    c.reseff_weight     = 0.0;

    NestedDFXAllocator a;
    a.init(f, c, [](int, int) { return 1.0; });
    return a;
}

void runSlotSemanticsTests() {
    auto a    = makeTestAllocator();
    int p     = X::parent(a);
    int d     = X::child(a, p, {0, 0, 5, 10});
    int s     = X::sub(a, p, {5, 0, 5, 10});
    int whole = a.subparents().at(s).full_slot;
    X::layer(a, d, 7, 5, 20);
    X::layer(a, whole, 3, 10, 30);

    std::string error;
    require(a.validate(&error), "initial disjoint C3/C7 history");

    auto follow = a;
    X::resizeFollow(follow, s, {0, 0, 10, 10});
    require(!follow.validate(&error), "global full-cover growth rejects historical conflict");

    require(X::expand(a, s, {0, 0, 10, 10}), "retained-slot candidate resolves history cleanly");
    require(a.validate(&error), "retained candidate validates");
    require(a.childDrrs().at(whole).retained && !a.childDrrs().at(whole).full_cover &&
            a.childDrrs().at(whole).rect.x == 5 && a.childDrrs().at(whole).rect.w == 5,
            "old physical slot identity and boundary preserved");
    require(a.childDrrs().at(whole).owner_sub == s && a.childDrrs().at(d).owner_sub == s,
            "explicit adopted ownership after global expansion");
    require(a.subparents().at(s).full_slot != whole && a.subLayers().at(s).empty(),
            "new full-cover slot and no direct SubParent residence");

    auto overlap = a;
    X::child(overlap, p, {5, 0, 5, 10});
    require(!overlap.validate(&error), "duplicate sibling slot rejected even when old slot idle");

    auto config = makeTestAllocator();
    p = X::parent(config);
    s = X::sub(config, p, {0, 0, 10, 10});
    int left  = X::child(config, p, {0, 0, 3, 10});
    int right = X::child(config, p, {7, 0, 3, 10});
    int full  = config.subparents().at(s).full_slot;
    X::layer(config, left, 1, 0, 10);
    X::layer(config, full, 2, 10, 20);
    X::layer(config, right, 3, 20, 30);
    require(config.validate(&error), "legal split-whole-split switching on half-open endpoints");

    auto bad = config;
    X::layer(bad, right, 4, 15, 20);
    require(!bad.validate(&error), "incompatible configurations cannot coexist");

    auto same = makeTestAllocator();
    p = X::parent(same);
    d = X::child(same, p, {0, 0, 4, 4});
    X::layer(same, d, 1, 0, -1, {0, 0, 2, 2});
    X::layer(same, d, 2, 0, -1, {2, 0, 2, 2});
    require(same.validate(&error), "composite members share reservation with disjoint internal placements");

    bool rejected = false;
    try {
        same.releaseGroup({1}, 10);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected && same.childDrrs().at(d).active_comm_ids.size() == 2,
            "partial group release rejected without mutation");

    same.releaseGroup({1, 2}, 10);
    require(same.validate(&error), "atomic full group release");
    const auto out = slotSemanticsJson(same);
    require(out["groups"].size() == 1 && out["events"].size() == 2 &&
            out["groups"][0]["members"].size() == 2,
            "one load and unload exported per composite RM");

    auto nested = makeTestAllocator();
    p    = X::parent(nested);
    s    = X::sub(nested, p, {5, 0, 5, 10});
    left = X::child(nested, p, {5, 0, 2, 10});
    full = nested.subparents().at(s).full_slot;
    X::layer(nested, left, 1, 0, 5);
    X::layer(nested, full, 2, 5, 15);
    require(X::expand(nested, s, {0, 0, 10, 10}, true) && nested.validate(&error),
            "retaining covering slot with existing children uses independent configuration");
    require(nested.childDrrs().at(left).configuration_id !=
            nested.childDrrs().at(full).configuration_id,
            "covering and covered children never siblings in one configuration");
}

void runFullCoverIccTests() {
    constexpr int k = FINAL_FULL_COVER_DMC_COST;
    auto a     = makeTestAllocator();
    int p      = X::parent(a);
    int s      = X::sub(a, p, {5, 0, 5, 10});
    int child  = X::child(a, p, {5, 0, 2, 10});
    int whole  = a.subparents().at(s).full_slot;
    auto c     = X::snapshot(a, child, 0, 5);
    auto w     = X::snapshot(a, whole, 5, 15);

    require(!samePhysicalDrrLayer(c, w) && finalFullCoverReuse(c, w) &&
            dmcRelationForLayers(c, w) == k,
            "independent configuration full-cover reuse uses selected coefficient");
    require(finalFullCoverReuse(w, c) && dmcRelationForLayers(w, c) == k,
            "full-cover ICC metric symmetric in edge direction");

    auto cross = w;
    cross.layout_version++;
    require(!finalFullCoverReuse(c, cross) && dmcRelationForLayers(c, cross) == 2,
            "cross-layout-version coefficient remains two");

    auto stale = w;
    stale.owner_sub_final_rect = {0, 0, 10, 10};
    require(!finalFullCoverReuse(c, stale) && dmcRelationForLayers(c, stale) == 2,
            "old full-cover boundary cannot qualify after owner grows");

    auto demoted = w;
    demoted.verified_full_cover = false;
    require(!finalFullCoverReuse(c, demoted) && dmcRelationForLayers(c, demoted) == 2,
            "demoted role loses full-cover discount despite same old coordinates");

    auto other = w;
    other.owner_sub++;
    require(!finalFullCoverReuse(c, other) && dmcRelationForLayers(c, other) == 2,
            "different SubParent does not qualify");

    other = w;
    other.parent_configuration++;
    require(!finalFullCoverReuse(c, other) && dmcRelationForLayers(c, other) == 2,
            "incompatible ancestor does not qualify");

    other = w;
    other.parent_id++;
    require(!finalFullCoverReuse(c, other) && dmcRelationForLayers(c, other) == 3,
            "different Parent sequential coefficient unchanged");

    auto bad = c;
    bad.final_rect = {0, 0, 20, 10};
    require(!finalFullCoverReuse(bad, w), "other slot must be inside final whole boundary");

    other = w;
    other.t_on = 2;
    require(!finalFullCoverReuse(c, other) && dmcRelationForLayers(c, other) == 1,
            "overlap does not become same-slot zero cost or full-cover reuse");

    auto reused = c;
    reused.t_on  = 5;
    reused.t_off = 10;
    require(dmcRelationForLayers(c, reused) == 1 && !finalFullCoverReuse(c, reused),
            "same-slot sequential reuse unchanged");

    reused = c;
    require(dmcRelationForLayers(c, reused) == 0,
            "same-slot overlapping coefficient unchanged");

    a.setTaskCommunicationEdges({TaskCommunicationEdge{0, 1, 2, 7.0}});
    X::boundLayer(a, child, 1, 1, 0, 5);
    require(X::increment(a, whole, 2, 2, 5) == 7.0 * k,
            "candidate estimate uses full-cover rule");
    X::boundLayer(a, whole, 2, 2, 5, 15);

    layout::SearchParams params;
    require(a.recomputeTotalCommCost(params) == 7.0 * k,
            "final recompute uses selected coefficient");
    auto details = a.recomputeCommCostDetail(params);
    require(details.first == 7.0 * k && details.second.size() == 1 &&
            details.second[0].final_full_cover_reuse &&
            !details.second[0].same_drr_reuse &&
            details.second[0].relation == "same_subparent_final_full_cover_reuse",
            "export distinguishes metric discount from actual slot identity");

    require(X::expand(a, s, {0, 0, 10, 10}, true), "real retained-slot expansion succeeds");
    require(a.recomputeTotalCommCost(params) == 14.0,
            "final role revision removes discount from previously resolved edge");
    details = a.recomputeCommCostDetail(params);
    require(details.first == 14.0 && !details.second[0].final_full_cover_reuse &&
            details.second[0].relation_cost == 2,
            "final edge detail agrees after full-cover demotion");

    auto after      = X::snapshot(a, whole, 5, 15);
    auto finalwhole = X::snapshot(a, a.subparents().at(s).full_slot, 30, 40);
    require(finalFullCoverReuse(after, finalwhole) &&
            dmcRelationForLayers(after, finalwhole) == k,
            "retained slot may reuse a different genuinely full-cover counterpart");
}

} // namespace

int main() {
    try {
        runSlotSemanticsTests();
        runFullCoverIccTests();
        std::cout << "ALL CPS SLOT & ICC REGRESSION TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
