// Included inside namespace nested_dfx in nested_dfx_allocator.hpp.
// Authoritative hierarchical slot and configuration consistency model.

inline int NestedDFXAllocator::makeConfiguration(int pid, int sid, int version, bool whole) {
    RegionConfiguration c;
    c.id = next_configuration_++;
    c.parent_id = pid;
    c.owner_sub = sid;
    c.layout_version = version;
    c.whole = whole;
    if (sid >= 0) c.parent_configuration = parents_.at(pid).split_configuration;
    configurations_[c.id] = c;
    return c.id;
}

inline int NestedDFXAllocator::ensureFullSlot(int pid, int sid) {
    auto& p = parents_.at(pid);
    int& existing = sid >= 0 ? subparents_.at(sid).full_slot : p.full_slot;
    if (existing >= 0) return existing;
    const Rect r = sid >= 0 ? subparents_.at(sid).rect : p.rect;
    const int id = createChildDRR(pid, r);
    auto& slot = child_drrs_.at(id);
    slot.owner_sub = sid;
    slot.full_cover = true;
    slot.parent_full = sid < 0;
    slot.configuration_id = makeConfiguration(pid, sid, p.current_layout_version, true);
    auto& list = p.child_drr_ids;
    list.erase(std::remove(list.begin(), list.end(), id), list.end());
    existing = id;
    return id;
}

inline void NestedDFXAllocator::reconcileSlotHierarchy(int pid) {
    auto& p = parents_.at(pid);
    if (p.split_configuration < 0)
        p.split_configuration = makeConfiguration(pid, -1, p.current_layout_version, false);
    for (int sid : p.subparent_ids) {
        auto& sub = subparents_.at(sid);
        if (sub.split_configuration < 0)
            sub.split_configuration = makeConfiguration(pid, sid, sub.layout_version, false);
    }
    for (int id : p.child_drr_ids) {
        auto& slot = child_drrs_.at(id);
        if (slot.layout_version != p.current_layout_version) continue;
        int owner = -1;
        for (int sid : p.subparent_ids)
            if (rectContains(subparents_.at(sid).rect, slot.rect)) { owner = sid; break; }
        if (owner != slot.owner_sub) {
            slot.owner_sub = owner;
            slot.configuration_id = -1;
        }
        if (!slot.retained || slot.configuration_id < 0) {
            const int split = owner < 0 ? p.split_configuration : subparents_.at(owner).split_configuration;
            bool intersects = false;
            if (slot.retained) {
                for (int other : p.child_drr_ids) {
                    if (other == id) continue;
                    const auto& b = child_drrs_.at(other);
                    if (b.layout_version == slot.layout_version && Rect::overlap(slot.rect, b.rect)) {
                        intersects = true;
                        break;
                    }
                }
            }
            slot.configuration_id = intersects
                ? makeConfiguration(pid, owner, slot.layout_version, false) : split;
        }
    }
    for (auto& [id, slot] : child_drrs_) {
        (void)id;
        if (slot.parent_id != pid || !slot.full_cover) continue;
        slot.rect = slot.owner_sub >= 0 ? subparents_.at(slot.owner_sub).rect : p.rect;
    }
    recomputeSubContainments(pid);
}

inline std::map<std::string, int> NestedDFXAllocator::configurationRequirements(int id) const {
    const auto& slot = child_drrs_.at(id);
    const auto& config = configurations_.at(slot.configuration_id);
    std::map<std::string, int> result;
    result["P" + std::to_string(slot.parent_id)] = slot.owner_sub < 0
        ? config.id : config.parent_configuration;
    if (slot.owner_sub >= 0) result["S" + std::to_string(slot.owner_sub)] = config.id;
    return result;
}

inline bool NestedDFXAllocator::validateSlots(std::string* error) const {
    auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (!sub_layers_.empty()) {
        for (const auto& [id, ls] : sub_layers_) {
            (void)id;
            if (!ls.empty()) return fail("community directly resident on SubParent; missing physical slot");
        }
    }
    for (const auto& [id, ls] : parent_mono_layers_) {
        (void)id;
        if (!ls.empty()) return fail("community directly resident on Parent; missing physical slot");
    }
    for (const auto& [id, c] : configurations_) {
        if (id != c.id || !parents_.count(c.parent_id)) return fail("orphan configuration");
        if (c.owner_sub >= 0) {
            auto s = subparents_.find(c.owner_sub);
            auto p = configurations_.find(c.parent_configuration);
            if (s == subparents_.end() || s->second.parent_id != c.parent_id ||
                s->second.layout_version != c.layout_version || p == configurations_.end() ||
                p->second.owner_sub >= 0 || p->second.whole ||
                p->second.parent_id != c.parent_id || p->second.layout_version != c.layout_version)
                return fail("incompatible ancestor configuration");
        }
    }
    for (const auto& [id, slot] : child_drrs_) {
        (void)id;
        auto ci = configurations_.find(slot.configuration_id);
        if (ci == configurations_.end()) return fail("slot lacks owner configuration");
        const auto& c = ci->second;
        if (c.parent_id != slot.parent_id || c.owner_sub != slot.owner_sub ||
            c.layout_version != slot.layout_version) return fail("slot/configuration ownership mismatch");
        if (slot.owner_sub >= 0) {
            const auto& sub = subparents_.at(slot.owner_sub);
            if (!rectContains(sub.rect, slot.rect)) return fail("slot outside explicit owner SubParent");
            if (slot.full_cover && !rectsEqual(slot.rect, sub.rect)) return fail("stale full-cover slot");
        } else if (slot.full_cover && !rectsEqual(slot.rect, parents_.at(slot.parent_id).rect))
            return fail("stale Parent full-cover slot");
        if (slot.full_cover != c.whole) return fail("full-cover/configuration role mismatch");
        if (slot.owner_sub < 0 && !slot.parent_full) {
            for (const auto& [sid, sub] : subparents_) {
                (void)sid;
                if (sub.parent_id != slot.parent_id || sub.layout_version != slot.layout_version) continue;
                auto sc = configurations_.find(sub.split_configuration);
                if (sc != configurations_.end() && sc->second.parent_configuration == slot.configuration_id &&
                    Rect::overlap(slot.rect, sub.rect))
                    return fail("unowned child intersects sibling SubParent");
            }
        }
    }
    for (auto a = subparents_.begin(); a != subparents_.end(); ++a)
        for (auto b = std::next(a); b != subparents_.end(); ++b)
            if (a->second.parent_id == b->second.parent_id &&
                a->second.layout_version == b->second.layout_version &&
                Rect::overlap(a->second.rect, b->second.rect))
                return fail("SubParent siblings overlap in a historical configuration");
    for (auto a = child_drrs_.begin(); a != child_drrs_.end(); ++a) {
        for (auto b = std::next(a); b != child_drrs_.end(); ++b) {
            if (a->second.configuration_id == b->second.configuration_id &&
                Rect::overlap(a->second.rect, b->second.rect))
                return fail("overlapping sibling slots in same configuration: D" +
                            std::to_string(a->first) + " / D" + std::to_string(b->first));
        }
    }
    struct Ref {
        int slot;
        const TimeLayer* layer;
    };
    std::vector<Ref> refs;
    for (const auto& [id, ls] : child_layers_) {
        if (!child_drrs_.count(id)) return fail("residence references missing slot");
        for (const auto& l : ls) {
            if (l.layout_version != child_drrs_.at(id).layout_version)
                return fail("residence layout lineage mismatch");
            refs.push_back({id, &l});
        }
    }
    auto off = [](const TimeLayer& l) {
        return l.active || l.t_off < 0 ? std::numeric_limits<double>::infinity() : l.t_off;
    };
    for (size_t i = 0; i < refs.size(); ++i) {
        for (size_t j = i + 1; j < refs.size(); ++j) {
            const auto& a = refs[i];
            const auto& b = refs[j];
            if (std::max(a.layer->t_on, b.layer->t_on) >=
                std::min(off(*a.layer), off(*b.layer)) - NDFX_EPS)
                continue;
            const auto ca = configurationRequirements(a.slot);
            const auto cb = configurationRequirements(b.slot);
            for (const auto& [owner, config] : ca) {
                auto bi = cb.find(owner);
                if (bi != cb.end() && bi->second != config)
                    return fail("incompatible simultaneous configurations at " + owner + ": C" +
                                std::to_string(a.layer->comm_id) + " / C" +
                                std::to_string(b.layer->comm_id));
            }
            if (a.slot != b.slot && Rect::overlap(reservationRect(a.slot), reservationRect(b.slot)))
                return fail("global reservation collision: C" +
                            std::to_string(a.layer->comm_id) + " / C" +
                            std::to_string(b.layer->comm_id));
        }
    }
    return true;
}

inline void NestedDFXAllocator::releaseGroup(const std::vector<int>& communities, double time) {
    const std::set<int> requested(communities.begin(), communities.end());
    for (int cid : communities)
        for (int peer : activeCommsInSameChildDrr(cid))
            if (!requested.count(peer)) throw std::logic_error("partial cohort release");
    for (int cid : communities) release(cid, time);
}
