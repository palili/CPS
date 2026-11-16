#pragma once

#include "cps/nested_dfx_allocator.hpp"
#include "nlohmann/json.hpp"

inline nlohmann::json slotSemanticsJson(const nested_dfx::NestedDFXAllocator& a) {
    using nlohmann::json;
    auto rect = [](const auto& r) { return json::array({r.x, r.y, r.w, r.h}); };
    json out = {
        {"schema", "explicit_nested_slots_v2"},
        {"layout_revision", a.layoutRevision()},
        {"reconfiguration_duration", 0},
        {"duration_model", "ideal_zero_latency_not_measured_hardware"}
    };
    out["configurations"]   = json::array();
    out["slots"]            = json::array();
    out["planning_changes"] = json::array();
    out["residences"]       = json::array();

    for (const auto& [id, c] : a.configurations()) {
        json slots = json::array();
        for (const auto& [did, d] : a.childDrrs())
            if (d.configuration_id == id) slots.push_back(did);
        out["configurations"].push_back({
            {"id", id},
            {"parent", c.parent_id},
            {"owner_sub", c.owner_sub},
            {"parent_configuration", c.parent_configuration},
            {"layout_version", c.layout_version},
            {"mode", c.whole ? "WHOLE" : "SUBDIVIDED"},
            {"slots", slots}
        });
    }
    for (const auto& [id, d] : a.childDrrs()) {
        out["slots"].push_back({
            {"id", id},
            {"parent", d.parent_id},
            {"owner_sub", d.owner_sub},
            {"configuration", d.configuration_id},
            {"full_cover", d.full_cover},
            {"parent_full", d.parent_full},
            {"retained", d.retained},
            {"layout_version", d.layout_version},
            {"reservation_rect", rect(d.rect)},
            {"requirements", a.configurationRequirements(id)}
        });
    }
    for (const auto& c : a.planningChanges()) {
        out["planning_changes"].push_back({
            {"time", c.time},
            {"operation", c.operation},
            {"sub", c.sub},
            {"slot", c.slot},
            {"before", rect(c.before)},
            {"after", rect(c.after)}
        });
    }

    std::map<std::pair<int, double>, json> groups;
    for (const auto& [id, ls] : a.childLayers()) {
        for (const auto& l : ls) {
            out["residences"].push_back({
                {"community", l.comm_id},
                {"slot", id},
                {"layer", l.layer_idx},
                {"configuration", a.childDrrs().at(id).configuration_id},
                {"on", l.t_on},
                {"off", l.t_off},
                {"active", l.active},
                {"placement_rect", rect(l.rect)},
                {"reservation_rect", rect(a.reservationRect(id))},
                {"requirements", a.configurationRequirements(id)}
            });
            auto& g = groups[{id, l.t_on}];
            if (g.is_null()) {
                g = {
                    {"slot", id},
                    {"on", l.t_on},
                    {"off", l.t_off},
                    {"members", json::array()},
                    {"requirements", a.configurationRequirements(id)}
                };
            }
            g["members"].push_back(l.comm_id);
        }
    }
    out["groups"] = json::array();
    out["events"] = json::array();
    for (const auto& [key, g] : groups) {
        (void)key;
        out["groups"].push_back(g);
        out["events"].push_back({
            {"time", g["on"]},
            {"type", "LOAD_GROUP"},
            {"slot", g["slot"]},
            {"members", g["members"]},
            {"requirements", g["requirements"]}
        });
        if (g["off"].get<double>() >= 0) {
            out["events"].push_back({
                {"time", g["off"]},
                {"type", "UNLOAD_GROUP"},
                {"slot", g["slot"]},
                {"members", g["members"]}
            });
        }
    }
    auto& events = out["events"];
    std::sort(events.begin(), events.end(), [](const json& a_ev, const json& b_ev) {
        if (a_ev["time"] != b_ev["time"])
            return a_ev["time"].get<double>() < b_ev["time"].get<double>();
        if (a_ev["type"] != b_ev["type"])
            return a_ev["type"] == "UNLOAD_GROUP";
        return a_ev["slot"].get<int>() < b_ev["slot"].get<int>();
    });
    json switches = json::array();
    std::map<std::string, int> loaded;
    for (const auto& e : events) {
        if (e["type"] == "LOAD_GROUP") {
            const auto req = e["requirements"].get<std::map<std::string, int>>();
            for (const auto& [owner, id] : req) {
                if (!loaded.count(owner) || loaded.at(owner) != id) {
                    const int before = loaded.count(owner) ? loaded.at(owner) : -1;
                    switches.push_back({
                        {"time", e["time"]},
                        {"owner", owner},
                        {"from", before},
                        {"to", id},
                        {"duration", 0}
                    });
                    if (owner[0] == 'P') {
                        const int pid = std::stoi(owner.substr(1));
                        for (const auto& [sid, sub] : a.subparents())
                            if (sub.parent_id == pid) loaded.erase("S" + std::to_string(sid));
                    }
                    loaded[owner] = id;
                }
            }
        }
    }
    out["configuration_switches"] = switches;
    return out;
}
