#pragma once

#include <stdexcept>
#include <string>

namespace cps {

enum class Preset {
    STANDARD,       // Default primary mode: strict lexicographic (Makespan, ICC, -RTU) + pure Eq. (16) slot cost
    COMM_AWARE,     // Inner allocator enables incremental ICC guidance
    PACKING_AWARE,  // Inner allocator enables area-time compactness + 3% outer ICC log-bucket
    BALANCED        // Joint ICC guidance + area-time compactness + 1% outer ICC log-bucket
};

struct PresetParameters {
    Preset preset = Preset::STANDARD;
    bool enable_dmc_scoring = false;
    double new_parent_weight = 0.20;
    double dmc_weight = 0.0;
    double reseff_weight = 0.0;
    double reseff_dmc_bucket = 0.0;
};

inline const char* presetName(Preset p) {
    switch (p) {
        case Preset::STANDARD:      return "standard";
        case Preset::COMM_AWARE:    return "comm-aware";
        case Preset::PACKING_AWARE: return "packing-aware";
        case Preset::BALANCED:      return "balanced";
    }
    return "standard";
}

inline Preset parsePreset(const std::string& text) {
    if (text == "standard" || text == "default")      return Preset::STANDARD;
    if (text == "comm-aware" || text == "dmc")        return Preset::COMM_AWARE;
    if (text == "packing-aware" || text == "reseff")  return Preset::PACKING_AWARE;
    if (text == "balanced" || text == "dmc_reseff")   return Preset::BALANCED;
    throw std::invalid_argument(
        "Unknown --preset '" + text + "'. Expected: standard | comm-aware | packing-aware | balanced");
}

inline PresetParameters makePresetParameters(Preset p) {
    PresetParameters out;
    out.preset = p;
    out.new_parent_weight = 0.20;
    switch (p) {
        case Preset::STANDARD:
            break;
        case Preset::COMM_AWARE:
            out.enable_dmc_scoring = true;
            out.dmc_weight = 1.0;
            break;
        case Preset::PACKING_AWARE:
            out.reseff_weight = 1.0;
            out.reseff_dmc_bucket = 0.03;
            break;
        case Preset::BALANCED:
            out.enable_dmc_scoring = true;
            out.dmc_weight = 1.0;
            out.reseff_weight = 1.0;
            out.reseff_dmc_bucket = 0.01;
            break;
    }
    return out;
}

} // namespace cps
