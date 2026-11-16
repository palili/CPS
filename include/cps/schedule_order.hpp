#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace scheduling {

inline bool feasible(double makespan) {
    return std::isfinite(makespan) && makespan >= 0.0;
}

inline bool betterSchedule(double a_time, double a_comm,
                           double b_time, double b_comm) {
    const bool a_ok = feasible(a_time), b_ok = feasible(b_time);
    if (a_ok != b_ok) return a_ok;
    if (!a_ok) return false;
    if (a_time != b_time) return a_time < b_time;
    const bool a_finite = std::isfinite(a_comm), b_finite = std::isfinite(b_comm);
    if (a_finite != b_finite) return a_finite;
    if (!a_finite) return false;
    return a_comm < b_comm;
}

template <class Result>
bool betterResult(const Result& a, const Result& b) {
    return betterSchedule(a.makespan, a.comm_cost_sum, b.makespan, b.comm_cost_sum);
}

// Diagnostic scalar summary for logging only; selection always uses Objective::better.
inline double displayFitness(double makespan, double communication, double epsilon = 1e-3) {
    if (!feasible(makespan) || !std::isfinite(communication)) return 0.0;
    const double obj = makespan + epsilon * communication;
    return 1.0 / (obj > 1e-6 ? obj : 1e-6);
}

template<class T, class = void> struct detail_has_box : std::false_type {};
template<class T> struct detail_has_box<T, std::void_t<decltype(std::declval<const T&>().resource_box_sum)>>
    : std::true_type {};

// Lexicographic schedule comparator:
// - Default (`reseff_dmc_bucket == 0`): strict (Makespan, ICC, -RTU).
// - Packing-aware presets (`reseff_dmc_bucket > 0`): within equal Makespan,
//   compares transitive log-buckets of ICC first, then smaller resource_box_sum
//   (higher RTU), then exact ICC.
struct Objective {
    double reseff_dmc_bucket = 0.0;

    long long dmcBucket(double dmc) const {
        if (!(reseff_dmc_bucket > 0.0) || !(dmc > 0.0)) return 0;
        return static_cast<long long>(std::floor(std::log(dmc) / std::log1p(reseff_dmc_bucket)));
    }

    bool accepts(double time, double dmc) const {
        return feasible(time) && std::isfinite(dmc);
    }

    template<class Result>
    bool better(const Result& a, const Result& b) const {
        if constexpr (detail_has_box<Result>::value) {
            const bool aok = feasible(a.makespan) && std::isfinite(a.comm_cost_sum);
            const bool bok = feasible(b.makespan) && std::isfinite(b.comm_cost_sum);
            if (aok && bok && a.makespan == b.makespan) {
                if (reseff_dmc_bucket > 0.0) {
                    const long long ab = dmcBucket(a.comm_cost_sum);
                    const long long bb = dmcBucket(b.comm_cost_sum);
                    if (ab != bb) return ab < bb;
                    if (a.resource_box_sum != b.resource_box_sum)
                        return a.resource_box_sum < b.resource_box_sum;
                    return a.comm_cost_sum < b.comm_cost_sum;
                }
                if (a.comm_cost_sum != b.comm_cost_sum) return a.comm_cost_sum < b.comm_cost_sum;
                return a.resource_box_sum < b.resource_box_sum;
            }
        }
        return betterSchedule(a.makespan, a.comm_cost_sum, b.makespan, b.comm_cost_sum);
    }
};

} // namespace scheduling
