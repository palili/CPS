#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>

namespace scheduling {

// Computes the earliest strictly-future event timestamp across running tasks
// and predecessor-satisfied tasks awaiting their earliest start time (EST).
template <class TaskMap>
double nextEventTime(const TaskMap& tasks, double t_now) {
    double next = std::numeric_limits<double>::infinity();
    for (const auto& [id, task] : tasks) {
        using State = std::decay_t<decltype(task.state)>;
        if (task.state == State::RUNNING) {
            next = std::min(next, task.finish_time);
        } else if ((task.state == State::IDLE ||
                    task.state == State::WAITING ||
                    task.state == State::READY) && task.EST > t_now + 1e-9) {
            const bool ready = std::all_of(
                task.predecessors.begin(), task.predecessors.end(),
                [&](int pred) { return tasks.at(pred).state == State::DONE; });
            if (ready) next = std::min(next, task.EST);
        }
    }
    return next;
}

} // namespace scheduling
