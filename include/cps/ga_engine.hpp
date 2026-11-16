#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cps/chromosome_io.hpp"
#include "cps/event_decoder.hpp"
#include "cps/preset_config.hpp"
#include "cps/schedule_order.hpp"
#include "cps/slot_export.hpp"
#include "nlohmann/json.hpp"

namespace cps {

// ============================================================================
// Stage 1: Feasibility-Guided Initial Policy Population Sampling
// ============================================================================

struct InitialSeedSolution {
    Chromosome chr;
    double makespan         = -1.0;
    double comm_cost_sum    = 0.0;
    double resource_box_sum = 0.0;
    double repair_fail_sum  = 0.0;
    int attempt_id          = 0;
};

struct InitialSampleResult {
    std::vector<InitialSeedSolution> solutions;
    std::vector<Chromosome> seeds;
    int attempts_used = 0;
    double elapsed_seconds = 0.0;
};

inline InitialSampleResult sampleInitialPopulation(
    unsigned seed,
    int target_solutions = 40,
    int max_attempts = 20000,
    const std::vector<double>& fixed_genes = {},
    bool verbose = true) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 gen(seed);

    std::vector<InitialSeedSolution> all_feasible;
    all_feasible.reserve(static_cast<size_t>(std::max(0, target_solutions)));

    if (verbose) {
        std::cout << "[Stage 1] Sampling feasible initial chromosomes (target="
                  << target_solutions << ", max_attempts=" << max_attempts
                  << ", seed=" << seed << ")...\n";
    }

    int attempt = 0;
    while (attempt < max_attempts &&
           static_cast<int>(all_feasible.size()) < target_solutions) {
        ++attempt;
        Chromosome chr = randomCommunityChromosome(gen, fixed_genes);
        const double ms = runSimulationSilent(chr);
        if (!scheduling::feasible(ms)) continue;

        InitialSeedSolution sol;
        sol.chr              = chr;
        sol.makespan         = ms;
        sol.attempt_id       = attempt;
        sol.comm_cost_sum    = recomputeFinalCommCostFromState();
        sol.resource_box_sum = computeFinalResourceBox();
        sol.repair_fail_sum  = g_total_fail_cost;
        all_feasible.push_back(sol);
    }

    std::stable_sort(all_feasible.begin(), all_feasible.end(),
                     scheduling::betterResult<InitialSeedSolution>);
    if (static_cast<int>(all_feasible.size()) > target_solutions)
        all_feasible.resize(static_cast<size_t>(target_solutions));

    InitialSampleResult out;
    out.attempts_used = attempt;
    out.solutions     = std::move(all_feasible);
    out.seeds.reserve(out.solutions.size());
    for (const auto& s : out.solutions) {
        Chromosome seed_chr = s.chr;
        seed_chr.fitness  = 0.0;
        seed_chr.makespan = -1.0;
        out.seeds.push_back(seed_chr);
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    out.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();

    if (verbose) {
        std::cout << "[Stage 1] Found " << out.solutions.size() << "/" << target_solutions
                  << " feasible seeds in " << out.attempts_used << " attempts ("
                  << std::fixed << std::setprecision(2) << out.elapsed_seconds << " s)";
        if (!out.solutions.empty()) {
            std::cout << " | Best seed: Makespan=" << std::setprecision(1)
                      << out.solutions.front().makespan
                      << ", ICC=" << std::setprecision(1)
                      << out.solutions.front().comm_cost_sum;
        }
        std::cout << "\n";
    }
    return out;
}

// ============================================================================
// Stage 2: Genetic Policy Evolution (SBX + Polynomial Mutation + Restart)
// ============================================================================

struct GAConfig {
    int population_size      = 60;
    int max_generations      = 100;
    double crossover_rate    = 0.90;
    double mutation_rate     = 0.20;
    double sbx_eta           = 15.0;
    double pm_eta            = 10.0;
    int tournament_size      = 3;
    int elite_count          = 2;
    int convergence_gens     = 30;
    int diversity_stagnation = 5;
    int diversity_keep_top   = 5;
    int diversity_boost_gens = 3;
    double diversity_boost_eta = 0.5;
    double epsilon_comm      = 1e-3;
    int evaluation_budget    = 0;
    unsigned seed            = 42;
    bool random_search_only  = false;
    PresetParameters preset_params = makePresetParameters(Preset::STANDARD);
    std::vector<double> fixed_genes;
};

struct GAResult {
    Chromosome best;
    std::vector<double> best_makespan_history;
    std::vector<double> best_icc_history;
    std::vector<double> best_rtu_pct_history;
    std::vector<double> avg_makespan_history;
    std::vector<int>    feasible_count_history;
    int generations_run            = 0;
    double elapsed_seconds         = 0.0;
    int evaluations                = 0;
    int feasible_evaluations       = 0;
    int initialization_evaluations = 0;
    int restart_evaluations        = 0;
    int offspring_evaluations      = 0;
    std::string stop_reason        = "max_generations_reached";
};

inline void evaluateFitness(Chromosome& chr, double epsilon_comm = 1e-3) {
    chr.makespan      = runSimulationSilent(chr);
    chr.comm_cost_sum = recomputeFinalCommCostFromState();
    chr.resource_box_sum = scheduling::feasible(chr.makespan)
                               ? computeFinalResourceBox()
                               : std::numeric_limits<double>::infinity();
    chr.repair_fail_sum = g_total_fail_cost;
    chr.fitness = scheduling::displayFitness(chr.makespan, chr.comm_cost_sum, epsilon_comm);
}

inline std::vector<Chromosome> initializePopulation(
    int pop_size,
    const std::vector<Chromosome>& seeds,
    std::mt19937& rng,
    const std::vector<double>& fixed_genes = {}) {
    std::vector<Chromosome> pop;
    pop.reserve(static_cast<size_t>(pop_size));
    for (size_t i = 0; i < seeds.size() && static_cast<int>(pop.size()) < pop_size; ++i)
        pop.push_back(seeds[i]);
    while (static_cast<int>(pop.size()) < pop_size)
        pop.push_back(randomCommunityChromosome(rng, fixed_genes));
    return pop;
}

inline int tournamentSelect(const std::vector<Chromosome>& pop,
                            int tourn_size,
                            std::mt19937& rng,
                            const scheduling::Objective& objective) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(pop.size()) - 1);
    int best = dist(rng);
    for (int i = 1; i < tourn_size; ++i) {
        const int challenger = dist(rng);
        if (objective.better(pop[challenger], pop[best]))
            best = challenger;
    }
    return best;
}

inline std::pair<Chromosome, Chromosome> sbxCrossover(
    const Chromosome& p1,
    const Chromosome& p2,
    double eta,
    std::mt19937& rng,
    const std::vector<double>& fixed_genes = {}) {
    const std::vector<double> v1 = p1.toVector(), v2 = p2.toVector();
    std::vector<double> c1(CHROM_LEN), c2(CHROM_LEN);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    constexpr double lb = 0.0, ub = 1.0;

    for (int i = 0; i < CHROM_LEN; ++i) {
        const double fg = fixedGeneValue(fixed_genes, i);
        if (fg >= 0.0) {
            c1[i] = c2[i] = fg;
            continue;
        }
        if (dist01(rng) > 0.5 || std::abs(v1[i] - v2[i]) < 1e-14) {
            c1[i] = v1[i];
            c2[i] = v2[i];
            continue;
        }

        const double x1   = std::min(v1[i], v2[i]);
        const double x2   = std::max(v1[i], v2[i]);
        const double diff = x2 - x1;

        const double beta1  = 1.0 + (2.0 * (x1 - lb) / diff);
        const double alpha1 = 2.0 - std::pow(beta1, -(eta + 1.0));
        double u = dist01(rng);
        const double betaq1 = (u <= 1.0 / alpha1)
                                  ? std::pow(u * alpha1, 1.0 / (eta + 1.0))
                                  : std::pow(1.0 / (2.0 - u * alpha1), 1.0 / (eta + 1.0));

        const double beta2  = 1.0 + (2.0 * (ub - x2) / diff);
        const double alpha2 = 2.0 - std::pow(beta2, -(eta + 1.0));
        u = dist01(rng);
        const double betaq2 = (u <= 1.0 / alpha2)
                                  ? std::pow(u * alpha2, 1.0 / (eta + 1.0))
                                  : std::pow(1.0 / (2.0 - u * alpha2), 1.0 / (eta + 1.0));

        c1[i] = std::clamp(0.5 * ((1.0 + betaq1) * x1 + (1.0 - betaq1) * x2), lb, ub);
        c2[i] = std::clamp(0.5 * ((1.0 - betaq2) * x1 + (1.0 + betaq2) * x2), lb, ub);
    }

    Chromosome child1, child2;
    child1.fromVector(c1); child1.fitness = 0.0; child1.makespan = -1.0;
    child2.fromVector(c2); child2.fitness = 0.0; child2.makespan = -1.0;
    return {child1, child2};
}

inline void polynomialMutate(
    Chromosome& chr,
    double mutation_rate,
    double eta,
    std::mt19937& rng,
    const std::vector<double>& fixed_genes = {}) {
    std::vector<double> v = chr.toVector();
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    constexpr double lb = 0.0, ub = 1.0;

    for (int i = 0; i < CHROM_LEN; ++i) {
        const double fg = fixedGeneValue(fixed_genes, i);
        if (fg >= 0.0) {
            v[i] = fg;
            continue;
        }
        if (dist01(rng) >= mutation_rate) continue;

        const double x      = v[i];
        const double delta1 = (x - lb) / (ub - lb);
        const double delta2 = (ub - x) / (ub - lb);
        const double u      = dist01(rng);
        double deltaq       = 0.0;

        if (u < 0.5) {
            const double val = 2.0 * u + (1.0 - 2.0 * u) * std::pow(1.0 - delta1, eta + 1.0);
            deltaq = std::pow(val, 1.0 / (eta + 1.0)) - 1.0;
        } else {
            const double val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * std::pow(1.0 - delta2, eta + 1.0);
            deltaq = 1.0 - std::pow(val, 1.0 / (eta + 1.0));
        }

        v[i] = std::clamp(x + deltaq * (ub - lb), lb, ub);
    }
    chr.fromVector(v);
}

inline GAResult runRandomSearchBaseline(const GAConfig& cfg,
                                        const std::vector<Chromosome>& seeds,
                                        bool verbose = true) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    scheduling::Objective objective;
    objective.reseff_dmc_bucket = cfg.preset_params.reseff_dmc_bucket;

    const int budget = (cfg.evaluation_budget > 0)
                           ? cfg.evaluation_budget
                           : (cfg.population_size + cfg.max_generations * (cfg.population_size - cfg.elite_count));
    std::mt19937 rng(cfg.seed);

    GAResult result;
    Chromosome best_seen;
    bool have_eval = false;
    const double total_work = totalTaskWeightedWork();

    auto eval_one = [&](Chromosome chr, bool is_init) {
        if (!cfg.fixed_genes.empty()) {
            auto v = chr.toVector();
            for (int i = 0; i < CHROM_LEN; ++i)
                if (cfg.fixed_genes[i] >= 0.0) v[i] = cfg.fixed_genes[i];
            chr.fromVector(v);
        }
        evaluateFitness(chr, cfg.epsilon_comm);
        ++result.evaluations;
        if (is_init) ++result.initialization_evaluations;
        else ++result.offspring_evaluations;
        if (scheduling::feasible(chr.makespan)) ++result.feasible_evaluations;
        if (!have_eval || objective.better(chr, best_seen)) best_seen = chr;
        have_eval = true;
    };

    for (size_t i = 0; i < seeds.size() && result.evaluations < budget; ++i)
        eval_one(seeds[i], true);
    while (result.evaluations < budget)
        eval_one(randomCommunityChromosome(rng, cfg.fixed_genes), false);

    result.best            = best_seen;
    result.generations_run = 1;
    result.stop_reason     = "random_search_budget_exhausted";
    if (have_eval) {
        result.best_makespan_history.push_back(best_seen.makespan);
        result.best_icc_history.push_back(best_seen.comm_cost_sum);
        const double rtu_pct = (scheduling::feasible(best_seen.makespan) && best_seen.resource_box_sum > 1e-9)
                                   ? (total_work / best_seen.resource_box_sum * 100.0)
                                   : 0.0;
        result.best_rtu_pct_history.push_back(rtu_pct);
        result.avg_makespan_history.push_back(best_seen.makespan);
        result.feasible_count_history.push_back(result.feasible_evaluations);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();
    if (verbose) {
        std::cout << "[Stage 2] Random-search baseline finished: " << result.evaluations
                  << " evaluations (" << std::fixed << std::setprecision(2)
                  << result.elapsed_seconds << " s)\n";
    }
    return result;
}

inline GAResult runGA(const GAConfig& cfg,
                      const std::vector<Chromosome>& seeds,
                      bool verbose = true) {
    if (cfg.population_size < 2 || cfg.max_generations < 1 || cfg.tournament_size < 1)
        throw std::invalid_argument("GA requires population_size >= 2 and positive generations/tournament_size");
    if (cfg.evaluation_budget < 0 ||
        (cfg.evaluation_budget > 0 && cfg.evaluation_budget < cfg.population_size) ||
        cfg.elite_count < 0 || cfg.elite_count >= cfg.population_size) {
        throw std::invalid_argument("Invalid GA evaluation_budget or elite_count");
    }
    if (!cfg.fixed_genes.empty() && static_cast<int>(cfg.fixed_genes.size()) != CHROM_LEN)
        throw std::invalid_argument("fixed_genes mask must contain 20 entries");
    for (double x : cfg.fixed_genes) {
        if (!std::isfinite(x) || (x != -1.0 && (x < 0.0 || x > 1.0)))
            throw std::invalid_argument("Invalid fixed_genes entry (expected -1 or [0, 1])");
    }

    if (cfg.random_search_only)
        return runRandomSearchBaseline(cfg, seeds, verbose);

    scheduling::Objective objective;
    objective.reseff_dmc_bucket = cfg.preset_params.reseff_dmc_bucket;
    const auto better = [&](const Chromosome& a, const Chromosome& b) {
        return objective.better(a, b);
    };
    auto project = [&](Chromosome& chr) {
        if (!cfg.fixed_genes.empty()) {
            auto v = chr.toVector();
            for (int i = 0; i < CHROM_LEN; ++i)
                if (cfg.fixed_genes[i] >= 0.0) v[i] = cfg.fixed_genes[i];
            chr.fromVector(v);
        }
    };

    const auto start_time = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(cfg.seed);
    const double total_work = totalTaskWeightedWork();

    if (verbose) {
        std::cout << "[Stage 2] Starting GA optimization (preset=" << presetName(cfg.preset_params.preset)
                  << ", pop=" << cfg.population_size
                  << ", max_gen=" << cfg.max_generations
                  << ", max_stag=" << cfg.convergence_gens
                  << ", seed=" << cfg.seed << ")...\n";
    }

    std::vector<Chromosome> pop = initializePopulation(cfg.population_size, seeds, rng, cfg.fixed_genes);
    for (auto& chr : pop) project(chr);

    GAResult result;
    Chromosome best_seen;
    bool have_evaluation = false;
    enum class EvalPhase { INIT, RESTART, OFFSPRING };
    EvalPhase phase = EvalPhase::INIT;

    auto budget_exhausted = [&]() {
        return cfg.evaluation_budget > 0 && result.evaluations >= cfg.evaluation_budget;
    };
    auto evaluate = [&](Chromosome& chr) -> bool {
        if (budget_exhausted()) return false;
        project(chr);
        evaluateFitness(chr, cfg.epsilon_comm);
        ++result.evaluations;
        if (phase == EvalPhase::INIT) ++result.initialization_evaluations;
        else if (phase == EvalPhase::RESTART) ++result.restart_evaluations;
        else ++result.offspring_evaluations;

        if (scheduling::feasible(chr.makespan)) ++result.feasible_evaluations;
        if (!have_evaluation || better(chr, best_seen)) best_seen = chr;
        have_evaluation = true;
        return true;
    };

    for (auto& chr : pop) evaluate(chr);

    Chromosome previous_best;
    int stagnation = 0;

    for (int gen = 0;
         cfg.evaluation_budget > 0 ? !budget_exhausted() : gen < cfg.max_generations;
         ++gen) {
        std::stable_sort(pop.begin(), pop.end(), better);

        double sum_ms = 0.0;
        int cnt_ok    = 0;
        for (const auto& c : pop) {
            if (scheduling::feasible(c.makespan)) {
                sum_ms += c.makespan;
                ++cnt_ok;
            }
        }
        const double avg_ms = cnt_ok > 0 ? (sum_ms / cnt_ok) : -1.0;
        const double best_rtu_pct =
            (scheduling::feasible(best_seen.makespan) && best_seen.resource_box_sum > 1e-9)
                ? (total_work / best_seen.resource_box_sum * 100.0)
                : 0.0;

        result.best_makespan_history.push_back(best_seen.makespan);
        result.best_icc_history.push_back(best_seen.comm_cost_sum);
        result.best_rtu_pct_history.push_back(best_rtu_pct);
        result.avg_makespan_history.push_back(avg_ms);
        result.feasible_count_history.push_back(cnt_ok);

        if (verbose && (gen == 0 || (gen + 1) % 10 == 0)) {
            std::cout << "  Gen " << std::setw(3) << (gen + 1)
                      << " | Best Makespan=" << std::fixed << std::setprecision(1) << best_seen.makespan
                      << " | ICC=" << std::setprecision(1) << best_seen.comm_cost_sum
                      << " | RTU=" << std::setprecision(2) << best_rtu_pct << "%"
                      << " | Feasible=" << cnt_ok << "/" << pop.size() << "\n";
        }

        if (better(best_seen, previous_best)) stagnation = 0;
        else ++stagnation;
        previous_best = best_seen;

        const bool target_met = objective.accepts(best_seen.makespan, best_seen.comm_cost_sum);
        if (cfg.evaluation_budget <= 0 && target_met &&
            cfg.convergence_gens > 0 && stagnation >= cfg.convergence_gens) {
            result.stop_reason     = "feasible_objective_converged";
            result.generations_run = gen + 1;
            break;
        }

        // Diversity restart on stagnation plateau
        bool just_restarted = false;
        if (cfg.diversity_stagnation > 0 &&
            stagnation > 0 &&
            stagnation % cfg.diversity_stagnation == 0 &&
            (cfg.evaluation_budget > 0 || !target_met ||
             cfg.convergence_gens <= 0 || stagnation < cfg.convergence_gens)) {
            int keep = std::max(cfg.elite_count, cfg.diversity_keep_top);
            keep = std::min(keep, static_cast<int>(pop.size()));
            for (int i = keep; i < static_cast<int>(pop.size()); ++i) {
                Chromosome chr = randomCommunityChromosome(rng, cfg.fixed_genes);
                project(chr);
                pop[i] = chr;
            }
            phase = EvalPhase::RESTART;
            for (int i = keep; i < static_cast<int>(pop.size()); ++i)
                if (!evaluate(pop[i])) break;
            if (budget_exhausted()) {
                result.generations_run = gen + 1;
                break;
            }
            std::stable_sort(pop.begin(), pop.end(), better);
            just_restarted = true;
        }

        double current_pm_eta = cfg.pm_eta;
        const int gens_since_restart = stagnation % std::max(cfg.diversity_stagnation, 1);
        if (cfg.diversity_stagnation > 0 &&
            (just_restarted || gens_since_restart < cfg.diversity_boost_gens) &&
            stagnation >= cfg.diversity_stagnation) {
            current_pm_eta = cfg.pm_eta * cfg.diversity_boost_eta;
        }

        std::vector<Chromosome> new_pop;
        new_pop.reserve(static_cast<size_t>(cfg.population_size));
        phase = EvalPhase::OFFSPRING;

        for (int i = 0; i < cfg.elite_count && i < static_cast<int>(pop.size()); ++i)
            new_pop.push_back(pop[i]);

        std::uniform_real_distribution<double> dist01(0.0, 1.0);
        while (static_cast<int>(new_pop.size()) < cfg.population_size && !budget_exhausted()) {
            const int p1_idx = tournamentSelect(pop, cfg.tournament_size, rng, objective);
            int p2_idx       = tournamentSelect(pop, cfg.tournament_size, rng, objective);
            while (p2_idx == p1_idx)
                p2_idx = tournamentSelect(pop, cfg.tournament_size, rng, objective);

            Chromosome child1, child2;
            if (dist01(rng) < cfg.crossover_rate) {
                auto [c1, c2] = sbxCrossover(pop[p1_idx], pop[p2_idx],
                                             cfg.sbx_eta, rng, cfg.fixed_genes);
                child1 = c1;
                child2 = c2;
            } else {
                child1 = pop[p1_idx];
                child2 = pop[p2_idx];
            }

            polynomialMutate(child1, cfg.mutation_rate, current_pm_eta, rng, cfg.fixed_genes);
            polynomialMutate(child2, cfg.mutation_rate, current_pm_eta, rng, cfg.fixed_genes);

            if (!evaluate(child1)) break;
            if (static_cast<int>(new_pop.size()) < cfg.population_size)
                new_pop.push_back(child1);

            if (!evaluate(child2)) break;
            if (static_cast<int>(new_pop.size()) < cfg.population_size)
                new_pop.push_back(child2);
        }

        pop = std::move(new_pop);
        result.generations_run = gen + 1;
    }

    std::stable_sort(pop.begin(), pop.end(), better);
    result.best = best_seen;
    if (cfg.evaluation_budget > 0)
        result.stop_reason = "fixed_evaluation_budget_exhausted";
    else if (!scheduling::feasible(result.best.makespan))
        result.stop_reason = "no_feasible_solution";
    else if (!objective.accepts(result.best.makespan, result.best.comm_cost_sum))
        result.stop_reason = "objective_target_not_met";
    else if (result.stop_reason != "feasible_objective_converged")
        result.stop_reason = "max_generations_reached";

    const auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

    if (verbose) {
        std::cout << "[Stage 2] Finished in " << result.generations_run << " generations ("
                  << result.evaluations << " evaluations, " << std::fixed << std::setprecision(2)
                  << result.elapsed_seconds << " s, stop=" << result.stop_reason << ")\n";
    }
    return result;
}

// ============================================================================
// Structured Artifact Writers (JSON, CSV, Markdown Report)
// ============================================================================

inline void writeInitialSolutionsJson(
    const std::string& path,
    unsigned seed,
    int target_solutions,
    int max_attempts,
    const PresetParameters& preset_params,
    const InitialSampleResult& sample,
    const std::vector<double>& fixed_genes = {}) {
    using nlohmann::json;
    json j_arr = json::array();
    for (size_t i = 0; i < sample.solutions.size(); ++i) {
        const auto& sol = sample.solutions[i];
        json j_sol = chromosomeGenesJSON(sol.chr);
        j_sol["id"]               = static_cast<int>(i + 1);
        j_sol["attempt_id"]       = sol.attempt_id;
        j_sol["makespan"]         = sol.makespan;
        j_sol["icc"]              = sol.comm_cost_sum;
        j_sol["resource_box_sum"] = sol.resource_box_sum;
        j_arr.push_back(j_sol);
    }

    json root;
    root["preset"]                  = presetName(preset_params.preset);
    root["seed"]                    = seed;
    root["selection_order"]         = {"feasible", "makespan", "icc"};
    root["decoder_semantics"]       = "explicit_nested_slots_v2_single_residency";
    root["initial_solution_target"] = target_solutions;
    root["initial_max_attempts"]    = max_attempts;
    root["initial_attempts"]        = sample.attempts_used;
    root["elapsed_seconds"]         = sample.elapsed_seconds;
    appendCommunityScoringMetadata(root, fixed_genes);
    root["solutions"] = j_arr;

    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write initial solutions JSON: " + path);
    f << root.dump(2) << "\n";
}

inline void writeConvergenceCsv(const std::string& path, const GAResult& result) {
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write convergence CSV: " + path);
    f << "generation,best_makespan,best_icc,best_rtu_pct,avg_feasible_makespan,feasible_population_count\n";
    for (size_t i = 0; i < result.best_makespan_history.size(); ++i) {
        f << (i + 1) << ","
          << std::fixed << std::setprecision(4) << result.best_makespan_history[i] << ","
          << std::setprecision(4) << result.best_icc_history[i] << ","
          << std::setprecision(4) << result.best_rtu_pct_history[i] << ","
          << std::setprecision(4) << result.avg_makespan_history[i] << ","
          << result.feasible_count_history[i] << "\n";
    }
}

struct PipelineTimings {
    double clustering_seconds = 0.0;
    double stage1_seconds     = 0.0;
    double stage2_seconds     = 0.0;
    double total_seconds      = 0.0;
};

struct ScheduleSummary {
    bool feasible     = false;
    double makespan   = -1.0;
    double icc        = 0.0;
    double rtu        = 0.0;
    double rtu_pct    = 0.0;
};

inline ScheduleSummary writeScheduleResultAndReport(
    const std::string& json_path,
    const std::string& report_md_path,
    const std::string& dag_file,
    unsigned cluster_seed,
    const GAConfig& cfg,
    const InitialSampleResult& stage1,
    const GAResult& ga_result,
    const PipelineTimings& timings) {
    using nlohmann::json;

    std::ostringstream trace_stream;
    const double replayed_ms = runSimulationVerbose(ga_result.best, trace_stream);
    const bool is_feasible   = scheduling::feasible(replayed_ms);

    double icc_total = 0.0;
    std::vector<nested_dfx::CommCostPair> icc_edges;
    if (is_feasible) {
        auto [tot, pairs] = recomputeFinalCommCostDetailFromState();
        icc_total = tot;
        icc_edges = std::move(pairs);
    } else {
        icc_total = recomputeFinalCommCostFromState();
    }

    const double total_work = totalTaskWeightedWork();
    const double total_box  = is_feasible ? computeFinalResourceBox() : 0.0;
    const double rtu        = (is_feasible && total_box > 1e-9) ? (total_work / total_box) : 0.0;
    const double rtu_pct    = rtu * 100.0;

    int active_comms = 0;
    for (const auto& [cid, comm] : g_communities) {
        (void)cid;
        if (comm.activation_count > 0) ++active_comms;
    }

    std::string val_err;
    const bool layout_valid = is_feasible ? g_nested_alloc.validate(&val_err) : false;

    // Build schedule_result.json
    json j;
    j["schema"]            = "cps_schedule_result_v1";
    j["decoder_semantics"] = "explicit_nested_slots_v2_single_residency";
    j["dag_file"]          = dag_file;
    j["preset"]            = presetName(cfg.preset_params.preset);
    j["preset_parameters"] = {
        {"candidate_scoring", cfg.preset_params.enable_dmc_scoring ? "action_role_dmc" : "action_role"},
        {"new_parent_weight", cfg.preset_params.new_parent_weight},
        {"dmc_weight",        cfg.preset_params.dmc_weight},
        {"reseff_weight",     cfg.preset_params.reseff_weight},
        {"reseff_dmc_bucket", cfg.preset_params.reseff_dmc_bucket}
    };
    j["seeds"] = {
        {"cluster_seed", cluster_seed},
        {"ga_seed",      cfg.seed}
    };
    j["summary"] = {
        {"feasible",           is_feasible},
        {"makespan",           is_feasible ? json(replayed_ms) : json(nullptr)},
        {"icc",                is_feasible ? json(icc_total) : json(nullptr)},
        {"rtu",                is_feasible ? json(rtu) : json(nullptr)},
        {"rtu_pct",            is_feasible ? json(rtu_pct) : json(nullptr)},
        {"weighted_task_work", total_work},
        {"resource_box_sum",   is_feasible ? json(total_box) : json(nullptr)},
        {"total_tasks",        static_cast<int>(g_tasks.size())},
        {"total_edges",        static_cast<int>(g_edges.size())},
        {"total_communities",  static_cast<int>(g_communities.size())},
        {"active_communities", active_comms},
        {"layout_valid",       layout_valid},
        {"stop_reason",        ga_result.stop_reason}
    };
    j["runtime_seconds"] = {
        {"clustering",       timings.clustering_seconds},
        {"initial_sampling", timings.stage1_seconds},
        {"ga_evolution",     timings.stage2_seconds},
        {"total",            timings.total_seconds}
    };
    j["ga_statistics"] = {
        {"population_size",            cfg.population_size},
        {"max_generations",            cfg.max_generations},
        {"generations_run",            ga_result.generations_run},
        {"evaluations",                ga_result.evaluations},
        {"feasible_evaluations",       ga_result.feasible_evaluations},
        {"initialization_evaluations", ga_result.initialization_evaluations},
        {"restart_evaluations",        ga_result.restart_evaluations},
        {"offspring_evaluations",      ga_result.offspring_evaluations},
        {"initial_attempts",           stage1.attempts_used},
        {"initial_feasible_seeds",     static_cast<int>(stage1.solutions.size())},
        {"random_search_only",         cfg.random_search_only}
    };
    appendCommunityScoringMetadata(j, cfg.fixed_genes);
    j["best_chromosome"] = chromosomeGenesJSON(ga_result.best);

    // Per-community placement & execution metrics
    json j_comms = json::array();
    for (const auto& [cid, comm] : g_communities) {
        layout::CommunityResource demand{
            static_cast<double>(comm.CLB),
            static_cast<double>(comm.BRAM),
            static_cast<double>(comm.DSP)
        };
        const auto sclass = g_nested_alloc.classifyCommunity(demand);
        const double res  = cumulativeResidenceTime(comm);
        const double work = weightedTaskWork(cid, 2.5, 2.5);
        const auto drr    = lookupCommDrr(cid);

        json jc = {
            {"id",               cid},
            {"task_ids",         comm.task_ids},
            {"resources",        {{"CLB", comm.CLB}, {"BRAM", comm.BRAM}, {"DSP", comm.DSP}}},
            {"size_class",       nested_dfx::NestedDFXAllocator::sizeClassName(sclass)},
            {"activation_count", comm.activation_count},
            {"activation_time",  comm.activation_time},
            {"completion_time",  comm.completion_time},
            {"residence_time",   res}
        };
        if (drr.valid) {
            const double area = static_cast<double>(drr.drr_rect.w) * drr.drr_rect.h;
            const double box  = (res > 0.0) ? (res * area) : 0.0;
            const layout::RectCap cap = nested_dfx::rectCapacity(g_layout_fpga, drr.drr_rect);
            jc["placement"] = {
                {"mode",          nested_dfx::placementModeName(static_cast<nested_dfx::PlacementMode>(drr.mode_tag))},
                {"container_kind", nested_dfx::containerKindName(drr.kind)},
                {"parent_id",     drr.parent_id},
                {"slot_id",       drr.container_id},
                {"drr_rect",      {{"x", drr.drr_rect.x}, {"y", drr.drr_rect.y},
                                   {"w", drr.drr_rect.w}, {"h", drr.drr_rect.h}}},
                {"occupied_rect", {{"x", drr.occupied_rect.x}, {"y", drr.occupied_rect.y},
                                   {"w", drr.occupied_rect.w}, {"h", drr.occupied_rect.h}}},
                {"slot_capacity", {{"CLB", cap.clb}, {"BRAM", cap.bram}, {"DSP", cap.dsp}}},
                {"community_rtu_pct", (box > 1e-9) ? (work / box * 100.0) : 0.0}
            };
        }
        j_comms.push_back(jc);
    }
    j["communities"] = j_comms;

    // Per-task execution timeline
    json j_tasks = json::array();
    for (const auto& [tid, task] : g_tasks) {
        json jt = {
            {"id",           tid},
            {"community_id", task.community_id},
            {"exec_time",    task.exec_time},
            {"EST",          task.EST},
            {"LST",          task.LST},
            {"LFT",          task.LFT},
            {"slack",        task.slack},
            {"start_time",   task.start_time},
            {"finish_time",  task.finish_time}
        };
        if (g_nested_alloc.taskBindings().count(tid)) {
            const auto& b = g_nested_alloc.taskBindings().at(tid);
            jt["residence"] = {
                {"parent_id",      b.placement.parent_id},
                {"slot_id",        b.placement.container_id},
                {"layer_idx",      b.placement.layer_idx},
                {"layout_version", b.layout_version}
            };
        }
        j_tasks.push_back(jt);
    }
    j["tasks"] = j_tasks;

    // Per-edge ICC breakdown
    json j_edges = json::array();
    for (const auto& p : icc_edges) {
        j_edges.push_back({
            {"edge_id",                p.edge_id},
            {"source_task",            p.source_task},
            {"target_task",            p.target_task},
            {"comm_a",                 p.comm_a},
            {"comm_b",                 p.comm_b},
            {"weight",                 p.weight},
            {"parent_a",               p.parent_a},
            {"parent_b",               p.parent_b},
            {"slot_a",                 p.container_a},
            {"slot_b",                 p.container_b},
            {"version_a",              p.version_a},
            {"version_b",              p.version_b},
            {"time_overlap",           p.time_overlap},
            {"same_drr_reuse",         p.same_drr_reuse},
            {"final_full_cover_reuse", p.final_full_cover_reuse},
            {"relation",               p.relation},
            {"relation_cost",          p.relation_cost},
            {"icc_cost",               p.cost}
        });
    }
    j["communication_edges"] = {
        {"total_icc",  icc_total},
        {"edge_count", static_cast<int>(icc_edges.size())},
        {"edges",      j_edges}
    };

    if (is_feasible) {
        j["nested_dfx_slots"] = slotSemanticsJson(g_nested_alloc);
    }

    {
        std::ofstream fj(json_path);
        if (!fj.is_open())
            throw std::runtime_error("Cannot write schedule result JSON: " + json_path);
        fj << j.dump(2) << "\n";
    }

    // Write human-readable Markdown schedule report
    {
        std::ofstream md(report_md_path);
        if (!md.is_open())
            throw std::runtime_error("Cannot write schedule report Markdown: " + report_md_path);

        md << "# CPS Schedule Report\n\n";
        md << "## Summary\n\n";
        md << "| Metric | Value |\n";
        md << "|:-------|------:|\n";
        md << "| DAG Input | `" << dag_file << "` |\n";
        md << "| Preset | `" << presetName(cfg.preset_params.preset) << "` |\n";
        md << "| Seeds (`cluster_seed` / `ga_seed`) | `" << cluster_seed << "` / `" << cfg.seed << "` |\n";
        md << "| Feasible | " << (is_feasible ? "Yes" : "No") << " |\n";
        if (is_feasible) {
            md << "| **Makespan** | **" << std::fixed << std::setprecision(1) << replayed_ms << "** |\n";
            md << "| **ICC (Inter-Community Comm. Cost)** | **" << std::fixed << std::setprecision(1) << icc_total << "** |\n";
            md << "| **RTU (Resource-Time Utilization)** | **" << std::fixed << std::setprecision(2) << rtu_pct << "%** |\n";
        }
        md << "| Communities (Active / Total) | " << active_comms << " / " << g_communities.size() << " |\n";
        md << "| Tasks / Dependency Edges | " << g_tasks.size() << " / " << g_edges.size() << " |\n";
        md << "| GA Generations / Evaluations | " << ga_result.generations_run << " / " << ga_result.evaluations << " |\n";
        md << "| Stop Reason | `" << ga_result.stop_reason << "` |\n";
        md << "| Nested-DFX Invariant Check | " << (layout_valid ? "PASS" : (is_feasible ? ("FAIL: " + val_err) : "N/A")) << " |\n";
        md << "| Runtime (Cluster / Stage1 / Stage2 / Total) | "
           << std::fixed << std::setprecision(2)
           << timings.clustering_seconds << "s / "
           << timings.stage1_seconds << "s / "
           << timings.stage2_seconds << "s / "
           << timings.total_seconds << "s |\n\n";

        md << "## Community Execution & Nested-DFX Placement\n\n";
        md << "| Community | Size Class | Tasks | Resources (CLB/BRAM/DSP) | `[t_on, t_off)` | Mode | Parent | Slot | DRR Rect `(x,y,w,h)` | Comm RTU |\n";
        md << "|:---------:|:----------:|------:|:------------------------:|:---------------:|:-----|------:|-----:|:--------------------:|---------:|\n";
        for (const auto& [cid, comm] : g_communities) {
            layout::CommunityResource d{
                static_cast<double>(comm.CLB),
                static_cast<double>(comm.BRAM),
                static_cast<double>(comm.DSP)
            };
            const auto sclass = g_nested_alloc.classifyCommunity(d);
            const auto drr    = lookupCommDrr(cid);
            const double res  = cumulativeResidenceTime(comm);
            const double work = weightedTaskWork(cid, 2.5, 2.5);
            const double area = drr.valid ? (static_cast<double>(drr.drr_rect.w) * drr.drr_rect.h) : 0.0;
            const double crtu = (res > 0.0 && area > 0.0) ? (work / (res * area) * 100.0) : 0.0;

            md << "| C" << cid
               << " | " << nested_dfx::NestedDFXAllocator::sizeClassName(sclass)
               << " | " << comm.task_ids.size()
               << " | " << comm.CLB << "/" << comm.BRAM << "/" << comm.DSP << " | ";
            if (comm.activation_time >= 0.0) {
                md << "[" << std::fixed << std::setprecision(1) << comm.activation_time
                   << ", " << comm.completion_time << ")";
            } else {
                md << "-";
            }
            if (drr.valid) {
                md << " | `" << nested_dfx::placementModeName(static_cast<nested_dfx::PlacementMode>(drr.mode_tag)) << "`"
                   << " | P" << drr.parent_id
                   << " | S" << drr.container_id
                   << " | `(" << drr.drr_rect.x << "," << drr.drr_rect.y
                   << "," << drr.drr_rect.w << "," << drr.drr_rect.h << ")`"
                   << " | " << std::setprecision(2) << crtu << "% |\n";
            } else {
                md << " | - | - | - | - | - |\n";
            }
        }
        md << "\n" << trace_stream.str();
    }

    ScheduleSummary summary;
    summary.feasible = is_feasible;
    summary.makespan = replayed_ms;
    summary.icc      = icc_total;
    summary.rtu      = rtu;
    summary.rtu_pct  = rtu_pct;
    return summary;
}

} // namespace cps
