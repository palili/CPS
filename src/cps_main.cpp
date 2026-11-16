#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cps/clustering.hpp"
#include "cps/event_decoder.hpp"
#include "cps/ga_engine.hpp"
#include "cps/preset_config.hpp"

namespace {

struct CliOptions {
    std::string dag_file;
    std::string output_dir = "./cps_out";
    std::string communities_file;
    cps::Preset preset = cps::Preset::STANDARD;
    unsigned cluster_seed = 42;
    unsigned ga_seed      = 42;
    int max_generations   = 100;
    int max_stagnation    = 30;
    int evaluation_budget = 0;
    std::vector<double> fixed_genes;
    bool random_search = false;
    bool dump_debug    = false;
    bool quiet         = false;
    bool show_help     = false;
};

void printUsage(const char* prog) {
    std::cout
        << "CPS: Community-level Policy Scheduling for Nested-DFX FPGAs\n\n"
        << "Usage:\n"
        << "  " << prog << " --dag <dag.json> [options]\n\n"
        << "Core Options:\n"
        << "  --dag <path>               Input task DAG JSON file (required)\n"
        << "  --output-dir <dir>         Output directory (default: ./cps_out)\n"
        << "  --preset <name>            Scheduling preset (default: standard)\n"
        << "                               standard      : strict (Makespan, ICC, -RTU) + Eq.(16) slot cost\n"
        << "                               comm-aware    : inner incremental ICC guidance\n"
        << "                               packing-aware : inner area-time compactness + 3% ICC log-bucket\n"
        << "                               balanced      : joint ICC + compactness + 1% ICC log-bucket\n"
        << "  --cluster-seed <uint>      RNG seed for Stage-0 Louvain clustering (default: 42)\n"
        << "  --seed <uint>              RNG seed for Stage-1/Stage-2 GA search (default: 42)\n"
        << "  --communities <path>       Use precomputed clustering JSON instead of Stage 0\n\n"
        << "Advanced / Ablation Options:\n"
        << "  --max-generations <int>    Maximum GA generations (default: 100)\n"
        << "  --max-stagnation <int>     Early-stop stagnation generations (default: 30)\n"
        << "  --evaluation-budget <int>  Fixed decoder evaluation budget (0 = disabled)\n"
        << "  --fixed-genes <csv>        20 comma-separated values (-1 = active, [0,1] = fixed)\n"
        << "  --random-search            Disable GA operators; run pure random sampling baseline\n"
        << "  --dump-debug               Write clustering debug matrices to <output-dir>/debug\n"
        << "  --quiet                    Suppress intermediate stage logs\n"
        << "  -h, --help                 Show this help message\n";
}

std::vector<double> parseFixedGenesCsv(const std::string& text) {
    std::vector<double> out;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty())
            throw std::invalid_argument("Empty entry in --fixed-genes CSV");
        out.push_back(std::stod(token));
    }
    if (static_cast<int>(out.size()) != cps::CHROM_LEN)
        throw std::invalid_argument("--fixed-genes requires exactly 20 comma-separated numbers");
    return out;
}

CliOptions parseCli(int argc, char* argv[]) {
    CliOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireVal = [&](const char* flag) -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument(std::string("Missing value after ") + flag);
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            opt.show_help = true;
            return opt;
        } else if (arg == "--dag") {
            opt.dag_file = requireVal("--dag");
        } else if (arg == "--output-dir" || arg == "-o") {
            opt.output_dir = requireVal("--output-dir");
        } else if (arg == "--preset") {
            opt.preset = cps::parsePreset(requireVal("--preset"));
        } else if (arg == "--cluster-seed") {
            opt.cluster_seed = static_cast<unsigned>(std::stoul(requireVal("--cluster-seed")));
        } else if (arg == "--seed") {
            opt.ga_seed = static_cast<unsigned>(std::stoul(requireVal("--seed")));
        } else if (arg == "--communities") {
            opt.communities_file = requireVal("--communities");
        } else if (arg == "--max-generations") {
            opt.max_generations = std::stoi(requireVal("--max-generations"));
        } else if (arg == "--max-stagnation") {
            opt.max_stagnation = std::stoi(requireVal("--max-stagnation"));
        } else if (arg == "--evaluation-budget") {
            opt.evaluation_budget = std::stoi(requireVal("--evaluation-budget"));
        } else if (arg == "--fixed-genes") {
            opt.fixed_genes = parseFixedGenesCsv(requireVal("--fixed-genes"));
        } else if (arg == "--random-search") {
            opt.random_search = true;
        } else if (arg == "--dump-debug") {
            opt.dump_debug = true;
        } else if (arg == "--quiet") {
            opt.quiet = true;
        } else {
            throw std::invalid_argument("Unknown CLI argument: " + arg);
        }
    }
    if (!opt.show_help && opt.dag_file.empty())
        throw std::invalid_argument("Missing required --dag <dag.json> argument (use --help for usage)");
    return opt;
}

} // namespace

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    CliOptions opt;
    try {
        opt = parseCli(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
    if (opt.show_help) {
        printUsage(argv[0]);
        return 0;
    }

    try {
        const auto wall_t0 = std::chrono::high_resolution_clock::now();
        const fs::path out_dir(opt.output_dir);
        fs::create_directories(out_dir);

        cps::resetAllDecoderTemplates();
        cps::loadDAG(opt.dag_file);

        // Stage 0: Multi-level Resource-Constrained Task Clustering (or external JSON)
        double clustering_sec = 0.0;
        if (opt.communities_file.empty()) {
            const auto c_t0 = std::chrono::high_resolution_clock::now();
            const std::string cluster_out = (out_dir / "clustering_result.json").string();
            const std::string debug_dir   = (out_dir / "debug").string();
            if (opt.dump_debug) fs::create_directories(debug_dir);

            if (!opt.quiet) {
                std::cout << "[Stage 0] Running multi-level task clustering (cluster_seed="
                          << opt.cluster_seed << ")...\n";
            }
            const nlohmann::json cluster_json = cps::clustering::runClusteringPipeline(
                opt.dag_file,
                opt.cluster_seed,
                cluster_out,
                opt.dump_debug,
                debug_dir,
                !opt.quiet);
            cps::loadCommunitiesFromJson(cluster_json);
            const auto c_t1 = std::chrono::high_resolution_clock::now();
            clustering_sec = std::chrono::duration<double>(c_t1 - c_t0).count();
        } else {
            if (!opt.quiet) {
                std::cout << "[Stage 0] Loading precomputed communities from "
                          << opt.communities_file << "...\n";
            }
            cps::loadCommunities(opt.communities_file);
        }

        // Initialize Event-Driven Decoder & Nested-DFX Physical Slot Allocator
        const cps::PresetParameters preset_params = cps::makePresetParameters(opt.preset);
        cps::initDecoderWorkspace(preset_params);

        // Stage 1: Feasibility-Guided Initial Population Sampling
        const cps::InitialSampleResult stage1 = cps::sampleInitialPopulation(
            opt.ga_seed,
            /*target_solutions=*/40,
            /*max_attempts=*/20000,
            opt.fixed_genes,
            !opt.quiet);
        cps::writeInitialSolutionsJson(
            (out_dir / "initial_solutions.json").string(),
            opt.ga_seed,
            40,
            20000,
            preset_params,
            stage1,
            opt.fixed_genes);

        // Stage 2: Genetic Policy Evolution
        cps::GAConfig ga_cfg;
        ga_cfg.preset_params      = preset_params;
        ga_cfg.seed               = opt.ga_seed;
        ga_cfg.max_generations    = opt.max_generations;
        ga_cfg.convergence_gens   = opt.max_stagnation;
        ga_cfg.evaluation_budget  = opt.evaluation_budget;
        ga_cfg.fixed_genes        = opt.fixed_genes;
        ga_cfg.random_search_only = opt.random_search;

        const cps::GAResult ga_result = cps::runGA(ga_cfg, stage1.seeds, !opt.quiet);
        cps::writeConvergenceCsv((out_dir / "convergence.csv").string(), ga_result);

        const auto wall_t1 = std::chrono::high_resolution_clock::now();
        cps::PipelineTimings timings;
        timings.clustering_seconds = clustering_sec;
        timings.stage1_seconds     = stage1.elapsed_seconds;
        timings.stage2_seconds     = ga_result.elapsed_seconds;
        timings.total_seconds      = std::chrono::duration<double>(wall_t1 - wall_t0).count();

        const cps::ScheduleSummary summary = cps::writeScheduleResultAndReport(
            (out_dir / "schedule_result.json").string(),
            (out_dir / "schedule_report.md").string(),
            opt.dag_file,
            opt.cluster_seed,
            ga_cfg,
            stage1,
            ga_result,
            timings);

        if (summary.feasible) {
            std::cout << "[CPS Summary] FEASIBLE | Preset=" << cps::presetName(opt.preset)
                      << " | Makespan=" << std::fixed << std::setprecision(1) << summary.makespan
                      << " | ICC=" << std::setprecision(1) << summary.icc
                      << " | RTU=" << std::setprecision(2) << summary.rtu_pct << "%"
                      << " | Output=" << out_dir.string() << "\n";
            return 0;
        } else {
            std::cout << "[CPS Summary] INFEASIBLE under single-residency constraint | Preset="
                      << cps::presetName(opt.preset)
                      << " | Output=" << out_dir.string() << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
