#pragma once

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

namespace cps {

inline constexpr int ALPHA_LEN = 7;
inline constexpr int BETA_LEN  = 11;
inline constexpr int CHROM_LEN = ALPHA_LEN + BETA_LEN + 2; // 20 genes in [0, 1]

// 20-dimensional real-valued policy chromosome:
//   alpha[0..6] : static community topological/resource feature weights (s1..s7)
//   beta[0..10] : dynamic runtime & reservation feature weights (d1..d11)
//   lambda      : convex static-vs-dynamic priority balance in [0, 1]
//   theta       : activation score gating threshold in [0, 1]
struct Chromosome {
    double alpha[ALPHA_LEN]{};
    double beta[BETA_LEN]{};
    double lambda = 0.0;
    double theta  = 0.0;

    double fitness          = 0.0;
    double makespan         = -1.0;
    double comm_cost_sum    = 0.0;
    double resource_box_sum = 0.0;
    double repair_fail_sum  = 0.0;

    std::vector<double> toVector() const {
        std::vector<double> v(CHROM_LEN);
        for (int i = 0; i < ALPHA_LEN; ++i) v[i] = alpha[i];
        for (int i = 0; i < BETA_LEN; ++i)  v[ALPHA_LEN + i] = beta[i];
        v[ALPHA_LEN + BETA_LEN]     = lambda;
        v[ALPHA_LEN + BETA_LEN + 1] = theta;
        return v;
    }

    void fromVector(const std::vector<double>& v) {
        if (static_cast<int>(v.size()) != CHROM_LEN)
            throw std::invalid_argument("Expected a 20-dimensional chromosome");
        for (double gene : v) {
            if (!std::isfinite(gene))
                throw std::invalid_argument("Chromosome genes must be finite");
        }
        for (int i = 0; i < ALPHA_LEN; ++i) alpha[i] = v[i];
        for (int i = 0; i < BETA_LEN; ++i)  beta[i]  = v[ALPHA_LEN + i];
        lambda = v[ALPHA_LEN + BETA_LEN];
        theta  = v[ALPHA_LEN + BETA_LEN + 1];
    }
};

inline std::string geneName(int index) {
    if (index < 0 || index >= CHROM_LEN)
        throw std::out_of_range("Chromosome gene index out of range");
    if (index < ALPHA_LEN) return "alpha" + std::to_string(index + 1);
    if (index < ALPHA_LEN + BETA_LEN) return "beta" + std::to_string(index - ALPHA_LEN + 1);
    return (index == CHROM_LEN - 2) ? "lambda" : "theta";
}

inline double fixedGeneValue(const std::vector<double>& fixed_genes, int index) {
    if (fixed_genes.empty()) return -1.0;
    if (static_cast<int>(fixed_genes.size()) != CHROM_LEN)
        throw std::invalid_argument("fixed_genes mask must be empty or have length 20");
    return fixed_genes[index];
}

inline Chromosome randomCommunityChromosome(
    std::mt19937& rng,
    const std::vector<double>& fixed_genes = {}) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> genes(CHROM_LEN);
    for (int i = 0; i < CHROM_LEN; ++i) {
        const double fg = fixedGeneValue(fixed_genes, i);
        genes[i] = (fg >= 0.0) ? fg : uniform(rng);
    }
    Chromosome chr;
    chr.fromVector(genes);
    chr.fitness  = 0.0;
    chr.makespan = -1.0;
    return chr;
}

inline double chromosomeGeneFromJSON(const nlohmann::json& value,
                                     const std::string& name) {
    if (!value.is_number())
        throw std::invalid_argument("Chromosome " + name + " must be numeric");
    const double gene = value.get<double>();
    if (!std::isfinite(gene) || gene < 0.0 || gene > 1.0)
        throw std::invalid_argument("Chromosome " + name + " must be finite and in [0, 1]");
    return gene;
}

inline Chromosome chromosomeFromJSON(const nlohmann::json& data) {
    if (!data.is_object() || !data.contains("alpha") || !data.contains("beta") ||
        !data.contains("lambda") || !data.contains("theta")) {
        throw std::invalid_argument("Chromosome JSON requires alpha, beta, lambda, and theta");
    }
    const auto& alpha = data.at("alpha");
    const auto& beta  = data.at("beta");
    if (!alpha.is_array() || static_cast<int>(alpha.size()) != ALPHA_LEN)
        throw std::invalid_argument("Chromosome alpha must contain 7 weights");
    if (!beta.is_array() || static_cast<int>(beta.size()) != BETA_LEN)
        throw std::invalid_argument("Chromosome beta must contain 11 weights");

    Chromosome result{};
    for (int i = 0; i < ALPHA_LEN; ++i)
        result.alpha[i] = chromosomeGeneFromJSON(alpha[i], "alpha[" + std::to_string(i) + "]");
    for (int i = 0; i < BETA_LEN; ++i)
        result.beta[i]  = chromosomeGeneFromJSON(beta[i], "beta[" + std::to_string(i) + "]");
    result.lambda = chromosomeGeneFromJSON(data.at("lambda"), "lambda");
    result.theta  = chromosomeGeneFromJSON(data.at("theta"), "theta");
    return result;
}

inline nlohmann::json chromosomeGenesJSON(const Chromosome& chr) {
    return {
        {"alpha", std::vector<double>(chr.alpha, chr.alpha + ALPHA_LEN)},
        {"beta",  std::vector<double>(chr.beta,  chr.beta  + BETA_LEN)},
        {"lambda", chr.lambda},
        {"theta",  chr.theta}
    };
}

inline void appendCommunityScoringMetadata(
    nlohmann::json& data,
    const std::vector<double>& fixed_genes = {}) {
    std::vector<std::string> active_names;
    nlohmann::json fixed_obj = nlohmann::json::object();
    for (int i = 0; i < CHROM_LEN; ++i) {
        const double fg = fixedGeneValue(fixed_genes, i);
        if (fg >= 0.0) {
            fixed_obj[geneName(i)] = fg;
        } else {
            active_names.push_back(geneName(i));
        }
    }
    data["community_scoring"]           = "weighted20_v1";
    data["chromosome_dimension"]        = CHROM_LEN;
    data["active_chromosome_dimension"] = active_names.size();
    data["active_gene_names"]           = active_names;
    data["fixed_coefficients"]          = fixed_obj;
    data["community_sort_order"]        = {"score_desc", "community_id_asc_on_exact_tie"};
    data["community_scoring_parameters"] = {
        {"formula", "lambda*Pri + (1-lambda)*(DynBase + beta8*d8 + beta9*d9 - beta10*d10 - beta11*d11)"},
        {"candidate_pool", "latent_ready_capacity_feasible_before_theta"},
        {"ranking", "score_desc_then_cid"},
        {"alpha_feature_names", {
            "cross_comm_volume", "cross_comm_edges", "mean_inv_slack",
            "internal_cp_length", "global_cp_ratio", "inv_entry_depth", "downstream_closure"
        }},
        {"beta_feature_names", {
            "ready_ratio", "unfinished_urgency", "latent_unlock", "own_resource_pressure",
            "active_unlock", "residence_efficiency", "external_wait_risk",
            "cp_ready_urgency", "ready_lst_urgency", "urgent_block_count", "urgent_block_pressure"
        }},
        {"beta_signs", {1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1}},
        {"threshold", "score > theta; system-idle fallback bypasses theta"}
    };
}

} // namespace cps
