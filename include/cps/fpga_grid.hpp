#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace layout {

constexpr double LAYOUT_EPS = 1e-9;

enum ColType : uint8_t { COL_CLB = 0, COL_BRAM = 1, COL_DSP = 2 };

// Heterogeneous column-based FPGA resource architecture (W = 146 cols, H = 350 CLB rows).
struct FPGAModel {
    int W = 146;
    int H = 350;
    double clb_cell_h  = 1.0;
    double bram_cell_h = 2.5;
    double dsp_cell_h  = 2.5;
    std::vector<ColType> col_type;

    std::vector<int> pre_clb;
    std::vector<int> pre_bram;
    std::vector<int> pre_dsp;

    static FPGAModel makeDefault() {
        FPGAModel m;
        m.W = 146;
        m.H = 350;
        m.clb_cell_h  = 1.0;
        m.bram_cell_h = 2.5;
        m.dsp_cell_h  = 2.5;
        m.col_type.assign(m.W, COL_CLB);
        const std::vector<int> bram_cols = {4,10,22,28,36,47,65,76,87,98,109,117,123,135,141};
        const std::vector<int> dsp_cols  = {13,19,25,33,39,44,50,62,68,73,79,84,90,95,101,106,112,120,126,132};
        for (int c : bram_cols) if (c >= 0 && c < m.W) m.col_type[c] = COL_BRAM;
        for (int c : dsp_cols)  if (c >= 0 && c < m.W) m.col_type[c] = COL_DSP;
        m.buildPrefix();
        return m;
    }

    void buildPrefix() {
        pre_clb.assign(W + 1, 0);
        pre_bram.assign(W + 1, 0);
        pre_dsp.assign(W + 1, 0);
        for (int i = 0; i < W; i++) {
            pre_clb[i + 1]  = pre_clb[i]  + (col_type[i] == COL_CLB  ? 1 : 0);
            pre_bram[i + 1] = pre_bram[i] + (col_type[i] == COL_BRAM ? 1 : 0);
            pre_dsp[i + 1]  = pre_dsp[i]  + (col_type[i] == COL_DSP  ? 1 : 0);
        }
    }

    inline int numCLB(int l, int r)  const { return pre_clb[r + 1]  - pre_clb[l]; }
    inline int numBRAM(int l, int r) const { return pre_bram[r + 1] - pre_bram[l]; }
    inline int numDSP(int l, int r)  const { return pre_dsp[r + 1]  - pre_dsp[l]; }
};

struct CommunityResource {
    double clb_req  = 0.0;
    double bram_req = 0.0;
    double dsp_req  = 0.0;
};

struct Rect {
    int x = 0, y = 0, w = 0, h = 0;
    bool valid() const { return w > 0 && h > 0; }
    inline int xr() const { return x + w; }
    inline int yt() const { return y + h; }
    inline int cx() const { return x + w / 2; }
    inline int cy() const { return y + h / 2; }

    static bool overlap(const Rect& a, const Rect& b) {
        if (a.xr() <= b.x || b.xr() <= a.x) return false;
        if (a.yt() <= b.y || b.yt() <= a.y) return false;
        return true;
    }

    bool withinFPGA(const FPGAModel& m) const {
        return x >= 0 && y >= 0 && xr() <= m.W && yt() <= m.H;
    }
};

struct PlacedComm {
    int    comm_id      = -1;
    Rect   rect;
    double t_on         = 0.0;
    double t_off        = -1.0;
    int    time_layer   = 0;
    int    drr_id       = 0;
    bool   still_active = false;
    double last_place_cost = 0.0;
    CommunityResource demand;
    bool   has_exact_demand = false;
};

struct TimeCoeffs {
    double alpha_d = 0.0;
    double beta_t  = 0.0;
};

struct SearchParams {
    double T_max = 10000.0;
    TimeCoeffs cc_overlap  { 1.0, 0.0 };
    TimeCoeffs cc_separate { 3.0, 1.5 };
    bool adaptive_T_max = true;
};

struct RectCap {
    double clb  = 0.0;
    double bram = 0.0;
    double dsp  = 0.0;
};

inline RectCap rectCapacity(const FPGAModel& m, int l, int r, int h) {
    RectCap c;
    c.clb  = static_cast<double>(m.numCLB(l, r))  * std::floor(static_cast<double>(h) / std::max(LAYOUT_EPS, m.clb_cell_h));
    c.bram = static_cast<double>(m.numBRAM(l, r)) * std::floor(static_cast<double>(h) / std::max(LAYOUT_EPS, m.bram_cell_h));
    c.dsp  = static_cast<double>(m.numDSP(l, r))  * std::floor(static_cast<double>(h) / std::max(LAYOUT_EPS, m.dsp_cell_h));
    return c;
}

} // namespace layout
