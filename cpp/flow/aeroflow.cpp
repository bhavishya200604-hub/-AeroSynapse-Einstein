// aeroflow: 3D compressible flow solver for any AeroSynapse design.
//
// Finite volumes on a stretched Cartesian grid; HLLC fluxes with MUSCL (van Albada) reconstruction and
// Thornber's low-Mach correction of the reconstructed velocities;
// SSP Runge-Kutta 3; local time steps for steady states, a global step for time-accurate runs.
// The body enters through the geometry kernel's signed distance as an immersed boundary: ghost cells
// mirror an image point in the fluid (no flow through the wall, zero normal pressure gradient,
// adiabatic). Turbulence: WALE subgrid viscosity (large-eddy simulation, implicit on coarse grids) and an
// equilibrium wall model (Spalding's law) that removes wall shear stress from the first fluid cells, so
// friction drag and its effect on the flow come from the same solution. Optional inflow gust and
// synthetic inflow turbulence, sideslip, full or half model.
//
// Non-dimensional: free-stream density 1, speed of sound 1, temperature 1; lengths in metres.
//
// Limits (stated in every output): the grid is far too coarse to resolve boundary layers, so the wall
// model supplies friction from the law of the wall and separation from smooth surfaces is only
// approximate; HLLC can show a carbuncle on strong normal shocks; low Mach (< 0.2) is slow and diffusive.
//
//   aeroflow case.json            run a case (see python aerosynapse.cfd for the format)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <array>

#include "../common/par.hpp"
#include "../common/sdf.hpp"

using geo::V3;

static const double GAM = 1.4;
static const int G = 2;  // ghost layers at the domain edge

struct Axis {
    std::vector<double> f, c, d;  // faces (n+1), centres, widths
    int n = 0;
    void finish() { n = (int)f.size() - 1; c.resize((size_t)n); d.resize((size_t)n); for (int i = 0; i < n; ++i) { c[(size_t)i] = 0.5 * (f[(size_t)i] + f[(size_t)i + 1]); d[(size_t)i] = f[(size_t)i + 1] - f[(size_t)i]; } }
    // uniform spacing h0 on [a, b], geometric growth (ratio, capped at hmax) out to [lo, hi]
    void stretched(double lo, double a, double b, double hi, double h0, double ratio, double hmax) {
        int m = std::max(1, (int)std::ceil((b - a) / h0));
        double h = (b - a) / m;
        std::vector<double> right, left;
        for (int i = 0; i <= m; ++i) right.push_back(a + i * h);
        double s = h, x = b;
        while (x < hi - 1e-9) { s = std::min(s * ratio, hmax); x += s; right.push_back(x); }
        s = h; x = a;
        while (x > lo + 1e-9) { s = std::min(s * ratio, hmax); x -= s; left.push_back(x); }
        f.assign(left.rbegin(), left.rend());
        f.insert(f.end(), right.begin(), right.end());
        finish();
    }
    void uniform(double lo, double hi, int n0) { f.clear(); for (int i = 0; i <= n0; ++i) f.push_back(lo + (hi - lo) * i / n0); finish(); }
    int locate(double x) const {  // index of the cell centre at or below x
        if (x <= c[0]) return 0;
        if (x >= c[(size_t)n - 1]) return n - 1;
        return (int)(std::upper_bound(c.begin(), c.end(), x) - c.begin()) - 1;
    }
};

enum BC { FARFIELD, EXTRAP, PERIODIC, SYMMETRY };

struct Ghost {  // immersed-boundary ghost cell
    size_t cell;
    int nb[8];
    double w[8];
    double nx, ny, nz, dw, ell;  // outward normal, distance below the wall, image distance above it
    double tw_x = 0, tw_y = 0, tw_z = 0, cf = 0, cp = 0, Tw = 1, yplus = 0;
    double sum_cp = 0, sum_cp2 = 0, sum_cf = 0, sum_T = 0, sep = 0;
    double area = 0;  // wall area this ghost represents
    bool first = false;
};

struct Solver {
    // settings
    double mach = 0.5, alpha = 0, beta = 0, Re = 1e6, Lref = 1, Tinf_dim = 288.15, pinf_dim = 101325, cfl = 0.8;
    bool viscous = true, lts = true, wallmodel = true, sutherland = true, low_mach = true, thin_on = true;
    std::string sgs = "wale";
    double Cw = 0.5, Prt = 0.9, Pr = 0.72;
    double gust_amp = 0, gust_len = 10, tu = 0, tu_len = 1, tu_x0 = 0, tu_x1 = 0, tu_relax = 0.25;  // forcing zone of the inflow turbulence
    double Sref = 1, cref = 1, bref = 1;
    V3 mref;
    bool half = false;
    BC bc[3][2] = {{FARFIELD, FARFIELD}, {FARFIELD, FARFIELD}, {FARFIELD, FARFIELD}};
    std::string out = "aeroflow_out", kase = "body";
    // grid and state
    Axis ax[3];
    int nx = 0, ny = 0, nz = 0, NX = 0, NY = 0, NZ = 0;
    size_t N = 0;
    std::vector<double> U[5], W[5], R[5], U0[5], dt, mut, Sx, Sy, Sz;  // S*: wall-shear momentum sink
    std::vector<double> gU[9], gT[3];
    std::vector<unsigned char> tag;  // 0 fluid, 1 ghost, 2 solid
    size_t sealed = 0;  // cells made solid so that no wall lets air through between two cell centres
    // face walls: bit a of fw[c] blocks the face between cell c and its +a neighbour (both fluid); the body passes
    // between their centres. fwl lists them for the forces, with whether the face lies across the surface (loaded)
    bool facewall = false;
    std::vector<unsigned char> fw;
    struct FaceWall { size_t c; int a; bool load; };
    std::vector<FaceWall> fwl;
    std::vector<Ghost> ghosts;
    std::vector<size_t> fluid;
    // thin surfaces (wings, fins, blades thinner than a cell): one-cell layers treated as two-sided walls
    std::vector<size_t> thin;
    std::vector<int> thin_id;  // per cell: index into thin / thin_n, or -1
    std::vector<V3> thin_n;
    double thin_fric[6] = {0, 0, 0, 0, 0, 0};  // friction force and moment on thin surfaces (latest wall-model pass)
    geo::Model model;
    bool has_body = false;
    double Winf[5] = {1, 0, 0, 0, 1 / GAM}, mu_inf = 0, time = 0;
    double t_avg0 = -1; long n_avg = 0;
    std::vector<float> mean[4], rms[4];  // u v w p
    // gust / turbulence modes
    struct Mode { double kx, ky, kz, ax, ay, az, ph; };
    std::vector<Mode> modes;

    size_t id(int i, int j, int k) const { return ((size_t)(k + G) * NY + (j + G)) * NX + (i + G); }

    double mu_lam(double T) const {
        if (!sutherland) return mu_inf;
        double S = 110.4 / Tinf_dim;
        return mu_inf * std::pow(T, 1.5) * (1 + S) / (T + S);
    }

    void prim(size_t c) {
        double r = std::max(U[0][c], 1e-8), u = U[1][c] / r, v = U[2][c] / r, w = U[3][c] / r;
        double p = (GAM - 1) * (U[4][c] - 0.5 * r * (u * u + v * v + w * w));
        W[0][c] = r; W[1][c] = u; W[2][c] = v; W[3][c] = w; W[4][c] = std::max(p, 1e-8);
    }
    void cons_from(const double* q, size_t c) {
        U[0][c] = q[0]; U[1][c] = q[0] * q[1]; U[2][c] = q[0] * q[2]; U[3][c] = q[0] * q[3];
        U[4][c] = q[4] / (GAM - 1) + 0.5 * q[0] * (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (int v = 0; v < 5; ++v) W[v][c] = q[v];
    }

    void allocate() {
        nx = ax[0].n; ny = ax[1].n; nz = ax[2].n;
        NX = nx + 2 * G; NY = ny + 2 * G; NZ = nz + 2 * G;
        N = (size_t)NX * NY * NZ;
        for (int v = 0; v < 5; ++v) { U[v].assign(N, 0); W[v].assign(N, 0); R[v].assign(N, 0); U0[v].assign(N, 0); }
        dt.assign(N, 0); mut.assign(N, 0); tag.assign(N, 0); fw.assign(N, 0); Sx.assign(N, 0); Sy.assign(N, 0); Sz.assign(N, 0);
        if (viscous) { for (auto& g : gU) g.assign(N, 0); for (auto& g : gT) g.assign(N, 0); }
    }

    // ---------------------------------------------------------------- immersed boundary
    void build_body() {
        facewall = getenv("AEROFLOW_NOFACEWALL") == nullptr;  // face walls are the default; the variable restores snapped sheet cells
        std::vector<double> sd(N, 1e9);
        par::parallel_for(nz, [&](int k) {
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) sd[id(i, j, k)] = model.eval(V3(ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]));
        }, 1);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) tag[id(i, j, k)] = sd[id(i, j, k)] < 0 ? 2 : 0;
        // surfaces thinner than a cell: a cell whose sub-samples (4 x 4 x 4) touch the body joins the solid when no
        // face neighbour is solid, so a thin wing never has holes
        std::vector<unsigned char> sheet(N, 0);
        par::parallel_for(nz, [&](int k) {
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    size_t c = id(i, j, k);
                    double hx = ax[0].d[(size_t)i], hy = ax[1].d[(size_t)j], hz = ax[2].d[(size_t)k];
                    if (facewall || sd[c] < 0 || sd[c] > 0.9 * std::max(hx, std::max(hy, hz))) continue;
                    V3 c0(ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]);
                    // a surface within half a cell of the centre crosses the cell (along that axis, for a sheet lying along
                    // the grid): sheets thinner than the sub-sample spacing (a quarter cell) were missed entirely
                    if (sd[c] <= 0.5 * std::min(hx, std::min(hy, hz))) sheet[c] = 1;
                    for (int q = 0; q < 64 && !sheet[c]; ++q) {
                        V3 p(c0.x + (((q & 3) + 0.5) / 4 - 0.5) * hx, c0.y + ((((q >> 2) & 3) + 0.5) / 4 - 0.5) * hy, c0.z + ((((q >> 4) & 3) + 0.5) / 4 - 0.5) * hz);
                        if (model.eval(p) < 0) sheet[c] = 1;
                    }
                }
        }, 1);

        auto inside = [&](int I, int J, int K) { return I >= 0 && J >= 0 && K >= 0 && I < nx && J < ny && K < nz; };
        // a sheet straddling a cell face marks both cells: keep the one whose centre is nearer the sheet, so every
        // sheet is one cell thick and gets the two-sided treatment
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    size_t c = id(i, j, k);
                    if (!sheet[c]) continue;
                    // only across the sheet: the axis along which the distance changes most
                    V3 p0(ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]);
                    double f0 = model.eval(p0), bv = -1;
                    int a = 0;
                    for (int b = 0; b < 3; ++b) {
                        double h = ax[b].d[b == 0 ? (size_t)i : b == 1 ? (size_t)j : (size_t)k];
                        V3 e; e[b] = h;
                        double v = std::fabs(model.eval(p0 + e) - f0) + std::fabs(model.eval(p0 - e) - f0);
                        if (v > bv) { bv = v; a = b; }
                    }
                    for (int sg = -1; sg <= 1; sg += 2) {
                        int I = i + (a == 0) * sg, J = j + (a == 1) * sg, K = k + (a == 2) * sg;
                        if (!inside(I, J, K)) continue;
                        size_t o = id(I, J, K);
                        if (sheet[o] && (sd[o] < sd[c] || (sd[o] == sd[c] && o < c))) { sheet[c] = 0; break; }
                    }
                }
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    size_t c = id(i, j, k);
                    if (!sheet[c]) continue;
                    // only where the body is thin: no cell centre of the 26 neighbours lies inside it (otherwise the
                    // cell is a corner of a thick body, which the ghost cells already represent)
                    bool solid_nb = false;
                    for (int dk = -1; dk <= 1 && !solid_nb; ++dk)
                        for (int dj = -1; dj <= 1 && !solid_nb; ++dj)
                            for (int di = -1; di <= 1; ++di) {
                                int I = i + di, J = j + dj, K = k + dk;
                                if (inside(I, J, K) && sd[id(I, J, K)] < 0) { solid_nb = true; break; }
                            }
                    if (!solid_nb) tag[c] = 2;
                }
        // watertight walls: where the body passes between the centres of two neighbouring fluid cells, air could flow
        // through it (a slot between the thick middle of a thin section and its sheet-like leading or trailing edge,
        // where the rules above keep neither cell). The cell nearer the body joins the solid. The distance is
        // 1-Lipschitz, so a crossing needs sd(c) + sd(c') <= h, and only those pairs are sampled.
        if (!getenv("AEROFLOW_NOSEAL")) {
            for (int pass = 0; pass < 3; ++pass) {
                size_t changed = 0;
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i) {
                            size_t c = id(i, j, k);
                            if (tag[c] != 0) continue;
                            for (int b = 0; b < 3 && tag[c] == 0; ++b) {
                                int I = i + (b == 0), J = j + (b == 1), K = k + (b == 2);
                                if (!inside(I, J, K)) continue;
                                size_t o = id(I, J, K);
                                if (tag[o] != 0) continue;
                                V3 p0(ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]), p1(ax[0].c[(size_t)I], ax[1].c[(size_t)J], ax[2].c[(size_t)K]);
                                double h = geo::len(p1 - p0);
                                if (sd[c] + sd[o] > h) continue;
                                bool cross = false;
                                for (int q = 1; q < 16 && !cross; ++q) if (model.eval(p0 + (p1 - p0) * (q / 16.0)) < 0) cross = true;
                                if (!cross) continue;
                                if (facewall) {
                                    if (fw[c] >> b & 1) continue;
                                    fw[c] |= (unsigned char)(1u << b);
                                    // loaded when the face lies across the surface: the distance changes most along b
                                    // (one-sided changes from the face centre; a central gradient vanishes inside a sheet)
                                    V3 pm = (p0 + p1) * 0.5;
                                    double f0 = model.eval(pm), best = -1;
                                    int ab = b;
                                    for (int a2 = 0; a2 < 3; ++a2) {
                                        V3 e; e[a2] = 0.5 * h;
                                        double v = std::fabs(model.eval(pm + e) - f0) + std::fabs(model.eval(pm - e) - f0);
                                        if (v > best) { best = v; ab = a2; }
                                    }
                                    fwl.push_back({c, b, ab == b});
                                    ++sealed;
                                    continue;
                                }
                                tag[sd[c] <= sd[o] ? c : o] = 2;
                                ++sealed; ++changed;
                            }
                        }
                if (!changed) break;
            }
        }
        // a solid cell with fluid on both sides along some axis is a one-cell layer: a two-sided thin wall
        if (getenv("AEROFLOW_DEBUG")) {
            size_t ns = 0, nsheet = 0, n2z = 0;
            for (int k = 1; k + 1 < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
                size_t c = id(i, j, k);
                if (sheet[c]) ++nsheet;
                if (tag[c] != 2) continue;
                ++ns;
                if (tag[id(i, j, k - 1)] == 0 && tag[id(i, j, k + 1)] == 0) ++n2z;
            }
            fprintf(stderr, "solid %u sheet-candidates %u fluid-above-and-below %u\n", (unsigned)ns, (unsigned)nsheet, (unsigned)n2z);
        }
        thin_id.assign(N, -1);
        std::vector<unsigned char> t0 = tag;
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    size_t c = id(i, j, k);
                    if (t0[c] != 2) continue;
                    if (!thin_on) continue;
                    V3 p(ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]);
                    // a sheet has fluid on both sides along an axis AND the distance rises in both directions across
                    // it (a central gradient vanishes on a sheet's mid-plane, so one-sided rises are used); a one-cell
                    // bump on a thick surface has fluid fore and aft but the distance hardly changes along that axis
                    double f0 = model.eval(p);
                    int axis = -1;
                    double best = 0.3;  // about 1 across a sheet (0.5 when it lies on a cell face), near 0 along a smooth surface
                    for (int a = 0; a < 3; ++a) {
                        int I0 = i - (a == 0), J0 = j - (a == 1), K0 = k - (a == 2), I1 = i + (a == 0), J1 = j + (a == 1), K1 = k + (a == 2);
                        if (!(inside(I0, J0, K0) && inside(I1, J1, K1) && t0[id(I0, J0, K0)] == 0 && t0[id(I1, J1, K1)] == 0)) continue;
                        double h = ax[a].d[a == 0 ? (size_t)i : a == 1 ? (size_t)j : (size_t)k];
                        V3 e; e[a] = h;
                        double rise = 0.5 * (std::fabs(model.eval(p + e) - f0) + std::fabs(model.eval(p - e) - f0)) / h;  // about 1 across a sheet, near 0 along a surface
                        if (rise > best) { best = rise; axis = a; }
                    }
                    if (axis < 0) continue;
                    // a bump on a thick body has the body itself next to it (a neighbour deep inside); a sheet does not
                    bool thick_nb = false;  // checked only for borderline cases: a clear sheet changes distance by about a cell
                    for (int a = 0; a < 3 && !thick_nb && best < 0.6; ++a) {
                        if (a == axis) continue;
                        for (int sg = -1; sg <= 1; sg += 2) {
                            int I = i + (a == 0) * sg, J = j + (a == 1) * sg, K = k + (a == 2) * sg;
                            double h = ax[a].d[a == 0 ? (size_t)i : a == 1 ? (size_t)j : (size_t)k];
                            if (inside(I, J, K) && sd[id(I, J, K)] < -0.5 * h) thick_nb = true;
                        }
                    }
                    if (thick_nb) continue;
                    V3 nn; nn[axis] = 1.0;
                    tag[c] = 3;
                    thin_id[c] = (int)thin.size();
                    thin.push_back(c);
                    thin_n.push_back(nn);
                    if (getenv("AEROFLOW_DEBUG")) fprintf(stderr, "thin cell i%d j%d k%d at (%.3f %.3f %.3f) sd %.4f n (%.2f %.2f %.2f)\n", i, j, k, p.x, p.y, p.z, sd[c], thin_n.back().x, thin_n.back().y, thin_n.back().z);
                }
        // solid cells within two cells of the fluid along an axis become ghosts
        int off[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    size_t c = id(i, j, k);
                    if (tag[c] != 2) continue;
                    bool first = false, any = false;
                    for (auto& o : off)
                        for (int s = 1; s <= 2; ++s) {
                            int I = i + o[0] * s, J = j + o[1] * s, K = k + o[2] * s;
                            if (I < 0 || J < 0 || K < 0 || I >= nx || J >= ny || K >= nz) continue;
                            if (tag[id(I, J, K)] == 0) { any = true; if (s == 1) first = true; }
                        }
                    if (!any) continue;
                    Ghost g; g.cell = c; g.first = first;
                    V3 p(ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]);
                    double hloc = std::min(ax[0].d[(size_t)i], std::min(ax[1].d[(size_t)j], ax[2].d[(size_t)k]));
                    V3 n = model.grad(p, 0.25 * hloc);
                    double ln = geo::len(n);
                    n = ln > 1e-12 ? n / ln : V3(0, 0, 1);
                    g.nx = n.x; g.ny = n.y; g.nz = n.z;
                    g.dw = std::fabs(sd[c]) / std::max(ln, 0.3);
                    double hmax = std::max(ax[0].d[(size_t)i], std::max(ax[1].d[(size_t)j], ax[2].d[(size_t)k]));
                    g.ell = std::max(g.dw, 1.2 * hmax);
                    ghosts.push_back(g);
                }
        for (auto& g : ghosts) tag[g.cell] = 1;
        for (auto& g : ghosts) {  // image stencils once every ghost is tagged, so they only ever use fluid cells
            size_t c = g.cell;
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            V3 p(ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]);
            interp_stencil(p + V3(g.nx, g.ny, g.nz) * (g.dw + g.ell), g);
        }
        if (const char* tp = getenv("AEROFLOW_TAGDUMP")) {  // cell map for diagnosis: nx ny nz, axis centres, then tags
            if (FILE* fp = fopen(tp, "wb")) {
                int d3[3] = {nx, ny, nz};
                fwrite(d3, sizeof(int), 3, fp);
                for (int a = 0; a < 3; ++a) fwrite(ax[a].c.data(), sizeof(double), (size_t)ax[a].n, fp);
                for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) { unsigned char t = tag[id(i, j, k)]; fwrite(&t, 1, 1, fp); }
                fclose(fp);
            }
        }
        // wall area: Cartesian faces between a first-layer ghost and the fluid, projected on the normal
        for (auto& g : ghosts) {
            size_t c = g.cell;
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            double nn[3] = {g.nx, g.ny, g.nz};
            for (int a = 0; a < 3; ++a)
                for (int s = -1; s <= 1; s += 2) {
                    int I = i + (a == 0) * s, J = j + (a == 1) * s, K = k + (a == 2) * s;
                    if (I < 0 || J < 0 || K < 0 || I >= nx || J >= ny || K >= nz || tag[id(I, J, K)] != 0) continue;
                    double A = (a == 0 ? ax[1].d[(size_t)j] * ax[2].d[(size_t)k] : a == 1 ? ax[0].d[(size_t)i] * ax[2].d[(size_t)k] : ax[0].d[(size_t)i] * ax[1].d[(size_t)j]);
                    g.area += A * std::fabs(nn[a]);
                }
        }
    }

    void interp_stencil(const V3& pi, Ghost& g) {
        int i0 = ax[0].locate(pi.x), j0 = ax[1].locate(pi.y), k0 = ax[2].locate(pi.z);
        i0 = std::min(i0, nx - 2); j0 = std::min(j0, ny - 2); k0 = std::min(k0, nz - 2);
        double tx = geo::clampd((pi.x - ax[0].c[(size_t)i0]) / (ax[0].c[(size_t)i0 + 1] - ax[0].c[(size_t)i0]), 0, 1);
        double ty = geo::clampd((pi.y - ax[1].c[(size_t)j0]) / (ax[1].c[(size_t)j0 + 1] - ax[1].c[(size_t)j0]), 0, 1);
        double tz = geo::clampd((pi.z - ax[2].c[(size_t)k0]) / (ax[2].c[(size_t)k0 + 1] - ax[2].c[(size_t)k0]), 0, 1);
        double ws = 0;
        for (int q = 0; q < 8; ++q) {
            int I = i0 + (q & 1), J = j0 + (q >> 1 & 1), K = k0 + (q >> 2 & 1);
            size_t c = id(I, J, K);
            double w = ((q & 1) ? tx : 1 - tx) * ((q >> 1 & 1) ? ty : 1 - ty) * ((q >> 2 & 1) ? tz : 1 - tz);
            if (tag[c] != 0) w = 0;  // only fluid cells feed the image point
            g.nb[q] = (int)c; g.w[q] = w; ws += w;
        }
        if (ws < 1e-6) {  // no fluid corner: use the nearest fluid neighbour of the ghost
            size_t c = g.cell; size_t best = c; double bd = 1e300;
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            for (int dk = -2; dk <= 2; ++dk) for (int dj = -2; dj <= 2; ++dj) for (int di = -2; di <= 2; ++di) {
                int I = i + di, J = j + dj, K = k + dk;
                if (I < 0 || J < 0 || K < 0 || I >= nx || J >= ny || K >= nz) continue;
                size_t cc = id(I, J, K);
                if (tag[cc] != 0) continue;
                double d2 = di * di + dj * dj + dk * dk + 0.01 * (dk * dk);
                if (d2 < bd) { bd = d2; best = cc; }
            }
            for (int q = 0; q < 8; ++q) { g.nb[q] = (int)best; g.w[q] = q == 0 ? 1 : 0; }
            ws = 1;
        }
        for (int q = 0; q < 8; ++q) g.w[q] /= ws;
    }

    // ghost states from their image points; the wall model fills the shear sink of the fluid cells
    void thin_states() {  // thin cells carry the tangential average of their fluid neighbours (for gradients)
        par::parallel_for((int)thin.size(), [&](int q) {
            size_t c = thin[(size_t)q];
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            double s[5] = {0, 0, 0, 0, 0};
            int n = 0;
            for (int a = 0; a < 3; ++a)
                for (int sg = -1; sg <= 1; sg += 2) {
                    int I = i + (a == 0) * sg, J = j + (a == 1) * sg, K = k + (a == 2) * sg;
                    if (I < 0 || J < 0 || K < 0 || I >= nx || J >= ny || K >= nz) continue;
                    size_t f = id(I, J, K);
                    if (tag[f] != 0) continue;
                    for (int v = 0; v < 5; ++v) s[v] += W[v][f];
                    ++n;
                }
            if (!n) return;
            for (double& v : s) v /= n;
            V3 nn = thin_n[(size_t)q];
            double un = s[1] * nn.x + s[2] * nn.y + s[3] * nn.z;
            s[1] -= un * nn.x; s[2] -= un * nn.y; s[3] -= un * nn.z;
            cons_from(s, c);
        });
    }

    void apply_ib(bool update_wall) {
        thin_states();
        par::parallel_for((int)ghosts.size(), [&](int q) {
            Ghost& g = ghosts[(size_t)q];
            double s[5] = {0, 0, 0, 0, 0};
            for (int m = 0; m < 8; ++m) { if (g.w[m] == 0) continue; size_t c = (size_t)g.nb[m]; for (int v = 0; v < 5; ++v) s[v] += g.w[m] * W[v][c]; }
            double vn = s[1] * g.nx + s[2] * g.ny + s[3] * g.nz;
            double ut[3] = {s[1] - vn * g.nx, s[2] - vn * g.ny, s[3] - vn * g.nz};
            double f = g.dw / g.ell;
            double gq[5];
            gq[1] = ut[0] - f * vn * g.nx; gq[2] = ut[1] - f * vn * g.ny; gq[3] = ut[2] - f * vn * g.nz;
            gq[4] = s[4]; gq[0] = s[0];
            cons_from(gq, g.cell);
            if (update_wall) {
                double T = GAM * s[4] / s[0];
                g.cp = (s[4] - 1 / GAM) / (0.5 * mach * mach);
                double Ut = std::sqrt(ut[0] * ut[0] + ut[1] * ut[1] + ut[2] * ut[2]);
                g.Tw = T + std::cbrt(Pr) * Ut * Ut * (GAM - 1) / 2;  // turbulent recovery temperature at the wall
                if (wallmodel && Ut > 1e-9) {
                    double nu = mu_lam(T) / s[0], y = g.dw + g.ell;
                    double ut_ = spalding(Ut, y, nu);
                    double tw = s[0] * ut_ * ut_;
                    g.tw_x = tw * ut[0] / Ut; g.tw_y = tw * ut[1] / Ut; g.tw_z = tw * ut[2] / Ut;
                    g.cf = tw / (0.5 * mach * mach);
                    g.yplus = y * ut_ / nu;
                } else { g.tw_x = g.tw_y = g.tw_z = g.cf = 0; }
                g.sep = (ut[0] * std::cos(alpha) + ut[2] * std::sin(alpha)) < 0 ? 1 : 0;  // tangential flow against the stream
            }
        });
    }

    static double spalding(double U, double y, double nu) {
        const double k = 0.41, B = 5.2, e = std::exp(-k * B);
        double ut = std::max(std::sqrt(nu * U / y), 1e-12);  // laminar guess
        double ul = U / (std::log(std::max(y * U / nu, 1.0)) / k + B);  // log-law guess
        if (y * ut / nu > 11) ut = std::max(ul, 1e-12);
        for (int it = 0; it < 30; ++it) {
            double up = U / ut, ku = k * up;
            double yp = up + e * (std::exp(ku) - 1 - ku - ku * ku / 2 - ku * ku * ku / 6);
            double F = yp - y * ut / nu;
            double dyp = 1 + e * k * (std::exp(ku) - 1 - ku - ku * ku / 2);  // d yp / d up
            double dF = dyp * (-U / (ut * ut)) - y / nu;
            double st = F / dF;
            ut = std::max(ut - st, 0.2 * ut);
            if (std::fabs(st) < 1e-10 * ut) break;
        }
        return ut;
    }

    void wall_sink() {
        std::fill(Sx.begin(), Sx.end(), 0.0); std::fill(Sy.begin(), Sy.end(), 0.0); std::fill(Sz.begin(), Sz.end(), 0.0);
        for (double& v : thin_fric) v = 0;
        if (!wallmodel) return;
        // the wall model samples the velocity one cell further out (at 1.5 cells, like the ghost cells' image points at
        // 1.2 cells or more): sampled in the first cell, which the wall shear itself slows, the shear and the velocity
        // ran down together (flat plate: 0.00208 at step 10, 0.00055 settled against Schoenherr's 0.00211). This is
        // the log-layer mismatch of Kawai & Larsson (2012). The first cell keeps receiving the shear.
        auto sample = [&](int I, int J, int K, int a, int dir, size_t& fs, double& y) {
            double h1 = a == 0 ? ax[0].d[(size_t)I] : a == 1 ? ax[1].d[(size_t)J] : ax[2].d[(size_t)K];
            fs = id(I, J, K); y = 0.5 * h1;
            int I2 = I + (a == 0) * dir, J2 = J + (a == 1) * dir, K2 = K + (a == 2) * dir;
            if (I2 < 0 || J2 < 0 || K2 < 0 || I2 >= nx || J2 >= ny || K2 >= nz) return;
            size_t f2 = id(I2, J2, K2);
            if (tag[f2] != 0) return;
            fs = f2; y = h1 + 0.5 * (a == 0 ? ax[0].d[(size_t)I2] : a == 1 ? ax[1].d[(size_t)J2] : ax[2].d[(size_t)K2]);
        };
        for (size_t c : thin) {  // both sides of every thin surface
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            for (int a = 0; a < 3; ++a)
                for (int sg = -1; sg <= 1; sg += 2) {
                    int I = i + (a == 0) * sg, J = j + (a == 1) * sg, K = k + (a == 2) * sg;
                    if (I < 0 || J < 0 || K < 0 || I >= nx || J >= ny || K >= nz) continue;
                    size_t f = id(I, J, K);
                    if (tag[f] != 0) continue;
                    V3 n = thin_normal(c, a, sg);
                    size_t fs; double y;
                    sample(I, J, K, a, sg, fs, y);
                    double u0 = W[1][fs], u1 = W[2][fs], u2 = W[3][fs], un = u0 * n.x + u1 * n.y + u2 * n.z;
                    double ut[3] = {u0 - un * n.x, u1 - un * n.y, u2 - un * n.z};
                    double Ut = std::sqrt(ut[0] * ut[0] + ut[1] * ut[1] + ut[2] * ut[2]);
                    if (Ut < 1e-9) continue;
                    double T = GAM * W[4][fs] / W[0][fs], nu = mu_lam(T) / W[0][fs];
                    double utau = spalding(Ut, y, nu), tw = W[0][f] * utau * utau;
                    double A = (a == 0 ? ax[1].d[(size_t)j] * ax[2].d[(size_t)k] : a == 1 ? ax[0].d[(size_t)i] * ax[2].d[(size_t)k] : ax[0].d[(size_t)i] * ax[1].d[(size_t)j]) * std::fabs(n[a]);
                    double V = ax[0].d[(size_t)I] * ax[1].d[(size_t)J] * ax[2].d[(size_t)K];
                    double fx = tw * ut[0] / Ut * A, fy = tw * ut[1] / Ut * A, fz = tw * ut[2] / Ut * A;
                    Sx[f] -= fx / V; Sy[f] -= fy / V; Sz[f] -= fz / V;
                    double rx = ax[0].c[(size_t)i] - mref.x, ry = ax[1].c[(size_t)j] - mref.y, rz = ax[2].c[(size_t)k] - mref.z;
                    thin_fric[0] += fx; thin_fric[1] += fy; thin_fric[2] += fz;
                    thin_fric[3] += ry * fz - rz * fy; thin_fric[4] += rz * fx - rx * fz; thin_fric[5] += rx * fy - ry * fx;
                }
        }
        for (const FaceWall& e : fwl) {  // both sides of every loaded face wall, as for a thin cell
            if (!e.load) continue;
            int a = e.a;
            int k = (int)(e.c / ((size_t)NX * NY)) - G, j = (int)((e.c / NX) % NY) - G, i = (int)(e.c % NX) - G;
            double A = a == 0 ? ax[1].d[(size_t)j] * ax[2].d[(size_t)k] : a == 1 ? ax[0].d[(size_t)i] * ax[2].d[(size_t)k] : ax[0].d[(size_t)i] * ax[1].d[(size_t)j];
            double P[3] = {ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]};
            P[a] += 0.5 * ax[a].d[a == 0 ? (size_t)i : a == 1 ? (size_t)j : (size_t)k];
            for (int side = 0; side < 2; ++side) {
                int I = i + (side && a == 0), J = j + (side && a == 1), K = k + (side && a == 2);
                size_t f = id(I, J, K), fs; double y;
                sample(I, J, K, a, side ? 1 : -1, fs, y);
                double ut[3] = {W[1][fs], W[2][fs], W[3][fs]};
                ut[a] = 0;  // the wall normal is the face axis
                double Ut = std::sqrt(ut[0] * ut[0] + ut[1] * ut[1] + ut[2] * ut[2]);
                if (Ut < 1e-9) continue;
                double T = GAM * W[4][fs] / W[0][fs], nu = mu_lam(T) / W[0][fs];
                double utau = spalding(Ut, y, nu), tw = W[0][f] * utau * utau;
                double V = ax[0].d[(size_t)I] * ax[1].d[(size_t)J] * ax[2].d[(size_t)K];
                double fx = tw * ut[0] / Ut * A, fy = tw * ut[1] / Ut * A, fz = tw * ut[2] / Ut * A;
                Sx[f] -= fx / V; Sy[f] -= fy / V; Sz[f] -= fz / V;
                double rx = P[0] - mref.x, ry = P[1] - mref.y, rz = P[2] - mref.z;
                thin_fric[0] += fx; thin_fric[1] += fy; thin_fric[2] += fz;
                thin_fric[3] += ry * fz - rz * fy; thin_fric[4] += rz * fx - rx * fz; thin_fric[5] += rx * fy - ry * fx;
            }
        }
        for (auto& g : ghosts) {
            if (!g.first || g.area <= 0) continue;
            size_t c = g.cell;
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            // share the wall force among the fluid face neighbours by their projected face areas
            double share[6]; size_t cells[6]; int n = 0; double tot = 0; double nn[3] = {g.nx, g.ny, g.nz};
            for (int a = 0; a < 3; ++a)
                for (int s = -1; s <= 1; s += 2) {
                    int I = i + (a == 0) * s, J = j + (a == 1) * s, K = k + (a == 2) * s;
                    if (I < 0 || J < 0 || K < 0 || I >= nx || J >= ny || K >= nz) continue;
                    size_t cc = id(I, J, K);
                    if (tag[cc] != 0) continue;
                    double A = (a == 0 ? ax[1].d[(size_t)j] * ax[2].d[(size_t)k] : a == 1 ? ax[0].d[(size_t)i] * ax[2].d[(size_t)k] : ax[0].d[(size_t)i] * ax[1].d[(size_t)j]) * std::fabs(nn[a]);
                    cells[n] = cc; share[n] = A; tot += A; ++n;
                }
            for (int m = 0; m < n; ++m) {
                size_t cc = cells[m];
                int K = (int)(cc / ((size_t)NX * NY)) - G, J = (int)((cc / NX) % NY) - G, I = (int)(cc % NX) - G;
                double V = ax[0].d[(size_t)I] * ax[1].d[(size_t)J] * ax[2].d[(size_t)K];
                double F = g.area * share[m] / tot / V;
                Sx[cc] -= g.tw_x * F; Sy[cc] -= g.tw_y * F; Sz[cc] -= g.tw_z * F;
            }
        }
    }

    // ---------------------------------------------------------------- boundaries of the domain
    void free_stream_at(double x, double y, double z, double* q) const {
        for (int v = 0; v < 5; ++v) q[v] = Winf[v];
        double U = mach;
        if (gust_amp != 0) { double ph = 2 * geo::PI * (x - U * time) / gust_len; q[3] += gust_amp * U * std::sin(-ph); }
        for (auto& m : modes) {
            double ph = m.kx * (x - U * time) + m.ky * y + m.kz * z + m.ph, s = std::cos(ph);
            q[1] += m.ax * s; q[2] += m.ay * s; q[3] += m.az * s;
        }
    }
    void boundaries() {
        // fill the G edge layers on each side; edges and corners take values filled by earlier axes
        for (int a = 0; a < 3; ++a) {
            int o1 = a == 0 ? 1 : 0, o2 = a == 2 ? 1 : 2;
            int n0 = a == 0 ? nx : a == 1 ? ny : nz, n1 = o1 == 0 ? nx : ny, n2 = o2 == 1 ? ny : nz;
            for (int side = 0; side < 2; ++side) {
                BC b = bc[a][side];
                par::parallel_for(n2 + 2 * G, [&](int p2) {
                    int q2 = p2 - G;
                    for (int q1 = -G; q1 < n1 + G; ++q1)
                        for (int l = 1; l <= G; ++l) {
                            auto cell = [&](int along) { int x[3]; x[a] = along; x[o1] = q1; x[o2] = q2; return id(x[0], x[1], x[2]); };
                            size_t g = cell(side ? n0 - 1 + l : -l);
                            if (b == PERIODIC) { size_t s = cell(side ? l - 1 : n0 - l); for (int v = 0; v < 5; ++v) { U[v][g] = U[v][s]; W[v][g] = W[v][s]; } continue; }
                            if (b == EXTRAP) { size_t s = cell(side ? n0 - 1 : 0); for (int v = 0; v < 5; ++v) { U[v][g] = U[v][s]; W[v][g] = W[v][s]; } continue; }
                            if (b == SYMMETRY) {
                                size_t s = cell(side ? n0 - l : l - 1); double q[5]; for (int v = 0; v < 5; ++v) q[v] = W[v][s];
                                q[1 + a] = -q[1 + a]; cons_from(q, g); continue;
                            }
                            // far field: one-dimensional characteristics normal to the boundary
                            size_t s = cell(side ? n0 - 1 : 0);
                            double qi[5]; for (int v = 0; v < 5; ++v) qi[v] = W[v][s];
                            int x[3]; x[a] = side ? n0 - 1 : 0; x[o1] = std::max(0, std::min(n1 - 1, q1)); x[o2] = std::max(0, std::min(n2 - 1, q2));
                            double qe[5]; free_stream_at(ax[0].c[(size_t)x[0]], ax[1].c[(size_t)x[1]], ax[2].c[(size_t)x[2]], qe);
                            double sn = side ? 1.0 : -1.0;  // outward normal along a
                            double uni = qi[1 + a] * sn, une = qe[1 + a] * sn;
                            double ci = std::sqrt(GAM * qi[4] / qi[0]), ce = std::sqrt(GAM * qe[4] / qe[0]);
                            double q[5];
                            if (une <= -ce) { for (int v = 0; v < 5; ++v) q[v] = qe[v]; }        // supersonic inflow
                            else if (uni >= ci) { for (int v = 0; v < 5; ++v) q[v] = qi[v]; }    // supersonic outflow
                            else {
                                double Rp = uni + 2 * ci / (GAM - 1), Rm = une - 2 * ce / (GAM - 1);
                                double un = 0.5 * (Rp + Rm), c = 0.25 * (GAM - 1) * (Rp - Rm);
                                const double* up = un > 0 ? qi : qe;  // entropy and tangential velocity from upstream
                                double sent = up[4] / std::pow(up[0], GAM);
                                double r = std::pow(c * c / (GAM * sent), 1 / (GAM - 1));
                                for (int v = 1; v <= 3; ++v) q[v] = up[v];
                                q[1 + a] = un * sn; q[0] = r; q[4] = r * c * c / GAM;
                            }
                            cons_from(q, g);
                        }
                });
            }
        }
    }

    // normal of thin cell c oriented into the fluid on side `side` (+1 or -1) along axis a
    V3 thin_normal(size_t c, int a, int side) const {
        V3 n = thin_n[(size_t)thin_id[c]];
        if (n[a] * side < 0) n = n * -1.0;
        if (std::fabs(n[a]) < 0.3) { n = V3(0, 0, 0); n[a] = side; }
        return n;
    }
    static void reflect(const double* q, const V3& n, double* r) {
        double un = q[1] * n.x + q[2] * n.y + q[3] * n.z;
        r[0] = q[0]; r[4] = q[4];
        r[1] = q[1] - 2 * un * n.x; r[2] = q[2] - 2 * un * n.y; r[3] = q[3] - 2 * un * n.z;
    }
    // flux through the face between fluid cell f and thin cell t along axis a (fluid on the left if fluid_left)
    void thin_flux(size_t f, size_t t, int a, bool fluid_left, double* F) const {
        double q[5], r[5];
        for (int v = 0; v < 5; ++v) q[v] = W[v][f];
        V3 n; n[a] = 1.0;  // reflect about the face itself: no mass crosses a wall face (the surface normal sets only the shear direction)
        (void)t;
        reflect(q, n, r);
        if (fluid_left) hllc(q, r, a, F); else hllc(r, q, a, F);
    }

    // ---------------------------------------------------------------- fluxes
    static double vanalbada(double a, double b) {
        if (a * b <= 0) return 0;
        const double e = 1e-12;
        return (a * (b * b + e) + b * (a * a + e)) / (a * a + b * b + 2 * e);
    }
    static void hllc(const double* L, const double* Rr, int d, double* F) {
        double rL = L[0], pL = L[4], rR = Rr[0], pR = Rr[4];
        double uL = L[1 + d], uR = Rr[1 + d];
        double cL = std::sqrt(GAM * pL / rL), cR = std::sqrt(GAM * pR / rR);
        double EL = pL / (GAM - 1) + 0.5 * rL * (L[1] * L[1] + L[2] * L[2] + L[3] * L[3]);
        double ER = pR / (GAM - 1) + 0.5 * rR * (Rr[1] * Rr[1] + Rr[2] * Rr[2] + Rr[3] * Rr[3]);
        double sl = std::min(uL - cL, uR - cR), sr = std::max(uL + cL, uR + cR);
        double ss = (pR - pL + rL * uL * (sl - uL) - rR * uR * (sr - uR)) / (rL * (sl - uL) - rR * (sr - uR));
        auto flux = [&](const double* Q, double r, double un, double E, double* f) {
            f[0] = r * un; f[1] = r * Q[1] * un; f[2] = r * Q[2] * un; f[3] = r * Q[3] * un; f[1 + d] += Q[4]; f[4] = (E + Q[4]) * un;
        };
        if (sl >= 0) { flux(L, rL, uL, EL, F); return; }
        if (sr <= 0) { flux(Rr, rR, uR, ER, F); return; }
        const double* Q = ss >= 0 ? L : Rr;
        double r = ss >= 0 ? rL : rR, un = ss >= 0 ? uL : uR, E = ss >= 0 ? EL : ER, s = ss >= 0 ? sl : sr;
        double f[5]; flux(Q, r, un, E, f);
        double k = r * (s - un) / (s - ss);
        double Us[5] = {k, k * Q[1], k * Q[2], k * Q[3], k * (E / r + (ss - un) * (ss + Q[4] / (r * (s - un))))};
        Us[1 + d] = k * ss;
        double Uk[5] = {r, r * Q[1], r * Q[2], r * Q[3], E};
        for (int v = 0; v < 5; ++v) F[v] = f[v] + s * (Us[v] - Uk[v]);
    }

    void gradients() {
        par::parallel_for(nz, [&](int k) {
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    size_t c = id(i, j, k);
                    if (tag[c] != 0) continue;
                    size_t n[6] = {id(i + 1, j, k), id(i - 1, j, k), id(i, j + 1, k), id(i, j - 1, k), id(i, j, k + 1), id(i, j, k - 1)};
                    double dx[3] = {
                        (i + 1 < nx ? ax[0].c[(size_t)i + 1] : ax[0].c[(size_t)i] + ax[0].d[(size_t)i]) - (i > 0 ? ax[0].c[(size_t)i - 1] : ax[0].c[(size_t)i] - ax[0].d[(size_t)i]),
                        (j + 1 < ny ? ax[1].c[(size_t)j + 1] : ax[1].c[(size_t)j] + ax[1].d[(size_t)j]) - (j > 0 ? ax[1].c[(size_t)j - 1] : ax[1].c[(size_t)j] - ax[1].d[(size_t)j]),
                        (k + 1 < nz ? ax[2].c[(size_t)k + 1] : ax[2].c[(size_t)k] + ax[2].d[(size_t)k]) - (k > 0 ? ax[2].c[(size_t)k - 1] : ax[2].c[(size_t)k] - ax[2].d[(size_t)k])};
                    for (int a = 0; a < 3; ++a) {
                        size_t p = n[2 * a], m = n[2 * a + 1];
                        for (int v = 0; v < 3; ++v) gU[3 * v + a][c] = (W[1 + v][p] - W[1 + v][m]) / dx[a];
                        gT[a][c] = (W[4][p] / W[0][p] - W[4][m] / W[0][m]) * GAM / dx[a];
                    }
                    // WALE subgrid viscosity
                    if (sgs == "none") { mut[c] = 0; continue; }
                    double g[3][3];
                    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) g[a][b] = gU[3 * a + b][c];
                    double D = std::cbrt(ax[0].d[(size_t)i] * ax[1].d[(size_t)j] * ax[2].d[(size_t)k]);
                    double SS = 0;
                    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) { double s = 0.5 * (g[a][b] + g[b][a]); SS += s * s; }
                    if (sgs == "smagorinsky") { mut[c] = W[0][c] * std::pow(0.17 * D, 2) * std::sqrt(2 * SS); continue; }
                    double g2[3][3];
                    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) { g2[a][b] = 0; for (int m = 0; m < 3; ++m) g2[a][b] += g[a][m] * g[m][b]; }
                    double tr = (g2[0][0] + g2[1][1] + g2[2][2]) / 3, SD = 0;
                    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) { double s = 0.5 * (g2[a][b] + g2[b][a]) - (a == b ? tr : 0); SD += s * s; }
                    double den = std::pow(SS, 2.5) + std::pow(SD, 1.25);
                    mut[c] = den > 1e-30 ? W[0][c] * std::pow(Cw * D, 2) * std::pow(SD, 1.5) / den : 0;
                }
        }, 1);
    }

    // residual R = -(div F) + sources, for every fluid cell, and the local time step
    void residual(bool want_dt) {
        if (viscous) gradients();
        for (int v = 0; v < 5; ++v) std::fill(R[v].begin(), R[v].end(), 0.0);
        for (int a = 0; a < 3; ++a) sweep(a);
        par::parallel_for(nz, [&](int k) {
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    size_t c = id(i, j, k);
                    if (tag[c] != 0) continue;
                    R[1][c] += Sx[c]; R[2][c] += Sy[c]; R[3][c] += Sz[c];
                    if (!want_dt) continue;
                    double cs = std::sqrt(GAM * W[4][c] / W[0][c]), lam = 0, vis = 0;
                    double mu = (viscous ? mu_lam(GAM * W[4][c] / W[0][c]) + mut[c] : 0) / W[0][c];
                    double h[3] = {ax[0].d[(size_t)i], ax[1].d[(size_t)j], ax[2].d[(size_t)k]};
                    for (int a = 0; a < 3; ++a) { lam += (std::fabs(W[1 + a][c]) + cs) / h[a]; vis += 2 * mu * GAM / Pr / (h[a] * h[a]); }
                    dt[c] = cfl / (lam + vis);
                }
        }, 1);
    }

    void sweep(int a) {
        // faces normal to axis a; rows are processed in parallel so no two threads touch one cell
        int na = a == 0 ? nx : a == 1 ? ny : nz;
        int o1 = a == 0 ? 1 : 0, o2 = a == 2 ? 1 : 2;  // the two other axes
        int n1 = o1 == 0 ? nx : ny, n2 = o2 == 1 ? ny : nz;
        const long long st = a == 0 ? 1 : a == 1 ? (long long)NX : (long long)NX * NY;
        par::parallel_for(n2, [&](int q2) {
            double wl[5], wr[5], F[5];
            for (int q1 = 0; q1 < n1; ++q1)
                for (int s = 0; s <= na; ++s) {  // face between cells s-1 and s
                    int idx[3]; idx[o1] = q1; idx[o2] = q2; idx[a] = 0;
                    const long long base = (long long)id(idx[0], idx[1], idx[2]);
                    auto cell = [&](int t) { return (size_t)(base + (long long)t * st); };
                    size_t cL = cell(s - 1), cR = cell(s);
                    bool inL = s - 1 >= 0 && tag[cL] == 0, inR = s < na && tag[cR] == 0;
                    if (!inL && !inR) continue;
                    if (inL && inR && (fw[cL] >> a & 1)) {  // a face wall: each side sees its own reflection
                        thin_flux(cL, cL, a, true, F);
                        { double r = 1.0 / ax[a].d[(size_t)s - 1]; for (int v = 0; v < 5; ++v) R[v][cL] -= F[v] * r; }
                        thin_flux(cR, cR, a, false, F);
                        { double r = 1.0 / ax[a].d[(size_t)s]; for (int v = 0; v < 5; ++v) R[v][cR] += F[v] * r; }
                        continue;
                    }
                    if ((inL && s < na && tag[cR] == 3) || (inR && s - 1 >= 0 && tag[cL] == 3)) {  // a thin wall: each side sees its own reflection
                        thin_flux(inL ? cL : cR, inL ? cR : cL, a, inL, F);
                        if (inL) { double r = 1.0 / ax[a].d[(size_t)s - 1]; for (int v = 0; v < 5; ++v) R[v][cL] -= F[v] * r; }
                        else { double r = 1.0 / ax[a].d[(size_t)s]; for (int v = 0; v < 5; ++v) R[v][cR] += F[v] * r; }
                        continue;
                    }
                    size_t cLL = cell(s - 2), cRR = cell(s + 1);
                    bool order1 = tag[cLL] == 3 || tag[cRR] == 3;
                    bool o1L = order1 || (fw[cLL] >> a & 1), o1R = order1 || (fw[cR] >> a & 1);  // no slope across a face wall
                    for (int v = 0; v < 5; ++v) {
                        double qLL = W[v][cLL], qL = W[v][cL], qR = W[v][cR], qRR = W[v][cRR];
                        wl[v] = o1L ? qL : qL + 0.5 * vanalbada(qL - qLL, qR - qL);
                        wr[v] = o1R ? qR : qR - 0.5 * vanalbada(qR - qL, qRR - qR);
                    }
                    if (wl[0] <= 0 || wl[4] <= 0 || wr[0] <= 0 || wr[4] <= 0) for (int v = 0; v < 5; ++v) { wl[v] = W[v][cL]; wr[v] = W[v][cR]; }
                    if (low_mach) {  // Thornber et al. (2008): scale the velocity jump by the local Mach number
                        double qL = std::sqrt(wl[1] * wl[1] + wl[2] * wl[2] + wl[3] * wl[3]) / std::sqrt(GAM * wl[4] / wl[0]);
                        double qR = std::sqrt(wr[1] * wr[1] + wr[2] * wr[2] + wr[3] * wr[3]) / std::sqrt(GAM * wr[4] / wr[0]);
                        double z = std::min(1.0, std::max(qL, qR));
                        for (int v = 1; v <= 3; ++v) { double m = 0.5 * (wl[v] + wr[v]), d = 0.5 * (wl[v] - wr[v]); wl[v] = m + z * d; wr[v] = m - z * d; }
                    }
                    hllc(wl, wr, a, F);
                    if (viscous && inL && inR) visc_flux(cL, cR, a, s, idx, F);
                    // flux per unit area divided by the cell width = flux per unit volume
                    if (inL) { double r = 1.0 / ax[a].d[(size_t)s - 1]; for (int v = 0; v < 5; ++v) R[v][cL] -= F[v] * r; }
                    if (inR) { double r = 1.0 / ax[a].d[(size_t)s]; for (int v = 0; v < 5; ++v) R[v][cR] += F[v] * r; }
                }
        }, 1);
    }

    void visc_flux(size_t cL, size_t cR, int a, int s, const int* idx, double* F) {
        double dn = ax[a].c[(size_t)s] - ax[a].c[(size_t)s - 1];
        double TL = GAM * W[4][cL] / W[0][cL], TR = GAM * W[4][cR] / W[0][cR];
        double mu = 0.5 * (mu_lam(TL) + mu_lam(TR)), mt = 0.5 * (mut[cL] + mut[cR]);
        double me = mu + mt, k = (mu / Pr + mt / Prt) / (GAM - 1);
        double g[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) g[i][j] = 0.5 * (gU[3 * i + j][cL] + gU[3 * i + j][cR]);
        for (int i = 0; i < 3; ++i) g[i][a] = (W[1 + i][cR] - W[1 + i][cL]) / dn;  // normal derivatives taken directly
        double div = g[0][0] + g[1][1] + g[2][2];
        double tau[3];
        for (int i = 0; i < 3; ++i) tau[i] = me * (g[i][a] + g[a][i]) - (i == a ? 2.0 / 3.0 * me * div : 0);
        double u[3] = {0.5 * (W[1][cL] + W[1][cR]), 0.5 * (W[2][cL] + W[2][cR]), 0.5 * (W[3][cL] + W[3][cR])};
        double q = -k * (TR - TL) / dn;
        for (int i = 0; i < 3; ++i) F[1 + i] -= tau[i];
        F[4] -= u[0] * tau[0] + u[1] * tau[1] + u[2] * tau[2] - q;
        (void)idx;
    }

    // ---------------------------------------------------------------- forces
    struct Forces { double fx = 0, fy = 0, fz = 0, mx = 0, my = 0, mz = 0, fxv = 0, fzv = 0, fyv = 0; };
    Forces forces() {
        Forces f;
        for (int a = 0; a < 3; ++a) {
            // pressure and momentum flux through every fluid-ghost face, from the cell states
            for (auto& g : ghosts) {
                if (!g.first) continue;
                size_t c = g.cell;
                int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
                for (int s = -1; s <= 1; s += 2) {
                    int I = i + (a == 0) * s, J = j + (a == 1) * s, K = k + (a == 2) * s;
                    if (I < 0 || J < 0 || K < 0 || I >= nx || J >= ny || K >= nz) continue;
                    size_t cf = id(I, J, K);
                    if (tag[cf] != 0) continue;
                    // wall pressure (from the image point) on the closed staircase: a uniform pressure sums to zero
                    double A = (a == 0 ? ax[1].d[(size_t)j] * ax[2].d[(size_t)k] : a == 1 ? ax[0].d[(size_t)i] * ax[2].d[(size_t)k] : ax[0].d[(size_t)i] * ax[1].d[(size_t)j]);
                    double sg = s > 0 ? -1.0 : 1.0;  // fluid on the + side pushes the body towards -a
                    double pw = W[4][c] - 1 / GAM;
                    double F[3] = {0, 0, 0}; F[a] = sg * pw * A;
                    double fx = F[0], fy = F[1], fz = F[2];
                    double P[3] = {ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]};
                    add(f, fx, fy, fz, P, false);
                }
            }
        }
        for (auto& g : ghosts) {
            if (!g.first || g.area <= 0) continue;
            size_t c = g.cell;
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            double P[3] = {ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]};
            add(f, g.tw_x * g.area, g.tw_y * g.area, g.tw_z * g.area, P, true);
        }
        for (size_t c : thin) {  // momentum flux into thin walls from each side
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            double P[3] = {ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]};
            const V3& sn = thin_n[(size_t)thin_id[c]];
            for (int a = 0; a < 3; ++a) {
                // only the faces across the sheet carry its load; the in-plane faces (leading and trailing edges a
                // whole cell high) are an artefact of the cell size, and a real sheet's edges carry almost no force
                if (std::fabs(sn[a]) < 0.5) continue;
                for (int sg = -1; sg <= 1; sg += 2) {
                    int I = i + (a == 0) * sg, J = j + (a == 1) * sg, K = k + (a == 2) * sg;
                    if (I < 0 || J < 0 || K < 0 || I >= nx || J >= ny || K >= nz) continue;
                    size_t fcell = id(I, J, K);
                    if (tag[fcell] != 0) continue;
                    double F[5];
                    thin_flux(fcell, c, a, sg < 0, F);
                    double A = (a == 0 ? ax[1].d[(size_t)j] * ax[2].d[(size_t)k] : a == 1 ? ax[0].d[(size_t)i] * ax[2].d[(size_t)k] : ax[0].d[(size_t)i] * ax[1].d[(size_t)j]);
                    double s2 = sg < 0 ? 1.0 : -1.0;  // fluid on the left pushes the wall towards +a
                    double Fm[3] = {F[1], F[2], F[3]};
                    Fm[a] -= 1 / GAM;  // free-stream pressure: sums to zero on closed surfaces, subtracted for accuracy
                    add(f, s2 * Fm[0] * A, s2 * Fm[1] * A, s2 * Fm[2] * A, P, false);
                }
            }
        }
        for (const FaceWall& e : fwl) {
            if (!e.load) continue;
            size_t c = e.c;
            int a = e.a;
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            size_t o = id(i + (a == 0), j + (a == 1), k + (a == 2));
            double A = (a == 0 ? ax[1].d[(size_t)j] * ax[2].d[(size_t)k] : a == 1 ? ax[0].d[(size_t)i] * ax[2].d[(size_t)k] : ax[0].d[(size_t)i] * ax[1].d[(size_t)j]);
            double P[3] = {ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k]};
            P[a] += 0.5 * ax[a].d[a == 0 ? (size_t)i : a == 1 ? (size_t)j : (size_t)k];
            double F[5];
            thin_flux(c, c, a, true, F);  // fluid on the left pushes the wall towards +a
            double Fm[3] = {F[1], F[2], F[3]};
            Fm[a] -= 1 / GAM;
            add(f, Fm[0] * A, Fm[1] * A, Fm[2] * A, P, false);
            thin_flux(o, o, a, false, F);  // fluid on the right pushes it towards -a
            double Gm[3] = {F[1], F[2], F[3]};
            Gm[a] -= 1 / GAM;
            add(f, -Gm[0] * A, -Gm[1] * A, -Gm[2] * A, P, false);
        }
        f.fx += thin_fric[0]; f.fy += thin_fric[1]; f.fz += thin_fric[2]; f.mx += thin_fric[3]; f.my += thin_fric[4]; f.mz += thin_fric[5];
        f.fxv += thin_fric[0]; f.fyv += thin_fric[1]; f.fzv += thin_fric[2];
        if (half) { f.fx *= 2; f.fz *= 2; f.fy = 0; f.my *= 2; f.mx = 0; f.mz = 0; f.fxv *= 2; f.fzv *= 2; f.fyv = 0; }
        return f;
    }
    void add(Forces& f, double fx, double fy, double fz, const double* P, bool visc) {
        f.fx += fx; f.fy += fy; f.fz += fz;
        if (visc) { f.fxv += fx; f.fyv += fy; f.fzv += fz; }
        double rx = P[0] - mref.x, ry = P[1] - mref.y, rz = P[2] - mref.z;
        f.mx += ry * fz - rz * fy; f.my += rz * fx - rx * fz; f.mz += rx * fy - ry * fx;
    }
    void coefficients(const Forces& f, double* C) {  // CL CD CY Cl Cm Cn CDf CLf
        double q = 0.5 * mach * mach * Sref;
        double ca = std::cos(alpha), sa = std::sin(alpha), cb = std::cos(beta), sb = std::sin(beta);
        // wind axes: drag along the free stream, lift normal to it in the x-z plane
        double D = (f.fx * ca + f.fz * sa) * cb - f.fy * sb, L = -f.fx * sa + f.fz * ca, Y = (f.fx * ca + f.fz * sa) * sb + f.fy * cb;
        double Dv = (f.fxv * ca + f.fzv * sa) * cb - f.fyv * sb, Lv = -f.fxv * sa + f.fzv * ca;  // friction parts
        C[0] = L / q; C[1] = D / q; C[2] = Y / q; C[3] = -f.mx / (q * bref); C[4] = f.my / (q * cref); C[5] = -f.mz / (q * bref); C[6] = Dv / q; C[7] = Lv / q;  // x aft, z up: standard roll and yaw signs flip
    }

    // ---------------------------------------------------------------- time stepping
    double step(bool unsteady) {
        double dtg = 0;
        for (size_t c : fluid) for (int v = 0; v < 5; ++v) U0[v][c] = U[v][c];
        const double a1[3] = {0, 0.75, 1.0 / 3.0}, b1[3] = {1, 0.25, 2.0 / 3.0};
        double res = 0;
        for (int st = 0; st < 3; ++st) {
            boundaries();
            apply_ib(st == 0);
            if (st == 0) wall_sink();
            residual(st == 0);
            if (st == 0) {
                if (unsteady) { double m = 1e300; for (size_t c : fluid) m = std::min(m, dt[c]); dtg = m; for (size_t c : fluid) dt[c] = m; }
            }
            par::parallel_for((int)fluid.size(), [&](int q) {
                size_t c = fluid[(size_t)q];
                for (int v = 0; v < 5; ++v) U[v][c] = a1[st] * U0[v][c] + b1[st] * (U[v][c] + dt[c] * R[v][c]);
                prim(c);
            });
        }
        for (size_t c : fluid) { double d = (U[0][c] - U0[0][c]) / dt[c]; res += d * d; }
        if (unsteady) time += dtg;
        return std::sqrt(res / std::max<size_t>(fluid.size(), 1));
    }

    void init_free_stream() {
        double a = alpha, b = beta;
        Winf[0] = 1; Winf[1] = mach * std::cos(a) * std::cos(b); Winf[2] = -mach * std::sin(b); Winf[3] = mach * std::sin(a) * std::cos(b); Winf[4] = 1 / GAM;
        for (int k = -G; k < nz + G; ++k)
            for (int j = -G; j < ny + G; ++j)
                for (int i = -G; i < nx + G; ++i) cons_from(Winf, id(i, j, k));
        if (tu > 0) {  // synthetic frozen turbulence: random Fourier modes with a von Karman-like spread
            srand(12345);
            int nm = 24;
            double amp = tu * mach * std::sqrt(2.0 / nm);
            for (int m = 0; m < nm; ++m) {
                double kk = 2 * geo::PI / tu_len * std::pow(4.0, (double)m / nm);
                double th = std::acos(2.0 * rand() / RAND_MAX - 1), ph = 2 * geo::PI * rand() / RAND_MAX;
                V3 kv(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th));
                V3 r(2.0 * rand() / RAND_MAX - 1, 2.0 * rand() / RAND_MAX - 1, 2.0 * rand() / RAND_MAX - 1);
                V3 av = geo::cross(kv, r); double l = geo::len(av); av = l > 0 ? av / l : V3(0, 1, 0);  // divergence free
                double e = std::pow(kk * tu_len / (2 * geo::PI), -5.0 / 6.0);
                modes.push_back({kk * kv.x, kk * kv.y, kk * kv.z, amp * e * av.x, amp * e * av.y, amp * e * av.z, 2 * geo::PI * rand() / RAND_MAX});
            }
            double s = 0; for (auto& m : modes) s += 0.5 * (m.ax * m.ax + m.ay * m.ay + m.az * m.az);
            double fct = s > 0 ? tu * mach / std::sqrt(2 * s / 3) : 1;  // match the requested intensity
            for (auto& m : modes) { m.ax *= fct; m.ay *= fct; m.az *= fct; }
            // seed the whole field with the turbulence, so it does not have to cross the coarse outer grid first
            for (int k = -G; k < nz + G; ++k)
                for (int j = -G; j < ny + G; ++j)
                    for (int i = -G; i < nx + G; ++i) {
                        double q[5];
                        free_stream_at(ax[0].c[(size_t)std::max(0, std::min(nx - 1, i))], ax[1].c[(size_t)std::max(0, std::min(ny - 1, j))], ax[2].c[(size_t)std::max(0, std::min(nz - 1, k))], q);
                        cons_from(q, id(i, j, k));
                    }
        }
    }

    // keep the inflow turbulence alive: relax the velocity towards the synthetic field in a slab upstream of the body,
    // where the grid is fine enough to carry the eddies (the far-field boundary alone feeds them through metre-sized cells)
    void force_turbulence(double dt_frac) {
        if (tu <= 0 || tu_x1 <= tu_x0) return;
        int i0 = ax[0].locate(tu_x0), i1 = ax[0].locate(tu_x1);
        double rl = std::min(1.0, tu_relax * dt_frac);
        par::parallel_for(nz, [&](int k) {
            for (int j = 0; j < ny; ++j)
                for (int i = i0; i <= i1; ++i) {
                    size_t c = id(i, j, k);
                    if (tag[c]) continue;
                    double q[5], t[5];
                    for (int v = 0; v < 5; ++v) q[v] = W[v][c];
                    free_stream_at(ax[0].c[(size_t)i], ax[1].c[(size_t)j], ax[2].c[(size_t)k], t);
                    double xi = (ax[0].c[(size_t)i] - tu_x0) / (tu_x1 - tu_x0), wgt = rl * std::sin(geo::PI * std::max(0.0, std::min(1.0, xi)));  // smooth in and out
                    for (int v = 1; v <= 3; ++v) q[v] += wgt * (t[v] - q[v]);
                    cons_from(q, c);
                }
        });
    }

    void collect_fluid() { fluid.clear(); for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) if (tag[id(i, j, k)] == 0) fluid.push_back(id(i, j, k)); }

    void accumulate() {
        if (mean[0].empty()) for (int v = 0; v < 4; ++v) { mean[v].assign(N, 0); rms[v].assign(N, 0); }
        ++n_avg;
        par::parallel_for((int)fluid.size(), [&](int q) {
            size_t c = fluid[(size_t)q];
            for (int v = 0; v < 4; ++v) { double x = W[v == 3 ? 4 : 1 + v][c]; mean[v][c] += (float)x; rms[v][c] += (float)(x * x); }
        });
        for (auto& g : ghosts) { g.sum_cp += g.cp; g.sum_cp2 += g.cp * g.cp; g.sum_cf += g.cf; g.sum_T += g.Tw; }
    }

    // ---------------------------------------------------------------- output
    void write_plane(int a, double at, const std::string& name) {
        int n0 = a == 0 ? nx : a == 1 ? ny : nz, idx = ax[a].locate(at);
        idx = std::max(0, std::min(n0 - 1, idx));
        int o1 = a == 0 ? 1 : 0, o2 = a == 2 ? 1 : 2;
        int n1 = o1 == 0 ? nx : ny, n2 = o2 == 1 ? ny : nz;
        std::vector<float> buf;
        const char* names[] = {"rho", "u", "v", "w", "p", "mach", "vort", "solid", "tke", "p_rms"};
        int nv = n_avg > 1 ? 10 : 8;
        for (int v = 0; v < nv; ++v)
            for (int q2 = 0; q2 < n2; ++q2)
                for (int q1 = 0; q1 < n1; ++q1) {
                    int x[3]; x[a] = idx; x[o1] = q1; x[o2] = q2;
                    size_t c = id(x[0], x[1], x[2]);
                    double val = 0;
                    if (v < 5) val = W[v][c];
                    else if (v == 5) val = std::sqrt(W[1][c] * W[1][c] + W[2][c] * W[2][c] + W[3][c] * W[3][c]) / std::sqrt(GAM * W[4][c] / W[0][c]);
                    else if (v == 6) val = viscous && tag[c] == 0 ? std::sqrt(std::pow(gU[7][c] - gU[5][c], 2) + std::pow(gU[2][c] - gU[6][c], 2) + std::pow(gU[3][c] - gU[1][c], 2)) : 0;
                    else if (v == 7) val = tag[c] != 0;
                    else if (v == 8) { double s = 0; for (int m = 0; m < 3; ++m) { double mu = mean[m][c] / n_avg; s += std::max(0.0, rms[m][c] / n_avg - mu * mu); } val = 0.5 * s / (mach * mach); }
                    else { double mu = mean[3][c] / n_avg; val = std::sqrt(std::max(0.0, rms[3][c] / n_avg - mu * mu)) / (0.5 * mach * mach); }
                    if (tag[c] != 0 && v != 7) val = 0;
                    buf.push_back((float)val);
                }
        js::Value h = js::Value::object();
        h["axis"] = a == 0 ? "x" : a == 1 ? "y" : "z"; h["at"] = ax[a].c[(size_t)idx];
        js::Value c1 = js::Value::array(), c2 = js::Value::array(), vn = js::Value::array();
        for (int q = 0; q < n1; ++q) c1.push(ax[o1].c[(size_t)q]);
        for (int q = 0; q < n2; ++q) c2.push(ax[o2].c[(size_t)q]);
        for (int v = 0; v < nv; ++v) vn.push(names[v]);
        h["u_axis"] = o1 == 0 ? "x" : "y"; h["v_axis"] = o2 == 1 ? "y" : "z"; h["u"] = c1; h["v"] = c2; h["vars"] = vn;
        js::write_file(out + "/" + name + ".json", js::dump(h, 0));
        FILE* f = fopen((out + "/" + name + ".bin").c_str(), "wb");
        fwrite(buf.data(), 4, buf.size(), f);
        fclose(f);
    }

    void write_volume() {  // every cell, float32: rho u v w p solid, for training data on disk
        FILE* f = fopen((out + "/volume.bin").c_str(), "wb");
        std::vector<float> row((size_t)nx);
        int nv = n_avg > 1 ? 10 : 6;  // rho u v w p solid, and for averaged runs tke p_rms u_rms vort
        for (int v = 0; v < nv; ++v)
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        size_t c = id(i, j, k);
                        double val = 0;
                        if (v < 5) val = tag[c] ? 0.0 : W[v][c];
                        else if (v == 5) val = tag[c] != 0;
                        else if (tag[c]) val = 0;
                        else if (v == 6) { double t = 0; for (int m = 0; m < 3; ++m) { double mu = mean[m][c] / n_avg; t += std::max(0.0, rms[m][c] / n_avg - mu * mu); } val = 0.5 * t / (mach * mach); }  // k / U^2
                        else if (v == 7) { double mu = mean[3][c] / n_avg; val = std::sqrt(std::max(0.0, rms[3][c] / n_avg - mu * mu)) / (0.5 * mach * mach); }  // p_rms / q
                        else if (v == 8) { double mu = mean[0][c] / n_avg; val = std::sqrt(std::max(0.0, rms[0][c] / n_avg - mu * mu)) / mach; }  // u_rms / U
                        else val = viscous ? std::sqrt(std::pow(gU[7][c] - gU[5][c], 2) + std::pow(gU[2][c] - gU[6][c], 2) + std::pow(gU[3][c] - gU[1][c], 2)) : 0;  // |vorticity|
                        row[(size_t)i] = (float)val;
                    }
                    fwrite(row.data(), 4, row.size(), f);
                }
        fclose(f);
        js::Value h = js::Value::object();
        for (int a = 0; a < 3; ++a) { js::Value c = js::Value::array(); for (double x : ax[a].c) c.push(x); h[a == 0 ? "x" : a == 1 ? "y" : "z"] = c; }
        js::Value vn = js::Value::array(); for (const char* s : {"rho", "u", "v", "w", "p", "solid", "tke", "p_rms", "u_rms", "vort"}) if (vn.size() < (size_t)nv) vn.push(s);
        h["averaged_samples"] = (double)n_avg;
        h["vars"] = vn; h["order"] = "var, z, y, x";
        js::write_file(out + "/volume.json", js::dump(h, 0));
    }

    // grid sequencing: start from the field of an earlier (coarser) run of the same flight instead of the free stream.
    // Trilinear between the earlier run's cell centres, using only its fluid cells; a half-model field fills a full
    // model by mirror symmetry. Returns the number of cells filled.
    size_t init_from_volume(const std::string& dir) {
        js::Value h;
        try { h = js::parse(js::read_file(dir + "/volume.json")); } catch (...) { return 0; }
        std::vector<double> cx[3] = {h.nums("x"), h.nums("y"), h.nums("z")};
        for (int a = 0; a < 3; ++a) if (cx[a].size() < 2) return 0;
        size_t mx = cx[0].size(), my = cx[1].size(), mz = cx[2].size(), M = mx * my * mz;
        FILE* f = fopen((dir + "/volume.bin").c_str(), "rb");
        if (!f) return 0;
        std::vector<float> q6(6 * M);
        size_t got = fread(q6.data(), 4, q6.size(), f);
        fclose(f);
        if (got != q6.size()) return 0;
        bool src_half = cx[1].front() > 0;
        auto loc = [](const std::vector<double>& c, double x, int& i, double& w) {
            if (x <= c.front()) { i = 0; w = 0; return; }
            if (x >= c.back()) { i = (int)c.size() - 2; w = 1; return; }
            i = (int)(std::upper_bound(c.begin(), c.end(), x) - c.begin()) - 1;
            i = std::max(0, std::min((int)c.size() - 2, i));
            w = (x - c[(size_t)i]) / std::max(c[(size_t)i + 1] - c[(size_t)i], 1e-12);
        };
        std::vector<char> done_c(fluid.size(), 0);
        par::parallel_for((int)fluid.size(), [&](int qn) {
            size_t c = fluid[(size_t)qn];
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            double x = ax[0].c[(size_t)i], y = ax[1].c[(size_t)j], z = ax[2].c[(size_t)k];
            double sy = 1;
            if (src_half && y < 0) { y = -y; sy = -1; }
            int I, J, K; double wx, wy, wz;
            loc(cx[0], x, I, wx); loc(cx[1], y, J, wy); loc(cx[2], z, K, wz);
            double acc[5] = {0, 0, 0, 0, 0}, wsum = 0;
            for (int dk = 0; dk < 2; ++dk) for (int dj = 0; dj < 2; ++dj) for (int di = 0; di < 2; ++di) {
                size_t m = ((size_t)(K + dk) * my + (size_t)(J + dj)) * mx + (size_t)(I + di);
                if (q6[5 * M + m] > 0.5f) continue;  // a solid cell of the earlier run
                double w = (di ? wx : 1 - wx) * (dj ? wy : 1 - wy) * (dk ? wz : 1 - wz);
                if (w <= 0) continue;
                for (int v = 0; v < 5; ++v) acc[v] += w * q6[(size_t)v * M + m];
                wsum += w;
            }
            if (wsum < 1e-6) return;  // no fluid nearby in the earlier run: keep the free stream
            double q[5];
            for (int v = 0; v < 5; ++v) q[v] = acc[v] / wsum;
            q[2] *= sy;
            if (!(q[0] > 0) || !(q[4] > 0)) return;
            cons_from(q, c);
            done_c[(size_t)qn] = 1;
        });
        size_t n = 0; for (char v : done_c) n += (size_t)v;
        return n;
    }

    void write_surface() {
        FILE* f = fopen((out + "/surface.csv").c_str(), "w");
        fprintf(f, "x,y,z,nx,ny,nz,cp,cp_rms,cf,T_wall_K,yplus,separated,area\n");
        for (auto& g : ghosts) {
            if (!g.first) continue;
            size_t c = g.cell;
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            double cp = n_avg ? g.sum_cp / n_avg : g.cp, cf = n_avg ? g.sum_cf / n_avg : g.cf, T = n_avg ? g.sum_T / n_avg : g.Tw;
            double cpr = n_avg ? std::sqrt(std::max(0.0, g.sum_cp2 / n_avg - cp * cp)) : 0;
            double px = ax[0].c[(size_t)i] + g.nx * g.dw, py = ax[1].c[(size_t)j] + g.ny * g.dw, pz = ax[2].c[(size_t)k] + g.nz * g.dw;
            fprintf(f, "%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.5f,%.5f,%.6f,%.2f,%.1f,%d,%.6f\n", px, py, pz, g.nx, g.ny, g.nz, cp, cpr, cf, T * Tinf_dim, g.yplus, (int)g.sep, g.area);
        }
        for (size_t c : thin) {  // thin surfaces: one row per wetted face, from the fluid cell on that side
            int k = (int)(c / ((size_t)NX * NY)) - G, j = (int)((c / NX) % NY) - G, i = (int)(c % NX) - G;
            for (int a = 0; a < 3; ++a)
                for (int sg = -1; sg <= 1; sg += 2) {
                    int I = i + (a == 0) * sg, J = j + (a == 1) * sg, K = k + (a == 2) * sg;
                    if (I < 0 || J < 0 || K < 0 || I >= nx || J >= ny || K >= nz) continue;
                    size_t fc = id(I, J, K);
                    if (tag[fc] != 0) continue;
                    V3 n = thin_normal(c, a, sg);
                    double F[5]; thin_flux(fc, c, a, sg < 0, F);
                    double un = W[1][fc] * n.x + W[2][fc] * n.y + W[3][fc] * n.z;
                    double pw = F[1 + a] - W[0][fc] * un * un;  // face pressure of the reflected Riemann problem
                    double ut2 = W[1][fc] * W[1][fc] + W[2][fc] * W[2][fc] + W[3][fc] * W[3][fc] - un * un;
                    double T = GAM * W[4][fc] / W[0][fc] + std::cbrt(Pr) * ut2 * (GAM - 1) / 2;
                    double A = (a == 0 ? ax[1].d[(size_t)j] * ax[2].d[(size_t)k] : a == 1 ? ax[0].d[(size_t)i] * ax[2].d[(size_t)k] : ax[0].d[(size_t)i] * ax[1].d[(size_t)j]) * std::fabs(n[a]);
                    double px = ax[0].c[(size_t)i] + 0.5 * sg * (a == 0) * ax[0].d[(size_t)i], py = ax[1].c[(size_t)j] + 0.5 * sg * (a == 1) * ax[1].d[(size_t)j], pz = ax[2].c[(size_t)k] + 0.5 * sg * (a == 2) * ax[2].d[(size_t)k];
                    int sep = (W[1][fc] - un * n.x) * std::cos(alpha) + (W[3][fc] - un * n.z) * std::sin(alpha) < 0 ? 1 : 0;
                    fprintf(f, "%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.5f,%.5f,%.6f,%.2f,%.1f,%d,%.6f\n", px, py, pz, n.x, n.y, n.z, (pw - 1 / GAM) / (0.5 * mach * mach), 0.0, 0.0, T * Tinf_dim, 0.0, sep, A);
                }
        }
        fclose(f);
    }
};

// ---------------------------------------------------------------- set-up and main
static void isa(double h, double& T, double& p, double& rho) {
    if (h < 11000) { T = 288.15 - 0.0065 * h; p = 101325 * std::pow(T / 288.15, 5.2559); }
    else if (h < 20000) { T = 216.65; p = 22632.1 * std::exp(-9.80665 * (h - 11000) / (287.05 * 216.65)); }
    else { T = 216.65 + 0.001 * (h - 20000); p = 5474.89 * std::pow(T / 216.65, -34.1632); }
    rho = p / (287.05 * T);
}

static BC bc_of(const std::string& s) { return s == "periodic" ? PERIODIC : s == "extrapolate" ? EXTRAP : s == "symmetry" ? SYMMETRY : FARFIELD; }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: aeroflow case.json\n"); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    Solver S;
    js::Value cs;
    try { cs = js::parse(js::read_file(argv[1])); } catch (std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 2; }
    const js::Value empty = js::Value::object();
    const js::Value& fl = cs.find("flow") ? *cs.find("flow") : empty;
    const js::Value& gr = cs.find("grid") ? *cs.find("grid") : empty;
    const js::Value& md = cs.find("model") ? *cs.find("model") : empty;
    const js::Value& rn = cs.find("run") ? *cs.find("run") : empty;
    const js::Value& rf = cs.find("reference") ? *cs.find("reference") : empty;
    S.kase = cs.str("case", "body");
    S.out = rn.str("output", "aeroflow_out");
    { std::string cmd = "mkdir \"" + S.out + "\" 2>nul"; for (auto& ch : cmd) if (ch == '/') ch = '\\'; if (system(cmd.c_str())) {} }
    S.mach = fl.num("mach", 0.5); S.alpha = fl.num("alpha", 0) * geo::PI / 180; S.beta = fl.num("beta", 0) * geo::PI / 180;
    S.gust_amp = fl.num("gust_amplitude", 0); S.gust_len = fl.num("gust_wavelength", 20);
    S.tu = fl.num("turbulence_intensity", 0); S.tu_len = fl.num("turbulence_length", 2);
    S.Lref = rf.num("length", 1); S.Sref = rf.num("area", 1); S.cref = rf.num("chord", 1); S.bref = rf.num("span", 1);
    { std::vector<double> m = rf.nums("moment_point"); if (m.size() == 3) S.mref = V3(m[0], m[1], m[2]); }
    double alt = fl.num("altitude", 0), T, p, rho;
    isa(alt, T, p, rho);
    T += fl.num("delta_T", 0); rho = p / (287.05 * T);
    S.Tinf_dim = T; S.pinf_dim = p;
    double a_dim = std::sqrt(1.4 * 287.05 * T), mu_dim = 1.458e-6 * std::pow(T, 1.5) / (T + 110.4);
    S.Re = fl.num("reynolds", rho * S.mach * a_dim * S.Lref / mu_dim);
    S.viscous = md.flag("viscous", true); S.wallmodel = md.flag("wall_model", true); S.sgs = md.str("sgs", "wale");
    S.cfl = md.num("cfl", 0.8); S.sutherland = md.flag("sutherland", true); S.thin_on = md.flag("thin_surfaces", true); S.low_mach = md.flag("low_mach_fix", rn.flag("unsteady", false) || S.kase == "tgv");  // steady runs keep the extra dissipation (it converges them)
    bool unsteady = rn.flag("unsteady", false);
    int steps = (int)rn.num("steps", 2000), avg_from = (int)rn.num("average_from", -1), every = (int)rn.num("log_every", 10);
    double t_end = rn.num("t_end", -1), tol = rn.num("residual_drop", 1e-4);
    // > 0: a steady run also stops when the mean CL and CD over the last force_window samples agree within this
    // fraction with the means over the window before. The forces of these runs swing slowly round a steady value;
    // the window means settle long before the swings die out, and the result is then the last window's mean
    double force_band = rn.num("force_band", 0);
    int force_window = std::max(10, (int)rn.num("force_window", 60));

    // grid
    if (const js::Value* u = gr.find("uniform")) {
        std::vector<double> n = u->nums("n"), lo = u->nums("lo"), hi = u->nums("hi");
        for (int a = 0; a < 3; ++a) S.ax[a].uniform(lo[(size_t)a], hi[(size_t)a], (int)n[(size_t)a]);
        std::vector<std::string> keys = {"x", "y", "z"};
        if (const js::Value* b = gr.find("bc"))
            for (int a = 0; a < 3; ++a) { const js::Value* q = b->find(keys[(size_t)a]); if (q && q->is_arr() && q->size() == 2) { S.bc[a][0] = bc_of(q->at(0).s); S.bc[a][1] = bc_of(q->at(1).s); } }
    }
    if (S.kase == "body") {
        try {
            if (const js::Value* m = cs.find("design")) S.model.load(m->is_str() ? js::parse(js::read_file(m->s)) : *m);
        } catch (std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 2; }
        if (!S.model.errors.empty()) { fprintf(stderr, "design error: %s\n", S.model.errors[0].c_str()); return 3; }
        S.has_body = true;
        geo::Box b = S.model.bounds;
        double L = std::max(b.hi.x - b.lo.x, 1e-3);
        S.tu_x0 = b.lo.x - fl.num("turbulence_zone_start", 0.45) * L; S.tu_x1 = b.lo.x - fl.num("turbulence_zone_end", 0.15) * L;
        S.half = gr.flag("half", std::fabs(S.beta) < 1e-9 && std::fabs(b.lo.y + b.hi.y) < 1e-3 * L);
        double cpl = gr.num("cells_per_length", 64), h0 = L / cpl, ratio = gr.num("stretch", 1.12), hmax = gr.num("max_spacing", 0.5 * L);
        std::vector<double> dom = gr.nums("domain");  // upstream, downstream, lateral, vertical, in body lengths
        if (dom.size() != 4) dom = {S.mach > 1.05 ? 0.4 : 1.5, S.mach > 1.05 ? 0.6 : 2.5, S.mach > 1.05 ? 0.8 : 1.5, S.mach > 1.05 ? 0.8 : 1.5};
        double pad = gr.num("core_pad", 0.06) * L;
        S.ax[0].stretched(b.lo.x - dom[0] * L, b.lo.x - pad, b.hi.x + pad, b.hi.x + dom[1] * L, h0, ratio, hmax);
        double span = std::max(b.hi.y, -b.lo.y);
        if (S.half) S.ax[1].stretched(0, 0, span + pad, span + dom[2] * L, h0, ratio, hmax);
        else S.ax[1].stretched(-span - dom[2] * L, -span - pad, span + pad, span + dom[2] * L, h0, ratio, hmax);
        S.ax[2].stretched(b.lo.z - dom[3] * L, b.lo.z - pad, b.hi.z + pad, b.hi.z + dom[3] * L, h0, ratio, hmax);
        if (S.half) S.bc[1][0] = SYMMETRY;
        if (S.mach > 1.05) S.bc[0][1] = EXTRAP;
    }
    size_t cells = (size_t)S.ax[0].n * S.ax[1].n * S.ax[2].n;
    double max_cells = gr.num("max_cells", 2.5e6);
    if (cells > max_cells) { fprintf(stderr, "grid has %u cells, above max_cells %.0f: lower cells_per_length\n", (unsigned)cells, max_cells); return 4; }
    S.allocate();
    S.mu_inf = S.mach * S.Lref / S.Re;
    if (S.kase == "tgv") {  // Taylor-Green vortex: viscosity from the requested Reynolds number at length 1
        S.mu_inf = S.mach * 1.0 / S.Re; S.sutherland = false; S.wallmodel = false;
    }
    S.init_free_stream();
    if (S.kase == "tgv") {
        for (int k = 0; k < S.nz; ++k) for (int j = 0; j < S.ny; ++j) for (int i = 0; i < S.nx; ++i) {
            double x = S.ax[0].c[(size_t)i], y = S.ax[1].c[(size_t)j], z = S.ax[2].c[(size_t)k], U0 = S.mach;
            double q[5]; q[1] = U0 * std::sin(x) * std::cos(y) * std::cos(z); q[2] = -U0 * std::cos(x) * std::sin(y) * std::cos(z); q[3] = 0;
            q[4] = 1 / GAM + U0 * U0 / 16 * (std::cos(2 * x) + std::cos(2 * y)) * (std::cos(2 * z) + 2); q[0] = GAM * q[4];
            S.cons_from(q, S.id(i, j, k));
        }
    } else if (S.kase == "sod") {
        for (int k = 0; k < S.nz; ++k) for (int j = 0; j < S.ny; ++j) for (int i = 0; i < S.nx; ++i) {
            bool left = S.ax[0].c[(size_t)i] < 0.5;
            double q[5] = {left ? 1.0 : 0.125, 0, 0, 0, left ? 1.0 : 0.1};
            S.cons_from(q, S.id(i, j, k));
        }
        S.mach = 1;  // only used to scale outputs
    }
    if (S.has_body) S.build_body();
    S.collect_fluid();
    bool sequenced = false;
    if (const js::Value* f0 = rn.find("init_from")) {  // grid sequencing: start from a coarser run's field
        size_t n0 = S.init_from_volume(f0->s);
        sequenced = n0 > S.fluid.size() / 2;
        fprintf(stderr, "initialised %u of %u fluid cells from %s\n", (unsigned)n0, (unsigned)S.fluid.size(), f0->s.c_str());
    }
    if (const js::Value* r0 = rn.find("restart")) {  // continue from a saved state on the same grid
        FILE* f = fopen(r0->s.c_str(), "rb");
        if (f) {
            unsigned long long n = 0; if (fread(&n, 8, 1, f) == 1 && n == S.N) { for (int v = 0; v < 5; ++v) if (fread(S.U[v].data(), 8, S.N, f) != S.N) break; for (size_t c = 0; c < S.N; ++c) S.prim(c); fprintf(stderr, "restarted from %s\n", r0->s.c_str()); }
            fclose(f);
        }
    }
    // probes: given in the case, or placed in the wake, the tip wakes, over the wing and upstream
    struct Probe { std::string name; double x, y, z; size_t c; };
    std::vector<Probe> probes;
    if (S.kase == "body") {
        geo::Box b = S.model.bounds;
        double L = std::max(b.hi.x - b.lo.x, 1e-3), span = std::max(b.hi.y, -b.lo.y), zc = 0.5 * (b.lo.z + b.hi.z);
        std::vector<Probe> want;
        if (const js::Value* pr = rn.find("probes")) {
            for (size_t q = 0; q < pr->size(); ++q) { std::vector<double> v = pr->at(q).nums("at"); if (v.size() == 3) want.push_back({pr->at(q).str("name", "probe"), v[0], v[1], v[2], 0}); }
        } else {
            want = {{"wake, centre", b.hi.x + 0.3 * L, 0, zc, 0}, {"wake, far", b.hi.x + 0.8 * L, 0, zc, 0}, {"right tip wake", b.hi.x + 0.1 * L, 0.9 * span, zc, 0},
                    {"left tip wake", b.hi.x + 0.1 * L, -0.9 * span, zc, 0}, {"over the wing", 0.5 * (b.lo.x + b.hi.x) + 0.1 * L, 0.4 * span, b.hi.z + 0.05 * L, 0},
                    {"upstream", b.lo.x - 0.08 * L, 0.3 * span, zc, 0}};
        }
        for (auto& q : want) {
            double pt[3] = {q.x, q.y, q.z}; bool in = true;
            for (int a = 0; a < 3; ++a) if (pt[a] < S.ax[a].c.front() || pt[a] > S.ax[a].c.back()) in = false;
            if (!in) continue;  // outside the grid (a half model has no left side)
            q.c = S.id(S.ax[0].locate(q.x), S.ax[1].locate(q.y), S.ax[2].locate(q.z));
            probes.push_back(q);
        }
    }
    FILE* pf = nullptr;
    if (!probes.empty()) {
        pf = fopen((S.out + "/probes.csv").c_str(), "w");
        fprintf(pf, "step,time");
        for (size_t q = 0; q < probes.size(); ++q) fprintf(pf, ",u%u,v%u,w%u,p%u", (unsigned)q, (unsigned)q, (unsigned)q, (unsigned)q);
        fprintf(pf, "\n");
        js::Value pj = js::Value::array();
        for (auto& q : probes) { js::Value e = js::Value::object(); e["name"] = q.name; js::Value at = js::Value::array(); at.push(q.x); at.push(q.y); at.push(q.z); e["at"] = at; pj.push(e); }
        js::Value ph = js::Value::object(); ph["probes"] = pj; ph["mach"] = S.mach; ph["length"] = S.Lref;
        ph["time_unit_s"] = S.Lref > 0 ? 1.0 / std::sqrt(1.4 * 287.05 * S.Tinf_dim) : 1.0;  // solver time is metres / speed of sound
        ph["note"] = "u v w in units of the speed of sound, p in units of rho_inf a_inf^2";
        js::write_file(S.out + "/probes.json", js::dump(ph, 1));
    }
    fprintf(stderr, "case %s  grid %d x %d x %d = %u cells, fluid %u, wall cells %u, thin-surface cells %u, Re %.3g, threads %d\n", S.kase.c_str(), S.nx, S.ny, S.nz, (unsigned)cells, (unsigned)S.fluid.size(), (unsigned)S.ghosts.size(), (unsigned)S.thin.size(), S.Re, par::threads());

    FILE* hist = fopen((S.out + "/history.csv").c_str(), "w");
    fprintf(hist, S.kase == "tgv" ? "step,time,residual,kinetic_energy,enstrophy\n" : "step,time,residual,CL,CD,CY,Cl,Cm,Cn,CDf\n");
    double res0 = -1, C[8] = {0}, Csum[8] = {0}, Csq[8] = {0}; long nC = 0;
    int done = 0; bool converged = false;
    std::vector<double> cl_hist, cd_hist;
    std::vector<std::array<double, 8>> c_hist;
    bool window_mean = false;
    std::string stop_reason = "steps";
    double ref_cfl = S.cfl;
    for (int it = 1; it <= steps; ++it) {
        S.cfl = ref_cfl * std::min(1.0, (sequenced ? 0.6 : 0.2) + it / 150.0);  // gentle start (shorter from a sequenced field)
        double res = S.step(unsteady || S.kase != "body");
        if (unsteady) S.force_turbulence(1.0);
        if (!std::isfinite(res)) { fprintf(stderr, "diverged at step %d\n", it); fprintf(hist, "# diverged\n"); break; }
        if (res0 < 0) res0 = res;
        done = it;
        bool averaging = avg_from >= 0 && it >= avg_from;
        if (S.kase == "body") {
            if (it % every == 0 || it == steps || averaging) {
                S.coefficients(S.forces(), C);
                if (averaging) { S.accumulate(); for (int q = 0; q < 8; ++q) { Csum[q] += C[q]; Csq[q] += C[q] * C[q]; } ++nC; }
                if (it % every == 0) fprintf(hist, "%d,%.6g,%.5e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", it, S.time, res / res0, C[0], C[1], C[2], C[3], C[4], C[5], C[6]);
                cl_hist.push_back(C[0]); cd_hist.push_back(C[1]);
                if (it % every == 0) { std::array<double, 8> a; for (int q = 0; q < 8; ++q) a[(size_t)q] = C[q]; c_hist.push_back(a); }
            }
            if (!unsteady && avg_from < 0 && cl_hist.size() > 60 && res / res0 < tol) {
                double mx = -1e9, mn = 1e9; for (size_t q = cl_hist.size() - 40; q < cl_hist.size(); ++q) { mx = std::max(mx, cl_hist[q]); mn = std::min(mn, cl_hist[q]); }
                if (mx - mn < 2e-4) { converged = true; stop_reason = "residual"; break; }
            }
            size_t nw = (size_t)force_window;
            if (!unsteady && avg_from < 0 && force_band > 0 && it % every == 0 && c_hist.size() >= 2 * nw) {  // the mean forces have settled
                double m1[8] = {0}, m0[8] = {0};
                for (size_t q = 0; q < nw; ++q) for (int v = 0; v < 8; ++v) { m1[v] += c_hist[c_hist.size() - 1 - q][(size_t)v] / nw; m0[v] += c_hist[c_hist.size() - 1 - nw - q][(size_t)v] / nw; }
                if (std::fabs(m1[0] - m0[0]) < force_band * std::max(0.02, std::fabs(m1[0])) && std::fabs(m1[1] - m0[1]) < force_band * std::max(0.002, std::fabs(m1[1]))) {
                    for (int v = 0; v < 8; ++v) C[v] = m1[v];
                    converged = true; stop_reason = "forces"; window_mean = true; break;
                }
            }
        } else if (S.kase == "tgv" && (it % every == 0 || it == 1)) {
            double ke = 0, en = 0, vol = 0;
            for (size_t c : S.fluid) {
                int k = (int)(c / ((size_t)S.NX * S.NY)) - G, j = (int)((c / S.NX) % S.NY) - G, i = (int)(c % S.NX) - G;
                double V = S.ax[0].d[(size_t)i] * S.ax[1].d[(size_t)j] * S.ax[2].d[(size_t)k];
                ke += 0.5 * S.W[0][c] * (S.W[1][c] * S.W[1][c] + S.W[2][c] * S.W[2][c] + S.W[3][c] * S.W[3][c]) * V;
                double wx = S.gU[7][c] - S.gU[5][c], wy = S.gU[2][c] - S.gU[6][c], wz = S.gU[3][c] - S.gU[1][c];
                en += 0.5 * S.W[0][c] * (wx * wx + wy * wy + wz * wz) * V; vol += V;
            }
            double U0 = S.mach;
            fprintf(hist, "%d,%.6g,%.5e,%.8f,%.8f\n", it, S.time * U0, res / res0, ke / vol / (U0 * U0), en / vol / (U0 * U0));
        }
        if (pf && (unsteady || it % every == 0)) {
            fprintf(pf, "%d,%.7g", it, S.time);
            for (auto& q : probes) fprintf(pf, ",%.6g,%.6g,%.6g,%.6g", S.W[1][q.c], S.W[2][q.c], S.W[3][q.c], S.W[4][q.c]);
            fprintf(pf, "\n");
        }
        if (it % 100 == 0) { fflush(hist); fprintf(stderr, "step %d  residual %.3e  time %.4g  %.0f s\n", it, res / res0, S.time, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()); }
        if (t_end > 0 && S.time >= t_end) break;
    }
    fclose(hist);
    if (pf) fclose(pf);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    js::Value o = js::Value::object();
    o["case"] = S.kase; o["steps"] = (double)done; o["converged"] = converged; o["stop_reason"] = stop_reason; o["sequenced"] = sequenced;
    if (window_mean) { o["forces_from"] = "mean of the last window"; o["force_window_steps"] = (double)(force_window * every); } o["seconds"] = sec; o["time"] = S.time;
    o["cells"] = (double)cells; o["fluid_cells"] = (double)S.fluid.size(); o["wall_cells"] = (double)S.ghosts.size(); o["thin_surface_cells"] = (double)S.thin.size(); o["sealed_cells"] = (double)S.sealed; o["wall_faces"] = (double)S.fwl.size();
    js::Value dims = js::Value::array(); dims.push(S.nx); dims.push(S.ny); dims.push(S.nz); o["grid"] = dims;
    o["mach"] = S.mach; o["alpha_deg"] = S.alpha * 180 / geo::PI; o["beta_deg"] = S.beta * 180 / geo::PI; o["reynolds"] = S.Re;
    o["T_inf_K"] = S.Tinf_dim; o["p_inf_Pa"] = S.pinf_dim; o["half_model"] = S.half; o["threads"] = par::threads();
    o["turbulence_intensity"] = S.tu; o["turbulence_length"] = S.tu_len; o["probes"] = (double)probes.size();
    o["sgs"] = S.sgs; o["wall_model"] = S.wallmodel; o["viscous"] = S.viscous; o["low_mach_fix"] = S.low_mach;
    if (S.kase == "body") {
        const char* nm[8] = {"CL", "CD", "CY", "Cl", "Cm", "Cn", "CD_friction", "CL_friction"};
        js::Value last = js::Value::object(), mean = js::Value::object(), rms = js::Value::object();
        for (int q = 0; q < 8; ++q) {
            last[nm[q]] = C[q];
            if (nC > 1) { double m = Csum[q] / nC; mean[nm[q]] = m; rms[nm[q]] = std::sqrt(std::max(0.0, Csq[q] / nC - m * m)); }
        }
        o["coefficients"] = nC > 1 ? mean : last; o["last"] = last;
        if (nC > 1) { o["rms"] = rms; o["samples"] = (double)nC; }
        double sep = 0, area = 0, Tmax = 0, cpmin = 1e9;
        for (auto& g : S.ghosts) if (g.first) { area += g.area; sep += g.sep * g.area; Tmax = std::max(Tmax, g.Tw * S.Tinf_dim); cpmin = std::min(cpmin, g.cp); }
        o["separated_fraction"] = area > 0 ? sep / area : 0.0; o["wall_area_m2"] = S.half ? 2 * area : area; o["T_wall_max_K"] = Tmax; o["cp_min"] = cpmin;
        S.write_surface();
        S.write_plane(1, S.half ? 0.0 : 0.5 * (S.model.bounds.lo.y + S.model.bounds.hi.y), "plane_symmetry");
        geo::Box b = S.model.bounds;
        for (int q = 1; q <= 3; ++q) { char nmq[32]; snprintf(nmq, sizeof nmq, "plane_x%d", q); S.write_plane(0, b.lo.x + (b.hi.x - b.lo.x) * (0.25 * q + 0.1), nmq); }
        S.write_plane(2, 0.5 * (b.lo.z + b.hi.z), "plane_z");
        if (rn.flag("save_volume", false)) S.write_volume();
    } else {
        S.write_plane(2, S.ax[2].c[(size_t)S.nz / 2], "plane_z");
        if (rn.flag("save_volume", false)) S.write_volume();
    }
    js::Value lim = js::Value::array();
    lim.push("boundary layers are not resolved: friction comes from an equilibrium wall model (law of the wall)");
    lim.push("separation from smooth surfaces is approximate on this grid; separation from sharp edges is captured");
    lim.push("turbulence: WALE large-eddy model on a coarse grid; resolved scales are limited by the cell size");
    o["limits"] = lim;
    if (rn.flag("save_state", false)) {
        FILE* f = fopen((S.out + "/state.bin").c_str(), "wb");
        unsigned long long n = S.N; fwrite(&n, 8, 1, f);
        for (int v = 0; v < 5; ++v) fwrite(S.U[v].data(), 8, S.N, f);
        fclose(f);
    }
    js::write_file(S.out + "/result.json", js::dump(o));
    printf("%s\n", js::dump(o).c_str());
    return 0;
}
