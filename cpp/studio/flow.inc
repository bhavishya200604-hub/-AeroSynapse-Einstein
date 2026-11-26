// Fluid flow round and through an assembly: a lattice-Boltzmann solver (D3Q19, BGK collision with the Smagorinsky
// large-eddy model, half-way bounce-back walls, moving walls by Ladd's momentum correction, forces by momentum
// exchange) on a grid cut from the parts' exact shapes. Two set-ups:
//   * a current past submerged parts (inflow upstream, outflow downstream, slip walls at the sides): wakes, vortex
//     shedding, drag and lift, the Strouhal number;
//   * a stirred vessel (closed, no-slip walls, a flat free surface on top): the turning parts stir the fluid from rest,
//     shear layers roll up into vortices and break down into turbulence; torque, power and power number.
// The turning parts turn in the flow at their motion-study speeds (their solid cells are re-cut from a local distance
// grid as they turn). Shown in the view: a slice of vorticity (or speed, or turbulence), the vortex cores (Q
// criterion surfaces) and particles carried by the flow; the report gives the numbers and how the turbulence grew.
//
// Limits, said in the report: the grid resolves the large eddies only (the Smagorinsky model carries the rest), walls
// are staircased on the grid (no wall functions), the free surface stays flat (no surface vortex dip), parts that
// translate (racks, followers) stand still in the flow, and the fluid is isothermal.

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

static const int LQ = 19;
static const int NPB = 180;  // angular bins of the loss-torque polar
static const int LC[LQ][3] = {{0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0},
                              {1, 0, 1}, {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, -1}, {0, 1, -1}, {0, -1, 1}};
static const int LOPP[LQ] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
static const float LW[LQ] = {1.f / 3, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 18, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36};
static int LREFL[3][LQ];  // the direction mirrored in each axis (slip walls)

enum { FL_FLUID = 0, FL_STATIC = 1, FL_MOVING = 2, FL_FREED = 3, FL_GAS = 4 };  // gas: above a free surface (bath mode)
static bool g_flow_bridge = getenv("AEROSYNAPSE_FLOW_NO_BRIDGE") == nullptr;  // bridge gaps one cell wide between parts (see movers_update)
enum { FACE_WALL = 0, FACE_SLIP = 1, FACE_IN = 2, FACE_OUT = 3 };

struct FlowCfg {
    int mode = 0;            // 0 current past the parts, 1 stirred vessel (full), 2 a partly filled bath (free surface: liquid and gas)
    int cells = 96;          // cells along the domain's longest side
    double U = 0.1;          // current (m/s)
    int dir = 0;             // 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z
    double duration = 0;     // simulated seconds (0: automatic)
    bool span_walls = false; // current: slip walls at the parts' ends along the long cross axis (a 2D-like flow)
    int span_axis = -1;      // that axis (-1: the longer cross axis)
    double margin = 0.04;    // vessel: margin round the parts (fraction of the size)
    int show = 1;            // the field shown: 0 vorticity, 1 speed, 2 turbulence, 3 none, 4 pressure, 5 temperature
    int view = 0;            // 0 the whole fluid (a coloured volume), 1 a slice, 2 both
    int heat = 1;            // 0 isothermal, 1 the turning parts hot, 2 every part hot
    double Thot = 0;         // the hot parts' temperature (C; 0: the Environment's part temperature)
    bool buoyancy = true;    // natural convection from the temperature differences (Boussinesq)
    int walls = 3;           // the cutaway's walls: 0 the shown field, 1 pressure, 2 velocity (speed and arrows), 3 pressure on x walls, velocity on y walls, the shown field on z walls
    int surf = 1;            // the parts' surfaces coloured by: 0 nothing, 1 friction (wall shear), 2 pressure, 3 temperature, 4 turbulence, 5 liquid fraction
    int slice_axis = 2;      // slice normal
    double slice_pos = 0.5;  // 0..1 across the domain
    bool cores = true, particles = false;
};
static FlowCfg g_fcfg;
static bool g_flow_auto_arg = false;  // --flow-auto: set the flow up from the Environment (any design)

struct FMover {
    int comp = -1;
    V3 p, d;                    // axis point and direction (world)
    double w = 0;               // rad/s about d
    Pose base;                  // the component's pose at time 0
    std::vector<float> sd;      // distance grid in the component's frame
    int lx = 0, ly = 0, lz = 0; double lh = 0; V3 lo;
    std::vector<int> cells;     // the cells it can sweep
    double R = 0;               // largest radius from the axis
    double Mx[9]; V3 mv;        // world cell -> local, this step
};

struct FlowSim {
    // grid
    int nx = 0, ny = 0, nz = 0; size_t N = 0; double dx = 0; V3 o;  // o: the centre of cell (0, 0, 0)
    int face[6] = {0, 0, 0, 0, 0, 0};  // -x, +x, -y, +y, -z, +z
    float uin[3] = {0, 0, 0}, uin0[3] = {0, 0, 0};
    int kick_axis = -1; long kick_steps = 0;
    std::vector<int> pinch; long pinch_sum = 0, pinch_upd = 0, pinch_last = 0;  // gaps one cell wide between parts, bridged as wall
    std::vector<unsigned char> nearwall;
    std::vector<float> sponge;
    std::vector<unsigned char> reach;  // a vessel: fluid connected to the turning parts  // a current: the relaxation time of each cell (raised in absorbing layers at the outlet and the cross-stream walls)  // fluid cells within 3 cells of a wall (no vortex cores drawn there)
    double smag = 0;  // 18 sqrt2 Cs^2 when the large-eddy model is on (0: every eddy resolved, plain BGK)
    double dt = 0, ulb = 0.06, Uref = 0, nu = 0, rho = 1000, tau0 = 0.6, Lref = 0, Re = 0, blockage = 0;
    std::string medium;
    int off[LQ];
    std::vector<float> fa, fb; float* src = nullptr; float* dst = nullptr;
    std::vector<unsigned char> flag; std::vector<short> owner;
    std::vector<FMover> mv;
    std::vector<int> bodies;          // components in the flow (for forces)
    // fields for the view
    std::vector<float> ucur, usnap, umean, kvar;  // ucur / usnap: ux, uy, uz, density per cell
    bool thermal = false; std::vector<float> Ta, Tb, Tsnap; float* Tc = nullptr; float* Tn = nullptr;
    double alpha_lb = 0, Tin = 20, Thot = 20, kf = 0.6, beta = 0, glb = 0;
    std::vector<float> compT;               // each component's wall temperature (NaN: adiabatic)
    std::vector<std::vector<double>> hq;    // per thread: per body the heat into the fluid (W) over a snapshot step
    std::vector<std::vector<double>> h_q;   // per body: heat (W) per snapshot
    std::vector<double> barea;              // per body: wetted area on the grid (m2)
    std::vector<float> bvel;                // belts and chains: the band's velocity in its cells (lattice units)
    int nbelts = 0;
    // free surface (Koerner et al. 2005, Thuerey 2007): liquid cells, interface cells (flag fluid, iface 1, mass M and
    // fill M / rho), gas cells (flag gas: the gas at a constant pressure, not simulated)
    bool fs = false; std::vector<unsigned char> iface; std::vector<float> Ma, Mb, Ra, Rb, phisnap;
    float* Mc = nullptr; float* Mn = nullptr; float* Rc = nullptr; float* Rn = nullptr;
    double gfs = 0, mass0 = 0, mass_now = 0, zsurf = 0; long conversions = 0;
    // the loss torque of turning gears by angular position (per tooth: the flank zone and the tip-circle zone)
    struct Polar { int comp = -1; V3 p, d, e1, e2; double rtip = 0, w = 0; int z = 0; };
    std::vector<Polar> pol; std::vector<int> polidx; std::vector<std::vector<double>> pacc; long pol_steps = 0;
    // threads
    int nthr = 1; volatile LONG bar_n = 0, bar_gen = 0;
    volatile LONG stop = 0, done = 0, failed = 0; volatile LONG bad_id = -1; float bad_r = 0;
    std::vector<std::vector<double>> frc;  // per thread: per body 6 (force, torque)
    std::vector<double> stat;              // per thread: 4 (kinetic energy, turbulent ke, max speed, eddy viscosity ratio sum)
    std::vector<double> fsum;              // forces summed over the snapshot interval
    size_t nfluid = 1; long last_step = 0;
    long step = 0, steps_total = 0; double t = 0; int snap_every = 20, mv_every = 4; bool snap_now = false;
    double ema = 0.02;
    // results (written by thread 0, read under the lock)
    std::vector<double> h_t, h_ke, h_tke, h_umax;
    std::vector<std::vector<double>> h_f;  // per body: per snapshot 6
    double ke = 0, tke = 0, umax = 0, nut = 0, mlups = 0; long clamps = 0;
    DWORD t0ms = 0;
    bool snap_new = false;
    CRITICAL_SECTION cs;
};
static FlowSim* g_flow = nullptr;
static std::vector<HANDLE> g_flow_threads;
static bool g_flow_on = false, g_flow_running = false;
static std::vector<Pose> g_flow_base;  // the poses at the start (put back at the end)
static std::string g_flow_msg;
static std::string g_flow_last_dir;

static inline float feq(int q, float r, float ux, float uy, float uz) {
    float cu = 3.f * (LC[q][0] * ux + LC[q][1] * uy + LC[q][2] * uz);
    return LW[q] * r * (1.f + cu + 0.5f * cu * cu - 1.5f * (ux * ux + uy * uy + uz * uz));
}
#include "lbm_collide.inc"
static void flow_barrier(FlowSim& S) {
    LONG g = S.bar_gen;
    if (InterlockedIncrement(&S.bar_n) == S.nthr) { S.bar_n = 0; InterlockedIncrement(&S.bar_gen); }
    else { int spins = 0; while (S.bar_gen == g) { if (++spins > 2000) Sleep(0); } }
}
static double rot_angle_of(const double R[9], V3& axis) {  // the rotation's angle and axis
    Quat q = quat_of(R);
    double s = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z), a = 2 * std::atan2(s, q.w);
    axis = s > 1e-15 ? V3(q.x, q.y, q.z) / s : V3(0, 0, 1);
    if (a > geo::PI) { a = 2 * geo::PI - a; axis = axis * -1.0; }  // q and -q are the same turn: take the short way
    return a;
}

// ---------------------------------------------------------------- the moving parts' solid cells
static void flow_snap_cell(FlowSim& S, int c, float ux, float uy, float uz) {  // a refilled cell's state in both snapshot buffers
    EnterCriticalSection(&S.cs);
    float* a = &S.ucur[4 * (size_t)c]; float* b = &S.usnap[4 * (size_t)c];
    a[0] = b[0] = ux; a[1] = b[1] = uy; a[2] = b[2] = uz; a[3] = b[3] = 1.f;
    LeaveCriticalSection(&S.cs);
}
static void movers_update(FlowSim& S, double t) {
    for (FMover& m : S.mv) {  // world cell -> the component's frame at time 0: turn back by w t about the axis
        double M[9]; rot_axis(m.d, -m.w * t, M);
        const double* B = m.base.R;
        double Bt[9] = {B[0], B[3], B[6], B[1], B[4], B[7], B[2], B[5], B[8]};
        mat_mul(Bt, M, m.Mx);
        V3 c = m.p - mat_vec(M, m.p) - m.base.t;
        m.mv = mat_vec(Bt, c);
    }
    std::vector<int> was_pinch; was_pinch.swap(S.pinch);
    for (int c : was_pinch) if (S.flag[(size_t)c] == FL_MOVING) S.flag[(size_t)c] = FL_FREED;
    for (FMover& m : S.mv) for (int c : m.cells) if (S.flag[(size_t)c] == FL_MOVING && S.owner[(size_t)c] == m.comp) S.flag[(size_t)c] = FL_FREED;
    for (FMover& m : S.mv)
        for (int c : m.cells) {
            unsigned char f = S.flag[(size_t)c];
            if (f == FL_STATIC || (f == FL_MOVING && S.owner[(size_t)c] != m.comp)) continue;  // (gas cells a part sweeps into become part)
            int i = c % S.nx, j = (c / S.nx) % S.ny, k = c / (S.nx * S.ny);
            V3 x = S.o + V3(i * S.dx, j * S.dx, k * S.dx), l = mat_vec(m.Mx, x) + m.mv;
            double gx = (l.x - m.lo.x) / m.lh, gy = (l.y - m.lo.y) / m.lh, gz = (l.z - m.lo.z) / m.lh;
            if (gx < 0 || gy < 0 || gz < 0 || gx >= m.lx - 1 || gy >= m.ly - 1 || gz >= m.lz - 1) continue;
            int a = (int)gx, b = (int)gy, e = (int)gz; double fx = gx - a, fy = gy - b, fz = gz - e;
            auto G = [&](int p, int q, int r) { return (double)m.sd[((size_t)r * m.ly + q) * m.lx + p]; };
            double d = (1 - fz) * ((1 - fy) * ((1 - fx) * G(a, b, e) + fx * G(a + 1, b, e)) + fy * ((1 - fx) * G(a, b + 1, e) + fx * G(a + 1, b + 1, e)))
                     + fz * ((1 - fy) * ((1 - fx) * G(a, b, e + 1) + fx * G(a + 1, b, e + 1)) + fy * ((1 - fx) * G(a, b + 1, e + 1) + fx * G(a + 1, b + 1, e + 1)));
            if (d < 0) { S.flag[(size_t)c] = FL_MOVING; S.owner[(size_t)c] = (short)m.comp; if (S.fs) S.iface[(size_t)c] = 0; }
        }
    for (FMover& m : S.mv)  // cells the part has just left: fluid moving with the part's surface
        for (int c : m.cells) {
            if (S.flag[(size_t)c] != FL_FREED) continue;
            int i = c % S.nx, j = (c / S.nx) % S.ny, k = c / (S.nx * S.ny);
            V3 x = S.o + V3(i * S.dx, j * S.dx, k * S.dx), uw = geo::cross(m.d * (m.w * S.dt), (x - m.p) / S.dx);  // lattice units: radians a step x cells
            for (int q = 0; q < LQ; ++q) S.src[(size_t)q * S.N + (size_t)c] = feq(q, 1.f, (float)uw.x, (float)uw.y, (float)uw.z);
            flow_snap_cell(S, c, (float)uw.x, (float)uw.y, (float)uw.z);
            if (S.thermal) { float tw = m.comp < (int)S.compT.size() ? S.compT[(size_t)m.comp] : NAN; S.Tc[c] = tw == tw ? tw : (float)S.Tin; }
            if (S.fs) {
                bool wet = false;
                for (int q = 1; q <= 6; ++q) { int a = i + LC[q][0], b = j + LC[q][1], e = k + LC[q][2]; if (a >= 0 && b >= 0 && e >= 0 && a < S.nx && b < S.ny && e < S.nz && S.flag[((size_t)e * S.ny + b) * S.nx + a] == FL_FLUID) wet = true; }
                S.iface[(size_t)c] = wet ? 1 : 0; S.Mc[(size_t)c] = 0; S.Rc[(size_t)c] = 1;
                if (!wet) { S.flag[(size_t)c] = FL_GAS; continue; }
            }
            S.flag[(size_t)c] = FL_FLUID;
        }
    for (int c : was_pinch) {  // bridged cells that no part covers now: fluid again, at rest
        if (S.flag[(size_t)c] != FL_FREED) continue;
        for (int q = 0; q < LQ; ++q) S.src[(size_t)q * S.N + (size_t)c] = feq(q, 1.f, 0.f, 0.f, 0.f);
        flow_snap_cell(S, c, 0.f, 0.f, 0.f);
        if (S.thermal) S.Tc[c] = (float)S.Tin;
        if (S.fs) { S.iface[(size_t)c] = 1; S.Mc[(size_t)c] = 0; S.Rc[(size_t)c] = 1; }
        S.flag[(size_t)c] = FL_FLUID;
    }
    if (!g_flow_bridge) return;
    // Gaps one cell wide between two solids, at least one of them moving (meshing teeth, a gear over its plate): the
    // lattice cannot carry a flow there (it gives speeds far beyond its valid range), so such a cell is bridged: it
    // becomes wall moving with the neighbouring part, and is freed again when the gap opens.
    auto solid = [&](int i, int j, int k, short& own) -> int {
        if (i < 0 || j < 0 || k < 0 || i >= S.nx || j >= S.ny || k >= S.nz) return 0;
        size_t id = ((size_t)k * S.ny + j) * S.nx + i; unsigned char f = S.flag[id];
        if (f == FL_MOVING) { own = S.owner[id]; return own >= 0 ? 2 : 1; }
        return f == FL_STATIC ? 1 : 0;
    };
    std::vector<std::pair<int, short>> cand;
    for (FMover& m : S.mv)
        for (int c : m.cells) {
            if (S.flag[(size_t)c] != FL_FLUID) continue;
            int i = c % S.nx, j = (c / S.nx) % S.ny, k = c / (S.nx * S.ny);
            int nsol = 0; short any = -1; bool hit = false;
            for (int q = 1; q < LQ && !hit; ++q) {
                short oa = -1, ob = -1, oc = -1;
                int a = solid(i + LC[q][0], j + LC[q][1], k + LC[q][2], oa);
                if (a) { ++nsol; if (a == 2) any = oa; } else continue;
                int b = solid(i - LC[q][0], j - LC[q][1], k - LC[q][2], ob);
                if (b && (a == 2 || b == 2)) { cand.push_back(std::make_pair(c, a == 2 ? oa : ob)); hit = true; break; }  // one cell wide
            }
        }
    for (auto& p : cand) {
        if (S.flag[(size_t)p.first] != FL_FLUID) continue;
        S.flag[(size_t)p.first] = FL_MOVING; S.owner[(size_t)p.first] = p.second; if (S.fs) S.iface[(size_t)p.first] = 0;
        S.pinch.push_back(p.first);
    }
    S.pinch_last = (long)S.pinch.size(); S.pinch_sum += S.pinch_last; ++S.pinch_upd;
}

// ---------------------------------------------------------------- one step on a slab of z
static void flow_kernel(FlowSim& S, int tid, int k0, int k1) {
    const size_t N = S.N; const int nx = S.nx, ny = S.ny, nz = S.nz;
    const float* src = S.src; float* dst = S.dst;
    const unsigned char* flag = S.flag.data(); const short* owner = S.owner.data();
    int off[LQ]; for (int q = 0; q < LQ; ++q) off[q] = S.off[q];
    const float* sponge = S.sponge.empty() ? nullptr : S.sponge.data();
    const float tau0 = (float)S.tau0, smag = (float)S.smag;
    std::vector<double>& F = S.frc[(size_t)tid];
    double* pacc = S.pacc.empty() ? nullptr : S.pacc[(size_t)tid].data();
    double ke = 0, tke = 0, umax = 0, nut = 0; long clamps = 0;
    const bool snap = S.snap_now; const float ema = (float)S.ema;
    const int nb = (int)S.bodies.size();
    const bool thermal = S.thermal, buoy = g_fcfg.buoyancy; const float* Tc = S.Tc; float* Tn = S.Tn;
    const float Tin = (float)S.Tin, Tref = (float)S.Tin, alpha = (float)S.alpha_lb, glbb = (float)(S.glb * S.beta);
    const float Tlo = (float)std::min(S.Tin, S.Thot) - 0.5f, Thi = (float)std::max(S.Tin, S.Thot) + 0.5f;
    const int nn3[3] = {nx, ny, nz}; const size_t st3[3] = {1, (size_t)nx, (size_t)nx * ny};
    const bool fs = S.fs; const unsigned char* ifc = fs ? S.iface.data() : nullptr; const float* Mc = S.Mc; const float* Rc = S.Rc; float* Mn = S.Mn; float* Rn = S.Rn;
    const float gfs = (float)S.gfs;
    // the wall velocity of a moving part's cell (lattice units) at a link's middle
    auto wall_u = [&](short own, double xi, double yi, double zi, float* uw) {
        uw[0] = uw[1] = uw[2] = 0;
        for (const FMover& m : S.mv) if (m.comp == own) {
            V3 x = S.o + V3(xi * S.dx, yi * S.dx, zi * S.dx), u = geo::cross(m.d * (m.w * S.dt), (x - m.p) / S.dx);
            uw[0] = (float)u.x; uw[1] = (float)u.y; uw[2] = (float)u.z; return;
        }
    };
    auto bidx = [&](short own) { for (int b = 0; b < nb; ++b) if (S.bodies[(size_t)b] == own) return b; return -1; };
    float f[LQ];
    for (int k = k0; k < k1; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                size_t id = ((size_t)k * ny + j) * nx + i;
                if (flag[id] != FL_FLUID) continue;
                bool edge = i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1;
                const bool isif = fs && ifc[id];
                float dm = 0, r0 = 1, u0[3] = {0, 0, 0};
                if (isif) {  // the interface cell's density and velocity from its last populations (for the gas side)
                    r0 = 0; for (int q = 0; q < LQ; ++q) { float v = src[(size_t)q * N + id]; r0 += v; u0[0] += v * LC[q][0]; u0[1] += v * LC[q][1]; u0[2] += v * LC[q][2]; }
                    if (r0 > 1e-6f) { u0[0] /= r0; u0[1] /= r0; u0[2] /= r0; }
                }
                for (int q = 0; q < LQ; ++q) {
                    int ni = i - LC[q][0], nj = j - LC[q][1], nk = k - LC[q][2];
                    if (edge && (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz)) {
                        int ft[3] = {-1, -1, -1};
                        if (ni < 0) ft[0] = S.face[0]; else if (ni >= nx) ft[0] = S.face[1];
                        if (nj < 0) ft[1] = S.face[2]; else if (nj >= ny) ft[1] = S.face[3];
                        if (nk < 0) ft[2] = S.face[4]; else if (nk >= nz) ft[2] = S.face[5];
                        bool in = false, out = false, wall = false;
                        for (int a = 0; a < 3; ++a) { in |= ft[a] == FACE_IN; out |= ft[a] == FACE_OUT; wall |= ft[a] == FACE_WALL; }
                        if (in) f[q] = feq(q, 1.f, S.uin[0], S.uin[1], S.uin[2]);
                        else if (out) f[q] = src[(size_t)q * N + id];
                        else if (wall) f[q] = src[(size_t)LOPP[q] * N + id];
                        else {  // a slip wall: the population that left the cell beside, along the wall, comes back mirrored
                            int r = q, si = ni, sj = nj, sk = nk;
                            for (int a = 0; a < 3; ++a) if (ft[a] == FACE_SLIP) r = LREFL[a][r];
                            if (ft[0] == FACE_SLIP) si = i;
                            if (ft[1] == FACE_SLIP) sj = j;
                            if (ft[2] == FACE_SLIP) sk = k;
                            size_t sid = id;
                            if (si >= 0 && sj >= 0 && sk >= 0 && si < nx && sj < ny && sk < nz) { size_t t2 = ((size_t)sk * ny + sj) * nx + si; if (flag[t2] == FL_FLUID) sid = t2; }
                            f[q] = src[(size_t)r * N + sid];
                        }
                        continue;
                    }
                    size_t n = id - (size_t)off[q];
                    unsigned char fl = flag[n];
                    if (fl == FL_FLUID) {
                        f[q] = src[(size_t)q * N + n];
                        if (isif) {  // mass across the link: in from the neighbour less out to it, weighted by the fill of both
                            float w = ifc[n] ? 0.5f * (Mc[id] / std::max(1e-6f, Rc[id]) + Mc[n] / std::max(1e-6f, Rc[n])) : 1.f;
                            dm += w * (f[q] - src[(size_t)LOPP[q] * N + id]);
                        }
                        continue;
                    }
                    if (fl == FL_GAS) {  // from the gas: rebuilt at the gas pressure (density 1) with the cell's velocity
                        f[q] = feq(q, 1.f, u0[0], u0[1], u0[2]) + feq(LOPP[q], 1.f, u0[0], u0[1], u0[2]) - src[(size_t)LOPP[q] * N + id];
                        continue;
                    }
                    // a wall between: the population that left this cell towards it comes back (moving wall: Ladd)
                    float fo = src[(size_t)LOPP[q] * N + id], uw[3] = {0, 0, 0};
                    if (fl == FL_MOVING) { if (owner[n] == -2 && !S.bvel.empty()) { uw[0] = S.bvel[3 * n]; uw[1] = S.bvel[3 * n + 1]; uw[2] = S.bvel[3 * n + 2]; } else wall_u(owner[n], i - 0.5 * LC[q][0], j - 0.5 * LC[q][1], k - 0.5 * LC[q][2], uw); }
                    f[q] = fo + 6.f * LW[q] * (LC[q][0] * uw[0] + LC[q][1] * uw[1] + LC[q][2] * uw[2]);
                    int b = bidx(owner[n]);
                    if (b >= 0) {  // momentum given to the part: -c (f_out + f_back), at the link's middle
                        double s = (fo + f[q] - 2 * LW[q]) * (isif ? std::max(0.f, std::min(1.f, Mc[id] / std::max(1e-6f, Rc[id]))) : 1.f), Fx = -LC[q][0] * s, Fy = -LC[q][1] * s, Fz = -LC[q][2] * s;  // an interface cell pushes with its liquid share only  // less the fluid at rest: gauge pressure (faces against the domain's walls carry none)
                        double rx = i - 0.5 * LC[q][0], ry = j - 0.5 * LC[q][1], rz = k - 0.5 * LC[q][2];
                        double* Fb = &F[(size_t)b * 6];
                        Fb[0] += Fx; Fb[1] += Fy; Fb[2] += Fz;
                        Fb[3] += ry * Fz - rz * Fy; Fb[4] += rz * Fx - rx * Fz; Fb[5] += rx * Fy - ry * Fx;
                        if (pacc && owner[n] >= 0 && owner[n] < (short)S.polidx.size() && S.polidx[(size_t)owner[n]] >= 0) {
                            int pi = S.polidx[(size_t)owner[n]];
                            const FlowSim::Polar& P = S.pol[(size_t)pi];
                            V3 rel = S.o + V3(rx * S.dx, ry * S.dx, rz * S.dx) - P.p; rel = rel - P.d * geo::dot(rel, P.d);
                            double rr = geo::len(rel), th = std::atan2(geo::dot(rel, P.e2), geo::dot(rel, P.e1));
                            int bin = std::min(NPB - 1, std::max(0, (int)((th + geo::PI) / (2 * geo::PI) * NPB)));
                            int zone = rr > P.rtip - 0.6 * S.dx ? 1 : 0;
                            V3 rc = rel / S.dx, Fv(Fx, Fy, Fz);
                            pacc[((size_t)pi * 2 + (size_t)zone) * NPB + (size_t)bin] += geo::dot(geo::cross(rc, Fv), P.d);
                        }
                    }
                }
                float r, ux, uy, uz, tau; bool bad = false;
                float T0 = thermal ? Tc[id] : 0.f; if (thermal && !(T0 == T0)) T0 = Tin;
                float az = (thermal && buoy ? glbb * (T0 - Tref) : 0.f) - (fs ? gfs : 0.f);  // buoyancy, and gravity on a free-surface liquid
                lbm_collide(f, sponge ? sponge[id] : tau0, smag, r, ux, uy, uz, tau, clamps, bad, 0.f, 0.f, az);
                if (thermal) {  // the temperature: carried by the flow (upwind), spread by conduction; hot walls by a ghost cell
                    float lap = 0, adv = 0, uu[3] = {ux, uy, uz};
                    int ii0[3] = {i, j, k};
                    for (int a = 0; a < 3; ++a) {
                        float tv[2];
                        for (int sd = 0; sd < 2; ++sd) {
                            int ii[3] = {ii0[0], ii0[1], ii0[2]}; ii[a] += sd ? 1 : -1;
                            if (ii[a] < 0 || ii[a] >= nn3[a]) { tv[sd] = S.face[2 * a + sd] == FACE_IN ? Tin : T0; continue; }
                            size_t nid = sd ? id + st3[a] : id - st3[a];
                            if (flag[nid] == FL_FLUID) { tv[sd] = Tc[nid]; continue; }
                            short ow = owner[nid];
                            float tw = ow >= 0 && ow < (short)S.compT.size() ? S.compT[(size_t)ow] : NAN;
                            if (tw == tw) {
                                tv[sd] = 2 * tw - T0;
                                if (snap) { int b = bidx(ow); if (b >= 0) S.hq[(size_t)tid][(size_t)b] += 2 * S.kf * S.dx * (tw - T0); }
                            } else tv[sd] = T0;
                        }
                        lap += tv[0] + tv[1] - 2 * T0;
                        adv += uu[a] > 0 ? uu[a] * (T0 - tv[0]) : uu[a] * (tv[1] - T0);
                    }
                    float cfl = std::fabs(ux) + std::fabs(uy) + std::fabs(uz) + 6 * alpha;  // explicit stability: keep the update a weighted average
                    if (cfl > 0.9f) adv *= 0.9f / cfl;
                    float tn = T0 - adv + alpha * lap;
                    Tn[id] = std::max(Tlo, std::min(Thi, tn));  // never outside the hottest and coldest (the maximum principle)
                }
                if (!bad && (r > 1.6f || r < 0.6f)) {  // a pocket the grid cannot resolve (a gap narrower than a cell between moving walls): reset it
                    float sc = std::sqrt(std::min(1.f, 0.0225f / std::max(1e-12f, ux * ux + uy * uy + uz * uz)));
                    ux *= sc; uy *= sc; uz *= sc;
                    for (int q = 0; q < LQ; ++q) f[q] = feq(q, 1.f, ux, uy, uz);
                    r = 1.f; clamps += 1;
                }
                if (bad) { S.failed = 1; if (S.bad_id < 0) { S.bad_id = (LONG)id; float rr = 0; for (int q = 0; q < LQ; ++q) rr += f[q]; S.bad_r = rr; } }
                float u2 = ux * ux + uy * uy + uz * uz;
                for (int q = 0; q < LQ; ++q) dst[(size_t)q * N + id] = f[q];
                if (fs) { Rn[id] = r; Mn[id] = isif ? Mc[id] + dm : r; }
                if (snap) {
                    float* uc = &S.ucur[4 * id]; uc[0] = ux; uc[1] = uy; uc[2] = uz; uc[3] = r;
                    float* m = &S.umean[3 * id];
                    m[0] += ema * (ux - m[0]); m[1] += ema * (uy - m[1]); m[2] += ema * (uz - m[2]);
                    float du = (ux - m[0]) * (ux - m[0]) + (uy - m[1]) * (uy - m[1]) + (uz - m[2]) * (uz - m[2]);
                    S.kvar[id] += ema * (0.5f * du - S.kvar[id]);
                    ke += 0.5 * r * u2; tke += S.kvar[id]; umax = std::max(umax, (double)u2); { float t0 = sponge ? sponge[id] : tau0; nut += (tau - t0) / (t0 - 0.5f); }  // against the cell's own molecular part (the absorbing layers raise it)
                }
            }
    double* st = &S.stat[(size_t)tid * 5];
    st[0] += ke; st[1] += tke; st[2] = std::max(st[2], umax); st[3] += nut; st[4] += (double)clamps;
}


// the free surface after a step: interface cells that filled become liquid (their gas neighbours interface), those that
// emptied become gas (their liquid neighbours interface); the mass over or under goes to the interface around
static void fs_update(FlowSim& S) {
    const size_t N = S.N;
    std::vector<size_t> fill, empty;
    for (size_t id = 0; id < N; ++id) {
        if (!S.iface[id] || S.flag[id] != FL_FLUID) continue;
        float phi = S.Mn[id] / std::max(1e-6f, S.Rn[id]);
        if (phi > 1.001f) fill.push_back(id); else if (phi < -0.001f) empty.push_back(id);
    }
    auto nb = [&](size_t id, int q, size_t& n) {
        int i = (int)(id % S.nx) + LC[q][0], j = (int)((id / S.nx) % S.ny) + LC[q][1], k = (int)(id / ((size_t)S.nx * S.ny)) + LC[q][2];
        if (i < 0 || j < 0 || k < 0 || i >= S.nx || j >= S.ny || k >= S.nz) return false;
        n = ((size_t)k * S.ny + j) * S.nx + i; return true;
    };
    for (size_t id : fill) {
        S.iface[id] = 0;
        for (int q = 1; q < LQ; ++q) {
            size_t n; if (!nb(id, q, n) || S.flag[n] != FL_GAS) continue;
            // a new interface cell: the average density and velocity of the wet cells around it
            float r = 0, u[3] = {0, 0, 0}; int c = 0;
            for (int q2 = 1; q2 < LQ; ++q2) {
                size_t m; if (!nb(n, q2, m) || S.flag[m] != FL_FLUID) continue;
                float rr = 0, uu[3] = {0, 0, 0};
                for (int a = 0; a < LQ; ++a) { float v = S.dst[(size_t)a * N + m]; rr += v; uu[0] += v * LC[a][0]; uu[1] += v * LC[a][1]; uu[2] += v * LC[a][2]; }
                if (rr > 1e-6f) { r += rr; for (int d = 0; d < 3; ++d) u[d] += uu[d] / rr; ++c; }
            }
            if (c) { r /= c; for (int d = 0; d < 3; ++d) u[d] /= c; } else r = 1;
            for (int a = 0; a < LQ; ++a) S.dst[(size_t)a * N + n] = feq(a, r, u[0], u[1], u[2]);
            S.flag[n] = FL_FLUID; S.iface[n] = 1; S.Mn[n] = 0; S.Rn[n] = r;
            ++S.conversions;
        }
    }
    for (size_t id : empty) {
        bool keep = false;  // next to a cell that just filled: stay interface this step
        for (int q = 1; q < LQ && !keep; ++q) { size_t n; if (nb(id, q, n) && S.flag[n] == FL_FLUID && !S.iface[n] && S.Mn[n] > 0.999f * S.Rn[n] && std::find(fill.begin(), fill.end(), n) != fill.end()) keep = true; }
        if (keep) continue;
        S.flag[id] = FL_GAS; S.iface[id] = 0;
        for (int q = 1; q < LQ; ++q) { size_t n; if (nb(id, q, n) && S.flag[n] == FL_FLUID && !S.iface[n]) S.iface[n] = 1; }
        ++S.conversions;
    }
    auto spread = [&](size_t id, float ex) {  // to the interface cells around, evenly
        size_t list[LQ]; int c = 0;
        for (int q = 1; q < LQ; ++q) { size_t n; if (nb(id, q, n) && S.flag[n] == FL_FLUID && S.iface[n]) list[c++] = n; }
        for (int k = 0; k < c; ++k) S.Mn[list[k]] += ex / c;
    };
    for (size_t id : fill) { float ex = S.Mn[id] - S.Rn[id]; S.Mn[id] = S.Rn[id]; spread(id, ex); }
    for (size_t id : empty) if (S.flag[id] == FL_GAS) { float ex = S.Mn[id]; S.Mn[id] = 0; spread(id, ex); }
}

static unsigned __stdcall flow_worker(void* arg) {
    FlowSim& S = *g_flow;
    int tid = (int)(INT_PTR)arg;
    int k0 = (int)((long long)S.nz * tid / S.nthr), k1 = (int)((long long)S.nz * (tid + 1) / S.nthr);
    for (;;) {
        if (tid == 0) {
            if (S.stop || S.failed || S.step >= S.steps_total) S.done = 1;
            else {
                if (!S.mv.empty() && S.step % S.mv_every == 0) movers_update(S, S.t);
                if (!S.pacc.empty() && S.step == S.steps_total / 2) { for (auto& v : S.pacc) std::fill(v.begin(), v.end(), 0.0); S.pol_steps = 0; }
                if (!S.pacc.empty()) ++S.pol_steps;
                if (S.kick_axis >= 0) { double ph = S.step < S.kick_steps ? std::sin(geo::PI * S.step / (double)S.kick_steps) : 0.0; S.uin[S.kick_axis] = S.uin0[S.kick_axis] + (float)(0.15 * S.ulb * ph * ph); }
                S.snap_now = (S.step + 1) % S.snap_every == 0;
                if (S.snap_now) std::fill(S.stat.begin(), S.stat.end(), 0.0);
            }
        }
        flow_barrier(S);
        if (S.done) break;
        flow_kernel(S, tid, k0, k1);
        flow_barrier(S);
        if (tid == 0) {
            if (S.fs) fs_update(S);
            std::swap(S.src, S.dst);
            if (S.fs) { std::swap(S.Mc, S.Mn); std::swap(S.Rc, S.Rn); }
            if (S.thermal) std::swap(S.Tc, S.Tn);
            for (int t = 0; t < S.nthr; ++t) { std::vector<double>& F = S.frc[(size_t)t]; for (size_t i = 0; i < F.size(); ++i) { S.fsum[i] += F[i]; F[i] = 0; } }
            ++S.step; S.t += S.dt;
            if (S.snap_now) {
                double ke = 0, tke = 0, um = 0, nut = 0; long cl = 0; size_t nf = 0;
                for (int t = 0; t < S.nthr; ++t) { const double* s = &S.stat[(size_t)t * 5]; ke += s[0]; tke += s[1]; um = std::max(um, s[2]); nut += s[3]; cl += (long)s[4]; }
                nf = S.nfluid;
                double cell_m = S.rho * S.dx * S.dx * S.dx, vel = S.dx / S.dt;  // lattice -> SI
                EnterCriticalSection(&S.cs);
                std::swap(S.ucur, S.usnap);
                S.ke = ke * cell_m * vel * vel; S.tke = tke * cell_m * vel * vel; S.umax = std::sqrt(um) * vel; S.nut = nf ? nut / nf : 0; S.clamps += cl;
                if (S.fs) {
                    double ms = 0;
                    for (size_t i = 0; i < S.N; ++i) { float ph = S.flag[i] == FL_FLUID ? (S.iface[i] ? std::max(0.f, std::min(1.f, S.Mc[i] / std::max(1e-6f, S.Rc[i]))) : 1.f) : 0.f; S.phisnap[i] = ph; if (S.flag[i] == FL_FLUID) ms += S.Mc[i]; }
                    S.mass_now = ms;
                }
                if (S.thermal) {
                    memcpy(S.Tsnap.data(), S.Tc, S.N * sizeof(float));
                    for (size_t b = 0; b < S.bodies.size(); ++b) { double q = 0; for (int t = 0; t < S.nthr; ++t) { q += S.hq[(size_t)t][b]; S.hq[(size_t)t][b] = 0; } S.h_q[b].push_back(q); }
                }
                S.h_t.push_back(S.t); S.h_ke.push_back(S.ke); S.h_tke.push_back(S.tke); S.h_umax.push_back(S.umax);
                double fu = S.rho * std::pow(S.dx, 4) / (S.dt * S.dt), tu = fu * S.dx;  // force and torque units
                for (size_t b = 0; b < S.bodies.size(); ++b) {
                    std::vector<double> v(6);
                    for (int a = 0; a < 3; ++a) v[(size_t)a] = S.fsum[b * 6 + (size_t)a] / S.snap_every * fu;
                    // torque about the grid origin -> about the world origin: r_world = o + r dx
                    V3 Fw(v[0], v[1], v[2]), Tg(S.fsum[b * 6 + 3] / S.snap_every * tu, S.fsum[b * 6 + 4] / S.snap_every * tu, S.fsum[b * 6 + 5] / S.snap_every * tu);
                    V3 T = Tg + geo::cross(S.o, Fw);
                    v[3] = T.x; v[4] = T.y; v[5] = T.z;
                    S.h_f[b].insert(S.h_f[b].end(), v.begin(), v.end());
                }
                std::fill(S.fsum.begin(), S.fsum.end(), 0.0);
                DWORD now = GetTickCount(), ms = now - S.t0ms;  // over the last interval
                if (ms > 0) S.mlups = 0.7 * S.mlups + 0.3 * (double)(S.step - S.last_step) * nf / (ms * 1e3);
                S.t0ms = now; S.last_step = S.step;
                S.snap_new = true;
                LeaveCriticalSection(&S.cs);
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------- setting up
static double flow_dir_sign(int d) { return d % 2 == 0 ? 1.0 : -1.0; }
static void flow_free() {
    if (!g_flow) return;
    g_flow->stop = 1;
    if (!g_flow_threads.empty()) { WaitForMultipleObjects((DWORD)g_flow_threads.size(), g_flow_threads.data(), TRUE, INFINITE); for (HANDLE h : g_flow_threads) CloseHandle(h); g_flow_threads.clear(); }
    DeleteCriticalSection(&g_flow->cs);
    delete g_flow; g_flow = nullptr;
}
static void flow_restore_poses() {
    if (g_flow_base.size() == g_ac.size()) for (size_t c = 0; c < g_ac.size(); ++c) g_ac[c].pose = g_flow_base[c];
    g_flow_base.clear();
}
// what turns, how fast and about which axis: from the motion study a moment after time 0
static std::vector<FMover> flow_movers() {
    std::vector<FMover> out;
    if (g_amot.empty()) return out;
    ms_reset();
    std::vector<Pose> p0(g_ac.size()); for (size_t c = 0; c < g_ac.size(); ++c) p0[c] = g_ac[c].pose;
    const double h = 1e-3;
    ms_eval(h);
    for (size_t c = 0; c < g_ac.size(); ++c) {
        const Pose &A = p0[c], &B = g_ac[c].pose;
        double At[9] = {A.R[0], A.R[3], A.R[6], A.R[1], A.R[4], A.R[7], A.R[2], A.R[5], A.R[8]}, Rr[9];
        mat_mul(B.R, At, Rr);
        V3 ax; double ang = rot_angle_of(Rr, ax);
        if (ang < 1e-7) continue;
        FMover m; m.comp = (int)c; m.d = ax; m.w = ang / h; m.base = A;
        V3 dsp = (B.t - A.t) / h, wv = ax * m.w;
        m.p = A.t + geo::cross(wv, dsp) / (m.w * m.w);  // the axis through the point that does not move
        out.push_back(m);
    }
    ms_reset();
    return out;
}
static geo::Box flow_wbox(const AComp& C) {  // the component's box in world coordinates
    geo::Box b = comp_box(C), w;
    if (!b.valid()) return w;
    for (int k = 0; k < 8; ++k) w.add(C.pose.apply(V3((k & 1) ? b.hi.x : b.lo.x, (k & 2) ? b.hi.y : b.lo.y, (k & 4) ? b.hi.z : b.lo.z)));
    return w;
}
static std::string flow_id(int c) { return c >= 0 && c < (int)g_ac.size() && g_flow_base.size() == g_ac.size() ? g_ac[(size_t)c].id : "part" + tostr(c + 1); }
static std::string flow_build(FlowSim& S) {
    // the parts' box
    geo::Box B; bool any = false;
    for (size_t c = 0; c < g_ac.size(); ++c) if (comp_ready(g_ac[c])) { geo::Box b = flow_wbox(g_ac[c]); B.add(b.lo); B.add(b.hi); any = true; }
    if (!any) return "nothing to put in the flow yet (the parts are still meshing)";
    env_from_doc();
    Fluid F = fluid_props(g_env.medium, g_env.Tc, g_env.alt);
    if (F.vacuum) return "a vacuum has no flow: choose a gas or a liquid in Environment";
    S.medium = F.name + " at " + fmtd(g_env.Tc) + " C"; S.rho = F.rho; S.nu = F.mu / F.rho;
    S.mv = flow_movers();
    V3 ext = B.hi - B.lo;
    geo::Box D = B;
    const FlowCfg& c = g_fcfg;
    int ax = c.dir / 2; double sg = flow_dir_sign(c.dir);
    int p1 = (ax + 1) % 3, p2 = (ax + 2) % 3;
    double Lref = 0;
    if (c.mode == 0) {
        int span = c.span_axis == p1 || c.span_axis == p2 ? c.span_axis : ext[p1] >= ext[p2] ? p1 : p2, other = span == p1 ? p2 : p1;
        double Dc = c.span_walls ? ext[other] : std::max(std::min(ext[p1], ext[p2]), 0.3 * std::max(ext[p1], ext[p2]));
        Lref = Dc;
        double up = 4 * Dc, down = 10 * Dc;
        V3 lo = B.lo, hi = B.hi;
        if (sg > 0) { lo[ax] -= up; hi[ax] += down; } else { lo[ax] -= down; hi[ax] += up; }
        lo[other] -= 6 * Dc; hi[other] += 6 * Dc;  // under 8 % blockage for a cylinder
        if (!c.span_walls) { lo[span] -= 6 * Dc; hi[span] += 6 * Dc; }
        D = geo::Box(); D.add(lo); D.add(hi);
        S.Uref = c.U;
        if (S.Uref <= 0) return "give the current a speed above 0";
    } else {
        double m = c.margin * std::max(ext.x, std::max(ext.y, ext.z));
        D = geo::Box(); D.add(B.lo - V3(m, m, m)); D.add(B.hi + V3(m, m, m));
        if (S.mv.empty() && !(c.heat > 0 && std::fabs((c.Thot > 0 ? c.Thot : g_env.Tpart) - g_env.Tc) > 1e-6))
            return "nothing moves the fluid: give a part a motor (Motion > Motor), a current (Environment), or hot parts (Heat)";
    }
    V3 de = D.hi - D.lo;
    double L = std::max(de.x, std::max(de.y, de.z));
    S.dx = L / std::max(16, c.cells);
    S.nx = std::max(4, (int)std::ceil(de.x / S.dx)); S.ny = std::max(4, (int)std::ceil(de.y / S.dx)); S.nz = std::max(4, (int)std::ceil(de.z / S.dx));
    S.N = (size_t)S.nx * S.ny * S.nz;
    if (S.N > 3500000) return "the grid would have " + tostr((int)(S.N / 1000)) + " thousand cells (" + tostr((int)(S.N * 190 / 1048576)) + " MB): use fewer cells";
    S.o = D.lo + V3(0.5 * S.dx, 0.5 * S.dx, 0.5 * S.dx);
    if (c.mode == 0 && c.span_walls) {  // the span: whole cells inside the parts' length (no ends for the flow to pass)
        int span = c.span_axis == p1 || c.span_axis == p2 ? c.span_axis : ext[p1] >= ext[p2] ? p1 : p2;
        int n = std::max(1, (int)std::floor(ext[span] / S.dx - 1e-9));
        int* nn[3] = {&S.nx, &S.ny, &S.nz}; *nn[span] = n;
        S.o[span] = 0.5 * (B.lo[span] + B.hi[span]) - 0.5 * (n - 1) * S.dx;
        S.N = (size_t)S.nx * S.ny * S.nz;
    }
    for (int q = 0; q < LQ; ++q) S.off[q] = LC[q][0] + LC[q][1] * S.nx + LC[q][2] * S.nx * S.ny;
    for (int a = 0; a < 3; ++a) for (int q = 0; q < LQ; ++q) { int cc[3] = {LC[q][0], LC[q][1], LC[q][2]}; cc[a] = -cc[a]; for (int r = 0; r < LQ; ++r) if (LC[r][0] == cc[0] && LC[r][1] == cc[1] && LC[r][2] == cc[2]) LREFL[a][q] = r; }
    // faces
    if (c.mode == 0) {
        for (int f = 0; f < 6; ++f) S.face[f] = FACE_SLIP;
        S.face[2 * ax + (sg > 0 ? 0 : 1)] = FACE_IN; S.face[2 * ax + (sg > 0 ? 1 : 0)] = FACE_OUT;
    } else {
        for (int f = 0; f < 6; ++f) S.face[f] = FACE_WALL;
        S.face[5] = FACE_SLIP;  // a flat free surface on top
    }
    // the speed scale, the time step and the relaxation time
    for (FMover& m : S.mv) {
        double R = 0;
        const AComp& C = g_ac[(size_t)m.comp];
        for (const APiece& pc : C.pieces) if (pc.mesh) for (const V3& v : pc.mesh->m.v) { V3 w = C.pose.apply(pc.local.apply(v)) - m.p; w = w - m.d * geo::dot(w, m.d); R = std::max(R, geo::len(w)); }
        m.R = R;
        if (c.mode >= 1) { if (std::fabs(m.w) * R > S.Uref) { S.Uref = std::fabs(m.w) * R; Lref = 2 * R; } }
        else S.Uref = std::max(S.Uref, std::fabs(m.w) * R);
    }
    if (S.Uref <= 0 && S.mv.empty() && c.mode >= 1) {  // only buoyancy: the free-convection velocity sqrt(g beta dT L)
        double L = std::max(ext.x, std::max(ext.y, ext.z)), dT = std::fabs((c.Thot > 0 ? c.Thot : g_env.Tpart) - g_env.Tc);
        S.Uref = std::max(1e-4, std::sqrt(g_env.g * std::max(F.beta, 1e-6) * dT * L)); Lref = L;
    }
    if (S.Uref <= 0) return "nothing moves the fluid";
    S.Lref = Lref;
    S.ulb = c.mode == 0 ? 0.08 : 0.06;
    S.dt = S.ulb * S.dx / S.Uref;
    double nulb = S.nu * S.dt / (S.dx * S.dx);
    S.tau0 = std::max(0.5 + 1e-4, 3 * nulb + 0.5);
    S.smag = S.tau0 - 0.5 >= 0.02 ? 0.0 : 18 * 1.41421356 * 0.16 * 0.16;  // the model only where the grid cannot carry the viscous scales (it over-damps laminar shear layers)
    if (c.mode >= 1 && !S.mv.empty()) { double Nrev = 0; for (FMover& m : S.mv) if (2 * m.R >= Lref * 0.999) Nrev = std::fabs(m.w) / (2 * geo::PI); S.Re = Nrev * Lref * Lref / S.nu; }
    else if (c.mode >= 1) S.Re = S.Uref * Lref / S.nu;
    else S.Re = S.Uref * Lref / S.nu;
    for (int a = 0; a < 3; ++a) S.uin[a] = 0;
    if (c.mode == 0) { S.uin[ax] = (float)(sg * S.ulb); for (int a = 0; a < 3; ++a) S.uin0[a] = S.uin[a]; S.kick_axis = (ax + 1) % 3 == 2 ? (ax + 2) % 3 : (ax + 1) % 3; S.kick_steps = (long)std::ceil(5 * Lref / S.Uref / S.dt); }
    // the solid cells: static parts from their exact distance functions, the turning ones from a local grid
    S.flag.assign(S.N, FL_FLUID); S.owner.assign(S.N, -1);
    std::vector<char> moving(g_ac.size(), 0); for (FMover& m : S.mv) moving[(size_t)m.comp] = 1;
    for (size_t ci = 0; ci < g_ac.size(); ++ci) {
        const AComp& C = g_ac[ci];
        if (!comp_ready(C)) continue;
        S.bodies.push_back((int)ci);
        geo::Box b = flow_wbox(C);
        if (moving[ci]) continue;
        int i0 = std::max(0, (int)std::floor((b.lo.x - S.o.x) / S.dx) - 1), i1 = std::min(S.nx - 1, (int)std::ceil((b.hi.x - S.o.x) / S.dx) + 1);
        int j0 = std::max(0, (int)std::floor((b.lo.y - S.o.y) / S.dx) - 1), j1 = std::min(S.ny - 1, (int)std::ceil((b.hi.y - S.o.y) / S.dx) + 1);
        int k0 = std::max(0, (int)std::floor((b.lo.z - S.o.z) / S.dx) - 1), k1 = std::min(S.nz - 1, (int)std::ceil((b.hi.z - S.o.z) / S.dx) + 1);
        for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i) {
            size_t id = ((size_t)k * S.ny + j) * S.nx + i;
            if (S.flag[id] != FL_FLUID) continue;
            if (comp_sd((int)ci, S.o + V3(i * S.dx, j * S.dx, k * S.dx)) < 0) { S.flag[id] = FL_STATIC; S.owner[id] = (short)ci; }
        }
    }
    for (FMover& m : S.mv) {  // a distance grid in the part's own frame, and the cells it can sweep
        const AComp& C = g_ac[(size_t)m.comp];
        geo::Box lb; for (const APiece& pc : C.pieces) if (pc.mesh) for (const V3& v : pc.mesh->m.v) lb.add(pc.local.apply(v));
        m.lh = 0.6 * S.dx; m.lo = lb.lo - V3(2 * m.lh, 2 * m.lh, 2 * m.lh);
        V3 le = lb.hi - lb.lo;
        m.lx = (int)std::ceil(le.x / m.lh) + 5; m.ly = (int)std::ceil(le.y / m.lh) + 5; m.lz = (int)std::ceil(le.z / m.lh) + 5;
        if ((double)m.lx * m.ly * m.lz > 6e6) { double s = std::cbrt((double)m.lx * m.ly * m.lz / 6e6); m.lh *= s; m.lx = (int)std::ceil(le.x / m.lh) + 5; m.ly = (int)std::ceil(le.y / m.lh) + 5; m.lz = (int)std::ceil(le.z / m.lh) + 5; }
        m.sd.assign((size_t)m.lx * m.ly * m.lz, 1.f);
        for (int k = 0; k < m.lz; ++k) for (int j = 0; j < m.ly; ++j) for (int i = 0; i < m.lx; ++i)
            m.sd[((size_t)k * m.ly + j) * m.lx + i] = (float)comp_sd(m.comp, C.pose.apply(m.lo + V3(i * m.lh, j * m.lh, k * m.lh)));
        double amin = 1e30, amax = -1e30;
        for (const APiece& pc : C.pieces) if (pc.mesh) for (const V3& v : pc.mesh->m.v) { double a = geo::dot(C.pose.apply(pc.local.apply(v)) - m.p, m.d); amin = std::min(amin, a); amax = std::max(amax, a); }
        for (size_t id = 0; id < S.N; ++id) {
            int i = (int)(id % S.nx), j = (int)((id / S.nx) % S.ny), k = (int)(id / ((size_t)S.nx * S.ny));
            V3 w = S.o + V3(i * S.dx, j * S.dx, k * S.dx) - m.p;
            double a = geo::dot(w, m.d); if (a < amin - S.dx || a > amax + S.dx) continue;
            if (geo::len(w - m.d * a) > m.R + 1.5 * S.dx) continue;
            m.cells.push_back((int)id);
        }
    }
    // populations at rest (vessel) or moving with the current, a slight sideways push behind the parts to start the shedding
    S.fa.assign(S.N * LQ, 0.f); S.fb.assign(S.N * LQ, 0.f); S.src = S.fa.data(); S.dst = S.fb.data();
    for (size_t id = 0; id < S.N; ++id) {
        float u[3] = {S.uin[0], S.uin[1], S.uin[2]};
        if (c.mode == 0) {
            int i = (int)(id % S.nx), j = (int)((id / S.nx) % S.ny);
            int across = (ax + 1) % 3 == 2 ? (ax + 2) % 3 : (ax + 1) % 3;
            int ii[3] = {i, j, (int)(id / ((size_t)S.nx * S.ny))};
            int nn[3] = {S.nx, S.ny, S.nz};
            (void)ii; (void)nn; (void)across;
        }
        for (int q = 0; q < LQ; ++q) { float e = feq(q, 1.f, u[0], u[1], u[2]); S.fa[(size_t)q * S.N + id] = e; S.fb[(size_t)q * S.N + id] = e; }
    }
    S.ucur.assign(4 * S.N, 0.f); S.usnap.assign(4 * S.N, 0.f);
    for (size_t i = 0; i < S.N; ++i) { S.ucur[4 * i + 3] = 1.f; S.usnap[4 * i + 3] = 1.f; }  // at rest, density 1 (cells that were never fluid show this)
    S.umean.assign(3 * S.N, 0.f); S.kvar.assign(S.N, 0.f);
    for (size_t id = 0; id < S.N; ++id) for (int a = 0; a < 3; ++a) S.umean[3 * id + (size_t)a] = S.uin[a];
    // run length: 60 crossings of the body by the current, or 6 turns of the fastest part
    double Tc = c.mode == 0 ? Lref / S.Uref : 2 * geo::PI / std::max(1e-9, S.mv.empty() ? 1.0 : [&] { double w = 0; for (FMover& m : S.mv) w = std::max(w, std::fabs(m.w)); return w; }());
    double dur = c.duration > 0 ? c.duration : (c.mode == 0 ? 60 * Tc : 6 * Tc);
    S.steps_total = std::max(10L, (long)std::ceil(dur / S.dt));
    S.snap_every = std::max(10, std::min(200, (int)std::round(Tc / S.dt / 40)));
    S.ema = std::min(0.2, S.snap_every * S.dt / (2 * Tc));  // the running mean over about two characteristic times
    double tip = 0; for (FMover& m : S.mv) tip = std::max(tip, std::fabs(m.w) * (m.R + S.dx) * S.dt / S.dx);
    S.mv_every = std::max(1, std::min(8, (int)std::floor(0.25 / std::max(1e-9, tip))));
    S.h_f.assign(S.bodies.size(), std::vector<double>());
    if (c.mode == 0) {  // the share of the cross-section the parts block
        size_t blocked = 0, tot = 0;
        int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3, nn[3] = {S.nx, S.ny, S.nz};
        for (int u1 = 0; u1 < nn[a1]; ++u1) for (int u2 = 0; u2 < nn[a2]; ++u2) {
            bool hit = false;
            for (int s = 0; s < nn[ax] && !hit; ++s) { int ii[3]; ii[ax] = s; ii[a1] = u1; ii[a2] = u2; size_t id = ((size_t)ii[2] * S.ny + ii[1]) * S.nx + ii[0]; hit = S.flag[id] != FL_FLUID; }
            if (hit) ++blocked; ++tot;
        }
        S.blockage = tot ? (double)blocked / tot : 0;
        // moving parts in a current are cut once at their place
    }
    // temperature: the fluid at the Environment's temperature, hot parts at theirs
    S.Tin = g_env.Tc; S.Thot = c.Thot > 0 ? c.Thot : g_env.Tpart; S.kf = F.k; S.beta = F.beta;
    S.thermal = c.heat > 0 && std::fabs(S.Thot - S.Tin) > 1e-6;
    S.compT.assign(g_ac.size(), NAN);
    if (S.thermal) {
        for (size_t ci = 0; ci < g_ac.size(); ++ci) { bool turning = false; for (FMover& m : S.mv) if (m.comp == (int)ci) turning = true; if (c.heat == 2 || turning) S.compT[ci] = (float)S.Thot; }
        S.alpha_lb = F.k / (F.rho * F.cp) * S.dt / (S.dx * S.dx);
        S.glb = g_env.g * S.dt * S.dt / S.dx;
        S.Ta.assign(S.N, (float)S.Tin); S.Tb.assign(S.N, (float)S.Tin); S.Tsnap.assign(S.N, (float)S.Tin);
        S.Tc = S.Ta.data(); S.Tn = S.Tb.data();
    }
    if (!S.mv.empty()) movers_update(S, 0);
    // belts and chains: a band moving along its loop at the driving wheel's surface speed
    for (const AMate& mt : g_am) {
        if (!mt.on || mt.type != MT_BELT) continue;
        DriveLoop D = drive_loop(mt);
        if (!D.ok) continue;
        double wn = 0; for (FMover& m : S.mv) if (m.comp == mt.a.comp) wn = m.w * geo::dot(m.d, D.n);
        double vb = D.rA * wn * S.dt / S.dx;  // lattice units
        double o0, o1, hw;
        if (mt.drive == 2) { const js::Value* sm = g_ac[(size_t)mt.a.comp].part.find("sprocket"); double hp = sm ? sm->num("plate_h", 0.012) : 0.012, b1 = sm ? sm->num("inner_width", 0.008) : 0.008, tp = sm ? sm->num("plate_t", 0.0016) : 0.0016; o0 = -0.5 * hp; o1 = 0.5 * hp; hw = 0.5 * b1 + 2 * tp; }
        else { o0 = -0.006; o1 = 0.002; hw = 0.0047; }
        if (S.bvel.empty()) S.bvel.assign(3 * S.N, 0.f);
        int ns = (int)std::ceil(D.L / (0.4 * S.dx));
        for (int si = 0; si < ns; ++si) {
            V3 t; V3 pp = drive_at(D, D.L * si / ns, t); V3 ov = geo::cross(t, D.n);
            for (double a = o0; a <= o1 + 1e-12; a += 0.45 * S.dx) for (double w = -hw; w <= hw + 1e-12; w += 0.45 * S.dx) {
                V3 x = pp + ov * a + D.n * w;
                int ci = (int)std::round((x.x - S.o.x) / S.dx), cj = (int)std::round((x.y - S.o.y) / S.dx), ck = (int)std::round((x.z - S.o.z) / S.dx);
                if (ci < 0 || cj < 0 || ck < 0 || ci >= S.nx || cj >= S.ny || ck >= S.nz) continue;
                size_t id = ((size_t)ck * S.ny + cj) * S.nx + ci;
                if (S.flag[id] == FL_STATIC) continue;
                S.flag[id] = FL_MOVING; S.owner[id] = -2;
                S.bvel[3 * id] = (float)(t.x * vb); S.bvel[3 * id + 1] = (float)(t.y * vb); S.bvel[3 * id + 2] = (float)(t.z * vb);
            }
        }
        ++S.nbelts;
    }
    if (S.nbelts > 0) for (int pass = 0; pass < 3; ++pass) {  // pockets of fluid shut inside a band join it (they would gain mass from the moving walls)
        std::vector<size_t> fill;
        for (size_t id = 0; id < S.N; ++id) {
            if (S.flag[id] != FL_FLUID) continue;
            int ii[3] = {(int)(id % S.nx), (int)((id / S.nx) % S.ny), (int)(id / ((size_t)S.nx * S.ny))};
            int solid = 0, belt = 0;
            for (int q = 1; q < LQ; ++q) {
                int a2 = ii[0] + LC[q][0], b2 = ii[1] + LC[q][1], e2 = ii[2] + LC[q][2];
                if (a2 < 0 || b2 < 0 || e2 < 0 || a2 >= S.nx || b2 >= S.ny || e2 >= S.nz) continue;
                size_t n = ((size_t)e2 * S.ny + b2) * S.nx + a2;
                if (S.flag[n] != FL_FLUID) { ++solid; if (S.owner[n] == -2) ++belt; }
            }
            if (belt > 0 && solid >= 10) fill.push_back(id);
        }
        if (fill.empty()) break;
        for (size_t id : fill) {
            int ii[3] = {(int)(id % S.nx), (int)((id / S.nx) % S.ny), (int)(id / ((size_t)S.nx * S.ny))};
            float v[3] = {0, 0, 0}; int nv = 0;
            for (int q = 1; q < LQ; ++q) {
                int a2 = ii[0] + LC[q][0], b2 = ii[1] + LC[q][1], e2 = ii[2] + LC[q][2];
                if (a2 < 0 || b2 < 0 || e2 < 0 || a2 >= S.nx || b2 >= S.ny || e2 >= S.nz) continue;
                size_t n = ((size_t)e2 * S.ny + b2) * S.nx + a2;
                if (S.owner[n] == -2 && S.flag[n] == FL_MOVING) { for (int c3 = 0; c3 < 3; ++c3) v[c3] += S.bvel[3 * n + c3]; ++nv; }
            }
            S.flag[id] = FL_MOVING; S.owner[id] = -2;
            for (int c3 = 0; c3 < 3; ++c3) S.bvel[3 * id + c3] = nv ? v[c3] / nv : 0.f;
        }
    }
    // each body's wetted area on the grid (faces between its cells and the fluid)
    S.barea.assign(S.bodies.size(), 0.0);
    for (size_t id = 0; id < S.N; ++id) {
        if (S.flag[id] != FL_FLUID) continue;
        int ii[3] = {(int)(id % S.nx), (int)((id / S.nx) % S.ny), (int)(id / ((size_t)S.nx * S.ny))}, nn[3] = {S.nx, S.ny, S.nz}; size_t stp[3] = {1, (size_t)S.nx, (size_t)S.nx * S.ny};
        for (int a = 0; a < 3; ++a) for (int sd = -1; sd <= 1; sd += 2) {
            int q = ii[a] + sd; if (q < 0 || q >= nn[a]) continue;
            size_t nid = sd > 0 ? id + stp[a] : id - stp[a];
            if (S.flag[nid] == FL_FLUID) continue;
            for (size_t b = 0; b < S.bodies.size(); ++b) if (S.bodies[b] == S.owner[nid]) S.barea[b] += S.dx * S.dx;
        }
    }
    S.h_q.assign(S.bodies.size(), std::vector<double>());
    if (c.mode == 2) {  // the bath: liquid below the surface, gas above, interface cells between (filled to their share)
        S.fs = true;
        S.zsurf = g_env.surface > S.o.z && g_env.surface < S.o.z + (S.nz - 1) * S.dx ? g_env.surface : S.o.z + 0.4 * (S.nz - 1) * S.dx;
        S.gfs = g_env.g * S.dt * S.dt / S.dx;
        S.iface.assign(S.N, 0); S.Ma.assign(S.N, 1.f); S.Mb.assign(S.N, 1.f); S.Ra.assign(S.N, 1.f); S.Rb.assign(S.N, 1.f); S.phisnap.assign(S.N, 0.f);
        S.Mc = S.Ma.data(); S.Mn = S.Mb.data(); S.Rc = S.Ra.data(); S.Rn = S.Rb.data();
        for (size_t id = 0; id < S.N; ++id) {
            if (S.flag[id] != FL_FLUID) continue;
            int k = (int)(id / ((size_t)S.nx * S.ny));
            double zc = S.o.z + k * S.dx, fill = geo::clampd((S.zsurf - (zc - 0.5 * S.dx)) / S.dx, 0, 1);
            if (fill <= 0) { S.flag[id] = FL_GAS; S.Ma[id] = S.Mb[id] = 0; }
            else if (fill < 1) { S.iface[id] = 1; S.Ma[id] = S.Mb[id] = (float)fill; }
        }
        for (size_t id = 0; id < S.N; ++id) {  // no liquid cell may touch the gas: those that do become interface
            if (S.flag[id] != FL_FLUID || S.iface[id]) continue;
            int ii[3] = {(int)(id % S.nx), (int)((id / S.nx) % S.ny), (int)(id / ((size_t)S.nx * S.ny))};
            for (int q = 1; q < LQ; ++q) { int a = ii[0] + LC[q][0], b2 = ii[1] + LC[q][1], e = ii[2] + LC[q][2]; if (a >= 0 && b2 >= 0 && e >= 0 && a < S.nx && b2 < S.ny && e < S.nz && S.flag[((size_t)e * S.ny + b2) * S.nx + a] == FL_GAS) { S.iface[id] = 1; break; } }
        }
        S.mass0 = 0; for (size_t id = 0; id < S.N; ++id) if (S.flag[id] == FL_FLUID) S.mass0 += S.Ma[id];
        S.mass_now = S.mass0;
    }
    S.polidx.assign(g_ac.size(), -1);
    for (FMover& m : S.mv) {
        const js::Value* gm = g_ac[(size_t)m.comp].part.find("gear");
        if (!gm || gm->str("kind", "") == "rack" || gm->str("kind", "") == "worm") continue;
        FlowSim::Polar P; P.comp = m.comp; P.p = m.p; P.d = m.d; P.w = m.w; P.z = (int)gm->num("teeth", 20); P.rtip = gm->num("tip_radius", m.R);
        V3 up(0, 0, 1); if (std::fabs(geo::dot(up, m.d)) > 0.9) up = V3(0, 1, 0);
        P.e2 = up - m.d * geo::dot(up, m.d); P.e2 = P.e2 / geo::len(P.e2); P.e1 = geo::cross(P.e2, m.d);  // e2 up (theta 90 deg), e1 across
        S.polidx[(size_t)m.comp] = (int)S.pol.size(); S.pol.push_back(P);
    }
    S.nfluid = 0; for (size_t i = 0; i < S.N; ++i) if (S.flag[i] == FL_FLUID) ++S.nfluid;
    if (c.mode == 0) {  // absorbing layers: the sound waves of the (weakly compressible) lattice fluid die there instead of ringing
        int span = c.span_axis == p1 || c.span_axis == p2 ? c.span_axis : ext[p1] >= ext[p2] ? p1 : p2, other = span == p1 ? p2 : p1, nn[3] = {S.nx, S.ny, S.nz};
        int Lo = std::max(6, nn[ax] / 7), Ls = std::max(4, (int)std::round(0.6 * Lref / S.dx));
        S.sponge.assign(S.N, (float)S.tau0);
        for (size_t id = 0; id < S.N; ++id) {
            int ii[3] = {(int)(id % S.nx), (int)((id / S.nx) % S.ny), (int)(id / ((size_t)S.nx * S.ny))};
            int dout = sg > 0 ? nn[ax] - 1 - ii[ax] : ii[ax];
            double w = std::max(0.0, 1.0 - dout / (double)Lo);
            int dside = std::min(ii[other], nn[other] - 1 - ii[other]);
            w = std::max(w, std::max(0.0, 1.0 - dside / (double)Ls));
            if (!c.span_walls) { int ds2 = std::min(ii[span], nn[span] - 1 - ii[span]); w = std::max(w, std::max(0.0, 1.0 - ds2 / (double)Ls)); }
            S.sponge[id] = (float)std::min(1.0, 0.5 + (S.tau0 - 0.5) * (1 + 60 * w * w));
        }
    }
    if (c.mode == 1 && !S.mv.empty()) {
        S.reach.assign(S.N, 0);
        std::vector<size_t> st;
        for (FMover& m : S.mv) for (int c2 : m.cells) if (S.flag[(size_t)c2] == FL_FLUID && !S.reach[(size_t)c2]) { S.reach[(size_t)c2] = 1; st.push_back((size_t)c2); }
        while (!st.empty()) {
            size_t id = st.back(); st.pop_back();
            int ii[3] = {(int)(id % S.nx), (int)((id / S.nx) % S.ny), (int)(id / ((size_t)S.nx * S.ny))};
            for (int q = 1; q <= 6; ++q) {
                int a = ii[0] + LC[q][0], b2 = ii[1] + LC[q][1], e = ii[2] + LC[q][2];
                if (a < 0 || b2 < 0 || e < 0 || a >= S.nx || b2 >= S.ny || e >= S.nz) continue;
                size_t n = ((size_t)e * S.ny + b2) * S.nx + a;
                if (S.reach[n] || S.flag[n] != FL_FLUID) continue;
                bool wallside = false;  // a cell touching a fixed wall does not pass the fill on (thin staircased walls leak otherwise)
                for (int q2 = 1; q2 <= 6 && !wallside; ++q2) { int a2 = a + LC[q2][0], b3 = b2 + LC[q2][1], e2 = e + LC[q2][2]; if (a2 >= 0 && b3 >= 0 && e2 >= 0 && a2 < S.nx && b3 < S.ny && e2 < S.nz) wallside = S.flag[((size_t)e2 * S.ny + b3) * S.nx + a2] == FL_STATIC; }
                S.reach[n] = wallside ? 2 : 1;
                if (!wallside) st.push_back(n);
            }
        }
    }
    S.nearwall.assign(S.N, 0);  // within 3 cells of a wall: a box dilation of the solid cells
    {
        std::vector<unsigned char> a(S.N), b2(S.N);
        for (size_t i = 0; i < S.N; ++i) a[i] = S.flag[i] != FL_FLUID;
        for (FMover& m : S.mv) for (int c2 : m.cells) a[(size_t)c2] = 1;  // the space the turning parts sweep
        int n3[3] = {S.nx, S.ny, S.nz}; size_t st[3] = {1, (size_t)S.nx, (size_t)S.nx * S.ny};
        for (int axd = 0; axd < 3; ++axd) {
            for (size_t i = 0; i < S.N; ++i) {
                int ii[3] = {(int)(i % S.nx), (int)((i / S.nx) % S.ny), (int)(i / ((size_t)S.nx * S.ny))};
                unsigned char v = a[i];
                for (int d = -3; d <= 3 && !v; ++d) { int q = ii[axd] + d; if (q >= 0 && q < n3[axd]) v = a[i + (long long)d * (long long)st[axd]]; }
                b2[i] = v;
            }
            a.swap(b2);
        }
        S.nearwall = a;
    }
    return "";    return "";
}

// ---------------------------------------------------------------- the view: slice, vortex cores, particles
struct FPart { float p[3]; float tr[10][3]; int n = 0; float s = 0; };
static std::vector<FPart> g_fparts;
static GLuint g_ftex = 0;
static int g_ftex_w = 0, g_ftex_h = 0;
static float g_fscale = 1;
static std::vector<float> g_fiso_v, g_fiso_n, g_fiso_c; static std::vector<unsigned> g_fiso_t;
static DWORD g_fiso_ms = 0;
static double g_flow_tvis = 0;
static double flow_nice_step(double raw);
static std::vector<float> g_fu;  // the view's copy of the last velocity snapshot (ux, uy, uz, density)
static std::vector<float> g_fT;  // and of the temperature
static std::vector<float> g_fphi;  // and of the liquid fraction (a bath)
static float g_flo = 0, g_fhi = 1;  // the colour scale of the field shown (SI units)
static std::vector<GLuint> g_fvol; static int g_fvol_axis = -1, g_fvol_step = 1, g_fvol_n = 0, g_fvol_w = 0, g_fvol_h = 0;
static DWORD g_fvol_ms = 0;
static unsigned g_frng = 12345;
static double frnd() { g_frng = g_frng * 1664525u + 1013904223u; return (g_frng >> 8) / 16777216.0; }

static bool flow_vel(const FlowSim& S, const std::vector<float>& U, double x, double y, double z, double* u) {  // SI velocity at a world point
    double gx = (x - S.o.x) / S.dx, gy = (y - S.o.y) / S.dx, gz = (z - S.o.z) / S.dx;
    if (gx < 0 || gy < 0 || gz < 0 || gx > S.nx - 1.001 || gy > S.ny - 1.001 || gz > S.nz - 1.001) return false;
    int i = (int)gx, j = (int)gy, k = (int)gz; double fx = gx - i, fy = gy - j, fz = gz - k;
    double s = S.dx / S.dt; u[0] = u[1] = u[2] = 0; bool solid = false;
    for (int c = 0; c < 8; ++c) {
        int a = i + (c & 1), b = j + ((c >> 1) & 1), e = k + ((c >> 2) & 1);
        size_t id = ((size_t)e * S.ny + b) * S.nx + a;
        double w = ((c & 1) ? fx : 1 - fx) * (((c >> 1) & 1) ? fy : 1 - fy) * (((c >> 2) & 1) ? fz : 1 - fz);
        if (S.flag[id] != FL_FLUID && w > 0.4) solid = true;
        for (int d = 0; d < 3; ++d) u[d] += w * U[4 * id + (size_t)d] * s;
    }
    return !solid;
}
static void flow_seed(FPart& p, const FlowSim& S, bool inlet) {
    for (int tries = 0; tries < 40; ++tries) {
        double r[3] = {frnd(), frnd(), frnd()};
        if (inlet && g_fcfg.mode == 0) { int ax = g_fcfg.dir / 2; r[ax] = flow_dir_sign(g_fcfg.dir) > 0 ? 0.01 * frnd() : 1 - 0.01 * frnd(); }
        double x = S.o.x + r[0] * (S.nx - 1) * S.dx, y = S.o.y + r[1] * (S.ny - 1) * S.dx, z = S.o.z + r[2] * (S.nz - 1) * S.dx;
        int i = (int)std::round((x - S.o.x) / S.dx), j = (int)std::round((y - S.o.y) / S.dx), k = (int)std::round((z - S.o.z) / S.dx);
        size_t cid = ((size_t)k * S.ny + j) * S.nx + i;
        if (S.flag[cid] != FL_FLUID || (!S.reach.empty() && S.reach[cid] != 1)) continue;
        p.p[0] = (float)x; p.p[1] = (float)y; p.p[2] = (float)z; p.n = 0; return;
    }
}
static void flow_particles(double dtv) {
    FlowSim& S = *g_flow;
    if (!g_fcfg.particles) return;
    const size_t NP = 4500;
    if (g_fparts.size() != NP) { g_fparts.assign(NP, FPart()); for (FPart& p : g_fparts) flow_seed(p, S, false); }
    int sub = 3; double h = dtv / sub;
    for (FPart& p : g_fparts) {
        for (int s = 0; s < sub; ++s) {
            double u[3], u2[3];
            if (!flow_vel(S, g_fu, p.p[0], p.p[1], p.p[2], u)) { flow_seed(p, S, true); break; }
            double mid[3] = {p.p[0] + 0.5 * h * u[0], p.p[1] + 0.5 * h * u[1], p.p[2] + 0.5 * h * u[2]};
            if (!flow_vel(S, g_fu, mid[0], mid[1], mid[2], u2)) { flow_seed(p, S, true); break; }
            for (int a = 0; a < 3; ++a) p.p[a] += (float)(h * u2[a]);
            p.s = (float)std::sqrt(u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2]);
        }
        for (int k = std::min(p.n, 9); k > 0; --k) for (int a = 0; a < 3; ++a) p.tr[k][a] = p.tr[k - 1][a];
        for (int a = 0; a < 3; ++a) p.tr[0][a] = p.p[a];
        p.n = std::min(p.n + 1, 10);
    }
}
// vorticity at a cell (lattice units / cell), from the snapshot
static void flow_curl(const FlowSim& S, const std::vector<float>& U, int i, int j, int k, double* w) {
    auto at = [&](int a, int b, int e, int d) { a = std::max(0, std::min(S.nx - 1, a)); b = std::max(0, std::min(S.ny - 1, b)); e = std::max(0, std::min(S.nz - 1, e)); return (double)U[4 * (((size_t)e * S.ny + b) * S.nx + a) + (size_t)d]; };
    double dwdy = 0.5 * (at(i, j + 1, k, 2) - at(i, j - 1, k, 2)), dvdz = 0.5 * (at(i, j, k + 1, 1) - at(i, j, k - 1, 1));
    double dudz = 0.5 * (at(i, j, k + 1, 0) - at(i, j, k - 1, 0)), dwdx = 0.5 * (at(i + 1, j, k, 2) - at(i - 1, j, k, 2));
    double dvdx = 0.5 * (at(i + 1, j, k, 1) - at(i - 1, j, k, 1)), dudy = 0.5 * (at(i, j + 1, k, 0) - at(i, j - 1, k, 0));
    w[0] = dwdy - dvdz; w[1] = dudz - dwdx; w[2] = dvdx - dudy;
}
static void diverging(float t, float& r, float& g, float& b) {  // blue - white - red for -1 .. 1
    t = std::max(-1.f, std::min(1.f, t));
    if (t < 0) { r = 1 + 0.85f * t; g = 1 + 0.6f * t; b = 1 + 0.05f * t; } else { r = 1 - 0.05f * t; g = 1 - 0.75f * t; b = 1 - 0.85f * t; }
}

static const char* FLOW_FIELD[7] = {"vorticity", "speed", "turbulence", "none", "pressure", "temperature", "liquid fraction"};
static const char* FLOW_UNIT[7] = {"1/s", "m/s", "m/s", "", "Pa", "C", "0 gas, 1 liquid"};
// the shown field at a cell, in SI units: vorticity (the component along `axis`, or the magnitude for axis < 0),
// speed, turbulent velocity sqrt(2k), gauge pressure, temperature
static double flow_value(const FlowSim& S, int fld, int i, int j, int k, int axis) {
    size_t id = ((size_t)k * S.ny + j) * S.nx + i;
    double vel = S.dx / S.dt;
    switch (fld) {
        case 0: { double w[3]; flow_curl(S, g_fu, i, j, k, w); double sc = 1 / S.dt; return axis >= 0 ? w[axis] * sc : std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) * sc; }
        case 1: { const float* q = &g_fu[4 * id]; return std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) * vel; }
        case 2: return std::sqrt(2 * std::max(0.f, S.kvar[id])) * vel;
        case 4: return (g_fu[4 * id + 3] - 1.0) / 3.0 * S.rho * vel * vel;
        case 5: return g_fT.size() == S.N ? (double)g_fT[id] : S.Tin;
        case 6: return g_fphi.size() == S.N ? (double)g_fphi[id] : 1.0;
    }
    return 0;
}
static void flow_colour(int fld, double v, float& r, float& g, float& b, float& t) {  // blue - green - red over the scale
    t = (float)((v - g_flo) / std::max(1e-30, (double)(g_fhi - g_flo)));
    t = std::max(0.f, std::min(1.f, t));
    if (fld == 0 && g_flo < 0) { diverging(2 * t - 1, r, g, b); return; }
    turbo(t, r, g, b);
}
// the scale: the 2nd to 98th percentile of the fluid's values (vorticity components symmetric about zero)
static void flow_scale(const FlowSim& S, int fld, int axis) {
    std::vector<float> v; v.reserve(S.N / 7 + 1);
    for (size_t id = 0; id < S.N; id += 7) {
        if (S.flag[id] != FL_FLUID) continue;
        int i = (int)(id % S.nx), j = (int)((id / S.nx) % S.ny), k = (int)(id / ((size_t)S.nx * S.ny));
        v.push_back((float)flow_value(S, fld, i, j, k, axis));
    }
    if (v.size() < 4) { g_flo = 0; g_fhi = 1; return; }
    size_t a = (size_t)(0.02 * (v.size() - 1)), b = (size_t)(0.98 * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + (long)a, v.end()); float lo = v[a];
    std::nth_element(v.begin(), v.begin() + (long)b, v.end()); float hi = v[b];
    if (fld == 0 && axis >= 0) { float m = std::max(std::fabs(lo), std::fabs(hi)); lo = -m; hi = m; }
    if (fld == 1 || fld == 2 || (fld == 0 && axis < 0)) lo = 0;
    if (fld == 5) { lo = std::min(lo, (float)S.Tin); hi = std::max(hi, (float)S.Tin + 0.01f); }
    if (fld == 6) { lo = 0; hi = 1; }
    if (hi - lo < 1e-12f) hi = lo + 1e-12f;
    g_flo = lo; g_fhi = hi;
}
// the whole fluid: a stack of coloured, translucent slices across the axis nearest the line of sight, drawn back to
// front; strong values more opaque, so the flow's structure shows through the volume
static void flow_volume_build() {
    if (!g_flow || g_fcfg.view == 1 || g_fcfg.show == 3) return;
    FlowSim& S = *g_flow;
    if (g_fu.size() != 4 * S.N) return;  // (views 0, 2, 3 and 4 carry the volume)
    double vz[3] = {g_mv[2], g_mv[6], g_mv[10]};
    int ax = 0; for (int a = 1; a < 3; ++a) if (std::fabs(vz[a]) > std::fabs(vz[ax])) ax = a;
    int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3; if (a1 > a2) std::swap(a1, a2);
    int nn[3] = {S.nx, S.ny, S.nz};
    int step = std::max(1, nn[ax] / 96), nsl = (nn[ax] + step - 1) / step;
    int fld = g_fcfg.show;
    flow_scale(S, fld, -1);
    int W = nn[a1], H = nn[a2];
    double keep = 0.42, full = 0.95;  // how much of the light the whole depth lets through: quiet fluid, strong regions
    float ab = (float)(1 - std::pow(1 - keep, 1.0 / nsl)), af = (float)(1 - std::pow(1 - full, 1.0 / nsl));
    if ((int)g_fvol.size() != nsl) { if (!g_fvol.empty()) glDeleteTextures((GLsizei)g_fvol.size(), g_fvol.data()); g_fvol.assign((size_t)nsl, 0); glGenTextures(nsl, g_fvol.data()); }
    std::vector<unsigned char> px((size_t)W * H * 4);
    for (int sl = 0; sl < nsl; ++sl) {
        int c = std::min(nn[ax] - 1, sl * step);
        for (int v = 0; v < H; ++v) for (int u = 0; u < W; ++u) {
            int ii[3]; ii[ax] = c; ii[a1] = u; ii[a2] = v;
            size_t id = ((size_t)ii[2] * S.ny + ii[1]) * S.nx + ii[0], o = 4 * ((size_t)v * W + u);
            if (S.flag[id] != FL_FLUID || (!S.reach.empty() && !S.reach[id])) { px[o + 3] = 0; continue; }
            float r, g, b, t; flow_colour(fld, flow_value(S, fld, ii[0], ii[1], ii[2], -1), r, g, b, t);
            float e = fld == 4 ? std::fabs(2 * t - 1) : t;  // pressure: both the low and the high stand out
            float al = ab + (af - ab) * e * std::sqrt(e);
            px[o] = (unsigned char)(255 * r); px[o + 1] = (unsigned char)(255 * g); px[o + 2] = (unsigned char)(255 * b); px[o + 3] = (unsigned char)(255 * std::min(1.f, al * (float)step));
        }
        glBindTexture(GL_TEXTURE_2D, g_fvol[(size_t)sl]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    }
    g_fvol_axis = ax; g_fvol_step = step; g_fvol_n = nsl; g_fvol_w = W; g_fvol_h = H;
}
static void flow_volume_draw() {
    if (!g_flow || g_fvol_axis < 0 || g_fvol.empty() || g_fcfg.view == 1 || g_fcfg.show == 3) return;
    FlowSim& S = *g_flow;
    int ax = g_fvol_axis, a1 = (ax + 1) % 3, a2 = (ax + 2) % 3; if (a1 > a2) std::swap(a1, a2);
    int nn[3] = {S.nx, S.ny, S.nz};
    double vz[3] = {g_mv[2], g_mv[6], g_mv[10]};
    bool up = vz[ax] > 0;  // the eye looks down -z: larger eye z is nearer, so draw from the far end
    glEnable(GL_TEXTURE_2D); glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); glDepthMask(GL_FALSE); glColor4f(1, 1, 1, 1);
    double hu = 0.5 / g_fvol_w, hv = 0.5 / g_fvol_h;
    for (int q = 0; q < g_fvol_n; ++q) {
        int sl = up ? q : g_fvol_n - 1 - q;
        int c = std::min(nn[ax] - 1, sl * g_fvol_step);
        double pa = S.o[ax] + c * S.dx;
        auto P = [&](double u, double v) { V3 x; x[ax] = pa; x[a1] = S.o[a1] + u * (nn[a1] - 1) * S.dx; x[a2] = S.o[a2] + v * (nn[a2] - 1) * S.dx; glVertex3d(x.x, x.y, x.z); };
        glBindTexture(GL_TEXTURE_2D, g_fvol[(size_t)sl]);
        glBegin(GL_QUADS);
        glTexCoord2d(hu, hv); P(0, 0); glTexCoord2d(1 - hu, hv); P(1, 0); glTexCoord2d(1 - hu, 1 - hv); P(1, 1); glTexCoord2d(hu, 1 - hv); P(0, 1);
        glEnd();
    }
    glDisable(GL_TEXTURE_2D); glDepthMask(GL_TRUE);
}
static void flow_legend(int W, int H) {  // the colour bar of the field shown
    if (!g_flow || g_fcfg.show == 3) return;
    float x0 = (float)W - 70, y0 = 120, h = std::min(260.f, (float)H - 260), w = 18;
    if (h < 60) return;
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(1, 1, 1, 0.85f); glBegin(GL_QUADS); glVertex2f(x0 - 56, y0 - 34); glVertex2f(x0 + w + 46, y0 - 34); glVertex2f(x0 + w + 46, y0 + h + 30); glVertex2f(x0 - 56, y0 + h + 30); glEnd();
    glDisable(GL_BLEND);
    glBegin(GL_QUAD_STRIP);
    for (int k = 0; k <= 32; ++k) { float r, g, b, t; flow_colour(g_fcfg.show, g_flo + (g_fhi - g_flo) * k / 32.0, r, g, b, t); glColor3f(r, g, b); glVertex2f(x0, y0 + h * k / 32); glVertex2f(x0 + w, y0 + h * k / 32); }
    glEnd();
    glColor3f(0.1f, 0.12f, 0.2f);
    char b[64];
    for (int k = 0; k <= 4; ++k) { snprintf(b, sizeof b, "%.3g", g_flo + (g_fhi - g_flo) * k / 4.0); gl_text(x0 - 52, y0 + h * k / 4 - 4, b); }
    snprintf(b, sizeof b, "%s", FLOW_FIELD[g_fcfg.show]); gl_text(x0 - 40, y0 + h + 14, b);
    snprintf(b, sizeof b, "%s", FLOW_UNIT[g_fcfg.show]); gl_text(x0 - 40, y0 - 24, b);
}
// a texture of the shown field across axis a at cell layer s (the gas of a bath pale, solids clear)
static void flow_slice_make(int a, int s, GLuint& tex, int& W, int& H, int fld = -1) {
    if (fld < 0) fld = g_fcfg.show;
    FlowSim& S = *g_flow;
    int a1 = (a + 1) % 3, a2 = (a + 2) % 3, nn[3] = {S.nx, S.ny, S.nz};
    if (a1 > a2) std::swap(a1, a2);
    W = nn[a1]; H = nn[a2]; s = std::max(0, std::min(nn[a] - 1, s));
    std::vector<float> val((size_t)W * H, 0.f); std::vector<char> sol((size_t)W * H, 0);
    for (int v = 0; v < H; ++v) for (int u = 0; u < W; ++u) {
        int ii[3]; ii[a] = s; ii[a1] = u; ii[a2] = v;
        size_t id = ((size_t)ii[2] * S.ny + ii[1]) * S.nx + ii[0];
        if (S.flag[id] == FL_GAS && fld == 6) { val[(size_t)v * W + u] = 0; continue; }
        if (S.flag[id] != FL_FLUID) { sol[(size_t)v * W + u] = 1; continue; }
        val[(size_t)v * W + u] = (float)flow_value(S, fld, ii[0], ii[1], ii[2], a);
    }
    std::vector<unsigned char> px((size_t)W * H * 4, 0);
    for (size_t i = 0; i < val.size(); ++i) {
        if (sol[i]) continue;
        float r, g, bb, t; flow_colour(fld, val[i], r, g, bb, t);
        float al = fld == 0 ? std::min(0.92f, 0.08f + 0.9f * std::fabs(2 * t - 1)) : 0.85f;
        if (fld == 6) { al = 0.2f + 0.7f * t; if (t < 0.05f) { r = 0.78f; g = 0.86f; bb = 1.0f; } }
        px[4 * i] = (unsigned char)(255 * r); px[4 * i + 1] = (unsigned char)(255 * g); px[4 * i + 2] = (unsigned char)(255 * bb); px[4 * i + 3] = (unsigned char)(255 * al);
    }
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
}
// x, y and z slices through the slice position, and the six walls of the box (each from the layer just inside)
static GLuint g_ftex3[3] = {0, 0, 0}, g_fwall[6] = {0, 0, 0, 0, 0, 0};
static float g_wlo[7], g_whi[7]; static int g_wfld[6] = {-1, -1, -1, -1, -1, -1};  // each wall's field and the fields' scales
static int flow_wall_layer(int f, const FlowSim& S) { int nn[3] = {S.nx, S.ny, S.nz}, a = f / 2, d = std::max(1, std::min(4, nn[a] / 12)); return f % 2 ? nn[a] - 1 - d : d; }
static int flow_wall_field(int f) {  // the field painted on wall f (axis f / 2)
    int a = f / 2, sh = g_fcfg.show == 3 ? 1 : g_fcfg.show;
    switch (g_fcfg.walls) {
        case 1: return 4;
        case 2: return 1;
        case 3: {
            if (a == 0) return 4;
            if (a == 1) return 1;
            if (sh != 1 && sh != 4) return sh;
            return g_flow && g_flow->thermal ? 5 : g_flow && g_flow->fs ? 6 : 2;
        }
    }
    return sh;
}

static int g_ftex3_w[3], g_ftex3_h[3], g_fwall_w[6], g_fwall_h[6];
static void flow_slice_texture() {
    FlowSim& S = *g_flow;
    if (g_fcfg.show == 3 || g_fcfg.view == 0 || g_fu.size() != 4 * S.N) return;
    int nn[3] = {S.nx, S.ny, S.nz};
    flow_scale(S, g_fcfg.show, g_fcfg.view == 1 || g_fcfg.view == 2 ? g_fcfg.slice_axis : -1);
    if (g_fcfg.view == 1 || g_fcfg.view == 2) { int a = g_fcfg.slice_axis; flow_slice_make(a, (int)std::round(g_fcfg.slice_pos * (nn[a] - 1)), g_ftex, g_ftex_w, g_ftex_h); }
    if (g_fcfg.view == 3) for (int a = 0; a < 3; ++a) flow_slice_make(a, (int)std::round(g_fcfg.slice_pos * (nn[a] - 1)), g_ftex3[a], g_ftex3_w[a], g_ftex3_h[a]);
    if (g_fcfg.view == 4) {
        float lo0 = g_flo, hi0 = g_fhi;
        bool done[7] = {false, false, false, false, false, false, false};
        for (int f = 0; f < 6; ++f) {
            int a = f / 2, fld = flow_wall_field(f);
            g_wfld[f] = fld;
            if (!done[fld]) {  // the scale from the values on the walls that show this field (2nd to 98th percentile)
                std::vector<float> v;
                for (int f2 = 0; f2 < 6; ++f2) {
                    if (flow_wall_field(f2) != fld) continue;
                    int b2 = f2 / 2, b1 = (b2 + 1) % 3, b3 = (b2 + 2) % 3, lay = flow_wall_layer(f2, S);
                    for (int u = 0; u < nn[b1]; u += 2) for (int w = 0; w < nn[b3]; w += 2) {
                        int ii[3]; ii[b2] = lay; ii[b1] = u; ii[b3] = w;
                        size_t id = ((size_t)ii[2] * S.ny + ii[1]) * S.nx + ii[0];
                        if (S.flag[id] == FL_FLUID) v.push_back((float)flow_value(S, fld, ii[0], ii[1], ii[2], -1));
                    }
                }
                float lo = 0, hi = 1;
                if (v.size() > 4) { size_t i1 = (size_t)(0.02 * (v.size() - 1)), i2 = (size_t)(0.98 * (v.size() - 1)); std::nth_element(v.begin(), v.begin() + (long)i1, v.end()); lo = v[i1]; std::nth_element(v.begin(), v.begin() + (long)i2, v.end()); hi = v[i2]; }
                if (fld == 1 || fld == 2) lo = 0;
                if (fld == 6) { lo = 0; hi = 1; }
                if (hi - lo < 1e-12f) hi = lo + 1e-12f;
                g_wlo[fld] = lo; g_whi[fld] = hi; done[fld] = true;
            }
            g_flo = g_wlo[fld]; g_fhi = g_whi[fld];
            flow_slice_make(a, flow_wall_layer(f, S), g_fwall[f], g_fwall_w[f], g_fwall_h[f], fld);
        }
        g_flo = lo0; g_fhi = hi0;
    }
}
// a field texture drawn across axis a at cell layer c
static void flow_tex_quad(int a, double coord, GLuint tex, int W, int H) {
    FlowSim& S = *g_flow;
    int a1 = (a + 1) % 3, a2 = (a + 2) % 3, nn[3] = {S.nx, S.ny, S.nz};
    if (a1 > a2) std::swap(a1, a2);
    double lo1 = S.o[a1] - 0.5 * S.dx, hi1 = S.o[a1] + (nn[a1] - 0.5) * S.dx, lo2 = S.o[a2] - 0.5 * S.dx, hi2 = S.o[a2] + (nn[a2] - 0.5) * S.dx;
    auto P = [&](double u, double v) { V3 q; q[a] = coord; q[a1] = lo1 + u * (hi1 - lo1); q[a2] = lo2 + v * (hi2 - lo2); glVertex3d(q.x, q.y, q.z); };
    (void)W; (void)H;
    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, tex); glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glDepthMask(GL_FALSE); glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2d(0, 0); P(0, 0); glTexCoord2d(1, 0); P(1, 0); glTexCoord2d(1, 1); P(1, 1); glTexCoord2d(0, 1); P(0, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D); glDepthMask(GL_TRUE);
}
// vortex cores: where rotation beats strain (Q > a threshold), meshed by surface nets on the grid
static void flow_cores() {
    FlowSim& S = *g_flow;
    g_fiso_v.clear(); g_fiso_n.clear(); g_fiso_c.clear(); g_fiso_t.clear();
    if (!g_fcfg.cores) return;
    int nx = S.nx, ny = S.ny, nz = S.nz;
    std::vector<float> Qf(S.N, 0.f);
    const std::vector<float>& U = g_fu;
    auto at = [&](int a, int b, int e, int d) { a = std::max(0, std::min(nx - 1, a)); b = std::max(0, std::min(ny - 1, b)); e = std::max(0, std::min(nz - 1, e)); return (double)U[4 * (((size_t)e * ny + b) * nx + a) + (size_t)d]; };
    std::vector<float> pos;
    for (int k = 1; k < nz - 1; ++k) for (int j = 1; j < ny - 1; ++j) for (int i = 1; i < nx - 1; ++i) {
        size_t id = ((size_t)k * ny + j) * nx + i;
        if (S.flag[id] != FL_FLUID) continue;
        bool nearwall = S.nearwall[id] != 0;
        if (nearwall) continue;
        double G[3][3];
        for (int d = 0; d < 3; ++d) { G[d][0] = 0.5 * (at(i + 1, j, k, d) - at(i - 1, j, k, d)); G[d][1] = 0.5 * (at(i, j + 1, k, d) - at(i, j - 1, k, d)); G[d][2] = 0.5 * (at(i, j, k + 1, d) - at(i, j, k - 1, d)); }
        double q = 0; for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) q -= 0.5 * G[r][c] * G[c][r];  // Q = -1/2 du_i/dx_j du_j/dx_i (incompressible)
        Qf[id] = (float)q; if (q > 0) pos.push_back((float)q);
    }
    if (pos.size() < 20) return;
    size_t qi = (size_t)(0.99 * (pos.size() - 1)); std::nth_element(pos.begin(), pos.begin() + (long)qi, pos.end());
    double qref = 0.05 * std::pow(S.ulb / std::max(1.0, S.Lref / S.dx), 2);  // a vortex as strong as the body's own scale
    float thr = std::max(pos[qi], (float)qref);
    auto phi = [&](int i, int j, int k) { size_t id = ((size_t)k * ny + j) * nx + i; return S.flag[id] != FL_FLUID ? 1.f : thr - Qf[id]; };  // < 0 inside a core
    std::vector<int> vid((size_t)nx * ny * nz, -1);
    double sp = S.dx / S.dt;
    for (int k = 0; k < nz - 1; ++k) for (int j = 0; j < ny - 1; ++j) for (int i = 0; i < nx - 1; ++i) {
        float c[8]; int neg = 0;
        for (int q = 0; q < 8; ++q) { c[q] = phi(i + (q & 1), j + ((q >> 1) & 1), k + ((q >> 2) & 1)); if (c[q] < 0) ++neg; }
        if (neg == 0 || neg == 8) continue;
        static const int E[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        double px = 0, py = 0, pz = 0; int ne = 0;
        for (auto& e : E) {
            float a = c[e[0]], b = c[e[1]];
            if ((a < 0) == (b < 0)) continue;
            double t = a / (a - b);
            double ax = (e[0] & 1), ay = ((e[0] >> 1) & 1), az = ((e[0] >> 2) & 1), bx = (e[1] & 1), by = ((e[1] >> 1) & 1), bz = ((e[1] >> 2) & 1);
            px += ax + (bx - ax) * t; py += ay + (by - ay) * t; pz += az + (bz - az) * t; ++ne;
        }
        px /= ne; py /= ne; pz /= ne;
        double gx = (c[1] + c[3] + c[5] + c[7]) - (c[0] + c[2] + c[4] + c[6]), gy = (c[2] + c[3] + c[6] + c[7]) - (c[0] + c[1] + c[4] + c[5]), gz = (c[4] + c[5] + c[6] + c[7]) - (c[0] + c[1] + c[2] + c[3]);
        double gl = std::sqrt(gx * gx + gy * gy + gz * gz) + 1e-30;
        vid[((size_t)k * ny + j) * nx + i] = (int)(g_fiso_v.size() / 3);
        V3 w = S.o + V3((i + px) * S.dx, (j + py) * S.dx, (k + pz) * S.dx);
        g_fiso_v.push_back((float)w.x); g_fiso_v.push_back((float)w.y); g_fiso_v.push_back((float)w.z);
        g_fiso_n.push_back((float)(gx / gl)); g_fiso_n.push_back((float)(gy / gl)); g_fiso_n.push_back((float)(gz / gl));
        size_t id0 = ((size_t)k * ny + j) * nx + i;
        float sp0 = (float)(std::sqrt(U[4 * id0] * U[4 * id0] + U[4 * id0 + 1] * U[4 * id0 + 1] + U[4 * id0 + 2] * U[4 * id0 + 2]) * sp);
        g_fiso_c.push_back(sp0);
    }
    // quads across every grid edge that changes sign
    for (int k = 1; k < nz - 1; ++k) for (int j = 1; j < ny - 1; ++j) for (int i = 1; i < nx - 1; ++i) {
        float p0 = phi(i, j, k);
        for (int a = 0; a < 3; ++a) {
            int di = a == 0, dj = a == 1, dk = a == 2;
            float p1 = phi(i + di, j + dj, k + dk);
            if ((p0 < 0) == (p1 < 0)) continue;
            int cc[4][3];
            if (a == 0) { int t[4][3] = {{i, j - 1, k - 1}, {i, j, k - 1}, {i, j, k}, {i, j - 1, k}}; memcpy(cc, t, sizeof t); }
            else if (a == 1) { int t[4][3] = {{i - 1, j, k - 1}, {i - 1, j, k}, {i, j, k}, {i, j, k - 1}}; memcpy(cc, t, sizeof t); }
            else { int t[4][3] = {{i - 1, j - 1, k}, {i, j - 1, k}, {i, j, k}, {i - 1, j, k}}; memcpy(cc, t, sizeof t); }
            int v[4]; bool ok = true;
            for (int q = 0; q < 4; ++q) { v[q] = vid[((size_t)cc[q][2] * ny + cc[q][1]) * nx + cc[q][0]]; if (v[q] < 0) ok = false; }
            if (!ok) continue;
            if (p0 < 0) std::swap(v[1], v[3]);
            g_fiso_t.insert(g_fiso_t.end(), {(unsigned)v[0], (unsigned)v[1], (unsigned)v[2], (unsigned)v[0], (unsigned)v[2], (unsigned)v[3]});
        }
    }
}
static void flow_poses() {  // the turning parts drawn where the flow has them
    FlowSim& S = *g_flow;
    if (g_flow_base.size() != g_ac.size()) return;
    for (const FMover& m : S.mv) {
        if (m.comp < 0 || m.comp >= (int)g_ac.size()) continue;
        Pose P = g_flow_base[(size_t)m.comp];
        double M[9]; rot_axis(m.d, m.w * g_flow_tvis, M); pose_turn(P, M, m.p);
        g_ac[(size_t)m.comp].pose = P;
    }
}

// the loss torque on one tooth as it passes each angle: the bins within one tooth pitch round it, averaged over the
// second half of the run, in N m, positive when it resists the turn
static void flow_polar_tooth(const FlowSim& S, size_t pi, int zone, std::vector<double>& out) {
    out.assign(NPB, 0.0);
    if (S.pacc.empty() || S.pol_steps <= 0) return;
    const FlowSim::Polar& P = S.pol[pi];
    std::vector<double> bin(NPB, 0.0);
    double tu = S.rho * std::pow(S.dx, 5) / (S.dt * S.dt), sg = P.w >= 0 ? -1.0 : 1.0;
    for (const auto& v : S.pacc) for (int b = 0; b < NPB; ++b) bin[(size_t)b] += v[(pi * 2 + (size_t)zone) * NPB + (size_t)b];
    int half = std::max(0, (int)std::round(0.5 * NPB / std::max(1, P.z)));
    for (int b = 0; b < NPB; ++b) {
        double sum = 0; for (int k = -half; k <= half; ++k) sum += bin[(size_t)((b + k + NPB) % NPB)];
        out[(size_t)b] = sg * sum * tu / S.pol_steps * (double)NPB / (P.z * (2 * half + 1));
    }
}
static size_t flow_polar_main(const FlowSim& S) { size_t best = 0; for (size_t i = 1; i < S.pol.size(); ++i) if (S.pol[i].rtip > S.pol[best].rtip) best = i; return best; }
static void flow_polar_draw() {
    if (!g_flow) return;
    FlowSim& S = *g_flow;
    for (size_t pi = 0; pi < S.pol.size(); ++pi) {
        if (pi != flow_polar_main(S)) continue;
        { V3 ez(g_mv[2], g_mv[6], g_mv[10]); if (std::fabs(geo::dot(ez, S.pol[pi].d)) < 0.45) continue; }  // edge on: a polar would be a line
        std::vector<double> fl, tp; flow_polar_tooth(S, pi, 0, fl); flow_polar_tooth(S, pi, 1, tp);
        double mx = 1e-30; for (int b = 0; b < NPB; ++b) mx = std::max(mx, std::max(std::fabs(fl[(size_t)b]), std::fabs(tp[(size_t)b])));
        if (S.pol_steps <= 0) continue;
        const FlowSim::Polar& P = S.pol[pi];
        double R0 = 1.35 * P.rtip, sc = 0.45 * P.rtip / mx;
        auto at = [&](double th, double r) { V3 x = P.p + (P.e1 * std::cos(th) + P.e2 * std::sin(th)) * r; glVertex3d(x.x, x.y, x.z); };
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND);
        for (int k = -2; k <= 2; ++k) {  // reference circles: 0, half and the whole of the largest
            glColor4f(0.25f, 0.28f, 0.35f, k == 0 ? 0.7f : 0.35f); glLineWidth(k == 0 ? 1.4f : 1.f);
            glBegin(GL_LINE_LOOP); for (int b = 0; b < 120; ++b) at(2 * geo::PI * b / 120, R0 + sc * mx * k / 2); glEnd();
        }
        glLineWidth(2.2f);
        glColor4f(0.1f, 0.35f, 0.85f, 0.95f); glBegin(GL_LINE_LOOP); for (int b = 0; b < NPB; ++b) at(-geo::PI + 2 * geo::PI * (b + 0.5) / NPB, R0 + sc * fl[(size_t)b]); glEnd();
        glColor4f(0.72f, 0.08f, 0.1f, 0.95f); glBegin(GL_LINE_LOOP); for (int b = 0; b < NPB; ++b) at(-geo::PI + 2 * geo::PI * (b + 0.5) / NPB, R0 + sc * tp[(size_t)b]); glEnd();
        glLineWidth(1.f); glEnable(GL_DEPTH_TEST);
    }
}
static void flow_polar_labels() {  // the reference circles' values, at the top of each polar
    if (!g_flow) return;
    FlowSim& S = *g_flow;
    for (size_t pi = 0; pi < S.pol.size(); ++pi) {
        if (pi != flow_polar_main(S)) continue;
        if (S.pol_steps <= 0) continue;
        { V3 ez(g_mv[2], g_mv[6], g_mv[10]); if (std::fabs(geo::dot(ez, S.pol[pi].d)) < 0.45) continue; }
        std::vector<double> fl, tp; flow_polar_tooth(S, pi, 0, fl); flow_polar_tooth(S, pi, 1, tp);
        double mx = 1e-30; for (int b = 0; b < NPB; ++b) mx = std::max(mx, std::max(std::fabs(fl[(size_t)b]), std::fabs(tp[(size_t)b])));
        const FlowSim::Polar& P = S.pol[pi];
        double R0 = 1.35 * P.rtip, sc = 0.45 * P.rtip / mx;
        for (int k = -2; k <= 2; ++k) {
            V3 x = P.p + P.e2 * (R0 + sc * mx * k / 2);
            double wx, wy, wz;
            if (!gluProject(x.x, x.y, x.z, g_mv, g_pj, g_vp, &wx, &wy, &wz)) continue;
            char b[48]; snprintf(b, sizeof b, "%.3g", mx * k / 2); glColor3f(0.15f, 0.18f, 0.28f); gl_text((float)wx + 4, (float)wy + 2, b);
        }
        V3 x = P.p - P.e2 * (R0 + sc * mx * 1.2);  // the title under the polar
        double wx, wy, wz;
        if (gluProject(x.x, x.y, x.z, g_mv, g_pj, g_vp, &wx, &wy, &wz)) { glColor3f(0.1f, 0.12f, 0.2f); gl_text((float)wx - 160, (float)wy - 18, flow_id(P.comp) + ": loss torque per tooth, N m (blue flank, red tip circle)"); }
    }
}

// ---------------------------------------------------------------- the parts' surfaces coloured by the flow next to them
static const char* SURF_NAME[7] = {"", "friction (wall shear)", "pressure", "temperature", "turbulence", "liquid fraction", "heat-transfer coefficient"};
static const char* SURF_UNIT[7] = {"", "Pa", "Pa", "C", "m/s", "0 gas, 1 liquid", "W/m2K"};
static std::vector<std::vector<std::vector<float>>> g_fsc;  // per component, per piece: r g b per vertex
static float g_slo = 0, g_shi = 1;
static void flow_surface_build() {
    g_fsc.clear();
    if (!g_flow || g_fcfg.surf == 0) return;
    FlowSim& S = *g_flow;
    if (g_fu.size() != 4 * S.N || g_flow_base.size() != g_ac.size()) return;
    double vel = S.dx / S.dt, mu = S.rho * S.nu;
    std::vector<std::vector<std::vector<float>>> val(g_ac.size());
    std::vector<float> all;
    for (int c : S.bodies) {
        const AComp& C = g_ac[(size_t)c];
        const FMover* mv = nullptr; for (const FMover& m : S.mv) if (m.comp == c) mv = &m;
        val[(size_t)c].resize(C.pieces.size());
        for (size_t k = 0; k < C.pieces.size(); ++k) {
            const APiece& pc = C.pieces[k];
            if (!pc.mesh) continue;
            const geo::Mesh& M = pc.mesh->m;
            std::vector<float>& v = val[(size_t)c][k];
            v.assign(M.v.size(), NAN);
            for (size_t i = 0; i < M.v.size(); ++i) {
                V3 X = C.pose.apply(pc.local.apply(M.v[i])), Nn = C.pose.rot(pc.local.rot(M.n[i]));
                for (double off : {0.8, 1.6}) {
                    V3 Y = X + Nn * (off * S.dx);
                    int ci = (int)std::round((Y.x - S.o.x) / S.dx), cj = (int)std::round((Y.y - S.o.y) / S.dx), ck = (int)std::round((Y.z - S.o.z) / S.dx);
                    if (ci < 0 || cj < 0 || ck < 0 || ci >= S.nx || cj >= S.ny || ck >= S.nz) continue;
                    size_t id = ((size_t)ck * S.ny + cj) * S.nx + ci;
                    unsigned char fl = S.flag[id];
                    if (fl != FL_FLUID && !(fl == FL_GAS && g_fcfg.surf == 5)) continue;
                    double x = 0;
                    switch (g_fcfg.surf) {
                        case 1: {
                            if (fl != FL_FLUID) { x = 0; break; }
                            V3 u(g_fu[4 * id] * vel, g_fu[4 * id + 1] * vel, g_fu[4 * id + 2] * vel), uw;
                            if (mv) uw = geo::cross(mv->d * mv->w, X - mv->p);
                            V3 cc = S.o + V3(ci * S.dx, cj * S.dx, ck * S.dx);
                            x = mu * geo::len(u - uw) / std::max(0.5 * S.dx, geo::len(cc - X));
                            break;
                        }
                        case 2: x = (g_fu[4 * id + 3] - 1.0) / 3.0 * S.rho * vel * vel; break;
                        case 3: x = g_fT.size() == S.N ? g_fT[id] : S.Tin; break;
                        case 4: x = std::sqrt(2 * std::max(0.f, S.kvar[id])) * vel; break;
                        case 5: x = fl == FL_GAS ? 0.0 : g_fphi.size() == S.N ? g_fphi[id] : 1.0; break;
                        case 6: {  // the local film coefficient: the conduction into the first fluid cell over the wall-to-fluid difference
                            float tw = c < (int)S.compT.size() ? S.compT[(size_t)c] : NAN;
                            if (!(tw == tw) || g_fT.size() != S.N || std::fabs(tw - S.Tin) < 1e-9 || fl != FL_FLUID) { x = NAN; break; }
                            V3 cc = S.o + V3(ci * S.dx, cj * S.dx, ck * S.dx);
                            x = S.kf * (tw - g_fT[id]) / std::max(0.5 * S.dx, geo::len(cc - X)) / (tw - S.Tin);
                            break;
                        }
                    }
                    v[i] = (float)x;
                    if ((i & 3) == 0 && x == x) all.push_back((float)x);
                    break;
                }
            }
            for (int pass = 0; pass < 3; ++pass) {  // neighbour averaging over the triangles: a smooth field, not the cell-size checker
                std::vector<float> s(v.size(), 0.f), w(v.size(), 0.f);
                for (size_t t = 0; t + 2 < M.t.size(); t += 3) for (int e = 0; e < 3; ++e) {
                    int i0 = M.t[t + (size_t)e], i1 = M.t[t + (size_t)((e + 1) % 3)];
                    float a0 = v[(size_t)i0], a1 = v[(size_t)i1];
                    if (a1 == a1) { s[(size_t)i0] += a1; w[(size_t)i0] += 1; }
                    if (a0 == a0) { s[(size_t)i1] += a0; w[(size_t)i1] += 1; }
                }
                for (size_t i = 0; i < v.size(); ++i) { if (w[i] <= 0) continue; v[i] = v[i] == v[i] ? 0.5f * v[i] + 0.5f * s[i] / w[i] : s[i] / w[i]; }
            }
        }
    }
    if (all.size() < 4) return;
    size_t a = (size_t)(0.02 * (all.size() - 1)), b = (size_t)(0.98 * (all.size() - 1));
    std::nth_element(all.begin(), all.begin() + (long)a, all.end()); float lo = all[a];
    std::nth_element(all.begin(), all.begin() + (long)b, all.end()); float hi = all[b];
    if (g_fcfg.surf == 1 || g_fcfg.surf == 4 || g_fcfg.surf == 6) lo = 0;
    if (g_fcfg.surf == 5) { lo = 0; hi = 1; }
    if (hi - lo < 1e-12f) hi = lo + 1e-12f;
    g_slo = lo; g_shi = hi;
    g_fsc.assign(g_ac.size(), std::vector<std::vector<float>>());
    for (int c : S.bodies) {
        g_fsc[(size_t)c].resize(val[(size_t)c].size());
        for (size_t k = 0; k < val[(size_t)c].size(); ++k) {
            const std::vector<float>& v = val[(size_t)c][k];
            std::vector<float>& col = g_fsc[(size_t)c][k];
            col.assign(3 * v.size(), 0.6f);
            for (size_t i = 0; i < v.size(); ++i) {
                if (!(v[i] == v[i])) continue;
                float t = std::max(0.f, std::min(1.f, (v[i] - lo) / (hi - lo))), r, g, bb; turbo(t, r, g, bb);
                col[3 * i] = r; col[3 * i + 1] = g; col[3 * i + 2] = bb;
            }
        }
    }
}
static void flow_surface_draw() {
    if (g_fsc.size() != g_ac.size()) return;
    glEnable(GL_LIGHTING); glEnable(GL_COLOR_MATERIAL); glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-1, -2);
    for (size_t c = 0; c < g_ac.size(); ++c) {
        const AComp& C = g_ac[c];
        if (C.hidden || g_fsc[c].empty()) continue;
        glPushMatrix(); gl_pose(C.pose, V3());
        for (size_t k = 0; k < C.pieces.size() && k < g_fsc[c].size(); ++k) {
            const APiece& pc = C.pieces[k];
            if (!pc.mesh || g_fsc[c][k].size() != 3 * pc.mesh->m.v.size()) continue;
            const geo::Mesh& M = pc.mesh->m;
            glPushMatrix(); gl_pose(pc.local, V3());
            static std::vector<float> vb, nb;
            vb.resize(M.v.size() * 3); nb.resize(M.v.size() * 3);
            for (size_t i = 0; i < M.v.size(); ++i) { vb[3 * i] = (float)M.v[i].x; vb[3 * i + 1] = (float)M.v[i].y; vb[3 * i + 2] = (float)M.v[i].z; nb[3 * i] = (float)M.n[i].x; nb[3 * i + 1] = (float)M.n[i].y; nb[3 * i + 2] = (float)M.n[i].z; }
            glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY); glEnableClientState(GL_COLOR_ARRAY);
            glVertexPointer(3, GL_FLOAT, 0, vb.data()); glNormalPointer(GL_FLOAT, 0, nb.data()); glColorPointer(3, GL_FLOAT, 0, g_fsc[c][k].data());
            glDrawElements(GL_TRIANGLES, (GLsizei)M.t.size(), GL_UNSIGNED_INT, M.t.data());
            glDisableClientState(GL_COLOR_ARRAY); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_VERTEX_ARRAY);
            glPopMatrix();
        }
        glPopMatrix();
    }
    glDisable(GL_POLYGON_OFFSET_FILL);
}
static void flow_surface_legend(int W, int H) {
    if (g_fsc.empty() || g_fcfg.surf == 0) return;
    float x0 = (float)W - 250, y0 = 120, h = std::min(260.f, (float)H - 260), w = 18;
    if (h < 60) return;
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(1, 1, 1, 0.85f); glBegin(GL_QUADS); glVertex2f(x0 - 56, y0 - 34); glVertex2f(x0 + w + 50, y0 - 34); glVertex2f(x0 + w + 50, y0 + h + 30); glVertex2f(x0 - 56, y0 + h + 30); glEnd();
    glDisable(GL_BLEND);
    glBegin(GL_QUAD_STRIP);
    for (int k = 0; k <= 32; ++k) { float r, g, b; turbo(k / 32.f, r, g, b); glColor3f(r, g, b); glVertex2f(x0, y0 + h * k / 32); glVertex2f(x0 + w, y0 + h * k / 32); }
    glEnd();
    glColor3f(0.1f, 0.12f, 0.2f);
    char b[64];
    for (int k = 0; k <= 4; ++k) { snprintf(b, sizeof b, "%.3g", g_slo + (g_shi - g_slo) * k / 4.0); gl_text(x0 - 52, y0 + h * k / 4 - 4, b); }
    gl_text(x0 - 52, y0 + h + 14, std::string("surface: ") + SURF_NAME[g_fcfg.surf]); gl_text(x0 - 52, y0 + h + 28, "");
    gl_text(x0 - 40, y0 - 24, SURF_UNIT[g_fcfg.surf]);
}
static void flow_render() {
    if (!g_flow_on || !g_flow || g_flow_base.size() != g_ac.size()) return;
    FlowSim& S = *g_flow;
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    // the domain's edges
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    V3 lo = S.o - V3(0.5 * S.dx, 0.5 * S.dx, 0.5 * S.dx), hi = lo + V3(S.nx * S.dx, S.ny * S.dx, S.nz * S.dx);
    glColor4f(0.15f, 0.35f, 0.7f, 0.5f); glLineWidth(1.2f);
    glBegin(GL_LINES);
    for (int e = 0; e < 12; ++e) {
        int a = e / 4, b = e % 4; V3 p0, p1;
        double c1 = (b & 1) ? hi[(a + 1) % 3] : lo[(a + 1) % 3], c2 = (b & 2) ? hi[(a + 2) % 3] : lo[(a + 2) % 3];
        p0[a] = lo[a]; p1[a] = hi[a]; p0[(a + 1) % 3] = p1[(a + 1) % 3] = c1; p0[(a + 2) % 3] = p1[(a + 2) % 3] = c2;
        glVertex3d(p0.x, p0.y, p0.z); glVertex3d(p1.x, p1.y, p1.z);
    }
    glEnd();
    {   // the whole system's x, y and z: coloured axes along three edges from the low corner, with ticks
        double Lmax = std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z)), tk = 0.018 * Lmax;
        static const float AC[3][3] = {{0.92f, 0.2f, 0.22f}, {0.12f, 0.68f, 0.3f}, {0.18f, 0.42f, 0.95f}};
        glLineWidth(3.0f);
        for (int a = 0; a < 3; ++a) {
            glColor4f(AC[a][0], AC[a][1], AC[a][2], 0.95f);
            V3 e = lo; e[a] = hi[a] + 0.06 * Lmax;
            glBegin(GL_LINES); glVertex3d(lo.x, lo.y, lo.z); glVertex3d(e.x, e.y, e.z); glEnd();
            double step = flow_nice_step((hi[a] - lo[a]) / 5);
            int b = (a + 1) % 3;
            glLineWidth(1.6f); glBegin(GL_LINES);
            for (double v = std::ceil(lo[a] / step) * step; v <= hi[a] + 1e-12; v += step) { V3 p = lo; p[a] = v; V3 q = p; q[b] -= tk; glVertex3d(p.x, p.y, p.z); glVertex3d(q.x, q.y, q.z); }
            glEnd(); glLineWidth(3.0f);
        }
    }
    // particles and their short trails
    if (g_fcfg.particles && !g_fparts.empty()) {
        float vmax = (float)std::max(1e-9, S.Uref * 1.5);
        auto pcol = [&](float s, float& r, float& g, float& b) { float t = std::min(1.f, s / vmax); if (t < 0.6f) { float u = t / 0.6f; r = 0.08f + 0.2f * u; g = 0.22f + 0.5f * u; b = 0.55f + 0.4f * u; } else { float u = (t - 0.6f) / 0.4f; r = 0.28f + 0.72f * u; g = 0.72f + 0.28f * u; b = 0.95f + 0.05f * u; } };
        glLineWidth(1.2f);
        glBegin(GL_LINES);
        for (const FPart& p : g_fparts) {
            float r, g, b; pcol(p.s, r, g, b);
            for (int k = 0; k + 1 < std::min(p.n, 7); ++k) { float a = 0.6f * (1 - k / 7.f); glColor4f(r, g, b, a); glVertex3fv(p.tr[k]); glColor4f(r, g, b, a * 0.7f); glVertex3fv(p.tr[k + 1]); }
        }
        glEnd();
        glPointSize(2.0f); glBegin(GL_POINTS);
        for (const FPart& p : g_fparts) { float r, g, b; pcol(p.s, r, g, b); glColor4f(r, g, b, 0.85f); glVertex3fv(p.p); }
        glEnd();
    }
    flow_surface_draw();
    flow_volume_draw();
    flow_polar_draw();
    // the slices and the cutaway walls
    if (g_fcfg.show != 3) {
        int nn[3] = {S.nx, S.ny, S.nz};
        if ((g_fcfg.view == 1 || g_fcfg.view == 2) && g_ftex) { int a = g_fcfg.slice_axis; flow_tex_quad(a, S.o[a] + std::round(g_fcfg.slice_pos * (nn[a] - 1)) * S.dx, g_ftex, g_ftex_w, g_ftex_h); }
        if (g_fcfg.view == 3) for (int a = 0; a < 3; ++a) if (g_ftex3[a]) flow_tex_quad(a, S.o[a] + std::round(g_fcfg.slice_pos * (nn[a] - 1)) * S.dx, g_ftex3[a], g_ftex3_w[a], g_ftex3_h[a]);
        if (g_fcfg.view == 4) {  // the walls the eye looks into (their outward normal points away from it)
            double ez[3] = {g_mv[2], g_mv[6], g_mv[10]};
            for (int f = 0; f < 6; ++f) {
                int a = f / 2; double nrm = f % 2 ? 1 : -1;
                if (nrm * ez[a] >= 0 || !g_fwall[f]) continue;
                flow_tex_quad(a, f % 2 ? S.o[a] + (nn[a] - 0.5) * S.dx : S.o[a] - 0.5 * S.dx, g_fwall[f], g_fwall_w[f], g_fwall_h[f]);
                if (g_wfld[f] == 1 && g_fu.size() == 4 * S.N) {  // the flow along the wall: arrows of its in-plane velocity
                    int a1 = (a + 1) % 3, a2 = (a + 2) % 3; if (a1 > a2) std::swap(a1, a2);
                    int st = std::max(1, std::max(nn[a1], nn[a2]) / 28), lay = flow_wall_layer(f, S);
                    double vel = S.dx / S.dt, vmax = std::max(1e-12, (double)g_whi[1]), len = 0.9 * st * S.dx;
                    double wc = f % 2 ? S.o[a] + (nn[a] - 0.9) * S.dx : S.o[a] - 0.1 * S.dx;
                    glDisable(GL_LIGHTING); glColor4f(0.06f, 0.08f, 0.14f, 0.85f); glLineWidth(1.3f);
                    glBegin(GL_LINES);
                    for (int u = st / 2; u < nn[a1]; u += st) for (int v = st / 2; v < nn[a2]; v += st) {
                        int ii[3]; ii[a] = lay; ii[a1] = u; ii[a2] = v;
                        size_t id = ((size_t)ii[2] * S.ny + ii[1]) * S.nx + ii[0];
                        if (S.flag[id] != FL_FLUID) continue;
                        double w1 = g_fu[4 * id + (size_t)a1] * vel, w2 = g_fu[4 * id + (size_t)a2] * vel, sp = std::sqrt(w1 * w1 + w2 * w2);
                        if (sp < 0.03 * vmax) continue;
                        double k = len * std::min(1.0, sp / vmax) / sp;
                        V3 p0; p0[a] = wc; p0[a1] = S.o[a1] + u * S.dx; p0[a2] = S.o[a2] + v * S.dx;
                        V3 d; d[a1] = w1 * k; d[a2] = w2 * k;
                        V3 p1 = p0 + d, sd; sd[a1] = -d[a2] * 0.35; sd[a2] = d[a1] * 0.35;
                        glVertex3d(p0.x, p0.y, p0.z); glVertex3d(p1.x, p1.y, p1.z);
                        V3 h1 = p1 - d * 0.35 + sd, h2 = p1 - d * 0.35 - sd;
                        glVertex3d(p1.x, p1.y, p1.z); glVertex3d(h1.x, h1.y, h1.z);
                        glVertex3d(p1.x, p1.y, p1.z); glVertex3d(h2.x, h2.y, h2.z);
                    }
                    glEnd(); glLineWidth(1.f);
                }
            }
        }
    }
    // vortex cores, lit, translucent
    if (g_fcfg.cores && !g_fiso_t.empty()) {
        glEnable(GL_LIGHTING); glEnable(GL_COLOR_MATERIAL); glDepthMask(GL_FALSE);
        float vmax = (float)std::max(1e-9, S.Uref * 1.4); (void)vmax;
        glBegin(GL_TRIANGLES);
        for (unsigned t : g_fiso_t) {
            float s2 = std::min(1.f, g_fiso_c[t] / vmax);
            glColor4f(0.30f + 0.35f * s2, 0.62f + 0.2f * s2, 0.92f, 0.5f); glNormal3fv(&g_fiso_n[3 * (size_t)t]); glVertex3fv(&g_fiso_v[3 * (size_t)t]);
        }
        glEnd();
        glDepthMask(GL_TRUE);
    }
    glPopAttrib();
}
static void flow_plot(float x0, float y0, float w, float h, const std::vector<double>& t, const std::vector<double>& v, float r, float g, float b) {
    if (v.size() < 2) return;
    double vm = 1e-30, vn = 0; for (double q : v) { vm = std::max(vm, q); vn = std::min(vn, q); }
    double tm = t.back();
    glColor3f(r, g, b); glLineWidth(1.6f); glBegin(GL_LINE_STRIP);
    for (size_t i = 0; i < v.size(); ++i) glVertex2f(x0 + (float)(w * t[i] / tm), y0 + (float)(h * (v[i] - vn) / (vm - vn)));
    glEnd(); glLineWidth(1.f);
}
static std::string flow_regime(double Re, int mode) {
    if (mode == 1) return Re < 10 ? "laminar" : Re < 1e4 ? "transitional" : "turbulent";
    if (mode == 2) return Re < 10 ? "laminar" : Re < 1e4 ? "transitional" : "turbulent";
    return Re < 47 ? "steady wake" : Re < 190 ? "laminar vortex street" : Re < 3e5 ? "turbulent wake (subcritical)" : "turbulent (supercritical)";
}
static double flow_nice_step(double raw) {  // 1, 2 or 5 times a power of ten
    if (!(raw > 0)) return 1;
    double p = std::pow(10.0, std::floor(std::log10(raw))), f = raw / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
}
static void flow_axes_labels() {  // tick values (mm) and X, Y, Z at the ends of the box's axes
    FlowSim& S = *g_flow;
    V3 lo = S.o - V3(0.5 * S.dx, 0.5 * S.dx, 0.5 * S.dx), hi = lo + V3(S.nx * S.dx, S.ny * S.dx, S.nz * S.dx);
    double Lmax = std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z));
    static const float AC[3][3] = {{0.85f, 0.12f, 0.15f}, {0.05f, 0.55f, 0.2f}, {0.1f, 0.3f, 0.85f}};
    static const char* AN[3] = {"X", "Y", "Z"};
    GLdouble wx, wy, wz;
    for (int a = 0; a < 3; ++a) {
        glColor3f(AC[a][0], AC[a][1], AC[a][2]);
        V3 e = lo; e[a] = hi[a] + 0.09 * Lmax;
        if (gluProject(e.x, e.y, e.z, g_mv, g_pj, g_vp, &wx, &wy, &wz) && wz < 1) gl_text((float)wx - 4, (float)wy - 4, std::string(AN[a]) + " (mm)");
        double step = flow_nice_step((hi[a] - lo[a]) / 5);
        int b = (a + 1) % 3;
        for (double v = std::ceil(lo[a] / step) * step; v <= hi[a] + 1e-12; v += step) {
            V3 p = lo; p[a] = v; p[b] -= 0.05 * Lmax;
            char t[32]; snprintf(t, sizeof t, "%.4g", v * 1000);
            if (gluProject(p.x, p.y, p.z, g_mv, g_pj, g_vp, &wx, &wy, &wz) && wz < 1) gl_text((float)wx - 12, (float)wy - 4, t);
        }
    }
}
static void flow_overlay(int W, int H) {
    if (!g_flow_on || !g_flow || g_flow_base.size() != g_ac.size()) return;
    FlowSim& S = *g_flow;
    EnterCriticalSection(&S.cs);
    std::vector<double> ht = S.h_t, hk = S.h_ke, htk = S.h_tke;
    double ke = S.ke, tke = S.tke, mlups = S.mlups, nut = S.nut, umax = S.umax;
    std::vector<double> lastf;
    for (size_t b = 0; b < S.h_f.size(); ++b) { if (S.h_f[b].size() >= 6) lastf.insert(lastf.end(), S.h_f[b].end() - 6, S.h_f[b].end()); else lastf.insert(lastf.end(), 6, 0.0); }
    LeaveCriticalSection(&S.cs);
    flow_axes_labels();
    char b[300];
    float x = 14, y = (float)H - 58;  // under the view's toolbar, at the left
    {   // a light panel behind the lines, so they read over the fluid
        int nl = 4 + (S.fs ? 1 : 0) + std::min(5, (int)S.mv.size()) + (S.thermal ? 5 : 0) + (g_fcfg.mode == 0 ? (int)S.bodies.size() : 0);
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(1, 1, 1, 0.72f); glBegin(GL_QUADS); glVertex2f(x - 8, y + 16); glVertex2f(x + 660, y + 16); glVertex2f(x + 660, y - 16.f * nl + 6); glVertex2f(x - 8, y - 16.f * nl + 6); glEnd();
        glDisable(GL_BLEND);
    }
    glColor3f(0.08f, 0.12f, 0.25f);
    snprintf(b, sizeof b, "FLOW (lattice Boltzmann D3Q19, %s): %s, %s Re %.3g (%s)", S.smag > 0 ? "Smagorinsky LES" : "direct", S.medium.c_str(), g_fcfg.mode == 0 ? "current past the parts," : g_fcfg.mode == 2 ? "bath with a free surface," : "stirred vessel,", S.Re, flow_regime(S.Re, g_fcfg.mode).c_str());
    gl_text(x, y, b); y -= 16;
    snprintf(b, sizeof b, "grid %d x %d x %d (dx %.3g mm), t = %.4g s of %.4g s (%ld steps, %.0f Mcells/s)%s%s", S.nx, S.ny, S.nz, S.dx * 1000, S.t, S.steps_total * S.dt, S.step, mlups,
             S.failed ? "  DIVERGED" : S.done ? "  finished" : "", g_flow_running ? "" : "");
    gl_text(x, y, b); y -= 16;
    if (S.fs) { snprintf(b, sizeof b, "liquid level %.4g m at the start; liquid mass %.2f %% of the start (cells converted %ld)", S.zsurf, S.mass0 > 0 ? 100 * S.mass_now / S.mass0 : 0.0, S.conversions); gl_text(x, y, b); y -= 16; }
    snprintf(b, sizeof b, "kinetic energy %.4g J, fluctuating %.3g J (%.1f %%), fastest %.3g m/s, eddy viscosity %.2g x molecular", ke, tke, ke > 0 ? 100 * tke / ke : 0.0, umax, nut);
    gl_text(x, y, b); y -= 16;
    int shown = 0;
    for (const FMover& m : S.mv) {
        if (++shown > 4) { gl_text(x, y, "... (every part in the report)"); y -= 16; break; }
        size_t bi = 0; for (; bi < S.bodies.size(); ++bi) if (S.bodies[bi] == m.comp) break;
        if (bi >= S.bodies.size() || lastf.size() < 6 * (bi + 1)) continue;
        V3 T(lastf[6 * bi + 3], lastf[6 * bi + 4], lastf[6 * bi + 5]), Fw(lastf[6 * bi], lastf[6 * bi + 1], lastf[6 * bi + 2]);
        double Ta = geo::dot(T - geo::cross(m.p, Fw), m.d), P = -Ta * m.w;
        snprintf(b, sizeof b, "%s: %.0f rpm, the fluid's torque %.4g N m, power %.4g W", flow_id(m.comp).c_str(), std::fabs(m.w) * 30 / geo::PI, Ta, P);
        gl_text(x, y, b); y -= 16;
    }
    if (S.thermal) {
        EnterCriticalSection(&S.cs);
        int nh = 0;
        for (size_t bi = 0; bi < S.bodies.size(); ++bi) {
            if (S.h_q[bi].empty() || !(S.compT[(size_t)S.bodies[bi]] == S.compT[(size_t)S.bodies[bi]])) continue;
            if (++nh > 4) break;
            double q = S.h_q[bi].back(), h = S.barea[bi] > 0 ? q / (S.barea[bi] * (S.Thot - S.Tin)) : 0;
            snprintf(b, sizeof b, "%s at %.0f C: gives %.3g W to the fluid (h %.3g W/m2K)", flow_id(S.bodies[bi]).c_str(), S.Thot, q, h);
            gl_text(x, y, b); y -= 16;
        }
        LeaveCriticalSection(&S.cs);
    }
    if (g_fcfg.mode == 0) for (size_t bi = 0; bi < S.bodies.size() && lastf.size() >= 6 * (bi + 1); ++bi) {
        int ax = g_fcfg.dir / 2; double sg = flow_dir_sign(g_fcfg.dir);
        double Fd = lastf[6 * bi + (size_t)ax] * sg;
        if (std::fabs(Fd) < 1e-12) continue;
        snprintf(b, sizeof b, "%s: drag %.4g N", flow_id(S.bodies[bi]).c_str(), Fd);
        gl_text(x, y, b); y -= 16;
    }
    // the history: kinetic energy (blue), turbulent (red)
    float pw = 260, ph = 70, px = (float)W - pw - 30, py = (float)H - ph - 40;
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(1, 1, 1, 0.8f); glBegin(GL_QUADS); glVertex2f(px - 6, py - 22); glVertex2f(px + pw + 6, py - 22); glVertex2f(px + pw + 6, py + ph + 6); glVertex2f(px - 6, py + ph + 6); glEnd();
    glDisable(GL_BLEND);
    flow_plot(px, py, pw, ph, ht, hk, 0.1f, 0.35f, 0.8f);
    flow_plot(px, py, pw, ph, ht, htk, 0.85f, 0.25f, 0.1f);
    flow_legend(W, H);
    if (g_fcfg.view == 4) {
        bool seen[7] = {false, false, false, false, false, false, false}; int row = 0;
        static const char* WN[7] = {"vorticity", "velocity (arrows along the wall)", "turbulence", "", "pressure (gauge)", "temperature", "liquid fraction"};
        for (int f = 0; f < 6; ++f) {
            int fld = g_wfld[f]; if (fld < 0 || seen[fld]) continue; seen[fld] = true;
            float x0 = (float)W - 345, y0 = (float)H - 168 - 52.f * row, w = 300, h = 12;
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glColor4f(1, 1, 1, 0.85f); glBegin(GL_QUADS); glVertex2f(x0 - 8, y0 - 22); glVertex2f(x0 + w + 8, y0 - 22); glVertex2f(x0 + w + 8, y0 + h + 22); glVertex2f(x0 - 8, y0 + h + 22); glEnd();
            glDisable(GL_BLEND);
            float lo0 = g_flo, hi0 = g_fhi; g_flo = g_wlo[fld]; g_fhi = g_whi[fld];
            glBegin(GL_QUAD_STRIP);
            for (int k = 0; k <= 32; ++k) { float r, g, b2, t; flow_colour(fld, g_flo + (g_fhi - g_flo) * k / 32.0, r, g, b2, t); glColor3f(r, g, b2); glVertex2f(x0 + w * k / 32, y0); glVertex2f(x0 + w * k / 32, y0 + h); }
            glEnd();
            char tb[160];
            glColor3f(0.1f, 0.12f, 0.2f);
            snprintf(tb, sizeof tb, "walls: %s, %s", WN[fld], FLOW_UNIT[fld]); gl_text(x0, y0 + h + 6, tb);
            snprintf(tb, sizeof tb, "%.3g", g_flo); gl_text(x0, y0 - 16, tb);
            snprintf(tb, sizeof tb, "%.3g", g_fhi); gl_text(x0 + w - 44, y0 - 16, tb);
            g_flo = lo0; g_fhi = hi0;
            ++row;
        }
    }
    flow_surface_legend(W, H);
    flow_polar_labels();
    glColor3f(0.1f, 0.12f, 0.2f); gl_text(px, py - 16, "energy (blue) and its fluctuating part (red)");
}

// ---------------------------------------------------------------- results
static std::string flow_summary() {
    FlowSim& S = *g_flow;
    std::string r; char b[600];
    auto line = [&](const std::string& s) { r += s; r += "\n"; };
    snprintf(b, sizeof b, "FLOW SIMULATION: %s in %s", g_fcfg.mode == 0 ? "a current past the parts" : g_fcfg.mode == 2 ? "a partly filled bath (a free surface, the gas above at constant pressure)" : "a stirred vessel", S.medium.c_str()); line(b);
    if (S.fs) { snprintf(b, sizeof b, "  liquid surface at z = %.4g m at the start; the liquid's mass at the end %.2f %% of the start (%ld cells changed between liquid and gas)", S.zsurf, S.mass0 > 0 ? 100 * S.mass_now / S.mass0 : 0.0, S.conversions); line(b); }
    snprintf(b, sizeof b, "  rho %.4g kg/m3, nu %.3g m2/s; Reynolds number %.4g (%s), on the %s", S.rho, S.nu, S.Re, flow_regime(S.Re, g_fcfg.mode).c_str(),
             g_fcfg.mode == 0 ? ("body size " + fmtd(std::round(S.Lref * 1e5) / 100) + " mm at " + fmtd(S.Uref) + " m/s").c_str() : ("stirrer diameter " + fmtd(std::round(S.Lref * 1e5) / 100) + " mm").c_str()); line(b);
    snprintf(b, sizeof b, "  grid %d x %d x %d = %.3g cells, dx %.4g mm (%.1f cells across the body), dt %.4g s, tau %.5f; %ld steps, %.4g s simulated", S.nx, S.ny, S.nz, (double)S.N, S.dx * 1000, S.Lref / S.dx, S.dt, S.tau0, S.step, S.t); line(b);
    double res_re = S.ulb * (S.Lref / S.dx) / ((S.tau0 - 0.5) / 3);
    snprintf(b, sizeof b, "  the grid alone would carry Re %.3g; mean eddy viscosity %.2g x molecular: %s", res_re, S.nut,
             S.smag == 0 ? "direct simulation: the grid resolves every eddy, no turbulence model" : S.nut > 0.5 ? "large-eddy simulation (the Smagorinsky model carries the small eddies)" : "large-eddy model on, adding little"); line(b);
    if (S.tau0 <= 0.5 + 1.01e-4) line("  note: the molecular viscosity is below what the grid can hold; the relaxation time sits at its floor (the effective viscosity is higher)");
    if (S.failed) line("  THE SOLUTION DIVERGED: the grid is too coarse for this Reynolds number. Use more cells, a slower current or a smaller stirrer speed.");
    if (S.clamps > 0) { snprintf(b, sizeof b, "  %ld cell updates were limited (speeds over Mach 0.5, or pockets narrower than a cell between moving walls reset): numbers there are not reliable", S.clamps); line(b); }
    if (g_fcfg.mode == 0) { snprintf(b, sizeof b, "  blockage: the parts fill %.1f %% of the cross-section (slip walls); above 5 %% it raises drag and shedding frequency", S.blockage * 100); line(b); }
    {   // the whole volume at the end: every cell of the x, y, z grid at once
        size_t nf = 0; double vs = 0, vmax = 0, Tsum = 0, Tmax = -1e30, Tmin = 1e30, ksum = 0, pmin = 1e30, pmax = -1e30, liq = 0, warm = 0;
        V3 Tat, vat; double vel = S.dx / S.dt, cell = S.dx * S.dx * S.dx;
        bool hasT = S.thermal && S.Tsnap.size() == S.N;
        for (size_t id = 0; id < S.N; ++id) {
            if (S.flag[id] != FL_FLUID || S.usnap.size() != 4 * S.N) continue;
            ++nf;
            const float* u = &S.usnap[4 * id];
            double sp = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) * vel, pr = (u[3] - 1.0) / 3.0 * S.rho * vel * vel;
            int i = (int)(id % S.nx), j = (int)((id / S.nx) % S.ny), k = (int)(id / ((size_t)S.nx * S.ny));
            V3 x = S.o + V3(i * S.dx, j * S.dx, k * S.dx);
            vs += sp; if (sp > vmax) { vmax = sp; vat = x; }
            ksum += S.kvar[id] * vel * vel; pmin = std::min(pmin, pr); pmax = std::max(pmax, pr);
            if (hasT) { double T = S.Tsnap[id]; Tsum += T; if (T > Tmax) { Tmax = T; Tat = x; } Tmin = std::min(Tmin, T); if (T > 0.5 * (S.Tin + S.Thot)) warm += 1; }
            if (S.fs) liq += S.iface[id] ? std::max(0.f, std::min(1.f, S.Mc[id] / std::max(1e-6f, S.Rc[id]))) : 1.0;
        }
        if (nf) {
            line("");
            snprintf(b, sizeof b, "THE WHOLE VOLUME at the end (%zu fluid cells over %d x %d x %d; %.4g litres of fluid)", nf, S.nx, S.ny, S.nz, nf * cell * 1000); line(b);
            snprintf(b, sizeof b, "  speed: mean %.4g m/s, highest %.4g m/s at (%.4g, %.4g, %.4g) m; turbulent energy %.4g J/kg on average; pressure %.4g to %.4g Pa",
                     vs / nf, vmax, vat.x, vat.y, vat.z, ksum / nf, pmin, pmax); line(b);
            if (hasT) { snprintf(b, sizeof b, "  temperature: mean %.4g C, %.4g to %.4g C, the hottest fluid at (%.4g, %.4g, %.4g) m; %.1f %% of the volume warmer than %.4g C", Tsum / nf, Tmin, Tmax, Tat.x, Tat.y, Tat.z, 100 * warm / nf, 0.5 * (S.Tin + S.Thot)); line(b); }
            if (S.fs) { snprintf(b, sizeof b, "  liquid: %.4g litres (%.1f %% of the box's fluid space), the rest gas", liq * cell * 1000, 100 * liq / nf); line(b); }
        }
    }
    line("");
    // forces, averaged over the second half
    size_t nh = S.h_t.size(), h0 = nh / 2;
    int ax = g_fcfg.dir / 2; double sg = flow_dir_sign(g_fcfg.dir);
    line("PART                 MEAN FORCE N (x, y, z)                    TORQUE ABOUT ITS AXIS N m   POWER W   NOTES");
    for (size_t bi = 0; bi < S.bodies.size(); ++bi) {
        int c = S.bodies[bi]; const std::vector<double>& H = S.h_f[bi];
        if (H.size() < 6 * nh || nh < 4) continue;
        V3 Fm, Tm; double n = 0;
        for (size_t i = h0; i < nh; ++i) { Fm = Fm + V3(H[6 * i], H[6 * i + 1], H[6 * i + 2]); Tm = Tm + V3(H[6 * i + 3], H[6 * i + 4], H[6 * i + 5]); n += 1; }
        Fm = Fm / n; Tm = Tm / n;
        std::string note; double Ta = 0, P = 0;
        const FMover* mvp = nullptr; for (const FMover& m : S.mv) if (m.comp == c) mvp = &m;
        if (mvp) {
            Ta = geo::dot(Tm - geo::cross(mvp->p, Fm), mvp->d); P = -Ta * mvp->w;
            bool stirrer = true; for (const FMover& m2 : S.mv) if (std::fabs(m2.w) * m2.R > std::fabs(mvp->w) * mvp->R * 1.0001) stirrer = false;
            if (g_fcfg.mode == 1 && stirrer) {
                double Nr = std::fabs(mvp->w) / (2 * geo::PI), Np = P / (S.rho * Nr * Nr * Nr * std::pow(S.Lref, 5));
                snprintf(b, sizeof b, "power number Np = P / (rho N^3 D^5) = %.3g (stirrers in baffled tanks, turbulent: Rushton about 5, pitched blade about 1.3)", Np); note = b;
            }
        }
        if (g_fcfg.mode == 0 && !mvp) {
            double Fd = Fm[ax] * sg;
            // the projected area by the grid: solid cells of this part seen along the current
            int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3, nn[3] = {S.nx, S.ny, S.nz}; size_t cnt = 0;
            for (int u1 = 0; u1 < nn[a1]; ++u1) for (int u2 = 0; u2 < nn[a2]; ++u2) { bool hit = false; for (int s = 0; s < nn[ax] && !hit; ++s) { int ii[3]; ii[ax] = s; ii[a1] = u1; ii[a2] = u2; size_t id = ((size_t)ii[2] * S.ny + ii[1]) * S.nx + ii[0]; hit = S.owner[id] == c; } if (hit) ++cnt; }
            double A = cnt * S.dx * S.dx, q = 0.5 * S.rho * S.Uref * S.Uref;
            if (A > 0) {
                // lift: the cross component that swings most; the shedding frequency from its zero crossings
                int lc = -1; double best = 0;
                for (int a = 0; a < 3; ++a) { if (a == ax) continue; double m = 0, v = 0; for (size_t i = h0; i < nh; ++i) m += H[6 * i + (size_t)a]; m /= (nh - h0); for (size_t i = h0; i < nh; ++i) v += std::pow(H[6 * i + (size_t)a] - m, 2); v = std::sqrt(v / (nh - h0)); if (v > best) { best = v; lc = a; } }
                snprintf(b, sizeof b, "Cd %.3g on %.4g mm2 projected; lift rms %.3g N (Cl rms %.3g)", Fd / (q * A), A * 1e6, best, best / (q * A)); note = b;
                if (lc >= 0 && best > 1e-12) {
                    double m = 0; for (size_t i = h0; i < nh; ++i) m += H[6 * i + (size_t)lc]; m /= (nh - h0);
                    bool longrun = S.h_t[nh - 1] - S.h_t[h0] >= 15 * S.Lref / S.Uref;
                    double fbest = 0, abest = 0, T = S.h_t[nh - 1] - S.h_t[h0];
                    std::vector<double> spec;
                    if (longrun) for (int kf = 16; kf < 4 * (int)(nh - h0); ++kf) {  // the strongest line of the lift's spectrum (Hann window, 8x zero-padded)
                        double f = kf / (8 * T), re = 0, im = 0;
                        for (size_t i = h0; i < nh; ++i) { double wv = 0.5 - 0.5 * std::cos(2 * geo::PI * (i - h0) / (double)(nh - h0 - 1)), v = (H[6 * i + (size_t)lc] - m) * wv; re += v * std::cos(2 * geo::PI * f * S.h_t[i]); im -= v * std::sin(2 * geo::PI * f * S.h_t[i]); }
                        double amp = re * re + im * im; spec.push_back(amp); if (amp > abest) { abest = amp; fbest = f; }
                    }
                    if (fbest > 0 && spec.size() > 2) {  // parabolic peak
                        size_t kb = (size_t)std::round(fbest * 8 * T) - 16;
                        if (kb > 0 && kb + 1 < spec.size()) { double a0 = spec[kb - 1], a1 = spec[kb], a2 = spec[kb + 1], den = a0 - 2 * a1 + a2; if (den < 0) fbest += 0.5 * (a0 - a2) / den / (8 * T); }
                    }
                    if (!longrun) note += "; too short a run to measure the shedding (it needs 30 or more body lengths of travel)";
                    double tfirst = 0, tlast = fbest > 0 ? 1 : 0; int cross = 3;
                    if (fbest > 0 && tlast > tfirst) {
                        double f = fbest, St = f * S.Lref / S.Uref; (void)cross;
                        snprintf(b, sizeof b, "; shedding %.3g Hz, Strouhal %.3g (a circular cylinder: 0.16 to 0.17 at Re 100, about 0.2 from Re 300 to 2e5)", f, St); note += b;
                    }
                }
            }
        }
        snprintf(b, sizeof b, "%-18.18s %10.4g %10.4g %10.4g   %16s %12s   %s", flow_id(c).c_str(), Fm.x, Fm.y, Fm.z, mvp ? fmtd(Ta).c_str() : "-", mvp ? fmtd(P).c_str() : "-", note.c_str());
        line(b);
    }
    for (size_t pi = 0; pi < S.pol.size(); ++pi) {
        std::vector<double> fl, tp; flow_polar_tooth(S, pi, 0, fl); flow_polar_tooth(S, pi, 1, tp);
        if (S.pol_steps <= 0) continue;
        const FlowSim::Polar& P = S.pol[pi];
        double sf = 0, st = 0, mx = -1e30; int bm = 0;
        for (int bb = 0; bb < NPB; ++bb) { sf += fl[(size_t)bb]; st += tp[(size_t)bb]; double v = fl[(size_t)bb] + tp[(size_t)bb]; if (v > mx) { mx = v; bm = bb; } }
        sf *= (double)P.z / NPB; st *= (double)P.z / NPB;  // all the teeth together = the gear's loss torque
        line("");
        snprintf(b, sizeof b, "NO-LOAD LOSS TORQUE OF %s (%d teeth, %.0f rpm; averaged over the second half, by the angle round the gear: 90 deg is up)", flow_id(P.comp).c_str(), P.z, std::fabs(P.w) * 30 / geo::PI); line(b);
        snprintf(b, sizeof b, "  flank zone %.4g N m, tip-circle zone %.4g N m, together %.4g N m (%.4g W at this speed); the flanks carry %.0f %%", sf, st, sf + st, (sf + st) * std::fabs(P.w), sf + st != 0 ? 100 * sf / (sf + st) : 0.0); line(b);
        snprintf(b, sizeof b, "  a tooth loses most at %.0f deg (%.4g N m per tooth there); squeezing where teeth enter the liquid or the mesh, and drag through it, make the peaks", -180 + 360.0 * (bm + 0.5) / NPB, mx); line(b);
    }
    if (S.thermal) {
        line("");
        snprintf(b, sizeof b, "HEAT (hot parts at %.4g C in fluid at %.4g C; k %.3g W/mK; %s)", S.Thot, S.Tin, S.kf, g_fcfg.buoyancy ? "with buoyancy" : "no buoyancy"); line(b);
        line("PART                 HEAT TO THE FLUID W   WETTED AREA mm2   h W/m2K   Nu (on the body size)");
        for (size_t bi = 0; bi < S.bodies.size(); ++bi) {
            int c = S.bodies[bi]; if (!(S.compT[(size_t)c] == S.compT[(size_t)c]) || S.h_q[bi].size() < 4) continue;
            double q = 0, n = 0; for (size_t i = S.h_q[bi].size() / 2; i < S.h_q[bi].size(); ++i) { q += S.h_q[bi][i]; n += 1; } q /= n;
            double h = S.barea[bi] > 0 ? q / (S.barea[bi] * (S.Thot - S.Tin)) : 0;
            snprintf(b, sizeof b, "%-18.18s %18.4g %17.4g %9.4g %9.4g", flow_id(c).c_str(), q, S.barea[bi] * 1e6, h, h * S.Lref / S.kf); line(b);
        }
        line("  The temperature is carried by the computed flow (first-order upwind: numerically diffusive where the flow is fast) and");
        line("  conducted with the fluid's diffusivity; walls of hot parts by a ghost cell, the others insulated. h on the staircased");
        line("  wetted area; the grid seldom resolves the thermal boundary layer, so take h as an estimate (the Environment report gives");
        line("  the correlation values to compare).");
    }
    {  // the same parts by the Environment analysis' correlations
        bool env_shown = g_env.on; std::vector<EBody> eb_keep = g_eb;
        env_analyse(false);
        line("");
        line("COMPARED WITH THE ENVIRONMENT ANALYSIS (its correlations for the same parts in the same fluid)");
        line("PART                 QUANTITY                         THIS FLOW SIMULATION   CORRELATION   SOURCE OF THE CORRELATION");
        size_t nh2 = S.h_t.size(), h02 = nh2 / 2;
        for (size_t bi = 0; bi < S.bodies.size(); ++bi) {
            int c = S.bodies[bi];
            const EBody* e = nullptr; for (auto& q : g_eb) if (q.comp == c) e = &q;
            const FMover* mvp = nullptr; for (const FMover& m : S.mv) if (m.comp == c) mvp = &m;
            const std::vector<double>& H = S.h_f[bi];
            if (H.size() < 6 * nh2 || nh2 < 4) continue;
            V3 Fm, Tm; double n = 0;
            for (size_t i = h02; i < nh2; ++i) { Fm = Fm + V3(H[6 * i], H[6 * i + 1], H[6 * i + 2]); Tm = Tm + V3(H[6 * i + 3], H[6 * i + 4], H[6 * i + 5]); n += 1; }
            Fm = Fm / n; Tm = Tm / n;
            if (mvp) {  // churning: a disc of the part's radius (von Karman), the wetted share
                double w = std::fabs(mvp->w), R = mvp->R, Rw = w * R * R / S.nu, Cm = Rw < 3e5 ? 3.87 / std::sqrt(std::max(Rw, 1.0)) : 0.146 * std::pow(Rw, -0.2);
                double frac = S.fs && e && e->V > 0 ? e->Vsub / e->V : 1.0, Pd = Cm * 0.5 * S.rho * w * w * std::pow(R, 5) * w * frac;
                double Pf = -geo::dot(Tm - geo::cross(mvp->p, Fm), mvp->d) * mvp->w;
                snprintf(b, sizeof b, "%-18.18s   %-30s %20.4g %13.4g   %s", flow_id(c).c_str(), "churning loss W", Pf, Pd, S.fs ? "disc (von Karman) x submerged share" : "disc (von Karman)"); line(b);
                const js::Value* gm = g_ac[(size_t)c].part.find("gear");
                if (S.fs && gm && std::fabs(geo::dot(mvp->d, V3(0, 0, 1))) < 0.2) {  // a gear on a horizontal axis dipping in the bath: Changenet and Velex (2007)
                    double Rp = gm->num("pitch_radius", mvp->R), Ra = gm->num("tip_radius", mvp->R), bw = gm->num("width", 0.01), Dp = 2 * Rp;
                    double hh = geo::clampd(S.zsurf - (mvp->p.z - Ra), 0, 2 * Ra);
                    if (hh > 0) {
                        double ac = std::acos(geo::clampd((Ra - hh) / Ra, -1, 1)), seg = Ra * Ra * ac - (Ra - hh) * std::sqrt(std::max(0.0, 2 * Ra * hh - hh * hh));
                        double Sm = 2 * seg + 2 * Ra * ac * bw;  // the immersed faces and tip surface
                        double V0 = S.mass0 * S.dx * S.dx * S.dx, Rec = w * Rp * bw / S.nu, Fr = w * w * Rp / g_env.g;
                        double Cm = Rec < 7500 ? 1.366 * std::pow(hh / Dp, 0.45) * std::pow(V0 / (Dp * Dp * Dp), 0.1) * std::pow(Fr, -0.6) * std::pow(Rec, -0.21)
                                               : 3.644 * std::pow(hh / Dp, 0.1) * std::pow(V0 / (Dp * Dp * Dp), -0.35) * std::pow(Fr, -0.88) * std::pow(bw / Dp, 0.85);
                        double Tch = 0.5 * S.rho * w * w * Sm * Rp * Rp * Rp * Cm;
                        snprintf(b, sizeof b, "%-18.18s   %-30s %20.4g %13.4g   %s (Re %.3g, Fr %.3g, h/D %.2f)", flow_id(c).c_str(), "churning loss W (dip)", Pf, Tch * w, "Changenet and Velex 2007 (dip-lubricated spur gear)", Rec, Fr, hh / Dp); line(b);
                    }
                }
            }
            if (S.thermal && c < (int)S.compT.size() && S.compT[(size_t)c] == S.compT[(size_t)c] && S.h_q[bi].size() >= 4 && e) {
                double q = 0, m = 0; for (size_t i = S.h_q[bi].size() / 2; i < S.h_q[bi].size(); ++i) { q += S.h_q[bi][i]; m += 1; } q /= m;
                double hf = S.barea[bi] > 0 ? q / (S.barea[bi] * (S.Thot - S.Tin)) : 0, hc = e->h_liq > 0 ? e->h_liq : e->h_gas + e->h_rad;
                snprintf(b, sizeof b, "%-18.18s   %-30s %20.4g %13.4g   %s", flow_id(c).c_str(), "heat transfer h W/m2K", hf, hc, "sphere correlations on 6V/A (Whitaker, Churchill)"); line(b);
            }
            if (g_fcfg.mode == 0 && !mvp && e && e->drag > 0) {
                int ax = g_fcfg.dir / 2; double sg = flow_dir_sign(g_fcfg.dir);
                snprintf(b, sizeof b, "%-18.18s   %-30s %20.4g %13.4g   %s", flow_id(c).c_str(), "drag N", Fm[ax] * sg, e->drag, "sphere drag law (Clift and Gauvin) on 6V/A"); line(b);
            }
        }
        line("  The correlations treat each part as an equivalent sphere or disc alone in the fluid; the simulation sees its real");
        line("  shape, its neighbours and the vessel. A factor of two between them is common for shapes far from compact.");
        if (!env_shown) { g_env.on = false; g_eb = eb_keep; }  // the comparison does not switch the Environment display on
    }
    line("");
    // how the turbulence grew
    line("HOW THE TURBULENCE DEVELOPED (the fluctuating energy: turbulence, and at low Reynolds numbers the periodic unsteadiness of");
    line("the turning parts themselves)");
    double kmax = 0; for (double v : S.h_ke) kmax = std::max(kmax, v);
    double t_on = -1, t_sat = -1, tmax = 0; for (double v : S.h_tke) tmax = std::max(tmax, v);
    for (size_t i = 0; i < nh; ++i) { if (t_on < 0 && S.h_ke[i] > 0 && S.h_tke[i] > 0.01 * S.h_ke[i]) t_on = S.h_t[i]; if (t_sat < 0 && tmax > 0 && S.h_tke[i] > 0.8 * tmax) t_sat = S.h_t[i]; }
    if (g_fcfg.mode >= 1) line("  The stirrer starts in still fluid. Its blades drag the fluid round; at their tips the fluid leaves in thin shear layers");
    else line("  The current meets the parts at rest relative to them. Boundary layers grow on their faces and separate at the edges;");
    if (g_fcfg.mode >= 1) line("  that roll up into trailing vortices, the vortices stretch and break, and their fluctuations spread through the vessel.");
    else line("  the free shear layers roll up into vortices that are shed alternately into the wake and break down further downstream.");
    if (t_on >= 0) { snprintf(b, sizeof b, "  Here: the fluctuating (turbulent) energy passed 1 %% of the kinetic energy at t = %.3g s", t_on); line(b); }
    else line("  Here: the fluctuations stayed under 1 % of the kinetic energy (the flow stayed laminar or the run was too short)");
    if (t_sat >= 0 && t_on >= 0) { snprintf(b, sizeof b, "  and reached 80 %% of its peak at t = %.3g s; at the end it is %.1f %% of the kinetic energy (%.4g of %.4g J).", t_sat, S.h_ke.empty() ? 0.0 : 100 * S.h_tke.back() / std::max(1e-30, S.h_ke.back()), S.h_tke.empty() ? 0.0 : S.h_tke.back(), S.h_ke.empty() ? 0.0 : S.h_ke.back()); line(b); }
    line("");
    line("METHOD: lattice Boltzmann D3Q19, BGK with the Smagorinsky model (Cs 0.16, Hou et al. 1996), half-way bounce-back walls");
    line("(staircased on the grid), moving walls by Ladd's correction, forces by momentum exchange; Mach 0.1 in lattice units. The");
    line("turbulent energy is 1/2 <u'^2> about a running mean over about two flow times. Limits: the grid resolves the large eddies");
    line("only; no wall functions (thin boundary layers at high Re are under-resolved, drag and torque then carry tens of per cent);");
    line(g_fcfg.mode == 2 ? "translating parts stand still; the gas is not simulated (a constant pressure); meshing teeth closer than a cell pass torque" : "the free surface stays flat; translating parts stand still; isothermal.");
    if (g_fcfg.mode == 2) line("through the liquid between them (unresolved): take the losses of a meshing pair in a bath as unreliable.");
    return r;
}
// The whole volume by axis: for every plane across x, y and z the mean speed, the mean of each velocity component,
// the largest speed, the temperature and the liquid fraction, written as profile_x/y/z.csv; returns the report lines.
static double g_flow_badfrac = 0;
static std::string g_flow_bad_dump;
static std::string flow_axis_profiles(const std::string& dir) {
    FlowSim& S = *g_flow;
    if (g_fu.size() != 4 * S.N) return "";
    double vel = S.dx / S.dt;
    int nn[3] = {S.nx, S.ny, S.nz};
    std::string rep = "\nTHE WHOLE VOLUME BY AXIS (x, y and z; profiles in profile_x.csv, profile_y.csv, profile_z.csv)\n";
    double e[3] = {0, 0, 0}, etot = 0;
    for (size_t id = 0; id < S.N; ++id) if (S.flag[id] == FL_FLUID) for (int a = 0; a < 3; ++a) { double u = g_fu[4 * id + (size_t)a] * vel; e[a] += u * u; etot += u * u; }
    static const char* AN[3] = {"x", "y", "z"};
    long nbad = 0, nfl = 0;
    for (size_t id = 0; id < S.N; ++id) if (S.flag[id] == FL_FLUID) ++nfl;
    for (int a = 0; a < 3; ++a) {
        int n = nn[a];
        std::vector<double> cnt((size_t)n, 0), sp((size_t)n, 0), uc[3] = {std::vector<double>((size_t)n, 0), std::vector<double>((size_t)n, 0), std::vector<double>((size_t)n, 0)}, mx((size_t)n, 0), tt((size_t)n, 0), ph((size_t)n, 0);
        for (int k = 0; k < S.nz; ++k) for (int j = 0; j < S.ny; ++j) for (int i = 0; i < S.nx; ++i) {
            size_t id = ((size_t)k * S.ny + j) * S.nx + i;
            if (S.flag[id] != FL_FLUID) continue;
            int p = a == 0 ? i : a == 1 ? j : k;
            double ux = g_fu[4 * id] * vel, uy = g_fu[4 * id + 1] * vel, uz = g_fu[4 * id + 2] * vel, s = std::sqrt(ux * ux + uy * uy + uz * uz);
            if (s > 0.35 * vel) {
                if (a == 0) {
                    ++nbad;
                    if (getenv("AEROSYNAPSE_FLOW_BADDUMP")) {  // a diagnosis: where the cells beyond the valid range are and what is round them
                        int ns = 0, nm = 0, nfr = 0; std::string own;
                        for (int q = 1; q < LQ; ++q) { int a2 = i + LC[q][0], b2 = j + LC[q][1], e2 = k + LC[q][2]; if (a2 < 0 || b2 < 0 || e2 < 0 || a2 >= S.nx || b2 >= S.ny || e2 >= S.nz) continue; size_t n2 = ((size_t)e2 * S.ny + b2) * S.nx + a2; unsigned char f2 = S.flag[n2]; if (f2 == FL_STATIC) ++ns; if (f2 == FL_MOVING) { ++nm; own += tostr((int)S.owner[n2]) + " "; } if (f2 == FL_FLUID) ++nfr; }
                        char b[200]; snprintf(b, sizeof b, "%d,%d,%d,%.5g,%.5g,%d,%d,%d,%s\n", i, j, k, s, g_fu[4 * id + 3], ns, nm, nfr, own.c_str()); g_flow_bad_dump += b;
                    }
                }
                continue;
            }  // beyond the lattice's valid speeds: an unresolved gap or a cell just uncovered by a moving part
            cnt[(size_t)p] += 1; sp[(size_t)p] += s; uc[0][(size_t)p] += ux; uc[1][(size_t)p] += uy; uc[2][(size_t)p] += uz; mx[(size_t)p] = std::max(mx[(size_t)p], s);
            if (g_fT.size() == S.N) tt[(size_t)p] += g_fT[id];
            if (g_fphi.size() == S.N) ph[(size_t)p] += g_fphi[id];
        }
        std::string c = std::string(AN[a]) + "_mm,mean_speed_m_s,mean_ux_m_s,mean_uy_m_s,mean_uz_m_s,max_speed_m_s,fluid_cells";
        if (g_fT.size() == S.N) c += ",mean_T";
        if (g_fphi.size() == S.N) c += ",liquid_fraction";
        c += "\n";
        int pk = 0; double best = -1;
        for (int p = 0; p < n; ++p) {
            double w = std::max(1.0, cnt[(size_t)p]), pos = ((a == 0 ? S.o.x : a == 1 ? S.o.y : S.o.z) + p * S.dx) * 1000;
            char b[260]; snprintf(b, sizeof b, "%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%d", pos, sp[(size_t)p] / w, uc[0][(size_t)p] / w, uc[1][(size_t)p] / w, uc[2][(size_t)p] / w, mx[(size_t)p], (int)cnt[(size_t)p]);
            c += b;
            if (g_fT.size() == S.N) { snprintf(b, sizeof b, ",%.5g", tt[(size_t)p] / w); c += b; }
            if (g_fphi.size() == S.N) { snprintf(b, sizeof b, ",%.4g", ph[(size_t)p] / w); c += b; }
            c += "\n";
            if (cnt[(size_t)p] > 0 && sp[(size_t)p] / w > best) { best = sp[(size_t)p] / w; pk = p; }
        }
        try { js::write_file(dir + "\\profile_" + AN[a] + ".csv", c); } catch (...) {}
        char b[240];
        snprintf(b, sizeof b, "  along %s: %d planes; %.1f %% of the kinetic energy is in u%s; the fastest plane is at %s = %.4g mm (mean speed %.4g m/s)\n",
                 AN[a], n, etot > 0 ? 100 * e[a] / etot : 0.0, AN[a], AN[a], ((a == 0 ? S.o.x : a == 1 ? S.o.y : S.o.z) + pk * S.dx) * 1000, std::max(0.0, best));
        rep += b;
    }
    g_flow_badfrac = (double)nbad / std::max(1L, nfl);
    if (!g_flow_bad_dump.empty()) { try { js::write_file(dir + "\\bad_cells.csv", "i,j,k,speed_m_s,density,static_nb,moving_nb,fluid_nb,owners\n" + g_flow_bad_dump); } catch (...) {} g_flow_bad_dump.clear(); }
    if (S.pinch_upd > 0) {
        char b[300]; snprintf(b, sizeof b, "  gaps one cell wide between parts, bridged as wall: %.0f cells on average per update (%.3f %% of the fluid), %ld at the end\n",
                              (double)S.pinch_sum / S.pinch_upd, 100.0 * S.pinch_sum / S.pinch_upd / std::max(1L, nfl), S.pinch_last);
        rep += b;
    }
    if (nbad) {
        char b[300]; snprintf(b, sizeof b, "  %ld of %ld fluid cells (%.3f %%) had speeds beyond the lattice's valid range (above 0.35 dx/dt = %.3g m/s): cells in gaps\n"
                                           "  narrower than the grid or just uncovered by a moving part. They are left out of these profiles; use more cells to resolve them.\n",
                 nbad, nfl, 100.0 * nbad / std::max(1L, nfl), 0.35 * vel);
        rep += b;
    }
    return rep;
}
// the last run's grid and results, for the comparison Refine grid writes
struct FlowRunMark { bool valid = false; int cells = 0; double dx = 0, bad = 0, pinch = 0, ke = 0; std::vector<std::string> ids; std::vector<double> T, F; };
static FlowRunMark g_flow_prev;
static std::string flow_grid_compare(FlowSim& S) {
    FlowRunMark now; now.valid = true; now.cells = g_fcfg.cells; now.dx = S.dx; now.bad = g_flow_badfrac; now.ke = S.ke;
    now.pinch = S.pinch_upd ? (double)S.pinch_sum / S.pinch_upd : 0;
    for (size_t b = 0; b < S.bodies.size(); ++b) {
        const std::vector<double>& h = S.h_f[b]; size_t n = h.size() / 6, k0 = n / 2; double T[3] = {0, 0, 0}, F[3] = {0, 0, 0}; int m = 0;
        for (size_t i = k0; i < n; ++i, ++m) for (int a = 0; a < 3; ++a) { F[a] += h[6 * i + (size_t)a]; T[a] += h[6 * i + 3 + (size_t)a]; }
        m = std::max(1, m);
        now.ids.push_back(flow_id(S.bodies[b])); now.T.push_back(std::sqrt(T[0] * T[0] + T[1] * T[1] + T[2] * T[2]) / m); now.F.push_back(std::sqrt(F[0] * F[0] + F[1] * F[1] + F[2] * F[2]) / m);
    }
    std::string rep;
    if (g_flow_prev.valid && g_flow_prev.ids == now.ids && g_flow_prev.cells != now.cells) {
        char b[400];
        rep += "\nGRID REFINEMENT (the same set-up on two grids; second-half means)\n";
        snprintf(b, sizeof b, "  %-22s %14s %14s %10s\n", "", (tostr(g_flow_prev.cells) + " cells").c_str(), (tostr(now.cells) + " cells").c_str(), "change"); rep += b;
        snprintf(b, sizeof b, "  %-22s %11.3f mm %11.3f mm\n", "cell size", g_flow_prev.dx * 1000, now.dx * 1000); rep += b;
        snprintf(b, sizeof b, "  %-22s %12.4f %% %12.4f %%\n", "invalid-speed cells", 100 * g_flow_prev.bad, 100 * now.bad); rep += b;
        snprintf(b, sizeof b, "  %-22s %14.0f %14.0f\n", "bridged gap cells", g_flow_prev.pinch, now.pinch); rep += b;
        snprintf(b, sizeof b, "  %-22s %12.4g J %12.4g J %9.1f %%\n", "kinetic energy", g_flow_prev.ke, now.ke, g_flow_prev.ke > 0 ? 100 * (now.ke / g_flow_prev.ke - 1) : 0.0); rep += b;
        for (size_t i = 0; i < now.ids.size(); ++i) {
            snprintf(b, sizeof b, "  %-22s %10.4g N m %10.4g N m %9.1f %%\n", (now.ids[i] + " torque").c_str(), g_flow_prev.T[i], now.T[i], g_flow_prev.T[i] > 0 ? 100 * (now.T[i] / g_flow_prev.T[i] - 1) : 0.0); rep += b;
        }
        rep += "  Where the values change little between the grids, the coarser one was fine enough; where they change a lot, refine again.\n";
    }
    g_flow_prev = now;
    return rep;
}
static void flow_start();
static void flow_cfg_to_doc();
static void flow_refine() {  // Refine grid: the same set-up again with 1.5 times the cells, compared in the report
    if (!g_flow_prev.valid) { status("Refine grid: run a flow simulation first; this runs it again with 1.5 times the cells and compares the two grids"); return; }
    if (g_fcfg.cells >= 320) { status("Refine grid: already at 320 cells along the longest side (the limit)"); return; }
    if (g_flow) flow_stop(false);
    g_fcfg.cells = std::min(320, (int)std::lround(g_fcfg.cells * 1.5));
    flow_cfg_to_doc();
    status("Refine grid: the same flow on " + tostr(g_fcfg.cells) + " cells along the longest side (the report compares it with the last run)");
    flow_start();
}
static void flow_write_results() {
    if (!g_flow || g_selftest) return;
    FlowSim& S = *g_flow;
    SYSTEMTIME tm; GetLocalTime(&tm);
    char nm[64]; snprintf(nm, sizeof nm, "flow_%04d%02d%02d_%02d%02d%02d", tm.wYear, tm.wMonth, tm.wDay, tm.wHour, tm.wMinute, tm.wSecond);
    std::string dir = g_root + "\\results\\studio\\" + nm;
    make_dirs(dir);
    std::string sum; EnterCriticalSection(&S.cs); sum = flow_summary();
    std::string csv = "t_s,kinetic_J,turbulent_J,max_speed_m_s";
    for (int c : S.bodies) { std::string n = flow_id(c); csv += "," + n + "_Fx," + n + "_Fy," + n + "_Fz," + n + "_Tx," + n + "_Ty," + n + "_Tz"; }
    csv += "\n";
    for (size_t i = 0; i < S.h_t.size(); ++i) {
        char b[160]; snprintf(b, sizeof b, "%.6g,%.6g,%.6g,%.6g", S.h_t[i], S.h_ke[i], S.h_tke[i], S.h_umax[i]); csv += b;
        for (size_t bi = 0; bi < S.bodies.size(); ++bi) for (int a = 0; a < 6; ++a) { double v = S.h_f[bi].size() > 6 * i + (size_t)a ? S.h_f[bi][6 * i + (size_t)a] : 0; snprintf(b, sizeof b, ",%.6g", v); csv += b; }
        csv += "\n";
    }
    LeaveCriticalSection(&S.cs);
    sum += flow_axis_profiles(dir);
    sum += flow_grid_compare(S);
    try { js::write_file(dir + "\\summary.txt", sum); js::write_file(dir + "\\history.csv", csv); } catch (...) {}
    snap_view_png(dir + "\\00_view_3d.png");
    g_flow_last_dir = dir;
    open_simview(dir, "Flow simulation");
}

// ---------------------------------------------------------------- running it
static void flow_stop(bool keep_view) {
    if (!g_flow) return;
    if (g_flow_running) { g_flow->stop = 1; WaitForMultipleObjects((DWORD)g_flow_threads.size(), g_flow_threads.data(), TRUE, INFINITE); for (HANDLE h : g_flow_threads) CloseHandle(h); g_flow_threads.clear(); g_flow_running = false; KillTimer(g_main, 44); flow_write_results(); }
    if (!keep_view) { flow_restore_poses(); flow_free(); g_flow_on = false; g_fparts.clear(); g_fiso_t.clear(); }
    InvalidateRect(g_view, nullptr, FALSE);
}
// Smooth playback: the view blends the last two snapshots over the time between them (a frame every 25 ms), so the
// colours change smoothly, the particles move every frame and the parts turn steadily instead of jumping per snapshot.
static std::vector<float> g_fu_a, g_fu_b, g_fu_new, g_fT_a, g_fT_b, g_fT_new;
static double g_ft_a = 0, g_ft_b = 0, g_fblend = 1;
static DWORD g_fw_b = 0, g_fw_gap = 400;
static bool g_flow_smooth = true;
static void flow_smooth_reset() { g_fu_a.clear(); g_fu_b.clear(); g_fu_new.clear(); g_fT_a.clear(); g_fT_b.clear(); g_fT_new.clear(); g_ft_a = g_ft_b = 0; g_fblend = 1; }
static void flow_smooth_final() {  // the newest snapshot as it is (the end of a run)
    if (!g_flow || g_fu_b.size() != 4 * g_flow->N) return;
    g_fu = g_fu_b; if (g_fT_b.size() == g_flow->N) g_fT = g_fT_b;
    g_flow_tvis = g_ft_b; g_fblend = 1;
    flow_slice_texture(); flow_volume_build(); flow_surface_build();
}
static void flow_tick() {
    if (!g_flow) { KillTimer(g_main, 44); return; }
    FlowSim& S = *g_flow;
    if (g_flow_base.size() != g_ac.size()) { status("the assembly changed: the flow simulation stopped"); flow_stop(false); return; }
    bool fresh = false; double tnow;
    EnterCriticalSection(&S.cs);
    fresh = S.snap_new; S.snap_new = false; tnow = S.t;
    if (fresh) { g_fu_new = S.usnap; if (S.thermal) g_fT_new = S.Tsnap; if (S.fs) g_fphi = S.phisnap; }
    LeaveCriticalSection(&S.cs);
    DWORD now = GetTickCount();
    if (fresh && g_fu_new.size() == 4 * S.N) {
        bool first = g_fu_b.size() != g_fu_new.size();
        g_fu_a.swap(g_fu_b); g_fu_b.swap(g_fu_new);  // a the previous snapshot, b the newest
        if (first) g_fu_a = g_fu_b;
        if (S.thermal && g_fT_new.size() == S.N) { g_fT_a.swap(g_fT_b); g_fT_b.swap(g_fT_new); if (g_fT_a.size() != g_fT_b.size()) g_fT_a = g_fT_b; }
        g_ft_a = first ? tnow : g_ft_b; g_ft_b = tnow;
        g_fw_gap = first ? 1 : std::max<DWORD>(30, std::min<DWORD>(4000, now - g_fw_b));
        g_fw_b = first ? now - 1 : now;
        g_fblend = -1;
    }
    if (g_fu_b.size() == 4 * S.N) {
        double a = g_flow_smooth ? std::min(1.0, (now - g_fw_b) / (double)g_fw_gap) : 1.0;
        if (a != g_fblend) {
            float fa = (float)a, fb = 1 - fa;
            g_fu.resize(g_fu_b.size());
            for (size_t i = 0; i < g_fu.size(); ++i) g_fu[i] = fb * g_fu_a[i] + fa * g_fu_b[i];
            if (S.thermal && g_fT_b.size() == S.N && g_fT_a.size() == S.N) { g_fT.resize(S.N); for (size_t i = 0; i < S.N; ++i) g_fT[i] = fb * g_fT_a[i] + fa * g_fT_b[i]; }
            double tv = g_ft_a + a * (g_ft_b - g_ft_a);
            flow_particles(std::max(0.0, tv - g_flow_tvis));
            g_flow_tvis = tv; g_fblend = a;
            flow_slice_texture();
            if (now - g_fvol_ms > (g_flow_smooth ? 300u : 700u)) { flow_volume_build(); flow_surface_build(); g_fvol_ms = now; }
            if (now - g_fiso_ms > 1500) { flow_cores(); g_fiso_ms = now; }
        }
    }
    flow_poses();
    if (S.done && g_flow_running) {
        WaitForMultipleObjects((DWORD)g_flow_threads.size(), g_flow_threads.data(), TRUE, INFINITE);
        for (HANDLE h : g_flow_threads) CloseHandle(h);
        g_flow_threads.clear(); g_flow_running = false; KillTimer(g_main, 44);
        flow_smooth_final();
        std::string where;
        if (S.failed && S.bad_id >= 0) {
            size_t id = (size_t)S.bad_id; int i = (int)(id % S.nx), j = (int)((id / S.nx) % S.ny), k = (int)(id / ((size_t)S.nx * S.ny));
            int nfl[4] = {0, 0, 0, 0};
            for (int q = 1; q < LQ; ++q) { int a = i - LC[q][0], b = j - LC[q][1], e = k - LC[q][2]; if (a < 0 || b < 0 || e < 0 || a >= S.nx || b >= S.ny || e >= S.nz) continue; nfl[S.flag[((size_t)e * S.ny + b) * S.nx + a] & 3]++; }
            where = " (first at cell " + tostr(i) + "," + tostr(j) + "," + tostr(k) + ", density " + fmtd(S.bad_r) + ", neighbours fluid " + tostr(nfl[0]) + " static " + tostr(nfl[1]) + " moving " + tostr(nfl[2]) + ")";
            g_flow_msg = where;
        }
        status(S.failed ? "the flow diverged at t = " + fmtd(S.t) + " s" + where + ": use more cells or a slower flow" : "flow simulation finished: " + fmtd(S.t) + " s simulated");
        flow_write_results();
    }
    InvalidateRect(g_view, nullptr, FALSE);
}
static void flow_start() {
    flow_stop(false);
    ms_reset();
    FlowSim* S = new FlowSim();
    InitializeCriticalSection(&S->cs);
    g_flow = S;
    status("flow: cutting the parts into the grid...");
    std::string err = flow_build(*S);
    if (!err.empty()) { flow_free(); MessageBoxA(g_main, err.c_str(), "Flow simulation", MB_ICONINFORMATION); return; }
    g_flow_base.clear(); for (const AComp& c : g_ac) g_flow_base.push_back(c.pose);
    SYSTEM_INFO si; GetSystemInfo(&si);
    S->nthr = g_selftest ? 1 : std::max(1, std::min(16, (int)si.dwNumberOfProcessors - 1));
    S->frc.assign((size_t)S->nthr, std::vector<double>(S->bodies.size() * 6, 0.0));
    S->fsum.assign(S->bodies.size() * 6, 0.0);
    S->stat.assign((size_t)S->nthr * 5, 0.0);
    S->hq.assign((size_t)S->nthr, std::vector<double>(S->bodies.size(), 0.0));
    if (!S->pol.empty()) S->pacc.assign((size_t)S->nthr, std::vector<double>(S->pol.size() * 2 * NPB, 0.0));
    S->t0ms = GetTickCount();
    g_flow_on = true; g_flow_tvis = 0; g_fparts.clear(); g_fiso_t.clear(); g_fscale = 1; flow_smooth_reset();
    if (g_selftest) {  // a few steps in this thread: the code runs, the test stays quick
        S->steps_total = std::min(S->steps_total, 12L); S->snap_every = 4;
        flow_worker((void*)0);
        g_st_log.push_back("flow: " + tostr((int)S->step) + " steps on " + tostr(S->nx) + "x" + tostr(S->ny) + "x" + tostr(S->nz) + (S->failed ? " DIVERGED" : " ok"));
        flow_tick();
        return;
    }
    for (int t = 0; t < S->nthr; ++t) g_flow_threads.push_back((HANDLE)_beginthreadex(nullptr, 0, flow_worker, (void*)(INT_PTR)t, 0, nullptr));
    g_flow_running = true;
    SetTimer(g_main, 44, 25, nullptr);  // 40 frames a second: smooth playback
    status("flow running: " + tostr(S->nx) + " x " + tostr(S->ny) + " x " + tostr(S->nz) + " cells, Re " + fmtd(std::round(S->Re)) + ", " + fmtd(std::round(S->steps_total * S->dt * 1000) / 1000) + " s to simulate on " + tostr(S->nthr) + " threads (Stop flow ends it early)");
}
static void flow_cfg_from_doc() {
    if (const js::Value* f = g_doc.find("flow")) {
        g_fcfg.mode = (int)f->num("mode", g_fcfg.mode); g_fcfg.cells = (int)f->num("cells", g_fcfg.cells); g_fcfg.U = f->num("current_m_s", g_fcfg.U); g_fcfg.dir = (int)f->num("dir", g_fcfg.dir);
        g_fcfg.duration = f->num("duration_s", 0); g_fcfg.span_walls = f->flag("span_walls", false); g_fcfg.span_axis = (int)f->num("span_axis", -1); g_fcfg.margin = f->num("margin", g_fcfg.margin);
        g_fcfg.show = (int)f->num("show", 1); g_fcfg.view = (int)f->num("view", 0); g_fcfg.heat = (int)f->num("heat", 1); g_fcfg.Thot = f->num("hot_C", 0); g_fcfg.buoyancy = f->flag("buoyancy", true); g_fcfg.surf = (int)f->num("surface_field", 1); g_fcfg.walls = (int)f->num("walls", 3);
        g_fcfg.slice_axis = (int)f->num("slice_axis", 2); g_fcfg.slice_pos = f->num("slice_pos", 0.5); g_fcfg.cores = f->flag("cores", true); g_fcfg.particles = f->flag("particles", false);
    }
}
static void flow_cfg_to_doc() {
    js::Value f = js::Value::object();
    f["mode"] = (double)g_fcfg.mode; f["cells"] = (double)g_fcfg.cells; f["current_m_s"] = g_fcfg.U; f["dir"] = (double)g_fcfg.dir; f["duration_s"] = g_fcfg.duration; f["span_walls"] = g_fcfg.span_walls; f["span_axis"] = (double)g_fcfg.span_axis;
    f["margin"] = g_fcfg.margin; f["show"] = (double)g_fcfg.show; f["view"] = (double)g_fcfg.view; f["heat"] = (double)g_fcfg.heat; f["hot_C"] = g_fcfg.Thot; f["buoyancy"] = g_fcfg.buoyancy; f["surface_field"] = (double)g_fcfg.surf; f["walls"] = (double)g_fcfg.walls;
    f["slice_axis"] = (double)g_fcfg.slice_axis; f["slice_pos"] = g_fcfg.slice_pos; f["cores"] = g_fcfg.cores; f["particles"] = g_fcfg.particles;
    g_doc["flow"] = f;
}
// the flow set-up the Environment implies: a liquid surface across the parts = a bath (free surface), a current = parts
// in a current, otherwise a closed box of the fluid; hot parts when their temperature differs from the fluid's
// The grid from the finest gap of the parts: a gear's tooth space is about 1.6 modules wide, so three cells across it
// need dx <= module / 2; other parts get at least ten cells across their smallest size. Capped at 200 cells along the
// longest side (Refine grid goes further). g_flow_grid_note says what was chosen and why.
static std::string g_flow_grid_note;
static int flow_auto_cells(double margin) {
    double need = 1e9, smallest = 1e9; std::string why;
    geo::Box B;
    for (const AComp& c : g_ac) {
        if (!comp_ready(c)) continue;
        geo::Box b = flow_wbox(c); B.add(b.lo); B.add(b.hi);
        double sz = std::min(b.hi.x - b.lo.x, std::min(b.hi.y - b.lo.y, b.hi.z - b.lo.z)); if (sz > 0) smallest = std::min(smallest, sz);
        for (const APiece& pc : c.pieces) {
            size_t at = 0;
            while ((at = pc.key.find("\"module\"", at)) != std::string::npos) {
                size_t col = pc.key.find(':', at); if (col == std::string::npos) break;
                double m = atof(pc.key.c_str() + col + 1);
                if (m > 0 && 0.5 * m < need) { need = 0.5 * m; why = "the gear teeth of " + c.id + " (module " + fmtd(m * 1000) + " mm: three cells across a tooth space)"; }
                at = col;
            }
        }
    }
    if (smallest < 1e8 && 0.1 * smallest < need) { need = 0.1 * smallest; why = "ten cells across the smallest part"; }
    if (!B.valid() || need > 1e8) { g_flow_grid_note = ""; return g_fcfg.cells; }
    double L = std::max(B.hi.x - B.lo.x, std::max(B.hi.y - B.lo.y, B.hi.z - B.lo.z)) * (1 + 2 * margin);
    int want = (int)std::ceil(L / need), cells = std::max(64, std::min(200, want));
    g_flow_grid_note = "Grid: " + tostr(cells) + " cells along the longest side, chosen from " + why + (want > 200 ? "; " + tostr(want) + " would resolve it fully (Refine grid goes to 320)" : "") +
                       ". Gaps still one cell wide between parts are bridged as wall.";
    return cells;
}
static void flow_cfg_auto() {
    env_from_doc();
    geo::Box B; for (const AComp& c : g_ac) if (comp_ready(c)) { geo::Box b = flow_wbox(c); B.add(b.lo); B.add(b.hi); }
    Fluid F = fluid_props(g_env.medium, g_env.Tc, g_env.alt);
    bool turning = !g_amot.empty();
    if (F.liquid && B.valid() && g_env.surface > B.lo.z && g_env.surface < B.hi.z) { g_fcfg.mode = 2; g_fcfg.show = 6; }
    else if (g_env.U > 0) { g_fcfg.mode = 0; g_fcfg.U = g_env.U; g_fcfg.dir = g_env.dir; g_fcfg.show = 1; }
    else { g_fcfg.mode = 1; g_fcfg.show = 1; }
    bool hot = std::fabs(g_env.Tpart - g_env.Tc) > 1e-6;
    g_fcfg.heat = hot ? (turning ? 1 : 2) : 0; g_fcfg.Thot = 0;
    if (hot && g_fcfg.mode != 2) g_fcfg.show = 5;
    g_fcfg.view = 4; g_fcfg.surf = hot ? 6 : 1; g_fcfg.margin = g_fcfg.mode == 2 ? 0.2 : 0.15;
    g_fcfg.cells = flow_auto_cells(g_fcfg.margin);
    flow_cfg_to_doc();
}
// any design: a part design becomes the first part of an assembly for the flow (the Environment kept; Undo returns)
static bool flow_ensure_assembly() {
    if (g_asm_on) return true;
    js::Value env; bool has_env = false;
    if (const js::Value* e = g_doc.find("environment")) { env = e->clone(); has_env = true; }
    if (!g_selftest && !g_flow_auto_arg && MessageBoxA(g_main, "The flow simulation runs on an assembly: the design becomes its first part (the Environment is kept; Undo brings the design back). Go on?", "Flow simulation", MB_OKCANCEL | MB_ICONQUESTION) != IDOK) return false;
    asm_from_design();
    if (!g_asm_on) return false;
    if (has_env) g_doc["environment"] = env;
    return true;
}
static void flow_setup() {
    if (!flow_ensure_assembly()) return;
    if (!g_doc.find("flow")) flow_cfg_auto();
    flow_cfg_from_doc();
    env_from_doc();
    std::vector<FField> f;
    static const char* MODES[3] = {"Parts submerged in a current", "Closed vessel, the turning parts stir it", "Partly filled bath (free surface, gas above)"};
    static const int SHOWN[6] = {1, 4, 5, 2, 0, 6};
    static const char* SHOW[6] = {"Speed", "Pressure", "Temperature", "Turbulence", "Vorticity", "Liquid fraction (bath)"};
    static const char* VIEWS[5] = {"Whole fluid (coloured volume)", "Slice", "Volume and a slice", "x, y and z slices with the volume", "Cutaway box (walls and volume)"};
    static const char* HEAT[3] = {"None (isothermal)", "The turning parts are hot", "Every part is hot"};
    static const char* AX[3] = {"x", "y", "z"};
    int si = 0; for (int i = 0; i < 6; ++i) if (SHOWN[i] == g_fcfg.show) si = i;
    f.push_back({"Set-up", {MODES[0], MODES[1], MODES[2]}, MODES[std::max(0, std::min(2, g_fcfg.mode))]});
    f.push_back({"Cells along the longest side", {"48", "64", "96", "128", "160", "200", "240"}, tostr(g_fcfg.cells)});
    f.push_back({"Current: speed (m/s)", {}, fmtd(g_fcfg.U)});
    f.push_back({"Current: direction", {FLOW_DIR[0], FLOW_DIR[1], FLOW_DIR[2], FLOW_DIR[3], FLOW_DIR[4], FLOW_DIR[5]}, FLOW_DIR[g_fcfg.dir]});
    f.push_back({"Simulated time (s; 0 = automatic)", {}, fmtd(g_fcfg.duration)});
    f.push_back({"Show", {SHOW[0], SHOW[1], SHOW[2], SHOW[3], SHOW[4], SHOW[5]}, SHOW[si]});
    f.push_back({"As", {VIEWS[0], VIEWS[1], VIEWS[2], VIEWS[3], VIEWS[4]}, VIEWS[std::max(0, std::min(4, g_fcfg.view))]});
    f.push_back({"Slice across / at (0 to 1)", {AX[0], AX[1], AX[2]}, AX[g_fcfg.slice_axis]});
    f.push_back({"Slice position (0 to 1)", {}, fmtd(g_fcfg.slice_pos)});
    f.push_back({"Heat", {HEAT[0], HEAT[1], HEAT[2]}, HEAT[g_fcfg.heat]});
    f.push_back({"Hot parts at (C; 0 = Environment's)", {}, fmtd(g_fcfg.Thot)});
    f.push_back({"Natural convection (buoyancy)", {"yes", "no"}, g_fcfg.buoyancy ? "yes" : "no"});
    static const char* SURFS[7] = {"Nothing", "Friction (wall shear stress)", "Pressure", "Temperature", "Turbulence", "Liquid fraction (bath)", "Heat-transfer coefficient (hot parts)"};
    f.push_back({"Parts' surfaces show", {SURFS[0], SURFS[1], SURFS[2], SURFS[3], SURFS[4], SURFS[5], SURFS[6]}, SURFS[std::max(0, std::min(6, g_fcfg.surf))]});
    static const char* WALLS[4] = {"The field shown", "Pressure", "Velocity (speed and arrows)", "Pressure (x), velocity (y), the field (z)"};
    f.push_back({"Cutaway box walls show", {WALLS[0], WALLS[1], WALLS[2], WALLS[3]}, WALLS[std::max(0, std::min(3, g_fcfg.walls))]});
    f.push_back({"Vortex cores and particles", {"vortex cores", "both", "particles", "neither"}, g_fcfg.cores && g_fcfg.particles ? "both" : g_fcfg.cores ? "vortex cores" : g_fcfg.particles ? "particles" : "neither"});
    std::string intro = "The parts sit wholly in the Environment's fluid (" + std::string(MED_NAME[g_env.medium]) + " at " + fmtd(g_env.Tc) + " C). Turning parts, belts and chains move it at their motion-study speeds. More cells: finer eddies, slower." + (g_flow_grid_note.empty() ? std::string() : "\n" + g_flow_grid_note);
    if (!form_dialog("Flow simulation", intro, f)) return;
    g_fcfg.mode = f[0].value == MODES[2] ? 2 : f[0].value == MODES[1] ? 1 : 0;
    g_fcfg.cells = std::max(16, std::min(400, atoi(f[1].value.c_str())));
    g_fcfg.U = std::max(0.0, atof(f[2].value.c_str()));
    for (int i = 0; i < 6; ++i) if (f[3].value == FLOW_DIR[i]) g_fcfg.dir = i;
    g_fcfg.duration = std::max(0.0, atof(f[4].value.c_str()));
    for (int i = 0; i < 6; ++i) if (f[5].value == SHOW[i]) g_fcfg.show = SHOWN[i];
    for (int i = 0; i < 5; ++i) if (f[6].value == VIEWS[i]) g_fcfg.view = i;
    for (int i = 0; i < 3; ++i) if (f[7].value == AX[i]) g_fcfg.slice_axis = i;
    g_fcfg.slice_pos = geo::clampd(atof(f[8].value.c_str()), 0, 1);
    for (int i = 0; i < 3; ++i) if (f[9].value == HEAT[i]) g_fcfg.heat = i;
    g_fcfg.Thot = atof(f[10].value.c_str());
    g_fcfg.buoyancy = f[11].value == "yes";
    g_fcfg.cores = f[14].value == "both" || f[14].value == "vortex cores"; g_fcfg.particles = f[14].value == "both" || f[14].value == "particles";
    for (int i = 0; i < 4; ++i) if (f[13].value == WALLS[i]) g_fcfg.walls = i;
    for (int i = 0; i < 7; ++i) if (f[12].value == SURFS[i]) g_fcfg.surf = i;
    if (g_selftest) g_fcfg.cells = 24;
    flow_cfg_to_doc();
    flow_start();
}
static void flow_cycle_view() {  // the field: speed -> pressure -> temperature -> turbulence -> vorticity
    static const int ORDER[6] = {6, 1, 4, 5, 2, 0};
    int k = 0; for (int i = 0; i < 6; ++i) if (ORDER[i] == g_fcfg.show) k = i;
    for (int t = 1; t <= 6; ++t) {  // skip what this run does not have (temperature when isothermal, the liquid fraction without a bath)
        int cand = ORDER[(k + t) % 6];
        if (cand == 5 && g_flow && !g_flow->thermal) continue;
        if (cand == 6 && g_flow && !g_flow->fs) continue;
        g_fcfg.show = cand; break;
    }
    if (g_flow && g_fu.size() == 4 * g_flow->N) { flow_slice_texture(); flow_volume_build(); }
    status(std::string("flow: showing ") + FLOW_FIELD[g_fcfg.show] + (g_fcfg.view == 0 ? " through the whole fluid" : g_fcfg.view == 1 ? " on the slice" : " in the fluid and on the slice"));
    InvalidateRect(g_view, nullptr, FALSE);
}
static void flow_cycle_surface() {  // the parts' surfaces: friction -> pressure -> temperature -> turbulence -> liquid fraction -> nothing
    for (int t = 1; t <= 7; ++t) {
        int cand = (g_fcfg.surf + t) % 7;
        if ((cand == 3 || cand == 6) && g_flow && !g_flow->thermal) continue;
        if (cand == 5 && g_flow && !g_flow->fs) continue;
        g_fcfg.surf = cand; break;
    }
    flow_surface_build();
    status(std::string("parts' surfaces: ") + (g_fcfg.surf ? SURF_NAME[g_fcfg.surf] : "plain"));
    InvalidateRect(g_view, nullptr, FALSE);
}
static void flow_cycle_walls() {  // the cutaway's walls: the field -> pressure -> velocity -> pressure, velocity and the field
    g_fcfg.walls = (g_fcfg.walls + 1) % 4;
    if (g_fcfg.view != 4) g_fcfg.view = 4;
    if (g_flow && g_fu.size() == 4 * g_flow->N) flow_slice_texture();
    static const char* W[4] = {"the field shown", "pressure", "velocity (speed and arrows)", "pressure on the x walls, velocity on the y walls, the field on the floor and top"};
    status(std::string("box walls: ") + W[g_fcfg.walls]);
    InvalidateRect(g_view, nullptr, FALSE);
}
static void flow_cycle_mode() {  // the whole fluid -> a slice -> both
    g_fcfg.view = (g_fcfg.view + 1) % 5;
    if (g_flow && g_fu.size() == 4 * g_flow->N) { flow_slice_texture(); flow_volume_build(); }
    static const char* V[5] = {"the whole fluid as a coloured volume", "a slice", "the volume and a slice", "x, y and z slices through the volume", "a cutaway box: the walls you look into, and the volume"};
    status(std::string("flow view: ") + V[g_fcfg.view]);
    InvalidateRect(g_view, nullptr, FALSE);
}

// ---------------------------------------------------------------- examples
// a circular cylinder (10 mm) across a water channel at Re 100, wall to wall: the Karman vortex street
static void flow_demo_cylinder() {
    flow_stop(false);
    asm_new();
    if (!g_asm_on) return;
    js::Value fl = js::Value::array(); fl.push(cyl_f("cylinder", "add", 0.005, 0.005, V3()));  // a slice of an endless cylinder: slip walls at its ends
    asm_add_component(make_part("circular cylinder 10 mm (2D slice 5 mm)", fl, "body", js::Value::object(), "stainless_316"), "cylinder", "", nullptr);
    g_ac[0].fixed = true;
    js::Value env = js::Value::object();
    env["medium"] = "Fresh water"; env["temperature_C"] = 20.0; env["flow_m_s"] = 0.01; env["flow_dir"] = "+x"; env["altitude_m"] = 0.0;
    env["surface_z_m"] = 1.0; env["find_float"] = false; env["part_temperature_C"] = 20.0; env["heat_W"] = 0.0; env["lubrication"] = "dry"; env["gravity_m_s2"] = 9.81;
    g_doc["environment"] = env;
    g_fcfg = FlowCfg(); g_fcfg.mode = 0; g_fcfg.U = 0.01004; g_fcfg.dir = 0; g_fcfg.cells = 220; g_fcfg.span_walls = true; g_fcfg.span_axis = 2; g_fcfg.show = 0; g_fcfg.view = 2; g_fcfg.slice_axis = 2; g_fcfg.slice_pos = 0.5; g_fcfg.duration = 140;
    flow_cfg_to_doc();
    g_asm_sel = -1; asm_changed(false); g_asm_fitted = false; asm_request_meshes();
    status("vortex street: a 10 mm cylinder across water at 10 mm/s (Re 100). Flow simulation runs it (the Karman street, Strouhal about 0.165)");
}
// a mixing tank: a Rushton turbine (D = T/3, six blades) at 120 rpm in a baffled cylindrical tank of water
static void flow_demo_tank() {
    flow_stop(false);
    asm_new();
    if (!g_asm_on) return;
    const double T = 0.2, Ht = 0.2, wl = 0.004, Dimp = T / 3, C = T / 3;
    js::Value tank = js::Value::object(), tf = js::Value::array();
    tank["name"] = "baffled tank T200"; tank["units"] = "m"; tank["material"] = "stainless_316";
    tf.push(cyl_f("shell", "add", 0.5 * T + wl, Ht + wl, V3(0, 0, 0.5 * (Ht + wl) - wl)));
    tf.push(cyl_f("inside", "cut", 0.5 * T, Ht + 0.01, V3(0, 0, 0.5 * Ht + 0.005)));
    for (int k = 0; k < 4; ++k) {  // four baffles, T/10 wide, against the wall
        double a = k * geo::PI / 2, r = 0.5 * T - 0.5 * T / 10;
        js::Value bx = box_f(("baffle" + tostr(k + 1)).c_str(), "add", k % 2 ? V3(0.004, T / 10, Ht) : V3(T / 10, 0.004, Ht), V3(r * std::cos(a), r * std::sin(a), 0.5 * Ht));
        tf.push(bx);
    }
    tank["features"] = tf;
    asm_add_component(tank, "tank", "", nullptr);  // 0
    g_ac[0].fixed = true;
    js::Value imp = js::Value::object(), ifl = js::Value::array();
    imp["name"] = "Rushton turbine D67"; imp["units"] = "m"; imp["material"] = "stainless_316";
    ifl.push(cyl_f("hub", "add", 0.012, 0.024, V3()));
    ifl.push(cyl_f("disc", "add", 0.375 * Dimp, 0.003, V3()));
    for (int k = 0; k < 6; ++k) {
        double a = k * geo::PI / 3, r = 0.5 * Dimp - Dimp / 8;
        js::Value bl = box_f(("blade" + tostr(k + 1)).c_str(), "add", V3(Dimp / 4, 0.003, Dimp / 5), V3(r * std::cos(a), r * std::sin(a), 0));
        bl["rotate"] = v3json(V3(0, 0, a * 180 / geo::PI));
        bl["translate"] = v3json(V3(r * std::cos(a), r * std::sin(a), 0));
        ifl.push(bl);
    }
    ifl.push(cyl_f("bore", "cut", 0.0061, 0.05, V3()));
    imp["features"] = ifl;
    { Pose p; p.t = V3(0, 0, C); asm_add_component(imp, "impeller", "", &p); }                                 // 1
    { js::Value sf = js::Value::array(); sf.push(cyl_f("shaft", "add", 0.006, Ht - C + 0.02, V3()));
      Pose p; p.t = V3(0, 0, C + 0.5 * (Ht - C + 0.02) - 0.012); asm_add_component(make_part("stirrer shaft 12", sf, "shaft", js::Value::object(), "stainless_316"), "shaft", "", &p); }  // 2
    AMate lk; lk.type = MT_LOCK; lk.a.comp = 2; lk.b.comp = 1; lk.id = "lock1";
    { const Pose &Pa = g_ac[2].pose, &Pb = g_ac[1].pose; double Rt[9] = {Pa.R[0], Pa.R[3], Pa.R[6], Pa.R[1], Pa.R[4], Pa.R[7], Pa.R[2], Pa.R[5], Pa.R[8]}; mat_mul(Rt, Pb.R, lk.rel.R); lk.rel.t = Pa.inv(Pb.t); }
    g_am.push_back(lk);
    AMotor mo; mo.comp = 2; mo.p = V3(); mo.d = V3(0, 0, 1); mo.speed = 720; mo.power = 0; g_amot.push_back(mo);
    g_ms_T = 4;
    js::Value env = js::Value::object();
    env["medium"] = "Fresh water"; env["temperature_C"] = 20.0; env["flow_m_s"] = 0.0; env["flow_dir"] = "+x"; env["altitude_m"] = 0.0;
    env["surface_z_m"] = Ht; env["find_float"] = false; env["part_temperature_C"] = 20.0; env["heat_W"] = 0.0; env["lubrication"] = "dry"; env["gravity_m_s2"] = 9.81;
    g_doc["environment"] = env;
    g_ac[0].hidden = true;  // the tank still holds the water; hidden so the flow inside shows (Show it from the tree)
    g_fcfg = FlowCfg(); g_fcfg.mode = 1; g_fcfg.cells = 96; g_fcfg.margin = 0.0; g_fcfg.show = 0; g_fcfg.slice_axis = 1; g_fcfg.slice_pos = 0.5; g_fcfg.duration = 3;
    flow_cfg_to_doc();
    g_asm_sel = -1; asm_changed(false); g_asm_fitted = false; asm_request_meshes();
    status("mixing tank: a Rushton turbine (67 mm) at 120 rpm in a baffled 200 mm tank of water (Re 8900). Flow simulation stirs it from rest");
}
// gears turning in oil: the 20:40 gear pair at 300 rpm, hot (80 C) in 40 C engine oil filling a closed box round it
static void flow_demo_gears_oil() {
    flow_stop(false);
    asm_demo_gears();
    if (!g_asm_on || g_amot.empty()) return;
    g_amot[0].speed = 1800;  // 300 rpm
    js::Value env = js::Value::object();
    env["medium"] = "Engine oil SAE 30"; env["temperature_C"] = 40.0; env["flow_m_s"] = 0.0; env["flow_dir"] = "+x"; env["altitude_m"] = 0.0;
    env["surface_z_m"] = 1.0; env["find_float"] = false; env["part_temperature_C"] = 80.0; env["heat_W"] = 0.0; env["lubrication"] = "oil-lubricated"; env["gravity_m_s2"] = 9.81;
    g_doc["environment"] = env;
    g_fcfg = FlowCfg(); g_fcfg.mode = 1; g_fcfg.cells = 110; g_fcfg.margin = 0.12; g_fcfg.show = 5; g_fcfg.view = 0; g_fcfg.heat = 1; g_fcfg.Thot = 80; g_fcfg.duration = 1.0;
    flow_cfg_to_doc();
    asm_changed(false);
    status("gears in oil: a 20:40 spur pair at 300 rpm, hot (80 C), wholly in 40 C SAE 30 oil. Flow simulation shows the churning, the heat leaving the gears and the turbulence");
}
// a belt and a chain drive running wholly in water
static void flow_demo_belt_fluid() {
    flow_stop(false);
    asm_demo_belts();
    if (!g_asm_on) return;
    js::Value env = js::Value::object();
    env["medium"] = "Fresh water"; env["temperature_C"] = 20.0; env["flow_m_s"] = 0.0; env["flow_dir"] = "+x"; env["altitude_m"] = 0.0;
    env["surface_z_m"] = 1.0; env["find_float"] = false; env["part_temperature_C"] = 50.0; env["heat_W"] = 0.0; env["lubrication"] = "dry"; env["gravity_m_s2"] = 9.81;
    g_doc["environment"] = env;
    g_fcfg = FlowCfg(); g_fcfg.mode = 1; g_fcfg.cells = 128; g_fcfg.margin = 0.06; g_fcfg.show = 1; g_fcfg.view = 0; g_fcfg.heat = 1; g_fcfg.Thot = 50; g_fcfg.duration = 2.0;
    flow_cfg_to_doc();
    asm_changed(false);
    status("belt and chain in water: the V-belt, the roller chain and their wheels stir the water they run in. Flow simulation shows the speed, pressure, temperature and turbulence");
}
// a spur gear dipping in an oil bath, the classic churning-loss test: 36 teeth, module 3, 20 mm wide, on a horizontal
// axis, a quarter of its diameter in gear oil ISO VG 220 at 40 C with air above, at 600 rpm, in a closed box. Shows the
// liquid fraction, the splash, the friction on the teeth and a tooth's loss torque round the gear; the report sets the
// loss beside Changenet and Velex's correlation
static void flow_demo_splash() {
    flow_stop(false);
    asm_new();
    if (!g_asm_on) return;
    const double m = 0.003, bw = 0.02; const int z = 36;
    double r = 0.5 * m * z;
    Pose Ry; Ry.R[0] = 1; Ry.R[1] = 0; Ry.R[2] = 0; Ry.R[3] = 0; Ry.R[4] = 0; Ry.R[5] = 1; Ry.R[6] = 0; Ry.R[7] = -1; Ry.R[8] = 0;  // local z -> world y
    { Pose p = Ry; p.t = V3(0, 0, 0); asm_add_component(gear_part(m, z, bw, 0.03), "gear", "", &p); }
    g_ac[0].material = "steel_mild"; g_ac[0].fixed = false;
    AMotor mo; mo.comp = 0; mo.p = V3(); mo.d = V3(0, 0, 1); mo.speed = 3600; mo.power = 0; g_amot.push_back(mo);  // 600 rpm
    g_ms_T = 1;
    js::Value env = js::Value::object();
    env["medium"] = "Gear oil ISO VG 220"; env["temperature_C"] = 40.0; env["flow_m_s"] = 0.0; env["flow_dir"] = "+x"; env["altitude_m"] = 0.0;
    env["surface_z_m"] = -(r + m) + 0.5 * r; env["find_float"] = false; env["part_temperature_C"] = 40.0; env["heat_W"] = 0.0; env["lubrication"] = "oil-lubricated"; env["gravity_m_s2"] = 9.81;
    g_doc["environment"] = env;
    g_fcfg = FlowCfg(); g_fcfg.mode = 2; g_fcfg.cells = 110; g_fcfg.margin = 0.3; g_fcfg.show = 6; g_fcfg.view = 2; g_fcfg.slice_axis = 1; g_fcfg.slice_pos = 0.5;
    g_fcfg.heat = 0; g_fcfg.surf = 1; g_fcfg.duration = 0.2; g_fcfg.cores = false;
    flow_cfg_to_doc();
    g_asm_sel = -1; asm_changed(false); g_asm_fitted = false; asm_request_meshes();
    status("gear splash: a 36-tooth spur gear at 600 rpm dipping a quarter of its diameter in gear oil, air above. Flow simulation shows the oil fraction, the splash and a tooth's loss torque round the gear");
}
// headless: --flow-run starts the saved flow set-up once everything is meshed
static bool g_flow_pending = false;
static double g_flow_time_arg = 0;  // --flow-time: the simulated time, overriding the set-up
static int g_flow_cells_arg = 0;  // --flow-cells: the cells along the longest side, overriding the set-up
static void flow_run_tick() {
    if (!g_flow_pending) { KillTimer(g_main, 45); return; }
    if (g_flow_auto_arg && !g_asm_on) { if (!g_mesh.v.empty()) { if (!flow_ensure_assembly()) { g_flow_pending = false; return; } } return; }
    if (!g_asm_on || g_ac.empty()) return;
    for (const AComp& c : g_ac) if (!comp_ready(c)) return;
    g_flow_pending = false;
    flow_cfg_from_doc();
    if (g_flow_auto_arg) flow_cfg_auto();
    if (g_flow_time_arg > 0) g_fcfg.duration = g_flow_time_arg;
    if (g_flow_cells_arg > 0) g_fcfg.cells = g_flow_cells_arg;
    flow_start();
}
