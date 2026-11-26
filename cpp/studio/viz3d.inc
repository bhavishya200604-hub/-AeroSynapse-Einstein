// 3D results in the viewport: the full flow volume (Mach slices you can move, particles flowing all around the
// aircraft), streamlines with travelling light pulses, isosurfaces (vortex cores, sonic surface), the deformed
// structure and result colour maps. Everything a simulation writes into viz3d.json, and the solver's volume.bin.

static void turbo(double t, float& r, float& g, float& b);

struct Vol3 {
    std::vector<double> x, y, z;
    std::vector<float> m, u, v, w, s;
    size_t nx = 0, ny = 0, nz = 0;
    size_t id(size_t i, size_t j, size_t k) const { return (k * ny + j) * nx + i; }
};
static Vol3 g_vol;
static bool g_vol_ok = false;
static int g_slice_axis = 1;
static bool g_slice_on[3] = {true, true, false};
static size_t g_slice_i[3] = {0, 0, 0};
static double g_crop_lo[3], g_crop_hi[3], g_vuinf = 1;
static float g_vm_lo = 0, g_vm_hi = 1, g_vminf = 0.5f;
struct P3 { double p[3]; float tr[14][3]; int n = 0; };
static std::vector<P3> g_p3;

struct VLine { std::vector<float> p, s; float r = 1, g = 1, b = 1; bool sc = false; };
struct VMesh { std::vector<float> v, n; std::vector<unsigned> t; float c[4] = {1, 1, 1, 0.4f}; std::string label; };
static std::vector<VLine> g_vlines;
static std::vector<VMesh> g_vmeshes;
static float g_vl_lo = 0, g_vl_hi = 1;
static std::string g_viz_title;
static bool g_show_lines = true, g_show_meshes = true, g_cmap_turbo = false;
static double g_phase = 0;
static bool g_vol_half = false;  // a half-model volume (y >= 0): mirrored for display and sampling
struct SLine { std::vector<float> p, m, t; };  // points, Mach, flow time (L / U units) along each streamline
static std::vector<SLine> g_sl;
static bool g_sl_on = false, g_sweep_on = false;
static int g_sweep_dir = 1;
static std::vector<double> g_def_y, g_def_dz;
static double g_def_scale = 0, g_def_y0 = 0;
static unsigned g_rng3 = 777;
static double frand3() { g_rng3 = g_rng3 * 1664525u + 1013904223u; return (g_rng3 >> 8) / 16777216.0; }

// ---------------------------------------------------------------- the flow volume
static bool vol_load(const std::string& dir) {
    std::string hj;
    try { hj = js::read_file(dir + "\\volume.json"); } catch (...) { return false; }
    js::Value h = js::parse(hj);
    Vol3 V;
    V.x = h.nums("x"); V.y = h.nums("y"); V.z = h.nums("z");
    V.nx = V.x.size(); V.ny = V.y.size(); V.nz = V.z.size();
    size_t n = V.nx * V.ny * V.nz;
    if (n == 0) return false;
    FILE* f = fopen((dir + "\\volume.bin").c_str(), "rb");
    if (!f) return false;
    std::vector<float> buf(n);
    std::vector<float> rho(n), p(n);
    std::vector<float>* dst[6] = {&rho, &V.u, &V.v, &V.w, &p, &V.s};
    for (int q = 0; q < 6; ++q) {
        dst[q]->resize(n);
        if (fread(dst[q]->data(), sizeof(float), n, f) != n) { fclose(f); return false; }
    }
    fclose(f);
    V.m.resize(n);
    for (size_t c = 0; c < n; ++c) {
        if (V.s[c] > 0.5f || rho[c] <= 0 || p[c] <= 0) { V.m[c] = 0; continue; }
        V.m[c] = std::sqrt(V.u[c] * V.u[c] + V.v[c] * V.v[c] + V.w[c] * V.w[c]) / std::sqrt(1.4f * p[c] / rho[c]);
    }
    g_vol = V;
    g_vol_half = V.y.front() > -1e-9;
    g_sl.clear();
    // crop box around the model, freestream from the inflow face, Mach range from the 2nd-98th percentile
    double L = std::max(1.0, g_props.hi.x - g_props.lo.x), S = std::max(1.0, g_props.hi.y - g_props.lo.y), Hh = std::max(0.5, g_props.hi.z - g_props.lo.z);
    g_crop_lo[0] = g_props.lo.x - 0.3 * L; g_crop_hi[0] = g_props.hi.x + 0.7 * L;
    g_crop_lo[1] = g_props.lo.y - 0.35 * S; g_crop_hi[1] = g_props.hi.y + 0.35 * S;
    g_crop_lo[2] = g_props.lo.z - 0.8 * Hh - 0.15 * L; g_crop_hi[2] = g_props.hi.z + 0.8 * Hh + 0.15 * L;
    double mi = 0, us = 0; int cnt = 0;
    for (size_t k = 0; k < V.nz; ++k) for (size_t j = 0; j < V.ny; ++j) { size_t c = V.id(0, j, k); if (V.s[c] < 0.5f) { mi += V.m[c]; us += std::sqrt(V.u[c] * V.u[c] + V.v[c] * V.v[c] + V.w[c] * V.w[c]); ++cnt; } }
    g_vminf = cnt ? (float)(mi / cnt) : 0.5f; g_vuinf = cnt ? us / cnt : 1.0;
    std::vector<float> ms;
    for (size_t k = 0; k < V.nz; ++k) for (size_t j = 0; j < V.ny; ++j) for (size_t i = 0; i < V.nx; ++i) {
        if (V.x[i] < g_crop_lo[0] || V.x[i] > g_crop_hi[0] || V.y[j] < g_crop_lo[1] || V.y[j] > g_crop_hi[1] || V.z[k] < g_crop_lo[2] || V.z[k] > g_crop_hi[2]) continue;
        size_t c = V.id(i, j, k);
        if (V.s[c] < 0.5f) ms.push_back(V.m[c]);
    }
    if (ms.size() > 20) { std::sort(ms.begin(), ms.end()); g_vm_lo = ms[ms.size() * 2 / 100]; g_vm_hi = ms[ms.size() * 99 / 100]; }
    if (g_vm_hi <= g_vm_lo + 1e-3f) g_vm_hi = g_vm_lo + 0.05f;
    auto nearest = [](const std::vector<double>& a, double v) { return (size_t)(std::min_element(a.begin(), a.end(), [v](double p_, double q_) { return std::fabs(p_ - v) < std::fabs(q_ - v); }) - a.begin()); };
    g_slice_i[0] = nearest(V.x, 0.5 * (g_props.lo.x + g_props.hi.x));
    g_slice_i[1] = nearest(V.y, 0.5 * (g_props.lo.y + g_props.hi.y));
    g_slice_i[2] = nearest(V.z, 0.5 * (g_props.lo.z + g_props.hi.z));
    g_p3.assign(1400, P3());
    for (auto& q : g_p3) { q.p[0] = g_crop_lo[0] + frand3() * (g_crop_hi[0] - g_crop_lo[0]); q.p[1] = g_props.lo.y - 0.2 * S + frand3() * 1.4 * S; q.p[2] = g_props.lo.z - 0.4 * Hh + frand3() * 1.8 * Hh; q.n = 0; }
    g_vol_ok = true;
    return true;
}

static bool vol_sample(const double* p0, double* vel, double& m, bool& solid) {
    const Vol3& V = g_vol;
    double p[3] = {p0[0], p0[1], p0[2]};
    bool flip = g_vol_half && p[1] < 0;  // the other half of a half model: its mirror image
    if (g_vol_half) { p[1] = std::fabs(p[1]); if (!V.y.empty() && p[1] <= V.y.front()) p[1] = V.y.front() + 1e-9; }
    size_t ix[3]; double fr[3];
    const std::vector<double>* ax[3] = {&V.x, &V.y, &V.z};
    for (int a = 0; a < 3; ++a) {
        const std::vector<double>& c = *ax[a];
        if (p[a] <= c.front() || p[a] >= c.back()) return false;
        size_t i = (size_t)(std::upper_bound(c.begin(), c.end(), p[a]) - c.begin()) - 1;
        ix[a] = i; fr[a] = (p[a] - c[i]) / (c[i + 1] - c[i]);
    }
    double acc[5] = {0, 0, 0, 0, 0};
    for (int c = 0; c < 8; ++c) {
        size_t i = ix[0] + (c & 1), j = ix[1] + ((c >> 1) & 1), k = ix[2] + ((c >> 2) & 1);
        double wgt = ((c & 1) ? fr[0] : 1 - fr[0]) * (((c >> 1) & 1) ? fr[1] : 1 - fr[1]) * (((c >> 2) & 1) ? fr[2] : 1 - fr[2]);
        size_t id = V.id(i, j, k);
        acc[0] += wgt * V.u[id]; acc[1] += wgt * V.v[id]; acc[2] += wgt * V.w[id]; acc[3] += wgt * V.m[id]; acc[4] += wgt * V.s[id];
    }
    vel[0] = acc[0]; vel[1] = flip ? -acc[1] : acc[1]; vel[2] = acc[2]; m = acc[3]; solid = acc[4] > 0.4;
    return true;
}

// ---------------------------------------------------------------- 3D streamlines from the volume
// seeds: a rake upstream of the aircraft and points just off its upstream surfaces; integrated with RK2 along the
// velocity until they leave the crop box or touch the body; the flow time along each line drives the pulses
static void vol_streamlines() {
    g_sl.clear();
    if (!g_vol_ok) return;
    double L = std::max(1.0, g_props.hi.x - g_props.lo.x), S = std::max(1.0, g_props.hi.y - g_props.lo.y), Hh = std::max(0.5, g_props.hi.z - g_props.lo.z);
    std::vector<V3> seeds;
    double x0 = g_props.lo.x - 0.1 * L;
    double ylo = g_vol_half ? -0.95 * g_props.hi.y : g_props.lo.y + 0.05 * S, yhi = g_vol_half ? 0.95 * g_props.hi.y : g_props.hi.y - 0.05 * S;
    for (int j = 0; j < 11; ++j) for (int k = 0; k < 4; ++k)  // a rake across the span, from just below to just above the aircraft
        seeds.push_back(V3(x0, ylo + (yhi - ylo) * j / 10.0, g_props.lo.z - 0.05 * Hh + 1.1 * Hh * k / 3.0));
    unsigned rng = 12345;
    auto rnd = [&]() { rng = rng * 1664525u + 1013904223u; return (rng >> 8) / 16777216.0; };
    for (int q = 0; q < 6000 && seeds.size() < 300 && !g_mesh.v.empty(); ++q) {  // just off the front half of the surface
        size_t i = (size_t)(rnd() * g_mesh.v.size()) % g_mesh.v.size();
        const V3& p = g_mesh.v[i]; const V3& n = g_mesh.n[i];
        if (p.x > g_props.lo.x + 0.55 * L || n.x > 0.2) continue;
        seeds.push_back(p + n * (0.02 * L));
    }
    double ds = 0.006 * L;
    for (auto& s0 : seeds) {
        SLine sl; double p[3] = {s0.x, s0.y, s0.z}, t = 0;
        for (int it = 0; it < 700; ++it) {
            double v1[3], m1; bool so1;
            if (!vol_sample(p, v1, m1, so1) || so1) break;
            double s1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
            if (s1 < 1e-9) break;
            sl.p.push_back((float)p[0]); sl.p.push_back((float)p[1]); sl.p.push_back((float)p[2]); sl.m.push_back((float)m1); sl.t.push_back((float)t);
            double mid[3]; for (int a = 0; a < 3; ++a) mid[a] = p[a] + 0.5 * ds * v1[a] / s1;
            double v2[3], m2; bool so2;
            if (!vol_sample(mid, v2, m2, so2) || so2) break;
            double s2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
            if (s2 < 1e-9) break;
            for (int a = 0; a < 3; ++a) p[a] += ds * v2[a] / s2;
            t += ds / s2 * g_vuinf / L;
            if (p[0] > g_crop_hi[0] || p[1] < -g_crop_hi[1] - S || p[1] > g_crop_hi[1] || p[2] < g_crop_lo[2] || p[2] > g_crop_hi[2]) break;
        }
        if (sl.m.size() > 8) g_sl.push_back(sl);
    }
}
static void sl_render() {
    if (!g_sl_on || g_sl.empty()) return;
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glEnable(GL_LINE_SMOOTH);
    glLineWidth(1.6f);
    for (const SLine& s : g_sl) {
        glBegin(GL_LINE_STRIP);
        for (size_t i = 0; i < s.m.size(); ++i) { float r, g, b; turbo((s.m[i] - g_vm_lo) / (g_vm_hi - g_vm_lo), r, g, b); glColor4f(r, g, b, 0.8f); glVertex3fv(&s.p[3 * i]); }
        glEnd();
    }
    // pulses: bright dashes that travel along every line at the local flow speed
    const float period = 0.55f, width = 0.07f, now = (float)(g_phase * 1.9);
    glLineWidth(3.6f);
    glBegin(GL_LINES);
    for (const SLine& s : g_sl)
        for (size_t i = 0; i + 1 < s.m.size(); ++i) {
            float f = std::fmod(s.t[i] - now, period); if (f < 0) f += period;
            if (f > width) continue;
            float k = 1.0f - f / width;
            float r, g, b; turbo((s.m[i] - g_vm_lo) / (g_vm_hi - g_vm_lo), r, g, b);
            glColor4f(0.55f * r + 0.45f, 0.55f * g + 0.45f, 0.55f * b + 0.45f, 0.95f * k);
            glVertex3fv(&s.p[3 * i]); glVertex3fv(&s.p[3 * i + 3]);
        }
    glEnd();
    glLineWidth(1.0f); glDisable(GL_LINE_SMOOTH); glDisable(GL_BLEND); glEnable(GL_LIGHTING);
}
static void sweep_tick() {  // the active slice glides through the volume and back
    if (!g_sweep_on || !g_vol_ok) return;
    static int sub = 0; if (++sub % 2) return;
    const std::vector<double>* ax[3] = {&g_vol.x, &g_vol.y, &g_vol.z};
    const std::vector<double>& c = *ax[g_slice_axis];
    long i = (long)g_slice_i[g_slice_axis] + g_sweep_dir;
    double lo = g_slice_axis == 1 && g_vol_half ? c.front() : g_crop_lo[g_slice_axis], hi = g_crop_hi[g_slice_axis];
    if (i <= 0 || i >= (long)c.size() - 1 || c[(size_t)i] < lo || c[(size_t)i] > hi) { g_sweep_dir = -g_sweep_dir; i = (long)g_slice_i[g_slice_axis] + g_sweep_dir; }
    g_slice_i[g_slice_axis] = (size_t)std::max(0L, std::min((long)c.size() - 1, i));
}


static void vol_step() {
    double L = std::max(1.0, g_props.hi.x - g_props.lo.x), S = std::max(1.0, g_props.hi.y - g_props.lo.y), Hh = std::max(0.5, g_props.hi.z - g_props.lo.z);
    double dt = 0.0065 * L / std::max(g_vuinf, 1e-9);
    for (auto& q : g_p3) {
        double vel[3], m; bool solid;
        bool ok = vol_sample(q.p, vel, m, solid);
        if (!ok || solid || q.p[0] > g_crop_hi[0] || frand3() < 0.003) {
            q.p[0] = g_crop_lo[0] + frand3() * 0.12 * L; q.p[1] = (g_vol_half ? -g_props.hi.y : g_props.lo.y) - 0.2 * S + frand3() * (g_vol_half ? 2 * g_props.hi.y + 0.4 * S : 1.4 * S); q.p[2] = g_props.lo.z - 0.4 * Hh + frand3() * 1.8 * Hh; q.n = 0;
            continue;
        }
        for (int k = std::min(q.n, 13); k > 0; --k) for (int a = 0; a < 3; ++a) q.tr[k][a] = q.tr[k - 1][a];
        for (int a = 0; a < 3; ++a) q.tr[0][a] = (float)q.p[a];
        q.n = std::min(q.n + 1, 14);
        for (int a = 0; a < 3; ++a) q.p[a] += vel[a] * dt;
    }
}

static void vol_colour(float mach, float alpha_base, float& r, float& g, float& b, float& a) {
    turbo((mach - g_vm_lo) / (g_vm_hi - g_vm_lo), r, g, b);
    float dev = (float)std::min(1.0, std::fabs(mach - g_vminf) / std::max(0.02, 0.3 * (g_vm_hi - g_vm_lo)));
    a = alpha_base * (0.06f + 0.94f * dev);
}

static void vol_render(bool particles) {
    const Vol3& V = g_vol;
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
    const std::vector<double>* ax[3] = {&V.x, &V.y, &V.z};
    for (int mirror = 0; mirror < (g_vol_half ? 2 : 1); ++mirror) {
    if (mirror) { glPushMatrix(); glScaled(1, -1, 1); }
    for (int a = 0; a < 3; ++a) {
        if (!g_slice_on[a] || (mirror && a == 1)) continue;
        int ua = (a + 1) % 3, va = (a + 2) % 3;
        const std::vector<double>&U = *ax[ua], &W = *ax[va];
        size_t si = std::min(g_slice_i[a], (*ax[a]).size() - 1);
        float base = a == g_slice_axis ? 0.7f : 0.42f;
        for (size_t jv = 0; jv + 1 < W.size(); ++jv) {
            if (W[jv + 1] < g_crop_lo[va] || W[jv] > g_crop_hi[va]) continue;
            glBegin(GL_QUAD_STRIP);
            for (size_t iu = 0; iu < U.size(); ++iu) {
                if (U[iu] < g_crop_lo[ua] || U[iu] > g_crop_hi[ua]) continue;
                for (size_t jj = jv; jj <= jv + 1; ++jj) {
                    size_t idx[3]; idx[a] = si; idx[ua] = iu; idx[va] = jj;
                    size_t c = V.id(idx[0], idx[1], idx[2]);
                    float r, g, b, al; vol_colour(V.m[c], base, r, g, b, al);
                    if (V.s[c] > 0.5f) al = 0;
                    glColor4f(r, g, b, al);
                    double p[3]; p[a] = (*ax[a])[si]; p[ua] = U[iu]; p[va] = W[jj];
                    glVertex3dv(p);
                }
            }
            glEnd();
        }
        // the slice's outline, in the scheme's pink when it is the active one
        glLineWidth(a == g_slice_axis ? 2.0f : 1.0f);
        if (g_calm) glColor4f(0.62f, 0.7f, 0.8f, a == g_slice_axis ? 0.75f : 0.4f);
        else glColor4f(a == g_slice_axis ? 0.93f : 0.5f, a == g_slice_axis ? 0.29f : 0.7f, a == g_slice_axis ? 0.6f : 0.9f, 0.9f);
        glBegin(GL_LINE_LOOP);
        double c4[4][2] = {{g_crop_lo[ua], g_crop_lo[va]}, {g_crop_hi[ua], g_crop_lo[va]}, {g_crop_hi[ua], g_crop_hi[va]}, {g_crop_lo[ua], g_crop_hi[va]}};
        for (auto& q : c4) { double p[3]; p[a] = (*ax[a])[si]; p[ua] = q[0]; p[va] = q[1]; glVertex3dv(p); }
        glEnd();
        glLineWidth(1.0f);
    }
    if (mirror) glPopMatrix();
    }
    if (particles) {
        glLineWidth(g_calm ? 1.0f : 1.5f);
        glBegin(GL_LINES);
        for (auto& q : g_p3) {
            for (int k = 0; k + 1 < q.n; ++k) {
                double pp[3] = {q.tr[k][0], q.tr[k][1], q.tr[k][2]}, vel[3], m; bool s;
                float r = 0.8f, g = 0.95f, b = 1.0f;
                if (k == 0 && vol_sample(pp, vel, m, s)) { turbo((m - g_vm_lo) / (g_vm_hi - g_vm_lo), r, g, b); r = 0.4f + 0.6f * r; g = 0.4f + 0.6f * g; b = 0.4f + 0.6f * b; }
                float al = 1.0f - (float)k / 14.0f;
                float amax = g_calm ? 0.5f : 0.8f;
                glColor4f(r, g, b, amax * al); glVertex3fv(q.tr[k]);
                glColor4f(r, g, b, amax * al * 0.6f); glVertex3fv(q.tr[k + 1]);
            }
        }
        glEnd();
        glPointSize(3.6f); glEnable(GL_POINT_SMOOTH);  // heads in the local Mach colour
        glBegin(GL_POINTS);
        for (auto& q : g_p3) if (q.n) { double pp[3] = {q.tr[0][0], q.tr[0][1], q.tr[0][2]}, vel[3], m; bool s; float r = 1, g = 1, b = 1;
            if (vol_sample(pp, vel, m, s)) turbo((m - g_vm_lo) / (g_vm_hi - g_vm_lo), r, g, b);
            glColor4f(0.25f + 0.75f * r, 0.25f + 0.75f * g, 0.25f + 0.75f * b, 0.95f); glVertex3fv(q.tr[0]); }
        glEnd(); glDisable(GL_POINT_SMOOTH);
        glLineWidth(1.0f);
    }
    glDepthMask(GL_TRUE); glDisable(GL_BLEND);
    sl_render();
}

static void slice_move(int d) {
    if (!g_vol_ok) return;
    const std::vector<double>* ax[3] = {&g_vol.x, &g_vol.y, &g_vol.z};
    long i = (long)g_slice_i[g_slice_axis] + d;
    g_slice_i[g_slice_axis] = (size_t)std::max(0L, std::min((long)(*ax[g_slice_axis]).size() - 1, i));
    char b[96]; snprintf(b, sizeof b, "slice normal to %c at %.2f m (Ctrl + wheel moves it)", "xyz"[g_slice_axis], (*ax[g_slice_axis])[g_slice_i[g_slice_axis]]);
    status(b);
}

// ---------------------------------------------------------------- viz3d.json overlays
static bool load_obj_mesh(const std::string& path, VMesh& M) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        float a, b, c; unsigned i, j, k;
        if (line[0] == 'v' && line[1] == ' ' && sscanf(line + 2, "%f %f %f", &a, &b, &c) == 3) { M.v.push_back(a); M.v.push_back(b); M.v.push_back(c); }
        else if (line[0] == 'f' && sscanf(line + 2, "%u %u %u", &i, &j, &k) == 3) { M.t.push_back(i - 1); M.t.push_back(j - 1); M.t.push_back(k - 1); }
    }
    fclose(f);
    M.n.assign(M.v.size(), 0.0f);
    for (size_t q = 0; q + 2 < M.t.size(); q += 3) {
        const float* A = &M.v[3 * M.t[q]], *B = &M.v[3 * M.t[q + 1]], *Cc = &M.v[3 * M.t[q + 2]];
        float e1[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]}, e2[3] = {Cc[0] - A[0], Cc[1] - A[1], Cc[2] - A[2]};
        float nn[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        for (int r = 0; r < 3; ++r) for (int a = 0; a < 3; ++a) M.n[3 * M.t[q + r] + a] += nn[a];
    }
    return !M.t.empty();
}

static double interp1(const std::vector<double>& x, const std::vector<double>& y, double v) {
    if (x.empty()) return 0;
    if (v <= x.front()) return y.front();
    if (v >= x.back()) return y.back();
    size_t i = (size_t)(std::upper_bound(x.begin(), x.end(), v) - x.begin()) - 1;
    double t = (v - x[i]) / (x[i + 1] - x[i]);
    return y[i] + t * (y[i + 1] - y[i]);
}

static void viz_clear() {
    g_vlines.clear(); g_vmeshes.clear(); g_def_y.clear(); g_def_dz.clear(); g_def_scale = 0; g_viz_title.clear(); g_cmap_turbo = false;
}

static void viz_colour(const js::Value& c) {
    if (g_mesh.v.empty()) return;
    std::string kind = c.str("kind", "");
    std::vector<double> val(g_mesh.v.size(), std::nan(""));
    if (kind == "span") {
        std::vector<double> ys = c.nums("y"), vs = c.nums("value");
        double y0 = c.num("y0", 0);
        for (size_t i = 0; i < g_mesh.v.size(); ++i) { double ay = std::fabs(g_mesh.v[i].y); if (ay > y0) val[i] = interp1(ys, vs, ay); }
    } else if (kind == "zones") {
        if (const js::Value* zs = c.find("zones"))
            for (size_t i = 0; i < g_mesh.v.size(); ++i) {
                const V3& p = g_mesh.v[i];
                for (size_t z = 0; z < zs->size(); ++z) {
                    std::vector<double> bx = zs->at(z).nums("box");
                    if (bx.size() == 6 && p.x >= bx[0] && p.x <= bx[1] && p.y >= bx[2] && p.y <= bx[3] && p.z >= bx[4] && p.z <= bx[5]) { val[i] = zs->at(z).num("value", 0); break; }
                }
            }
    } else return;
    double lo = 1e300, hi = -1e300;
    for (double v : val) if (v == v) { lo = std::min(lo, v); hi = std::max(hi, v); }
    if (lo > hi) return;
    if (hi <= lo) hi = lo + 1e-6;
    g_colors.clear();
    for (double v : val) {
        float r = 0.55f, g = 0.58f, b = 0.63f;
        if (v == v) turbo((v - lo) / (hi - lo), r, g, b);
        g_colors.push_back(r); g_colors.push_back(g); g_colors.push_back(b);
    }
    g_rmin = lo; g_rmax = hi; g_result_label = c.str("label", "result"); g_cmap_turbo = true;
}

static bool viz_load(const std::string& dir) {
    std::string t;
    try { t = js::read_file(dir + "\\viz3d.json"); } catch (...) { return false; }
    js::Value v;
    try { v = js::parse(t); } catch (...) { return false; }
    viz_clear();
    g_viz_title = v.str("title", "");
    if (const js::Value* ls = v.find("lines"))
        for (size_t i = 0; i < ls->size(); ++i) {
            const js::Value& L = ls->at(i);
            VLine vl;
            if (const js::Value* pts = L.find("pts")) for (size_t k = 0; k < pts->size(); ++k) { const js::Value& q = pts->at(k); if (q.size() >= 3) { vl.p.push_back((float)q.at(0).n); vl.p.push_back((float)q.at(1).n); vl.p.push_back((float)q.at(2).n); } }
            std::vector<double> s = L.nums("s");
            if (!s.empty()) { vl.sc = true; for (double x : s) vl.s.push_back((float)x); }
            std::vector<double> col = L.nums("color");
            if (col.size() >= 3) { vl.r = (float)col[0]; vl.g = (float)col[1]; vl.b = (float)col[2]; }
            if (vl.p.size() >= 6) g_vlines.push_back(vl);
        }
    std::vector<double> lr = v.nums("line_range");
    if (lr.size() == 2) { g_vl_lo = (float)lr[0]; g_vl_hi = (float)std::max(lr[1], lr[0] + 1e-3); }
    if (const js::Value* ms = v.find("meshes"))
        for (size_t i = 0; i < ms->size(); ++i) {
            VMesh M;
            std::vector<double> col = ms->at(i).nums("color");
            for (size_t a = 0; a < 4 && a < col.size(); ++a) M.c[a] = (float)col[a];
            M.label = ms->at(i).str("label", "");
            if (load_obj_mesh(dir + "\\" + ms->at(i).str("obj", ""), M)) g_vmeshes.push_back(M);
        }
    if (const js::Value* d = v.find("deform")) { g_def_y = d->nums("y"); g_def_dz = d->nums("dz"); g_def_scale = d->num("scale", 1); g_def_y0 = d->num("y0", 0); }
    if (const js::Value* c = v.find("colour")) viz_colour(*c);
    if (!g_vlines.empty()) SetTimer(g_main, 7, 33, nullptr);
    InvalidateRect(g_view, nullptr, FALSE);
    return true;
}

static void viz_render() {
    if (g_show_lines && !g_vlines.empty()) {
        glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
        glLineWidth(g_calm ? 1.0f : 1.3f);
        size_t stride = g_vlines.size() > 200 ? 2 : 1;
        for (size_t li = 0; li < g_vlines.size(); li += stride) {
            const VLine& L = g_vlines[li];
            size_t n = L.p.size() / 3;
            glBegin(GL_LINE_STRIP);
            for (size_t i = 0; i < n; ++i) {
                float r = L.r, g = L.g, b = L.b;
                if (L.sc && i < L.s.size()) { float t = (L.s[i] - g_vl_lo) / (g_vl_hi - g_vl_lo); t = std::max(0.0f, std::min(1.0f, t)); r = 0.15f + 0.85f * t; g = 0.85f - 0.55f * t; b = 1.0f; }
                double ph = std::fmod(g_phase - i * 0.018 + li * 0.137, 1.0); if (ph < 0) ph += 1;
                float pulse = g_calm ? 0.0f : (float)std::max(0.0, 1.0 - ph * 7.0);
                if (g_calm) { float gy = 0.3f * r + 0.59f * g + 0.11f * b; r = 0.55f * r + 0.45f * gy; g = 0.55f * g + 0.45f * gy; b = 0.55f * b + 0.45f * gy; }
                glColor4f(std::min(1.0f, r + 0.6f * pulse), std::min(1.0f, g + 0.6f * pulse), std::min(1.0f, b + 0.6f * pulse),
                          g_calm ? (L.sc ? 0.32f : 0.6f) : (L.sc ? 0.13f : 0.5f) + (L.sc ? 0.62f : 0.5f) * pulse);
                glVertex3fv(&L.p[3 * i]);
            }
            glEnd();
        }
        glLineWidth(1.0f);
        glDepthMask(GL_TRUE); glDisable(GL_BLEND);
    }
    if (g_show_meshes && !g_vmeshes.empty()) {
        glEnable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
        for (const VMesh& M : g_vmeshes) {
            glColor4f(M.c[0], M.c[1], M.c[2], M.c[3]);
            glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY);
            glVertexPointer(3, GL_FLOAT, 0, M.v.data()); glNormalPointer(GL_FLOAT, 0, M.n.data());
            glDrawElements(GL_TRIANGLES, (GLsizei)M.t.size(), GL_UNSIGNED_INT, M.t.data());
            glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_VERTEX_ARRAY);
        }
        glDepthMask(GL_TRUE); glDisable(GL_BLEND);
    }
}

// the model as drawn: deformed by a structural result when one is shown
static void draw_mesh(const geo::Mesh& M, bool colours);
static void draw_model() {
    if (g_def_scale > 0 && g_def_y.size() >= 2 && !g_mesh.v.empty()) {
        static geo::Mesh D;
        D = g_mesh;
        for (auto& v : D.v) { double ay = std::fabs(v.y); if (ay > g_def_y0) v.z += g_def_scale * interp1(g_def_y, g_def_dz, ay); }
        draw_mesh(D, true);
    } else draw_mesh(g_mesh, true);
}
