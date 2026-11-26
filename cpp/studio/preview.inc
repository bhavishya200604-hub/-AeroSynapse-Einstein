// Previews of simulation and optimisation setups in the 3D view, while their form is open: what a run will do is
// visible before it starts, and changes as the fields change.
//   3D flow: the computational domain, the refined core around the aircraft, the free-stream direction (angle of
//            attack and sideslip), the probes, and the grid size the solver will build (same rules as aeroflow).
//   Wing optimisation: the main wing's planform now, and at the low and high end of every variable that is varied.

static bool setup_num(const std::string& key, double& out) {
    for (auto& r : g_srows) if (r.key == key) { js::Value v; std::string e; if (srow_get(r, v, e) && v.kind == js::Value::Num) { out = v.n; return true; } return false; }
    return false;
}
static bool setup_bool(const std::string& key, bool dflt) {
    for (auto& r : g_srows) if (r.key == key) { js::Value v; std::string e; if (srow_get(r, v, e)) return v.kind == js::Value::Bool ? v.b : dflt; }
    return dflt;
}
static std::string setup_str(const std::string& key, const std::string& dflt) {
    for (auto& r : g_srows) if (r.key == key) { js::Value v; std::string e; if (srow_get(r, v, e) && v.kind == js::Value::Str) return v.s; }
    return dflt;
}
// cells along one stretched axis, as aeroflow's Axis::stretched builds it: uniform h0 over the core, then cells
// growing by `ratio` up to hmax out to each end
static int axis_cells(double a0, double c0, double c1, double a1, double h0, double ratio, double hmax) {
    int m = std::max(1, (int)std::ceil(std::max(0.0, c1 - c0) / h0)), n = m;
    double hc = (c1 - c0) / m;  // the core's actual spacing, where the growth starts
    for (double D : {c0 - a0, a1 - c1}) { double s = 0, h = hc; while (s < D - 1e-9) { h = std::min(h * ratio, hmax); s += h; ++n; if (n > 100000) break; } }
    return n;
}
struct FlowPreview { bool ok = false; V3 dlo, dhi, clo, chi; double h0 = 0, cells = 0; bool half = false; V3 dir; std::vector<V3> probes; };
static FlowPreview flow_preview() {
    FlowPreview P;
    if (g_mesh.v.empty()) return P;
    V3 lo = g_props.lo, hi = g_props.hi;
    double L = std::max(hi.x - lo.x, 1e-3);
    double mach = 0.85, alpha = 2, beta = 0, cpl = 30, stretch = 1.15;
    setup_num("mach", mach); setup_num("alpha", alpha); setup_num("beta", beta); setup_num("stretch", stretch);
    std::string q = setup_str("quality", "draft");
    if (q == "custom") setup_num("cells_per_length", cpl);
    else cpl = q == "live" ? 18 : q == "draft" ? 30 : q == "standard" ? 48 : 70;
    double d[4] = {0, 0, 0, 0};
    setup_num("domain_up", d[0]); setup_num("domain_down", d[1]); setup_num("domain_side", d[2]); setup_num("domain_vert", d[3]);
    if (!(d[0] > 0 && d[1] > 0 && d[2] > 0 && d[3] > 0)) {
        bool sup = mach > 1.05;
        d[0] = sup ? 0.4 : 1.5; d[1] = sup ? 0.6 : 2.5; d[2] = sup ? 0.8 : 1.5; d[3] = sup ? 0.8 : 1.5;
    }
    bool full = setup_bool("full_model", true) || std::fabs(beta) > 0;
    double pad = 0.06 * L, h0 = L / cpl, hmax = 0.5 * L, span = std::max(hi.y, -lo.y);
    P.half = !full;
    P.clo = V3(lo.x - pad, full ? -span - pad : 0, lo.z - pad); P.chi = V3(hi.x + pad, span + pad, hi.z + pad);
    P.dlo = V3(lo.x - d[0] * L, full ? -span - d[2] * L : 0, lo.z - d[3] * L); P.dhi = V3(hi.x + d[1] * L, span + d[2] * L, hi.z + d[3] * L);
    double nx = axis_cells(P.dlo.x, P.clo.x, P.chi.x, P.dhi.x, h0, stretch, hmax);
    double ny = full ? axis_cells(P.dlo.y, P.clo.y, P.chi.y, P.dhi.y, h0, stretch, hmax) : axis_cells(0, 0, P.chi.y, P.dhi.y, h0, stretch, hmax);
    double nz = axis_cells(P.dlo.z, P.clo.z, P.chi.z, P.dhi.z, h0, stretch, hmax);
    P.cells = nx * ny * nz; P.h0 = h0;
    double a = alpha * geo::PI / 180, b = beta * geo::PI / 180;
    P.dir = V3(std::cos(a) * std::cos(b), -std::sin(b), std::sin(a) * std::cos(b));
    for (const char* k : {"probe_1", "probe_2", "probe_3"}) {
        std::string t = setup_str(k, "");
        double x, y, z; if (sscanf(t.c_str(), "%lf , %lf , %lf", &x, &y, &z) == 3 || sscanf(t.c_str(), "%lf %lf %lf", &x, &y, &z) == 3) P.probes.push_back(V3(x, y, z));
    }
    P.ok = true;
    return P;
}
static void box_edges(const V3& lo, const V3& hi) {
    auto V = [&](int k) { return V3((k & 1) ? hi.x : lo.x, (k & 2) ? hi.y : lo.y, (k & 4) ? hi.z : lo.z); };
    int ed[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    glBegin(GL_LINES); for (auto& e : ed) { V3 a = V(e[0]), b = V(e[1]); glVertex3d(a.x, a.y, a.z); glVertex3d(b.x, b.y, b.z); } glEnd();
}

// the main wing: the wing feature with the largest span x root chord
static int main_wing_index() {
    const js::Value& fl = features(); int best = -1; double area = -1;
    for (size_t i = 0; i < fl.size(); ++i) {
        const js::Value& f = fl.at(i);
        if (f.str("type", "") != "wing" || f.str("op", "add") != "add" || !f.flag("enabled", true)) continue;
        double a = f.num("span", 0) * f.num("root_chord", 0);
        if (a > area) { area = a; best = (int)i; }
    }
    return best;
}
static void planform_line(const V3& t, double b, double cr, double lam, double sw_deg, bool mirror) {
    double sw = sw_deg * geo::PI / 180;
    for (int side = 0; side < (mirror ? 2 : 1); ++side) {
        double s = side ? -1 : 1;
        V3 rle(t.x, s * t.y, t.z), rte(t.x + cr, s * t.y, t.z), tle(t.x + b * std::tan(sw), s * (t.y + b), t.z), tte(tle.x + cr * lam, tle.y, t.z);
        glBegin(GL_LINE_LOOP); glVertex3d(rle.x, rle.y, rle.z); glVertex3d(tle.x, tle.y, tle.z); glVertex3d(tte.x, tte.y, tte.z); glVertex3d(rte.x, rte.y, rte.z); glEnd();
    }
}

static bool preview_active() { return g_setup && IsWindowVisible(g_setup) && (g_setup_kind == "cfd" || g_setup_kind == "optimise"); }
static void setup_preview_render() {
    if (!preview_active()) return;
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glEnable(GL_LINE_SMOOTH);
    if (g_setup_kind == "cfd") {
        FlowPreview P = flow_preview();
        if (!P.ok) return;
        glLineWidth(1.5f); glColor4f(0.10f, 0.35f, 0.75f, 0.85f); box_edges(P.dlo, P.dhi);   // the computational domain
        glColor4f(0.92f, 0.50f, 0.10f, 0.9f); box_edges(P.clo, P.chi);                        // the refined core (uniform cells)
        // a few core cells on its floor, to show the cell size
        glColor4f(0.92f, 0.50f, 0.10f, 0.35f); glLineWidth(1.0f); glBegin(GL_LINES);
        for (double x = P.clo.x; x <= P.chi.x + 1e-9; x += P.h0) { glVertex3d(x, P.clo.y, P.clo.z); glVertex3d(x, P.chi.y, P.clo.z); }
        for (double y = P.clo.y; y <= P.chi.y + 1e-9; y += P.h0) { glVertex3d(P.clo.x, y, P.clo.z); glVertex3d(P.chi.x, y, P.clo.z); }
        glEnd();
        // the free stream: three arrows ahead of the aircraft along (alpha, beta)
        double L = P.chi.x - P.clo.x;
        glLineWidth(2.0f); glColor4f(0.05f, 0.55f, 0.35f, 0.95f);
        for (int k = -1; k <= 1; ++k) {
            V3 tip = V3(P.clo.x - 0.05 * L, 0, 0.5 * (P.clo.z + P.chi.z) + k * 0.15 * L), tail = tip - P.dir * (0.35 * L);
            V3 side = geo::cross(P.dir, V3(0, 1, 0)); double ls = geo::len(side); side = ls > 0 ? side / ls : V3(0, 0, 1);
            V3 h1 = tip - P.dir * (0.06 * L) + side * (0.025 * L), h2 = tip - P.dir * (0.06 * L) - side * (0.025 * L);
            glBegin(GL_LINES); glVertex3d(tail.x, tail.y, tail.z); glVertex3d(tip.x, tip.y, tip.z); glVertex3d(tip.x, tip.y, tip.z); glVertex3d(h1.x, h1.y, h1.z);
            glVertex3d(tip.x, tip.y, tip.z); glVertex3d(h2.x, h2.y, h2.z); glEnd();
        }
        glColor4f(0.80f, 0.10f, 0.10f, 1.0f);  // probes
        for (const V3& p : P.probes) { double r = 0.02 * L; glBegin(GL_LINES); glVertex3d(p.x - r, p.y, p.z); glVertex3d(p.x + r, p.y, p.z); glVertex3d(p.x, p.y - r, p.z); glVertex3d(p.x, p.y + r, p.z); glVertex3d(p.x, p.y, p.z - r); glVertex3d(p.x, p.y, p.z + r); glEnd(); }
    } else {
        int k = main_wing_index();
        if (k < 0) return;
        const js::Value& f = features().at((size_t)k);
        V3 t = vtr(f, "translate");
        double b0 = f.num("span", 1), c0 = f.num("root_chord", 1), l0 = f.num("taper", 0.5), s0 = f.num("sweep", 30);
        bool mir = f.flag("mirror", false), keep = setup_bool("keep_area", true);
        double area = 0.5 * b0 * c0 * (1 + l0);
        auto draw = [&](double b, double lam, double sw, float r, float g, float bl, float a, float w) {
            double cr = keep ? 2 * area / (b * (1 + lam)) : c0;
            glColor4f(r, g, bl, a); glLineWidth(w); planform_line(t, b, cr, lam, sw, mir);
        };
        double lo, hi;
        if (setup_bool("vary_sweep", true) && setup_num("sweep_min", lo) && setup_num("sweep_max", hi)) { draw(b0, l0, lo, 0.85f, 0.35f, 0.10f, 0.75f, 1.3f); draw(b0, l0, hi, 0.85f, 0.35f, 0.10f, 0.75f, 1.3f); }
        if (setup_bool("vary_taper", true) && setup_num("taper_min", lo) && setup_num("taper_max", hi)) { draw(b0, lo, s0, 0.10f, 0.55f, 0.35f, 0.75f, 1.3f); draw(b0, hi, s0, 0.10f, 0.55f, 0.35f, 0.75f, 1.3f); }
        if (setup_bool("vary_span", false) && setup_num("span_min", lo) && setup_num("span_max", hi)) { draw(b0 * lo, l0, s0, 0.45f, 0.20f, 0.70f, 0.75f, 1.3f); draw(b0 * hi, l0, s0, 0.45f, 0.20f, 0.70f, 0.75f, 1.3f); }
        draw(b0, l0, s0, 0.05f, 0.10f, 0.20f, 1.0f, 2.4f);  // the wing as it is
    }
    glLineWidth(1.0f); glDisable(GL_LINE_SMOOTH); glDisable(GL_BLEND); glEnable(GL_LIGHTING); glEnable(GL_DEPTH_TEST);
}
static void setup_preview_overlay(int W, int H) {
    if (!preview_active()) return;
    char buf[256];
    if (g_setup_kind == "cfd") {
        FlowPreview P = flow_preview();
        if (!P.ok) return;
        snprintf(buf, sizeof buf, "Flow setup preview: domain %.0f x %.0f x %.0f m (%s), core cells %.0f mm, about %.3f million cells",
                 P.dhi.x - P.dlo.x, P.dhi.y - P.dlo.y, P.dhi.z - P.dlo.z, P.half ? "half model" : "full model", P.h0 * 1000, P.cells / 1e6);
        glColor3f(0.10f, 0.30f, 0.62f); gl_text(96, (float)H - 50, buf);
        glColor3f(0.30f, 0.33f, 0.38f);
        gl_text(96, (float)H - 66, "blue: domain   orange: refined core with its cells   green: free stream (angle of attack, sideslip)   red: probes");
    } else {
        glColor3f(0.10f, 0.30f, 0.62f);
        gl_text(96, (float)H - 50, "Wing optimisation preview: dark = the wing now; orange = sweep range, green = taper range, purple = span range");
        glColor3f(0.30f, 0.33f, 0.38f);
        gl_text(96, (float)H - 66, "every CFD run of the study is one wing between these outlines (twist and dihedral do not change the planform)");
    }
    (void)W;
}
