// Interference and fits: every pair of parts whose boxes overlap is sampled on a grid with both parts' exact
// distance functions (from their part designs, not their meshes), so overlapping material is found to the grid's
// resolution; pairs that only touch are reported as contacts. Every shaft-in-hole mate is classed by its diametral
// clearance: loose (running), sliding, transition (line to line) or interference (press). The overlap shows in red.

#include <map>
struct ClashRes { int a = -1, b = -1; double vol = 0; bool interfere = false; std::vector<V3> pts; };
static std::vector<ClashRes> g_clash;
static std::string g_clash_doc;  // the assembly the result belongs to (shown only while it is unchanged)
static std::map<std::string, std::shared_ptr<geo::Model>> g_models;
static const geo::Model* piece_model(const APiece& pc) {
    auto it = g_models.find(pc.key);
    if (it != g_models.end()) return it->second.get();
    auto m = std::make_shared<geo::Model>();
    try { m->load(pc.part); } catch (...) {}
    g_models[pc.key] = m;
    return m.get();
}
static double comp_sd(int c, const V3& w) {  // signed distance to a component (all its pieces), world point
    const AComp& C = g_ac[(size_t)c];
    V3 l = C.pose.inv(w);
    double d = 1e30;
    for (const APiece& pc : C.pieces) { const geo::Model* M = piece_model(pc); if (M->bounds.valid()) d = std::min(d, M->eval(pc.local.inv(l))); }
    return d;
}
static double mate_radius(const ARef& r, int c) {  // a round reference's radius, or the part's own nominal size
    if (r.r > 0) return r.r;
    const js::Value& P = g_ac[(size_t)c].part;
    if (const js::Value* f = P.find("fastener")) return 0.5 * f->num("d", 0);
    if (const js::Value* f = P.find("shaft")) return 0.5 * f->num("d", 0);
    return 0;
}
static std::string fit_class(double hole_d, double shaft_d, double& clr) {
    clr = hole_d - shaft_d;
    double d = std::max(hole_d, shaft_d), tol = 2e-4 * d + 1e-6;
    if (clr < -tol) return "interference (press fit)";
    if (clr <= tol) return "transition (line to line)";
    if (clr <= 3e-3 * d) return "clearance (sliding)";
    return "clearance (loose, running)";
}
static void asm_check() {
    if (!g_asm_on) { MessageBoxA(g_main, "The interference check works on an assembly.", "Interference", MB_ICONINFORMATION); return; }
    for (const AComp& c : g_ac) if (!c.hidden && !comp_ready(c)) { status("the parts are still meshing: try again in a moment"); return; }
    g_clash.clear();
    size_t n = g_ac.size();
    std::vector<geo::Box> wb(n);
    for (size_t c = 0; c < n; ++c) comp_world_box(g_ac[c], wb[c]);
    int pairs = 0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            if (g_ac[i].hidden || g_ac[j].hidden || !wb[i].valid() || !wb[j].valid()) continue;
            geo::Box I; I.lo = V3(std::max(wb[i].lo.x, wb[j].lo.x), std::max(wb[i].lo.y, wb[j].lo.y), std::max(wb[i].lo.z, wb[j].lo.z));
            I.hi = V3(std::min(wb[i].hi.x, wb[j].hi.x), std::min(wb[i].hi.y, wb[j].hi.y), std::min(wb[i].hi.z, wb[j].hi.z));
            V3 e = I.hi - I.lo;
            double s = std::max(1e-4, std::max(e.x, std::max(e.y, e.z)) / 40);  // the sample spacing: 40 across the overlap
            I.lo = I.lo - V3(0.5 * s, 0.5 * s, 0.5 * s); I.hi = I.hi + V3(0.5 * s, 0.5 * s, 0.5 * s);
            e = I.hi - I.lo;
            if (e.x <= 0 || e.y <= 0 || e.z <= 0) continue;
            ++pairs;
            int nx = std::max(1, (int)std::ceil(e.x / s)), ny = std::max(1, (int)std::ceil(e.y / s)), nz = std::max(1, (int)std::ceil(e.z / s));
            double hx = e.x / nx, hy = e.y / ny, hz = e.z / nz, h = std::max(hx, std::max(hy, hz));
            ClashRes r; r.a = (int)i; r.b = (int)j;
            long in = 0, touch = 0;
            for (int k = 0; k < nz; ++k) for (int jj = 0; jj < ny; ++jj) for (int ii = 0; ii < nx; ++ii) {
                V3 p = I.lo + V3((ii + 0.5) * hx, (jj + 0.5) * hy, (k + 0.5) * hz);
                double da = comp_sd((int)i, p);
                if (da > 0.6 * h) continue;
                double db = comp_sd((int)j, p);
                if (da < -0.02 * h && db < -0.02 * h) { ++in; if (r.pts.size() < 600) r.pts.push_back(p); }
                else if (da < 0.6 * h && db < 0.6 * h) ++touch;
            }
            r.vol = in * hx * hy * hz;
            if (in >= 2) { r.interfere = true; g_clash.push_back(r); }
            else if (touch > 0) { r.pts.clear(); g_clash.push_back(r); }
        }
    // the report
    std::string out; char b[512];
    auto line = [&](const std::string& t) { out += t; out += "\n"; };
    int nint = 0; for (auto& c : g_clash) if (c.interfere) ++nint;
    snprintf(b, sizeof b, "INTERFERENCE CHECK: %d parts, %d pairs close enough to test, %d interfering, %d touching", (int)n, pairs, nint, (int)g_clash.size() - nint); line(b);
    line("");
    line("PAIR                                 RESULT          OVERLAP mm3");
    for (auto& c : g_clash) {
        snprintf(b, sizeof b, "%-16.16s / %-16.16s   %-14s %12.4g", g_ac[(size_t)c.a].id.c_str(), g_ac[(size_t)c.b].id.c_str(), c.interfere ? "INTERFERE" : "touch", c.vol * 1e9); line(b);
    }
    if (g_clash.empty()) line("  none: no two parts touch or overlap");
    line("");
    line("FITS (shaft in hole, at 20 C)        HOLE mm   SHAFT mm   CLEARANCE um   CLASS");
    int nfit = 0;
    for (const AMate& m : g_am) {
        if (!m.on || (m.type != MT_CONCENTRIC && m.type != MT_INSERT) || m.a.comp < 0 || m.b.comp < 0) continue;
        double ra = mate_radius(m.a, m.a.comp), rb = mate_radius(m.b, m.b.comp);
        if (ra <= 0 || rb <= 0 || std::fabs(ra - rb) > 0.2 * std::max(ra, rb)) continue;
        bool ahole = m.a.hole || (!m.b.hole && ra >= rb);
        double hd = 2 * (ahole ? ra : rb), sd = 2 * (ahole ? rb : ra), clr;
        std::string cls = fit_class(hd, sd, clr);
        snprintf(b, sizeof b, "%-14.14s / %-14.14s   %8.3f   %8.3f   %12.1f   %s", g_ac[(size_t)m.a.comp].id.c_str(), g_ac[(size_t)m.b.comp].id.c_str(), hd * 1000, sd * 1000, clr * 1e6, cls.c_str());
        line(b); ++nfit;
    }
    if (!nfit) line("  no shaft-in-hole mates with known diameters");
    line("");
    line("Overlap: both parts' exact distance functions sampled on a grid of 40 across each overlapping box (red points in the");
    line("view). A gear pair that interferes has its teeth out of phase or its centres too close; a shaft that interferes");
    line("with its hole is a press fit (intended or not). Classes: line to line within 0.02 % of the diameter; sliding up to");
    line("0.3 %; loose beyond.");
    g_clash_doc = g_asm_loaded;
    if (!g_selftest) {
        SYSTEMTIME t; GetLocalTime(&t);
        char nm[64]; snprintf(nm, sizeof nm, "check_%04d%02d%02d_%02d%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
        std::string dir = g_root + "\\results\\studio\\" + nm;
        make_dirs(dir);
        try { js::write_file(dir + "\\summary.txt", out); } catch (...) {}
        snap_view_png(dir + "\\00_view_3d.png"); open_simview(dir, "Interference and fits");
    }
    status(nint ? tostr(nint) + " pairs interfere (red in the view); " + tostr((int)g_clash.size() - nint) + " touch; " + tostr(nfit) + " fits classed" : "no interference; " + tostr((int)g_clash.size()) + " pairs touch; " + tostr(nfit) + " fits classed");
    InvalidateRect(g_view, nullptr, FALSE);
}
static void check_render() {
    if (g_clash.empty() || g_clash_doc != g_asm_loaded) return;
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(5.0f); glEnable(GL_POINT_SMOOTH);
    glBegin(GL_POINTS);
    for (auto& c : g_clash) { if (!c.interfere) continue; glColor4f(0.92f, 0.1f, 0.08f, 0.9f); for (const V3& p : c.pts) glVertex3d(p.x, p.y, p.z); }
    glEnd();
    glDisable(GL_POINT_SMOOTH); glPointSize(1.0f);
    glEnable(GL_DEPTH_TEST); glDisable(GL_BLEND); glEnable(GL_LIGHTING);
}
