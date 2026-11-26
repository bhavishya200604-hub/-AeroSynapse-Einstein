// Simulation in Studio: launching the analytic, PINN and structural simulations, the results viewer, the 3D flow
// field display (Mach-number slices from the flow solver drawn in the 3D view) and animated flow particles carried by
// the solver's velocity on the symmetry plane. Also the report and CAD-package exports.

// ---------------------------------------------------------------- colour map (Google's turbo, polynomial fit)
static void turbo(double t, float& r, float& g, float& b) {
    t = std::max(0.0, std::min(1.0, t));
    r = (float)(0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))));
    g = (float)(0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))));
    b = (float)(0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))));
    r = std::max(0.0f, std::min(1.0f, r)); g = std::max(0.0f, std::min(1.0f, g)); b = std::max(0.0f, std::min(1.0f, b));
}

// ---------------------------------------------------------------- flow field
struct FieldPlane {
    char axis = 'y', ua = 'x', va = 'z';
    double at = 0;
    std::vector<double> u, v;
    std::vector<float> mach, vu, vv, vw, solid;  // [j * nu + i]
    size_t nu() const { return u.size(); }
};
struct Particle { double a = 0, b = 0; float tr[12][2]; int n = 0; };
static std::vector<FieldPlane> g_planes;
static std::vector<Particle> g_parts;
static bool g_field_on = false, g_particles_on = false;
static float g_mach_lo = 0, g_mach_hi = 1;
static std::string g_field_dir, g_last_cfd_dir;
static double g_box_lo[3], g_box_hi[3], g_uinf = 1, g_minf = 0.5;
static unsigned g_rng = 12345;
static double frand() { g_rng = g_rng * 1664525u + 1013904223u; return (g_rng >> 8) / 16777216.0; }

static bool load_plane(const std::string& dir, const std::string& name, FieldPlane& P) {
    std::string hj;
    try { hj = js::read_file(dir + "\\" + name + ".json"); } catch (...) { return false; }
    js::Value h = js::parse(hj);
    P.axis = h.str("axis", "y")[0]; P.ua = h.str("u_axis", "x")[0]; P.va = h.str("v_axis", "z")[0]; P.at = h.num("at", 0);
    P.u = h.nums("u"); P.v = h.nums("v");
    std::vector<std::string> vars;
    if (const js::Value* vs = h.find("vars")) for (size_t i = 0; i < vs->size(); ++i) vars.push_back(vs->at(i).s);
    size_t n = P.u.size() * P.v.size();
    FILE* f = fopen((dir + "\\" + name + ".bin").c_str(), "rb");
    if (!f || n == 0) { if (f) fclose(f); return false; }
    std::vector<float> all(n * vars.size());
    size_t got = fread(all.data(), sizeof(float), all.size(), f);
    fclose(f);
    if (got != all.size()) return false;
    auto var = [&](const char* k, std::vector<float>& out) {
        for (size_t i = 0; i < vars.size(); ++i) if (vars[i] == k) { out.assign(all.begin() + (long)(i * n), all.begin() + (long)((i + 1) * n)); return; }
        out.assign(n, 0.0f);
    };
    var("mach", P.mach); var("u", P.vu); var("v", P.vv); var("w", P.vw); var("solid", P.solid);
    return true;
}

static void field_seed() {
    g_parts.assign(420, Particle());
    for (auto& p : g_parts) { p.a = g_box_lo[0] + frand() * (g_box_hi[0] - g_box_lo[0]); p.b = g_box_lo[2] + frand() * (g_box_hi[2] - g_box_lo[2]); p.n = 0; }
}

static bool field_load(const std::string& dir) {
    if (file_exists(dir + "\\volume.json") && vol_load(dir)) { g_field_dir = dir; viz_load(dir); return true; }  // the whole 3D volume
    std::vector<FieldPlane> ps;
    const char* names[] = {"plane_symmetry", "plane_x1", "plane_x2", "plane_x3", "plane_z"};
    for (const char* n : names) { FieldPlane P; if (load_plane(dir, n, P)) ps.push_back(P); }
    if (ps.empty()) return false;
    g_planes = ps; g_field_dir = dir;
    double L = std::max(1.0, g_props.hi.x - g_props.lo.x);
    g_box_lo[0] = g_props.lo.x - 0.25 * L; g_box_hi[0] = g_props.hi.x + 0.45 * L;
    g_box_lo[1] = -0.05; g_box_hi[1] = g_props.hi.y + 0.25 * L;
    g_box_lo[2] = g_props.lo.z - 0.3 * L; g_box_hi[2] = g_props.hi.z + 0.3 * L;
    // Mach range (2nd to 98th percentile in the display box), freestream Mach and speed from the inflow column
    std::vector<float> ms;
    double us = 0, mi = 0; int un = 0;
    for (auto& P : g_planes) {
        int ia = P.ua - 'x', ib = P.va - 'x';
        for (size_t j = 0; j < P.v.size(); ++j)
            for (size_t i = 0; i < P.nu(); ++i) {
                size_t k = j * P.nu() + i;
                if (P.axis == 'y' && i == 0) { us += std::sqrt(P.vu[k] * P.vu[k] + P.vw[k] * P.vw[k]); mi += P.mach[k]; ++un; }
                if (P.solid[k] > 0.5 || P.u[i] < g_box_lo[ia] || P.u[i] > g_box_hi[ia] || P.v[j] < g_box_lo[ib] || P.v[j] > g_box_hi[ib]) continue;
                ms.push_back(P.mach[k]);
            }
    }
    g_uinf = un ? us / un : 1.0;
    g_minf = un ? mi / un : 0.5;
    if (ms.size() > 10) {
        std::sort(ms.begin(), ms.end());
        g_mach_lo = ms[ms.size() * 2 / 100]; g_mach_hi = ms[ms.size() * 98 / 100];
    } else { g_mach_lo = 0; g_mach_hi = 1; }
    if (g_mach_hi <= g_mach_lo + 1e-3f) g_mach_hi = g_mach_lo + 0.05f;
    field_seed();
    return true;
}

static bool sample_sym(double x, double z, double& vx, double& vz, double& m, bool& solid) {
    const FieldPlane* S = nullptr;
    for (auto& P : g_planes) if (P.axis == 'y') S = &P;
    if (!S || S->u.size() < 2 || S->v.size() < 2) return false;
    if (x <= S->u.front() || x >= S->u.back() || z <= S->v.front() || z >= S->v.back()) return false;
    size_t i = (size_t)(std::upper_bound(S->u.begin(), S->u.end(), x) - S->u.begin()) - 1, j = (size_t)(std::upper_bound(S->v.begin(), S->v.end(), z) - S->v.begin()) - 1;
    double fx = (x - S->u[i]) / (S->u[i + 1] - S->u[i]), fz = (z - S->v[j]) / (S->v[j + 1] - S->v[j]);
    size_t nu = S->nu();
    auto bil = [&](const std::vector<float>& a) {
        return (1 - fx) * (1 - fz) * a[j * nu + i] + fx * (1 - fz) * a[j * nu + i + 1] + (1 - fx) * fz * a[(j + 1) * nu + i] + fx * fz * a[(j + 1) * nu + i + 1];
    };
    vx = bil(S->vu); vz = bil(S->vw); m = bil(S->mach);
    solid = bil(S->solid) > 0.4;
    return true;
}

static void field_step() {
    if (g_vol_ok) { vol_step(); return; }
    double L = std::max(1.0, g_box_hi[0] - g_box_lo[0]), dt = 0.006 * L / std::max(g_uinf, 1e-6);
    for (auto& p : g_parts) {
        double vx, vz, m; bool solid;
        bool ok = sample_sym(p.a, p.b, vx, vz, m, solid);
        if (!ok || solid || p.a > g_box_hi[0] || p.b < g_box_lo[2] || p.b > g_box_hi[2] || frand() < 0.004) {
            p.a = g_box_lo[0] + frand() * 0.15 * L; p.b = g_box_lo[2] + frand() * (g_box_hi[2] - g_box_lo[2]); p.n = 0;
            continue;
        }
        for (int k = std::min(p.n, 11); k > 0; --k) { p.tr[k][0] = p.tr[k - 1][0]; p.tr[k][1] = p.tr[k - 1][1]; }
        p.tr[0][0] = (float)p.a; p.tr[0][1] = (float)p.b;
        p.n = std::min(p.n + 1, 12);
        p.a += vx * dt; p.b += vz * dt;
    }
}

static void field_render() {
    if (!g_field_on) return;
    if (g_vol_ok) { vol_render(g_particles_on); return; }
    if (g_planes.empty()) return;
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
    auto put = [](const FieldPlane& P, double a, double b, double mirror) {
        double p[3]; int ia = P.ua - 'x', ib = P.va - 'x', ic = P.axis - 'x';
        p[ia] = a; p[ib] = b; p[ic] = P.at; p[1] *= mirror;
        glVertex3dv(p);
    };
    for (auto& P : g_planes) {
        int ia = P.ua - 'x', ib = P.va - 'x';
        for (int side = 0; side < (P.axis == 'y' ? 1 : 2); ++side) {
            double mir = side ? -1.0 : 1.0;
            for (size_t j = 0; j + 1 < P.v.size(); ++j) {
                if (P.v[j + 1] < g_box_lo[ib] || P.v[j] > g_box_hi[ib]) continue;
                glBegin(GL_QUAD_STRIP);
                for (size_t i = 0; i < P.nu(); ++i) {
                    if (P.u[i] < g_box_lo[ia] - 1 || P.u[i] > g_box_hi[ia] + 1) continue;
                    for (size_t jj = j; jj <= j + 1; ++jj) {
                        size_t k = jj * P.nu() + i;
                        float r, g, b; turbo((P.mach[k] - g_mach_lo) / (g_mach_hi - g_mach_lo), r, g, b);
                        // undisturbed flow stays nearly clear; shocks, expansions and wakes glow
                        float dev = (float)std::min(1.0, std::fabs(P.mach[k] - g_minf) / std::max(0.02, 0.35 * (g_mach_hi - g_mach_lo)));
                        float base = P.axis == 'y' ? 0.78f : 0.5f;
                        glColor4f(r, g, b, P.solid[k] > 0.5f ? 0.0f : base * (0.08f + 0.92f * dev));
                        put(P, P.u[i], P.v[jj], mir);
                    }
                }
                glEnd();
            }
        }
    }
    if (g_particles_on) {
        glDisable(GL_DEPTH_TEST);
        glLineWidth(1.6f);
        glBegin(GL_LINES);
        for (auto& p : g_parts) {
            for (int k = 0; k + 1 < p.n; ++k) {
                double vx, vz, m; bool s;
                float r = 0.8f, g = 0.95f, b = 1.0f;
                if (sample_sym(p.tr[k][0], p.tr[k][1], vx, vz, m, s)) { turbo((m - g_mach_lo) / (g_mach_hi - g_mach_lo), r, g, b); r = 0.35f + 0.65f * r; g = 0.35f + 0.65f * g; b = 0.35f + 0.65f * b; }
                float a = 1.0f - (float)k / 12.0f;
                glColor4f(r, g, b, 0.75f * a); glVertex3d(p.tr[k][0], 0.12, p.tr[k][1]);
                glColor4f(r, g, b, 0.75f * a * 0.6f); glVertex3d(p.tr[k + 1][0], 0.12, p.tr[k + 1][1]);
            }
        }
        glEnd();
        glPointSize(3.0f);
        glBegin(GL_POINTS); glColor4f(1, 1, 1, 0.95f);
        for (auto& p : g_parts) if (p.n) glVertex3d(p.tr[0][0], 0.12, p.tr[0][1]);
        glEnd();
        glLineWidth(1.0f); glEnable(GL_DEPTH_TEST);
    }
    glDepthMask(GL_TRUE); glDisable(GL_BLEND);
}

static void field_legend(int W, int H) {
    if (!g_field_on || (g_planes.empty() && !g_vol_ok)) return;
    (void)W; (void)H;
    float g_mach_lo = g_vol_ok ? g_vm_lo : ::g_mach_lo, g_mach_hi = g_vol_ok ? g_vm_hi : ::g_mach_hi;
    double g_minf = g_vol_ok ? g_vminf : ::g_minf;
    glColor3f(0.15f, 0.18f, 0.24f);
    gl_text(12, 36, "Mach");
    for (int i = 0; i < 100; ++i) {
        float r, g, b; turbo(i / 99.0, r, g, b); glColor3f(r, g, b);
        glBegin(GL_QUADS); glVertex2f(56.0f + 2 * i, 34); glVertex2f(58.0f + 2 * i, 34); glVertex2f(58.0f + 2 * i, 46); glVertex2f(56.0f + 2 * i, 46); glEnd();
    }
    char b[64];
    glColor3f(0.15f, 0.18f, 0.24f);
    snprintf(b, sizeof b, "%.2f", g_mach_lo); gl_text(56, 52, b);
    snprintf(b, sizeof b, "%.2f", g_mach_hi); gl_text(232, 52, b);
    snprintf(b, sizeof b, "freestream %.2f", g_minf); gl_text(270, 36, b);
}


static std::string newest_results(const char* marker) {
    std::string best; FILETIME bt = {0, 0};
    const char* roots[] = {"\\results\\studio\\", "\\results\\cfd3d\\", "\\results\\optimise\\"};
    for (const char* r : roots) {
        std::string base = g_root + r;
        WIN32_FIND_DATAA fd; HANDLE h = FindFirstFileA((base + "*").c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || fd.cFileName[0] == '.') continue;
            std::string d = base + fd.cFileName, f = d + "\\" + marker;
            WIN32_FILE_ATTRIBUTE_DATA a;
            if (!GetFileAttributesExA(f.c_str(), GetFileExInfoStandard, &a)) continue;
            if (CompareFileTime(&a.ftLastWriteTime, &bt) > 0) { bt = a.ftLastWriteTime; best = d; }
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
    return best;
}
static std::string find_field_dir() {
    if (!g_last_cfd_dir.empty() && (file_exists(g_last_cfd_dir + "\\volume.json") || file_exists(g_last_cfd_dir + "\\plane_symmetry.json"))) return g_last_cfd_dir;
    if (!g_wf_dir.empty()) {
        try {
            js::Value r = js::parse(js::read_file(g_wf_dir + "\\07_simulation\\result.json"));
            std::string d = r.str("studio_results_dir", "");
            for (char& c : d) if (c == '/') c = '\\';
            if (!d.empty() && file_exists(g_wf_dir + "\\" + d + "\\plane_symmetry.json")) return g_wf_dir + "\\" + d;
        } catch (...) {}
    }
    {   // the newest 3D flow run on disk, so the tools work straight away
        std::string v = newest_results("volume.json");
        if (v.empty()) v = newest_results("plane_symmetry.json");
        if (!v.empty()) { status("flow field from the newest run: " + v); return v; }
    }
    std::string p = g_root + "\\results\\studio\\plane_symmetry.json";
    if (!ask_path(false, "Flow solver field (plane_symmetry.json)\0plane_symmetry.json\0All files\0*.*\0", "json", p)) return "";
    size_t at = p.find_last_of("\\/");
    return at == std::string::npos ? p : p.substr(0, at);
}

static void toggle_field(bool particles) {
    if (g_planes.empty() && !g_vol_ok) {
        std::string d = find_field_dir();
        if (d.empty()) return;
        if (!field_load(d)) { MessageBoxA(g_main, ("No flow-field planes in " + d + ". Run a 3D flow simulation first (Simulate tab).").c_str(), "Flow field", MB_ICONINFORMATION); return; }
        g_field_on = true;
    }
    if (particles) { g_particles_on = !g_particles_on; g_field_on = true; if (g_particles_on) SetTimer(g_main, 7, 33, nullptr); else KillTimer(g_main, 7); }
    else { g_field_on = !g_field_on; if (!g_field_on) { g_particles_on = false; KillTimer(g_main, 7); } }
    status(g_field_on ? "flow field: Mach number on the solver's planes (" + g_field_dir + ")" + (g_particles_on ? ", particles on" : "") : "flow field hidden");
    InvalidateRect(g_view, nullptr, FALSE);
}

#include "resview.inc"

static void run_simulation(const char* kind) {
    std::string d = work_dir(std::string("sim_") + kind);
    js::write_file(d + "\\design.json", js::dump(g_doc, 2));
    char w0[64] = "";
    double W0 = g_doc.num("W0_kg", 0);
    if (W0 > 0) snprintf(w0, sizeof w0, " --W0 %.0f", W0);
    launch("\"" + python() + "\" -m aerosynapse.cli simulate " + kind + " \"" + d + "\\design.json\" --out \"" + d + "\"" + w0, std::string("simulation ") + kind, d);
}

static void export_report(const char* fmt) {
    if (!ensure_project(g_main)) return;
    launch("\"" + python() + "\" -m aerosynapse.cli workflow report \"" + g_wf_dir + "\" --format " + fmt, std::string("report ") + fmt, g_wf_dir + "\\08_export");
}

static void export_package() {
    std::string d = work_dir("cad_package");
    js::write_file(d + "\\design.json", js::dump(g_doc, 2));
    launch("\"" + python() + "\" -m aerosynapse.cli export-design \"" + d + "\\design.json\" --out \"" + d + "\"", "CAD package", d);
}

// called from job_done for the jobs started here; returns true when it handled the job
static bool sim_job_done(DWORD code) {
    if (g_job_kind.compare(0, 11, "simulation ") == 0) {
        std::string k = g_job_kind.substr(11);
        if (!file_exists(g_job_dir + "\\summary.txt")) {
            if (k != "live") MessageBoxA(g_main, ("The simulation failed (exit code " + tostr((long)code) + "). Files: " + g_job_dir).c_str(), "Simulation", MB_ICONERROR);
            return true;
        }
        if (k.compare(0, 3, "cfd") == 0 || k == "live") {  // full-3D flow: surface pressure, the volume, streamlines and vortex cores in the 3D view
            g_last_cfd_dir = g_job_dir; g_planes.clear(); g_vol_ok = false;
            load_surface_csv(g_job_dir + "\\surface.csv");
            if (field_load(g_job_dir)) { g_field_on = true; g_particles_on = k != "live"; SetTimer(g_main, 7, 33, nullptr); }
            if (k != "live") open_simview(g_job_dir, "3D flow analysis: " + k.substr(k.find(' ') == std::string::npos ? 0 : k.find(' ') + 1));
            else status("live simulation updated: surface pressure and flow field of the current design");
            return true;
        }
        open_simview(g_job_dir, k == "analytic" ? "Analytic simulation" : k == "pinn" ? "Simulation with the parametric PINN" : k == "fin_pinn" ? "Fin PINN: every optimisation setting, 2D and 3D" :
                                k == "turbulence" ? "Turbulence and flow instability" : "Structural simulation");
        viz_load(g_job_dir);
        return true;
    }
    if (g_job_kind.compare(0, 9, "engineer ") == 0) {
        std::string k = g_job_kind.substr(9);
        if (!file_exists(g_job_dir + "\\summary.txt")) { MessageBoxA(g_main, ("The tool failed (exit code " + tostr((long)code) + "). Files: " + g_job_dir).c_str(), "Engineering", MB_ICONERROR); return true; }
        if (k == "skeleton") {
            try { std::string keep = g_path; set_doc(js::parse(js::read_file(g_job_dir + "\\design.json")), keep); g_modified = true; status("geometry rebuilt from the skeleton parameters"); }
            catch (std::exception& e) { MessageBoxA(g_main, e.what(), "Skeleton", MB_ICONERROR); }
            return true;
        }
        open_simview(g_job_dir, k == "clash" ? "Clash analysis" : k == "rules" ? "Design rules (knowledgeware)" : k == "composites" ? "Composite laminate" : "RFLP traceability");
        return true;
    }
    if (g_job_kind.compare(0, 7, "report ") == 0) {
        std::string ext = g_job_kind.substr(7), src = g_job_dir + "\\report." + ext;
        if (!file_exists(src)) { MessageBoxA(g_main, ("The report was not written (exit code " + tostr((long)code) + ").").c_str(), "Report", MB_ICONERROR); return true; }
        std::string nm = g_wf_dir.substr(g_wf_dir.find_last_of("\\/") + 1), p = nm + "_report." + ext;
        const char* filt = ext == "pdf" ? "PDF document (*.pdf)\0*.pdf\0" : ext == "docx" ? "Word document (*.docx)\0*.docx\0" : "Text (*.txt)\0*.txt\0";
        if (ask_path(true, filt, ext.c_str(), p)) {
            CopyFileA(src.c_str(), p.c_str(), FALSE);
            ShellExecuteA(g_main, "open", p.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            status("report saved: " + p);
        }
        return true;
    }
    if (g_job_kind == "CAD package") {
        if (!file_exists(g_job_dir + "\\design.step")) { MessageBoxA(g_main, ("The export failed (exit code " + tostr((long)code) + "). Files: " + g_job_dir).c_str(), "Export", MB_ICONERROR); return true; }
        ShellExecuteA(g_main, "open", g_job_dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
        status("CAD package written: " + g_job_dir);
        return true;
    }
    return false;
}

static void register_sim_classes(HINSTANCE hi, HICON icon) {
    WNDCLASSA a; memset(&a, 0, sizeof a);
    a.lpfnWndProc = sv_proc; a.hInstance = hi; a.hCursor = LoadCursor(nullptr, IDC_ARROW); a.hbrBackground = g_br_panel; a.lpszClassName = "AeroSynapseSimView"; a.hIcon = icon;
    RegisterClassA(&a);
    a.lpfnWndProc = simimage_proc; a.hbrBackground = nullptr; a.lpszClassName = "AeroSynapseSimImage";
    RegisterClassA(&a);
    a.lpfnWndProc = fig_proc; a.style = CS_DBLCLKS; a.lpszClassName = "AeroSynapseFigure";
    RegisterClassA(&a);
}
