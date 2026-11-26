// AeroSynapse Studio: feature-based design editor for whole aircraft (Win32 + OpenGL 1.1).
//
// The design is the same JSON feature tree the optimiser and the flow solver use (cpp/common/sdf.hpp):
// bodies, wings, canards, tails, fins, strakes, blades and primitives that are added, cut or intersected,
// with blends, mirror, shell (hollow), patterns and twist. Every edit rebuilds the surface in a background
// thread; undo/redo keep whole-document snapshots.
//
// Mouse: left drag orbit, right drag pan, wheel zoom. Keys: arrows / PgUp PgDn move the selected feature,
// + - scale, R rotate about z, T about x, Del delete, Ctrl+D duplicate, Ctrl+Z / Ctrl+Y undo / redo, W wireframe,
// C section cut, X Y Z cut axis, [ ] move the cut, 0-6 standard views, F fit, H mesh resolution.
#define _WIN32_WINNT 0x0601
#define _WIN32_IE 0x0600
#include "../common/mesh.hpp"
#include "../common/tess.hpp"

#include <commctrl.h>
#include <commdlg.h>
#include <shellapi.h>
#include <shlobj.h>

// self-test mode (--selftest): dialogs, file pickers, opened folders and started processes are recorded, not shown
static bool g_selftest = false;
static std::vector<std::string> g_st_log;
static int st_MessageBoxA(HWND h, LPCSTR t, LPCSTR c, UINT f) {
    if (g_selftest) {
        g_st_log.push_back(std::string((f & 0xF0) == MB_ICONERROR ? "ERROR " : (f & 0xF0) == MB_ICONWARNING ? "WARNING " : "MSG ") + (c ? c : "") + ": " + (t ? t : ""));
        UINT b = f & 0xF; return b == MB_YESNO || b == MB_YESNOCANCEL ? IDYES : IDOK;
    }
    return MessageBoxA(h, t, c, f);
}
static BOOL st_GetOpenFileNameA(LPOPENFILENAMEA o) { if (g_selftest) { g_st_log.push_back("DIALOG open file"); return FALSE; } return GetOpenFileNameA(o); }
static BOOL st_GetSaveFileNameA(LPOPENFILENAMEA o) { if (g_selftest) { g_st_log.push_back("DIALOG save file"); return FALSE; } return GetSaveFileNameA(o); }
static LPITEMIDLIST st_SHBrowseForFolderA(LPBROWSEINFOA b) { if (g_selftest) { g_st_log.push_back("DIALOG choose folder"); return nullptr; } return SHBrowseForFolderA(b); }
static HINSTANCE st_ShellExecuteA(HWND h, LPCSTR op, LPCSTR f, LPCSTR a, LPCSTR d, INT n) { if (g_selftest) { g_st_log.push_back(std::string("OPEN ") + (f ? f : "")); return (HINSTANCE)33; } return ShellExecuteA(h, op, f, a, d, n); }
static BOOL st_CreateProcessA(LPCSTR app, LPSTR cmd, LPSECURITY_ATTRIBUTES pa, LPSECURITY_ATTRIBUTES ta, BOOL inh, DWORD fl, LPVOID env, LPCSTR dir, LPSTARTUPINFOA si, LPPROCESS_INFORMATION pi) {
    if (g_selftest) {  // record the job; a placeholder that waits stands in, so no result handler runs
        g_st_log.push_back(std::string("JOB ") + (cmd ? cmd : ""));
        char dummy[] = "cmd /c ping -n 60 127.0.0.1 >nul";
        return CreateProcessA(nullptr, dummy, pa, ta, FALSE, (fl & ~CREATE_SUSPENDED) | CREATE_NO_WINDOW, env, dir, si, pi);
    }
    return CreateProcessA(app, cmd, pa, ta, inh, fl, env, dir, si, pi);
}
#define MessageBoxA st_MessageBoxA
#define GetOpenFileNameA st_GetOpenFileNameA
#define GetSaveFileNameA st_GetSaveFileNameA
#define SHBrowseForFolderA st_SHBrowseForFolderA
#define ShellExecuteA st_ShellExecuteA
#define CreateProcessA st_CreateProcessA
#include <GL/gl.h>
#include <GL/glu.h>
// line widths and point sizes grow with an ultra-clear picture (hq.inc), so they keep their look at any size
static float g_gl_lscale = 1.0f;
static inline void gl_line_width_scaled(GLfloat w) { glLineWidth(w * g_gl_lscale); }
static inline void gl_point_size_scaled(GLfloat s) { glPointSize(s * g_gl_lscale); }
#define glLineWidth gl_line_width_scaled
#define glPointSize gl_point_size_scaled
static bool g_hr_on = false; static int g_hr_W = 0, g_hr_H = 0, g_hr_lW = 0, g_hr_lH = 0;  // drawing off screen at W x H for a window of lW x lH

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <functional>
#include <tuple>
#include <set>
#include <memory>
#include <string>
#include <vector>

using geo::V3;

enum {
    ID_LIST = 100, ID_STATUS, ID_APPLY, ID_JSON, ID_APPLYJSON, ID_BTN_UP, ID_BTN_DOWN, ID_BTN_DEL, ID_BTN_DUP, ID_BTN_TOGGLE, ID_BTN_OP,
    ID_FILE_NEW = 1000, ID_FILE_OPEN, ID_FILE_SAVE, ID_FILE_SAVEAS, ID_FILE_STL, ID_FILE_OBJ, ID_FILE_SVG, ID_FILE_EXIT,
    ID_EDIT_UNDO = 1100, ID_EDIT_REDO, ID_EDIT_DEL, ID_EDIT_DUP, ID_EDIT_UP, ID_EDIT_DOWN, ID_EDIT_TOGGLE, ID_EDIT_OP,
    ID_INS_BASE = 2000,  // + index in geo::feature_kinds()
    ID_INS_SLOT = 2100, ID_INS_HOLE, ID_INS_CAVITY, ID_MOD_SHELL, ID_MOD_PATTERN, ID_MOD_MIRROR, ID_MOD_BLEND, ID_MOD_TWIST,
    ID_VIEW_ISO = 3000, ID_VIEW_FRONT, ID_VIEW_SIDE, ID_VIEW_TOP, ID_VIEW_FIT, ID_VIEW_WIRE, ID_VIEW_CUT, ID_VIEW_CUTX, ID_VIEW_CUTY, ID_VIEW_CUTZ, ID_VIEW_RES, ID_VIEW_RESULTS_OFF,
    ID_TOOL_MEASURE = 4000, ID_TOOL_CHECK,
    ID_AN_QUICK = 5000, ID_AN_CFD_CRUISE, ID_AN_CFD_TRANS, ID_AN_CFD_SUPER, ID_AN_CFD_ALPHA, ID_AN_LOAD,
    ID_HELP_ABOUT = 6000, ID_HELP_KEYS,
    ID_INS_AIRFOIL = 2200, ID_RES_CP = 5100, ID_RES_CPRMS, ID_RES_CF, ID_RES_T,
    ID_OPT_CENTRE = 7000, ID_OPT_TASK0 = 7001,  // + task index
    ID_JOB_LIST = 7500, ID_JOB_ARGS, ID_JOB_RUN, ID_JOB_STOP, ID_JOB_OPEN, ID_JOB_LOG, ID_JOB_DESC,
    ID_MOD_TAPER = 2120, ID_MOD_BEND, ID_MOD_SHEAR, ID_MOD_STRETCH, ID_MOD_PIVOT, ID_MOD_INFILL, ID_MOD_CLEARX,
    ID_WF_LIST = 8000, ID_WF_RUN, ID_WF_ALL, ID_WF_FROM, ID_WF_STOP, ID_WF_LOAD, ID_WF_REPORT, ID_WF_FOLDER, ID_WF_NEW, ID_WF_OPEN, ID_WF_PROFILE, ID_WF_LOG, ID_WF_DESC,
    ID_WF_PROJ, ID_WF_START, ID_ASK_EDIT, ID_ASK_OK, ID_ASK_CANCEL,
    ID_WFM_WINDOW = 8100, ID_WFM_NEW, ID_WFM_OPEN, ID_WFM_NEXT, ID_WFM_REPORT, ID_WFM_STAGE0 = 8200,  // + stage index
    ID_SK_REVOLVE = 8300, ID_SK_EXTRUDE, ID_SK_LOFT, ID_SIM_FIELD, ID_SIM_PARTICLES, ID_SIM_ANALYTIC, ID_SIM_PINN, ID_SIM_STRUCT, ID_SIM_VIEWER,
    ID_EXP_PACKAGE, ID_EXP_PDF, ID_EXP_DOCX, ID_EXP_TXT, ID_WF_RPDF, ID_WF_RDOCX, ID_WF_RTXT,
    ID_SIM_SIDESLIP = 8400, ID_SIM_GUST, ID_SIM_QUALITY, ID_SIM_LIVE, ID_SIM_SLICE, ID_SIM_LINES, ID_SIM_MESHES, ID_SIM_CLEAR,
    ID_START_PAGE = 9000, ID_NEW_BLANK, ID_NEW_FOLDER, ID_SKP_XY, ID_SKP_XZ, ID_SKP_YZ, ID_SKS_POLY, ID_SKS_RECT, ID_SKS_CIRCLE, ID_SKS_SLOT,
    ID_PD_PAD, ID_PD_POCKET, ID_PD_SHAFT, ID_PD_GROOVE, ID_PD_SLOT, ID_PD_HOLE, ID_MOD_CHAMFER, ID_MOD_DRAFT,
    ID_VIEW_ZEBRA, ID_VIEW_CURV, ID_VIEW_ANNOT, ID_VIEW_TREE, ID_VIEW_PROPS, ID_VIEW_PALETTE, ID_VIEW_FULL,
    ID_ENG_UNFOLD, ID_ENG_COMPOSITES, ID_ENG_CLASH, ID_ENG_RULES, ID_ENG_PARAMS, ID_ENG_RFLP, ID_ASM_RESERVE, ID_DMU_KIN, ID_ASM_PIPE, ID_ASM_HARNESS,
    ID_SIM_SETUP = 9050, ID_SIM_TURB, ID_PREFS, ID_SIM_OPT, ID_ROWEDT0 = 9800, ID_ROWCMB0 = 9850,
    ID_DRAFT = 16000, ID_DRAFT_VIEWS, ID_DRAFT_FLAT, ID_DRAFT_OPEN, ID_SHM_SKETCH, ID_SHM_CHANNEL, ID_SHM_ZBRACKET, ID_SHM_HAT, ID_SHM_LBRACKET, ID_FIN_PINN, ID_HQ_IMAGE, ID_LABS, ID_LABS_APP, ID_LABS_DESIGNER, ID_LABS_WB, ID_LABS_AIRFRANS,
    ID_VB_ZIN = 15000, ID_VB_ZOUT, ID_VB_DIMS, ID_VB_APPEAR, ID_OPT_APPLY = 15200, ID_APP0 = 15300,
    WM_MESHDONE = WM_APP + 1, WM_DETAILDONE = WM_APP + 30,
    NROWS = 26,
};

enum { ID_VIEW_SHADOW = 16530, ID_ASM_T0 = 16300, ID_SIM_SWEEP = 16500, ID_SIM_RECORD, ID_SIM_FLOWLINES, ID_ENV_SETUP = 16520, ID_ENV_CLEAR };  // the Assembly tools (asm.inc): ID_ASM_T0 + 0 ... 19
#include "tools.inc"
static std::string g_snap_png;  // when set, the next paint of the 3D view is saved there as a PNG
static void snap_view_png(const std::string& path);
static void open_labs(const std::string& app = "", const std::string& page = "");  // labs.inc: the browser apps inside Studio
// which app of the Labs window a command line of a browser app means ("" when it is not one)
static std::string labs_app_of(const std::string& args) {
    if (args.find("thermal.web.server") != std::string::npos) return "thermal_lab";
    if (args.find("thermal/app/main.py") != std::string::npos || args.find("thermal\\app\\main.py") != std::string::npos) return "thermal_app";
    if (args.find("cli designer") != std::string::npos || args.find("ui/designer.py") != std::string::npos) return "designer";
    if (args.find("ui/app.py") != std::string::npos) return "workbench";
    return "";
}
static bool hq_save_png(const std::string& path, double k, int ss, bool stamp);
#include "params.inc"
static bool g_show_palette = false, g_show_tree = true, g_show_props = false, g_fit_next = false;  // ribbon, tree and 3D view; the property panel on demand

// ------------------------------------------------------------------ application state
struct MeshResult { int gen; geo::Mesh mesh; geo::Mesh sel; geo::Props props; std::string error; double h; std::vector<float> creases;
                    std::vector<geo::TMesh> exact; geo::Mesh rest; std::string rest_text; bool rest_full = true;
                    std::vector<int> owners, rest_owners; std::vector<float> rest_creases; };  // owners: the part of each vertex (colours)
struct DetailResult { int gen, mgen; geo::Mesh mesh; std::vector<float> creases; geo::Box box; double h; std::vector<int> owners; };
static std::vector<int> mesh_owners(const geo::Model& m, const geo::Mesh& M);  // appearance.inc
static std::vector<int> g_mesh_owners, g_rest_owners, g_detail_owners;
// assemblies (asm.inc)
static bool g_asm_on = false;
static void asm_from_doc(); static void asm_fill_list(); static void asm_store(); static void asm_fit(); static void asm_tree_draw(DRAWITEMSTRUCT* d);
static std::vector<float> g_rest_creases;

static HWND g_main, g_list, g_view, g_status, g_json, g_lbl[NROWS], g_edt[NROWS], g_cmb[NROWS], g_apply, g_applyjson, g_btn[6];
static bool g_row_combo[NROWS];
static HFONT g_font;
static HDC g_hdc;
static HGLRC g_rc;
static js::Value g_doc;
static std::string g_path, g_root;
static std::vector<std::string> g_undo, g_redo;
static int g_sel = -1, g_gen = 0, g_building = 0, g_dirty_build = 0;
static geo::Mesh g_mesh, g_selmesh;
static geo::Props g_props;
static std::vector<geo::Seg> g_section;
static std::vector<float> g_colors;
static std::string g_result_label;
static double g_rmin = 0, g_rmax = 0;
static double g_h = 0.045, g_h_used = 0.045;  // chosen spacing, and the one the last mesh used (coarser for big designs)
static int g_res = 1;
static bool g_wire = false, g_cut = false, g_modified = false;
static int g_cutaxis = 1;
static double g_cutpos = 0;
static double g_yaw = -135, g_pitch = 25, g_dist = 30, g_tx = 8, g_ty = 0, g_tz = 0;
static int g_mx, g_my, g_mbtn = 0;
static std::vector<std::string> g_rowkeys;
static GLuint g_font_base = 0;
static PROCESS_INFORMATION g_job = {};
static std::string g_job_kind, g_job_dir, g_pending_results;
static int g_pending_view = -1;

// ------------------------------------------------------------------ palette (from the logo: navy, cyan, steel)
static const COLORREF C_PANEL = RGB(243, 244, 246), C_FIELD = RGB(255, 255, 255), C_INK = RGB(25, 28, 34), C_NAVY = RGB(22, 50, 92),
                      C_NAVY_DOWN = RGB(12, 30, 58), C_CYAN = RGB(0, 150, 214), C_MUTED = RGB(92, 110, 138);
static HBRUSH g_br_panel, g_br_field;

// ------------------------------------------------------------------ helpers
static std::string tostr(long v) { char b[32]; snprintf(b, sizeof b, "%ld", v); return b; }
static std::string fmtd(double v) { char b[64]; snprintf(b, sizeof b, "%.6g", v); return b; }

static void status(const std::string& s, int part = 0) { SendMessageA(g_status, SB_SETTEXTA, part, (LPARAM)s.c_str()); }

static js::Value& features() { return g_doc["features"]; }

static std::string read_edit(HWND h) {
    int n = GetWindowTextLengthA(h);
    std::string s((size_t)n + 1, '\0');
    GetWindowTextA(h, &s[0], n + 1);
    s.resize((size_t)n);
    return s;
}

static std::string crlf(const std::string& s) { std::string o; for (char c : s) { if (c == '\n') o += '\r'; o += c; } return o; }
#include "ui_gdip.inc"
#include "imgsave.inc"

static void push_undo() {
    g_undo.push_back(js::dump(g_doc, 0));
    if (g_undo.size() > 200) g_undo.erase(g_undo.begin());
    g_redo.clear();
    g_modified = true;
}

static std::string exe_dir() {
    char b[MAX_PATH]; GetModuleFileNameA(nullptr, b, MAX_PATH);
    std::string s(b); size_t p = s.find_last_of("\\/");
    return p == std::string::npos ? "." : s.substr(0, p);
}

static void update_title() {
    std::string t = "AeroSynapse Studio - " + (g_path.empty() ? std::string("untitled") : g_path) + (g_modified ? " *" : "");
    SetWindowTextA(g_main, t.c_str());
}

static std::string unique_id(const std::string& base) {
    js::Value& f = features();
    for (int k = 1;; ++k) {
        std::string id = base + (k > 1 ? "_" + tostr(k) : std::string());
        bool used = false;
        for (size_t i = 0; i < f.size(); ++i) if (f.at(i).str("id", "") == id) used = true;
        if (!used) return id;
    }
}

// ------------------------------------------------------------------ meshing in the background
// crisp edge lines: every mesh edge where the surface turns by more than `deg` (wing edges, fin roots, steps, holes)
static std::vector<float> crease_lines(const geo::Mesh& M, double deg, bool boundaries) {
    std::vector<float> out;
    size_t nt = M.t.size() / 3;
    if (!nt) return out;
    std::vector<V3> fn(nt);
    for (size_t t = 0; t < nt; ++t) {
        const V3 &a = M.v[(size_t)M.t[3 * t]], &b = M.v[(size_t)M.t[3 * t + 1]], &c = M.v[(size_t)M.t[3 * t + 2]];
        V3 n = geo::cross(b - a, c - a); double l = geo::len(n); fn[t] = l > 0 ? n / l : V3();
    }
    struct Ed { int a, b; unsigned t; };
    std::vector<Ed> es; es.reserve(nt * 3);
    for (size_t t = 0; t < nt; ++t)
        for (int e = 0; e < 3; ++e) { int a = M.t[3 * t + (size_t)e], b = M.t[3 * t + (size_t)((e + 1) % 3)]; if (a > b) std::swap(a, b); es.push_back({a, b, (unsigned)t}); }
    std::sort(es.begin(), es.end(), [](const Ed& x, const Ed& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; });
    double cmax = std::cos(deg * geo::PI / 180);
    auto seg = [&](int a, int b) { const V3 &p = M.v[(size_t)a], &q = M.v[(size_t)b]; float s[6] = {(float)p.x, (float)p.y, (float)p.z, (float)q.x, (float)q.y, (float)q.z}; out.insert(out.end(), s, s + 6); };
    for (size_t i = 0; i < es.size();) {
        size_t j = i + 1;
        while (j < es.size() && es[j].a == es[i].a && es[j].b == es[i].b) ++j;
        if (j - i == 2) { if (geo::dot(fn[es[i].t], fn[es[i + 1].t]) < cmax) seg(es[i].a, es[i].b); }
        else if (j - i == 1 && boundaries) seg(es[i].a, es[i].b);
        i = j;
    }
    return out;
}
static js::Value display_doc(const std::string& text, double h) { js::Value d = js::parse(text); d["display_h"] = h; return d; }

struct BuildJob { int gen; std::string text; int sel; double h; };

static unsigned __stdcall build_thread(void* p) {
    std::unique_ptr<BuildJob> job((BuildJob*)p);
    MeshResult* r = new MeshResult();
    r->gen = job->gen; r->h = job->h;
    try {
        geo::Model m0; m0.load(js::parse(job->text));
        if (m0.bounds.valid()) {  // large designs: a coarser grid, so it fits in memory (about 12 million cells); zooming in refines it
            V3 e = m0.bounds.hi - m0.bounds.lo;
            job->h = std::max(job->h, std::cbrt((e.x + 0.2) * (e.y + 0.2) * (e.z + 0.2) / 1.2e7));
            r->h = job->h;
        }
        geo::Model m; m.load(display_doc(job->text, job->h));  // thin parts drawn at a visible minimum thickness
        if (!m.errors.empty()) r->error = m.errors[0];
        if (m.bounds.valid()) {
            geo::Grid g = geo::sample(m, job->h);
            r->mesh = geo::surface_nets(m, g, 1);
            r->props = geo::properties(r->mesh, g);
            r->creases = crease_lines(r->mesh, 38.0, false);
            geo::split_sharp(r->mesh, 30.0);  // crisp shading on both sides of every edge (after the crease lines and properties)
            r->owners = mesh_owners(m, r->mesh);
            // exact surfaces where the definition allows it; the sampled mesh only for the rest (cut, deformed, other kinds)
            js::Value rd = display_doc(job->text, job->h);
            std::vector<int> exact_src;
            for (size_t k = 0; k < m.feats.size(); ++k) {
                if (!geo::exact_candidate(m.feats[k])) continue;
                geo::TMesh t = geo::tessellate(m.feats[k]);
                if (t.t.empty() || !geo::exact_ok(m, k, t)) continue;
                t.feature = m.feats[k].src;
                exact_src.push_back(t.feature);
                r->exact.push_back(std::move(t));
            }
            if (!exact_src.empty()) {
                for (int s : exact_src) rd["features"].at((size_t)s)["enabled"] = false;
                r->rest_full = false;
                r->rest_text = js::dump(rd, 0);
                geo::Model mr; mr.load(rd);
                if (mr.bounds.valid()) {
                    geo::Grid gr = geo::sample(mr, job->h); r->rest = geo::surface_nets(mr, gr, 2);
                    r->rest_creases = crease_lines(r->rest, 38.0, false);
                    geo::split_sharp(r->rest, 30.0);
                    r->rest_owners = mesh_owners(mr, r->rest);
                }
            }
            if (job->sel >= 0) {
                // the selected feature on its own, for the highlight
                js::Value d = js::parse(job->text);
                const js::Value& fl = d["features"];
                if (job->sel < (int)fl.size()) {
                    js::Value one = js::Value::object(); js::Value arr = js::Value::array();
                    js::Value f = fl.at((size_t)job->sel).clone(); f["op"] = "add"; f["enabled"] = true;
                    arr.push(f); one["features"] = arr;
                    one["display_h"] = job->h * 1.5;
                    geo::Model ms; ms.load(one);
                    if (ms.errors.empty() && ms.bounds.valid()) {  // only where the design is: a half-space or a long cut has no end of its own
                        geo::Box region = ms.bounds, mb = m.bounds.pad(0.1 * geo::len(m.bounds.hi - m.bounds.lo) + job->h);
                        region.lo = V3(std::max(region.lo.x, mb.lo.x), std::max(region.lo.y, mb.lo.y), std::max(region.lo.z, mb.lo.z));
                        region.hi = V3(std::min(region.hi.x, mb.hi.x), std::min(region.hi.y, mb.hi.y), std::min(region.hi.z, mb.hi.z));
                        if (region.hi.x > region.lo.x && region.hi.y > region.lo.y && region.hi.z > region.lo.z) {
                            geo::Grid gs = geo::sample(ms, job->h * 1.5, &region); r->sel = geo::surface_nets(ms, gs, 0); geo::split_sharp(r->sel, 30.0);
                        }
                    }
                }
            }
        }
    } catch (std::exception& e) { r->error = e.what(); }
    PostMessageA(g_main, WM_MESHDONE, 0, (LPARAM)r);
    return 0;
}

static void rebuild() {
    if (g_asm_on) { g_mesh = geo::Mesh(); g_props = geo::Props(); if (g_view) InvalidateRect(g_view, nullptr, FALSE); return; }  // components are meshed one by one (asm.inc)
    if (g_building) { g_dirty_build = 1; return; }
    g_building = 1; g_dirty_build = 0;
    BuildJob* j = new BuildJob{++g_gen, js::dump(g_doc, 0), g_sel, g_h};
    HANDLE t = (HANDLE)_beginthreadex(nullptr, 0, build_thread, j, 0, nullptr);
    if (t) CloseHandle(t);
    status("building surface...", 1);
}

static void compute_section() {
    g_section.clear();
    if (!g_cut) return;
    try {
        geo::Model m; m.load(g_doc);
        if (!m.bounds.valid()) return;
        geo::Box e;
        g_section = geo::slice(m, g_cutaxis, g_cutpos, std::max(g_h * 0.5, 0.01), e);
    } catch (...) {}
}

// ------------------------------------------------------------------ feature list and properties
static void fill_list() {
    asm_from_doc();  // an assembly document lists its components and mates instead
    if (g_asm_on) { asm_fill_list(); return; }
    SendMessageA(g_list, LB_RESETCONTENT, 0, 0);
    js::Value& f = features();
    for (size_t i = 0; i < f.size(); ++i) {
        const js::Value& o = f.at(i);
        std::string op = o.str("op", "add");
        std::string s = std::string(o.flag("enabled", true) ? "" : "(off) ") + (op == "cut" ? "[-] " : op == "intersect" ? "[&] " : "[+] ") + o.str("id", "?") + "  : " + o.str("type", "?");
        if (o.flag("mirror", false)) s += "  |mirror";
        if (o.num("shell", 0) > 0) s += "  |hollow";
        if (o.find("pattern")) s += "  |pattern";
        SendMessageA(g_list, LB_ADDSTRING, 0, (LPARAM)s.c_str());
    }
    if (g_sel >= (int)f.size()) g_sel = (int)f.size() - 1;
    SendMessageA(g_list, LB_SETCURSEL, g_sel, 0);
}

static std::string value_text(const js::Value& v) {
    if (v.kind == js::Value::Num) return fmtd(v.n);
    if (v.kind == js::Value::Str) return v.s;
    if (v.kind == js::Value::Bool) return v.b ? "true" : "false";
    if (v.kind == js::Value::Arr) {
        bool flat = true; for (size_t i = 0; i < v.size(); ++i) if (v.at(i).kind != js::Value::Num) flat = false;
        if (flat) { std::string s; for (size_t i = 0; i < v.size(); ++i) { if (i) s += ", "; s += fmtd(v.at(i).n); } return s; }
    }
    return js::dump(v, 0);
}

static void card_refresh();
static void props_layout();
static int g_rows_used = 0;
static void fill_props() {
    g_rowkeys.clear();
    js::Value& f = features();
    bool has = g_sel >= 0 && g_sel < (int)f.size();
    int r = 0;
    if (has) {
        const js::Value& o = f.at((size_t)g_sel);
        for (auto& kv : *o.o) {
            if (r >= NROWS) break;
            g_rowkeys.push_back(kv.first);
            const ParamHelp* ph = param_help(kv.first);
            std::string lab = kv.first + (ph && ph->unit[0] ? std::string(" [") + ph->unit + "]" : "");
            SetWindowTextA(g_lbl[r], lab.c_str());
            std::vector<std::string> ch = ph ? split_csv(ph->choices) : std::vector<std::string>();
            bool boolish = !ch.empty() && ch[0] == "true";
            g_row_combo[r] = !ch.empty() && (boolish ? kv.second.kind == js::Value::Bool : kv.second.kind == js::Value::Str);
            if (g_row_combo[r]) {
                SendMessageA(g_cmb[r], CB_RESETCONTENT, 0, 0);
                std::string cur = value_text(kv.second);
                int sel = -1;
                for (size_t q = 0; q < ch.size(); ++q) { SendMessageA(g_cmb[r], CB_ADDSTRING, 0, (LPARAM)ch[q].c_str()); if (ch[q] == cur) sel = (int)q; }
                if (sel < 0) { SendMessageA(g_cmb[r], CB_ADDSTRING, 0, (LPARAM)cur.c_str()); sel = (int)ch.size(); }
                SendMessageA(g_cmb[r], CB_SETCURSEL, sel, 0);
            } else SetWindowTextA(g_edt[r], value_text(kv.second).c_str());
            EnableWindow(g_edt[r], kv.first != "type");
            ShowWindow(g_lbl[r], g_show_props ? SW_SHOW : SW_HIDE);
            ShowWindow(g_edt[r], g_show_props && !g_row_combo[r] ? SW_SHOW : SW_HIDE); ShowWindow(g_cmb[r], g_show_props && g_row_combo[r] ? SW_SHOW : SW_HIDE);
            ++r;
        }
        SetWindowTextA(g_json, crlf(js::dump(o, 2)).c_str());
    } else SetWindowTextA(g_json, "");
    g_rows_used = r;
    g_param_key.clear();
    for (; r < NROWS; ++r) { ShowWindow(g_lbl[r], SW_HIDE); ShowWindow(g_edt[r], SW_HIDE); ShowWindow(g_cmb[r], SW_HIDE); g_row_combo[r] = false; }
    props_layout();
    card_refresh();
}

static bool apply_rows() {
    js::Value& f = features();
    if (g_sel < 0 || g_sel >= (int)f.size()) return false;
    js::Value o = f.at((size_t)g_sel).clone();
    for (size_t r = 0; r < g_rowkeys.size(); ++r) {
        const std::string& k = g_rowkeys[r];
        js::Value* cur = o.find(k);
        if (!cur || k == "type") continue;
        std::string t;
        if (g_row_combo[r]) { char b[256] = ""; int i = (int)SendMessageA(g_cmb[r], CB_GETCURSEL, 0, 0); if (i >= 0) SendMessageA(g_cmb[r], CB_GETLBTEXT, i, (LPARAM)b); t = b; }
        else t = read_edit(g_edt[r]);
        try {
            if (cur->kind == js::Value::Num) { char* e; double v = strtod(t.c_str(), &e); if (e == t.c_str()) throw std::runtime_error(k + " must be a number"); *cur = js::Value(v); }
            else if (cur->kind == js::Value::Str) *cur = js::Value(t);
            else if (cur->kind == js::Value::Bool) *cur = js::Value(t == "true" || t == "1" || t == "yes");
            else if (cur->kind == js::Value::Arr && !t.empty() && t[0] != '[') *cur = js::parse("[" + t + "]");
            else *cur = js::parse(t);
        } catch (std::exception& e) { MessageBoxA(g_main, e.what(), "Property not applied", MB_ICONWARNING); return false; }
    }
    push_undo();
    f.at((size_t)g_sel) = o;
    fill_list(); fill_props(); rebuild(); compute_section(); update_title();
    return true;
}

static void apply_json() {
    js::Value& f = features();
    if (g_sel < 0 || g_sel >= (int)f.size()) return;
    try {
        js::Value o = js::parse(read_edit(g_json));
        if (!o.is_obj()) throw std::runtime_error("a feature must be a JSON object");
        js::Value test = g_doc.clone(); test["features"].at((size_t)g_sel) = o;
        geo::Model m; m.load(test);
        if (!m.errors.empty()) throw std::runtime_error(m.errors[0]);
        push_undo();
        f.at((size_t)g_sel) = o;
    } catch (std::exception& e) { MessageBoxA(g_main, e.what(), "JSON not applied", MB_ICONWARNING); return; }
    fill_list(); fill_props(); rebuild(); compute_section(); update_title();
}

// ------------------------------------------------------------------ document
static js::Value default_design() {
    js::Value d = js::Value::object();
    d["name"] = "new design"; d["units"] = "m";
    js::Value f = js::Value::array();
    f.push(geo::template_for("fuselage"));
    js::Value w = geo::template_for("wing"); f.push(w);
    f.push(geo::template_for("tail"));
    js::Value fin = geo::template_for("fin"); fin["dihedral"] = 90.0; fin["mirror"] = false; fin["translate"] = js::parse("[12.0, 0, 0.4]"); f.push(fin);
    d["features"] = f;
    return d;
}

static void set_doc(const js::Value& d, const std::string& path) {
    g_doc = d;
    if (!g_doc.find("features")) g_doc["features"] = js::Value::array();
    g_path = path; g_undo.clear(); g_redo.clear(); g_sel = features().size() ? 0 : -1; g_modified = false; g_fit_next = true;
    g_colors.clear(); g_result_label.clear();
    fill_list(); fill_props(); rebuild(); update_title();
}

static bool ask_path(bool save, const char* filter, const char* ext, std::string& out) {
    char buf[MAX_PATH] = {0};
    if (!out.empty()) strncpy(buf, out.c_str(), MAX_PATH - 1);
    OPENFILENAMEA o; memset(&o, 0, sizeof o);
    o.lStructSize = sizeof o; o.hwndOwner = g_main; o.lpstrFilter = filter; o.lpstrFile = buf; o.nMaxFile = MAX_PATH; o.lpstrDefExt = ext;
    o.Flags = save ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST;
    BOOL ok = save ? GetSaveFileNameA(&o) : GetOpenFileNameA(&o);
    if (ok) out = buf;
    return ok != 0;
}

static void save(bool as) {
    std::string p = g_path;
    if (as || p.empty()) if (!ask_path(true, "Design (*.json)\0*.json\0All files\0*.*\0", "json", p)) return;
    asm_store();
    try { js::write_file(p, js::dump(g_doc, 2) + "\n"); g_path = p; g_modified = false; update_title(); status("saved " + p); }
    catch (std::exception& e) { MessageBoxA(g_main, e.what(), "Save failed", MB_ICONERROR); }
}

static void export_mesh(bool obj) {
    std::string p;
    if (!ask_path(true, obj ? "Wavefront OBJ (*.obj)\0*.obj\0" : "STL (*.stl)\0*.stl\0", obj ? "obj" : "stl", p)) return;
    try {
        geo::Model m; m.load(g_doc);
        geo::Grid g = geo::sample(m, std::min(g_h, 0.03));
        geo::Mesh M = geo::surface_nets(m, g, 2);
        if (obj) geo::write_obj(M, p); else geo::write_stl(M, p, g_doc.str("name", "design"));
        status("exported " + p + " (" + tostr((int)(M.t.size() / 3)) + " triangles)");
    } catch (std::exception& e) { MessageBoxA(g_main, e.what(), "Export failed", MB_ICONERROR); }
}

static void export_svg() {
    if (!g_cut) {  // no section yet: cut through the middle along x and use that
        g_cut = true; g_cutaxis = 0; g_cutpos = 0.5 * (g_props.lo.x + g_props.hi.x); compute_section(); InvalidateRect(g_view, nullptr, FALSE);
        status("section cut turned on at x = " + fmtd(g_cutpos) + " m (move it with [ and ])");
    }
    std::string p;
    if (!ask_path(true, "SVG (*.svg)\0*.svg\0", "svg", p)) return;
    FILE* f = fopen(p.c_str(), "w");
    if (!f) return;
    double lo0 = 1e9, lo1 = 1e9, hi0 = -1e9, hi1 = -1e9;
    for (auto& s : g_section) { lo0 = std::min(lo0, std::min(s.u0, s.u1)); hi0 = std::max(hi0, std::max(s.u0, s.u1)); lo1 = std::min(lo1, std::min(s.v0, s.v1)); hi1 = std::max(hi1, std::max(s.v0, s.v1)); }
    fprintf(f, "<svg xmlns='http://www.w3.org/2000/svg' viewBox='%g %g %g %g'><g stroke='#1a5fb4' stroke-width='%g' fill='none'>\n", lo0 - 0.1, -hi1 - 0.1, hi0 - lo0 + 0.2, hi1 - lo1 + 0.2, (hi0 - lo0) / 400);
    for (auto& s : g_section) fprintf(f, "<line x1='%.5f' y1='%.5f' x2='%.5f' y2='%.5f'/>\n", s.u0, -s.v0, s.u1, -s.v1);
    fprintf(f, "</g></svg>\n");
    fclose(f);
    status("section written to " + p);
}

// ------------------------------------------------------------------ editing operations
static void open_pad(int idx);
static int g_cmd_id = 0;  // the command being run (modifier tools open the pad, keyboard nudges do not)
static bool pad_modifier(int id) {
    const int ids[] = {ID_MOD_SHELL, ID_MOD_PATTERN, ID_MOD_MIRROR, ID_MOD_BLEND, ID_MOD_TWIST, ID_MOD_TAPER, ID_MOD_BEND, ID_MOD_SHEAR, ID_MOD_STRETCH,
                       ID_MOD_PIVOT, ID_MOD_INFILL, ID_MOD_CHAMFER, ID_MOD_DRAFT};
    for (int k : ids) if (k == id) return true;
    return false;
}

static void insert(js::Value f) {
    push_undo();
    f["id"] = unique_id(f.str("id", f.str("type", "feature")));
    features().push(f);
    g_sel = (int)features().size() - 1;
    fill_list(); fill_props(); rebuild(); update_title();
    open_pad(g_sel);  // every new feature: its pad, with a live preview; Cancel takes it away again
}

static void modify_selected(const std::function<void(js::Value&)>& fn) {
    bool pad = pad_modifier(g_cmd_id);  // read it now: the property refresh below sends nested notifications
    js::Value& f = features();
    if (g_sel < 0 || g_sel >= (int)f.size()) { status("select a feature first"); return; }
    push_undo();
    fn(f.at((size_t)g_sel));
    fill_list(); fill_props(); rebuild(); compute_section(); update_title();
    if (pad) open_pad(g_sel);
}

static void move_selected(int d) {
    js::Value& f = features();
    int j = g_sel + d;
    if (g_sel < 0 || j < 0 || j >= (int)f.size()) return;
    push_undo();
    std::swap(f.at((size_t)g_sel), f.at((size_t)j));
    g_sel = j;
    fill_list(); fill_props(); rebuild(); update_title();
}

static void nudge(int axis, double d) {
    modify_selected([&](js::Value& o) {
        std::vector<double> t = o.nums("translate"); if (t.size() < 3) t = {0, 0, 0};
        t[(size_t)axis] += d;
        js::Value a = js::Value::array(); for (double v : t) a.push(v); o["translate"] = a;
    });
}

static void rotate_sel(int axis, double deg) {
    modify_selected([&](js::Value& o) {
        std::vector<double> r = o.nums("rotate"); if (r.size() < 3) r = {0, 0, 0};
        r[(size_t)axis] += deg;
        js::Value a = js::Value::array(); for (double v : r) a.push(v); o["rotate"] = a;
    });
}

static void undo(bool redo_) {
    std::vector<std::string>& from = redo_ ? g_redo : g_undo;
    std::vector<std::string>& to = redo_ ? g_undo : g_redo;
    if (from.empty()) return;
    to.push_back(js::dump(g_doc, 0));
    g_doc = js::parse(from.back()); from.pop_back();
    if (g_sel >= (int)features().size()) g_sel = (int)features().size() - 1;
    g_modified = true;
    fill_list(); fill_props(); rebuild(); compute_section(); update_title();
}

// ------------------------------------------------------------------ results on the surface
static void colour_by(const std::vector<V3>& pts, const std::vector<double>& val, const std::string& label) {
    g_colors.clear();
    if (pts.empty() || g_mesh.v.empty()) return;
    // uniform hash grid for nearest-point lookup
    geo::Box b; for (auto& p : pts) b.add(p);
    double cell = std::max(0.05, std::cbrt((b.hi.x - b.lo.x + 1e-3) * (b.hi.y - b.lo.y + 1e-3) * (b.hi.z - b.lo.z + 1e-3) / std::max<size_t>(pts.size(), 1)) * 2);
    std::map<long long, std::vector<int>> H;
    auto key = [&](int i, int j, int k) { return ((long long)i * 73856093LL) ^ ((long long)j * 19349663LL) ^ ((long long)k * 83492791LL); };
    for (size_t i = 0; i < pts.size(); ++i) H[key((int)std::floor(pts[i].x / cell), (int)std::floor(pts[i].y / cell), (int)std::floor(pts[i].z / cell))].push_back((int)i);
    std::vector<double> sorted = val; std::sort(sorted.begin(), sorted.end());
    g_rmin = sorted[(size_t)(0.02 * (sorted.size() - 1))]; g_rmax = sorted[(size_t)(0.98 * (sorted.size() - 1))];
    if (g_rmax <= g_rmin) g_rmax = g_rmin + 1e-6;
    for (auto& v : g_mesh.v) {
        int ci = (int)std::floor(v.x / cell), cj = (int)std::floor(v.y / cell), ck = (int)std::floor(v.z / cell);
        double best = 1e300; int bi = -1;
        for (int r = 1; r <= 3 && bi < 0; ++r)
            for (int a = -r; a <= r; ++a) for (int bb = -r; bb <= r; ++bb) for (int c = -r; c <= r; ++c) {
                auto it = H.find(key(ci + a, cj + bb, ck + c)); if (it == H.end()) continue;
                for (int q : it->second) { V3 d = pts[(size_t)q] - v; double dd = geo::dot(d, d); if (dd < best) { best = dd; bi = q; } }
            }
        double t = bi < 0 ? 0.5 : geo::clampd((val[(size_t)bi] - g_rmin) / (g_rmax - g_rmin), 0, 1);
        // blue - white - red
        float r = (float)(t < 0.5 ? 2 * t : 1.0), g = (float)(t < 0.5 ? 2 * t : 2 * (1 - t)), bl = (float)(t < 0.5 ? 1.0 : 2 * (1 - t));
        g_colors.push_back(0.15f + 0.85f * r); g_colors.push_back(0.15f + 0.85f * g); g_colors.push_back(0.15f + 0.85f * bl);
    }
    g_result_label = label;
}

static std::vector<std::vector<double>> g_surf;  // rows of surface.csv: x y z nx ny nz cp cp_rms cf T_wall_K yplus separated area
static int g_surf_col = 6;

static void show_surface_column(int col) {
    if (g_surf.empty()) { status("load surface results first (Analysis menu)"); return; }
    static const char* names[13] = {"x", "y", "z", "nx", "ny", "nz", "pressure coefficient Cp", "pressure fluctuation (rms Cp)", "skin friction Cf", "wall temperature, K", "y+", "reversed flow", "area"};
    std::vector<V3> pts; std::vector<double> val;
    for (auto& r : g_surf) {
        if ((int)r.size() <= col) continue;
        pts.push_back(V3(r[0], r[1], r[2])); val.push_back(r[(size_t)col]);
        pts.push_back(V3(r[0], -r[1], r[2])); val.push_back(r[(size_t)col]);  // half-model runs: mirror
    }
    g_surf_col = col;
    colour_by(pts, val, names[col]);
    InvalidateRect(g_view, nullptr, FALSE);
    status(std::string(names[col]) + ": " + fmtd(g_rmin) + " to " + fmtd(g_rmax) + " over " + tostr((long)g_surf.size()) + " wall cells");
}

static void load_surface_csv(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { MessageBoxA(g_main, ("cannot open " + path).c_str(), "Results", MB_ICONWARNING); return; }
    char line[1024];
    if (!fgets(line, sizeof line, f)) { fclose(f); return; }
    g_surf.clear();
    while (fgets(line, sizeof line, f)) {
        std::vector<double> v(13, 0.0);
        int n = sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9], &v[10], &v[11], &v[12]);
        if (n >= 7) { v.resize((size_t)n); g_surf.push_back(v); }
    }
    fclose(f);
    show_surface_column(g_surf_col);
}

// ------------------------------------------------------------------ airfoil import: Selig or Lednicer .dat -> wing section table
static bool airfoil_section(const std::string& path, js::Value& sec, std::string& name) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[512];
    std::vector<std::pair<double, double>> pts;
    name = "airfoil";
    bool first = true;
    while (fgets(line, sizeof line, f)) {
        double x, y;
        if (sscanf(line, "%lf %lf", &x, &y) == 2 && x <= 1.5 && x >= -0.5) pts.push_back({x, y});
        else if (first) { name = line; while (!name.empty() && (name.back() == '\n' || name.back() == '\r' || name.back() == ' ')) name.pop_back(); }
        first = false;
    }
    fclose(f);
    if (pts.size() < 10) return false;
    // split at the leading edge (smallest x); one side is upper, the other lower
    size_t le = 0;
    for (size_t i = 1; i < pts.size(); ++i) if (pts[i].first < pts[le].first) le = i;
    std::vector<std::pair<double, double>> a(pts.begin(), pts.begin() + (long)le + 1), b(pts.begin() + (long)le, pts.end());
    std::reverse(a.begin(), a.end());
    auto interp = [](const std::vector<std::pair<double, double>>& c, double u) {
        for (size_t i = 0; i + 1 < c.size(); ++i) {
            double x0 = c[i].first, x1 = c[i + 1].first;
            if ((u >= std::min(x0, x1)) && (u <= std::max(x0, x1)) && x1 != x0) return c[i].second + (c[i + 1].second - c[i].second) * (u - x0) / (x1 - x0);
        }
        return u <= c.front().first ? c.front().second : c.back().second;
    };
    double x0 = pts[le].first, x1 = std::max(a.back().first, b.back().first), chord = std::max(x1 - x0, 1e-6);
    js::Value U = js::Value::array(), T = js::Value::array(), Z = js::Value::array();
    for (int k = 0; k <= 60; ++k) {
        double u = 0.5 * (1 - std::cos(geo::PI * k / 60)), xx = x0 + u * chord;
        double ya = interp(a, xx), yb = interp(b, xx);
        U.push(u); T.push(std::fabs(ya - yb) / chord); Z.push(0.5 * (ya + yb) / chord);
    }
    sec = js::Value::object(); sec["u"] = U; sec["t"] = T; sec["zc"] = Z;
    return true;
}

static void insert_airfoil_wing() {
    char buf[MAX_PATH] = {0};
    OPENFILENAMEA o; memset(&o, 0, sizeof o);
    o.lStructSize = sizeof o; o.hwndOwner = g_main; o.lpstrFilter = "Airfoil coordinates (*.dat)\0*.dat\0All files\0*.*\0"; o.lpstrFile = buf; o.nMaxFile = MAX_PATH; o.Flags = OFN_FILEMUSTEXIST;
    std::string init = g_root + "\\results\\shapeopt\\naca2412";
    o.lpstrInitialDir = init.c_str();
    if (!GetOpenFileNameA(&o)) return;
    js::Value sec; std::string nm;
    if (!airfoil_section(buf, sec, nm)) { MessageBoxA(g_main, "Could not read airfoil coordinates from that file (expected x y pairs, Selig or Lednicer order).", "Airfoil", MB_ICONWARNING); return; }
    js::Value w = geo::template_for("wing");
    w["id"] = "airfoil_wing"; w["name"] = nm; w["profile"] = "table"; w["section"] = sec;
    insert(w);
    status("wing built from " + std::string(buf));
}

// ------------------------------------------------------------------ external analysis
static std::string python() {
    char b[MAX_PATH];
    if (GetEnvironmentVariableA("AEROSYNAPSE_PYTHON", b, MAX_PATH)) return b;
    std::string venv = g_root + "\\.venv\\Scripts\\python.exe";
    if (GetFileAttributesA(venv.c_str()) != INVALID_FILE_ATTRIBUTES) return venv;
    return "python";
}

static bool launch(const std::string& cmd, const std::string& kind, const std::string& dir) {
    if (g_job.hProcess) { MessageBoxA(g_main, "An analysis is still running.", "Busy", MB_ICONINFORMATION); return false; }
    std::string pp = g_root + "\\src";
    SetEnvironmentVariableA("PYTHONPATH", pp.c_str());
    STARTUPINFOA si; memset(&si, 0, sizeof si); si.cb = sizeof si;
    std::string c = cmd;
    if (!CreateProcessA(nullptr, &c[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, g_root.c_str(), &si, &g_job)) {
        MessageBoxA(g_main, ("could not start: " + cmd).c_str(), "Analysis", MB_ICONERROR); return false;
    }
    g_job_kind = kind; g_job_dir = dir;
    SetTimer(g_main, 1, 1000, nullptr);
    status("running " + kind + "...", 1);
    return true;
}

static void make_dirs(const std::string& d) {  // every missing folder along the path
    for (size_t i = 3; i <= d.size(); ++i) if (i == d.size() || d[i] == '\\' || d[i] == '/') CreateDirectoryA(d.substr(0, i).c_str(), nullptr);
}
static std::string work_dir(const std::string& name) {
    std::string d = g_root + "\\results\\studio\\" + name;
    make_dirs(d);
    return d;
}

static void quick_analysis() {
    std::string d = work_dir("analysis");
    js::write_file(d + "\\design.json", js::dump(g_doc, 2));
    launch("\"" + python() + "\" -m aerosynapse.cli cad-analyse \"" + d + "\\design.json\" --out \"" + d + "\\analysis.txt\"", "quick analysis", d);
}


static bool sim_job_done(DWORD code);
static void cfd_finished(const std::string& dir);

static std::string g_cfd_quality = "draft";
static bool g_live = false;
static void run_cfd_case(const char* kase, const std::string& quality) {
    std::string d = work_dir(std::string("cfd_") + kase + "_" + quality);
    js::write_file(d + "\\design.json", js::dump(g_doc, 2));
    launch("\"" + python() + "\" -m aerosynapse.cli simulate cfd \"" + d + "\\design.json\" --case " + kase + " --quality " + quality + " --out \"" + d + "\"",
           quality == "live" ? std::string("simulation live") : std::string("simulation cfd ") + kase, d);
}

static void job_done() {
    DWORD code = 0;
    GetExitCodeProcess(g_job.hProcess, &code);
    CloseHandle(g_job.hProcess); CloseHandle(g_job.hThread);
    memset(&g_job, 0, sizeof g_job);
    KillTimer(g_main, 1);
    status("", 1);
    if (sim_job_done(code)) return;
    if (g_job_kind.compare(0, 4, "CFD ") == 0) cfd_finished(g_job_dir);
    if (g_job_kind == "quick analysis") {
        std::string t;
        try { t = js::read_file(g_job_dir + "\\analysis.txt"); } catch (...) { t = "the analysis wrote no report (exit code " + tostr((int)code) + ")"; }
        MessageBoxA(g_main, crlf(t).c_str(), "Quick aerodynamic analysis", MB_ICONINFORMATION);
    } else {
        std::string res;
        try { res = js::read_file(g_job_dir + "\\result.json"); } catch (...) {}
        if (res.empty()) { MessageBoxA(g_main, ("the solver failed (exit code " + tostr((int)code) + "); see " + g_job_dir).c_str(), "CFD", MB_ICONERROR); return; }
        load_surface_csv(g_job_dir + "\\surface.csv");
        js::Value r = js::parse(res);
        const js::Value* cc = r.find("coefficients");
        std::string msg = g_job_kind + "\n\n";
        if (cc) for (auto& kv : *cc->o) msg += kv.first + " = " + fmtd(kv.second.n) + "\n";
        msg += "\nseparated area " + fmtd(100 * r.num("separated_fraction", 0)) + " %, hottest wall " + fmtd(r.num("T_wall_max_K", 0)) + " K\n";
        msg += "grid " + fmtd(r.num("cells", 0)) + " cells, " + fmtd(r.num("seconds", 0)) + " s\nFiles: " + g_job_dir;
        MessageBoxA(g_main, msg.c_str(), "CFD result", MB_ICONINFORMATION);
    }
}

// ------------------------------------------------------------------ Optimisation and simulation centre
struct Task { const char* group; const char* name; const char* args; const char* about; };
static const Task TASKS[] = {
    {"Airfoil", "Airfoil shape optimisation", "-m aerosynapse.cli shapeopt --shape naca2412 --profile quick",
     "CMA-ES over CST sections, planform and twist across laminar, turbulent and extreme flow states (NeuralFoil, viscous-inviscid check). Shapes: nacaXXXX, circle, square, ellipse:0.3 or a .dat file path."},
    {"Airfoil", "Airfoil check with XFOIL", "scripts/shapeopt_xfoil.py --shape naca2412", "Runs XFOIL on the optimised section over the hold-out flow states."},
    {"Airfoil", "Airfoil verification (viscous-inviscid and Navier-Stokes)", "scripts/shapeopt_verify.py --shape naca2412 --ns quick", "Independent checks of an optimised section."},
    {"Topology", "Topology optimisation of a 2D body", "scripts/shapeopt_topology.py --shape square --profile quick", "Density topology optimisation with Brinkman Navier-Stokes and an exact adjoint. Shapes as for airfoils."},
    {"Aircraft", "Whole aircraft study (airliner and supersonic)", "-m aerosynapse.cli aircraft --profile quick", "Tube-and-wing, blended wing body, flying wing and free-form bodies against their missions."},
    {"Aircraft", "Fighter study (five families)", "-m aerosynapse.cli fighter --profile quick", "Every topology family against every flight situation, with materials and constraints."},
    {"3D flow", "3D flow of this design", "-m aerosynapse.cli cfd {design} --mach 0.85 --alpha 2 --altitude 11000 --cells 36 --steps 1200 --out results/studio/cfd_centre",
     "Compressible flow with turbulence model and wall model. Change Mach, incidence, sideslip (--beta), gust (--gust 0.05) or inflow turbulence (--tu 0.05)."},
    {"3D flow", "3D flow verification suite", "-m aerosynapse.cli cfd-verify --profile quick", "Shock tube, spheroid, cone, slab friction, thin wing, delta wing and Taylor-Green vortex."},
    {"3D flow", "Fighter 3D flow comparison", "scripts/fighter_cfd.py --profile quick", "Reference against the best fighter in several flight states, beside the fast model."},
    {"Thermal", "Skin heating with the PINN", "-m aerosynapse.cli thermal --profile quick", "Trains the parametric PINN for skin panels and applies it zone by zone with the selected materials."},
    {"Thermal", "THERMAL LAB (merged PINN project, inside Studio)", "-m aerosynapse.thermal.web.server --port 8766",
     "The whole convection-diffusion PINN project: verified solvers, parametric PINNs, inverse identification of alpha, k and h, real data."},
    {"Thermal", "Fin PINN study (every optimisation setting)", "-m aerosynapse.thermal.fin_pinn --out results/studio/fin_pinn --profile quick",
     "Runs nine optimisers on the cooling fin, keeps every design they evaluate as training data, trains a parametric PINN on it and analyses it per optimiser in 2D and 3D."},
    {"Thermal", "Parametric 2D PINN study", "-m aerosynapse.thermal.experiments.p4_5_parametric --profile smoke", "One network for a range of Re, Pr and Bi (Graetz channel, cavity)."},
    {"Thermal", "Inverse identification of alpha, k, h", "-m aerosynapse.thermal.experiments.p5_2_pinn_inverse --tag studio_quick --set pinn.adam.epochs=800 --set pinn.lbfgs.iterations=100 --set pinn.two_stage.fit_epochs=300 --set pinn.two_stage.param_epochs=200 --set pinn.staged.warmup_epochs=200", "Recover material properties from sparse noisy temperatures."},
    {"Prediction", "AirfRANS flow surrogates (the Kaggle data in the project)", "-m aerosynapse.ml.airfrans_graph all --profile quick",
     "Reads the 1000 AirfRANS simulations (graph_airfrans_data_batch_1..10) safely, trains a point-wise MLP and a PointNet on 800, validates on 100 and tests on 100 held-out airfoils at full mesh resolution."},
    {"Prediction", "Train surrogates from the data store", "-m aerosynapse.cli surrogate --profile quick", "Streams training data from shards on disk; bounded memory."},
    {"Prediction", "Predict this design's performance (surrogate)", "-m aerosynapse.cli surrogate-predict {design}", "Instant estimate of take-off mass, drag, climb and more for designs built by the fighter design space."},
    {"Analysis", "Quick aerodynamics of this design", "-m aerosynapse.cli cad-analyse {design}", "Vortex lattice, Mach-plane area rule and friction for any feature tree."},
    {"Structure", "Wing box sizing and design codes (this design)", "-m aerosynapse.cli structure {design} --W0 12000 --code mil_fighter --material al7075 --vd 400",
     "V-n loads, covers and spar webs sized to zero margin for strength and buckling, then stiffened for flutter and divergence; frame finite elements; fuselage bending."},
    {"Workflow", "Whole design workflow of a project", "-m aerosynapse.cli workflow run falcon --profile quick",
     "Basic design, design, wing and body optimisation, structure, PINN skin, 3D flow and export, in order. The Workflow menu (F9) runs it with live status."},
    {"Apps", "Designer app (all workspaces, inside Studio)", "-m aerosynapse.cli designer", "Editor, studies, 3D flow, materials, skin heating, data store and the optimisation centre."},
    {"Apps", "Rebuild the fighter report", "scripts/make_fighter_report.py", "Regenerates results/fighter/report.html from the saved results."},
};
static const int NTASK = (int)(sizeof(TASKS) / sizeof(TASKS[0]));
static HWND g_jobs, g_job_list, g_job_args, g_job_log, g_job_desc, g_job_run, g_job_stop, g_job_open;
static HANDLE g_job_obj = nullptr, g_job_proc = nullptr;
static std::string g_job_logfile;
static int g_job_task = -1;

static void jobs_select(int k) {
    if (k < 0 || k >= NTASK) return;
    SendMessageA(g_job_list, LB_SETCURSEL, k, 0);
    SetWindowTextA(g_job_args, TASKS[k].args);
    SetWindowTextA(g_job_desc, (std::string(TASKS[k].group) + ": " + TASKS[k].about).c_str());
}

static void jobs_run() {
    if (g_job_proc) { MessageBoxA(g_jobs, "A task is already running. Stop it first.", "Optimisation centre", MB_ICONINFORMATION); return; }
    int k = (int)SendMessageA(g_job_list, LB_GETCURSEL, 0, 0);
    std::string args = read_edit(g_job_args);
    { std::string app = labs_app_of(args); if (!app.empty()) { open_labs(app, ""); return; } }  // a browser app: inside Studio
    std::string dir = work_dir("jobs");
    if (args.find("{design}") != std::string::npos) {
        std::string dp = dir + "\\current_design.json";
        js::write_file(dp, js::dump(g_doc, 2));
        size_t at;
        while ((at = args.find("{design}")) != std::string::npos) args.replace(at, 8, "\"" + dp + "\"");
    }
    g_job_logfile = dir + "\\task_" + tostr(k) + ".log";
    std::string cmd = "cmd /c \"\"" + python() + "\" " + args + " > \"" + g_job_logfile + "\" 2>&1\"";
    std::string pp = g_root + "\\src";
    SetEnvironmentVariableA("PYTHONPATH", pp.c_str());
    STARTUPINFOA si; memset(&si, 0, sizeof si); si.cb = sizeof si;
    PROCESS_INFORMATION pi;
    if (!CreateProcessA(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, g_root.c_str(), &si, &pi)) {
        MessageBoxA(g_jobs, ("could not start: " + cmd).c_str(), "Optimisation centre", MB_ICONERROR); return;
    }
    g_job_obj = CreateJobObjectA(nullptr, nullptr);  // the whole process tree stops together
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li; memset(&li, 0, sizeof li);
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(g_job_obj, JobObjectExtendedLimitInformation, &li, sizeof li);
    AssignProcessToJobObject(g_job_obj, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    g_job_proc = pi.hProcess; g_job_task = k;
    SetWindowTextA(g_job_log, ("running: python " + args + "\r\n").c_str());
    SetTimer(g_jobs, 2, 1000, nullptr);
}

static void jobs_poll(bool final_) {
    std::string t;
    try { t = js::read_file(g_job_logfile); } catch (...) { return; }
    if (t.size() > 12000) t = "...\n" + t.substr(t.size() - 12000);
    std::string head = std::string(final_ ? "finished: " : "running: ") + (g_job_task >= 0 ? TASKS[g_job_task].name : "") + "\n\n";
    SetWindowTextA(g_job_log, crlf(head + t).c_str());
    SendMessageA(g_job_log, EM_SETSEL, (WPARAM)-1, (LPARAM)-1);
    SendMessageA(g_job_log, WM_VSCROLL, SB_BOTTOM, 0);
}

static void jobs_stop() {
    if (g_job_obj) { TerminateJobObject(g_job_obj, 1); CloseHandle(g_job_obj); g_job_obj = nullptr; }
    if (g_job_proc) { CloseHandle(g_job_proc); g_job_proc = nullptr; }
    KillTimer(g_jobs, 2);
}

static LRESULT CALLBACK jobs_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            HINSTANCE hi = GetModuleHandleA(nullptr);
            g_job_list = CreateWindowExA(WS_EX_CLIENTEDGE, "LISTBOX", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | LBS_NOTIFY | LBS_NOINTEGRALHEIGHT, 10, 10, 330, 520, h, (HMENU)ID_JOB_LIST, hi, nullptr);
            for (int k = 0; k < NTASK; ++k) SendMessageA(g_job_list, LB_ADDSTRING, 0, (LPARAM)(std::string(TASKS[k].group) + "  |  " + TASKS[k].name).c_str());
            g_job_desc = CreateWindowExA(0, "STATIC", "", WS_CHILD | WS_VISIBLE, 350, 10, 620, 54, h, (HMENU)ID_JOB_DESC, hi, nullptr);
            g_job_args = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL, 350, 70, 620, 24, h, (HMENU)ID_JOB_ARGS, hi, nullptr);
            g_job_run = CreateWindowExA(0, "BUTTON", "Run", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 350, 102, 120, 28, h, (HMENU)ID_JOB_RUN, hi, nullptr);
            g_job_stop = CreateWindowExA(0, "BUTTON", "Stop", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 478, 102, 120, 28, h, (HMENU)ID_JOB_STOP, hi, nullptr);
            g_job_open = CreateWindowExA(0, "BUTTON", "Open results folder", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 606, 102, 170, 28, h, (HMENU)ID_JOB_OPEN, hi, nullptr);
            g_job_log = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY, 350, 138, 620, 392, h, (HMENU)ID_JOB_LOG, hi, nullptr);
            HWND all[] = {g_job_list, g_job_desc, g_job_args, g_job_run, g_job_stop, g_job_open, g_job_log};
            for (HWND x : all) SendMessageA(x, WM_SETFONT, (WPARAM)g_font, TRUE);
            jobs_select(g_job_task >= 0 ? g_job_task : 0);
            return 0;
        }
        case WM_COMMAND:
            if (LOWORD(w) == ID_JOB_LIST && HIWORD(w) == LBN_SELCHANGE) jobs_select((int)SendMessageA(g_job_list, LB_GETCURSEL, 0, 0));
            if (LOWORD(w) == ID_JOB_RUN) jobs_run();
            if (LOWORD(w) == ID_JOB_STOP && g_job_proc) { jobs_stop(); jobs_poll(true); SetWindowTextA(g_job_desc, "Stopped."); }
            if (LOWORD(w) == ID_JOB_OPEN) ShellExecuteA(h, "open", (g_root + "\\results").c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            return 0;
        case WM_TIMER:
            if (w == 2 && g_job_proc) {
                bool done = WaitForSingleObject(g_job_proc, 0) == WAIT_OBJECT_0;
                jobs_poll(done);
                if (done) jobs_stop();
            }
            return 0;
        case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkColor(dc, C_PANEL); SetTextColor(dc, C_INK); return (LRESULT)g_br_panel; }
        case WM_CTLCOLOREDIT: case WM_CTLCOLORLISTBOX: { HDC dc = (HDC)w; SetBkColor(dc, C_FIELD); SetTextColor(dc, C_INK); return (LRESULT)g_br_field; }
        case WM_DRAWITEM: {
            DRAWITEMSTRUCT* d = (DRAWITEMSTRUCT*)l;
            draw_button(d, d->CtlID == ID_JOB_RUN ? gcol(34, 197, 94) : gcol(22, 50, 92));
            return TRUE;
        }
        case WM_CLOSE: ShowWindow(h, SW_HIDE); return 0;  // keeps running tasks; reopen from the Optimise menu
    }
    return DefWindowProcA(h, m, w, l);
}

static void open_centre(int task) {
    if (!g_jobs) {
        g_jobs = CreateWindowExA(WS_EX_TOOLWINDOW, "AeroSynapseJobs", "Optimisation and simulation centre", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                                 CW_USEDEFAULT, CW_USEDEFAULT, 1000, 580, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
    }
    if (task >= 0 && !g_job_proc) jobs_select(task);
    ShowWindow(g_jobs, SW_SHOW); SetForegroundWindow(g_jobs);
}

// ------------------------------------------------------------------ Design workflow (projects/<name>, aerosynapse.workflow)
struct Stage { const char* key; const char* title; const char* about; };
static const Stage STAGE_LIST[] = {
    {"concept", "Basic design", "The starting design (a fighter-family baseline or the design open in Studio) and its first analysis: geometry, mass properties, quick aerodynamics and, for design-space aircraft, the full performance model."},
    {"design", "Design optimisation", "CMA-ES over the whole configuration: the fighter design space (weights, drag, turn, climb, gust and extreme states), or for any other body the main wing planform with the quick analysis."},
    {"wing", "Wing optimisation", "The airfoil of the main wing by the shape optimiser: CST section, NeuralFoil with compressibility at the normal Mach number, supersonic wave drag, thickness and spar depth held; XFOIL check; the section goes back into the 3D wing."},
    {"body", "Body optimisation", "Fuselage cross-sections for the supersonic area rule (Mach-plane wave drag at M1.2 and M1.6) and wetted area, with the internal volume and the engine and cockpit envelopes as constraints."},
    {"structure", "Structure and design codes", "Design-code loads and V-n diagram; wing box sized for strength, buckling, flutter and divergence with a 3D frame finite-element model; fuselage bending; material trade."},
    {"thermal", "Skin heating with the PINN", "Convergence study of the parametric PINN against the exact series, then the lightest skin material and gauge for every zone with the PINN in the loop."},
    {"simulation", "3D flow simulation", "The 3D flow solver on three grids (Richardson extrapolation and grid-convergence index), then manoeuvre and supersonic states. The finest cruise surface is loaded onto the model."},
    {"export", "Export and documentation", "STL, OBJ, glTF, STEP, DXF three-view, A3 drawing sheet, bill of materials, mass properties, the full report and a checksummed project package."},
};
static const int NSTAGE = 8;
static HWND g_wf, g_wf_list, g_wf_log, g_wf_desc, g_wf_proj, g_wf_profile, g_wf_start;
static HANDLE g_wf_obj = nullptr, g_wf_proc = nullptr;
static std::string g_wf_dir, g_wf_logfile, g_wf_running;  // project folder; log of the running command; stage key or "all"
static std::string g_wf_status[8], g_wf_secs[8];
static std::string g_pending_project;

static std::string wf_stage_dir(int k) { char b[16]; snprintf(b, sizeof b, "\\%02d_", k + 1); return g_wf_dir + b + STAGE_LIST[k].key; }

static bool file_exists(const std::string& f) { return GetFileAttributesA(f.c_str()) != INVALID_FILE_ATTRIBUTES; }

static void wf_refresh() {
    if (!g_wf) return;
    std::string name = g_wf_dir.empty() ? std::string("no project open") : g_wf_dir;
    for (int k = 0; k < NSTAGE; ++k) { g_wf_status[k] = "pending"; g_wf_secs[k].clear(); }
    if (!g_wf_dir.empty()) {
        try {
            js::Value m = js::parse(js::read_file(g_wf_dir + "\\project.json"));
            name = m.str("name", "") + "   (" + g_wf_dir + ")   " + m.str("origin", "");
            if (const js::Value* st = m.find("stages"))
                for (int k = 0; k < NSTAGE; ++k)
                    if (const js::Value* e = st->find(STAGE_LIST[k].key)) {
                        g_wf_status[k] = e->str("status", "pending");
                        double sec = e->num("seconds", -1);
                        if (sec >= 0) g_wf_secs[k] = sec < 120 ? tostr((long)(sec + 0.5)) + " s" : tostr((long)(sec / 60 + 0.5)) + " min";
                    }
        } catch (...) {}
    }
    SetWindowTextA(g_wf_proj, ("Project: " + name).c_str());
    InvalidateRect(g_wf_list, nullptr, TRUE);
}

static void wf_log_poll() {
    std::string t;
    try { t = js::read_file(g_wf_logfile); } catch (...) { return; }
    if (t.size() > 16000) t = "...\n" + t.substr(t.size() - 16000);
    SetWindowTextA(g_wf_log, crlf(t).c_str());
    SendMessageA(g_wf_log, EM_SETSEL, (WPARAM)-1, (LPARAM)-1);
    SendMessageA(g_wf_log, WM_VSCROLL, SB_BOTTOM, 0);
}

static void wf_load_stage(int k, bool quiet) {
    if (g_wf_dir.empty() || k < 0) return;
    std::string f = wf_stage_dir(k) + "\\design.json";
    if (!file_exists(f)) { if (!quiet) MessageBoxA(g_wf, "This stage has no design yet. Run it first.", "Design workflow", MB_ICONINFORMATION); return; }
    try { set_doc(js::parse(js::read_file(f)), f); status("loaded " + std::string(STAGE_LIST[k].title) + " design: " + f); }
    catch (std::exception& e) { MessageBoxA(g_wf, e.what(), "Load failed", MB_ICONERROR); return; }
    if (std::string(STAGE_LIST[k].key) == "simulation" || std::string(STAGE_LIST[k].key) == "export") {  // paint the finest cruise surface on the model
        try {
            js::Value r = js::parse(js::read_file(g_wf_dir + "\\07_simulation\\result.json"));
            std::string d = r.str("studio_results_dir", "");
            for (char& c : d) if (c == '/') c = '\\';
            if (!d.empty() && file_exists(g_wf_dir + "\\" + d + "\\surface.csv")) {
                g_pending_results = g_wf_dir + "\\" + d + "\\surface.csv";  // painted once the new mesh is built
                g_sel = -1; fill_list(); fill_props(); rebuild();  // no selection highlight over the results
            }
        } catch (...) {}
    }
}

static void wf_stop() {
    if (g_wf_obj) { TerminateJobObject(g_wf_obj, 1); CloseHandle(g_wf_obj); g_wf_obj = nullptr; }
    if (g_wf_proc) { CloseHandle(g_wf_proc); g_wf_proc = nullptr; }
    KillTimer(g_wf, 3);
}

static bool wf_launch(const std::string& args, const std::string& what) {
    if (g_wf_proc) { MessageBoxA(g_wf, "The workflow is already running. Stop it first.", "Design workflow", MB_ICONINFORMATION); return false; }
    g_wf_logfile = g_wf_dir + "\\studio_run.log";
    std::string cmd = "cmd /c \"\"" + python() + "\" -m aerosynapse.cli workflow " + args + " > \"" + g_wf_logfile + "\" 2>&1\"";
    std::string pp = g_root + "\\src";
    SetEnvironmentVariableA("PYTHONPATH", pp.c_str());
    STARTUPINFOA si; memset(&si, 0, sizeof si); si.cb = sizeof si;
    PROCESS_INFORMATION pi;
    if (!CreateProcessA(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, g_root.c_str(), &si, &pi)) {
        MessageBoxA(g_wf, ("could not start: " + cmd).c_str(), "Design workflow", MB_ICONERROR); return false;
    }
    g_wf_obj = CreateJobObjectA(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li; memset(&li, 0, sizeof li);
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(g_wf_obj, JobObjectExtendedLimitInformation, &li, sizeof li);
    AssignProcessToJobObject(g_wf_obj, pi.hProcess);
    ResumeThread(pi.hThread); CloseHandle(pi.hThread);
    g_wf_proc = pi.hProcess; g_wf_running = what;
    SetWindowTextA(g_wf_log, ("running: workflow " + args + "\r\n").c_str());
    SetTimer(g_wf, 3, 1500, nullptr);
    status("workflow: " + what + " running...", 1);
    return true;
}

static void wf_select(int k) {
    if (k < 0 || k >= NSTAGE || !g_wf_list) return;
    SendMessageA(g_wf_list, LB_SETCURSEL, k, 0);
    SetWindowTextA(g_wf_desc, STAGE_LIST[k].about);
}

static std::string wf_profile() { return SendMessageA(g_wf_profile, CB_GETCURSEL, 0, 0) == 1 ? "standard" : "quick"; }

static void make_dirs(const std::string& d);
// tools that need a workflow project offer to make one from the design in the editor
static void wf_set_project(const std::string& dir);
static bool ensure_project(HWND owner) {
    if (!g_wf_dir.empty()) return true;
    if (g_selftest) { g_st_log.push_back("MSG offers to create a project from the design"); return false; }
    if (MessageBoxA(owner, "No design project is open. Create one now from the design in the editor?\n\n(It goes in the projects folder; open another with Workflow > Open project.)",
                    "Design workflow", MB_YESNO | MB_ICONQUESTION) != IDYES) return false;
    std::string nm = g_doc.str("name", "design"), clean;
    for (char c : nm) clean += (isalnum((unsigned char)c) || c == '-' || c == '_') ? c : '_';
    if (clean.empty()) clean = "design";
    SYSTEMTIME t; GetLocalTime(&t);
    char stamp[32]; snprintf(stamp, sizeof stamp, "_%04d%02d%02d_%02d%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
    std::string name = clean + stamp, parent = g_root + "\\projects", dir = parent + "\\" + name;
    make_dirs(dir);
    js::write_file(dir + "\\design.json", js::dump(g_doc, 2));
    std::string cmd = "cmd /c \"\"" + python() + "\" -m aerosynapse.cli workflow new " + name + " --from \"" + dir + "\\design.json\" --where \"" + parent + "\" --overwrite > \"" + dir + "\\new.log\" 2>&1\"";
    SetEnvironmentVariableA("PYTHONPATH", (g_root + "\\src").c_str());
    STARTUPINFOA si; memset(&si, 0, sizeof si); si.cb = sizeof si; PROCESS_INFORMATION pi;
    if (CreateProcessA(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, g_root.c_str(), &si, &pi)) { WaitForSingleObject(pi.hProcess, 90000); CloseHandle(pi.hProcess); CloseHandle(pi.hThread); }
    if (!file_exists(dir + "\\project.json") && !g_selftest) { MessageBoxA(owner, ("The project could not be created; see " + dir + "\\new.log").c_str(), "Design workflow", MB_ICONERROR); return false; }
    wf_set_project(dir);
    status("project created from the design: " + dir);
    return true;
}

static void wf_run(int mode) {  // 0 selected stage, 1 all remaining, 2 rerun from the selected stage
    if (!ensure_project(g_wf ? g_wf : g_main)) return;
    int k = (int)SendMessageA(g_wf_list, LB_GETCURSEL, 0, 0);
    if (k < 0) k = 0;
    std::string a = "run \"" + g_wf_dir + "\" --profile " + wf_profile();
    if (mode == 0) wf_launch(a + " --stage " + STAGE_LIST[k].key, STAGE_LIST[k].key);
    else if (mode == 1) wf_launch(a, "all");
    else wf_launch(a + " --start " + STAGE_LIST[k].key, std::string("from ") + STAGE_LIST[k].key);
}

// a small modal text prompt
static std::string g_ask_text; static bool g_ask_ok, g_ask_done;
static LRESULT CALLBACK ask_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_COMMAND:
            if (LOWORD(w) == ID_ASK_OK) { g_ask_text = read_edit(GetDlgItem(h, ID_ASK_EDIT)); g_ask_ok = true; g_ask_done = true; DestroyWindow(h); }
            if (LOWORD(w) == ID_ASK_CANCEL) { g_ask_done = true; DestroyWindow(h); }
            return 0;
        case WM_CLOSE: g_ask_done = true; DestroyWindow(h); return 0;
        case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkColor(dc, C_PANEL); SetTextColor(dc, C_INK); return (LRESULT)g_br_panel; }
    }
    return DefWindowProcA(h, m, w, l);
}
static bool ask_text(HWND owner, const char* title, const char* prompt, std::string& io) {
    if (g_selftest) { g_st_log.push_back(std::string("DIALOG text: ") + title); return false; }
    HINSTANCE hi = GetModuleHandleA(nullptr);
    HWND d = CreateWindowExA(WS_EX_DLGMODALFRAME, "AeroSynapseAsk", title, WS_POPUP | WS_CAPTION | WS_SYSMENU, 300, 260, 460, 160, owner, nullptr, hi, nullptr);
    HWND st = CreateWindowExA(0, "STATIC", prompt, WS_CHILD | WS_VISIBLE, 14, 12, 420, 36, d, nullptr, hi, nullptr);
    HWND ed = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", io.c_str(), WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL | WS_TABSTOP, 14, 50, 420, 24, d, (HMENU)ID_ASK_EDIT, hi, nullptr);
    HWND ok = CreateWindowExA(0, "BUTTON", "OK", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON | WS_TABSTOP, 254, 84, 86, 28, d, (HMENU)ID_ASK_OK, hi, nullptr);
    HWND ca = CreateWindowExA(0, "BUTTON", "Cancel", WS_CHILD | WS_VISIBLE | WS_TABSTOP, 348, 84, 86, 28, d, (HMENU)ID_ASK_CANCEL, hi, nullptr);
    HWND all[] = {st, ed, ok, ca};
    for (HWND x : all) SendMessageA(x, WM_SETFONT, (WPARAM)g_font, TRUE);
    EnableWindow(owner, FALSE);
    ShowWindow(d, SW_SHOW); SetFocus(ed); SendMessageA(ed, EM_SETSEL, 0, -1);
    g_ask_ok = false; g_ask_done = false;
    MSG msg;
    while (!g_ask_done && GetMessageA(&msg, nullptr, 0, 0)) {
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_RETURN) { SendMessageA(d, WM_COMMAND, ID_ASK_OK, 0); continue; }
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_ESCAPE) { SendMessageA(d, WM_COMMAND, ID_ASK_CANCEL, 0); continue; }
        if (!IsDialogMessageA(d, &msg)) { TranslateMessage(&msg); DispatchMessageA(&msg); }
    }
    EnableWindow(owner, TRUE); SetForegroundWindow(owner);
    if (g_ask_ok) io = g_ask_text;
    return g_ask_ok;
}

static void open_workflow();
static void wf_set_project(const std::string& dir) {
    g_wf_dir = dir;
    while (!g_wf_dir.empty() && (g_wf_dir.back() == '\\' || g_wf_dir.back() == '/')) g_wf_dir.pop_back();
    open_workflow();
    g_wf_logfile = g_wf_dir + "\\workflow.log";
    wf_log_poll();
    wf_refresh();
}

static void wf_new() {
    std::string nm = g_doc.str("name", "design");
    std::string clean;
    for (char c : nm) clean += (isalnum((unsigned char)c) || c == '-' || c == '_') ? c : '_';
    if (clean.empty()) clean = "design";
    if (!ask_text(g_wf ? g_wf : g_main, "New design project", "Project name (letters, digits, - and _). The project starts from the choice in 'Start from'.", clean)) return;
    int start = g_wf_start ? (int)SendMessageA(g_wf_start, CB_GETCURSEL, 0, 0) : 0;
    static const char* fam[] = {"", "conventional", "canard_delta", "twin_tail_lerx", "tailless_blended", "freeform"};
    std::string args = "new " + clean + " --overwrite";
    if (start <= 0) {
        std::string dp = work_dir("jobs") + "\\workflow_input.json";
        js::write_file(dp, js::dump(g_doc, 2));
        args += " --from \"" + dp + "\"";
    } else args += std::string(" --family ") + fam[start];
    std::string cmd = "cmd /c \"\"" + python() + "\" -m aerosynapse.cli workflow " + args + " > \"" + work_dir("jobs") + "\\workflow_new.log\" 2>&1\"";
    SetEnvironmentVariableA("PYTHONPATH", (g_root + "\\src").c_str());
    STARTUPINFOA si; memset(&si, 0, sizeof si); si.cb = sizeof si; PROCESS_INFORMATION pi;
    if (!CreateProcessA(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, g_root.c_str(), &si, &pi)) { MessageBoxA(g_main, "could not start Python", "Design workflow", MB_ICONERROR); return; }
    WaitForSingleObject(pi.hProcess, 60000);
    DWORD code = 1; GetExitCodeProcess(pi.hProcess, &code); CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    std::string dir = g_root + "\\projects\\" + clean;
    if (code != 0 || !file_exists(dir + "\\project.json")) {
        std::string t; try { t = js::read_file(work_dir("jobs") + "\\workflow_new.log"); } catch (...) {}
        MessageBoxA(g_main, ("Could not create the project.\n\n" + t.substr(t.size() > 1500 ? t.size() - 1500 : 0)).c_str(), "Design workflow", MB_ICONERROR); return;
    }
    wf_set_project(dir);
    status("project created: " + dir);
}

static void wf_open() {
    std::string p = g_root + "\\projects\\project.json";
    if (!ask_path(false, "Design project (project.json)\0project.json\0All files\0*.*\0", "json", p)) return;
    size_t at = p.find_last_of("\\/");
    wf_set_project(at == std::string::npos ? p : p.substr(0, at));
}

static LRESULT CALLBACK wf_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            HINSTANCE hi = GetModuleHandleA(nullptr);
            g_wf_proj = CreateWindowExA(0, "STATIC", "Project:", WS_CHILD | WS_VISIBLE | SS_ENDELLIPSIS, 12, 10, 960, 20, h, (HMENU)ID_WF_PROJ, hi, nullptr);
            HWND b1 = CreateWindowExA(0, "BUTTON", "New project...", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 12, 36, 130, 28, h, (HMENU)ID_WF_NEW, hi, nullptr);
            HWND b2 = CreateWindowExA(0, "BUTTON", "Open project...", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 148, 36, 130, 28, h, (HMENU)ID_WF_OPEN, hi, nullptr);
            HWND l1 = CreateWindowExA(0, "STATIC", "Start from:", WS_CHILD | WS_VISIBLE, 292, 42, 70, 20, h, nullptr, hi, nullptr);
            g_wf_start = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST, 364, 38, 250, 200, h, (HMENU)ID_WF_START, hi, nullptr);
            const char* opts[] = {"the design open in Studio", "conventional fighter baseline", "canard-delta baseline", "twin-tail LERX baseline", "tailless blended baseline", "free-form baseline"};
            for (const char* o : opts) SendMessageA(g_wf_start, CB_ADDSTRING, 0, (LPARAM)o);
            SendMessageA(g_wf_start, CB_SETCURSEL, 0, 0);
            HWND l2 = CreateWindowExA(0, "STATIC", "Profile:", WS_CHILD | WS_VISIBLE, 630, 42, 52, 20, h, nullptr, hi, nullptr);
            g_wf_profile = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST, 684, 38, 288, 200, h, (HMENU)ID_WF_PROFILE, hi, nullptr);
            SendMessageA(g_wf_profile, CB_ADDSTRING, 0, (LPARAM)"quick (about 45 minutes in all)");
            SendMessageA(g_wf_profile, CB_ADDSTRING, 0, (LPARAM)"standard (several hours, finer grids)");
            SendMessageA(g_wf_profile, CB_SETCURSEL, 0, 0);
            g_wf_list = CreateWindowExA(WS_EX_CLIENTEDGE, "LISTBOX", "", WS_CHILD | WS_VISIBLE | LBS_NOTIFY | LBS_OWNERDRAWFIXED | LBS_HASSTRINGS | LBS_NOINTEGRALHEIGHT, 12, 74, 330, 8 * 46 + 4, h, (HMENU)ID_WF_LIST, hi, nullptr);
            for (int k = 0; k < NSTAGE; ++k) SendMessageA(g_wf_list, LB_ADDSTRING, 0, (LPARAM)STAGE_LIST[k].title);
            SendMessageA(g_wf_list, LB_SETCURSEL, 0, 0);
            g_wf_desc = CreateWindowExA(0, "STATIC", STAGE_LIST[0].about, WS_CHILD | WS_VISIBLE, 356, 74, 616, 52, h, (HMENU)ID_WF_DESC, hi, nullptr);
            struct { const char* t; int id; int x, y, w; } bt[] = {
                {"Run this stage", ID_WF_RUN, 356, 132, 120}, {"Run all remaining", ID_WF_ALL, 482, 132, 140}, {"Rerun from here", ID_WF_FROM, 628, 132, 130}, {"Stop", ID_WF_STOP, 764, 132, 80},
                {"Load stage design", ID_WF_LOAD, 356, 166, 150}, {"Open report", ID_WF_REPORT, 512, 166, 120}, {"Open project folder", ID_WF_FOLDER, 638, 166, 160},
                {"Report PDF", ID_WF_RPDF, 356, 200, 120}, {"Report Word", ID_WF_RDOCX, 482, 200, 120}, {"Report text", ID_WF_RTXT, 608, 200, 120}};
            std::vector<HWND> hs = {g_wf_proj, b1, b2, l1, l2, g_wf_start, g_wf_profile, g_wf_list, g_wf_desc};
            for (auto& b : bt) hs.push_back(CreateWindowExA(0, "BUTTON", b.t, WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, b.x, b.y, b.w, 28, h, (HMENU)(INT_PTR)b.id, hi, nullptr));
            g_wf_log = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY, 356, 236, 616, 208, h, (HMENU)ID_WF_LOG, hi, nullptr);
            hs.push_back(g_wf_log);
            for (HWND x : hs) SendMessageA(x, WM_SETFONT, (WPARAM)g_font, TRUE);
            return 0;
        }
        case WM_MEASUREITEM: { MEASUREITEMSTRUCT* mi = (MEASUREITEMSTRUCT*)l; if (mi->CtlID == ID_WF_LIST) { mi->itemHeight = 46; return TRUE; } break; }
        case WM_COMMAND: {
            int id = LOWORD(w);
            if (id == ID_WF_LIST && HIWORD(w) == LBN_SELCHANGE) { int k = (int)SendMessageA(g_wf_list, LB_GETCURSEL, 0, 0); if (k >= 0) SetWindowTextA(g_wf_desc, STAGE_LIST[k].about); }
            if (id == ID_WF_LIST && HIWORD(w) == LBN_DBLCLK) wf_load_stage((int)SendMessageA(g_wf_list, LB_GETCURSEL, 0, 0), false);
            if (id == ID_WF_RUN) wf_run(0);
            if (id == ID_WF_ALL) wf_run(1);
            if (id == ID_WF_FROM) wf_run(2);
            if (id == ID_WF_STOP && g_wf_proc) { wf_stop(); wf_log_poll(); wf_refresh(); status("workflow stopped", 1); }
            if (id == ID_WF_LOAD) wf_load_stage((int)SendMessageA(g_wf_list, LB_GETCURSEL, 0, 0), false);
            if (id == ID_WF_NEW) wf_new();
            if (id == ID_WF_OPEN) wf_open();
            if (id == ID_WF_REPORT) {
                std::string r = g_wf_dir + "\\08_export\\report.html";
                if (file_exists(r)) ShellExecuteA(h, "open", r.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
                else MessageBoxA(h, "The report is written by the Export stage.", "Design workflow", MB_ICONINFORMATION);
            }
            if (id == ID_WF_FOLDER && !g_wf_dir.empty()) ShellExecuteA(h, "open", g_wf_dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            if (id == ID_WF_RPDF) SendMessageA(g_main, WM_COMMAND, ID_EXP_PDF, 0);
            if (id == ID_WF_RDOCX) SendMessageA(g_main, WM_COMMAND, ID_EXP_DOCX, 0);
            if (id == ID_WF_RTXT) SendMessageA(g_main, WM_COMMAND, ID_EXP_TXT, 0);
            return 0;
        }
        case WM_TIMER:
            if (w == 3 && g_wf_proc) {
                bool done = WaitForSingleObject(g_wf_proc, 0) == WAIT_OBJECT_0;
                wf_log_poll();
                wf_refresh();
                if (done) {
                    DWORD code = 1; GetExitCodeProcess(g_wf_proc, &code);
                    wf_stop();
                    int last = -1;
                    for (int k = 0; k < NSTAGE; ++k) if (g_wf_status[k] == "done") last = k;
                    if (code == 0 && last >= 0) { wf_load_stage(last, true); wf_select(last); }
                    status(code == 0 ? "workflow: finished" : "workflow: failed (see the log)", 1);
                    FlashWindow(g_main, TRUE);
                }
            }
            return 0;
        case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkColor(dc, C_PANEL); SetTextColor(dc, C_INK); return (LRESULT)g_br_panel; }
        case WM_CTLCOLOREDIT: case WM_CTLCOLORLISTBOX: { HDC dc = (HDC)w; SetBkColor(dc, C_FIELD); SetTextColor(dc, C_INK); return (LRESULT)g_br_field; }
        case WM_DRAWITEM: {
            DRAWITEMSTRUCT* d = (DRAWITEMSTRUCT*)l;
            if (d->CtlID == ID_WF_LIST) {
                if ((int)d->itemID < 0) return TRUE;
                int k = (int)d->itemID;
                bool sel = (d->itemState & ODS_SELECTED) != 0;
                RECT rc = d->rcItem;
                int W = rc.right - rc.left, H = rc.bottom - rc.top;
                gp::Bitmap buf(W, H, PixelFormat32bppARGB);
                gp::Graphics g(&buf);
                quality(g);
                gp::LinearGradientBrush bgb(gp::Rect(0, 0, W, H), gcol(236, 241, 248), gcol(222, 230, 241), gp::LinearGradientModeVertical);
                g.FillRectangle(&bgb, 0, 0, W, H);
                const std::string& stt = g_wf_status[k];
                gp::Color c = stt == "done" ? gcol(22, 163, 74) : stt == "running" ? gcol(14, 165, 233) : stt == "failed" ? gcol(220, 38, 38) : stt == "stale" ? gcol(217, 119, 6) : gcol(148, 163, 184);
                gp::RectF card(5, 4, W - 10.0f, H - 9.0f);
                soft_shadow(g, card, 8, 4, sel ? 90 : 55);
                fill_round(g, card, 8, sel ? gcol(255, 255, 255) : gcol(250, 252, 255), sel ? gcol(219, 234, 254) : gcol(234, 240, 248));
                gp::GraphicsPath cp; round_path(cp, card, 8);
                gp::Pen cpen(sel ? gcol(59, 130, 246) : gcol(203, 213, 225), sel ? 2.0f : 1.0f); g.DrawPath(&cpen, &cp);
                gp::SolidBrush accb(c); g.FillRectangle(&accb, card.X + 1, card.Y + 6, 4.0f, card.Height - 12);
                gp::RectF ball(card.X + 12, card.Y + (card.Height - 30) / 2, 30, 30);
                gp::GraphicsPath bp; bp.AddEllipse(ball);
                gp::PathGradientBrush pg(&bp);
                pg.SetCenterPoint(gp::PointF(ball.X + 10, ball.Y + 9));
                pg.SetCenterColor(gshade(c, 0.55f));
                gp::Color edge = gshade(c, -0.25f); int ne = 1; pg.SetSurroundColors(&edge, &ne);
                g.FillEllipse(&pg, ball);
                char num[4]; snprintf(num, sizeof num, "%d", k + 1);
                gtext(g, num, ball, gcol(255, 255, 255), 14.0f, gp::FontStyleBold, gp::StringAlignmentCenter);
                gtext(g, STAGE_LIST[k].title, gp::RectF(card.X + 50, card.Y + 2, card.Width - 90, 20), gcol(15, 30, 55), 13.5f, gp::FontStyleBold);
                std::string sub = stt + (g_wf_secs[k].empty() ? "" : "  |  " + g_wf_secs[k]) + (g_wf_running == STAGE_LIST[k].key && g_wf_proc ? "  |  running now" : "");
                gtext(g, sub, gp::RectF(card.X + 50, card.Y + 20, card.Width - 90, 16), gshade(c, -0.15f), 11.5f, gp::FontStyleBold);
                char ic[16]; snprintf(ic, sizeof ic, "stage%d", k + 1);
                (void)ic;
                static const char* sic[8] = {"sketch_revolve", "opt_fighter", "opt_airfoil", "opt_topology", "structure", "pinn", "flow_field", "package"};
                if (gp::Bitmap* im = icon_png(sic[k])) g.DrawImage(im, card.X + card.Width - 36, card.Y + (card.Height - 30) / 2, 30.0f, 30.0f);
                gp::Graphics out(d->hDC);
                out.DrawImage(&buf, (INT)rc.left, (INT)rc.top);
                return TRUE;
            }
            bool primary = d->CtlID == ID_WF_RUN || d->CtlID == ID_WF_ALL || d->CtlID == ID_WF_NEW;
            bool report = d->CtlID == ID_WF_RPDF || d->CtlID == ID_WF_RDOCX || d->CtlID == ID_WF_RTXT;
            draw_button(d, primary ? gcol(59, 130, 246) : report ? gcol(202, 138, 4) : d->CtlID == ID_WF_STOP ? gcol(190, 40, 60) : gcol(22, 50, 92));
            return TRUE;
        }
        case WM_CLOSE: ShowWindow(h, SW_HIDE); return 0;  // a running workflow keeps running; reopen from the Workflow menu
    }
    return DefWindowProcA(h, m, w, l);
}

static void open_workflow() {
    if (!g_wf) {
        g_wf = CreateWindowExA(WS_EX_TOOLWINDOW, "AeroSynapseWorkflow", "Design workflow: basic design to export", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                               CW_USEDEFAULT, CW_USEDEFAULT, 1000, 495, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
    }
    wf_refresh();
    ShowWindow(g_wf, SW_SHOW); SetForegroundWindow(g_wf);
}

static void gl_text(float x, float y, const std::string& s);
#include "ui_tools.inc"
#include "apps.inc"
#include "ribbon.inc"
#include "appearance.inc"
#include "sketch.inc"
#include "viz3d.inc"
#include "simview.inc"

// ---------------------------------------------------------------- flow animations: streamlines, slice sweep, recording
static bool g_rec_on = false, g_pending_flowlines = false;
static int g_rec_n = 0;
static std::string g_rec_dir;
static bool flow_ready() {
    if (!g_vol_ok) toggle_field(false);
    if (!g_vol_ok) { MessageBoxA(g_main, "These animations need the solver's 3D volume (volume.bin): run a 3D CFD case first, or open its results.", "Flow animation", MB_ICONINFORMATION); return false; }
    return true;
}
static void flowlines_toggle() {
    if (!flow_ready()) return;
    g_field_on = true;
    g_sl_on = !g_sl_on;
    if (g_sl_on && g_sl.empty()) { status("tracing 3D streamlines through the flow volume...", 1); vol_streamlines(); }
    if (g_sl_on) SetTimer(g_main, 7, 33, nullptr);
    status(g_sl_on ? "3D streamlines: " + tostr((int)g_sl.size()) + " lines from a rake upstream and from the aircraft's surface, coloured by Mach, pulses at the local flow speed" : "3D streamlines off");
    InvalidateRect(g_view, nullptr, FALSE);
}
static void sweep_toggle() {
    if (!flow_ready()) return;
    g_field_on = true;
    g_sweep_on = !g_sweep_on;
    if (g_sweep_on) { g_slice_on[g_slice_axis] = true; SetTimer(g_main, 7, 33, nullptr); }
    status(g_sweep_on ? std::string("slice sweep: the ") + "xyz"[g_slice_axis] + " slice glides through the flow (Slice axis changes the direction)" : "slice sweep stopped");
}
static void record_toggle() {
    if (g_selftest) { g_st_log.push_back("RECORD skipped in the self-test (no frames written)"); return; }
    if (g_rec_on) {
        g_rec_on = false;
        status("recorded " + tostr(g_rec_n) + " frames; encoding the video...");
        launch("\"" + python() + "\" \"" + g_root + "\\scripts\\frames_to_mp4.py\" \"" + g_rec_dir + "\" --fps 30", "video encoding", g_rec_dir);
        return;
    }
    SYSTEMTIME t; GetLocalTime(&t);
    char nm[64]; snprintf(nm, sizeof nm, "anim_%04d%02d%02d_%02d%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
    g_rec_dir = g_root + "\\results\\studio\\" + nm;
    make_dirs(g_rec_dir);
    g_rec_n = 0; g_rec_on = true;
    SetTimer(g_main, 7, 33, nullptr);
    status("recording the 3D view (orbit, zoom or play while it records; click Record again to stop, at most 60 s) -> " + g_rec_dir);
}
static void snap_view_png(const std::string& path) {
    if (!g_view || g_selftest) return;
    if (hq_save_png(path, 2.0, 2, true)) return;  // twice the view, supersampled, with the logo
    g_snap_png = path; InvalidateRect(g_view, nullptr, FALSE); UpdateWindow(g_view);  // else the screen as it is
}
static void snap_frame(int W, int H) {  // called after the view is drawn
    if (g_snap_png.empty()) return;
    std::string p = g_snap_png; g_snap_png.clear();
    std::vector<unsigned char> px((size_t)W * H * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    gp::Bitmap bm(W, H, PixelFormat24bppRGB);
    gp::Rect r(0, 0, W, H); gp::BitmapData bd;
    if (bm.LockBits(&r, gp::ImageLockModeWrite, PixelFormat24bppRGB, &bd) != gp::Ok) return;
    for (int y = 0; y < H; ++y) {
        unsigned char* dst = (unsigned char*)bd.Scan0 + (size_t)y * bd.Stride;
        const unsigned char* src = &px[(size_t)(H - 1 - y) * W * 3];
        for (int x = 0; x < W; ++x) { dst[3 * x] = src[3 * x + 2]; dst[3 * x + 1] = src[3 * x + 1]; dst[3 * x + 2] = src[3 * x]; }
    }
    bm.UnlockBits(&bd);
    save_png_bitmap(&bm, p); forget_png(p);
}
static void record_frame(int W, int H) {
    if (!g_rec_on) return;
    if (g_rec_n >= 1800) { record_toggle(); return; }
    std::vector<unsigned char> px((size_t)W * H * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    char fn[64]; snprintf(fn, sizeof fn, "\\frame_%05d.ppm", g_rec_n++);
    FILE* f = fopen((g_rec_dir + fn).c_str(), "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int y = H - 1; y >= 0; --y) fwrite(&px[(size_t)y * W * 3], 1, (size_t)W * 3, f);
    fclose(f);
}
#include "workspace.inc"
#include "setup.inc"
#include "pad.inc"
#include "selftest.inc"
static std::string g_pending_simview, g_pending_field, g_pending_hq;
static bool g_pending_sketch = false;
static std::string g_pending_tip, g_pending_new_parent, g_pending_new_name;
static void cfd_finished(const std::string& dir) { g_last_cfd_dir = dir; g_planes.clear(); g_field_on = g_particles_on = false; KillTimer(g_main, 7); }

// ------------------------------------------------------------------ OpenGL view
static void fit_view() {
    if (g_asm_on) { asm_fit(); return; }
    geo::Box b;
    for (auto& v : g_mesh.v) b.add(v);
    if (!b.valid()) return;
    g_tx = 0.5 * (b.lo.x + b.hi.x); g_ty = 0.5 * (b.lo.y + b.hi.y); g_tz = 0.5 * (b.lo.z + b.hi.z);
    g_dist = 1.6 * geo::len(b.hi - b.lo);
}

static void gl_text(float x, float y, const std::string& s) {
    if (!g_font_base) return;
    glRasterPos2f(x, y);
    glListBase(g_font_base - 32);
    glCallLists((GLsizei)s.size(), GL_UNSIGNED_BYTE, s.c_str());
}

static void draw_mesh(const geo::Mesh& M, bool colours) {
    if (M.t.empty()) return;
    static std::vector<float> vb, nb;
    vb.resize(M.v.size() * 3); nb.resize(M.v.size() * 3);
    for (size_t i = 0; i < M.v.size(); ++i) { vb[3 * i] = (float)M.v[i].x; vb[3 * i + 1] = (float)M.v[i].y; vb[3 * i + 2] = (float)M.v[i].z; nb[3 * i] = (float)M.n[i].x; nb[3 * i + 1] = (float)M.n[i].y; nb[3 * i + 2] = (float)M.n[i].z; }
    glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, vb.data()); glNormalPointer(GL_FLOAT, 0, nb.data());
    if (colours && g_colors.size() == M.v.size() * 3) { glEnableClientState(GL_COLOR_ARRAY); glColorPointer(3, GL_FLOAT, 0, g_colors.data()); }
    glDrawElements(GL_TRIANGLES, (GLsizei)M.t.size(), GL_UNSIGNED_INT, M.t.data());
    glDisableClientState(GL_COLOR_ARRAY); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_VERTEX_ARRAY);
}

// ------------------------------------------------------------------ exact surfaces, crisp lines and detail at any zoom
struct GMesh { std::vector<float> v, n, e; std::vector<int> t; int feature = -1; };
static std::vector<GMesh> g_exact;           // parts drawn from their exact definition
static geo::Mesh g_rest;                     // the sampled mesh of everything else
static std::string g_rest_text;              // the design that g_rest shows (exact parts switched off)
static bool g_rest_full = true;              // no exact parts: g_rest is the whole sampled mesh
static void draw_gmesh(const GMesh& M) {
    if (M.t.empty()) return;
    glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, M.v.data()); glNormalPointer(GL_FLOAT, 0, M.n.data());
    glDrawElements(GL_TRIANGLES, (GLsizei)M.t.size(), GL_UNSIGNED_INT, M.t.data());
    glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_VERTEX_ARRAY);
}
static void draw_exact_edges() {
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glLineWidth(1.4f); glColor4f(0.06f, 0.1f, 0.18f, 0.9f);
    glEnableClientState(GL_VERTEX_ARRAY);
    for (const GMesh& M : g_exact) if (!M.e.empty()) { glVertexPointer(3, GL_FLOAT, 0, M.e.data()); glDrawArrays(GL_LINES, 0, (GLsizei)(M.e.size() / 3)); }
    glDisableClientState(GL_VERTEX_ARRAY);
    glLineWidth(1.0f); glDisable(GL_LINE_SMOOTH); glDisable(GL_BLEND); glEnable(GL_LIGHTING);
}
static geo::Mesh g_detail;
static std::vector<float> g_creases, g_detail_creases;
static std::vector<int> g_coarse_tris;
static geo::Box g_detail_box, g_detail_inner, g_detail_inner_req;  // inner: where the coarse mesh gives way to the detail
static bool g_detail_on = false;
static int g_detail_gen = 0;
static double g_detail_h = 0;
static volatile LONG g_detail_running = 0;  // one detail mesh at a time
static void draw_mesh_idx(const geo::Mesh& M, const std::vector<int>& idx, const std::vector<float>* cols = nullptr) {
    if (idx.empty()) return;
    static std::vector<float> vb, nb;
    vb.resize(M.v.size() * 3); nb.resize(M.v.size() * 3);
    for (size_t i = 0; i < M.v.size(); ++i) { vb[3 * i] = (float)M.v[i].x; vb[3 * i + 1] = (float)M.v[i].y; vb[3 * i + 2] = (float)M.v[i].z; nb[3 * i] = (float)M.n[i].x; nb[3 * i + 1] = (float)M.n[i].y; nb[3 * i + 2] = (float)M.n[i].z; }
    glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, vb.data()); glNormalPointer(GL_FLOAT, 0, nb.data());
    if (cols && cols->size() == M.v.size() * 3) { glEnableClientState(GL_COLOR_ARRAY); glColorPointer(3, GL_FLOAT, 0, cols->data()); }
    glDrawElements(GL_TRIANGLES, (GLsizei)idx.size(), GL_UNSIGNED_INT, idx.data());
    glDisableClientState(GL_COLOR_ARRAY); glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_VERTEX_ARRAY);
}
// a sampled mesh in its parts' colours (shade < 1: the darker caps of a section)
static void draw_owned(const geo::Mesh& M, const std::vector<int>& own, float shade) {
    if (M.t.empty()) return;
    if (own.size() != M.v.size()) { draw_mesh(M, false); return; }
    std::vector<float> cols = owner_colours(own, shade);
    static std::vector<int> all;
    all.assign(M.t.begin(), M.t.end());
    draw_mesh_idx(M, all, &cols);
}
static void draw_scene_parts(bool detail, float shade) {  // every part in its colour: exact surfaces, then the sampled rest
    for (const GMesh& M : g_exact) { float c[3]; part_colour(M.feature, c); glColor3f(c[0] * shade, c[1] * shade, c[2] * shade); draw_gmesh(M); }
    if (detail) {
        std::vector<float> cols = owner_colours(g_rest_owners, shade);
        draw_mesh_idx(g_rest, g_coarse_tris, g_rest_owners.size() == g_rest.v.size() ? &cols : nullptr);
        draw_owned(g_detail, g_detail_owners, shade);
    } else draw_owned(g_rest, g_rest_owners, shade);
}
static bool in_box(const geo::Box& b, double x, double y, double z) { return x > b.lo.x && x < b.hi.x && y > b.lo.y && y < b.hi.y && z > b.lo.z && z < b.hi.z; }
static void draw_creases(bool detail) {
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glLineWidth(1.3f);
    glColor4f(0.07f, 0.11f, 0.2f, 0.85f);
    const geo::Box& inner = g_detail_inner;
    glBegin(GL_LINES);
    const std::vector<float>& cr = g_rest_creases;
    for (size_t i = 0; i + 5 < cr.size(); i += 6) {
        if (detail && in_box(inner, 0.5 * (cr[i] + cr[i + 3]), 0.5 * (cr[i + 1] + cr[i + 4]), 0.5 * (cr[i + 2] + cr[i + 5]))) continue;
        glVertex3fv(&cr[i]); glVertex3fv(&cr[i + 3]);
    }
    if (detail) for (size_t i = 0; i + 5 < g_detail_creases.size(); i += 6) { glVertex3fv(&g_detail_creases[i]); glVertex3fv(&g_detail_creases[i + 3]); }
    glEnd();
    glLineWidth(1.0f); glDisable(GL_LINE_SMOOTH); glDisable(GL_BLEND); glEnable(GL_LIGHTING);
}
struct DetailJob { int gen, mgen; std::string text; double h; geo::Box box; };
static unsigned __stdcall detail_thread(void* p) {
    std::unique_ptr<DetailJob> job((DetailJob*)p);
    DetailResult* r = new DetailResult();
    r->gen = job->gen; r->mgen = job->mgen; r->box = job->box; r->h = job->h;
    try {
        geo::Model m; m.load(display_doc(job->text, job->h));
        if (m.bounds.valid()) {
            geo::Grid g = geo::sample(m, job->h, &job->box);
            r->mesh = geo::surface_nets(m, g, 1);
            r->creases = crease_lines(r->mesh, 38.0, false);
            geo::split_sharp(r->mesh, 30.0);
            r->owners = mesh_owners(m, r->mesh);
        }
    } catch (...) {}
    InterlockedExchange(&g_detail_running, 0);
    PostMessageA(g_main, WM_DETAILDONE, 0, (LPARAM)r);
    return 0;
}
static void detail_request() {  // timer 12: the camera has settled; mesh the visible part finely enough for the screen
    KillTimer(g_main, 12);
    if (g_mesh.v.empty() || g_building || g_detail_running) { SetTimer(g_main, 12, 600, nullptr); return; }
    if (g_rest.v.empty()) { if (g_detail_on) { g_detail_on = false; InvalidateRect(g_view, nullptr, FALSE); } return; }  // everything is exact
    geo::Box rb; for (const V3& p : g_rest.v) rb.add(p);
    RECT rc; GetClientRect(g_view, &rc);
    double H = std::max(1L, rc.bottom), W = std::max(1L, rc.right), half = g_dist * std::tan(16.0 * geo::PI / 180);
    double wpp = 2 * half / H, hd = std::max(g_h / 40, 2.2 * wpp);
    auto off = [&]() { if (g_detail_on) { g_detail_on = false; InvalidateRect(g_view, nullptr, FALSE); } };
    if (hd > 0.7 * g_h_used) { off(); return; }
    double R = 1.15 * half * std::max(1.0, W / H);
    geo::Box b; b.lo = V3(std::max(g_tx - R, rb.lo.x - g_h_used), std::max(g_ty - R, rb.lo.y - g_h_used), std::max(g_tz - R, rb.lo.z - g_h_used));
    b.hi = V3(std::min(g_tx + R, rb.hi.x + g_h_used), std::min(g_ty + R, rb.hi.y + g_h_used), std::min(g_tz + R, rb.hi.z + g_h_used));
    if (!(b.hi.x > b.lo.x && b.hi.y > b.lo.y && b.hi.z > b.lo.z)) { off(); return; }
    double vol = (b.hi.x - b.lo.x) * (b.hi.y - b.lo.y) * (b.hi.z - b.lo.z);
    if (vol / (hd * hd * hd) > 5e6) hd = std::cbrt(vol / 5e6);  // about 5 million cells at most
    if (hd > 0.7 * g_h_used) { off(); return; }
    if (g_detail_on && std::fabs(hd - g_detail_h) < 0.1 * hd && geo::len(b.lo - g_detail_box.lo) < 2 * hd && geo::len(b.hi - g_detail_box.hi) < 2 * hd) return;
    // the coarse mesh is hidden inside the detail box, shrunk only where the view cuts the model (on the model's own
    // extent the box edge is the surface itself: a top face there was drawn twice, coarse rim lines over the detail)
    g_detail_inner_req = b;
    for (int a = 0; a < 3; ++a) {
        g_detail_inner_req.lo[a] = b.lo[a] <= rb.lo[a] - 0.99 * g_h_used ? -1e30 : b.lo[a] + g_h_used;
        g_detail_inner_req.hi[a] = b.hi[a] >= rb.hi[a] + 0.99 * g_h_used ? 1e30 : b.hi[a] - g_h_used;
    }
    DetailJob* j = new DetailJob{++g_detail_gen, g_gen, g_rest_full ? js::dump(g_doc, 0) : g_rest_text, hd, b};
    InterlockedExchange(&g_detail_running, 1);
    HANDLE t = (HANDLE)_beginthreadex(nullptr, 0, detail_thread, j, 0, nullptr);
    if (t) CloseHandle(t); else InterlockedExchange(&g_detail_running, 0);
    status("sharpening the view: mesh " + fmtd(hd * 1000) + " mm in view...", 1);
}
static void detail_done(DetailResult* r) {
    std::unique_ptr<DetailResult> keep(r);
    if (r->gen != g_detail_gen || r->mgen != g_gen || r->mesh.t.empty()) return;
    g_detail = r->mesh; g_detail_creases = r->creases; g_detail_box = r->box; g_detail_h = r->h; g_detail_owners = r->owners;
    g_detail_inner = g_detail_inner_req;
    g_coarse_tris.clear(); g_coarse_tris.reserve(g_rest.t.size());
    const geo::Box& inner = g_detail_inner;
    for (size_t q = 0; q + 2 < g_rest.t.size(); q += 3) {
        V3 c = (g_rest.v[(size_t)g_rest.t[q]] + g_rest.v[(size_t)g_rest.t[q + 1]] + g_rest.v[(size_t)g_rest.t[q + 2]]) / 3.0;
        if (in_box(inner, c.x, c.y, c.z)) continue;
        g_coarse_tris.push_back(g_rest.t[q]); g_coarse_tris.push_back(g_rest.t[q + 1]); g_coarse_tris.push_back(g_rest.t[q + 2]);
    }
    g_detail_on = true;
    status("sharp view: mesh " + fmtd(r->h * 1000) + " mm in view", 1);
    InvalidateRect(g_view, nullptr, FALSE);
}

#include "dims.inc"
#include "preview.inc"
#include "asm.inc"
#include "envphys.inc"
#include "check.inc"
#include "power.inc"
#include "look.inc"
#include "flow.inc"
#include "draft.inc"
#include "labs.inc"
static void render() {
    RECT rc; GetClientRect(g_view, &rc);
    int W = std::max(1L, rc.right), H = std::max(1L, rc.bottom);
    if (g_hr_on) { W = g_hr_W; H = g_hr_H; }
    glViewport(0, 0, W, H);
    // background gradient
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glBegin(GL_QUADS);  // a light CAD background: white at the top to a cool grey at the bottom
    glColor3f(0.80f, 0.84f, 0.89f); glVertex2f(0, 0); glVertex2f(1, 0); glColor3f(0.93f, 0.945f, 0.96f); glVertex2f(1, 0.55f); glVertex2f(0, 0.55f);
    glColor3f(0.93f, 0.945f, 0.96f); glVertex2f(0, 0.55f); glVertex2f(1, 0.55f); glColor3f(0.995f, 0.997f, 1.0f); glVertex2f(1, 1); glVertex2f(0, 1);
    glEnd();
    glClear(GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    gluPerspective(32.0, (double)W / H, std::max(0.01, g_dist * 0.01), g_dist * 20);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    double yr = g_yaw * geo::PI / 180, pr = g_pitch * geo::PI / 180;
    double ex = g_tx + g_dist * std::cos(pr) * std::cos(yr), ey = g_ty + g_dist * std::cos(pr) * std::sin(yr), ez = g_tz + g_dist * std::sin(pr);
    gluLookAt(ex, ey, ez, g_tx, g_ty, g_tz, 0, 0, 1);
    glGetDoublev(GL_MODELVIEW_MATRIX, g_mv); glGetDoublev(GL_PROJECTION_MATRIX, g_pj); glGetIntegerv(GL_VIEWPORT, g_vp);
    if (!g_hr_on) {   // when the camera stops moving, mesh what is in view at the screen's resolution
        static double last[7] = {0};
        double now[7] = {g_dist, g_tx, g_ty, g_tz, g_yaw, g_pitch, (double)(W * 10000 + H)};
        bool moved = false; for (int k = 0; k < 7; ++k) if (std::fabs(now[k] - last[k]) > 1e-9) moved = true;
        if (moved) { memcpy(last, now, sizeof now); SetTimer(g_main, 12, 450, nullptr); }
    }
    glEnable(GL_DEPTH_TEST);
    // no ground grid: a clean CAD view
    // lights follow the camera
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_LIGHT1); glEnable(GL_NORMALIZE);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    GLfloat l0[4] = {(float)ex, (float)ey, (float)(ez + g_dist), 1}, l1[4] = {(float)(g_tx - (ex - g_tx)), (float)(g_ty - (ey - g_ty)), (float)(g_tz + 0.3 * g_dist), 1};
    GLfloat dif[4] = {0.72f, 0.72f, 0.72f, 1}, dif1[4] = {0.32f, 0.32f, 0.34f, 1}, amb[4] = {0.36f, 0.36f, 0.38f, 1}, spec[4] = {0.35f, 0.35f, 0.35f, 1};
    glLightfv(GL_LIGHT0, GL_POSITION, l0); glLightfv(GL_LIGHT0, GL_DIFFUSE, dif); glLightfv(GL_LIGHT0, GL_SPECULAR, spec); glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
    glLightfv(GL_LIGHT1, GL_POSITION, l1); glLightfv(GL_LIGHT1, GL_DIFFUSE, dif1);
    glEnable(GL_COLOR_MATERIAL); glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    GLfloat ms[4] = {0.42f, 0.42f, 0.42f, 1}; glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, ms); glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 48);  // a metallic sheen
    if (g_cut) {
        double eq[4] = {0, 0, 0, 0}; eq[g_cutaxis] = -1; eq[3] = g_cutpos;
        glClipPlane(GL_CLIP_PLANE0, eq); glEnable(GL_CLIP_PLANE0);
    }
    glColor3f(0.74f, 0.79f, 0.86f);
    if (g_wire) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(1, 1);
    // clean display: exact surfaces for the main parts, the sampled mesh (refined after a zoom) only for the rest;
    // results painted on the surface, deformations and zebra/curvature use the sampled mesh of the whole design
    bool clean = g_colors.empty() && g_def_scale <= 0;
    bool detail = clean && g_detail_on;
    if (g_asm_on) { asm_render(); flow_render(); clean = false; }
    else if (clean && g_cut && !g_wire) {
        // a section like a CAD program's: the outside shaded, and where the cut opens a part, its inside drawn flat in
        // a darker shade of the part's colour, so every cut face reads as a solid cap of that part
        glEnable(GL_CULL_FACE); glCullFace(GL_BACK);
        draw_scene_parts(detail, 1.0f);
        glCullFace(GL_FRONT); glDisable(GL_LIGHTING);
        draw_scene_parts(detail, 0.72f);
        glEnable(GL_LIGHTING); glDisable(GL_CULL_FACE);
    } else if (clean) draw_scene_parts(detail, 1.0f);
    else if (!g_asm_on) draw_model();
    glDisable(GL_POLYGON_OFFSET_FILL);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (clean && !g_wire) { draw_exact_edges(); draw_creases(detail); }  // true edges, and the creases of the sampled parts
    if (!g_wire && g_colors.empty()) shadow_render();  // the soft shadow on the floor
    // the selected feature: a light tint over it (not a wire mesh)
    if ((!g_selmesh.t.empty() || g_sel >= 0) && !g_view_mode && !g_asm_on) {  // not over zebra or curvature analysis
        glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
        glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-1, -4);
        glColor4f(0.1f, 0.55f, 1.0f, 0.13f);  // a light selection tint: the part keeps its colour
        bool done = false;
        for (const GMesh& M : g_exact) if (M.feature == g_sel) { draw_gmesh(M); done = true; }
        if (!done) draw_mesh(g_selmesh, false);
        glDisable(GL_POLYGON_OFFSET_FILL); glDepthMask(GL_TRUE); glDisable(GL_BLEND); glEnable(GL_LIGHTING);
    }
    glDisable(GL_CLIP_PLANE0);  // the selection tint is clipped by the section too
    field_render();
    viz_render();
    env_render();
    if (g_asm_on) check_render();
    annotations_render();
    dims_render();
    setup_preview_render();
    // section contour
    glDisable(GL_LIGHTING);
    if (g_cut) {
        glLineWidth(1.6f); glColor3f(0.12f, 0.13f, 0.18f);  // the cut outline, thin and dark as in a CAD section
        glBegin(GL_LINES);
        for (auto& s : g_section) {
            double a[3], b[3];
            int ua = g_cutaxis == 0 ? 1 : 0, va = g_cutaxis == 2 ? 1 : 2;
            a[g_cutaxis] = b[g_cutaxis] = g_cutpos; a[ua] = s.u0; a[va] = s.v0; b[ua] = s.u1; b[va] = s.v1;
            glVertex3dv(a); glVertex3dv(b);
        }
        glEnd(); glLineWidth(1.0f);
    }
    // axis triad in the lower-left corner, turning with the view
    glDisable(GL_DEPTH_TEST);
    {
        { int u = g_hr_on ? std::max(1, g_hr_W / std::max(1, g_hr_lW)) : 1; glViewport(8 * u, 8 * u, 76 * u, 76 * u); }
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-1.4, 1.4, -1.4, 1.4, -5, 5);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(std::cos(pr) * std::cos(yr), std::cos(pr) * std::sin(yr), std::sin(pr), 0, 0, 0, 0, 0, 1);
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        glColor3f(0.80f, 0.12f, 0.12f); glVertex3d(0, 0, 0); glVertex3d(1, 0, 0);
        glColor3f(0.12f, 0.58f, 0.12f); glVertex3d(0, 0, 0); glVertex3d(0, 1, 0);
        glColor3f(0.12f, 0.24f, 0.80f); glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
        glEnd();
        glLineWidth(1.0f);
        glColor3f(0.80f, 0.12f, 0.12f); glRasterPos3d(1.15, 0, 0); glListBase(g_font_base - 32); glCallLists(1, GL_UNSIGNED_BYTE, "X");
        glColor3f(0.12f, 0.58f, 0.12f); glRasterPos3d(0, 1.15, 0); glCallLists(1, GL_UNSIGNED_BYTE, "Y");
        glColor3f(0.12f, 0.24f, 0.80f); glRasterPos3d(0, 0, 1.15); glCallLists(1, GL_UNSIGNED_BYTE, "Z");
        glViewport(0, 0, W, H);
    }
    // overlay text (an ultra-clear picture lays it out in the window's units, drawn larger)
    if (g_hr_on) { W = g_hr_lW; H = g_hr_lH; g_vp[0] = 0; g_vp[1] = 0; g_vp[2] = W; g_vp[3] = H; }
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, W, 0, H, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    dims_labels(H);
    setup_preview_overlay(W, H);
    glColor3f(0.30f, 0.33f, 0.38f);
    char buf[256];
    if (g_asm_on) asm_info(buf, sizeof buf);
    else snprintf(buf, sizeof buf, "volume %.3f m3   wetted %.2f m2   length %.2f m   %d triangles   mesh %.3f m%s", g_props.volume, g_props.area, g_props.length, (int)(g_mesh.t.size() / 3), g_h_used, g_cut ? "   [section]" : "");
    gl_text(96, 10, buf);
    field_legend(W, H);
    asm_overlay(W, H);
    env_overlay(W, H);
    view_label(H);
    power_overlay(W, H);
    cam_overlay(W, H);
    flow_overlay(W, H);
    if (!g_result_label.empty() && !g_colors.empty() && g_view_mode == 1) {  // zebra: stripes, no scale
        glColor3f(0.85f, 0.9f, 0.96f);
        gl_text((float)std::max(10, W - 20 - 7 * (int)g_result_label.size()), 60, g_result_label);
    } else if (!g_result_label.empty() && !g_colors.empty()) {
        for (int i = 0; i < 100; ++i) {
            double t = i / 99.0;
            float r = (float)(t < 0.5 ? 2 * t : 1.0), g = (float)(t < 0.5 ? 2 * t : 2 * (1 - t)), bl = (float)(t < 0.5 ? 1.0 : 2 * (1 - t));
            if (g_cmap_turbo) { turbo(t, r, g, bl); glColor3f(r, g, bl); }
            else glColor3f(0.15f + 0.85f * r, 0.15f + 0.85f * g, 0.15f + 0.85f * bl);
            glBegin(GL_QUADS); glVertex2f((float)W - 40, 60.0f + 2 * i); glVertex2f((float)W - 20, 60.0f + 2 * i); glVertex2f((float)W - 20, 62.0f + 2 * i); glVertex2f((float)W - 40, 62.0f + 2 * i); glEnd();
        }
        glColor3f(0.85f, 0.9f, 0.96f);
        gl_text((float)std::max(10, std::min(W - 190, W - 20 - 7 * (int)g_result_label.size())), 262, g_result_label);
        snprintf(buf, sizeof buf, "%.3g", g_rmax); gl_text((float)W - 90, 256 - 12, buf);
        snprintf(buf, sizeof buf, "%.3g", g_rmin); gl_text((float)W - 90, 58, buf);
    }
    if (g_hr_on) return;
    record_frame(W, H);
    snap_frame(W, H);
    SwapBuffers(g_hdc);
}

#include "hq.inc"
static int g_click_x = 0, g_click_y = 0;
static bool g_click_moved = false;
static LRESULT view_proc_inner(HWND h, UINT m, WPARAM w, LPARAM l);
static LRESULT CALLBACK view_proc(HWND h, UINT m, WPARAM w, LPARAM l) {  // the 3D view: errors are shown, never fatal
    try { return view_proc_inner(h, m, w, l); }
    catch (std::exception& e) { status(std::string("3D view: ") + e.what()); }
    catch (...) { status("3D view: unknown error"); }
    return DefWindowProcA(h, m, w, l);
}
static LRESULT view_proc_inner(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            g_hdc = GetDC(h);
            PIXELFORMATDESCRIPTOR pfd; memset(&pfd, 0, sizeof pfd);
            pfd.nSize = sizeof pfd; pfd.nVersion = 1; pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
            pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 24; pfd.cDepthBits = 24; pfd.cStencilBits = 8;  // stencil: the soft floor shadow
            SetPixelFormat(g_hdc, ChoosePixelFormat(g_hdc, &pfd), &pfd);
            g_rc = wglCreateContext(g_hdc); wglMakeCurrent(g_hdc, g_rc);
            SelectObject(g_hdc, g_font);
            g_font_base = glGenLists(96);
            wglUseFontBitmapsA(g_hdc, 32, 96, g_font_base);
            return 0;
        }
        case WM_PAINT: { PAINTSTRUCT ps; BeginPaint(h, &ps); render(); EndPaint(h, &ps); return 0; }
        case WM_ERASEBKGND: return 1;
        case WM_SETCURSOR: if (g_pick_mode && LOWORD(l) == HTCLIENT) { SetCursor(LoadCursor(nullptr, IDC_CROSS)); return TRUE; } break;
        case WM_LBUTTONDOWN: case WM_RBUTTONDOWN: case WM_MBUTTONDOWN:
            if (asm_mouse(m, (short)LOWORD(l), (short)HIWORD(l), w)) return 0;
            if (g_pick_mode && m == WM_LBUTTONDOWN) {
                V3 p, n;
                if (pick_ray((short)LOWORD(l), (short)HIWORD(l), p, n)) pad_pick_result(p, n);
                else status("no surface under the mouse: click on the model");
                return 0;
            }
            if (g_pick_mode && m == WM_RBUTTONDOWN) { g_pick_mode = 0; status("picking cancelled"); return 0; }
            if (m == WM_LBUTTONDOWN) { int d = dims_hit((short)LOWORD(l), (short)HIWORD(l)); if (d >= 0) { dims_begin_edit(d); return 0; } }
            SetFocus(h); SetCapture(h); g_mx = (short)LOWORD(l); g_my = (short)HIWORD(l); g_mbtn = m == WM_LBUTTONDOWN ? 1 : 2;
            g_click_x = g_mx; g_click_y = g_my; g_click_moved = false;
            return 0;
        case WM_LBUTTONUP: case WM_RBUTTONUP: case WM_MBUTTONUP: {
            if (asm_mouse(m, (short)LOWORD(l), (short)HIWORD(l), w)) { g_mbtn = 0; return 0; }
            ReleaseCapture();
            bool was = g_mbtn != 0;
            g_mbtn = 0;
            int x = (short)LOWORD(l), y = (short)HIWORD(l);
            if (was && !g_click_moved && m != WM_MBUTTONUP && g_asm_on) { asm_click(m, x, y); return 0; }
            if (was && !g_click_moved && m != WM_MBUTTONUP) {  // a click, not a drag: pick the part under the mouse
                int k = pick_part(x, y);
                if (m == WM_LBUTTONUP) { select_part(k); if (k < 0) status("nothing under the mouse: click a part"); }
                else if (k >= 0) { select_part(k); POINT p = {x, y}; ClientToScreen(h, &p); part_context_menu(p.x, p.y); }
                InvalidateRect(h, nullptr, FALSE);
            }
            return 0;
        }
        case WM_LBUTTONDBLCLK: if (!g_asm_on && g_sel >= 0) { push_undo(); open_pad(g_sel); } return 0;
        case WM_MOUSEMOVE: {
            if (asm_mouse(m, (short)LOWORD(l), (short)HIWORD(l), w)) return 0;
            int x = (short)LOWORD(l), y = (short)HIWORD(l), dx = x - g_mx, dy = y - g_my;
            g_mx = x; g_my = y;
            if (g_mbtn && (std::abs(x - g_click_x) > 3 || std::abs(y - g_click_y) > 3)) g_click_moved = true;
            if (!g_mbtn) { int d = dims_hit(x, y); if (d != g_dim_hover) { g_dim_hover = d; InvalidateRect(h, nullptr, FALSE); if (d >= 0) status("click to change the " + g_dims[(size_t)d].label); } }
            if (g_mbtn == 1) { g_yaw -= dx * 0.4; g_pitch = geo::clampd(g_pitch + dy * 0.4, -89, 89); InvalidateRect(h, nullptr, FALSE); }
            else if (g_mbtn == 2) {
                double yr = g_yaw * geo::PI / 180, s = g_dist * 0.0015;
                g_tx += (-std::sin(yr) * -dx) * s * -1; g_ty += (std::cos(yr) * -dx) * s * -1; g_tz += dy * s;
                InvalidateRect(h, nullptr, FALSE);
            }
            return 0;
        }
        case WM_MOUSEWHEEL:
            if (GetKeyState(VK_CONTROL) < 0 && g_vol_ok) { slice_move((short)HIWORD(w) > 0 ? 1 : -1); InvalidateRect(h, nullptr, FALSE); return 0; }
            g_dist *= std::pow(0.88, (short)HIWORD(w) / 120.0); InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_KEYDOWN: {
            bool shift = GetKeyState(VK_SHIFT) < 0, ctrl = GetKeyState(VK_CONTROL) < 0;
            double st = shift ? 1.0 : 0.1;
            switch (w) {
                case VK_LEFT: nudge(0, -st); break;
                case VK_RIGHT: nudge(0, st); break;
                case VK_UP: nudge(1, st); break;
                case VK_DOWN: nudge(1, -st); break;
                case VK_PRIOR: nudge(2, st); break;
                case VK_NEXT: nudge(2, -st); break;
                case VK_DELETE: SendMessageA(g_main, WM_COMMAND, ID_EDIT_DEL, 0); break;
                case VK_OEM_PLUS: case VK_ADD: modify_selected([](js::Value& o) { o["scale"] = o.num("scale", 1) * 1.05; }); break;
                case VK_OEM_MINUS: case VK_SUBTRACT: modify_selected([](js::Value& o) { o["scale"] = o.num("scale", 1) / 1.05; }); break;
                case 'R': rotate_sel(2, shift ? -5 : 5); break;
                case 'T': rotate_sel(0, shift ? -5 : 5); break;
                case 'W': g_wire = !g_wire; break;
                case 'G': g_shadow_on = !g_shadow_on; status(g_shadow_on ? "floor shadow on (G)" : "floor shadow off (G)"); break;
                case VK_SPACE: if (g_asm_on) ms_play(); break;
                case 'E': if (g_asm_on) asm_toggle_explode(); break;
                case 'C': SendMessageA(g_main, WM_COMMAND, ID_VIEW_CUT, 0); break;
                case 'X': SendMessageA(g_main, WM_COMMAND, ID_VIEW_CUTX, 0); break;
                case 'Y': if (ctrl) undo(true); else SendMessageA(g_main, WM_COMMAND, ID_VIEW_CUTY, 0); break;
                case 'Z': if (ctrl) undo(false); else SendMessageA(g_main, WM_COMMAND, ID_VIEW_CUTZ, 0); break;
                case 'D': if (ctrl) SendMessageA(g_main, WM_COMMAND, ID_EDIT_DUP, 0); break;
                case VK_OEM_4: g_cutpos -= shift ? 1.0 : 0.1; compute_section(); break;
                case VK_OEM_6: g_cutpos += shift ? 1.0 : 0.1; compute_section(); break;
                case 'F': fit_view(); break;
                case 'H': SendMessageA(g_main, WM_COMMAND, ID_VIEW_RES, 0); break;
                case '0': g_yaw = -135; g_pitch = 25; break;
                case '1': g_yaw = 180; g_pitch = 0; break;
                case '2': g_yaw = 0; g_pitch = 0; break;
                case '3': g_yaw = -90; g_pitch = 0; break;
                case '4': g_yaw = 90; g_pitch = 0; break;
                case '5': g_yaw = -90; g_pitch = 89; break;
                case '6': g_yaw = -90; g_pitch = -89; break;
            }
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        }
        case WM_DESTROY: wglMakeCurrent(nullptr, nullptr); wglDeleteContext(g_rc); ReleaseDC(h, g_hdc); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

// ------------------------------------------------------------------ tree (right-click), best wing, task window entries
static WNDPROC g_list_old = nullptr;
static LRESULT CALLBACK tree_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_RBUTTONDOWN && g_asm_on) {
        int i = (int)SendMessageA(h, LB_ITEMFROMPOINT, 0, l);
        if (HIWORD(i) == 0) {
            SendMessageA(h, LB_SETCURSEL, LOWORD(i), 0); asm_tree_select(LOWORD(i));
            POINT p = {(short)LOWORD(l), (short)HIWORD(l)}; ClientToScreen(h, &p);
            if (g_asm_mate_sel >= 0) asm_mate_context(p.x, p.y); else asm_component_menu(p.x, p.y);
        }
        return 0;
    }
    if (m == WM_RBUTTONDOWN) {
        int i = (int)SendMessageA(h, LB_ITEMFROMPOINT, 0, l);
        if (HIWORD(i) == 0 && LOWORD(i) < (int)features().size()) {
            SendMessageA(h, LB_SETCURSEL, LOWORD(i), 0); select_part(LOWORD(i));
            POINT p = {(short)LOWORD(l), (short)HIWORD(l)}; ClientToScreen(h, &p); part_context_menu(p.x, p.y);
        }
        return 0;
    }
    return CallWindowProcA(g_list_old, h, m, w, l);
}
static void apply_best_wing() {  // the best wing of the newest CFD optimisation, into the editor (Ctrl+Z goes back)
    std::string d = newest_results("best_design.json");
    if (d.empty()) { MessageBoxA(g_main, "No CFD wing optimisation has finished yet. Run Optimise (Simulate or Optimise tab) first.", "Apply best wing", MB_ICONINFORMATION); return; }
    try {
        js::Value best = js::parse(js::read_file(d + "\\best_design.json"));
        if (!best.find("features")) throw std::runtime_error("best_design.json has no features");
        push_undo();
        g_doc = best; g_modified = true;
        fill_list(); fill_props(); rebuild(); update_title();
        std::string note;
        try { note = js::read_file(d + "\\summary.txt"); } catch (...) {}
        bool warn = note.find("WARNING") != std::string::npos || note.find("NOT RELIABLE") != std::string::npos;
        status("best wing applied from " + d + (warn ? "  (its study carries a reliability warning: read its summary)" : "  (Ctrl+Z restores the previous design)"));
        if (warn) MessageBoxA(g_main, ("The study this wing comes from carries a warning:\n\n" + note.substr(0, 900)).c_str(), "Apply best wing", MB_ICONWARNING);
    } catch (std::exception& e) { MessageBoxA(g_main, e.what(), "Apply best wing", MB_ICONERROR); }
}
static void jobs_show_custom(const std::string& args, const std::string& about) {
    open_centre(-1);
    if (g_job_proc) { MessageBoxA(g_jobs, "A task is running in the task window. Stop it or wait for it, then pick again.", "Applications", MB_ICONINFORMATION); return; }
    SendMessageA(g_job_list, LB_SETCURSEL, (WPARAM)-1, 0);
    SetWindowTextA(g_job_args, args.c_str());
    SetWindowTextA(g_job_desc, about.c_str());
}

// ------------------------------------------------------------------ main window
static HMENU make_menu() {
    HMENU bar = CreateMenu(), f = CreatePopupMenu(), e = CreatePopupMenu(), ins = CreatePopupMenu(), mod = CreatePopupMenu(), v = CreatePopupMenu(), t = CreatePopupMenu(), an = CreatePopupMenu(), hp = CreatePopupMenu();
    AppendMenuA(f, MF_STRING, ID_FILE_NEW, "&New\tCtrl+N"); AppendMenuA(f, MF_STRING, ID_FILE_OPEN, "&Open...\tCtrl+O"); AppendMenuA(f, MF_STRING, ID_FILE_SAVE, "&Save\tCtrl+S");
    AppendMenuA(f, MF_STRING, ID_FILE_SAVEAS, "Save &as..."); AppendMenuA(f, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(f, MF_STRING, ID_FILE_STL, "Export &STL..."); AppendMenuA(f, MF_STRING, ID_FILE_OBJ, "Export &OBJ..."); AppendMenuA(f, MF_STRING, ID_FILE_SVG, "Export section as S&VG...");
    AppendMenuA(f, MF_STRING, ID_EXP_PACKAGE, "Export CAD &package (STEP, glTF, DXF, drawing)");
    AppendMenuA(f, MF_STRING, ID_EXP_PDF, "Project report as P&DF..."); AppendMenuA(f, MF_STRING, ID_EXP_DOCX, "Project report as &Word..."); AppendMenuA(f, MF_STRING, ID_EXP_TXT, "Project report as te&xt...");
    AppendMenuA(f, MF_SEPARATOR, 0, nullptr); AppendMenuA(f, MF_STRING, ID_FILE_EXIT, "E&xit");
    AppendMenuA(e, MF_STRING, ID_EDIT_UNDO, "&Undo\tCtrl+Z"); AppendMenuA(e, MF_STRING, ID_EDIT_REDO, "&Redo\tCtrl+Y"); AppendMenuA(e, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(e, MF_STRING, ID_EDIT_DEL, "&Delete feature\tDel"); AppendMenuA(e, MF_STRING, ID_EDIT_DUP, "D&uplicate feature\tCtrl+D");
    AppendMenuA(e, MF_STRING, ID_EDIT_UP, "Move &up in the tree"); AppendMenuA(e, MF_STRING, ID_EDIT_DOWN, "Move do&wn in the tree");
    AppendMenuA(e, MF_STRING, ID_EDIT_TOGGLE, "&Suppress / restore"); AppendMenuA(e, MF_STRING, ID_EDIT_OP, "Cycle &operation (add, cut, intersect)");
    std::vector<std::string> kinds = geo::feature_kinds();
    for (size_t i = 0; i < kinds.size(); ++i) AppendMenuA(ins, MF_STRING, ID_INS_BASE + (int)i, kinds[i].c_str());
    AppendMenuA(ins, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(ins, MF_STRING, ID_INS_AIRFOIL, "Wing from &airfoil file (.dat)...");
    AppendMenuA(ins, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(ins, MF_STRING, ID_SK_REVOLVE, "Sketch and &revolve..."); AppendMenuA(ins, MF_STRING, ID_SK_EXTRUDE, "Sketch and e&xtrude..."); AppendMenuA(ins, MF_STRING, ID_SK_LOFT, "Sketch and &loft...");
    AppendMenuA(ins, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(ins, MF_STRING, ID_INS_SLOT, "Cut: &slot (box)"); AppendMenuA(ins, MF_STRING, ID_INS_HOLE, "Cut: &hole (cylinder)"); AppendMenuA(ins, MF_STRING, ID_INS_CAVITY, "Cut: hollow &cavity (ellipsoid)");
    AppendMenuA(mod, MF_STRING, ID_MOD_SHELL, "Make &hollow (shell 4 cm)"); AppendMenuA(mod, MF_STRING, ID_MOD_PATTERN, "Linear &pattern (3 copies)");
    AppendMenuA(mod, MF_STRING, ID_MOD_MIRROR, "Toggle &mirror"); AppendMenuA(mod, MF_STRING, ID_MOD_BLEND, "Add &blend (fillet 10 cm)"); AppendMenuA(mod, MF_STRING, ID_MOD_TWIST, "Add &twist (5 deg/m)");
    AppendMenuA(mod, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(mod, MF_STRING, ID_MOD_TAPER, "T&aper along x (cross-section grows 5 %/m)"); AppendMenuA(mod, MF_STRING, ID_MOD_BEND, "Be&nd (curvature 0.05 /m)");
    AppendMenuA(mod, MF_STRING, ID_MOD_SHEAR, "&Shear (sweep the cross-sections, 0.2)"); AppendMenuA(mod, MF_STRING, ID_MOD_STRETCH, "St&retch (non-uniform scale 1, 1.2, 1)");
    AppendMenuA(mod, MF_STRING, ID_MOD_PIVOT, "Rotate about a &pivot (variable sweep, folding: 20 deg)"); AppendMenuA(mod, MF_STRING, ID_MOD_INFILL, "Hollow with &gyroid lattice infill");
    AppendMenuA(mod, MF_STRING, ID_MOD_CLEARX, "&Clear all transformations");
    AppendMenuA(v, MF_STRING, ID_VIEW_ISO, "&Isometric\t0"); AppendMenuA(v, MF_STRING, ID_VIEW_FRONT, "&Front\t1"); AppendMenuA(v, MF_STRING, ID_VIEW_SIDE, "&Side\t3"); AppendMenuA(v, MF_STRING, ID_VIEW_TOP, "&Top\t5");
    AppendMenuA(v, MF_STRING, ID_VIEW_FIT, "Fi&t\tF"); AppendMenuA(v, MF_STRING, ID_VIEW_WIRE, "&Wireframe\tW"); AppendMenuA(v, MF_STRING, ID_VIEW_SHADOW, "Floor &shadow\tG"); AppendMenuA(v, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(v, MF_STRING, ID_VIEW_CUT, "Section &cut\tC"); AppendMenuA(v, MF_STRING, ID_VIEW_CUTX, "Cut normal to &x\tX"); AppendMenuA(v, MF_STRING, ID_VIEW_CUTY, "Cut normal to &y\tY"); AppendMenuA(v, MF_STRING, ID_VIEW_CUTZ, "Cut normal to &z\tZ");
    AppendMenuA(v, MF_SEPARATOR, 0, nullptr); AppendMenuA(v, MF_STRING, ID_VIEW_RES, "Mesh &resolution (draft, normal, fine)\tH"); AppendMenuA(v, MF_STRING, ID_VIEW_RESULTS_OFF, "Hide &results colours");
    AppendMenuA(t, MF_STRING, ID_TOOL_MEASURE, "&Measure: mass properties"); AppendMenuA(t, MF_STRING, ID_TOOL_CHECK, "&Check design");
    AppendMenuA(an, MF_STRING, ID_AN_QUICK, "&Quick aerodynamics (lattice, area rule, friction)"); AppendMenuA(an, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(an, MF_STRING, ID_AN_CFD_CRUISE, "3D flow: cruise M0.85, 2 deg, 11 km"); AppendMenuA(an, MF_STRING, ID_AN_CFD_TRANS, "3D flow: transonic turn M0.95, 6 deg, 5 km");
    AppendMenuA(an, MF_STRING, ID_AN_CFD_SUPER, "3D flow: supersonic M1.6, 2 deg, 12 km"); AppendMenuA(an, MF_STRING, ID_AN_CFD_ALPHA, "3D flow: high incidence M0.3, 25 deg, 6 km");
    AppendMenuA(an, MF_SEPARATOR, 0, nullptr); AppendMenuA(an, MF_STRING, ID_AN_LOAD, "&Load surface results (surface.csv)...");
    AppendMenuA(an, MF_STRING, ID_RES_CP, "Colour by pressure"); AppendMenuA(an, MF_STRING, ID_RES_CPRMS, "Colour by pressure fluctuation");
    AppendMenuA(an, MF_STRING, ID_RES_CF, "Colour by skin friction"); AppendMenuA(an, MF_STRING, ID_RES_T, "Colour by wall temperature");
    AppendMenuA(an, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(an, MF_STRING, ID_SIM_FIELD, "Show 3D &flow field (Mach slices)"); AppendMenuA(an, MF_STRING, ID_SIM_PARTICLES, "Animated flow &particles");
    AppendMenuA(an, MF_STRING, ID_SIM_SETUP, "3D flow &setup (every condition)...");
    AppendMenuA(an, MF_STRING, ID_SIM_TURB, "&Turbulence and flow instability...");
    AppendMenuA(an, MF_STRING, ID_SIM_OPT, "&Optimise the wing with CFD...");
    AppendMenuA(an, MF_STRING, ID_SIM_ANALYTIC, "&Analytic simulation (panel, lifting line, area rule, shocks, heating)");
    AppendMenuA(an, MF_STRING, ID_SIM_PINN, "Simulation with the &PINN (skin heating)"); AppendMenuA(an, MF_STRING, ID_SIM_STRUCT, "&Structural simulation (FEM, flutter)");
    AppendMenuA(an, MF_STRING, ID_SIM_VIEWER, "Simulation &results viewer...");
    AppendMenuA(hp, MF_STRING, ID_HELP_KEYS, "&Mouse and keys"); AppendMenuA(hp, MF_STRING, ID_HELP_ABOUT, "&About AeroSynapse Studio");
    AppendMenuA(bar, MF_POPUP, (UINT_PTR)f, "&File"); AppendMenuA(bar, MF_POPUP, (UINT_PTR)e, "&Edit"); AppendMenuA(bar, MF_POPUP, (UINT_PTR)ins, "&Insert");
    AppendMenuA(bar, MF_POPUP, (UINT_PTR)mod, "&Modify"); AppendMenuA(bar, MF_POPUP, (UINT_PTR)v, "&View"); AppendMenuA(bar, MF_POPUP, (UINT_PTR)t, "&Tools");
    HMENU wfm = CreatePopupMenu();
    AppendMenuA(wfm, MF_STRING, ID_WFM_WINDOW, "&Design workflow...\tF9"); AppendMenuA(wfm, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(wfm, MF_STRING, ID_WFM_NEW, "&New project from this design..."); AppendMenuA(wfm, MF_STRING, ID_WFM_OPEN, "&Open project...");
    AppendMenuA(wfm, MF_SEPARATOR, 0, nullptr);
    for (int k = 0; k < NSTAGE; ++k) { char b[96]; snprintf(b, sizeof b, "Run stage &%d: %s", k + 1, STAGE_LIST[k].title); AppendMenuA(wfm, MF_STRING, ID_WFM_STAGE0 + k, b); }
    AppendMenuA(wfm, MF_STRING, ID_WFM_NEXT, "Run &all remaining stages"); AppendMenuA(wfm, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(wfm, MF_STRING, ID_WFM_REPORT, "Open the project &report");
    AppendMenuA(bar, MF_POPUP, (UINT_PTR)wfm, "Wor&kflow");
    HMENU op = CreatePopupMenu();
    AppendMenuA(op, MF_STRING, ID_OPT_CENTRE, "&Optimisation and simulation centre...");
    AppendMenuA(op, MF_SEPARATOR, 0, nullptr);
    std::string grp;
    HMENU sub = nullptr;
    for (int k = 0; k < NTASK; ++k) {
        if (grp != TASKS[k].group) { grp = TASKS[k].group; sub = CreatePopupMenu(); AppendMenuA(op, MF_POPUP, (UINT_PTR)sub, TASKS[k].group); }
        AppendMenuA(sub, MF_STRING, ID_OPT_TASK0 + k, TASKS[k].name);
    }
    AppendMenuA(hp, MF_SEPARATOR, 0, nullptr); AppendMenuA(hp, MF_STRING, ID_PREFS, "&Preferences (charts, 3D display, tool bar)...");
    AppendMenuA(bar, MF_POPUP, (UINT_PTR)an, "&Analysis"); AppendMenuA(bar, MF_POPUP, (UINT_PTR)op, "&Optimise"); AppendMenuA(bar, MF_POPUP, (UINT_PTR)hp, "&Help");
    return bar;
}

static void layout() {
    RECT rc; GetClientRect(g_main, &rc);
    SendMessageA(g_status, WM_SIZE, 0, 0);
    RECT sr; GetWindowRect(g_status, &sr);
    int sh = sr.bottom - sr.top, W = rc.right, H = rc.bottom - sh;
    int lw = g_show_tree ? 270 : 0, pw = g_show_props ? 320 : 0, bh = 26, top = SCHEME_H;
    int x0 = 0;  // the ribbon holds the tools: no tool palette
    MoveWindow(g_schemebar, 0, 0, W, top, TRUE);
    ShowWindow(g_palette, SW_HIDE);
    ShowWindow(g_list, g_show_tree ? SW_SHOW : SW_HIDE);
    for (int i = 0; i < 6; ++i) ShowWindow(g_btn[i], g_show_tree ? SW_SHOW : SW_HIDE);
    HWND props[] = {g_card, g_apply, g_json, g_applyjson};
    for (HWND hp : props) ShowWindow(hp, g_show_props ? SW_SHOW : SW_HIDE);
    if (!g_show_props) for (int r = 0; r < NROWS; ++r) { ShowWindow(g_lbl[r], SW_HIDE); ShowWindow(g_edt[r], SW_HIDE); ShowWindow(g_cmb[r], SW_HIDE); }
    MoveWindow(g_list, x0 + 4, top + TREE_HEAD + 2, lw - 8, H - top - TREE_HEAD - 2 * (bh + 3) - 10, TRUE);
    { RECT hr = {0, top, lw + 4, top + TREE_HEAD + 2}; InvalidateRect(g_main, &hr, TRUE); }
    for (int i = 0; i < 6; ++i) MoveWindow(g_btn[i], x0 + 4 + (i % 3) * ((lw - 8) / 3), H - 2 * (bh + 3) - 4 + (i / 3) * (bh + 3), (lw - 8) / 3 - 2, bh, TRUE);
    MoveWindow(g_view, x0 + lw, top, W - x0 - lw - pw, H - top, TRUE);
    props_layout();
    viewbar_place();
}

// the property panel: info card, the rows in use (squeezed to fit), Apply, the feature JSON and Apply JSON, never overlapping
static void props_layout() {
    if (!g_main || !g_card || !g_show_props) return;
    RECT rc; GetClientRect(g_main, &rc);
    RECT sr; GetWindowRect(g_status, &sr);
    int W = rc.right, H = rc.bottom - (sr.bottom - sr.top), pw = 320, top = SCHEME_H;
    MoveWindow(g_card, W - pw + 2, top + 2, pw - 4, CARD_H, TRUE);
    int x = W - pw + 4, y = top + CARD_H + 6, n = std::max(1, g_rows_used);
    int json_min = 110, avail = H - y - 32 - json_min - 36;
    int rh = std::max(15, std::min(21, avail / n));
    int shown = std::min(n, std::max(0, avail / rh));
    for (int r = 0; r < NROWS; ++r) {
        bool vis = r < g_rows_used && r < shown;
        MoveWindow(g_lbl[r], x, y + r * rh + 3, 118, rh - 2, TRUE); MoveWindow(g_edt[r], x + 121, y + r * rh, pw - 129, rh - 1, TRUE);
        MoveWindow(g_cmb[r], x + 121, y + r * rh - 1, pw - 129, 220, TRUE);
        if (r < g_rows_used) {
            ShowWindow(g_lbl[r], vis ? SW_SHOW : SW_HIDE);
            ShowWindow(g_edt[r], vis && !g_row_combo[r] ? SW_SHOW : SW_HIDE); ShowWindow(g_cmb[r], vis && g_row_combo[r] ? SW_SHOW : SW_HIDE);
        }
    }
    int yb = y + std::min(n, shown) * rh + 4;
    MoveWindow(g_apply, x, yb, pw - 8, 28, TRUE);
    MoveWindow(g_json, x, yb + 32, pw - 8, std::max(60, H - yb - 70), TRUE);
    MoveWindow(g_applyjson, x, H - 33, pw - 8, 28, TRUE);
}

static void about() {
    MessageBoxA(g_main,
                "AeroSynapse Studio 2.0\n\n"
                "Design: sketcher (revolve, extrude, loft), bodies, wings and primitives, curves and surfaces, equation surfaces,\n"
                "superellipsoids, parametric sweeps and lattices, with cuts, fillets, shells, patterns and transformations.\n\n"
                "Simulate: 3D flow (with the Mach field and particles in the 3D view), analytic solutions, the parametric PINN,\n"
                "finite elements and flutter. Optimise: airfoil, body, wing and whole-aircraft optimisers.\n\n"
                "Workflow (F9): basic design to export in eight stages; reports as PDF, Word, text and web;\n"
                "CAD export as STEP, glTF, STL, OBJ, DXF and drawings.\n\n"
                "Part of AeroSynapse. Research and education use.",
                "About AeroSynapse Studio", MB_ICONINFORMATION);
}

static LRESULT main_proc_inner(HWND h, UINT m, WPARAM w, LPARAM l);
// any error in a command, timer or result handler is shown and the app keeps running (it used to close Studio)
static LRESULT CALLBACK main_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    try { return main_proc_inner(h, m, w, l); }
    catch (std::exception& e) { g_cmd_id = 0; status(std::string("error: ") + e.what()); MessageBoxA(h, e.what(), "AeroSynapse Studio: the action did not complete", MB_ICONERROR); }
    catch (...) { g_cmd_id = 0; MessageBoxA(h, "unknown error", "AeroSynapse Studio: the action did not complete", MB_ICONERROR); }
    return 0;
}
static LRESULT main_proc_inner(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            g_main = h;
            HINSTANCE hi = GetModuleHandleA(nullptr);
            g_status = CreateWindowExA(0, STATUSCLASSNAMEA, "", WS_CHILD | WS_VISIBLE | SBARS_SIZEGRIP, 0, 0, 0, 0, h, (HMENU)ID_STATUS, hi, nullptr);
            int parts[2] = {700, -1}; SendMessageA(g_status, SB_SETPARTS, 2, (LPARAM)parts);
            g_list = CreateWindowExA(0, "LISTBOX", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_CLIPSIBLINGS | LBS_NOTIFY | LBS_NOINTEGRALHEIGHT | LBS_OWNERDRAWFIXED | LBS_HASSTRINGS,
                                     0, 0, 0, 0, h, (HMENU)ID_LIST, hi, nullptr);
            g_list_old = (WNDPROC)SetWindowLongPtrA(g_list, GWLP_WNDPROC, (LONG_PTR)tree_proc);
            const char* bl[6] = {"Up", "Down", "Delete", "Duplicate", "On/Off", "+ / - / &&"};
            int bid[6] = {ID_BTN_UP, ID_BTN_DOWN, ID_BTN_DEL, ID_BTN_DUP, ID_BTN_TOGGLE, ID_BTN_OP};
            for (int i = 0; i < 6; ++i) g_btn[i] = CreateWindowExA(0, "BUTTON", bl[i], WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)(INT_PTR)bid[i], hi, nullptr);
            for (int r = 0; r < NROWS; ++r) {
                g_lbl[r] = CreateWindowExA(0, "STATIC", "", WS_CHILD, 0, 0, 0, 0, h, nullptr, hi, nullptr);
                g_edt[r] = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | ES_AUTOHSCROLL, 0, 0, 0, 0, h, (HMENU)(INT_PTR)(ID_ROWEDT0 + r), hi, nullptr);
                g_cmb[r] = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VSCROLL | CBS_DROPDOWNLIST, 0, 0, 0, 0, h, (HMENU)(INT_PTR)(ID_ROWCMB0 + r), hi, nullptr);
                SendMessageA(g_cmb[r], WM_SETFONT, (WPARAM)g_font, TRUE);
            }
            g_apply = CreateWindowExA(0, "BUTTON", "Apply properties", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)ID_APPLY, hi, nullptr);
            g_json = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN, 0, 0, 0, 0, h, (HMENU)ID_JSON, hi, nullptr);
            g_applyjson = CreateWindowExA(0, "BUTTON", "Apply feature JSON", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)ID_APPLYJSON, hi, nullptr);
            g_view = CreateWindowExA(0, "AeroSynapseView", "", WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_schemebar = CreateWindowExA(0, "AeroSynapseSchemes", "", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_palette = CreateWindowExA(0, "AeroSynapsePalette", "", WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_card = CreateWindowExA(0, "AeroSynapseCard", "", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_viewbar = CreateWindowExA(0, "AeroSynapseViewbar", "", WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS, 0, 0, 10, 10, h, nullptr, hi, nullptr);
            HWND all[] = {g_status, g_list, g_apply, g_json, g_applyjson};
            for (HWND x : all) SendMessageA(x, WM_SETFONT, (WPARAM)g_font, TRUE);
            for (int i = 0; i < 6; ++i) SendMessageA(g_btn[i], WM_SETFONT, (WPARAM)g_font, TRUE);
            for (int r = 0; r < NROWS; ++r) { SendMessageA(g_lbl[r], WM_SETFONT, (WPARAM)g_font, TRUE); SendMessageA(g_edt[r], WM_SETFONT, (WPARAM)g_font, TRUE); }
            return 0;
        }
        case WM_SIZE: layout(); return 0;
        case WM_GETMINMAXINFO: { MINMAXINFO* mm = (MINMAXINFO*)l; mm->ptMinTrackSize.x = 1200; mm->ptMinTrackSize.y = 720; return 0; }
        case WM_DETAILDONE: detail_done((DetailResult*)l); return 0;
        case WM_ASM_MESHED: asm_meshed(w, l); return 0;
        case WM_MESHDONE: {
            MeshResult* r = (MeshResult*)l;
            g_building = 0;
            if (g_asm_on) { delete r; return 0; }  // a part's mesh finishing after an assembly was opened
            if (r->gen == g_gen || g_mesh.v.empty()) {
                bool first = g_mesh.v.empty() || g_fit_next;
                if (r->gen == g_gen) g_fit_next = false;
                g_mesh = r->mesh; g_selmesh = r->sel; g_props = r->props; g_colors.clear(); g_result_label.clear();
                g_creases = r->creases; g_detail_on = false; ++g_detail_gen; SetTimer(g_main, 12, 300, nullptr);
                g_exact.clear();
                for (const geo::TMesh& t : r->exact) {
                    GMesh G; G.feature = t.feature; G.t = t.t; G.e = t.edges;
                    G.v.reserve(t.v.size() * 3); G.n.reserve(t.v.size() * 3);
                    for (size_t i = 0; i < t.v.size(); ++i) { G.v.insert(G.v.end(), {(float)t.v[i].x, (float)t.v[i].y, (float)t.v[i].z}); G.n.insert(G.n.end(), {(float)t.n[i].x, (float)t.n[i].y, (float)t.n[i].z}); }
                    g_exact.push_back(std::move(G));
                }
                g_rest_full = r->rest_full; g_rest_text = r->rest_text; g_h_used = r->h;
                if (r->gen == g_gen && !r->error.empty() && features().size()) g_st_last_error = r->error;
                g_rest = r->rest_full ? r->mesh : r->rest;
                g_mesh_owners = r->owners;
                g_rest_owners = r->rest_full ? r->owners : r->rest_owners;
                g_rest_creases = r->rest_full ? r->creases : r->rest_creases;
                if (first) { fit_view(); g_cutpos = 0.5 * (g_props.lo[g_cutaxis] + g_props.hi[g_cutaxis]); }
                if (!g_pending_results.empty() && r->gen == g_gen && !g_dirty_build) { std::string pr = g_pending_results; g_pending_results.clear(); load_surface_csv(pr); }
                if (!g_pending_field.empty() && r->gen == g_gen && !g_dirty_build && field_load(g_pending_field)) {
                    g_pending_field.clear(); g_field_on = g_particles_on = true; SetTimer(g_main, 7, 33, nullptr);
                    if (g_pending_flowlines) { g_pending_flowlines = false; g_particles_on = false; flowlines_toggle(); }
                    for (int k = 0; k < 160; ++k) field_step();
                }
                if (g_live && !g_job.hProcess && r->gen == g_gen && !g_dirty_build && !g_kin_on) SetTimer(g_main, 8, 2500, nullptr);  // live simulation after edits
                if (g_view_mode) analysis_colours();
                card_refresh();
                if (g_pending_view >= 0) { double yaws[4] = {-135, 180, -90, -90}, pitches[4] = {25, 0, 0, 89}; g_yaw = yaws[g_pending_view % 4]; g_pitch = pitches[g_pending_view % 4]; g_pending_view = -1; }
                if (features().size() == 0) status("empty design: pick Sketcher > Plane XY, XZ or YZ and draw your first body, or add a part from the palette");
                else status(r->error.empty() ? "ready" : "design error: " + r->error);
                status("", 1);
            }
            delete r;
            if (g_dirty_build) rebuild();
            InvalidateRect(g_view, nullptr, FALSE);
            return 0;
        }
        case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkColor(dc, C_PANEL); SetTextColor(dc, C_MUTED); return (LRESULT)g_br_panel; }
        case WM_CTLCOLOREDIT: case WM_CTLCOLORLISTBOX: { HDC dc = (HDC)w; SetBkColor(dc, C_FIELD); SetTextColor(dc, C_INK); return (LRESULT)g_br_field; }
        case WM_MEASUREITEM: { MEASUREITEMSTRUCT* mi = (MEASUREITEMSTRUCT*)l; if (mi->CtlID == ID_LIST) { mi->itemHeight = 24; return TRUE; } break; }
        case WM_PAINT: {  // the main window shows only the tree's header between its children
            PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps);
            FillRect(dc, &ps.rcPaint, g_br_panel);
            if (g_show_tree) tree_paint_header(dc, 4, SCHEME_H + 1, 262);
            EndPaint(h, &ps);
            return 0;
        }
        case WM_DRAWITEM: {
            DRAWITEMSTRUCT* d = (DRAWITEMSTRUCT*)l;
            if (d->CtlType == ODT_LISTBOX && d->CtlID == ID_LIST) { tree_draw_item(d); return TRUE; }
            if (d->CtlType != ODT_BUTTON) break;
            bool primary = d->CtlID == ID_APPLY || d->CtlID == ID_APPLYJSON;
            if (primary) draw_button(d, gcol(28, 92, 168));
            else draw_button_light(d, d->CtlID == ID_BTN_DEL);
            return TRUE;
        }
        case WM_TIMER:
            if (w == 1 && g_job.hProcess && WaitForSingleObject(g_job.hProcess, 0) == WAIT_OBJECT_0) job_done();
            if (w == 7 && (g_particles_on || !g_vlines.empty() || g_sl_on || g_sweep_on || g_rec_on)) { if (g_particles_on) field_step(); sweep_tick(); g_phase += 0.012; InvalidateRect(g_view, nullptr, FALSE); }
            if (w == 9) kin_tick();
            if (w == ASM_TIMER) ms_tick();
            if (w == 42) env_run_tick();
            if (w == 43) power_run_tick();
            if (w == 44) flow_tick();
            if (w == 45) flow_run_tick();
            if (w == 46) { KillTimer(h, 46); RECT r; GetClientRect(g_view, &r); bool ok = hq_save_png(g_pending_hq, 3840.0 / std::max(1L, r.right), 2, true); status((ok ? "ultra-clear image saved: " : "the ultra-clear image failed: ") + g_pending_hq); }
            if (w == 10) palette_tick();
            if (w == 12) detail_request();
            if (w == 13) selftest_tick();
            if (w == 11) palette_leave_tick();
            if (w == 8) { KillTimer(g_main, 8); if (g_live && !g_job.hProcess) run_cfd_case("cruise", "live"); }
            return 0;
        case WM_TOOL_INSERT: {
            int ti = (int)w;
            if (ti >= 0 && ti < NTOOL && TOOLS[ti].kind) { insert(geo::template_for(TOOLS[ti].kind)); status(std::string("inserted ") + TOOLS[ti].label + ": edit it on the right, or drag with the arrow keys"); }
            return 0;
        }
        case WM_COMMAND: {
            int id = LOWORD(w);
            if (!l) g_cmd_id = id;  // menus, accelerators and tools (control notifications carry their window in l)
            if (id >= ID_ASM_T0 && id < ID_ASM_T0 + 60) { asm_tool(id - ID_ASM_T0); return 0; }
            if (asm_command(id)) return 0;
            if (g_asm_on && id == ID_LIST && HIWORD(w) == LBN_SELCHANGE) { asm_tree_select((int)SendMessageA(g_list, LB_GETCURSEL, 0, 0)); return 0; }
            if (g_asm_on && id == ID_LIST && HIWORD(w) == LBN_DBLCLK) return 0;
            if (id == ID_LIST && HIWORD(w) == LBN_DBLCLK) { g_sel = (int)SendMessageA(g_list, LB_GETCURSEL, 0, 0); push_undo(); open_pad(g_sel); return 0; }
            if (((id >= ID_ROWEDT0 && id < ID_ROWEDT0 + NROWS && HIWORD(w) == EN_SETFOCUS) || (id >= ID_ROWCMB0 && id < ID_ROWCMB0 + NROWS && HIWORD(w) == CBN_SETFOCUS))) {
                size_t r = (size_t)(id >= ID_ROWCMB0 ? id - ID_ROWCMB0 : id - ID_ROWEDT0);
                if (r < g_rowkeys.size()) { g_param_key = g_rowkeys[r]; card_refresh(); }
                return 0;
            }
            if (id == ID_LIST && HIWORD(w) == LBN_SELCHANGE) { g_sel = (int)SendMessageA(g_list, LB_GETCURSEL, 0, 0); fill_props(); rebuild(); return 0; }
            if (id >= ID_OPT_TASK0 && id < ID_OPT_TASK0 + NTASK) { open_centre(id - ID_OPT_TASK0); return 0; }
            if (id >= ID_APP0 && id < ID_APP0 + (int)g_apps.size()) { app_open(id - ID_APP0); return 0; }
            if (id >= ID_CTX_APPEAR0 && id <= ID_CTX_APPEAR0 + NAPPEAR && g_sel >= 0) {
                push_undo(); js::Value& f = features().at((size_t)g_sel);
                if (id - ID_CTX_APPEAR0 < NAPPEAR) f["appearance"] = APPEAR[id - ID_CTX_APPEAR0].key; else f.erase("appearance");
                fill_props(); InvalidateRect(g_view, nullptr, FALSE); InvalidateRect(g_list, nullptr, FALSE); status(f.str("id", "") + ": " + part_appearance_name(g_sel)); return 0;
            }
            if (id >= ID_CTX_MAT0 && id <= ID_CTX_MAT0 + NMATERIAL && g_sel >= 0) {
                push_undo(); js::Value& f = features().at((size_t)g_sel);
                if (id - ID_CTX_MAT0 < NMATERIAL) f["material"] = MATERIALS[id - ID_CTX_MAT0].key; else f.erase("material");
                fill_props(); InvalidateRect(g_view, nullptr, FALSE); InvalidateRect(g_list, nullptr, FALSE); status(f.str("id", "") + ": " + part_appearance_name(g_sel)); return 0;
            }
            if (id >= ID_CTX_MODE0 && id < ID_CTX_MODE0 + 3) { g_appear_mode = id - ID_CTX_MODE0; InvalidateRect(g_view, nullptr, FALSE); InvalidateRect(g_list, nullptr, FALSE); return 0; }
            if (id >= ID_INS_BASE && id < ID_INS_BASE + 100) {
                std::vector<std::string> k = geo::feature_kinds();
                if ((size_t)(id - ID_INS_BASE) < k.size()) insert(geo::template_for(k[(size_t)(id - ID_INS_BASE)]));
                return 0;
            }
            switch (id) {
                case ID_FILE_NEW: if (!g_modified || MessageBoxA(h, "Discard unsaved changes?", "New design", MB_YESNO) == IDYES) set_doc(default_design(), ""); break;
                case ID_FILE_OPEN: {
                    std::string p;
                    if (ask_path(false, "Design (*.json)\0*.json\0All files\0*.*\0", "json", p)) {
                        try { set_doc(js::parse(js::read_file(p)), p); } catch (std::exception& e) { MessageBoxA(h, e.what(), "Open failed", MB_ICONERROR); }
                    }
                    break;
                }
                case ID_FILE_SAVE: save(false); break;
                case ID_FILE_SAVEAS: save(true); break;
                case ID_FILE_STL: export_mesh(false); break;
                case ID_FILE_OBJ: export_mesh(true); break;
                case ID_FILE_SVG: export_svg(); break;
                case ID_FILE_EXIT: SendMessageA(h, WM_CLOSE, 0, 0); break;
                case ID_EDIT_UNDO: undo(false); break;
                case ID_EDIT_REDO: undo(true); break;
                case ID_EDIT_DEL: case ID_BTN_DEL:
                    if (g_sel >= 0 && g_sel < (int)features().size()) { push_undo(); features().a->erase(features().a->begin() + g_sel); if (g_sel >= (int)features().size()) g_sel = (int)features().size() - 1; fill_list(); fill_props(); rebuild(); update_title(); }
                    break;
                case ID_EDIT_DUP: case ID_BTN_DUP:
                    if (g_sel >= 0 && g_sel < (int)features().size()) { js::Value c = features().at((size_t)g_sel).clone(); std::vector<double> t = c.nums("translate"); if (t.size() == 3) { t[2] += 0.5; js::Value a = js::Value::array(); for (double v : t) a.push(v); c["translate"] = a; } insert(c); }
                    break;
                case ID_EDIT_UP: case ID_BTN_UP: move_selected(-1); break;
                case ID_EDIT_DOWN: case ID_BTN_DOWN: move_selected(1); break;
                case ID_EDIT_TOGGLE: case ID_BTN_TOGGLE: modify_selected([](js::Value& o) { o["enabled"] = !o.flag("enabled", true); }); break;
                case ID_EDIT_OP: case ID_BTN_OP: modify_selected([](js::Value& o) { std::string op = o.str("op", "add"); o["op"] = op == "add" ? "cut" : op == "cut" ? "intersect" : "add"; }); break;
                case ID_INS_SLOT: { js::Value f = geo::template_for("box"); f["id"] = "slot"; f["op"] = "cut"; f["size"] = js::parse("[0.6, 0.08, 1.0]"); f["translate"] = js::parse("[8.5, 3.0, 0]"); f["mirror"] = true; insert(f); break; }
                case ID_INS_HOLE: { js::Value f = geo::template_for("cylinder"); f["id"] = "hole"; f["op"] = "cut"; f["axis"] = "z"; f["r"] = 0.15; f["length"] = 2.0; f["translate"] = js::parse("[9.0, 2.5, 0]"); f["mirror"] = true; insert(f); break; }
                case ID_INS_CAVITY: { js::Value f = geo::template_for("ellipsoid"); f["id"] = "cavity"; f["op"] = "cut"; f["radii"] = js::parse("[2.0, 0.4, 0.3]"); f["translate"] = js::parse("[8.0, 0, -0.1]"); insert(f); break; }
                case ID_MOD_SHELL: modify_selected([](js::Value& o) { o["shell"] = o.num("shell", 0) > 0 ? 0.0 : 0.04; }); break;
                case ID_MOD_PATTERN: modify_selected([](js::Value& o) { if (o.find("pattern")) o.erase("pattern"); else o["pattern"] = js::parse("{\"count\": 3, \"step\": [0.6, 0.4, 0], \"rotate\": 0}"); }); break;
                case ID_MOD_MIRROR: modify_selected([](js::Value& o) { o["mirror"] = !o.flag("mirror", false); }); break;
                case ID_MOD_BLEND: modify_selected([](js::Value& o) { o["blend"] = o.num("blend", 0) > 0 ? 0.0 : 0.1; }); break;
                case ID_MOD_TWIST: modify_selected([](js::Value& o) { o["twist_rate"] = o.num("twist_rate", 0) != 0 ? 0.0 : 5.0; }); break;
                case ID_MOD_TAPER: modify_selected([](js::Value& o) { o["taper_rate"] = o.num("taper_rate", 0) != 0 ? 0.0 : 0.05; }); break;
                case ID_MOD_BEND: modify_selected([](js::Value& o) { o["bend"] = o.num("bend", 0) != 0 ? 0.0 : 0.05; }); break;
                case ID_MOD_SHEAR: modify_selected([](js::Value& o) { if (o.find("shear")) o.erase("shear"); else o["shear"] = js::parse("[0.2, 0.0]"); }); break;
                case ID_MOD_STRETCH: modify_selected([](js::Value& o) { if (o.find("stretch")) o.erase("stretch"); else o["stretch"] = js::parse("[1.0, 1.2, 1.0]"); }); break;
                case ID_MOD_PIVOT: modify_selected([](js::Value& o) {
                    if (o.find("pivot")) { o.erase("pivot"); o["rotate"] = js::parse("[0, 0, 0]"); }
                    else { o["pivot"] = js::parse("[0, 0, 0]"); o["rotate"] = js::parse("[0, 0, 20]"); }
                }); break;
                case ID_MOD_INFILL: modify_selected([](js::Value& o) {
                    if (o.find("infill")) o.erase("infill");
                    else { if (o.num("shell", 0) <= 0) o["shell"] = 0.03; o["infill"] = js::parse("{\"type\": \"gyroid\", \"cell\": 0.25, \"wall\": 0.012}"); }
                }); break;
                case ID_MOD_CLEARX: modify_selected([](js::Value& o) {
                    const char* ks[] = {"taper_rate", "bend", "shear", "stretch", "pivot", "infill", "twist_rate"};
                    for (const char* k : ks) if (o.find(k)) o.erase(k);
                }); break;
                case ID_WFM_WINDOW: open_workflow(); break;
                case ID_WFM_NEW: open_workflow(); SendMessageA(g_wf_start, CB_SETCURSEL, 0, 0); wf_new(); break;
                case ID_WFM_OPEN: open_workflow(); wf_open(); break;
                case ID_WFM_NEXT: open_workflow(); wf_run(1); break;
                case ID_WFM_REPORT: open_workflow(); SendMessageA(g_wf, WM_COMMAND, ID_WF_REPORT, 0); break;
                case ID_VIEW_ISO: g_yaw = -135; g_pitch = 25; break;
                case ID_VIEW_FRONT: g_yaw = 180; g_pitch = 0; break;
                case ID_VIEW_SIDE: g_yaw = -90; g_pitch = 0; break;
                case ID_VIEW_TOP: g_yaw = -90; g_pitch = 89; break;
                case ID_VIEW_FIT: fit_view(); break;
                case ID_VIEW_WIRE: g_wire = !g_wire; break;
                case ID_VIEW_SHADOW: g_shadow_on = !g_shadow_on; break;
                case ID_VIEW_CUT: g_cut = !g_cut; if (g_cut && g_cutpos == 0) g_cutpos = 0.5 * (g_props.lo[g_cutaxis] + g_props.hi[g_cutaxis]); compute_section(); break;
                case ID_VIEW_CUTX: case ID_VIEW_CUTY: case ID_VIEW_CUTZ: g_cutaxis = id - ID_VIEW_CUTX; g_cut = true; g_cutpos = 0.5 * (g_props.lo[g_cutaxis] + g_props.hi[g_cutaxis]); compute_section(); break;
                case ID_VIEW_RES: { g_res = (g_res + 1) % 3; double hs[3] = {0.08, 0.045, 0.03}; g_h = hs[g_res]; rebuild(); status(std::string("mesh resolution ") + fmtd(g_h) + " m"); break; }
                case ID_VIEW_RESULTS_OFF: g_colors.clear(); g_result_label.clear(); break;
                case ID_TOOL_MEASURE: {
                    char b[1024];
                    snprintf(b, sizeof b, "Volume %.4f m3\nWetted area %.3f m2\nFrontal area %.3f m2\nPlanform area %.3f m2\nSide area %.3f m2\nLength %.3f m\n\nCentroid (%.3f, %.3f, %.3f) m\n"
                             "Inertia per unit density (m5):\nIxx %.4g  Iyy %.4g  Izz %.4g\nIxy %.4g  Ixz %.4g  Iyz %.4g\n\nBounds x %.2f..%.2f  y %.2f..%.2f  z %.2f..%.2f\nMesh %d triangles at %.3f m",
                             g_props.volume, g_props.area, g_props.frontal, g_props.planform, g_props.side, g_props.length, g_props.centroid.x, g_props.centroid.y, g_props.centroid.z,
                             g_props.I[0], g_props.I[1], g_props.I[2], g_props.I[3], g_props.I[4], g_props.I[5], g_props.lo.x, g_props.hi.x, g_props.lo.y, g_props.hi.y, g_props.lo.z, g_props.hi.z, (int)(g_mesh.t.size() / 3), g_h);
                    MessageBoxA(h, b, "Mass properties", MB_ICONINFORMATION);
                    break;
                }
                case ID_TOOL_CHECK: {
                    geo::Model mm; mm.load(g_doc);
                    std::string s = mm.errors.empty() ? "No errors. " + tostr((int)mm.feats.size()) + " active features." : "";
                    for (auto& e : mm.errors) s += e + "\n";
                    MessageBoxA(h, s.c_str(), "Check design", mm.errors.empty() ? MB_ICONINFORMATION : MB_ICONWARNING);
                    break;
                }
                case ID_AN_QUICK: quick_analysis(); break;
                case ID_AN_CFD_CRUISE: open_setup("cfd", "cruise"); break;
                case ID_AN_CFD_TRANS: open_setup("cfd", "transonic"); break;
                case ID_AN_CFD_SUPER: open_setup("cfd", "supersonic"); break;
                case ID_AN_CFD_ALPHA: open_setup("cfd", "high_alpha"); break;
                case ID_SIM_SIDESLIP: open_setup("cfd", "sideslip"); break;
                case ID_SIM_GUST: open_setup("cfd", "gust"); break;
                case ID_SIM_SETUP: open_setup("cfd"); break;
                case ID_SIM_TURB: open_setup("turbulence"); break;
                case ID_SIM_OPT: open_setup("optimise"); break;
                case ID_PREFS: open_prefs(); break;
                case ID_SIM_QUALITY: {
                    static const char* q[4] = {"draft", "standard", "fine", "live"};
                    int k = 0; for (int i = 0; i < 4; ++i) if (g_cfd_quality == q[i]) k = i;
                    g_cfd_quality = q[(k + 1) % 4];
                    status("3D flow quality: " + g_cfd_quality + (g_cfd_quality == "draft" ? " (length/30, 900 steps: minutes)" : g_cfd_quality == "standard" ? " (length/48, 2500 steps: about an hour)" :
                           g_cfd_quality == "fine" ? " (length/70, 3500 steps: hours)" : " (length/18, 260 steps: a minute)"));
                    break;
                }
                case ID_SIM_LIVE: g_live = !g_live; status(g_live ? "live simulation ON: every edit reruns a quick 3D flow and repaints the model" : "live simulation off"); if (g_live && !g_job.hProcess) SetTimer(g_main, 8, 300, nullptr); break;
                case ID_SIM_SLICE:
                    if (!g_vol_ok) { toggle_field(false); break; }
                    g_slice_axis = (g_slice_axis + 1) % 3; g_slice_on[g_slice_axis] = true; g_field_on = true; slice_move(0); break;
                case ID_SIM_LINES: g_show_lines = !g_show_lines; break;
                case ID_SIM_FLOWLINES: flowlines_toggle(); break;
                case ID_ENV_SETUP: env_setup(); break;
                case ID_ENV_CLEAR: env_clear(); break;
                case ID_SIM_SWEEP: sweep_toggle(); break;
                case ID_SIM_RECORD: record_toggle(); break;
                case ID_SIM_MESHES: g_show_meshes = !g_show_meshes; break;
                case ID_START_PAGE: open_start(); break;
                case ID_NEW_BLANK: set_doc(blank_doc(), ""); status("blank design: pick the Sketcher workbench and a plane to draw your first body"); break;
                case ID_NEW_FOLDER: new_in_folder(); break;
                case ID_SKP_XY: open_sketch(-1, 0, OP_PAD); break;
                case ID_SKP_XZ: open_sketch(-1, 1, OP_PAD); break;
                case ID_SKP_YZ: open_sketch(-1, 2, OP_PAD); break;
                case ID_SKS_POLY: open_sketch(-1, -1, -1, SH_POLY); break;
                case ID_SKS_RECT: open_sketch(-1, -1, -1, SH_RECT); break;
                case ID_SKS_CIRCLE: open_sketch(-1, -1, -1, SH_CIRCLE); break;
                case ID_SKS_SLOT: open_sketch(-1, -1, -1, SH_SLOT); break;
                case ID_PD_PAD: open_sketch(-1, -1, OP_PAD); break;
                case ID_PD_POCKET: open_sketch(-1, -1, OP_POCKET); break;
                case ID_PD_SHAFT: open_sketch(-1, 1, OP_SHAFT, SH_POLY); break;
                case ID_PD_GROOVE: open_sketch(-1, 1, OP_GROOVE, SH_POLY); break;
                case ID_PD_SLOT: open_sketch(-1, -1, OP_POCKET, SH_SLOT); break;
                case ID_PD_HOLE: insert(hole_feature(default_hole_spec(), "hole")); status("hole: type its size and direction in the pad, or pick its entry point on the model"); break;
                case ID_MOD_CHAMFER: modify_selected([](js::Value& o) { o["chamfer"] = o.num("chamfer", 0) > 0 ? 0.0 : 0.05; }); break;
                case ID_MOD_DRAFT: modify_selected([](js::Value& o) { o["draft"] = o.num("draft", 0) != 0 ? 0.0 : 5.0; }); break;
                case ID_VIEW_ZEBRA: g_view_mode = g_view_mode == 1 ? 0 : 1; if (!g_view_mode) { g_colors.clear(); g_result_label.clear(); } else analysis_colours(); break;
                case ID_VIEW_CURV: g_view_mode = g_view_mode == 2 ? 0 : 2; if (!g_view_mode) { g_colors.clear(); g_result_label.clear(); } else analysis_colours(); break;
                case ID_VIEW_ANNOT: g_annot = !g_annot; break;
                case ID_VIEW_TREE: toggle_panel(1); break;
                case ID_VIEW_PROPS: toggle_panel(2); break;
                case ID_VIEW_PALETTE: toggle_panel(0); break;
                case ID_VIEW_FULL: toggle_panel(3); break;
                case ID_DRAFT: open_draft(nullptr); break;
                case ID_HQ_IMAGE: hq_image_tool(); break;
                case ID_LABS: open_labs("thermal_lab", ""); break;
                case ID_LABS_APP: open_labs("thermal_app", ""); break;
                case ID_LABS_DESIGNER: open_labs("designer", ""); break;
                case ID_LABS_WB: open_labs("workbench", ""); break;
                case ID_LABS_AIRFRANS: open_labs("airfrans", ""); break;
                case ID_DRAFT_VIEWS: open_draft("VIEWS"); break;
                case ID_DRAFT_FLAT: open_draft("FLAT"); break;
                case ID_DRAFT_OPEN: open_draft("DXFIN"); break;
                case ID_SHM_SKETCH: open_sketch(SK_EXTRUDE, 1, OP_PAD, SH_SHEET); break;
                case ID_SHM_CHANNEL: case ID_SHM_ZBRACKET: case ID_SHM_HAT: case ID_SHM_LBRACKET: {  // sheet-metal presets: real bent plates
                    js::Value f = geo::template_for("bentplate");
                    const char* prof = id == ID_SHM_CHANNEL ? "[[0, 0.1], [0, 0], [0.2, 0], [0.2, 0.1]]" : id == ID_SHM_ZBRACKET ? "[[0, 0], [0.1, 0], [0.1, 0.15], [0.2, 0.15]]"
                                     : id == ID_SHM_HAT ? "[[0, 0], [0.06, 0], [0.06, 0.08], [0.18, 0.08], [0.18, 0], [0.24, 0]]" : "[[0, 0.12], [0, 0], [0.15, 0]]";
                    const char* nm = id == ID_SHM_CHANNEL ? "channel" : id == ID_SHM_ZBRACKET ? "z_bracket" : id == ID_SHM_HAT ? "hat_stiffener" : "l_bracket";
                    f["id"] = nm; f["op"] = "add"; f["mirror"] = false; f["profile"] = js::parse(prof); f["thickness"] = 0.003; f["bend_radius"] = 0.006; f["width"] = 0.4;
                    f["translate"] = js::parse("[0, 0, 0]");
                    insert(f);
                    status(std::string("sheet-metal ") + nm + ": 3 mm thick, R6 bends, 400 mm wide; flat blank " + fmtd(std::round(flat_length(f) * 10000) / 10) + " mm (Flat pattern draws it with its bend lines)");
                    break;
                }
                case ID_FIN_PINN: {
                    std::string have = g_root + "\\results\\studio\\fin_pinn";
                    if (file_exists(have + "_full\\summary.txt")) have += "_full"; if (file_exists(have.substr(0, have.find("fin_pinn") + 8) + "_v2\\summary.txt")) have = have.substr(0, have.find("fin_pinn") + 8) + "_v2";  // the newest: per-setting figures  // the longer training, when it is there
                    int a = file_exists(have + "\\summary.txt") ? MessageBoxA(h, "Fin PINN study: the trained results are on disk.\n\nYes: open them now\nNo: extract the data and train again (about 5 minutes)", "Fin PINN", MB_YESNOCANCEL | MB_ICONQUESTION) : IDNO;
                    if (a == IDYES) { open_simview(have, "Fin PINN: every optimisation setting, 2D and 3D"); break; }
                    if (a != IDNO) break;
                    std::string d = work_dir("fin_pinn");
                    launch("\"" + python() + "\" -m aerosynapse.thermal.fin_pinn --out \"" + d + "\" --profile quick", "simulation fin_pinn", d);
                    break;
                }
                case ID_ENG_UNFOLD: {
                    if (g_sel < 0 || features().at((size_t)g_sel).str("type", "") != "bentplate")
                        for (size_t q = 0; q < features().size(); ++q) if (features().at(q).str("type", "") == "bentplate") { g_sel = (int)q; fill_list(); fill_props(); rebuild(); break; }
                    if (g_sel < 0 || features().at((size_t)g_sel).str("type", "") != "bentplate") { MessageBoxA(h, "There is no bent plate in the design. Add one with Sheet & Composite > Bent plate, then Unfold.", "Unfold", MB_ICONINFORMATION); break; }
                    const js::Value& f = features().at((size_t)g_sel);
                    char b[256]; snprintf(b, sizeof b, "Flat blank of '%s': %.1f mm long x %.1f mm wide x %.1f mm thick\n(bend allowance with K = 0.44, inner bend radius %.1f mm)",
                                          f.str("id", "").c_str(), 1e3 * flat_length(f), 1e3 * f.num("width", 0.3), 1e3 * f.num("thickness", 0.003), 1e3 * f.num("bend_radius", 0.01));
                    MessageBoxA(h, b, "Unfold", MB_ICONINFORMATION);
                    break;
                }
                case ID_ENG_COMPOSITES: open_setup("composites"); break;
                case ID_ENG_CLASH: open_setup("clash"); break;
                case ID_ENG_RULES: run_engineer("rules"); break;
                case ID_ENG_RFLP: run_engineer("rflp"); break;
                case ID_ENG_PARAMS: open_params(); break;
                case ID_ASM_RESERVE: { js::Value f = geo::template_for("box"); f["id"] = "reservation"; f["reserve"] = true; f["enabled"] = false; f["size"] = js::parse("[1.6, 0.9, 0.8]"); f["round"] = 0.0; f["translate"] = js::parse("[8, 0, 0]"); insert(f); break; }
                case ID_DMU_KIN: kin_toggle(); break;
                case ID_ASM_PIPE: { js::Value f = geo::template_for("duct"); f["id"] = "pipe"; f["op"] = "add"; f["mirror"] = false; f["smooth"] = true; f["radius"] = js::parse("[0.04, 0.04, 0.04]"); insert(f); break; }
                case ID_ASM_HARNESS: { js::Value f = geo::template_for("duct"); f["id"] = "harness"; f["op"] = "add"; f["mirror"] = false; f["smooth"] = true; f["radius"] = js::parse("[0.02, 0.02, 0.02]"); f["path"] = js::parse("[[2.0, 0.3, 0.4], [6.0, 0.35, 0.5], [10.0, 0.3, 0.45], [13.0, 0.2, 0.3]]"); insert(f); break; }
                case ID_SIM_CLEAR: viz_clear(); g_colors.clear(); g_result_label.clear(); g_field_on = g_particles_on = false; KillTimer(g_main, 7); status("results cleared from the 3D view"); break;
                case ID_INS_AIRFOIL: insert_airfoil_wing(); break;
                case ID_RES_CP: show_surface_column(6); break;
                case ID_RES_CPRMS: show_surface_column(7); break;
                case ID_RES_CF: show_surface_column(8); break;
                case ID_RES_T: show_surface_column(9); break;
                case ID_OPT_CENTRE: open_centre(-1); break;
                case ID_VB_ZIN: g_dist *= 0.8; InvalidateRect(g_view, nullptr, FALSE); break;
                case ID_VB_ZOUT: g_dist *= 1.25; InvalidateRect(g_view, nullptr, FALSE); break;
                case ID_VB_DIMS: g_dims_on = !g_dims_on; status(g_dims_on ? "dimensions of the selected part: on (click a part, then a value)" : "dimensions: off"); InvalidateRect(g_view, nullptr, FALSE); InvalidateRect(g_viewbar, nullptr, FALSE); break;
                case ID_VB_APPEAR: g_appear_mode = (g_appear_mode + 1) % 3; status(std::string("part colours: ") + APPEAR_MODE_NAME[g_appear_mode]); InvalidateRect(g_view, nullptr, FALSE); InvalidateRect(g_list, nullptr, FALSE); break;
                case ID_OPT_APPLY: apply_best_wing(); break;
                case ID_CTX_EDIT: if (g_sel >= 0) { push_undo(); open_pad(g_sel); } break;
                case ID_CTX_DIMS: g_dims_on = true; g_dims_for = -2; InvalidateRect(g_view, nullptr, FALSE); break;
                case ID_CTX_HIDE: SendMessageA(h, WM_COMMAND, ID_EDIT_TOGGLE, 0); break;
                case ID_CTX_DEL: SendMessageA(h, WM_COMMAND, ID_EDIT_DEL, 0); break;
                case ID_CTX_DUP: SendMessageA(h, WM_COMMAND, ID_EDIT_DUP, 0); break;
                case ID_SK_REVOLVE: open_sketch(SK_REVOLVE); break;
                case ID_SK_EXTRUDE: open_sketch(SK_EXTRUDE); break;
                case ID_SK_LOFT: open_sketch(SK_LOFT); break;
                case ID_SIM_FIELD: toggle_field(false); break;
                case ID_SIM_PARTICLES: toggle_field(true); break;
                case ID_SIM_ANALYTIC: open_setup("analytic"); break;
                case ID_SIM_PINN: open_setup("pinn"); break;
                case ID_SIM_STRUCT: open_setup("structure"); break;
                case ID_SIM_VIEWER: {
                    std::string p = g_root + "\\results\\studio\\summary.txt";
                    if (g_sv && !g_sv_dir.empty()) { ShowWindow(g_sv, SW_SHOW); SetForegroundWindow(g_sv); break; }
                    { std::string d = newest_results("summary.txt"); if (!d.empty()) { open_simview(d, "Simulation results"); break; } }
                    if (ask_path(false, "Simulation summary (summary.txt)\0summary.txt\0", "txt", p)) { size_t at = p.find_last_of("\\/"); open_simview(p.substr(0, at), "Simulation results"); }
                    break;
                }
                case ID_EXP_PACKAGE: export_package(); break;
                case ID_EXP_PDF: export_report("pdf"); break;
                case ID_EXP_DOCX: export_report("docx"); break;
                case ID_EXP_TXT: export_report("txt"); break;
                default:
                    if (id >= ID_WFM_STAGE0 && id < ID_WFM_STAGE0 + NSTAGE) { open_workflow(); wf_select(id - ID_WFM_STAGE0); wf_run(0); }
                    break;
                case ID_AN_LOAD: { std::string p; if (ask_path(false, "Surface results (surface.csv)\0*.csv\0", "csv", p)) load_surface_csv(p); break; }
                case ID_APPLY: apply_rows(); break;
                case ID_APPLYJSON: apply_json(); break;
                case ID_HELP_ABOUT: about(); break;
                case ID_HELP_KEYS:
                    MessageBoxA(h, "Mouse: left drag orbit, right drag pan, wheel zoom.\n\nSelected feature: arrows move in x / y, PgUp PgDn in z (Shift: 1 m steps),\n+ / - scale, R rotate about z, T about x, Del delete, Ctrl+D duplicate.\n\n"
                                   "Ctrl+Z undo, Ctrl+Y redo, W wireframe, C section cut, X Y Z cut axis, [ ] move the cut,\n0 isometric, 1-6 standard views, F fit, H mesh resolution.\n\nEdit any property on the right and press Apply; the JSON box below edits the whole feature.",
                                "Mouse and keys", MB_ICONINFORMATION);
                    break;
            }
            g_cmd_id = 0;
            InvalidateRect(g_view, nullptr, FALSE);
            return 0;
        }
        case WM_CLOSE:
            if (g_modified && MessageBoxA(h, "Quit without saving?", "AeroSynapse Studio", MB_YESNO) != IDYES) return 0;
            DestroyWindow(h);
            return 0;
        case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

// ------------------------------------------------------------------ splash screen
static int g_splash_t = 0;
static LRESULT CALLBACK splash_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_PAINT) {
        PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps);
        RECT rc; GetClientRect(h, &rc);
        int W = rc.right, H = rc.bottom;
        gp::Bitmap buf(W, H, PixelFormat32bppARGB);
        gp::Graphics g(&buf);
        quality(g);
        gp::LinearGradientBrush bg(gp::Rect(0, 0, W, H), gcol(4, 10, 24), gcol(14, 70, 130), 35.0f);
        g.FillRectangle(&bg, 0, 0, W, H);
        for (int k = 0; k < NSCHEME; ++k) {  // the scheme colours as a band of light
            gp::SolidBrush sb(gcol(SCHEMES[k].r, SCHEMES[k].g, SCHEMES[k].b, 200));
            g.FillRectangle(&sb, (float)k * W / NSCHEME, (float)H - 6, (float)W / NSCHEME + 1, 6.0f);
        }
        gp::SolidBrush glow(gcol(0, 180, 255, 40)); g.FillEllipse(&glow, W / 2.0f - 110, 30.0f, 220.0f, 220.0f);
        if (gp::Bitmap* logo = load_png(g_root + "\\docs\\assets\\aerosynapse_logo.png")) g.DrawImage(logo, W / 2.0f - 70, 50.0f, 140.0f, 140.0f);
        gtext(g, "AeroSynapse Studio", gp::RectF(0, 205, (float)W, 40), gcol(255, 255, 255), 30.0f, gp::FontStyleBold, gp::StringAlignmentCenter);
        gtext(g, "aircraft design  |  CFD, analytic and PINN simulation  |  optimisation  |  CAD export", gp::RectF(0, 248, (float)W, 22), gcol(130, 215, 255), 13.0f,
              gp::FontStyleRegular, gp::StringAlignmentCenter);
        float t = std::min(1.0f, g_splash_t / 20.0f);
        gp::RectF bar(W * 0.2f, H - 48.0f, W * 0.6f, 6.0f);
        gp::SolidBrush track(gcol(255, 255, 255, 40)); g.FillRectangle(&track, bar);
        gp::LinearGradientBrush fill(bar, gcol(0, 188, 235), gcol(236, 72, 153), gp::LinearGradientModeHorizontal);
        g.FillRectangle(&fill, bar.X, bar.Y, bar.Width * t, bar.Height);
        gtext(g, t < 1 ? "loading the geometry kernel, solvers and tools..." : "ready", gp::RectF(0, H - 40.0f, (float)W, 20), gcol(180, 200, 230), 11.5f, gp::FontStyleRegular, gp::StringAlignmentCenter);
        gp::Graphics out(dc); out.DrawImage(&buf, 0, 0);
        EndPaint(h, &ps);
        return 0;
    }
    if (m == WM_ERASEBKGND) return 1;
    return DefWindowProcA(h, m, w, l);
}

static void splash(HINSTANCE hi) {
    WNDCLASSA a; memset(&a, 0, sizeof a);
    a.lpfnWndProc = splash_proc; a.hInstance = hi; a.hCursor = LoadCursor(nullptr, IDC_APPSTARTING); a.lpszClassName = "AeroSynapseSplash";
    RegisterClassA(&a);
    int W = 680, H = 380, sx = GetSystemMetrics(SM_CXSCREEN), sy = GetSystemMetrics(SM_CYSCREEN);
    HWND h = CreateWindowExA(WS_EX_TOOLWINDOW | WS_EX_TOPMOST, "AeroSynapseSplash", "", WS_POPUP, (sx - W) / 2, (sy - H) / 2, W, H, nullptr, nullptr, hi, nullptr);
    ShowWindow(h, SW_SHOW);
    for (g_splash_t = 0; g_splash_t <= 22; ++g_splash_t) {
        InvalidateRect(h, nullptr, FALSE); UpdateWindow(h);
        MSG msg; while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageA(&msg); }
        Sleep(55);
    }
    DestroyWindow(h);
}

int WINAPI WinMain(HINSTANCE hi, HINSTANCE, LPSTR cmd, int show) {
    INITCOMMONCONTROLSEX ic = {sizeof ic, ICC_BAR_CLASSES};
    InitCommonControlsEx(&ic);
    g_root = exe_dir() + "\\..";
    { char full[MAX_PATH]; if (GetFullPathNameA(g_root.c_str(), MAX_PATH, full, nullptr)) g_root = full; }
    g_font = CreateFontA(-13, 0, 0, 0, FW_NORMAL, 0, 0, 0, ANSI_CHARSET, 0, 0, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
    gdip_start();
    HICON icon = LoadIconA(hi, MAKEINTRESOURCEA(1));
    WNDCLASSA vc; memset(&vc, 0, sizeof vc);
    vc.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS; vc.lpfnWndProc = view_proc; vc.hInstance = hi; vc.hCursor = LoadCursor(nullptr, IDC_CROSS); vc.lpszClassName = "AeroSynapseView";
    RegisterClassA(&vc);
    WNDCLASSA mc; memset(&mc, 0, sizeof mc);
    g_br_panel = CreateSolidBrush(C_PANEL); g_br_field = CreateSolidBrush(C_FIELD);
    mc.lpfnWndProc = main_proc; mc.hInstance = hi; mc.hCursor = LoadCursor(nullptr, IDC_ARROW); mc.hbrBackground = g_br_panel; mc.lpszClassName = "AeroSynapseStudio"; mc.hIcon = icon;
    RegisterClassA(&mc);
    WNDCLASSA jc; memset(&jc, 0, sizeof jc);
    jc.lpfnWndProc = jobs_proc; jc.hInstance = hi; jc.hCursor = LoadCursor(nullptr, IDC_ARROW); jc.hbrBackground = g_br_panel; jc.lpszClassName = "AeroSynapseJobs"; jc.hIcon = icon;
    RegisterClassA(&jc);
    WNDCLASSA wc2; memset(&wc2, 0, sizeof wc2);
    wc2.lpfnWndProc = wf_proc; wc2.hInstance = hi; wc2.hCursor = LoadCursor(nullptr, IDC_ARROW); wc2.hbrBackground = g_br_panel; wc2.lpszClassName = "AeroSynapseWorkflow"; wc2.hIcon = icon;
    RegisterClassA(&wc2);
    WNDCLASSA ac; memset(&ac, 0, sizeof ac);
    ac.lpfnWndProc = ask_proc; ac.hInstance = hi; ac.hCursor = LoadCursor(nullptr, IDC_ARROW); ac.hbrBackground = g_br_panel; ac.lpszClassName = "AeroSynapseAsk"; ac.hIcon = icon;
    RegisterClassA(&ac);
    register_ui_classes(hi);
    register_sketch_classes(hi, icon);
    register_sim_classes(hi, icon);
    register_draft_classes(hi, icon);
    register_lab_classes(hi, icon);
    register_workspace_classes(hi, icon);
    register_setup_classes(hi, icon);
    register_pad_classes(hi, icon);
    prefs_load();
    g_pal_w = g_pal_target = g_autohide ? PAL_HANDLE : PALETTE_W;
    splash(hi);
    apps_load();
    WNDCLASSA vb; memset(&vb, 0, sizeof vb);
    vb.lpfnWndProc = viewbar_proc; vb.hInstance = hi; vb.hCursor = LoadCursor(nullptr, IDC_HAND); vb.lpszClassName = "AeroSynapseViewbar";
    RegisterClassA(&vb);
    HWND h = CreateWindowExA(0, "AeroSynapseStudio", "AeroSynapse Studio", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1500, 900, nullptr, nullptr, hi, nullptr);
    ribbon_set_menu(make_menu());  // the classic menus live under the ribbon's File tab
    // AeroSynapseStudio [design.json] [--results surface.csv] [--view iso|front|side|top]
    std::string arg;
    bool clean = false;
    std::vector<std::string> av;
    int asm_demo_arg = 0; bool flow_run_arg = false;
    bool env_run_arg = false, post_run = false;
    {
        int n = 0;
        LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n);
        for (int i = 0; w && i < n; ++i) {
            int len = WideCharToMultiByte(CP_ACP, 0, w[i], -1, nullptr, 0, nullptr, nullptr);
            std::string s8((size_t)std::max(len - 1, 0), ' ');
            if (len > 1) WideCharToMultiByte(CP_ACP, 0, w[i], -1, &s8[0], len, nullptr, nullptr);
            av.push_back(s8);
        }
        if (w) LocalFree(w);
    }
    for (size_t i = 1; i < av.size(); ++i) {
        const std::string& a = av[i];
        if (a == "--results" && i + 1 < av.size()) g_pending_results = av[++i];
        else if (a == "--view" && i + 1 < av.size()) { const std::string& v = av[++i]; g_pending_view = v == "front" ? 1 : v == "side" ? 2 : v == "top" ? 3 : 0; }
        else if (a == "--h" && i + 1 < av.size()) g_h = std::max(0.01, atof(av[++i].c_str()));
        else if (a == "--clean") clean = true;
        else if (a == "--project" && i + 1 < av.size()) g_pending_project = av[++i];
        else if (a == "--scheme" && i + 1 < av.size()) { std::string k = av[++i]; for (int q = 0; q < NSCHEME; ++q) if (k == SCHEMES[q].key) g_scheme = q; }
        else if (a == "--simview" && i + 1 < av.size()) g_pending_simview = av[++i];
        else if (a == "--hq-image" && i + 1 < av.size()) g_pending_hq = av[++i];
        else if (a == "--field" && i + 1 < av.size()) g_pending_field = av[++i];
        else if (a == "--sketch-demo") g_pending_sketch = true;
        else if (a == "--asm-demo") asm_demo_arg = 1;  // the worked assembly example (screenshots, tests)
        else if (a == "--asm-play") asm_demo_arg = 2;
        else if (a == "--asm-explode") asm_demo_arg = 3;
        else if (a == "--asm-gears") asm_demo_arg = 4;
        else if (a == "--env-run") env_run_arg = true;  // analyse the saved environment once meshed
        else if (a == "--env-medium" && i + 1 < av.size()) { std::string m = av[++i]; for (int q = 0; q < NMED; ++q) if (m == MED_NAME[q]) g_env.medium = q; }
        else if (a == "--asm-gears-play") asm_demo_arg = 5;
        else if (a == "--asm-gears2") asm_demo_arg = 6;
        else if (a == "--asm-gears2-play") asm_demo_arg = 7;
        else if (a == "--asm-drive") asm_demo_arg = 8;
        else if (a == "--power-run") { g_power_pending = true; post_run = true; }
        else if (a == "--check-run") { g_check_pending = true; post_run = true; }
        else if (a == "--asm-drive-play") asm_demo_arg = 9;
        else if (a == "--asm-belts") asm_demo_arg = 10;
        else if (a == "--asm-belts-play") asm_demo_arg = 11;
        else if (a == "--asm-worm") asm_demo_arg = 12;
        else if (a == "--asm-worm-play") asm_demo_arg = 13;
        else if (a == "--asm-cam") asm_demo_arg = 14;
        else if (a == "--asm-cam-play") asm_demo_arg = 15;
        else if (a == "--flow-cyl") asm_demo_arg = 16;
        else if (a == "--flow-tank") asm_demo_arg = 17;
        else if (a == "--flow-gears") asm_demo_arg = 18;
        else if (a == "--flow-belt") asm_demo_arg = 19;
        else if (a == "--flow-splash") asm_demo_arg = 20;
        else if (a == "--flow-run") flow_run_arg = true;
        else if (a == "--flow-auto") { flow_run_arg = true; g_flow_auto_arg = true; }
        else if (a == "--flow-time" && i + 1 < av.size()) g_flow_time_arg = atof(av[++i].c_str());
        else if (a == "--flow-cells" && i + 1 < av.size()) g_flow_cells_arg = atoi(av[++i].c_str());
        else if (a == "--flowlines") g_pending_flowlines = true;  // with --field: 3D streamlines straight away
        else if (a == "--selftest" && i + 1 < av.size()) { g_selftest = true; g_st_out = av[++i]; }
        else if (a == "--tip" && i + 1 < av.size()) g_pending_tip = av[++i];  // show the hover card of a palette tool (screenshots, tests)
        else if (a == "--new-project" && i + 2 < av.size()) { g_pending_new_parent = av[++i]; g_pending_new_name = av[++i]; }
        else arg = a;
    }
    (void)cmd;
    try { if (!arg.empty()) set_doc(js::parse(js::read_file(arg)), arg); else set_doc(default_design(), ""); if (clean) { g_sel = -1; fill_list(); fill_props(); rebuild(); } }
    catch (std::exception& e) { MessageBoxA(h, e.what(), "Open failed", MB_ICONERROR); set_doc(default_design(), ""); }
    ShowWindow(h, show); UpdateWindow(h);
    layout();
    if (arg.empty() && g_pending_project.empty() && g_pending_simview.empty() && g_pending_field.empty() && !g_pending_sketch && !clean && g_pending_new_name.empty() && !asm_demo_arg) open_start();
    if (!g_pending_new_name.empty()) make_project_folder(g_pending_new_parent, g_pending_new_name);
    if (g_selftest) SetTimer(g_main, 13, 50, nullptr);
    if (!g_pending_tip.empty()) {
        std::vector<int> ts = scheme_tools(g_scheme);
        for (size_t q = 0; q < ts.size(); ++q)
            if (g_pending_tip == TOOLS[ts[q]].icon || g_pending_tip == TOOLS[ts[q]].label) {
                g_hover_tool = (int)q; InvalidateRect(g_palette, nullptr, FALSE);
                RECT t = tool_rect((int)q); POINT pt = {PALETTE_W + 6, t.top}; ClientToScreen(g_palette, &pt);
                show_tip(ts[q], pt.x, pt.y);
                status(std::string(TOOLS[ts[q]].label) + ":  " + TOOLS[ts[q]].tip);
            }
    }
    if (!g_pending_simview.empty()) open_simview(g_pending_simview, "Simulation results");
    if (!g_pending_hq.empty()) SetTimer(g_main, 46, 9000, nullptr);  // --hq-image: a 4K picture once the design is shown
    if (env_run_arg) { g_env_pending = true; SetTimer(g_main, 42, 500, nullptr); }
    if (post_run) SetTimer(g_main, 43, 500, nullptr);
    if (flow_run_arg) { g_flow_pending = true; SetTimer(g_main, 45, 500, nullptr); }
    if (asm_demo_arg) { asm_tool(asm_demo_arg == 20 ? 55 : asm_demo_arg == 19 ? 53 : asm_demo_arg == 18 ? 52 : asm_demo_arg == 17 ? 50 : asm_demo_arg == 16 ? 49 : asm_demo_arg >= 14 ? 46 : asm_demo_arg >= 12 ? 42 : asm_demo_arg >= 10 ? 39 : asm_demo_arg >= 8 ? 33 : asm_demo_arg >= 6 ? 24 : asm_demo_arg >= 4 ? 20 : 11); if (asm_demo_arg == 2 || asm_demo_arg == 5 || asm_demo_arg == 7 || asm_demo_arg == 9 || asm_demo_arg == 11 || asm_demo_arg == 13 || asm_demo_arg == 15) ms_play(); if (asm_demo_arg == 3) asm_toggle_explode(); }
    if (g_pending_sketch) {  // a nacelle-like profile, to show the sketcher
        open_sketch(SK_REVOLVE);
        double pr[][2] = {{4.0, 0.0}, {4.3, 0.42}, {5.2, 0.78}, {7.0, 0.95}, {9.5, 0.92}, {11.2, 0.7}, {12.0, 0.45}};
        for (auto& q : pr) g_sk_pts.push_back(geo::P2(q[0], q[1]));
        InvalidateRect(g_sk_canvas, nullptr, FALSE);
    }
    if (!g_pending_project.empty()) {
        std::string d = g_pending_project;
        if (!file_exists(d + "\\project.json")) d = g_root + "\\projects\\" + d;
        wf_set_project(d);
        for (int k = NSTAGE - 1; k >= 0; --k) if (g_wf_status[k] == "done") { wf_load_stage(k, true); wf_select(k); break; }
    }
    MSG msg;
    ACCEL acc[] = {{FCONTROL | FVIRTKEY, 'N', ID_FILE_NEW}, {FCONTROL | FVIRTKEY, 'O', ID_FILE_OPEN}, {FCONTROL | FVIRTKEY, 'S', ID_FILE_SAVE}, {FVIRTKEY, VK_F9, ID_WFM_WINDOW},
                   {FVIRTKEY, VK_F2, ID_VIEW_TREE}, {FVIRTKEY, VK_F3, ID_VIEW_PROPS}, {FVIRTKEY, VK_F4, ID_VIEW_PALETTE}, {FVIRTKEY, VK_F11, ID_VIEW_FULL}};
    HACCEL ha = CreateAcceleratorTableA(acc, 8);
    while (GetMessageA(&msg, nullptr, 0, 0)) {
        if (!TranslateAcceleratorA(h, ha, &msg)) { TranslateMessage(&msg); DispatchMessageA(&msg); }
    }
    return 0;
}
