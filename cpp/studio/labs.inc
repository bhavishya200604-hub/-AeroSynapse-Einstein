// Labs: every browser app of AeroSynapse inside Studio, in a native window. THERMAL LAB, the PINN heat-transfer app,
// the Designer (aircraft, fighters, workflow, materials, data store) and the research workbench run their own page code
// in a Python worker (aerosynapse.nativelab) that answers each page's inputs and records what it shows; here the inputs
// are native fields and the output a native report (headings, numbers, figures, tables, downloads). The inputs start
// from the open design and assembly: the selected part, its material, the Environment's fluid and speed, the aircraft's
// fuselage and wings (nativelab.context), each marked "design". Left: the apps and their pages; middle: the page's
// inputs, Run, the design link; right: the report (wheel scrolls; a click on a figure or table opens it on its own).

struct LabPage { std::string app, app_title, page; };
struct LabWidget { std::string key, kind, label, value, source, help, section; std::vector<std::string> options; double lo = NAN, hi = NAN; bool sidebar = false; HWND ctl = nullptr, lbl = nullptr, extra = nullptr; };
struct LabBlock { std::string type, text, label, value, file, section; double prog = 0; std::vector<std::string> cols; int rows = 0; float y0 = 0, y1 = 0; };
static HWND g_lab, g_lab_list, g_lab_form, g_lab_rep, g_lab_run, g_lab_design, g_lab_status;
static std::vector<LabPage> g_lab_pages;
static std::vector<int> g_lab_row;  // list row -> page index (-1 a heading)
static int g_lab_cur = -1, g_lab_req = 0, g_lab_wait = -1;
static std::vector<LabWidget> g_lab_w;
static std::vector<LabBlock> g_lab_b;
static std::map<std::string, std::map<std::string, std::string>> g_lab_vals;  // page -> widget key -> value
static std::string g_lab_dir, g_lab_resdir, g_lab_press;
static HANDLE g_lab_proc = nullptr, g_lab_job = nullptr;
static DWORD g_lab_t0 = 0;
static float g_lab_scroll = 0, g_lab_total = 0;
static int g_lab_form_scroll = 0, g_lab_form_h = 0;
static bool g_lab_use_design = true;
static std::string lab_rid(int id) {  // request names unique to this Studio session (a new session never reads an old answer)
    static std::string pre = tostr((long)GetCurrentProcessId()) + "_" + tostr((long)GetTickCount());
    return pre + "_" + tostr(id);
}
enum { ID_LAB_LIST = 9600, ID_LAB_RUN, ID_LAB_DESIGN, ID_LAB_RESET, ID_LAB_W0 = 9700 };

// ---------------------------------------------------------------- the design and the assembly, for the labs
static js::Value lab_mat(const std::string& key) {
    js::Value m = js::Value::object();
    if (const EnvMat* e = envmat_of(key)) { m["key"] = e->key; m["label"] = e->label; m["rho"] = e->rho; m["k"] = e->k; m["cp"] = e->cp; }
    return m;
}
static js::Value labs_context() {
    js::Value c = js::Value::object();
    c["design"] = g_doc.clone();
    env_from_doc();
    Fluid F = fluid_props(g_env.medium, g_env.Tc, g_env.alt);
    js::Value env = js::Value::object(), fl = js::Value::object();
    env["medium"] = MED_NAME[g_env.medium]; env["T_fluid_C"] = g_env.Tc; env["T_part_C"] = g_env.Tpart; env["U_m_s"] = g_env.U;
    fl["rho"] = F.rho; fl["mu"] = F.mu; fl["k"] = F.k; fl["cp"] = F.cp; fl["Pr"] = F.Pr; fl["beta"] = F.beta;
    env["fluid"] = fl; c["environment"] = env;
    js::Value part = js::Value::object(), comps = js::Value::array();
    if (g_asm_on && !g_ac.empty()) {
        int pick = g_asm_sel; double best = -1;
        for (size_t i = 0; i < g_ac.size(); ++i) {
            const AComp& C = g_ac[i]; geo::Box b = comp_box(C); if (!b.valid()) continue;
            V3 s = b.hi - b.lo;
            js::Value o = js::Value::object(); o["id"] = C.id; o["material"] = lab_mat(part_material_key(C.part, C.material));
            js::Value sz = js::Value::array(); sz.push(js::Value(s.x)); sz.push(js::Value(s.y)); sz.push(js::Value(s.z)); o["size_m"] = sz;
            double rpm = 0; for (const AMotor& m : g_amot) if (m.comp == (int)i && !m.linear) rpm = std::fabs(m.speed) / 6.0;
            o["rpm"] = rpm; comps.push(o);
            double v = s.x * s.y * s.z; if (g_asm_sel < 0 && v > best) { best = v; pick = (int)i; }
        }
        if (pick >= 0 && pick < (int)comps.size()) part = comps.at((size_t)pick).clone();
    } else if (!g_mesh.v.empty()) {
        V3 s = g_props.hi - g_props.lo;
        part["id"] = g_path.empty() ? std::string("design") : g_path.substr(g_path.find_last_of("\\/") + 1);
        part["material"] = lab_mat(part_material_key(g_doc, ""));
        js::Value sz = js::Value::array(); sz.push(js::Value(s.x)); sz.push(js::Value(s.y)); sz.push(js::Value(s.z)); part["size_m"] = sz;
        part["volume_m3"] = g_props.volume;
    }
    if (g_env.heat > 0) part["loss_W"] = g_env.heat;
    c["part"] = part; c["components"] = comps;
    return c;
}
static std::string labs_context_line() {
    js::Value c = labs_context();
    js::Value& p = c["part"];
    std::string s = "Design input: ";
    if (p.str("id", "").empty()) return s + "none (open a part or an assembly; the Environment sets the fluid)";
    s += p.str("id", "") + " (" + p["material"].str("label", "material not set") + ")";
    s += ", in " + c["environment"].str("medium", "") + " at " + fmtd(c["environment"].num("T_fluid_C", 15)) + " C";
    if (c["environment"].num("U_m_s", 0) > 0) s += ", current " + fmtd(c["environment"].num("U_m_s", 0)) + " m/s";
    if (p.num("rpm", 0) > 0) s += ", " + fmtd(p.num("rpm", 0)) + " rpm";
    return s;
}

// ---------------------------------------------------------------- the worker
static bool labs_worker() {
    if (g_lab_proc && WaitForSingleObject(g_lab_proc, 0) == WAIT_TIMEOUT) return true;
    g_lab_dir = work_dir("labs\\worker");
    std::string log = g_lab_dir + "\\worker.log";
    std::string cmd = "cmd /c \"\"" + python() + "\" -m aerosynapse.nativelab serve \"" + g_lab_dir + "\" > \"" + log + "\" 2>&1\"";
    SetEnvironmentVariableA("PYTHONPATH", (g_root + "\\src").c_str());
    STARTUPINFOA si; memset(&si, 0, sizeof si); si.cb = sizeof si; PROCESS_INFORMATION pi;
    if (!CreateProcessA(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW | CREATE_SUSPENDED | BELOW_NORMAL_PRIORITY_CLASS, nullptr, g_root.c_str(), &si, &pi)) return false;
    if (!g_lab_job) {  // the worker stops with Studio
        g_lab_job = CreateJobObjectA(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li; memset(&li, 0, sizeof li); li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(g_lab_job, JobObjectExtendedLimitInformation, &li, sizeof li);
    }
    AssignProcessToJobObject(g_lab_job, pi.hProcess);
    ResumeThread(pi.hThread); CloseHandle(pi.hThread);
    g_lab_proc = pi.hProcess;
    return true;
}

static void lab_collect() {  // the fields' values into the page's value map
    if (g_lab_cur < 0) return;
    std::map<std::string, std::string>& V = g_lab_vals[g_lab_pages[(size_t)g_lab_cur].app + "|" + g_lab_pages[(size_t)g_lab_cur].page];
    for (LabWidget& w : g_lab_w) {
        if (!w.ctl || w.kind == "button") continue;
        if (w.kind == "checkbox") V[w.key] = SendMessageA(w.ctl, BM_GETCHECK, 0, 0) == BST_CHECKED ? "true" : "false";
        else if (w.kind == "selectbox" || w.kind == "radio") { int i = (int)SendMessageA(w.ctl, CB_GETCURSEL, 0, 0); if (i >= 0 && i < (int)w.options.size()) V[w.key] = w.options[(size_t)i]; }
        else V[w.key] = read_edit(w.ctl);
    }
}
static void lab_request(bool fresh) {
    if (g_lab_cur < 0 || g_lab_cur >= (int)g_lab_pages.size()) return;
    if (!labs_worker()) { SetWindowTextA(g_lab_status, "the Python worker could not be started"); return; }
    if (!fresh) lab_collect();
    const LabPage& P = g_lab_pages[(size_t)g_lab_cur];
    js::Value r = js::Value::object(), vals = js::Value::object();
    r["app"] = P.app; r["page"] = P.page; r["use_design"] = g_lab_use_design;
    for (auto& kv : g_lab_vals[P.app + "|" + P.page]) {
        std::string kind; for (const LabWidget& w : g_lab_w) if (w.key == kv.first) kind = w.kind;
        char* end = nullptr; double d = strtod(kv.second.c_str(), &end);
        if ((kind == "number_input" || kind == "slider") && end && *end == 0 && !kv.second.empty()) vals[kv.first] = d;
        else if (kind == "checkbox") vals[kv.first] = kv.second == "true";
        else vals[kv.first] = kv.second;
    }
    r["values"] = vals;
    if (!g_lab_press.empty()) r["press"] = g_lab_press;
    r["context"] = labs_context();
    g_lab_press.clear();
    int id = ++g_lab_req;
    std::string tmp = g_lab_dir + "\\tmp_" + lab_rid(id) + ".json", fin = g_lab_dir + "\\req_" + lab_rid(id) + ".json";
    try { js::write_file(tmp, js::dump(r, 1)); MoveFileExA(tmp.c_str(), fin.c_str(), MOVEFILE_REPLACE_EXISTING); } catch (...) { return; }
    g_lab_wait = id; g_lab_t0 = GetTickCount();
    SetTimer(g_lab, 1, 200, nullptr);
    SetWindowTextA(g_lab_status, ("running " + P.app_title + ": " + P.page + " ...").c_str());
    InvalidateRect(g_lab_rep, nullptr, FALSE);
}

// ---------------------------------------------------------------- the form
static void lab_form_layout() {
    RECT rc; GetClientRect(g_lab_form, &rc);
    int W = rc.right, y = 8 - g_lab_form_scroll;
    std::string sec = "\x01";
    for (LabWidget& w : g_lab_w) {
        std::string s = w.sidebar ? "Settings" : w.section;
        (void)s;
        if (w.kind == "button") { MoveWindow(w.ctl, 10, y, W - 20, 28, TRUE); y += 34; continue; }
        MoveWindow(w.lbl, 10, y, W - 20, 18, TRUE); y += 19;
        int h = w.kind == "text_area" ? 70 : (w.kind == "selectbox" || w.kind == "radio") ? 300 : 24;
        if (w.kind == "file_uploader") { MoveWindow(w.ctl, 10, y, W - 60, 24, TRUE); MoveWindow(w.extra, W - 46, y, 36, 24, TRUE); }
        else MoveWindow(w.ctl, 10, y, W - 20, h, TRUE);
        y += (w.kind == "text_area" ? 74 : 28) + 6;
    }
    g_lab_form_h = y + g_lab_form_scroll;
}
static void lab_build_form(const js::Value& ui) {
    for (LabWidget& w : g_lab_w) { if (w.ctl) DestroyWindow(w.ctl); if (w.lbl) DestroyWindow(w.lbl); if (w.extra) DestroyWindow(w.extra); }
    g_lab_w.clear(); g_lab_form_scroll = 0;
    const js::Value* ws = ui.find("widgets"); if (!ws) return;
    HINSTANCE hi = GetModuleHandleA(nullptr);
    std::map<std::string, std::string>& V = g_lab_vals[g_lab_pages[(size_t)g_lab_cur].app + "|" + g_lab_pages[(size_t)g_lab_cur].page];
    // the settings (sidebar) first, then the page's own inputs, as the apps lay them out
    std::vector<const js::Value*> order;
    for (int pass = 0; pass < 2; ++pass) for (size_t i = 0; i < ws->size(); ++i) if (ws->at(i).flag("sidebar", false) == (pass == 0)) order.push_back(&ws->at(i));
    int id = ID_LAB_W0;
    for (const js::Value* pw : order) {
        const js::Value& x = *pw;
        LabWidget w; w.key = x.str("key", ""); w.kind = x.str("kind", ""); w.label = x.str("label", ""); w.source = x.str("source", ""); w.help = x.str("help", ""); w.section = x.str("section", ""); w.sidebar = x.flag("sidebar", false);
        if (const js::Value* o = x.find("options")) for (size_t k = 0; k < o->size(); ++k) w.options.push_back(o->at(k).is_str() ? o->at(k).s : js::dump(o->at(k), -1));
        const js::Value* v = x.find("value");
        if (v) w.value = v->is_str() ? v->s : v->kind == js::Value::Bool ? (v->b ? "true" : "false") : v->is_arr() ? [&] { std::string t; for (size_t k = 0; k < v->size(); ++k) t += (k ? "; " : "") + (v->at(k).is_str() ? v->at(k).s : fmtd(v->at(k).n)); return t; }() : v->is_num() ? fmtd(v->n) : "";
        if (const js::Value* mn = x.find("min")) if (mn->is_num()) w.lo = mn->n;
        if (const js::Value* mx = x.find("max")) if (mx->is_num()) w.hi = mx->n;
        if (w.kind == "button") {
            w.ctl = CreateWindowExA(0, "BUTTON", w.label.c_str(), WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, g_lab_form, (HMENU)(INT_PTR)id++, hi, nullptr);
        } else {
            std::string lab = w.label;
            if (w.section.size()) lab = "[" + w.section + "]  " + lab;
            if (!std::isnan(w.lo) && !std::isnan(w.hi)) lab += "   (" + fmtd(w.lo) + " .. " + fmtd(w.hi) + ")";
            if (w.source.compare(0, 6, "design") == 0) lab += "   << from the design";
            w.lbl = CreateWindowExA(0, "STATIC", lab.c_str(), WS_CHILD | WS_VISIBLE | SS_LEFTNOWORDWRAP | SS_NOTIFY, 0, 0, 0, 0, g_lab_form, nullptr, hi, nullptr);
            if (w.kind == "checkbox") {
                w.ctl = CreateWindowExA(0, "BUTTON", "yes", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 0, 0, 0, 0, g_lab_form, (HMENU)(INT_PTR)id++, hi, nullptr);
                SendMessageA(w.ctl, BM_SETCHECK, w.value == "true" ? BST_CHECKED : BST_UNCHECKED, 0);
            } else if (w.kind == "selectbox" || w.kind == "radio") {
                w.ctl = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL, 0, 0, 0, 0, g_lab_form, (HMENU)(INT_PTR)id++, hi, nullptr);
                int sel = 0;
                for (size_t k = 0; k < w.options.size(); ++k) { SendMessageA(w.ctl, CB_ADDSTRING, 0, (LPARAM)w.options[k].c_str()); if (w.options[k] == w.value) sel = (int)k; }
                SendMessageA(w.ctl, CB_SETCURSEL, sel, 0);
            } else {
                DWORD st = WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL | (w.kind == "text_area" ? ES_MULTILINE | WS_VSCROLL | ES_AUTOVSCROLL : 0);
                w.ctl = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", w.value.c_str(), st, 0, 0, 0, 0, g_lab_form, (HMENU)(INT_PTR)id++, hi, nullptr);
                if (w.kind == "file_uploader") w.extra = CreateWindowExA(0, "BUTTON", "...", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, g_lab_form, (HMENU)(INT_PTR)id++, hi, nullptr);
            }
            SendMessageA(w.lbl, WM_SETFONT, (WPARAM)g_font, TRUE);
        }
        if (w.ctl) SendMessageA(w.ctl, WM_SETFONT, (WPARAM)g_font, TRUE);
        if (w.kind != "button") V[w.key] = w.value;
        g_lab_w.push_back(w);
    }
    lab_form_layout();
}
static LRESULT CALLBACK lab_form_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_COMMAND: {
            int id = LOWORD(w);
            for (LabWidget& x : g_lab_w) {
                if (x.ctl && GetDlgCtrlID(x.ctl) == id && x.kind == "button") { g_lab_press = x.key; lab_request(false); return 0; }
                if (x.extra && GetDlgCtrlID(x.extra) == id) { std::string p; if (ask_path(false, "All files\0*.*\0", "", p)) SetWindowTextA(x.ctl, p.c_str()); return 0; }
                if (x.ctl && GetDlgCtrlID(x.ctl) == id && (x.kind == "selectbox" || x.kind == "radio") && HIWORD(w) == CBN_SELCHANGE) { lab_request(false); return 0; }  // a choice re-runs at once, as in the apps
                if (x.ctl && GetDlgCtrlID(x.ctl) == id && x.kind == "checkbox" && HIWORD(w) == BN_CLICKED) { lab_request(false); return 0; }
            }
            return 0;
        }
        case WM_MOUSEWHEEL: {
            RECT rc; GetClientRect(h, &rc);
            g_lab_form_scroll = std::max(0, std::min(std::max(0, g_lab_form_h - (int)rc.bottom + 20), g_lab_form_scroll - GET_WHEEL_DELTA_WPARAM(w) / 2));
            lab_form_layout(); InvalidateRect(h, nullptr, TRUE); return 0;
        }
        case WM_DRAWITEM: { DRAWITEMSTRUCT* d = (DRAWITEMSTRUCT*)l; draw_button(d, gcol(234, 88, 12)); return TRUE; }
        case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkMode(dc, TRANSPARENT); SetTextColor(dc, C_INK); for (LabWidget& x : g_lab_w) if ((HWND)l == x.lbl && x.source.compare(0, 6, "design") == 0) SetTextColor(dc, RGB(22, 120, 70)); return (LRESULT)g_br_panel; }
        case WM_CTLCOLOREDIT: case WM_CTLCOLORLISTBOX: { HDC dc = (HDC)w; SetBkColor(dc, C_FIELD); SetTextColor(dc, C_INK); return (LRESULT)g_br_field; }
        case WM_KEYDOWN: if (w == VK_RETURN) lab_request(false); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

// ---------------------------------------------------------------- the report
static void lab_load_result(const std::string& dir) {
    g_lab_resdir = dir; g_lab_b.clear(); g_lab_scroll = 0;
    try {
        js::Value ui = js::parse(js::read_file(dir + "\\ui.json"));
        lab_build_form(ui);
        js::Value bl = js::parse(js::read_file(dir + "\\blocks.json"));
        if (const js::Value* a = bl.find("blocks"))
            for (size_t i = 0; i < a->size(); ++i) {
                const js::Value& x = a->at(i); LabBlock b;
                b.type = x.str("type", ""); b.text = x.str("text", ""); b.label = x.str("label", ""); b.value = x.str("value", ""); b.file = x.str("file", ""); b.section = x.str("section", "");
                b.prog = x.num("value", 0); b.rows = (int)x.num("rows", 0);
                if (const js::Value* c = x.find("cols")) for (size_t k = 0; k < c->size(); ++k) b.cols.push_back(c->at(k).is_str() ? c->at(k).s : std::string());
                if (b.type == "progress" && !g_lab_b.empty() && g_lab_b.back().type == "progress") { g_lab_b.back() = b; continue; }
                g_lab_b.push_back(b);
            }
        for (const LabBlock& b : g_lab_b) if (b.type == "figure") forget_png(dir + "\\" + b.file);
    } catch (std::exception& e) { LabBlock b; b.type = "error"; b.text = std::string("the result could not be read: ") + e.what(); g_lab_b.push_back(b); }
    InvalidateRect(g_lab_rep, nullptr, FALSE);
}
static float lab_text(gp::Graphics& g, const std::string& s, float x, float y, float w, gp::Color c, float px, int style = gp::FontStyleRegular, bool mono = false) {
    gp::FontFamily ff(mono ? L"Consolas" : L"Segoe UI"); gp::Font f(&ff, px, style, gp::UnitPixel);
    std::wstring ws = wide(s); gp::RectF box; gp::StringFormat sf;
    g.MeasureString(ws.c_str(), (INT)ws.size(), &f, gp::RectF(x, y, w, 100000), &sf, &box);
    gp::SolidBrush b(c); g.DrawString(ws.c_str(), (INT)ws.size(), &f, gp::RectF(x, y, w, box.Height + 4), &sf, &b);
    return box.Height;
}
static void paint_lab_report(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    int W = std::max(1L, rc.right), H = std::max(1L, rc.bottom);
    gp::Bitmap buf(W, H, PixelFormat32bppARGB); gp::Graphics g(&buf); quality(g);
    gp::SolidBrush bg(gcol(246, 248, 251)); g.FillRectangle(&bg, 0, 0, W, H);
    float x = 22, w = (float)W - 44, y = 16 - g_lab_scroll;
    if (g_lab_wait >= 0) {  // running: the time, and the page's progress bar when it has one
        std::string s = "Running... " + tostr((long)((GetTickCount() - g_lab_t0) / 1000)) + " s";
        double pv = -1; std::string pt;
        try { js::Value p = js::parse(js::read_file(g_lab_dir + "\\progress_" + lab_rid(g_lab_wait) + ".json")); pv = p.num("progress", -1); pt = p.str("text", ""); } catch (...) {}
        gp::RectF band(0, 0, (float)W, 40); gp::LinearGradientBrush gb(band, gcol(234, 88, 12), gcol(236, 72, 153), gp::LinearGradientModeHorizontal); g.FillRectangle(&gb, band);
        gtext(g, s + (pt.empty() ? "" : "   |   " + pt), gp::RectF(16, 0, (float)W - 32, 40), gcol(255, 255, 255), 13.0f, gp::FontStyleBold);
        if (pv >= 0) { gp::SolidBrush pb(gcol(255, 255, 255, 200)); g.FillRectangle(&pb, 0.0f, 36.0f, (float)(W * std::min(1.0, pv)), 4.0f); }
        y += 44;
    }
    std::string sec;
    for (LabBlock& b : g_lab_b) {
        if (!b.section.empty() && b.section != sec) { sec = b.section; y += 6; y += lab_text(g, sec, x, y, w, gcol(234, 88, 12), 15.0f, gp::FontStyleBold) + 4; gp::Pen pe(gcol(234, 88, 12, 120), 1.5f); g.DrawLine(&pe, x, y, x + w, y); y += 6; }
        b.y0 = y + g_lab_scroll;
        const std::string& t = b.type;
        if (t == "context") {
            gp::RectF r(x, y, w, 0); float th = lab_text(g, b.text, x + 12, y + 8, w - 24, gcol(20, 83, 45), 11.5f, gp::FontStyleRegular, true);
            r.Height = th + 16; gp::GraphicsPath p; round_path(p, r, 8); gp::SolidBrush cb(gcol(220, 252, 231, 230)); g.FillPath(&cb, &p); gp::Pen pe(gcol(22, 163, 74, 150), 1.2f); g.DrawPath(&pe, &p);
            lab_text(g, b.text, x + 12, y + 8, w - 24, gcol(20, 83, 45), 11.5f, gp::FontStyleRegular, true); y += r.Height + 10;
        } else if (t == "title" || t == "header") y += lab_text(g, b.text, x, y + 4, w, gcol(15, 23, 42), t == "title" ? 24.0f : 20.0f, gp::FontStyleBold) + 12;
        else if (t == "subheader") y += lab_text(g, b.text, x, y + 4, w, gcol(30, 41, 59), 16.0f, gp::FontStyleBold) + 8;
        else if (t == "caption") y += lab_text(g, b.text, x, y, w, gcol(100, 116, 139), 11.0f) + 6;
        else if (t == "markdown" || t == "text" || t == "latex") { std::string s = b.text; for (char& ch : s) if (ch == '*' || ch == '`' || ch == '#') ch = ' '; y += lab_text(g, s, x, y, w, gcol(30, 41, 59), 12.5f) + 8; }
        else if (t == "code" || t == "json") { float th = lab_text(g, b.text.substr(0, 6000), x + 10, y + 6, w - 20, gcol(226, 232, 240), 11.0f, gp::FontStyleRegular, true); gp::SolidBrush cb(gcol(15, 23, 42)); g.FillRectangle(&cb, x, y, w, th + 12); lab_text(g, b.text.substr(0, 6000), x + 10, y + 6, w - 20, gcol(226, 232, 240), 11.0f, gp::FontStyleRegular, true); y += th + 20; }
        else if (t == "info" || t == "success" || t == "warning" || t == "error") {
            gp::Color c = t == "info" ? gcol(37, 99, 235) : t == "success" ? gcol(22, 163, 74) : t == "warning" ? gcol(217, 119, 6) : gcol(220, 38, 38);
            float th = lab_text(g, b.text, x + 14, y + 8, w - 28, gcol(15, 23, 42), 12.0f);
            gp::SolidBrush cb(gcol(c.GetR(), c.GetG(), c.GetB(), 28)); g.FillRectangle(&cb, x, y, w, th + 16); gp::SolidBrush bar(c); g.FillRectangle(&bar, x, y, 4.0f, th + 16);
            lab_text(g, b.text, x + 14, y + 8, w - 28, gcol(15, 23, 42), 12.0f); y += th + 24;
        } else if (t == "metric") {  // metrics side by side as cards
            float cw = 210, chh = 64;
            static float mx = 0; static float my = 0;
            if (&b == &g_lab_b.front() || (&b - &g_lab_b.front() > 0 && (&b - 1)->type != "metric")) { mx = x; my = y; }
            if (mx + cw > x + w) { mx = x; my += chh + 10; }
            gp::RectF r(mx, my, cw, chh); gp::GraphicsPath p; round_path(p, r, 10); gp::SolidBrush cb(gcol(255, 255, 255)); soft_shadow(g, r, 10, 4, 40); g.FillPath(&cb, &p);
            gtext(g, b.label, gp::RectF(mx + 12, my + 6, cw - 24, 18), gcol(100, 116, 139), 11.0f);
            gtext(g, b.value, gp::RectF(mx + 12, my + 24, cw - 24, 30), gcol(15, 23, 42), 18.0f, gp::FontStyleBold);
            mx += cw + 10;
            bool last = &b == &g_lab_b.back() || (&b + 1)->type != "metric";
            y = last ? my + chh + 14 : my;
        } else if (t == "figure") {
            if (gp::Bitmap* im = load_png(g_lab_resdir + "\\" + b.file)) {
                float iw = (float)im->GetWidth(), ih = (float)im->GetHeight(), s = std::min(w / iw, 1.0f), hh = ih * s, ww = iw * s, fx = x + (w - ww) / 2;
                if (y + hh > -40 && y < H + 40) { gp::RectF fr(fx - 4, y - 4, ww + 8, hh + 8); soft_shadow(g, fr, 8, 5, 60); gp::SolidBrush wb(gcol(255, 255, 255)); g.FillRectangle(&wb, fr); g.DrawImage(im, fx, y, ww, hh); }
                y += hh + 18;
            }
        } else if (t == "table") {
            std::vector<std::vector<std::string>> rows;
            try {
                std::string s = js::read_file(g_lab_resdir + "\\" + b.file); size_t pos = 0;
                while (pos < s.size() && rows.size() < 26) { size_t e = s.find('\n', pos); if (e == std::string::npos) e = s.size(); std::string ln = s.substr(pos, e - pos); pos = e + 1; if (!ln.empty() && ln.back() == '\r') ln.pop_back(); std::vector<std::string> c; size_t a = 0; while (true) { size_t q = ln.find(',', a); c.push_back(ln.substr(a, q == std::string::npos ? std::string::npos : q - a)); if (q == std::string::npos) break; a = q + 1; } rows.push_back(c); }
            } catch (...) {}
            if (!rows.empty()) {
                size_t nc = std::min<size_t>(rows[0].size(), 9); float cw = w / std::max<size_t>(1, nc), rh = 20;
                for (size_t r = 0; r < rows.size(); ++r) {
                    gp::SolidBrush rb(r == 0 ? gcol(30, 41, 59) : r % 2 ? gcol(255, 255, 255) : gcol(241, 245, 249)); g.FillRectangle(&rb, x, y, w, rh);
                    for (size_t c = 0; c < nc && c < rows[r].size(); ++c) gtext(g, rows[r][c], gp::RectF(x + c * cw + 6, y, cw - 8, rh), r == 0 ? gcol(255, 255, 255) : gcol(30, 41, 59), 11.0f, r == 0 ? gp::FontStyleBold : gp::FontStyleRegular);
                    y += rh;
                }
                if (b.rows > 25 || rows[0].size() > nc) { y += lab_text(g, "(" + tostr(b.rows) + " rows x " + tostr((int)rows[0].size()) + " columns: click the table to see it all as charts, or Folder)", x, y + 2, w, gcol(100, 116, 139), 10.5f); }
                y += 14;
            }
        } else if (t == "download") { y += lab_text(g, "Download: " + b.label + "  ->  " + b.file + "  (click to open its folder)", x, y, w, gcol(37, 99, 235), 12.5f, gp::FontStyleBold) + 10; }
        else if (t == "progress") { gp::SolidBrush pb(gcol(226, 232, 240)); g.FillRectangle(&pb, x, y, w, 8.0f); gp::SolidBrush pf(gcol(234, 88, 12)); g.FillRectangle(&pf, x, y, (float)(w * std::min(1.0, b.prog)), 8.0f); y += 12 + lab_text(g, b.text, x, y + 10, w, gcol(100, 116, 139), 11.0f); }
        else if (t == "divider") { gp::Pen pe(gcol(203, 213, 225), 1.0f); g.DrawLine(&pe, x, y + 6, x + w, y + 6); y += 14; }
        b.y1 = y + g_lab_scroll;
    }
    if (g_lab_b.empty() && g_lab_wait < 0) gtext(g, "Pick an app's page on the left. Its inputs appear in the middle (the design's values marked), its results here.", gp::RectF(20, 20, (float)W - 40, 60), gcol(100, 116, 139), 14.0f, gp::FontStyleRegular, gp::StringAlignmentNear, gp::StringAlignmentNear, true);
    g_lab_total = y + g_lab_scroll + 30;
    gp::Graphics out(dc); out.DrawImage(&buf, 0, 0);
}
static LRESULT CALLBACK lab_rep_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_PAINT: { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_lab_report(h, dc); EndPaint(h, &ps); return 0; }
        case WM_ERASEBKGND: return 1;
        case WM_MOUSEWHEEL: { RECT rc; GetClientRect(h, &rc); g_lab_scroll = std::max(0.0f, std::min(std::max(0.0f, g_lab_total - rc.bottom), g_lab_scroll - GET_WHEEL_DELTA_WPARAM(w) * 0.8f)); InvalidateRect(h, nullptr, FALSE); return 0; }
        case WM_LBUTTONDOWN: SetFocus(h); return 0;
        case WM_LBUTTONUP: {
            float yy = (float)(short)HIWORD(l) + g_lab_scroll;
            for (const LabBlock& b : g_lab_b) {
                if (yy < b.y0 || yy > b.y1) continue;
                if (b.type == "figure") pop_out(g_lab_resdir, b.file, SVK_PNG, g_lab_cur >= 0 ? g_lab_pages[(size_t)g_lab_cur].page : "Labs");
                else if (b.type == "table") pop_out(g_lab_resdir, b.file, SVK_CSV, g_lab_cur >= 0 ? g_lab_pages[(size_t)g_lab_cur].page : "Labs");
                else if (b.type == "download") { std::string f = g_lab_resdir + "\\" + b.file; for (char& c : f) if (c == '/') c = '\\'; ShellExecuteA(h, "open", f.substr(0, f.find_last_of('\\')).c_str(), nullptr, nullptr, SW_SHOWNORMAL); }
                break;
            }
            return 0;
        }
        case WM_KEYDOWN: { RECT rc; GetClientRect(h, &rc); float d = w == VK_NEXT ? rc.bottom * 0.9f : w == VK_PRIOR ? -rc.bottom * 0.9f : w == VK_HOME ? -1e9f : w == VK_END ? 1e9f : 0; g_lab_scroll = std::max(0.0f, std::min(std::max(0.0f, g_lab_total - rc.bottom), g_lab_scroll + d)); InvalidateRect(h, nullptr, FALSE); return 0; }
    }
    return DefWindowProcA(h, m, w, l);
}

// ---------------------------------------------------------------- the window
static void lab_layout() {
    RECT rc; GetClientRect(g_lab, &rc);
    int W = rc.right, H = rc.bottom, lw = 250, fw = 380;
    MoveWindow(g_lab_list, 8, 8, lw - 16, H - 16, TRUE);
    MoveWindow(g_lab_status, lw, 8, fw - 8, 34, TRUE);
    MoveWindow(g_lab_run, lw, 46, 110, 30, TRUE);
    MoveWindow(GetDlgItem(g_lab, ID_LAB_RESET), lw + 116, 46, 90, 30, TRUE);
    MoveWindow(g_lab_design, lw + 212, 46, fw - 220, 30, TRUE);
    MoveWindow(g_lab_form, lw, 82, fw - 8, H - 90, TRUE);
    MoveWindow(g_lab_rep, lw + fw, 0, W - lw - fw, H, TRUE);
    if (g_lab_form) lab_form_layout();
}
static void lab_fill_list() {
    SendMessageA(g_lab_list, LB_RESETCONTENT, 0, 0); g_lab_row.clear();
    std::string last;
    for (size_t i = 0; i < g_lab_pages.size(); ++i) {
        if (g_lab_pages[i].app_title != last) { last = g_lab_pages[i].app_title; SendMessageA(g_lab_list, LB_ADDSTRING, 0, (LPARAM)last.c_str()); g_lab_row.push_back(-1); }
        SendMessageA(g_lab_list, LB_ADDSTRING, 0, (LPARAM)("      " + g_lab_pages[i].page).c_str()); g_lab_row.push_back((int)i);
    }
}
static void lab_select(int page) {
    if (page < 0 || page >= (int)g_lab_pages.size()) return;
    if (g_lab_cur >= 0) lab_collect();
    g_lab_cur = page;
    for (size_t r = 0; r < g_lab_row.size(); ++r) if (g_lab_row[r] == page) SendMessageA(g_lab_list, LB_SETCURSEL, (WPARAM)r, 0);
    lab_request(true);
}
static void lab_tick() {
    if (g_lab_wait < 0) { KillTimer(g_lab, 1); return; }
    std::string res = g_lab_dir + "\\res_" + lab_rid(g_lab_wait);
    if (file_exists(res + "\\done.json")) {
        KillTimer(g_lab, 1);
        int was = g_lab_wait; g_lab_wait = -1;
        std::string msg;
        try { js::Value d = js::parse(js::read_file(res + "\\done.json")); SYSTEMTIME tm; GetLocalTime(&tm); char clk[16]; snprintf(clk, sizeof clk, "%02d:%02d:%02d", tm.wHour, tm.wMinute, tm.wSecond);
            msg = d.flag("ok", false) ? "finished at " + std::string(clk) + ", in " + fmtd(std::round(d.num("seconds", 0) * 10) / 10) + " s: " + tostr((int)d.num("widgets", 0)) + " inputs, " + tostr((int)d.num("blocks", 0)) + " results" : "finished with an error (shown in red): " + d.str("error", ""); } catch (...) {}
        lab_load_result(res);
        SetWindowTextA(g_lab_status, (g_lab_pages[(size_t)g_lab_cur].app_title + ": " + g_lab_pages[(size_t)g_lab_cur].page + "  |  " + msg).c_str());
        (void)was;
        return;
    }
    if (g_lab_proc && WaitForSingleObject(g_lab_proc, 0) != WAIT_TIMEOUT) {  // the worker stopped: its log tells why
        KillTimer(g_lab, 1); g_lab_wait = -1; CloseHandle(g_lab_proc); g_lab_proc = nullptr;
        std::string t; try { t = js::read_file(g_lab_dir + "\\worker.log"); } catch (...) {}
        LabBlock b; b.type = "error"; b.text = "The Python worker stopped. Its log:\n" + (t.size() > 3000 ? t.substr(t.size() - 3000) : t); g_lab_b.clear(); g_lab_b.push_back(b);
    }
    InvalidateRect(g_lab_rep, nullptr, FALSE);
}
static LRESULT CALLBACK lab_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            HINSTANCE hi = GetModuleHandleA(nullptr);
            g_lab_list = CreateWindowExA(WS_EX_CLIENTEDGE, "LISTBOX", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | LBS_NOTIFY | LBS_NOINTEGRALHEIGHT, 0, 0, 0, 0, h, (HMENU)ID_LAB_LIST, hi, nullptr);
            g_lab_status = CreateWindowExA(0, "STATIC", "", WS_CHILD | WS_VISIBLE | SS_LEFT, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_lab_run = CreateWindowExA(0, "BUTTON", "Run", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)ID_LAB_RUN, hi, nullptr);
            CreateWindowExA(0, "BUTTON", "Reset", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)ID_LAB_RESET, hi, nullptr);
            g_lab_design = CreateWindowExA(0, "BUTTON", "Inputs from the design", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 0, 0, 0, 0, h, (HMENU)ID_LAB_DESIGN, hi, nullptr);
            SendMessageA(g_lab_design, BM_SETCHECK, BST_CHECKED, 0);
            g_lab_form = CreateWindowExA(WS_EX_CONTROLPARENT, "AeroSynapseLabForm", "", WS_CHILD | WS_VISIBLE | WS_CLIPCHILDREN, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_lab_rep = CreateWindowExA(0, "AeroSynapseLabReport", "", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            for (HWND c : {g_lab_list, g_lab_status, g_lab_design}) SendMessageA(c, WM_SETFONT, (WPARAM)g_font, TRUE);
            return 0;
        }
        case WM_SIZE: lab_layout(); return 0;
        case WM_TIMER: lab_tick(); return 0;
        case WM_COMMAND:
            if (LOWORD(w) == ID_LAB_LIST && HIWORD(w) == LBN_SELCHANGE) {
                int r = (int)SendMessageA(g_lab_list, LB_GETCURSEL, 0, 0);
                if (r >= 0 && r < (int)g_lab_row.size()) { int p = g_lab_row[(size_t)r]; if (p < 0 && r + 1 < (int)g_lab_row.size()) p = g_lab_row[(size_t)r + 1]; lab_select(p); }
            }
            if (LOWORD(w) == ID_LAB_RUN) lab_request(false);
            if (LOWORD(w) == ID_LAB_RESET && g_lab_cur >= 0) { g_lab_vals.erase(g_lab_pages[(size_t)g_lab_cur].app + "|" + g_lab_pages[(size_t)g_lab_cur].page); lab_request(true); }
            if (LOWORD(w) == ID_LAB_DESIGN) {
                g_lab_use_design = SendMessageA(g_lab_design, BM_GETCHECK, 0, 0) == BST_CHECKED;
                if (g_lab_cur >= 0) {  // the design's values replace what was typed (or the page's own defaults come back)
                    std::map<std::string, std::string>& V = g_lab_vals[g_lab_pages[(size_t)g_lab_cur].app + "|" + g_lab_pages[(size_t)g_lab_cur].page];
                    for (const LabWidget& x : g_lab_w) if (x.source.compare(0, 6, "design") == 0 || g_lab_use_design) V.erase(x.key);
                    for (LabWidget& x : g_lab_w) x.ctl = x.ctl;  // (the fields are rebuilt from the answer)
                    lab_request(true);
                }
            }
            return 0;
        case WM_DRAWITEM: { DRAWITEMSTRUCT* d = (DRAWITEMSTRUCT*)l; draw_button(d, d->CtlID == ID_LAB_RUN ? gcol(22, 163, 74) : gcol(100, 116, 139)); return TRUE; }
        case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkMode(dc, TRANSPARENT); SetTextColor(dc, C_INK); return (LRESULT)g_br_panel; }
        case WM_CTLCOLORLISTBOX: { HDC dc = (HDC)w; SetBkColor(dc, C_FIELD); SetTextColor(dc, C_INK); return (LRESULT)g_br_field; }
        case WM_CLOSE: ShowWindow(h, SW_HIDE); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}
static void open_labs(const std::string& app, const std::string& page) {
    if (!g_lab) {
        g_lab = CreateWindowExA(0, "AeroSynapseLabs", "Labs: THERMAL LAB, PINN heat transfer, Designer, research workbench (inside Studio)", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1560, 940, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
        lab_layout();
    }
    ShowWindow(g_lab, SW_SHOW); SetForegroundWindow(g_lab);
    if (g_lab_pages.empty()) {  // the apps and their pages, from the worker's package (once)
        std::string d = work_dir("labs"), f = d + "\\apps.json";
        WIN32_FILE_ATTRIBUTE_DATA fa, fb;
        bool stale = GetFileAttributesExA(f.c_str(), GetFileExInfoStandard, &fa) && GetFileAttributesExA((g_root + "\\src\\aerosynapse\\nativelab\\run.py").c_str(), GetFileExInfoStandard, &fb) &&
                     CompareFileTime(&fb.ftLastWriteTime, &fa.ftLastWriteTime) > 0;
        if (!file_exists(f) || stale || g_selftest) {
            std::string cmd = "cmd /c \"\"" + python() + "\" -m aerosynapse.nativelab list \"" + f + "\"\"";
            SetEnvironmentVariableA("PYTHONPATH", (g_root + "\\src").c_str());
            STARTUPINFOA si; memset(&si, 0, sizeof si); si.cb = sizeof si; PROCESS_INFORMATION pi;
            SetWindowTextA(g_lab_status, "reading the apps...");
            if (CreateProcessA(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, g_root.c_str(), &si, &pi)) { WaitForSingleObject(pi.hProcess, g_selftest ? 5000 : 180000); CloseHandle(pi.hProcess); CloseHandle(pi.hThread); }
        }
        try {
            js::Value a = js::parse(js::read_file(f));
            if (const js::Value* apps = a.find("apps"))
                for (size_t i = 0; i < apps->size(); ++i) { const js::Value& x = apps->at(i); if (const js::Value* p = x.find("pages")) for (size_t k = 0; k < p->size(); ++k) g_lab_pages.push_back(LabPage{x.str("key", ""), x.str("title", ""), p->at(k).is_str() ? p->at(k).s : std::string()}); }
        } catch (...) { SetWindowTextA(g_lab_status, "the apps could not be listed (see results/studio/labs)"); }
        lab_fill_list();
    }
    SetWindowTextA(g_lab_status, labs_context_line().c_str());
    for (size_t i = 0; i < g_lab_pages.size(); ++i) if (g_lab_pages[i].app == app && (page.empty() || g_lab_pages[i].page == page)) { lab_select((int)i); break; }
}
static void register_lab_classes(HINSTANCE hi, HICON icon) {
    WNDCLASSA a; memset(&a, 0, sizeof a);
    a.lpfnWndProc = lab_proc; a.hInstance = hi; a.hCursor = LoadCursor(nullptr, IDC_ARROW); a.hbrBackground = g_br_panel; a.lpszClassName = "AeroSynapseLabs"; a.hIcon = icon;
    RegisterClassA(&a);
    a.lpfnWndProc = lab_form_proc; a.lpszClassName = "AeroSynapseLabForm"; RegisterClassA(&a);
    a.lpfnWndProc = lab_rep_proc; a.hbrBackground = nullptr; a.style = CS_DBLCLKS; a.lpszClassName = "AeroSynapseLabReport"; RegisterClassA(&a);
}
