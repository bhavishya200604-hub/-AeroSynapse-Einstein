// Simulation setup forms, preferences, the parameter help of every feature property, and the sliding tool bar.
// The setup form is built from assets/sim_setup_schema.json (written by `aerosynapse.cli simulate schema -`), so every input the
// Python analyses accept is on the form, with its unit, range and help.

// ---------------------------------------------------------------- preferences (settings.json, shared with the Python side)
static void prefs_load() {
    try {
        js::Value s = js::parse(js::read_file(g_root + "\\settings.json"));
        g_pref_plot = s.str("plot_style", "plain"); g_pref_display = s.str("display", "calm"); g_pref_palette = s.str("palette", "fixed");
    } catch (...) {}
    g_calm = g_pref_display != "vivid"; g_autohide = g_pref_palette == "autohide";
}
static void prefs_save() {
    js::Value s = js::Value::object();
    try { s = js::parse(js::read_file(g_root + "\\settings.json")); } catch (...) {}
    s["plot_style"] = g_pref_plot; s["display"] = g_pref_display; s["palette"] = g_pref_palette;
    js::write_file(g_root + "\\settings.json", js::dump(s, 2));
    g_calm = g_pref_display != "vivid"; g_autohide = g_pref_palette == "autohide";
}

// ---------------------------------------------------------------- sliding tool bar
static void layout();
static void palette_target() {
    g_pal_target = !g_show_palette ? 0 : g_autohide ? (g_pal_open ? PALETTE_W : PAL_HANDLE) : PALETTE_W;
    if (g_pal_w != g_pal_target) SetTimer(g_main, 10, 15, nullptr);
}
static void palette_tick() {  // timer 10: slide the tool bar towards its target width
    int d = g_pal_target - g_pal_w, stepw = std::max(6, std::abs(d) / 3);
    g_pal_w = std::abs(d) <= stepw ? g_pal_target : g_pal_w + (d > 0 ? stepw : -stepw);
    if (g_pal_w == g_pal_target) KillTimer(g_main, 10);
    layout();
}
static void palette_hover(bool in) {  // auto-hide: slide out under the mouse, back after it leaves
    if (!g_autohide) return;
    if (in) { KillTimer(g_main, 11); if (!g_pal_open) { g_pal_open = true; palette_target(); } }
    else SetTimer(g_main, 11, 450, nullptr);
}
static void palette_leave_tick() {  // timer 11
    KillTimer(g_main, 11);
    POINT p; GetCursorPos(&p); RECT r; GetWindowRect(g_palette, &r);
    if (PtInRect(&r, p)) return;
    g_pal_open = false; palette_target();
}

// ---------------------------------------------------------------- simulation setup window
enum { ID_SETUP_RUN = 9600, ID_SETUP_DEFAULTS, ID_SETUP_SAVE, ID_SETUP_LOAD, ID_SETUP_CLOSE, ID_SETUP_KIND, ID_SETUP_ROW0 = 9700,
       ID_PREF_PLOT = 9660, ID_PREF_DISPLAY, ID_PREF_PALETTE, ID_PREF_OK };
struct SetupRow { std::string key, type, label, unit, help; HWND lbl = nullptr, ctl = nullptr, ul = nullptr; bool has_lo = false, has_hi = false; double lo = 0, hi = 0; std::vector<std::string> choices; };
static HWND g_setup = nullptr, g_setup_page = nullptr, g_setup_help = nullptr, g_setup_kindcb = nullptr;
static js::Value g_schema;
static std::string g_setup_kind;
static std::vector<SetupRow> g_srows;
static std::vector<HWND> g_sgroups;
static int g_setup_scroll = 0, g_setup_h = 0;
static HFONT g_font_bold = nullptr;
static const char* SETUP_KINDS[] = {"cfd", "optimise", "turbulence", "analytic", "pinn", "structure", "composites", "clash"};

static bool schema_load() {
    if (g_schema.is_obj()) return true;
    try { g_schema = js::parse(js::read_file(g_root + "\\assets\\sim_setup_schema.json")); return true; }
    catch (std::exception& e) { MessageBoxA(g_main, (std::string("The setup schema is missing (assets\\sim_setup_schema.json): ") + e.what()).c_str(), "Setup", MB_ICONERROR); return false; }
}
static std::string setup_store(const std::string& kind) {
    CreateDirectoryA((g_root + "\\results").c_str(), nullptr); CreateDirectoryA((g_root + "\\results\\studio").c_str(), nullptr);
    return g_root + "\\results\\studio\\setup_" + kind + ".json";
}
static std::string vtext(const js::Value& v) {
    if (v.kind == js::Value::Num) { char b[48]; snprintf(b, sizeof b, "%.6g", v.n); return b; }
    if (v.kind == js::Value::Bool) return v.b ? "true" : "false";
    if (v.kind == js::Value::Str) return v.s;
    return "";
}
static void srow_set(SetupRow& r, const js::Value& v) {
    if (r.type == "bool") SendMessageA(r.ctl, BM_SETCHECK, (v.kind == js::Value::Bool ? v.b : v.n != 0) ? BST_CHECKED : BST_UNCHECKED, 0);
    else if (r.type == "choice") {
        std::string s = vtext(v);
        for (size_t i = 0; i < r.choices.size(); ++i) if (r.choices[i] == s) SendMessageA(r.ctl, CB_SETCURSEL, i, 0);
    } else SetWindowTextA(r.ctl, vtext(v).c_str());
}
static bool srow_get(const SetupRow& r, js::Value& out, std::string& err) {
    if (r.type == "bool") { out = js::Value(SendMessageA(r.ctl, BM_GETCHECK, 0, 0) == BST_CHECKED); return true; }
    if (r.type == "choice") { int i = (int)SendMessageA(r.ctl, CB_GETCURSEL, 0, 0); out = js::Value(i >= 0 && i < (int)r.choices.size() ? r.choices[(size_t)i] : std::string()); return true; }
    std::string t = read_edit(r.ctl);
    while (!t.empty() && t.back() == ' ') t.pop_back();
    if (r.type == "text") { out = js::Value(t); return true; }
    char* e = nullptr; double v = strtod(t.c_str(), &e);
    if (t.empty() || e == t.c_str()) { err = r.label + ": type a number"; return false; }
    if ((r.has_lo && v < r.lo) || (r.has_hi && v > r.hi)) { char b[256]; snprintf(b, sizeof b, "%s = %g is outside %g to %g %s", r.label.c_str(), v, r.lo, r.hi, r.unit.c_str()); err = b; return false; }
    out = js::Value(r.type == "int" ? std::floor(v + 0.5) : v);
    return true;
}
static void setup_help_for(const SetupRow& r) {
    std::string t = r.label + (r.unit.empty() ? "" : "  [" + r.unit + "]") + "\r\n\r\n" + (r.help.empty() ? "" : r.help + "\r\n\r\n");
    if (r.has_lo || r.has_hi) { char b[128]; snprintf(b, sizeof b, "Range: %g to %g", r.lo, r.hi); t += b; }
    if (!r.choices.empty()) { t += "Choices: "; for (size_t i = 0; i < r.choices.size(); ++i) t += (i ? ", " : "") + r.choices[i]; }
    SetWindowTextA(g_setup_help, t.c_str());
}
static void setup_scroll_to(int y) {
    RECT rc; GetClientRect(g_setup_page, &rc);
    int mx = std::max(0, g_setup_h - (int)rc.bottom);
    y = std::max(0, std::min(mx, y));
    int dy = g_setup_scroll - y;
    g_setup_scroll = y;
    if (dy) ScrollWindowEx(g_setup_page, 0, dy, nullptr, nullptr, nullptr, nullptr, SW_SCROLLCHILDREN | SW_INVALIDATE | SW_ERASE);
    SCROLLINFO si; si.cbSize = sizeof si; si.fMask = SIF_ALL; si.nMin = 0; si.nMax = std::max(0, g_setup_h - 1); si.nPage = (UINT)rc.bottom; si.nPos = g_setup_scroll; si.nTrackPos = 0;
    SetScrollInfo(g_setup_page, SB_VERT, &si, TRUE);
}
static void setup_build(const std::string& kind) {
    if (!schema_load()) return;
    const js::Value* K = g_schema.find(kind);
    if (!K) return;
    g_setup_kind = kind;
    for (auto& r : g_srows) { DestroyWindow(r.lbl); DestroyWindow(r.ctl); if (r.ul) DestroyWindow(r.ul); }
    for (HWND gh : g_sgroups) DestroyWindow(gh);
    g_srows.clear(); g_sgroups.clear();
    setup_scroll_to(0);
    HINSTANCE hi = GetModuleHandleA(nullptr);
    RECT rc; GetClientRect(g_setup_page, &rc);
    int W = std::max(300L, rc.right), y = 8, lw = 210, cw = std::max(140, W - lw - 130), id = ID_SETUP_ROW0;
    std::string grp;
    const js::Value* fs = K->find("fields");
    for (size_t i = 0; fs && i < fs->size(); ++i) {
        const js::Value& f = fs->at(i);
        SetupRow r;
        r.key = f.str("key", ""); r.type = f.str("type", "num"); r.label = f.str("label", r.key); r.unit = f.str("unit", ""); r.help = f.str("help", "");
        if (const js::Value* v = f.find("min")) { r.has_lo = true; r.lo = v->n; }
        if (const js::Value* v = f.find("max")) { r.has_hi = true; r.hi = v->n; }
        if (const js::Value* c = f.find("choices")) for (size_t q = 0; q < c->size(); ++q) r.choices.push_back(c->at(q).s);
        std::string g = f.str("group", "");
        if (g != grp) {
            grp = g; y += 6;
            HWND gh = CreateWindowExA(0, "STATIC", g.c_str(), WS_CHILD | WS_VISIBLE, 8, y, W - 16, 20, g_setup_page, nullptr, hi, nullptr);
            SendMessageA(gh, WM_SETFONT, (WPARAM)g_font_bold, TRUE);
            g_sgroups.push_back(gh); y += 24;
        }
        r.lbl = CreateWindowExA(0, "STATIC", r.label.c_str(), WS_CHILD | WS_VISIBLE | SS_RIGHT, 8, y + 3, lw - 14, 20, g_setup_page, nullptr, hi, nullptr);
        if (r.type == "bool") r.ctl = CreateWindowExA(0, "BUTTON", "", WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX, lw, y + 2, 22, 20, g_setup_page, (HMENU)(INT_PTR)id, hi, nullptr);
        else if (r.type == "choice") {
            r.ctl = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST, lw, y, cw, 300, g_setup_page, (HMENU)(INT_PTR)id, hi, nullptr);
            for (auto& c : r.choices) SendMessageA(r.ctl, CB_ADDSTRING, 0, (LPARAM)c.c_str());
        } else r.ctl = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL, lw, y, cw, 22, g_setup_page, (HMENU)(INT_PTR)id, hi, nullptr);
        if (!r.unit.empty()) r.ul = CreateWindowExA(0, "STATIC", r.unit.c_str(), WS_CHILD | WS_VISIBLE, lw + cw + 6, y + 3, 120, 20, g_setup_page, nullptr, hi, nullptr);
        HWND hs[] = {r.lbl, r.ctl, r.ul};
        for (HWND x : hs) if (x) SendMessageA(x, WM_SETFONT, (WPARAM)g_font, TRUE);
        if (const js::Value* dv = f.find("default")) srow_set(r, *dv);
        g_srows.push_back(r); ++id; y += 27;
    }
    g_setup_h = y + 10;
    try {  // the values used last time
        js::Value last = js::parse(js::read_file(setup_store(kind)));
        for (auto& r : g_srows) if (const js::Value* v = last.find(r.key)) srow_set(r, *v);
    } catch (...) {}
    setup_scroll_to(0);
    SetWindowTextA(g_setup, ("Setup: " + K->str("title", kind)).c_str());
    for (size_t i = 0; i < sizeof SETUP_KINDS / sizeof *SETUP_KINDS; ++i) if (kind == SETUP_KINDS[i]) SendMessageA(g_setup_kindcb, CB_SETCURSEL, i, 0);
    SetWindowTextA(g_setup_help, ("Choose every condition of the analysis, then Run. Click a field to read what it means, its unit and its range.\r\n\r\n"
                                  "The values are remembered for next time (Defaults restores the original ones)."));
    InvalidateRect(g_setup, nullptr, TRUE);
}
static void setup_apply_preset(const std::string& name) {
    const js::Value* K = g_schema.find(g_setup_kind);
    const js::Value* ps = K ? K->find("presets") : nullptr;
    const js::Value* p = ps ? ps->find(name) : nullptr;
    if (!p) return;
    for (auto& kv : *p->o) for (auto& r : g_srows) if (r.key == kv.first) srow_set(r, kv.second);
}
static bool setup_collect(js::Value& v) {
    v = js::Value::object();
    for (auto& r : g_srows) {
        js::Value x; std::string err;
        if (!srow_get(r, x, err)) { MessageBoxA(g_setup, err.c_str(), "Setup", MB_ICONWARNING); SetFocus(r.ctl); setup_help_for(r); return false; }
        v[r.key] = x;
    }
    return true;
}
static void setup_run() {
    js::Value v;
    if (!setup_collect(v)) return;
    js::write_file(setup_store(g_setup_kind), js::dump(v, 1));
    std::string k = g_setup_kind, tag = k == "cfd" ? v.str("preset", "custom") + "_" + v.str("quality", "draft") : k;
    std::string d = work_dir((k == "composites" || k == "clash" ? "eng_" : k == "cfd" ? "cfd_" : k == "optimise" ? "opt_" : "sim_") + tag);
    js::write_file(d + "\\design.json", js::dump(g_doc, 2));
    js::write_file(d + "\\setup.json", js::dump(v, 1));
    std::string py = "\"" + python() + "\" -m aerosynapse.cli ";
    bool ok;
    if (k == "composites" || k == "clash")
        ok = launch(py + "engineer " + k + " \"" + d + "\\design.json\" --setup \"" + d + "\\setup.json\" --out \"" + d + "\"", "engineer " + k, d);
    else
        ok = launch(py + "simulate " + k + " \"" + d + "\\design.json\" --setup \"" + d + "\\setup.json\" --out \"" + d + "\"",
                    k == "cfd" ? "simulation cfd " + v.str("preset", "custom") : k == "optimise" ? "CFD optimisation (" + tostr((long)v.num("budget", 14)) + " runs)" : "simulation " + k, d);
    if (ok) { ShowWindow(g_setup, SW_HIDE); status("running " + k + " with your setup; the results open when it finishes"); }
}
static void setup_layout() {
    RECT rc; GetClientRect(g_setup, &rc);
    int W = rc.right, H = rc.bottom, hw = 270, top = 78, bh = 44;
    MoveWindow(g_setup_kindcb, 110, 44, 260, 300, TRUE);
    MoveWindow(g_setup_page, 0, top, W - hw, H - top - bh, TRUE);
    MoveWindow(g_setup_help, W - hw + 8, top + 4, hw - 16, H - top - bh - 8, TRUE);
    int x = 10, y = H - bh + 7;
    int ids[] = {ID_SETUP_RUN, ID_SETUP_DEFAULTS, ID_SETUP_LOAD, ID_SETUP_SAVE, ID_SETUP_CLOSE}, ws[] = {150, 100, 100, 100, 90};
    for (int i = 0; i < 5; ++i) { MoveWindow(GetDlgItem(g_setup, ids[i]), x, y, ws[i], 30, TRUE); x += ws[i] + 8; }
}
static LRESULT CALLBACK setup_page_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_VSCROLL: {
            SCROLLINFO si; si.cbSize = sizeof si; si.fMask = SIF_ALL; GetScrollInfo(h, SB_VERT, &si);
            int y = g_setup_scroll;
            switch (LOWORD(w)) { case SB_LINEUP: y -= 27; break; case SB_LINEDOWN: y += 27; break; case SB_PAGEUP: y -= (int)si.nPage; break; case SB_PAGEDOWN: y += (int)si.nPage; break;
                                 case SB_THUMBTRACK: case SB_THUMBPOSITION: y = si.nTrackPos; break; }
            setup_scroll_to(y); return 0;
        }
        case WM_MOUSEWHEEL: setup_scroll_to(g_setup_scroll - (short)HIWORD(w) / 2); return 0;
        case WM_SIZE: setup_scroll_to(g_setup_scroll); return 0;
        case WM_CTLCOLORSTATIC: {
            HDC dc = (HDC)w; SetBkColor(dc, C_PANEL);
            bool grp = std::find(g_sgroups.begin(), g_sgroups.end(), (HWND)l) != g_sgroups.end();
            SetTextColor(dc, grp ? C_NAVY : C_INK); return (LRESULT)g_br_panel;
        }
        case WM_CTLCOLOREDIT: case WM_CTLCOLORLISTBOX: { HDC dc = (HDC)w; SetBkColor(dc, C_FIELD); SetTextColor(dc, C_INK); return (LRESULT)g_br_field; }
        case WM_COMMAND: {
            int id = LOWORD(w), code = HIWORD(w), i = id - ID_SETUP_ROW0;
            if (i >= 0 && i < (int)g_srows.size()) {
                SetupRow& r = g_srows[(size_t)i];
                if (code == EN_SETFOCUS || code == CBN_SETFOCUS || code == BN_SETFOCUS || code == BN_CLICKED) setup_help_for(r);
                if (code == EN_CHANGE || code == CBN_SELCHANGE || code == BN_CLICKED) InvalidateRect(g_view, nullptr, FALSE);  // the preview follows the form
                if (code == CBN_SELCHANGE && r.key == "preset") { int s = (int)SendMessageA(r.ctl, CB_GETCURSEL, 0, 0); if (s >= 0) setup_apply_preset(r.choices[(size_t)s]); }
                // keep the field in view when the keyboard moves into it
                if (code == EN_SETFOCUS || code == CBN_SETFOCUS || code == BN_SETFOCUS) {
                    RECT cr; GetWindowRect(r.ctl, &cr); POINT p = {cr.left, cr.top}; ScreenToClient(h, &p);
                    RECT pr; GetClientRect(h, &pr);
                    if (p.y < 0 || p.y + 26 > pr.bottom) setup_scroll_to(g_setup_scroll + p.y - pr.bottom / 3);
                }
            }
            return 0;
        }
    }
    return DefWindowProcA(h, m, w, l);
}
static LRESULT CALLBACK setup_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            HINSTANCE hi = GetModuleHandleA(nullptr);
            if (!g_font_bold) g_font_bold = CreateFontA(-14, 0, 0, 0, FW_BOLD, 0, 0, 0, ANSI_CHARSET, 0, 0, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
            HWND lab = CreateWindowExA(0, "STATIC", "Analysis:", WS_CHILD | WS_VISIBLE | SS_RIGHT, 10, 48, 94, 20, h, nullptr, hi, nullptr);
            g_setup_kindcb = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | CBS_DROPDOWNLIST, 0, 0, 0, 0, h, (HMENU)ID_SETUP_KIND, hi, nullptr);
            if (schema_load()) for (const char* k : SETUP_KINDS) { const js::Value* K = g_schema.find(k); SendMessageA(g_setup_kindcb, CB_ADDSTRING, 0, (LPARAM)(K ? K->str("title", k) : std::string(k)).c_str()); }
            g_setup_page = CreateWindowExA(0, "AeroSynapseSetupPage", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_CLIPCHILDREN, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_setup_help = CreateWindowExA(0, "STATIC", "", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            const char* names[] = {"Run with this setup", "Defaults", "Load...", "Save...", "Close"};
            int ids[] = {ID_SETUP_RUN, ID_SETUP_DEFAULTS, ID_SETUP_LOAD, ID_SETUP_SAVE, ID_SETUP_CLOSE};
            for (int i = 0; i < 5; ++i) { HWND b = CreateWindowExA(0, "BUTTON", names[i], WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)(INT_PTR)ids[i], hi, nullptr); SendMessageA(b, WM_SETFONT, (WPARAM)g_font, TRUE); }
            HWND hs[] = {lab, g_setup_kindcb, g_setup_help};
            for (HWND x : hs) SendMessageA(x, WM_SETFONT, (WPARAM)g_font, TRUE);
            return 0;
        }
        case WM_SIZE: setup_layout(); return 0;
        case WM_PAINT: {
            PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps);
            RECT rc; GetClientRect(h, &rc);
            gp::Graphics g(dc); quality(g);
            gp::LinearGradientBrush bg(gp::Rect(0, 0, rc.right, 38), gcol(236, 72, 153), gcol(22, 50, 92), gp::LinearGradientModeHorizontal);
            g.FillRectangle(&bg, 0, 0, (int)rc.right, 38);
            const js::Value* K = g_schema.is_obj() ? g_schema.find(g_setup_kind) : nullptr;
            gtext(g, std::string("Setup: ") + (K ? K->str("title", g_setup_kind) : g_setup_kind) + "   (every input of the analysis; units on the right of each field)",
                  gp::RectF(10, 0, (float)rc.right - 20, 38), gcol(255, 255, 255), 12.5f, gp::FontStyleBold);
            EndPaint(h, &ps); return 0;
        }
        case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkColor(dc, C_PANEL); SetTextColor(dc, C_INK); return (LRESULT)g_br_panel; }
        case WM_DRAWITEM: { DRAWITEMSTRUCT* d = (DRAWITEMSTRUCT*)l; draw_button(d, d->CtlID == ID_SETUP_RUN ? gcol(34, 197, 94) : gcol(22, 50, 92)); return TRUE; }
        case WM_COMMAND:
            switch (LOWORD(w)) {
                case ID_SETUP_KIND: if (HIWORD(w) == CBN_SELCHANGE) { int i = (int)SendMessageA(g_setup_kindcb, CB_GETCURSEL, 0, 0); if (i >= 0) setup_build(SETUP_KINDS[i]); } break;
                case ID_SETUP_RUN: setup_run(); break;
                case ID_SETUP_DEFAULTS: DeleteFileA(setup_store(g_setup_kind).c_str()); setup_build(g_setup_kind); break;
                case ID_SETUP_SAVE: {
                    js::Value v; if (!setup_collect(v)) break;
                    std::string p = "setup_" + g_setup_kind + ".json";
                    if (ask_path(true, "Setup (*.json)\0*.json\0", "json", p)) { v["kind"] = g_setup_kind; js::write_file(p, js::dump(v, 1)); status("setup saved: " + p); }
                    break;
                }
                case ID_SETUP_LOAD: {
                    std::string p;
                    if (!ask_path(false, "Setup (*.json)\0*.json\0", "json", p)) break;
                    try {
                        js::Value v = js::parse(js::read_file(p));
                        if (v.str("kind", g_setup_kind) != g_setup_kind) setup_build(v.str("kind", g_setup_kind));
                        int n = 0; for (auto& r : g_srows) if (const js::Value* x = v.find(r.key)) { srow_set(r, *x); ++n; }
                        status("setup loaded: " + tostr((long)n) + " values from " + p);
                    } catch (std::exception& e) { MessageBoxA(h, e.what(), "Load setup", MB_ICONERROR); }
                    break;
                }
                case ID_SETUP_CLOSE: ShowWindow(h, SW_HIDE); InvalidateRect(g_view, nullptr, FALSE); break;
            }
            return 0;
        case WM_CLOSE: ShowWindow(h, SW_HIDE); InvalidateRect(g_view, nullptr, FALSE); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}
// open the form of an analysis; preset fills the flight condition of a 3D flow case
static void open_setup(const std::string& kind, const std::string& preset = "") {
    if (!schema_load()) return;
    if (!g_setup) g_setup = CreateWindowExA(WS_EX_TOOLWINDOW, "AeroSynapseSetup", "Setup", WS_OVERLAPPEDWINDOW & ~WS_MINIMIZEBOX, CW_USEDEFAULT, CW_USEDEFAULT, 900, 680, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
    setup_layout();
    setup_build(kind);
    if (!preset.empty()) {
        for (auto& r : g_srows) if (r.key == "preset") { srow_set(r, js::Value(preset)); setup_apply_preset(preset); }
        for (auto& r : g_srows) if (r.key == "quality" && !g_cfd_quality.empty()) srow_set(r, js::Value(g_cfd_quality));
    }
    if (kind == "turbulence") {  // the last 3D flow run is the natural source
        std::string src = !g_last_cfd_dir.empty() ? g_last_cfd_dir : g_field_dir;
        for (auto& r : g_srows) if (r.key == "flow_folder" && read_edit(r.ctl).empty() && !src.empty()) SetWindowTextA(r.ctl, src.c_str());
    }
    if (kind == "structure") { double W0 = g_doc.num("W0_kg", 0); if (W0 > 0) for (auto& r : g_srows) if (r.key == "W0_kg") srow_set(r, js::Value(W0)); }
    if (kind == "composites") { std::string lay = g_doc.str("layup", ""); if (!lay.empty()) for (auto& r : g_srows) if (r.key == "layup") srow_set(r, js::Value(lay)); }
    ShowWindow(g_setup, SW_SHOW); SetForegroundWindow(g_setup);
    InvalidateRect(g_view, nullptr, FALSE);  // the setup's preview in the 3D view
}

// ---------------------------------------------------------------- preferences window
static HWND g_prefs = nullptr, g_pref_cb[3];
static LRESULT CALLBACK prefs_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            HINSTANCE hi = GetModuleHandleA(nullptr);
            const char* labels[3] = {"Charts and reports", "3D display", "Tool bar"};
            const char* opts[3][2] = {{"Plain (white paper, thin lines)", "Dark (bright colours)"}, {"Calm (no glow or pulses)", "Vivid (glowing pulses)"},
                                      {"Fixed (click the workbench to slide it)", "Auto-hide (slides out under the mouse)"}};
            for (int i = 0; i < 3; ++i) {
                HWND lb = CreateWindowExA(0, "STATIC", labels[i], WS_CHILD | WS_VISIBLE | SS_RIGHT, 10, 58 + i * 40, 140, 20, h, nullptr, hi, nullptr);
                g_pref_cb[i] = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST, 160, 54 + i * 40, 300, 200, h, (HMENU)(INT_PTR)(ID_PREF_PLOT + i), hi, nullptr);
                for (int j = 0; j < 2; ++j) SendMessageA(g_pref_cb[i], CB_ADDSTRING, 0, (LPARAM)opts[i][j]);
                SendMessageA(lb, WM_SETFONT, (WPARAM)g_font, TRUE); SendMessageA(g_pref_cb[i], WM_SETFONT, (WPARAM)g_font, TRUE);
            }
            HWND b = CreateWindowExA(0, "BUTTON", "Save preferences", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 160, 186, 180, 32, h, (HMENU)ID_PREF_OK, hi, nullptr);
            SendMessageA(b, WM_SETFONT, (WPARAM)g_font, TRUE);
            HWND note = CreateWindowExA(0, "STATIC", "Charts use the new style from the next analysis you run.", WS_CHILD | WS_VISIBLE, 20, 232, 440, 20, h, nullptr, hi, nullptr);
            SendMessageA(note, WM_SETFONT, (WPARAM)g_font, TRUE);
            return 0;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps);
            gp::Graphics g(dc); quality(g);
            gp::LinearGradientBrush bg(gp::Rect(0, 0, 490, 38), gcol(100, 116, 139), gcol(22, 50, 92), gp::LinearGradientModeHorizontal);
            g.FillRectangle(&bg, 0, 0, 490, 38);
            gtext(g, "Preferences", gp::RectF(10, 0, 460, 38), gcol(255, 255, 255), 13.0f, gp::FontStyleBold);
            EndPaint(h, &ps); return 0;
        }
        case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkColor(dc, C_PANEL); SetTextColor(dc, C_INK); return (LRESULT)g_br_panel; }
        case WM_DRAWITEM: draw_button((DRAWITEMSTRUCT*)l, gcol(34, 197, 94)); return TRUE;
        case WM_COMMAND:
            if (LOWORD(w) == ID_PREF_OK) {
                g_pref_plot = SendMessageA(g_pref_cb[0], CB_GETCURSEL, 0, 0) == 1 ? "dark" : "plain";
                g_pref_display = SendMessageA(g_pref_cb[1], CB_GETCURSEL, 0, 0) == 1 ? "vivid" : "calm";
                g_pref_palette = SendMessageA(g_pref_cb[2], CB_GETCURSEL, 0, 0) == 1 ? "autohide" : "fixed";
                prefs_save();
                g_pal_open = false; palette_target(); layout();
                InvalidateRect(g_view, nullptr, FALSE);
                ShowWindow(h, SW_HIDE);
                status("preferences saved: charts " + g_pref_plot + ", 3D display " + g_pref_display + ", tool bar " + g_pref_palette);
            }
            return 0;
        case WM_CLOSE: ShowWindow(h, SW_HIDE); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}
static void open_prefs() {
    if (!g_prefs) g_prefs = CreateWindowExA(WS_EX_TOOLWINDOW, "AeroSynapsePrefs", "Preferences", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT, 500, 300, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
    SendMessageA(g_pref_cb[0], CB_SETCURSEL, g_pref_plot == "dark" ? 1 : 0, 0);
    SendMessageA(g_pref_cb[1], CB_SETCURSEL, g_pref_display == "vivid" ? 1 : 0, 0);
    SendMessageA(g_pref_cb[2], CB_SETCURSEL, g_pref_palette == "autohide" ? 1 : 0, 0);
    ShowWindow(g_prefs, SW_SHOW); SetForegroundWindow(g_prefs);
}

static void register_setup_classes(HINSTANCE hi, HICON icon) {
    WNDCLASSA a; memset(&a, 0, sizeof a);
    a.lpfnWndProc = setup_proc; a.hInstance = hi; a.hCursor = LoadCursor(nullptr, IDC_ARROW); a.hbrBackground = g_br_panel; a.lpszClassName = "AeroSynapseSetup"; a.hIcon = icon;
    RegisterClassA(&a);
    a.lpfnWndProc = setup_page_proc; a.lpszClassName = "AeroSynapseSetupPage";
    RegisterClassA(&a);
    a.lpfnWndProc = prefs_proc; a.lpszClassName = "AeroSynapsePrefs";
    RegisterClassA(&a);
}
