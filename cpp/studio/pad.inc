// The feature pad: every tool opens a pad with all the parameters of its feature. Every change is previewed live in the
// 3D view; OK keeps it, Cancel puts the design back as it was before the tool. A hole has its own section: diameter,
// depth or through-all, the entry point and the direction (any axis or any vector), and a point picked on the model.
// "Place at picked point" moves any feature to a point picked on the model.

struct PadRow { std::string key, kind, label, unit; HWND lbl = nullptr, ctl = nullptr, ul = nullptr; bool added = false; std::string def; std::vector<std::string> choices; };
static HWND g_pad = nullptr, g_pad_page = nullptr, g_pad_help = nullptr, g_pad_msg = nullptr;
static std::vector<PadRow> g_prows;
static std::vector<HWND> g_pgroups;
static int g_pad_idx = -1, g_pad_scroll = 0, g_pad_h = 0, g_pick_mode = 0;  // pick: 1 hole entry point, 2 place the feature
static std::string g_pad_snapshot;  // the whole design before the tool: Cancel restores it
static bool g_pad_hole = false, g_pad_building = false;
static double g_mv[16], g_pj[16];
static int g_vp[4];
enum { ID_PAD_OK = 9500, ID_PAD_CANCEL, ID_PAD_PICK, ID_PAD_ROW0 = 9520 };
static const char* HOLE_DIRS[] = {"-z (down)", "+z (up)", "-y", "+y", "-x (forward)", "+x (aft)", "custom vector"};

static std::string fmt3(const std::vector<double>& v) { std::string s; for (size_t i = 0; i < v.size(); ++i) { if (i) s += ", "; s += fmtd(v[i]); } return s; }
static bool parse_nums(const std::string& t, std::vector<double>& out) {
    out.clear(); std::string cur;
    for (size_t i = 0; i <= t.size(); ++i) {
        char c = i < t.size() ? t[i] : ',';
        if (c == ',' || c == ';' || c == ' ' || c == '[' || c == ']') { if (!cur.empty()) { char* e; double v = strtod(cur.c_str(), &e); if (*e) return false; out.push_back(v); cur.clear(); } }
        else cur += c;
    }
    return true;
}
static std::string prow_text(const PadRow& r) {
    if (r.kind == "bool") return SendMessageA(r.ctl, BM_GETCHECK, 0, 0) == BST_CHECKED ? "true" : "false";
    if (r.kind == "choice") { char b[128] = ""; int i = (int)SendMessageA(r.ctl, CB_GETCURSEL, 0, 0); if (i >= 0) SendMessageA(r.ctl, CB_GETLBTEXT, i, (LPARAM)b); return b; }
    return read_edit(r.ctl);
}
static void prow_set(PadRow& r, const std::string& t) {
    if (r.kind == "bool") SendMessageA(r.ctl, BM_SETCHECK, t == "true" ? BST_CHECKED : BST_UNCHECKED, 0);
    else if (r.kind == "choice") { for (size_t i = 0; i < r.choices.size(); ++i) if (r.choices[i] == t) SendMessageA(r.ctl, CB_SETCURSEL, i, 0); }
    else SetWindowTextA(r.ctl, t.c_str());
}
static PadRow* prow(const std::string& k) { for (auto& r : g_prows) if (r.key == k) return &r; return nullptr; }

// ---------------------------------------------------------------- picking a point on the model
static bool ray_mesh(const V3& o, const V3& d, V3& hit, V3& nrm) {  // the first surface along o + t d
    if (g_mesh.t.empty()) return false;
    double best = 1e300; bool any = false;
    for (size_t q = 0; q + 2 < g_mesh.t.size(); q += 3) {  // Moller-Trumbore against every triangle
        const V3 &a = g_mesh.v[(size_t)g_mesh.t[q]], &b = g_mesh.v[(size_t)g_mesh.t[q + 1]], &c = g_mesh.v[(size_t)g_mesh.t[q + 2]];
        V3 e1 = b - a, e2 = c - a, p = geo::cross(d, e2);
        double det = geo::dot(e1, p);
        if (std::fabs(det) < 1e-18) continue;
        double inv = 1 / det; V3 s = o - a;
        double u = geo::dot(s, p) * inv; if (u < 0 || u > 1) continue;
        V3 qv = geo::cross(s, e1);
        double v = geo::dot(d, qv) * inv; if (v < 0 || u + v > 1) continue;
        double t = geo::dot(e2, qv) * inv;
        if (t > 0 && t < best) { best = t; hit = o + d * t; nrm = geo::cross(e1, e2); any = true; }
    }
    if (!any) return false;
    double l = geo::len(nrm); nrm = l > 0 ? nrm / l : V3(0, 0, 1);
    if (geo::dot(nrm, d) > 0) nrm = nrm * -1;  // the face that looks at the camera: outward
    return true;
}
static bool pick_ray(int mx, int my, V3& hit, V3& nrm) {
    double x0, y0, z0, x1, y1, z1;
    double wy = g_vp[3] - my;
    if (!gluUnProject(mx, wy, 0.0, g_mv, g_pj, g_vp, &x0, &y0, &z0) || !gluUnProject(mx, wy, 1.0, g_mv, g_pj, g_vp, &x1, &y1, &z1)) return false;
    return ray_mesh(V3(x0, y0, z0), V3(x1 - x0, y1 - y0, z1 - z0), hit, nrm);
}

// ---------------------------------------------------------------- hole: its own parameters, compiled to a cylinder cut
static js::Value hole_feature(const js::Value& spec, const std::string& id) {
    std::vector<double> at = spec.nums("at"), dir = spec.nums("dir");
    if (at.size() != 3) at = {0, 0, 0};
    if (dir.size() != 3) dir = {0, 0, -1};
    V3 d(dir[0], dir[1], dir[2]); double l = geo::len(d); d = l > 1e-12 ? d / l : V3(0, 0, -1);
    double D = std::max(1e-4, spec.num("diameter", 0.1)), depth = std::max(1e-4, spec.num("depth", 0.5));
    V3 a(at[0], at[1], at[2]);
    if (spec.flag("through", false)) {  // through all: to where the drill line leaves the model's box
        double tex = 1e300;  // the drill line leaves the box through the first face it reaches
        for (int k = 0; k < 3; ++k) if (std::fabs(d[k]) > 1e-12) tex = std::min(tex, std::max(0.0, ((d[k] > 0 ? g_props.hi[k] : g_props.lo[k]) - a[k]) / d[k]));
        if (tex > 1e299) tex = geo::len(g_props.hi - g_props.lo);
        depth = std::max(0.05, tex + 0.05);
    }
    double e = std::max(0.01, 0.5 * D);  // start a little outside the surface, so no skin is left over the hole
    V3 c = a + d * (0.5 * (depth - e));
    double th = std::acos(geo::clampd(d.z, -1, 1)) * 180 / geo::PI, ph = std::atan2(d.y, d.x) * 180 / geo::PI;
    js::Value f = js::Value::object();
    f["id"] = id; f["type"] = "cylinder"; f["op"] = "cut"; f["axis"] = "z"; f["r"] = 0.5 * D; f["length"] = depth + e;
    js::Value t = js::Value::array(); t.push(c.x); t.push(c.y); t.push(c.z); f["translate"] = t;
    js::Value r = js::Value::array(); r.push(0.0); r.push(th); r.push(ph); f["rotate"] = r;
    f["hole"] = spec.clone();
    return f;
}
static js::Value default_hole_spec() {
    js::Value s = js::Value::object();
    V3 c = (g_props.lo + g_props.hi) * 0.5, top(c.x, 0.0, g_mesh.v.empty() ? 0.5 : g_props.hi.z), n;
    V3 hit;
    if (ray_mesh(V3(c.x, 0.0, g_props.hi.z + 1.0), V3(0, 0, -1), hit, n)) top = hit;  // drop onto the top surface below the middle
    js::Value at = js::Value::array(); at.push(top.x); at.push(top.y); at.push(top.z); s["at"] = at;
    js::Value d = js::Value::array(); d.push(0.0); d.push(0.0); d.push(-1.0); s["dir"] = d;
    s["diameter"] = 0.1; s["depth"] = 0.5; s["through"] = true;
    return s;
}

// ---------------------------------------------------------------- the pad window
static void pad_scroll_to(int y) {
    RECT rc; GetClientRect(g_pad_page, &rc);
    y = std::max(0, std::min(std::max(0, g_pad_h - (int)rc.bottom), y));
    int dy = g_pad_scroll - y; g_pad_scroll = y;
    if (dy) ScrollWindowEx(g_pad_page, 0, dy, nullptr, nullptr, nullptr, nullptr, SW_SCROLLCHILDREN | SW_INVALIDATE | SW_ERASE);
    SCROLLINFO si; si.cbSize = sizeof si; si.fMask = SIF_ALL; si.nMin = 0; si.nMax = std::max(0, g_pad_h - 1); si.nPage = (UINT)rc.bottom; si.nPos = g_pad_scroll; si.nTrackPos = 0;
    SetScrollInfo(g_pad_page, SB_VERT, &si, TRUE);
}
static void pad_clear() {
    for (auto& r : g_prows) { DestroyWindow(r.lbl); DestroyWindow(r.ctl); if (r.ul) DestroyWindow(r.ul); }
    for (HWND gh : g_pgroups) DestroyWindow(gh);
    g_prows.clear(); g_pgroups.clear(); g_pad_scroll = 0;
}
static int g_pad_y = 0;
static void pad_group(const std::string& t) {
    RECT rc; GetClientRect(g_pad_page, &rc);
    g_pad_y += 4;
    HWND gh = CreateWindowExA(0, "STATIC", t.c_str(), WS_CHILD | WS_VISIBLE, 8, g_pad_y, rc.right - 16, 20, g_pad_page, nullptr, GetModuleHandleA(nullptr), nullptr);
    SendMessageA(gh, WM_SETFONT, (WPARAM)g_font_bold, TRUE); g_pgroups.push_back(gh); g_pad_y += 22;
}
static void pad_row(const std::string& key, const std::string& kind, const std::string& label, const std::string& unit, const std::string& value,
                    const std::vector<std::string>& choices = {}, bool added = false) {
    RECT rc; GetClientRect(g_pad_page, &rc);
    HINSTANCE hi = GetModuleHandleA(nullptr);
    int lw = 128, cw = std::max(120, (int)rc.right - lw - 70), id = ID_PAD_ROW0 + (int)g_prows.size(), hgt = kind == "json" ? 64 : 22;
    PadRow r; r.key = key; r.kind = kind; r.label = label; r.unit = unit; r.added = added; r.def = value; r.choices = choices;
    r.lbl = CreateWindowExA(0, "STATIC", label.c_str(), WS_CHILD | WS_VISIBLE | SS_RIGHT, 4, g_pad_y + 3, lw - 8, 20, g_pad_page, nullptr, hi, nullptr);
    if (kind == "bool") r.ctl = CreateWindowExA(0, "BUTTON", "", WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX, lw, g_pad_y + 2, 22, 20, g_pad_page, (HMENU)(INT_PTR)id, hi, nullptr);
    else if (kind == "choice") {
        r.ctl = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST, lw, g_pad_y, cw, 260, g_pad_page, (HMENU)(INT_PTR)id, hi, nullptr);
        for (auto& c : choices) SendMessageA(r.ctl, CB_ADDSTRING, 0, (LPARAM)c.c_str());
    } else r.ctl = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL | (kind == "json" ? ES_MULTILINE | ES_AUTOVSCROLL | WS_VSCROLL : 0),
                                   lw, g_pad_y, cw, hgt, g_pad_page, (HMENU)(INT_PTR)id, hi, nullptr);
    if (!unit.empty()) r.ul = CreateWindowExA(0, "STATIC", unit.c_str(), WS_CHILD | WS_VISIBLE, lw + cw + 4, g_pad_y + 3, 60, 20, g_pad_page, nullptr, hi, nullptr);
    HWND hs[] = {r.lbl, r.ctl, r.ul};
    for (HWND x : hs) if (x) SendMessageA(x, WM_SETFONT, (WPARAM)g_font, TRUE);
    prow_set(r, value);
    g_prows.push_back(r);
    g_pad_y += hgt + 5;
}
static std::string pad_kind(const js::Value& v, const ParamHelp* ph) {
    if (v.kind == js::Value::Bool) return "bool";
    if (v.kind == js::Value::Num) return "num";
    if (v.kind == js::Value::Str) return ph && ph->choices && std::string(ph->choices) != "true,false" ? "choice" : "text";
    if (v.kind == js::Value::Arr) { bool flat = true; for (size_t i = 0; i < v.size(); ++i) if (v.at(i).kind != js::Value::Num) flat = false; if (flat && v.size() <= 4) return "vec"; }
    return "json";
}
static void pad_build() {
    pad_clear(); g_pad_y = 6;
    const js::Value& o = features().at((size_t)g_pad_idx);
    g_pad_hole = o.find("hole") != nullptr;
    if (g_pad_hole) {
        const js::Value& h = *o.find("hole");
        std::vector<double> dir = h.nums("dir"); if (dir.size() != 3) dir = {0, 0, -1};
        const double ax[6][3] = {{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}};
        int dsel = 6;
        for (int k = 0; k < 6; ++k) if (std::fabs(dir[0] - ax[k][0]) + std::fabs(dir[1] - ax[k][1]) + std::fabs(dir[2] - ax[k][2]) < 1e-9) dsel = k;
        std::vector<std::string> dirs(HOLE_DIRS, HOLE_DIRS + 7);
        pad_group("Hole");
        pad_row("h_diameter", "num", "Diameter", "m", fmtd(h.num("diameter", 0.1)));
        pad_row("h_through", "bool", "Through all", "", h.flag("through", false) ? "true" : "false");
        pad_row("h_depth", "num", "Depth", "m", fmtd(h.num("depth", 0.5)));
        pad_group("Position and direction");
        pad_row("h_at", "vec", "Entry point x, y, z", "m", fmt3(h.nums("at")));
        pad_row("h_dir", "choice", "Direction", "", dirs[(size_t)dsel], dirs);
        pad_row("h_vec", "vec", "Custom direction", "", fmt3(dir));
        pad_group("Join");
        pad_row("id", "text", "Name", "", o.str("id", "hole"));
        pad_row("enabled", "bool", "On", "", o.flag("enabled", true) ? "true" : "false");
        pad_row("chamfer", "num", "Edge chamfer", "m", fmtd(o.num("chamfer", 0)), {}, !o.find("chamfer"));
    } else {
        static const char* common[] = {"id", "type", "op", "enabled", "translate", "rotate", "scale", "pivot", "mirror", "blend", "chamfer", "shell", "draft"};
        auto is_common = [](const std::string& k) { for (const char* c : common) if (k == c) return true; return false; };
        pad_group("Shape: " + o.str("type", "feature"));
        for (auto& kv : *o.o) {
            if (is_common(kv.first)) continue;
            const ParamHelp* ph = param_help(kv.first);
            std::string kind = pad_kind(kv.second, ph);
            pad_row(kv.first, kind, kv.first, ph ? ph->unit : "", value_text(kv.second), kind == "choice" ? split_csv(ph->choices) : std::vector<std::string>());
        }
        pad_group("Position");
        auto vec_or = [&](const char* k, const char* def) { const js::Value* v = o.find(k); return v ? value_text(*v) : std::string(def); };
        pad_row("translate", "vec", "Move x, y, z", "m", vec_or("translate", "0, 0, 0"), {}, !o.find("translate"));
        pad_row("rotate", "vec", "Rotate x, y, z", "deg", vec_or("rotate", "0, 0, 0"), {}, !o.find("rotate"));
        pad_row("scale", "num", "Scale", "", vec_or("scale", "1"), {}, !o.find("scale"));
        pad_row("mirror", "bool", "Mirror (y = 0)", "", o.flag("mirror", false) ? "true" : "false", {}, !o.find("mirror"));
        if (o.find("pivot")) pad_row("pivot", "vec", "Pivot x, y, z", "m", vec_or("pivot", "0, 0, 0"));
        pad_group("Join");
        pad_row("id", "text", "Name", "", o.str("id", ""));
        pad_row("op", "choice", "Operation", "", o.str("op", "add"), {"add", "cut", "intersect"}, !o.find("op"));
        pad_row("enabled", "bool", "On", "", o.flag("enabled", true) ? "true" : "false", {}, !o.find("enabled"));
        pad_row("blend", "num", "Fillet radius", "m", fmtd(o.num("blend", 0)), {}, !o.find("blend"));
        pad_row("chamfer", "num", "Chamfer", "m", fmtd(o.num("chamfer", 0)), {}, !o.find("chamfer"));
        pad_row("shell", "num", "Shell wall", "m", fmtd(o.num("shell", 0)), {}, !o.find("shell"));
        if (o.str("type", "") == "extrude") pad_row("draft", "num", "Draft angle", "deg", fmtd(o.num("draft", 0)), {}, !o.find("draft"));
    }
    g_pad_h = g_pad_y + 8;
    pad_scroll_to(0);
    SetWindowTextA(g_pad, ("Pad: " + o.str("id", "") + "  (" + o.str("type", "") + ")").c_str());
    SetWindowTextA(GetDlgItem(g_pad, ID_PAD_PICK), g_pad_hole ? "Pick entry point on model" : "Place at picked point");
    SetWindowTextA(g_pad_msg, "");
}
// the feature as the pad describes it; false (with a message) when a field cannot be read
static bool pad_collect(js::Value& out, std::string& err) {
    const js::Value& cur = features().at((size_t)g_pad_idx);
    js::Value o = cur.clone();
    if (g_pad_hole) {
        js::Value spec = cur.find("hole") ? cur.find("hole")->clone() : default_hole_spec();
        std::vector<double> v;
        char* e; std::string t = prow_text(*prow("h_diameter")); double D = strtod(t.c_str(), &e);
        if (*e || t.empty() || D <= 0) { err = "Diameter: a positive number"; return false; }
        t = prow_text(*prow("h_depth")); double dp = strtod(t.c_str(), &e);
        if (*e || t.empty() || dp <= 0) { err = "Depth: a positive number"; return false; }
        if (!parse_nums(prow_text(*prow("h_at")), v) || v.size() != 3) { err = "Entry point: three numbers x, y, z"; return false; }
        js::Value at = js::Value::array(); for (double x : v) at.push(x);
        std::string ds = prow_text(*prow("h_dir"));
        const double ax[6][3] = {{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}};
        std::vector<double> dv;
        for (int k = 0; k < 6; ++k) if (ds == HOLE_DIRS[k]) dv = {ax[k][0], ax[k][1], ax[k][2]};
        if (dv.empty()) { if (!parse_nums(prow_text(*prow("h_vec")), dv) || dv.size() != 3 || std::fabs(dv[0]) + std::fabs(dv[1]) + std::fabs(dv[2]) < 1e-12) { err = "Custom direction: three numbers, not all zero"; return false; } }
        js::Value dir = js::Value::array(); for (double x : dv) dir.push(x);
        spec["diameter"] = D; spec["depth"] = dp; spec["through"] = prow_text(*prow("h_through")) == "true"; spec["at"] = at; spec["dir"] = dir;
        o = hole_feature(spec, prow_text(*prow("id")));
        o["enabled"] = prow_text(*prow("enabled")) == "true";
        t = prow_text(*prow("chamfer")); double ch = strtod(t.c_str(), &e);
        if (*e) { err = "Edge chamfer: a number"; return false; }
        if (ch > 0) o["chamfer"] = ch;
        out = o; return true;
    }
    for (auto& r : g_prows) {
        std::string t = prow_text(r);
        if (r.added && t == r.def) continue;  // a default the user did not touch: keep the feature file short
        try {
            if (r.kind == "num") { char* e; double v = strtod(t.c_str(), &e); if (t.empty() || *e) throw std::runtime_error(r.label + ": a number"); o[r.key] = v; }
            else if (r.kind == "bool") o[r.key] = t == "true";
            else if (r.kind == "text" || r.kind == "choice") o[r.key] = t;
            else if (r.kind == "vec") { std::vector<double> v; if (!parse_nums(t, v) || v.empty()) throw std::runtime_error(r.label + ": numbers separated by commas"); js::Value a = js::Value::array(); for (double x : v) a.push(x); o[r.key] = a; }
            else o[r.key] = js::parse(t);
        } catch (std::exception& ex) { err = std::string(ex.what()).find(':') != std::string::npos ? ex.what() : r.label + ": " + ex.what(); return false; }
    }
    out = o; return true;
}
static void pad_preview() {  // show the change in the 3D view now (it is not final until OK)
    if (g_pad_idx < 0 || g_pad_idx >= (int)features().size()) return;
    js::Value o; std::string err;
    if (!pad_collect(o, err)) { SetWindowTextA(g_pad_msg, ("Not previewed: " + err).c_str()); return; }
    features().at((size_t)g_pad_idx) = o;
    g_sel = g_pad_idx;
    g_pad_building = true;
    fill_list(); fill_props(); rebuild(); compute_section();
    g_pad_building = false;
    SetWindowTextA(g_pad_msg, "Preview: the 3D view shows the change. OK keeps it, Cancel undoes it.");
}
static void pad_close(bool keep) {
    KillTimer(g_pad, 1);
    g_pick_mode = 0;
    if (keep) { pad_preview(); g_modified = true; status("applied: " + features().at((size_t)std::max(0, g_pad_idx)).str("id", "")); }
    else if (!g_pad_snapshot.empty()) {
        g_doc = js::parse(g_pad_snapshot);
        if (!g_undo.empty() && g_undo.back() == g_pad_snapshot) g_undo.pop_back();
        if (g_sel >= (int)features().size()) g_sel = (int)features().size() - 1;
        fill_list(); fill_props(); rebuild(); compute_section();
        status("cancelled: the design is as it was");
    }
    g_pad_snapshot.clear(); g_pad_idx = -1;
    update_title();
    ShowWindow(g_pad, SW_HIDE);
}
static void pad_pick_result(const V3& p, const V3& n) {
    if (!g_pad || g_pad_idx < 0) return;
    if (g_pick_mode == 1 && g_pad_hole) {
        prow_set(*prow("h_at"), fmt3({p.x, p.y, p.z}));
        prow_set(*prow("h_vec"), fmt3({-n.x, -n.y, -n.z}));
        prow_set(*prow("h_dir"), HOLE_DIRS[6]);
        status("hole entry point picked; it is drilled into the surface (custom direction = the surface normal, inwards)");
    } else if (PadRow* r = prow("translate")) {
        prow_set(*r, fmt3({p.x, p.y, p.z}));
        status("the feature's origin is placed at the picked point");
    }
    g_pick_mode = 0;
    pad_preview();
}
static LRESULT CALLBACK pad_page_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_VSCROLL: {
            SCROLLINFO si; si.cbSize = sizeof si; si.fMask = SIF_ALL; GetScrollInfo(h, SB_VERT, &si);
            int y = g_pad_scroll;
            switch (LOWORD(w)) { case SB_LINEUP: y -= 27; break; case SB_LINEDOWN: y += 27; break; case SB_PAGEUP: y -= (int)si.nPage; break; case SB_PAGEDOWN: y += (int)si.nPage; break;
                                 case SB_THUMBTRACK: case SB_THUMBPOSITION: y = si.nTrackPos; break; }
            pad_scroll_to(y); return 0;
        }
        case WM_MOUSEWHEEL: pad_scroll_to(g_pad_scroll - (short)HIWORD(w) / 2); return 0;
        case WM_CTLCOLORSTATIC: {
            HDC dc = (HDC)w; SetBkColor(dc, C_PANEL);
            bool grp = std::find(g_pgroups.begin(), g_pgroups.end(), (HWND)l) != g_pgroups.end();
            SetTextColor(dc, grp ? C_NAVY : C_INK); return (LRESULT)g_br_panel;
        }
        case WM_CTLCOLOREDIT: case WM_CTLCOLORLISTBOX: { HDC dc = (HDC)w; SetBkColor(dc, C_FIELD); SetTextColor(dc, C_INK); return (LRESULT)g_br_field; }
        case WM_COMMAND: {
            int i = LOWORD(w) - ID_PAD_ROW0, code = HIWORD(w);
            if (i < 0 || i >= (int)g_prows.size()) return 0;
            const PadRow& r = g_prows[(size_t)i];
            if (code == EN_SETFOCUS || code == CBN_SETFOCUS || code == BN_SETFOCUS) {
                const ParamHelp* ph = param_help(r.key);
                std::string help = r.key == "h_diameter" ? "Hole diameter." : r.key == "h_depth" ? "Depth from the entry point along the direction (ignored for through-all)." :
                                   r.key == "h_through" ? "Through all: the hole goes right through the model." : r.key == "h_at" ? "Where the drill enters: pick it on the model or type it." :
                                   r.key == "h_dir" ? "Drilling direction: any axis, or a custom vector (picking fills it with the surface normal)." : r.key == "h_vec" ? "Drilling direction as a vector dx, dy, dz (any length)." :
                                   ph ? std::string(ph->what) + (ph->unit[0] ? std::string("  [") + ph->unit + "]" : "") + (ph->choices ? std::string("  choices: ") + ph->choices : "") : "A parameter of this feature.";
                SetWindowTextA(g_pad_help, (r.label + ": " + help).c_str());
            }
            if (code == EN_CHANGE || code == CBN_SELCHANGE || code == BN_CLICKED) SetTimer(g_pad, 1, 450, nullptr);  // preview shortly after typing stops
            return 0;
        }
    }
    return DefWindowProcA(h, m, w, l);
}
static void pad_layout() {
    RECT rc; GetClientRect(g_pad, &rc);
    int W = rc.right, H = rc.bottom;
    MoveWindow(g_pad_page, 0, 34, W, H - 34 - 128, TRUE);
    MoveWindow(g_pad_help, 8, H - 124, W - 16, 40, TRUE);
    MoveWindow(g_pad_msg, 8, H - 82, W - 16, 34, TRUE);
    MoveWindow(GetDlgItem(g_pad, ID_PAD_PICK), 8, H - 44, W - 16 - 196, 34, TRUE);
    MoveWindow(GetDlgItem(g_pad, ID_PAD_OK), W - 184, H - 44, 84, 34, TRUE);
    MoveWindow(GetDlgItem(g_pad, ID_PAD_CANCEL), W - 94, H - 44, 86, 34, TRUE);
}
static LRESULT CALLBACK pad_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            HINSTANCE hi = GetModuleHandleA(nullptr);
            if (!g_font_bold) g_font_bold = CreateFontA(-14, 0, 0, 0, FW_BOLD, 0, 0, 0, ANSI_CHARSET, 0, 0, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
            g_pad_page = CreateWindowExA(0, "AeroSynapsePadPage", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_CLIPCHILDREN, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_pad_help = CreateWindowExA(0, "STATIC", "Click a field to see what it means. Changes show in the 3D view as you type.", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_pad_msg = CreateWindowExA(0, "STATIC", "", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            const char* names[] = {"OK", "Cancel", "Place at picked point"};
            int ids[] = {ID_PAD_OK, ID_PAD_CANCEL, ID_PAD_PICK};
            for (int i = 0; i < 3; ++i) { HWND b = CreateWindowExA(0, "BUTTON", names[i], WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)(INT_PTR)ids[i], hi, nullptr); SendMessageA(b, WM_SETFONT, (WPARAM)g_font, TRUE); }
            SendMessageA(g_pad_help, WM_SETFONT, (WPARAM)g_font, TRUE); SendMessageA(g_pad_msg, WM_SETFONT, (WPARAM)g_font, TRUE);
            return 0;
        }
        case WM_SIZE: pad_layout(); return 0;
        case WM_PAINT: {
            PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps);
            RECT rc; GetClientRect(h, &rc);
            gp::Graphics g(dc); quality(g);
            gp::SolidBrush bg(gcol(22, 50, 92)); g.FillRectangle(&bg, 0, 0, (int)rc.right, 34);
            std::string t = g_pad_idx >= 0 && g_pad_idx < (int)features().size() ? features().at((size_t)g_pad_idx).str("id", "") : std::string();
            gtext(g, (g_pad_hole ? "Hole: " : "Feature pad: ") + t, gp::RectF(10, 0, (float)rc.right - 20, 34), gcol(255, 255, 255), 12.5f, gp::FontStyleBold);
            EndPaint(h, &ps); return 0;
        }
        case WM_CTLCOLORSTATIC: {
            HDC dc = (HDC)w; SetBkColor(dc, C_PANEL);
            SetTextColor(dc, (HWND)l == g_pad_msg ? RGB(150, 60, 20) : C_INK); return (LRESULT)g_br_panel;
        }
        case WM_DRAWITEM: { DRAWITEMSTRUCT* d = (DRAWITEMSTRUCT*)l; draw_button(d, d->CtlID == ID_PAD_OK ? gcol(34, 197, 94) : d->CtlID == ID_PAD_PICK ? gcol(14, 116, 144) : gcol(22, 50, 92)); return TRUE; }
        case WM_TIMER: if (w == 1) { KillTimer(h, 1); pad_preview(); } return 0;
        case WM_COMMAND:
            switch (LOWORD(w)) {
                case ID_PAD_OK: pad_close(true); break;
                case ID_PAD_CANCEL: pad_close(false); break;
                case ID_PAD_PICK:
                    g_pick_mode = g_pad_hole ? 1 : 2;
                    SetWindowTextA(g_pad_msg, g_pad_hole ? "Click the point on the model where the hole enters." : "Click a point on the model: the feature's origin moves there.");
                    SetForegroundWindow(g_main); SetFocus(g_view);
                    break;
            }
            return 0;
        case WM_CLOSE: pad_close(false); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}
// open the pad on feature idx; the design before the tool (the last undo state) is what Cancel restores
static void open_pad(int idx) {
    if (idx < 0 || idx >= (int)features().size() || g_pad_building) return;
    if (g_pad && IsWindowVisible(g_pad) && g_pad_idx >= 0 && g_pad_idx != idx) { KillTimer(g_pad, 1); pad_preview(); }  // keep the previous pad's edits
    if (!g_pad) {
        RECT vr; GetWindowRect(g_view, &vr);
        g_pad = CreateWindowExA(WS_EX_TOOLWINDOW, "AeroSynapsePad", "Pad", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME, vr.left + 12, vr.top + 12, 430, 600, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
    }
    g_pad_idx = idx;
    g_pad_snapshot = g_undo.empty() ? js::dump(g_doc, 0) : g_undo.back();
    pad_layout();
    pad_build();
    InvalidateRect(g_pad, nullptr, TRUE);
    ShowWindow(g_pad, SW_SHOWNA);
}
static void register_pad_classes(HINSTANCE hi, HICON icon) {
    WNDCLASSA a; memset(&a, 0, sizeof a);
    a.lpfnWndProc = pad_proc; a.hInstance = hi; a.hCursor = LoadCursor(nullptr, IDC_ARROW); a.hbrBackground = g_br_panel; a.lpszClassName = "AeroSynapsePad"; a.hIcon = icon;
    RegisterClassA(&a);
    a.lpfnWndProc = pad_page_proc; a.lpszClassName = "AeroSynapsePadPage";
    RegisterClassA(&a);
}
