// One-click dimensions. Click a part (in the 3D view or the tree) and its driving dimensions appear on it as
// dimension lines with values, as in a CAD program; click a value to type a new one, and the part rebuilds at once.
// Wing-like parts (wing, canard, tail, fin, strake, blade) show semi-span, root and tip chord, leading-edge sweep and
// dihedral on their real edges; primitives show their sizes; anything else shows its overall extents.
// The floating view toolbar over the 3D view (fit, zoom, orientations, display, section, dimensions, colours) is here too.

struct PartDim { std::string key, label, unit; int comp = -1; double value = 0; V3 a, b, at; int kind = 0; };  // kind 0 length, 1 angle, 2 derived (tip chord)
static std::vector<PartDim> g_dims;
static bool g_dims_on = true;  // show the selected part's dimensions
static int g_dims_for = -2, g_dims_gen = -1, g_dim_hover = -1;
struct DimLabel { RECT r; int idx; };
static std::vector<DimLabel> g_dim_labels;
static HWND g_dim_edit = nullptr;
static int g_dim_editing = -1;
static WNDPROC g_dim_edit_old = nullptr;

static V3 vtr(const js::Value& f, const char* k) { std::vector<double> t = f.nums(k); return V3(t.size() > 0 ? t[0] : 0, t.size() > 1 ? t[1] : 0, t.size() > 2 ? t[2] : 0); }
static bool wing_like(const std::string& ty) { return ty == "wing" || ty == "canard" || ty == "tail" || ty == "fin" || ty == "strake" || ty == "blade"; }

static void dims_compute() {
    g_dims.clear();
    const js::Value& fl = features();
    if (g_sel < 0 || g_sel >= (int)fl.size()) return;
    const js::Value& f = fl.at((size_t)g_sel);
    std::string ty = f.str("type", "");
    bool xformed = f.find("rotate") || f.find("scale") || f.find("stretch") || f.find("bend") || f.find("twist_rate");
    auto add = [&](const std::string& key, const std::string& label, const std::string& unit, int comp, double val, V3 a, V3 b, V3 at, int kind = 0) {
        PartDim d; d.key = key; d.label = label; d.unit = unit; d.comp = comp; d.value = val; d.a = a; d.b = b; d.at = at; d.kind = kind; g_dims.push_back(d);
    };
    if (wing_like(ty) && !xformed) {
        V3 t = vtr(f, "translate");
        double b = f.num("span", 1), cr = f.num("root_chord", 1), lam = f.num("taper", 0.5), sw = f.num("sweep", 30) * geo::PI / 180, di = f.num("dihedral", 0) * geo::PI / 180;
        // a mirrored part shows its dimensions on the side that faces the camera
        double sy = f.flag("mirror", false) && std::sin(g_yaw * geo::PI / 180) < 0 ? -1 : 1;
        if (sy < 0) t.y = -t.y;
        V3 rle = t, rte = t + V3(cr, 0, 0);
        V3 tle = t + V3(b * std::tan(sw), sy * b * std::cos(di), b * std::sin(di)), tte = tle + V3(cr * lam, 0, 0);
        V3 off = V3(0, 0, 0.06 * b + 0.05);
        V3 dy(0, sy * 0.05 * b, 0);
        add("root_chord", "root chord", "m", -1, cr, rle - dy, rte - dy, (rle + rte) * 0.5 - dy - off);
        add("taper", "tip chord", "m", -1, cr * lam, tle + dy, tte + dy, (tle + tte) * 0.5 + dy * 3.0 - off * 1.5, 2);
        V3 sa = rte + V3(0.12 * cr, 0, 0), sb = tte + V3(0.12 * cr, 0, 0);
        add("span", "semi-span", "m", -1, b, sa, sb, (sa + sb) * 0.5 + V3(0.1 * cr, 0, 0));
        add("sweep", "LE sweep", "deg", -1, f.num("sweep", 30), rle, tle, rle + (tle - rle) * 0.35 - V3(0.25 * cr, 0, 0), 1);
        if (std::fabs(di) > 1e-6 || ty == "wing") add("dihedral", "dihedral", "deg", -1, f.num("dihedral", 0), tle, tle, tle + V3(0, 0, 0.15 * b + 0.35), 1);
        return;
    }
    // everything else: from the kernel's box of this part
    geo::Model m; js::Value one = js::Value::object(); js::Value arr = js::Value::array();
    js::Value c = f.clone(); c["op"] = "add"; c["enabled"] = true; c["mirror"] = false; arr.push(c); one["features"] = arr;
    try { m.load(one); } catch (...) { return; }
    if (m.feats.empty()) return;
    geo::Box bx = m.boxes[0];
    if (!bx.valid()) return;
    V3 lo = bx.lo, hi = bx.hi, e = hi - lo;
    double o = 0.08 * std::max(e.x, std::max(e.y, e.z)) + 0.05;
    V3 ax(hi.x - lo.x, 0, 0);
    auto len_x = [&](const std::string& key, const std::string& label, int comp, double val) { add(key, label, "m", comp, val, V3(lo.x, lo.y - o, lo.z), V3(hi.x, lo.y - o, lo.z), V3(0.5 * (lo.x + hi.x), lo.y - 1.6 * o, lo.z)); };
    auto len_y = [&](const std::string& key, const std::string& label, int comp, double val) { add(key, label, "m", comp, val, V3(hi.x + o, lo.y, lo.z), V3(hi.x + o, hi.y, lo.z), V3(hi.x + 1.6 * o, 0.5 * (lo.y + hi.y), lo.z)); };
    auto len_z = [&](const std::string& key, const std::string& label, int comp, double val) { add(key, label, "m", comp, val, V3(hi.x + o, lo.y, lo.z), V3(hi.x + o, lo.y, hi.z), V3(hi.x + 1.6 * o, lo.y, 0.5 * (lo.z + hi.z))); };
    (void)ax;
    if (ty == "sphere") { len_x("r", "radius", -1, f.num("r", 0.5)); return; }
    if (ty == "ellipsoid") { std::vector<double> r = f.nums("radii"); if (r.size() == 3) { len_x("radii", "semi-axis x", 0, r[0]); len_y("radii", "semi-axis y", 1, r[1]); len_z("radii", "semi-axis z", 2, r[2]); } return; }
    if (ty == "box") { std::vector<double> s = f.nums("size"); if (s.size() == 3) { len_x("size", "length", 0, s[0]); len_y("size", "width", 1, s[1]); len_z("size", "height", 2, s[2]); } return; }
    if (ty == "cylinder") { len_x("length", "length", -1, f.num("length", 1)); len_z("r", "radius", -1, f.num("r", 0.3)); return; }
    if (ty == "cone") { len_x("length", "length", -1, f.num("length", 1)); len_z("r1", "radius 1", -1, f.num("r1", 0.5)); len_y("r2", "radius 2", -1, f.num("r2", 0.3)); return; }
    if (ty == "torus") { len_x("R", "ring radius", -1, f.num("R", 1)); len_z("r", "tube radius", -1, f.num("r", 0.2)); return; }
    if (ty == "fuselage") { len_x("@length", "length", -1, e.x); len_y("@width", "max width", -1, e.y); len_z("@height", "max height", -1, e.z); return; }
    len_x("@length", "length", -1, e.x); len_y("@width", "width", -1, e.y); len_z("@height", "height", -1, e.z);  // overall: scales the part
}

// a new value for dimension i: written into the feature, then rebuilt (the change shows at once)
static void dims_apply(int i, double v) {
    if (i < 0 || i >= (int)g_dims.size() || g_sel < 0) return;
    const PartDim& d = g_dims[(size_t)i];
    js::Value& f = features().at((size_t)g_sel);
    push_undo();
    if (d.kind == 2) { double cr = f.num("root_chord", 1); f["taper"] = cr > 0 ? std::max(0.01, v / cr) : 0.5; }
    else if (d.key[0] == '@') {  // overall size of a fuselage or another part: scale it about its first point
        double s = d.value > 1e-9 ? v / d.value : 1;
        int ax = d.key == "@length" ? 0 : d.key == "@width" ? 1 : 2;
        if (f.str("type", "") == "fuselage" && f.find("stations")) {
            js::Value& st = f["stations"];
            for (size_t k = 0; k < st.size(); ++k) {
                js::Value& row = st.at(k);
                if (ax == 0 && row.size() > 0) row.at(0) = js::Value(row.at(0).n * s);
                if (ax == 1 && row.size() > 1) row.at(1) = js::Value(row.at(1).n * s);
                if (ax == 2 && row.size() > 2) row.at(2) = js::Value(row.at(2).n * s);
            }
        } else {
            std::vector<double> st = f.nums("stretch"); if (st.size() != 3) st = {1, 1, 1};
            st[(size_t)ax] *= s;
            js::Value a = js::Value::array(); for (double q : st) a.push(js::Value(q)); f["stretch"] = a;
        }
    } else if (d.comp >= 0) {
        js::Value& a = f[d.key];
        if (a.size() > (size_t)d.comp) a.at((size_t)d.comp) = js::Value(v);
    } else f[d.key] = js::Value(v);
    g_modified = true;
    fill_props(); rebuild();
    status("changed " + d.label + " of " + f.str("id", "the part") + " to " + fmtd(v) + " " + d.unit + " (Ctrl+Z undoes)");
}

static bool project(const V3& p, double& sx, double& sy) {
    double wx, wy, wz;
    if (!gluProject(p.x, p.y, p.z, g_mv, g_pj, g_vp, &wx, &wy, &wz) || wz < 0 || wz > 1) return false;
    sx = wx; sy = wy; return true;
}

// drawn inside render(): lines in 3D, labels in the overlay pass (dims_labels)
static void dims_render() {
    if (!g_dims_on || g_sel < 0) { g_dim_labels.clear(); return; }
    static int side = 0;
    int now_side = std::sin(g_yaw * geo::PI / 180) < 0 ? -1 : 1;
    if (g_dims_for != g_sel || g_dims_gen != g_gen || side != now_side) { dims_compute(); g_dims_for = g_sel; g_dims_gen = g_gen; side = now_side; }
    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(1.2f);
    for (size_t i = 0; i < g_dims.size(); ++i) {
        const PartDim& d = g_dims[i];
        bool hov = (int)i == g_dim_hover;
        glColor4f(hov ? 0.85f : 0.05f, hov ? 0.35f : 0.30f, hov ? 0.05f : 0.62f, 0.95f);
        glBegin(GL_LINES);
        if (d.kind == 1) {  // an angle: the edge it measures, and a short arc-like tick
            glVertex3d(d.a.x, d.a.y, d.a.z); glVertex3d(d.b.x, d.b.y, d.b.z);
        } else {
            glVertex3d(d.a.x, d.a.y, d.a.z); glVertex3d(d.b.x, d.b.y, d.b.z);
            V3 u = d.b - d.a; double L = geo::len(u); if (L > 1e-9) u = u / L;
            V3 n = std::fabs(u.z) < 0.9 ? geo::cross(u, V3(0, 0, 1)) : geo::cross(u, V3(1, 0, 0)); double ln = geo::len(n); if (ln > 0) n = n / ln;
            double t = 0.04 * L + 0.02;
            for (int s = 0; s < 2; ++s) { V3 p = s ? d.b : d.a; glVertex3d(p.x - n.x * t, p.y - n.y * t, p.z - n.z * t); glVertex3d(p.x + n.x * t, p.y + n.y * t, p.z + n.z * t); }
        }
        glEnd();
    }
    glLineWidth(1.0f); glDisable(GL_BLEND); glDisable(GL_LINE_SMOOTH); glEnable(GL_DEPTH_TEST);
}
static void dims_labels(int H) {  // overlay pass: value boxes at the projected anchors (also the click targets)
    g_dim_labels.clear();
    if (!g_dims_on || g_sel < 0) return;
    for (size_t i = 0; i < g_dims.size(); ++i) {
        const PartDim& d = g_dims[i];
        double sx, sy;
        if (!project(d.at, sx, sy)) continue;
        char buf[96]; snprintf(buf, sizeof buf, "%s %.4g %s", d.label.c_str(), d.value, d.unit.c_str());
        int w = 7 * (int)strlen(buf) + 10, hh = 18;
        int x = (int)sx - w / 2, y = (int)sy - hh / 2;
        bool hov = (int)i == g_dim_hover;
        glColor4f(1, 1, 1, 0.92f); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBegin(GL_QUADS); glVertex2i(x, y); glVertex2i(x + w, y); glVertex2i(x + w, y + hh); glVertex2i(x, y + hh); glEnd();
        glDisable(GL_BLEND);
        if (hov) glColor3f(0.85f, 0.35f, 0.05f); else glColor3f(0.05f, 0.30f, 0.62f);
        glBegin(GL_LINE_LOOP); glVertex2i(x, y); glVertex2i(x + w, y); glVertex2i(x + w, y + hh); glVertex2i(x, y + hh); glEnd();
        gl_text((float)x + 5, (float)y + 5, buf);
        RECT r = {x, H - (y + hh), x + w, H - y};  // window coordinates (y down)
        g_dim_labels.push_back({r, (int)i});
    }
}
static int dims_hit(int x, int y) { for (auto& L : g_dim_labels) if (x >= L.r.left && x < L.r.right && y >= L.r.top && y < L.r.bottom) return L.idx; return -1; }

static LRESULT CALLBACK dim_edit_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_KEYDOWN && (w == VK_RETURN || w == VK_ESCAPE)) {
        if (w == VK_RETURN) {
            std::string t = read_edit(h); char* e; double v = strtod(t.c_str(), &e);
            if (e != t.c_str() && std::isfinite(v)) dims_apply(g_dim_editing, v);
            else status("not a number: " + t);
        }
        g_dim_editing = -1; DestroyWindow(h); g_dim_edit = nullptr; SetFocus(g_view);
        return 0;
    }
    if (m == WM_KILLFOCUS) { g_dim_editing = -1; PostMessageA(h, WM_CLOSE, 0, 0); }
    if (m == WM_CLOSE) { DestroyWindow(h); if (g_dim_edit == h) g_dim_edit = nullptr; return 0; }
    if (m == WM_GETDLGCODE) return DLGC_WANTALLKEYS;
    return CallWindowProcA(g_dim_edit_old, h, m, w, l);
}
static void dims_begin_edit(int i) {
    if (i < 0 || i >= (int)g_dims.size()) return;
    RECT r = {0, 0, 0, 0};
    for (auto& L : g_dim_labels) if (L.idx == i) r = L.r;
    if (g_dim_edit) DestroyWindow(g_dim_edit);
    g_dim_editing = i;
    g_dim_edit = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", fmtd(g_dims[(size_t)i].value).c_str(), WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL, r.left, r.top - 2,
                                 std::max(90L, r.right - r.left), 24, g_view, nullptr, GetModuleHandleA(nullptr), nullptr);
    SendMessageA(g_dim_edit, WM_SETFONT, (WPARAM)g_font, TRUE);
    g_dim_edit_old = (WNDPROC)SetWindowLongPtrA(g_dim_edit, GWLP_WNDPROC, (LONG_PTR)dim_edit_proc);
    SendMessageA(g_dim_edit, EM_SETSEL, 0, -1);
    SetFocus(g_dim_edit);
    status("type the new " + g_dims[(size_t)i].label + " in " + g_dims[(size_t)i].unit + " and press Enter (Esc cancels)");
}

// which part is under the mouse: the added feature with the lowest distance at the picked surface point
static int pick_part(int x, int y) {
    V3 p, n;
    if (!pick_ray(x, y, p, n)) return -1;
    try {
        geo::Model m; m.load(g_doc);
        double best = 1e30; int bi = -1;
        for (size_t k = 0; k < m.feats.size(); ++k) {
            if (m.feats[k].op != geo::ADD) continue;
            double v = m.feats[k].eval(p);
            if (v < best) { best = v; bi = m.feats[k].src; }
        }
        return bi;
    } catch (...) { return -1; }
}
static void select_part(int k) {
    if (k == g_sel) return;
    g_sel = k;
    SendMessageA(g_list, LB_SETCURSEL, k, 0);
    fill_props(); rebuild();
    if (k >= 0) status(features().at((size_t)k).str("id", "part") + " (" + part_appearance_name(k) + "): click a dimension to change it, right-click for more, double-click to edit");
}

// ---------------------------------------------------------------- floating view toolbar
static HWND g_viewbar = nullptr;
// ID_VB_* are in the main command list (studio.cpp)
struct VBItem { int cmd; const char* icon; const char* tip; };
static const VBItem VBAR[] = {
    {ID_VIEW_FIT, "view_fit", "Fit the model (F)"}, {ID_VB_ZIN, "zoom_in", "Zoom in"}, {ID_VB_ZOUT, "zoom_out", "Zoom out"},
    {ID_VIEW_ISO, "view_iso", "Isometric (0)"}, {ID_VIEW_FRONT, "view_front", "Front (1)"}, {ID_VIEW_SIDE, "view_side", "Side (3)"}, {ID_VIEW_TOP, "view_top", "Top (5)"},
    {ID_VIEW_WIRE, "view_wire", "Shaded / wireframe (W)"}, {ID_VIEW_CUT, "cut_y", "Section view (C); X, Y, Z pick the plane"},
    {ID_VB_DIMS, "annotate", "Dimensions of the selected part on or off"}, {ID_VB_APPEAR, "colour_t", "Colours: by part, by material, single"},
};
static const int NVB = (int)(sizeof(VBAR) / sizeof(VBAR[0])), VB_CELL = 26;
static int g_vb_hover = -1;
static void paint_viewbar(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    gp::Bitmap buf(rc.right, rc.bottom, PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    g.Clear(gcol(246, 247, 249));
    gp::Pen bd(gcol(190, 195, 202), 1.0f); g.DrawRectangle(&bd, 0, 0, rc.right - 1, rc.bottom - 1);
    for (int i = 0; i < NVB; ++i) {
        gp::RectF r(2.0f + i * VB_CELL, 2.0f, VB_CELL - 2.0f, VB_CELL - 2.0f);
        bool on = (VBAR[i].cmd == ID_VB_DIMS && g_dims_on) || (VBAR[i].cmd == ID_VIEW_WIRE && g_wire) || (VBAR[i].cmd == ID_VIEW_CUT && g_cut);
        if (i == g_vb_hover || on) { gp::SolidBrush hb(on ? gcol(200, 222, 246) : gcol(220, 233, 248)); g.FillRectangle(&hb, r); gp::Pen pn(gcol(135, 175, 222), 1.0f); g.DrawRectangle(&pn, r); }
        if (gp::Bitmap* ic = icon_flat(VBAR[i].icon)) g.DrawImage(ic, r.X + 3, r.Y + 3, 18.0f, 18.0f);
    }
    gp::Graphics out(dc); out.DrawImage(&buf, 0, 0);
}
static LRESULT CALLBACK viewbar_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_PAINT: { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_viewbar(h, dc); EndPaint(h, &ps); return 0; }
        case WM_ERASEBKGND: return 1;
        case WM_MOUSEMOVE: {
            int i = ((short)LOWORD(l) - 2) / VB_CELL; if (i < 0 || i >= NVB) i = -1;
            if (i != g_vb_hover) { g_vb_hover = i; InvalidateRect(h, nullptr, FALSE); if (i >= 0) status(VBAR[i].tip); }
            TRACKMOUSEEVENT te = {sizeof te, TME_LEAVE, h, 0}; TrackMouseEvent(&te);
            return 0;
        }
        case WM_MOUSELEAVE: g_vb_hover = -1; InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_LBUTTONUP: {
            int i = ((short)LOWORD(l) - 2) / VB_CELL;
            if (i >= 0 && i < NVB) { PostMessageA(g_main, WM_COMMAND, VBAR[i].cmd, 0); InvalidateRect(h, nullptr, FALSE); }
            return 0;
        }
    }
    return DefWindowProcA(h, m, w, l);
}
static void viewbar_place() {
    if (!g_viewbar || !g_view) return;
    RECT vr; GetWindowRect(g_view, &vr);
    POINT p = {vr.left, vr.top}; ScreenToClient(g_main, &p);
    int w = NVB * VB_CELL + 4, x = p.x + ((vr.right - vr.left) - w) / 2;
    SetWindowPos(g_viewbar, HWND_TOP, x, p.y + 6, w, VB_CELL + 4, SWP_NOACTIVATE | (IsWindowVisible(g_view) ? SWP_SHOWWINDOW : SWP_HIDEWINDOW));
}
