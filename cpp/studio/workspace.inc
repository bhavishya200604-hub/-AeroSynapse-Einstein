// Workspace of AeroSynapse Studio: the info card of the selected feature, palette tooltips, collapsible panels and full
// 3D view, the start page (templates, blank design, new project in a new folder), the skeleton-parameter table, the
// view analyses (zebra stripes, curvature map, 3D annotations), space reservations and DMU kinematics.

#include <shlobj.h>

// ---------------------------------------------------------------- what every feature is and what to do with it
struct FeatureHelp { const char* type; const char* icon; const char* title; const char* what; const char* todo; };
static const FeatureHelp FEATURE_HELP[] = {
    {"fuselage", "fuselage", "Fuselage", "A body through stations: x, half-width, half-height, centre height and section shape (2 round, 4 boxy).",
     "Edit 'stations' in the JSON panel; add stations for a longer nose or a waisted (area-ruled) body."},
    {"wing", "wing", "Wing", "A lifting surface: span from its root, root chord, taper, leading-edge sweep, dihedral, twist and section.",
     "Change span, chord or sweep on the left of each row; set 'profile' to naca, biconvex, diamond or table."},
    {"canard", "canard", "Canard", "A foreplane ahead of the wing: trim and pitch control, extra lift on deltas.", "Move it with the arrow keys; keep it above the wing's wake."},
    {"tail", "tail", "Tailplane", "The horizontal tail: pitch stability and trim.", "Size it with span and chord; dihedral -2 to 0 deg is usual."},
    {"fin", "fin", "Fin", "A vertical or canted fin: directional stability.", "dihedral 90 is vertical; 60-75 cants twin fins outward; 'mirror' makes a pair."},
    {"strake", "strake", "Strake", "A leading-edge extension: vortex lift at high incidence.", "Long, thin and very swept (70-80 deg) ahead of the wing root."},
    {"blade", "blade", "Blade", "A small fence or blade on the wing.", "Use 'pattern' to repeat it along the span."},
    {"sphere", "sphere", "Sphere", "A sphere of radius r.", "Set r; move it with the arrow keys (Shift for 1 m steps)."},
    {"ellipsoid", "ellipsoid", "Ellipsoid", "An ellipsoid with three radii: canopies, fairings, tanks.", "Edit 'radii' [x, y, z]; give it a blend to fair it into the body."},
    {"box", "box", "Box", "A box with rounded edges.", "Set 'size' and 'round'; switch to cut (+ / - / &) for a slot or bay."},
    {"cylinder", "cylinder", "Cylinder", "A cylinder of radius r and length along an axis.", "Set 'axis' x, y or z; as a cut it is a hole."},
    {"cone", "cone", "Cone", "A cone frustum along x from radius r1 to r2.", "Nose cones, nozzles and spinners."},
    {"torus", "torus", "Torus", "A ring about x: inlet lips, rings.", "R is the ring radius, r the tube radius."},
    {"capsule", "capsule", "Capsule", "A rounded rod between two points a and b.", "Edit 'a' and 'b'; good for struts and booms."},
    {"duct", "duct", "Duct / pipe", "A tube swept along a path of points with a radius at each.", "As a cut it makes intakes and channels; as an add it is a pipe or harness."},
    {"halfspace", "halfspace", "Half-space", "Everything on one side of a plane.", "As a cut it trims the design flat; set 'normal' and 'offset'."},
    {"revolve", "revolve", "Revolve (shaft)", "A profile (x, r) turned about the local x axis.", "Edit 'profile'; turn 'smooth' on for a spline through the points."},
    {"extrude", "extrude", "Extrude (pad)", "A closed outline extruded symmetrically about its sketch plane.", "Edit 'height' for the thickness and 'draft' (deg) for tapered walls; as a cut it is a pocket."},
    {"loft", "loft", "Loft", "A body through cross-sections at stations along x.", "Each section needs the same number of points; 'smooth' splines them."},
    {"sheet", "sheet", "Bezier sheet", "A curved surface from a 4 x 4 control net, with a thickness.", "Move the 16 control points in 'control'; check its fairness with Zebra."},
    {"lattice", "lattice", "Lattice", "A periodic lattice block: gyroid, Schwarz-P, diamond or struts.", "Set 'cell' and 'wall'; lattices print well by additive manufacturing."},
    {"implicit", "implicit", "Equation surface", "Any solid where f(x, y, z) <= 0, inside a bounding 'size' box.", "Edit 'expr': + - * / ^, sin cos exp sqrt abs min max, pi, x y z."},
    {"superellipsoid", "superellipsoid", "Superellipsoid", "Between an ellipsoid (n = 1) and a box (n near 0.1).", "Lower n1 and n2 for a boxier shape; 2 gives an octahedron."},
    {"bentplate", "booleans", "Bent plate (sheet metal)", "A sheet-metal part: an open profile with real bends, a thickness and a width.", "Edit 'profile', 'thickness', 'bend_radius'; the flat length (K = 0.44) is below."},
};

static const FeatureHelp* help_for(const std::string& type) {
    for (auto& h : FEATURE_HELP) if (type == h.type) return &h;
    return nullptr;
}

static double flat_length(const js::Value& f) {  // unfolded length of a bent plate, K factor 0.44
    std::vector<double> xs, zs;
    if (const js::Value* a = f.find("profile")) for (size_t i = 0; i < a->size(); ++i) if (a->at(i).size() >= 2) { xs.push_back(a->at(i).at(0).n); zs.push_back(a->at(i).at(1).n); }
    double t = f.num("thickness", 0.003), r = f.num("bend_radius", 0.01), L = 0;
    for (size_t i = 0; i + 1 < xs.size(); ++i) L += std::hypot(xs[i + 1] - xs[i], zs[i + 1] - zs[i]);
    for (size_t i = 1; i + 1 < xs.size(); ++i) {
        double ax = xs[i] - xs[i - 1], az = zs[i] - zs[i - 1], bx = xs[i + 1] - xs[i], bz = zs[i + 1] - zs[i];
        double th = std::acos(std::max(-1.0, std::min(1.0, (ax * bx + az * bz) / std::max(1e-12, std::hypot(ax, az) * std::hypot(bx, bz)))));
        double rc = r + t / 2, setback = rc * std::tan(th / 2);
        L += th * (r + 0.44 * t) - 2 * setback;
    }
    return L;
}

// ---------------------------------------------------------------- info card
static HWND g_card;
static const int CARD_H = 138;

static void paint_card(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    int W = std::max(1L, rc.right), H = std::max(1L, rc.bottom);
    gp::Bitmap buf(W, H, PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    g.Clear(gfrom(C_PANEL));
    gp::RectF card(3, 3, W - 7.0f, H - 8.0f);
    soft_shadow(g, card, 10, 4, 60);
    fill_round(g, card, 10, gcol(255, 255, 255), gcol(232, 240, 250));
    gp::GraphicsPath pth; round_path(pth, card, 10);
    gp::Pen pen(gcol(186, 205, 230), 1.0f); g.DrawPath(&pen, &pth);
    const js::Value* f = nullptr;
    if (g_sel >= 0 && g_sel < (int)features().size()) f = &features().at((size_t)g_sel);
    std::string type = f ? f->str("type", "") : "", icon = "sch_sketch", title = "No feature selected", what = "Pick a tool on the left to add a part, or a feature in the tree to edit it.",
                todo = "Workspaces: click a scheme tab; click the active tab again to hide its tools. F2 tree, F3 properties, F11 full view.";
    if (const FeatureHelp* hp = help_for(type)) { icon = hp->icon; title = hp->title; what = hp->what; todo = hp->todo; }
    else if (f) { icon = "sch_sketch"; title = type; what = "A feature of the design."; todo = "Edit it in the rows below or in the JSON panel."; }
    if (f) {
        char b[160] = "";
        if (type == "bentplate") snprintf(b, sizeof b, "  Flat length %.1f mm (K 0.44).", 1e3 * flat_length(*f));
        if (f->find("span") && f->find("root_chord")) {
            double s = f->num("span", 1), c = f->num("root_chord", 1), l = f->num("taper", 0.5), A = 0.5 * (1 + l) * c * s * (f->flag("mirror", true) ? 2 : 1);
            snprintf(b, sizeof b, "  Area %.2f m2, aspect ratio %.2f.", A, std::pow(s * (f->flag("mirror", true) ? 2 : 1), 2) / std::max(A, 1e-9));
        }
        title += std::string("  (") + f->str("id", "") + ")";
        what += b;
    }
    if (f && !g_param_key.empty()) {
        const ParamHelp* ph = param_help(g_param_key);
        todo = g_param_key + (ph && ph->unit[0] ? std::string(" [") + ph->unit + "]" : "") + ": " + (ph ? ph->what : "a parameter of this feature") +
               (ph && ph->choices ? std::string(" (") + ph->choices + ")" : "");
    }
    if (gp::Bitmap* ic = icon_png(icon)) g.DrawImage(ic, card.X + 8, card.Y + 10, 46.0f, 46.0f);
    gtext(g, title, gp::RectF(card.X + 62, card.Y + 6, card.Width - 68, 20), gcol(15, 30, 55), 14.0f, gp::FontStyleBold);
    gtext(g, what, gp::RectF(card.X + 62, card.Y + 26, card.Width - 68, 52), gcol(45, 60, 85), 11.0f, gp::FontStyleRegular, gp::StringAlignmentNear, gp::StringAlignmentNear, true);
    gp::RectF tip(card.X + 8, card.Y + 80, card.Width - 16, card.Height - 84);
    fill_round(g, tip, 6, gcol(224, 242, 254), gcol(207, 232, 252));
    gtext(g, (g_param_key.empty() || !f ? std::string("What to do: ") : std::string("")) + todo, gp::RectF(tip.X + 6, tip.Y + 1, tip.Width - 10, tip.Height - 2), gcol(7, 89, 133), 11.0f, gp::FontStyleRegular,
          gp::StringAlignmentNear, gp::StringAlignmentCenter, true);
    gp::Graphics out(dc); out.DrawImage(&buf, 0, 0);
}

static LRESULT CALLBACK card_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_PAINT) { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_card(h, dc); EndPaint(h, &ps); return 0; }
    if (m == WM_ERASEBKGND) return 1;
    return DefWindowProcA(h, m, w, l);
}
static void card_refresh() { if (g_card) InvalidateRect(g_card, nullptr, FALSE); }

// ---------------------------------------------------------------- palette tooltip
static HWND g_tip;
static int g_tip_tool = -1;

static void paint_tip(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    int W = rc.right, H = rc.bottom;
    gp::Bitmap buf(W, H, PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    g.Clear(gcol(8, 16, 32));
    if (g_tip_tool < 0) return;
    const ToolDef& td = TOOLS[g_tip_tool];
    int sk = 0; for (int k = 0; k < NSCHEME; ++k) if (std::string(td.scheme) == SCHEMES[k].key) sk = k;
    gp::Color c = scheme_col(sk);
    gp::LinearGradientBrush bg(gp::Rect(0, 0, W, H), gmix(gcol(10, 22, 44), c, 0.35f), gcol(8, 16, 32), gp::LinearGradientModeVertical);
    g.FillRectangle(&bg, 0, 0, W, H);
    gp::Pen pen(c, 2.0f); g.DrawRectangle(&pen, 1, 1, W - 3, H - 3);
    if (gp::Bitmap* ic = icon_png(td.icon)) g.DrawImage(ic, 10.0f, 10.0f, 56.0f, 56.0f);
    gtext(g, td.label, gp::RectF(76, 8, W - 84.0f, 22), gcol(255, 255, 255), 15.0f, gp::FontStyleBold);
    gtext(g, td.tip, gp::RectF(76, 30, W - 84.0f, H - 52.0f), gcol(205, 225, 245), 11.5f, gp::FontStyleRegular, gp::StringAlignmentNear, gp::StringAlignmentNear, true);
    gtext(g, td.kind ? "Click to insert, then edit it on the right" : "Click to run", gp::RectF(76, H - 22.0f, W - 84.0f, 18), gshade(c, 0.5f), 10.5f, gp::FontStyleItalic);
    gp::Graphics out(dc); out.DrawImage(&buf, 0, 0);
}
static LRESULT CALLBACK tip_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_PAINT) { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_tip(h, dc); EndPaint(h, &ps); return 0; }
    if (m == WM_ERASEBKGND) return 1;
    if (m == WM_MOUSEACTIVATE) return MA_NOACTIVATE;
    return DefWindowProcA(h, m, w, l);
}
static void show_tip(int tool, int screen_x, int screen_y) {
    if (!g_tip) g_tip = CreateWindowExA(WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_NOACTIVATE, "AeroSynapseTip", "", WS_POPUP, 0, 0, 340, 110, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
    if (tool < 0) { ShowWindow(g_tip, SW_HIDE); g_tip_tool = -1; return; }
    g_tip_tool = tool;
    SetWindowPos(g_tip, HWND_TOPMOST, screen_x, screen_y, 340, 110, SWP_NOACTIVATE | SWP_SHOWWINDOW);
    InvalidateRect(g_tip, nullptr, FALSE);
}

// ---------------------------------------------------------------- panels and full view
static void layout();
static void palette_target();
static void toggle_panel(int which) {  // 0 palette, 1 tree, 2 properties, 3 full view
    if (which == 0) g_show_palette = !g_show_palette;
    else if (which == 1) g_show_tree = !g_show_tree;
    else if (which == 2) g_show_props = !g_show_props;
    else { bool any = g_show_palette || g_show_tree || g_show_props; g_show_palette = g_show_tree = g_show_props = !any; }
    palette_target();  // the tool bar slides in or out
    layout();
    status(std::string("panels: tools ") + (g_show_palette ? "shown" : "hidden") + ", tree " + (g_show_tree ? "shown" : "hidden") + ", properties " + (g_show_props ? "shown" : "hidden") +
           "   (F4 tools, F2 tree, F3 properties, F11 full 3D view)");
}

// ---------------------------------------------------------------- view analysis: zebra, curvature, annotations, reservations
static int g_view_mode = 0;  // 0 shaded, 1 zebra, 2 curvature
static bool g_annot = false;

static void analysis_colours() {
    if (g_view_mode == 0 || g_mesh.v.empty()) return;
    g_colors.assign(g_mesh.v.size() * 3, 0.5f);
    if (g_view_mode == 1) {  // reflection stripes from a fixed direction: kinks and waviness show as broken stripes
        V3 d(0.35, 0.55, 0.76);
        for (size_t i = 0; i < g_mesh.v.size(); ++i) {
            double s = geo::dot(g_mesh.n[i], d) / std::max(1e-9, geo::len(g_mesh.n[i]));
            double a = std::acos(std::max(-1.0, std::min(1.0, s))) * 16.0 / geo::PI;
            float c = std::fmod(a, 1.0) < 0.5 ? 0.06f : 0.97f;
            g_colors[3 * i] = c; g_colors[3 * i + 1] = c; g_colors[3 * i + 2] = c;
        }
        g_result_label = "zebra (reflection lines): smooth stripes = curvature-continuous (G2) surface";
        g_rmin = 0; g_rmax = 1; g_cmap_turbo = false;
        return;
    }
    std::vector<double> k(g_mesh.v.size(), 0.0), cnt(g_mesh.v.size(), 0.0);
    for (size_t t = 0; t + 2 < g_mesh.t.size(); t += 3)
        for (int e = 0; e < 3; ++e) {
            unsigned a = g_mesh.t[t + e], b = g_mesh.t[t + (e + 1) % 3];
            V3 na = g_mesh.n[a], nb = g_mesh.n[b];
            double la = geo::len(na), lb = geo::len(nb), dl = geo::len(g_mesh.v[a] - g_mesh.v[b]);
            if (la < 1e-12 || lb < 1e-12 || dl < 1e-9) continue;
            double kk = geo::len(na / la - nb / lb) / dl;
            k[a] += kk; cnt[a] += 1; k[b] += kk; cnt[b] += 1;
        }
    std::vector<double> lv;
    for (size_t i = 0; i < k.size(); ++i) { k[i] = cnt[i] > 0 ? std::log10(1e-3 + k[i] / cnt[i]) : -3; lv.push_back(k[i]); }
    std::sort(lv.begin(), lv.end());
    double lo = lv[lv.size() / 50], hi = lv[lv.size() * 98 / 100];
    if (hi <= lo) hi = lo + 1e-6;
    for (size_t i = 0; i < k.size(); ++i) { float r, g, b; turbo((k[i] - lo) / (hi - lo), r, g, b); g_colors[3 * i] = r; g_colors[3 * i + 1] = g; g_colors[3 * i + 2] = b; }
    g_result_label = "curvature, log10 (1/m)"; g_rmin = lo; g_rmax = hi; g_cmap_turbo = true;
}

static void draw_text3d(double x, double y, double z, const std::string& s) {
    GLdouble mv[16], pj[16]; GLint vp[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, mv); glGetDoublev(GL_PROJECTION_MATRIX, pj); glGetIntegerv(GL_VIEWPORT, vp);
    GLdouble wx, wy, wz;
    if (gluProject(x, y, z, mv, pj, vp, &wx, &wy, &wz) == GL_TRUE && wz < 1.0) {
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, vp[2], 0, vp[3], -1, 1);
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
        glDisable(GL_DEPTH_TEST);
        gl_text((float)wx + 4, (float)wy + 4, s);
        glEnable(GL_DEPTH_TEST);
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);
    }
}

static void annotations_render() {
    glDisable(GL_LIGHTING);
    // space reservations: keep-out volumes drawn as amber glass boxes
    const js::Value& fl = features();
    for (size_t i = 0; i < fl.size(); ++i) {
        const js::Value& f = fl.at(i);
        if (!f.flag("reserve", false)) continue;
        std::vector<double> sz = f.nums("size"), t = f.nums("translate");
        if (sz.size() < 3) continue;
        double c[3] = {t.size() > 0 ? t[0] : 0, t.size() > 1 ? t[1] : 0, t.size() > 2 ? t[2] : 0};
        double lo[3] = {c[0] - sz[0] / 2, c[1] - sz[1] / 2, c[2] - sz[2] / 2}, hi[3] = {c[0] + sz[0] / 2, c[1] + sz[1] / 2, c[2] + sz[2] / 2};
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
        glColor4f(0.98f, 0.75f, 0.14f, 0.18f);
        int fc[6][4] = {{0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3}};
        auto V = [&](int k) { return V3((k & 1) ? hi[0] : lo[0], (k & 2) ? hi[1] : lo[1], (k & 4) ? hi[2] : lo[2]); };
        glBegin(GL_QUADS); for (auto& q : fc) for (int k : q) { V3 p = V(k); glVertex3d(p.x, p.y, p.z); } glEnd();
        glColor4f(0.98f, 0.75f, 0.14f, 0.9f);
        glBegin(GL_LINES); int ed[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        for (auto& e : ed) { V3 a = V(e[0]), b = V(e[1]); glVertex3d(a.x, a.y, a.z); glVertex3d(b.x, b.y, b.z); } glEnd();
        glDepthMask(GL_TRUE); glDisable(GL_BLEND);
        glColor3f(0.98f, 0.8f, 0.3f);
        draw_text3d(hi[0], hi[1], hi[2], "reserved: " + f.str("id", ""));
    }
    if (!g_annot || g_mesh.v.empty()) return;
    // model-based definition: overall dimensions in the 3D view
    V3 lo = g_props.lo, hi = g_props.hi;
    double off = 0.08 * (hi.x - lo.x);
    glColor3f(0.13f, 0.83f, 0.93f); glLineWidth(1.5f);
    glBegin(GL_LINES);
    glVertex3d(lo.x, lo.y - off, lo.z); glVertex3d(hi.x, lo.y - off, lo.z);
    glVertex3d(hi.x + off, lo.y, lo.z); glVertex3d(hi.x + off, hi.y, lo.z);
    glVertex3d(hi.x + off, lo.y - off, lo.z); glVertex3d(hi.x + off, lo.y - off, hi.z);
    for (double x : {lo.x, hi.x}) { glVertex3d(x, lo.y, lo.z); glVertex3d(x, lo.y - 1.3 * off, lo.z); }
    for (double y : {lo.y, hi.y}) { glVertex3d(hi.x, y, lo.z); glVertex3d(hi.x + 1.3 * off, y, lo.z); }
    glEnd(); glLineWidth(1.0f);
    char b[96];
    glColor3f(0.85f, 0.97f, 1.0f);
    snprintf(b, sizeof b, "length %.0f mm", 1e3 * (hi.x - lo.x)); draw_text3d(0.5 * (lo.x + hi.x), lo.y - off, lo.z, b);
    snprintf(b, sizeof b, "span %.0f mm", 1e3 * (hi.y - lo.y)); draw_text3d(hi.x + off, 0.5 * (lo.y + hi.y), lo.z, b);
    snprintf(b, sizeof b, "height %.0f mm", 1e3 * (hi.z - lo.z)); draw_text3d(hi.x + off, lo.y - off, 0.5 * (lo.z + hi.z), b);
    snprintf(b, sizeof b, "volume %.3f m3, wetted %.2f m2", g_props.volume, g_props.area); draw_text3d(lo.x, hi.y, hi.z, b);
}

// ---------------------------------------------------------------- DMU kinematics: swing the selected feature about its pivot
static bool g_kin_on = false;
static int g_kin_feat = -1;
static double g_kin_base = 0, g_kin_t = 0, g_kin_h = 0;
static void kin_toggle() {
    if (g_kin_on) {
        g_kin_on = false; KillTimer(g_main, 9);
        if (g_kin_feat >= 0 && g_kin_feat < (int)features().size()) { js::Value& f = features().at((size_t)g_kin_feat); js::Value r = js::Value::array(); std::vector<double> rr = f.nums("rotate"); r.push(js::Value(rr.size() > 0 ? rr[0] : 0.0)); r.push(js::Value(rr.size() > 1 ? rr[1] : 0.0)); r.push(js::Value(g_kin_base)); f["rotate"] = r; }
        g_h = g_kin_h; rebuild(); status("kinematics stopped");
        return;
    }
    if (g_sel < 0) { MessageBoxA(g_main, "Select the moving feature first (for example a wing with a pivot: Transform > Pivot rotate).", "DMU kinematics", MB_ICONINFORMATION); return; }
    g_kin_feat = g_sel; g_kin_on = true; g_kin_t = 0; g_kin_h = g_h;
    std::vector<double> rr = features().at((size_t)g_sel).nums("rotate");
    g_kin_base = rr.size() > 2 ? rr[2] : 0.0;
    g_h = std::max(g_h, 0.09);  // draft resolution while it moves
    SetTimer(g_main, 9, 60, nullptr);
    status("DMU kinematics: '" + features().at((size_t)g_sel).str("id", "") + "' swings +/- 25 deg about its pivot (click again to stop)");
}
static void kin_tick() {
    if (!g_kin_on || g_building || g_kin_feat < 0 || g_kin_feat >= (int)features().size()) return;
    g_kin_t += 0.25;
    js::Value& f = features().at((size_t)g_kin_feat);
    std::vector<double> rr = f.nums("rotate");
    js::Value r = js::Value::array();
    r.push(js::Value(rr.size() > 0 ? rr[0] : 0.0)); r.push(js::Value(rr.size() > 1 ? rr[1] : 0.0)); r.push(js::Value(g_kin_base + 25.0 * std::sin(g_kin_t)));
    f["rotate"] = r;
    rebuild();
}

// ---------------------------------------------------------------- skeleton parameters (top-down design)
static HWND g_par, g_par_edit;
enum { ID_PAR_REBUILD = 8800, ID_PAR_EDIT };
static LRESULT CALLBACK par_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            HINSTANCE hi = GetModuleHandleA(nullptr);
            g_par_edit = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN, 8, 40, 404, 470, h, (HMENU)ID_PAR_EDIT, hi, nullptr);
            HWND b = CreateWindowExA(0, "BUTTON", "Rebuild from parameters", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 8, 518, 404, 32, h, (HMENU)ID_PAR_REBUILD, hi, nullptr);
            static HFONT mono = CreateFontA(-13, 0, 0, 0, FW_NORMAL, 0, 0, 0, ANSI_CHARSET, 0, 0, CLEARTYPE_QUALITY, FIXED_PITCH, "Consolas");
            SendMessageA(g_par_edit, WM_SETFONT, (WPARAM)mono, TRUE); SendMessageA(b, WM_SETFONT, (WPARAM)g_font, TRUE);
            return 0;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps);
            gp::Graphics g(dc); quality(g);
            gp::LinearGradientBrush bg(gp::Rect(0, 0, 420, 36), gcol(34, 197, 94), gcol(22, 50, 92), gp::LinearGradientModeHorizontal);
            g.FillRectangle(&bg, 0, 0, 420, 36);
            gtext(g, "Skeleton: the design-space parameters drive the geometry", gp::RectF(10, 0, 400, 36), gcol(255, 255, 255), 12.5f, gp::FontStyleBold);
            EndPaint(h, &ps); return 0;
        }
        case WM_DRAWITEM: draw_button((DRAWITEMSTRUCT*)l, gcol(34, 197, 94)); return TRUE;
        case WM_COMMAND:
            if (LOWORD(w) == ID_PAR_REBUILD) {
                std::string t = read_edit(g_par_edit), line;
                js::Value v = js::Value::object();
                size_t p = 0;
                while (p <= t.size()) {
                    size_t e = t.find('\n', p); if (e == std::string::npos) e = t.size();
                    line = t.substr(p, e - p); p = e + 1;
                    size_t q = line.find('=');
                    if (q == std::string::npos) continue;
                    std::string k = line.substr(0, q); while (!k.empty() && (k.back() == ' ' || k.back() == '\r')) k.pop_back(); while (!k.empty() && k[0] == ' ') k.erase(0, 1);
                    if (!k.empty()) v[k] = atof(line.substr(q + 1).c_str());
                }
                g_doc["variables"] = v;
                std::string d = work_dir("skeleton");
                js::write_file(d + "\\design.json", js::dump(g_doc, 2));
                launch("\"" + python() + "\" -m aerosynapse.cli engineer skeleton \"" + d + "\\design.json\" --out \"" + d + "\"", "engineer skeleton", d);
            }
            return 0;
        case WM_CLOSE: ShowWindow(h, SW_HIDE); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}
static void open_params() {
    const js::Value* v = g_doc.find("variables");
    if (!v || !v->is_obj()) { MessageBoxA(g_main, "This design has no skeleton parameters. Start from a jet-fighter template (Start page) to get a parameter-driven design.", "Skeleton", MB_ICONINFORMATION); return; }
    if (!g_par) g_par = CreateWindowExA(WS_EX_TOOLWINDOW, "AeroSynapseParams", "Skeleton parameters", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT, 436, 600, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
    std::string t;
    for (auto& kv : *v->o) { char b[128]; snprintf(b, sizeof b, "%-14s = %.6g\r\n", kv.first.c_str(), kv.second.n); t += b; }
    SetWindowTextA(g_par_edit, t.c_str());
    ShowWindow(g_par, SW_SHOW); SetForegroundWindow(g_par);
}

// ---------------------------------------------------------------- engineering tools (clash, rules, composites, RFLP)
static void run_engineer(const char* kind) {
    std::string d = work_dir(std::string("eng_") + kind);
    std::string target;
    if (std::string(kind) == "rflp") {
        if (!ensure_project(g_main)) return;
        target = g_wf_dir;
    } else { target = d + "\\design.json"; js::write_file(target, js::dump(g_doc, 2)); }
    std::string extra = std::string(kind) == "composites" ? " --layup \"" + g_doc.str("layup", "[0/45/-45/90]s") + "\" --material cfrp_im7" : "";
    launch("\"" + python() + "\" -m aerosynapse.cli engineer " + kind + " \"" + target + "\" --out \"" + d + "\"" + extra, std::string("engineer ") + kind, d);
}

// ---------------------------------------------------------------- start page
struct Tpl { std::string key, group, title, desc; };
static std::vector<Tpl> g_tpl;
static HWND g_start;
static int g_start_hover = -1, g_start_scroll = 0;
static const int START_LEFT = 300;

static void tpl_load() {
    if (!g_tpl.empty()) return;
    try {
        js::Value idx = js::parse(js::read_file(g_root + "\\assets\\templates\\index.json"));
        for (size_t i = 0; i < idx.size(); ++i) g_tpl.push_back({idx.at(i).str("key", ""), idx.at(i).str("group", ""), idx.at(i).str("title", ""), idx.at(i).str("description", "")});
    } catch (...) {}
}

struct Hit { RECT r; int kind; int idx; };  // kind 0 action, 1 template
static std::vector<Hit> g_start_hits;
static const char* START_ACTIONS[4][3] = {{"Blank design", "Start from nothing: sketch a body, add parts.", "wf_new"},
                                          {"New project in a new folder...", "Pick a folder; the design and its workflow live there.", "wf_open"},
                                          {"Open a design...", "Any AeroSynapse design file (.json).", "save"},
                                          {"Open a project...", "A design workflow project (project.json).", "wf_window"}};

static void paint_start(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    int W = rc.right, H = rc.bottom;
    gp::Bitmap buf(std::max(1, W), std::max(1, H), PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    gp::LinearGradientBrush bg(gp::Rect(0, 0, W, H), gcol(5, 12, 28), gcol(14, 40, 80), 30.0f);
    g.FillRectangle(&bg, 0, 0, W, H);
    g_start_hits.clear();
    // left column: logo and actions
    if (gp::Bitmap* logo = load_png(g_root + "\\docs\\assets\\aerosynapse_logo.png")) g.DrawImage(logo, 24, 22, 64, 64);
    gtext(g, "AeroSynapse", gp::RectF(98, 24, START_LEFT - 104, 30), gcol(255, 255, 255), 21.0f, gp::FontStyleBold);
    gtext(g, "Studio  |  start a design", gp::RectF(99, 54, START_LEFT - 104, 22), gcol(120, 210, 255), 13.0f);
    for (int k = 0; k < 4; ++k) {
        RECT r = {20, 110 + k * 84, START_LEFT - 20, 110 + k * 84 + 70};
        gp::RectF fr((float)r.left, (float)r.top, (float)(r.right - r.left), (float)(r.bottom - r.top));
        bool hv = g_start_hover == 1000 + k;
        soft_shadow(g, fr, 10, 5, 90);
        fill_round(g, fr, 10, hv ? gcol(30, 80, 140) : gcol(22, 52, 96), hv ? gcol(14, 40, 80) : gcol(12, 30, 60));
        gp::GraphicsPath p; round_path(p, fr, 10); gp::Pen pen(hv ? gcol(34, 211, 238) : gcol(60, 100, 160), 1.3f); g.DrawPath(&pen, &p);
        if (gp::Bitmap* ic = icon_png(START_ACTIONS[k][2])) g.DrawImage(ic, fr.X + 10, fr.Y + 11, 48.0f, 48.0f);
        gtext(g, START_ACTIONS[k][0], gp::RectF(fr.X + 68, fr.Y + 10, fr.Width - 74, 22), gcol(255, 255, 255), 13.5f, gp::FontStyleBold);
        gtext(g, START_ACTIONS[k][1], gp::RectF(fr.X + 68, fr.Y + 32, fr.Width - 74, 34), gcol(170, 200, 235), 11.0f, gp::FontStyleRegular, gp::StringAlignmentNear, gp::StringAlignmentNear, true);
        g_start_hits.push_back({r, 0, k});
    }
    gtext(g, "Templates: pick a starting body, airfoil or airplane.  Esc closes this page.", gp::RectF(20, (float)H - 40, START_LEFT - 30.0f, 34), gcol(120, 160, 210), 10.5f,
          gp::FontStyleRegular, gp::StringAlignmentNear, gp::StringAlignmentNear, true);
    // right: template gallery by group
    int x0 = START_LEFT, cw = 232, ch = 186, gap = 16, cols = std::max(1, (W - x0 - 20) / (cw + gap)), y = 20 - g_start_scroll;
    std::string grp;
    int col = 0;
    for (size_t i = 0; i < g_tpl.size(); ++i) {
        if (g_tpl[i].group != grp) {
            if (col) { y += ch + gap; col = 0; }
            grp = g_tpl[i].group;
            gtext(g, grp, gp::RectF((float)x0, (float)y, 400, 26), gcol(255, 255, 255), 16.0f, gp::FontStyleBold);
            gp::SolidBrush ln(gcol(34, 211, 238)); g.FillRectangle(&ln, (float)x0, (float)y + 28, 60.0f, 3.0f);
            y += 40;
        }
        RECT r = {x0 + col * (cw + gap), y, x0 + col * (cw + gap) + cw, y + ch};
        gp::RectF fr((float)r.left, (float)r.top, (float)cw, (float)ch);
        bool hv = g_start_hover == (int)i;
        soft_shadow(g, fr, 10, 6, hv ? 140 : 90);
        fill_round(g, fr, 10, hv ? gcol(28, 70, 130) : gcol(16, 36, 68), gcol(9, 20, 40));
        gp::GraphicsPath p; round_path(p, fr, 10); gp::Pen pen(hv ? gcol(34, 211, 238) : gcol(50, 80, 130), hv ? 2.0f : 1.0f); g.DrawPath(&pen, &p);
        if (gp::Bitmap* im = load_png(g_root + "\\assets\\templates\\" + g_tpl[i].key + ".png")) g.DrawImage(im, fr.X + 6, fr.Y + 6, cw - 12.0f, 120.0f);
        gtext(g, g_tpl[i].title, gp::RectF(fr.X + 10, fr.Y + 128, cw - 20.0f, 20), gcol(255, 255, 255), 13.0f, gp::FontStyleBold);
        gtext(g, g_tpl[i].desc, gp::RectF(fr.X + 10, fr.Y + 147, cw - 20.0f, 36), gcol(165, 195, 230), 10.5f, gp::FontStyleRegular, gp::StringAlignmentNear, gp::StringAlignmentNear, true);
        g_start_hits.push_back({r, 1, (int)i});
        if (++col >= cols) { col = 0; y += ch + gap; }
    }
    gp::Graphics out(dc); out.DrawImage(&buf, 0, 0);
}

static void set_doc(const js::Value& d, const std::string& path);
static void wf_set_project(const std::string& dir);
static void start_action(int kind, int idx);

static LRESULT CALLBACK start_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_PAINT: { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_start(h, dc); EndPaint(h, &ps); return 0; }
        case WM_ERASEBKGND: return 1;
        case WM_MOUSEMOVE: {
            int x = (short)LOWORD(l), y = (short)HIWORD(l), hit = -1;
            for (auto& t : g_start_hits) if (x >= t.r.left && x < t.r.right && y >= t.r.top && y < t.r.bottom) hit = t.kind == 0 ? 1000 + t.idx : t.idx;
            if (hit != g_start_hover) { g_start_hover = hit; InvalidateRect(h, nullptr, FALSE); }
            TRACKMOUSEEVENT te = {sizeof te, TME_LEAVE, h, 0}; TrackMouseEvent(&te);
            return 0;
        }
        case WM_MOUSELEAVE: g_start_hover = -1; InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_LBUTTONUP: {
            int x = (short)LOWORD(l), y = (short)HIWORD(l);
            for (auto& t : g_start_hits) if (x >= t.r.left && x < t.r.right && y >= t.r.top && y < t.r.bottom) { start_action(t.kind, t.idx); return 0; }
            return 0;
        }
        case WM_MOUSEWHEEL: g_start_scroll = std::max(0, g_start_scroll - (short)HIWORD(w) / 2); InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_KEYDOWN: if (w == VK_ESCAPE) ShowWindow(h, SW_HIDE); return 0;
        case WM_CLOSE: ShowWindow(h, SW_HIDE); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

static void open_start() {
    tpl_load();
    if (!g_start) {
        RECT mr; GetWindowRect(g_main, &mr);
        int W = 1240, H = 760;
        g_start = CreateWindowExA(WS_EX_TOOLWINDOW, "AeroSynapseStart", "Start: AeroSynapse Studio", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME,
                                  mr.left + std::max(0L, (mr.right - mr.left - W) / 2), mr.top + 40, W, H, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
    }
    ShowWindow(g_start, SW_SHOW); SetForegroundWindow(g_start); SetFocus(g_start);
}

static js::Value blank_doc() { js::Value d = js::Value::object(); d["name"] = "new design"; d["units"] = "m"; d["features"] = js::Value::array(); return d; }

// a new project folder parent/name: design.json plus the eight-stage workflow project, then opened in Studio
static bool make_project_folder(const std::string& parent, const std::string& name) {
    for (char c : name) if (!isalnum((unsigned char)c) && c != '-' && c != '_') { MessageBoxA(g_main, "A project name uses letters, digits, - and _ only.", "New project", MB_ICONWARNING); return false; }
    std::string dir = parent + "\\" + name;
    if (name.empty() || (!CreateDirectoryA(dir.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS)) {
        MessageBoxA(g_main, ("Cannot create the folder " + dir).c_str(), "New project", MB_ICONERROR); return false;
    }
    js::Value d = blank_doc(); d["name"] = name;
    js::write_file(dir + "\\design.json", js::dump(d, 2));
    std::string cmd = "cmd /c \"\"" + python() + "\" -m aerosynapse.cli workflow new " + name + " --from \"" + dir + "\\design.json\" --where \"" + parent + "\" --overwrite > \"" + dir + "\\new.log\" 2>&1\"";
    SetEnvironmentVariableA("PYTHONPATH", (g_root + "\\src").c_str());
    STARTUPINFOA si; memset(&si, 0, sizeof si); si.cb = sizeof si; PROCESS_INFORMATION pi;
    if (CreateProcessA(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, g_root.c_str(), &si, &pi)) { WaitForSingleObject(pi.hProcess, 60000); CloseHandle(pi.hProcess); CloseHandle(pi.hThread); }
    set_doc(d, dir + "\\design.json");
    bool wf = file_exists(dir + "\\project.json");
    if (wf) g_wf_dir = dir;
    if (g_start) ShowWindow(g_start, SW_HIDE);
    status("new project folder: " + dir + (wf ? "  (design.json and the workflow project are there; sketch your body with the Sketcher)" : "  (design.json written; the workflow project failed, see new.log)"));
    return wf;
}

static void new_in_folder() {
    BROWSEINFOA bi; memset(&bi, 0, sizeof bi);
    char disp[MAX_PATH] = "";
    bi.hwndOwner = g_start && IsWindowVisible(g_start) ? g_start : g_main; bi.pszDisplayName = disp;
    bi.lpszTitle = "Choose the folder in which the new project folder is created"; bi.ulFlags = BIF_RETURNONLYFSDIRS;
    LPITEMIDLIST pid = SHBrowseForFolderA(&bi);
    if (!pid) return;
    char parent[MAX_PATH] = "";
    SHGetPathFromIDListA(pid, parent);
    CoTaskMemFree(pid);
    std::string name = "my_aircraft";
    if (!ask_text(bi.hwndOwner, "New project in a new folder", "Name of the new project folder (letters, digits, - and _):", name)) return;
    make_project_folder(parent, name);
}

static void start_action(int kind, int idx) {
    if (kind == 1 && idx >= 0 && idx < (int)g_tpl.size()) {
        try {
            js::Value d = js::parse(js::read_file(g_root + "\\assets\\templates\\" + g_tpl[(size_t)idx].key + ".json"));
            set_doc(d, "");
            status("template: " + g_tpl[(size_t)idx].title + ". Save it (Ctrl+S) under your own name.");
        } catch (std::exception& e) { MessageBoxA(g_start, e.what(), "Template", MB_ICONERROR); return; }
        ShowWindow(g_start, SW_HIDE);
        return;
    }
    if (idx == 0) { set_doc(blank_doc(), ""); ShowWindow(g_start, SW_HIDE); status("blank design: pick the Sketcher workbench and a plane to draw your first body"); }
    else if (idx == 1) new_in_folder();
    else if (idx == 2) { ShowWindow(g_start, SW_HIDE); SendMessageA(g_main, WM_COMMAND, ID_FILE_OPEN, 0); }
    else { ShowWindow(g_start, SW_HIDE); open_workflow(); wf_open(); }
}

static void register_workspace_classes(HINSTANCE hi, HICON icon) {
    WNDCLASSA a; memset(&a, 0, sizeof a);
    a.hInstance = hi; a.hCursor = LoadCursor(nullptr, IDC_ARROW); a.hIcon = icon;
    a.lpfnWndProc = card_proc; a.lpszClassName = "AeroSynapseCard"; RegisterClassA(&a);
    a.lpfnWndProc = tip_proc; a.lpszClassName = "AeroSynapseTip"; RegisterClassA(&a);
    a.lpfnWndProc = start_proc; a.lpszClassName = "AeroSynapseStart"; a.hCursor = LoadCursor(nullptr, IDC_HAND); RegisterClassA(&a);
    a.lpfnWndProc = par_proc; a.lpszClassName = "AeroSynapseParams"; a.hCursor = LoadCursor(nullptr, IDC_ARROW); a.hbrBackground = g_br_panel; RegisterClassA(&a);
}
