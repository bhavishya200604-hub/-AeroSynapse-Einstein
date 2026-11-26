// 2D drafting, in the manner of AutoCAD: a model space in millimetres with layers (OUTLINE, HIDDEN, CENTER, DIM, TEXT,
// BORDER), lines, polylines, rectangles, circles, arcs, text and linear/radius dimensions, drawn with the mouse or typed
// on the command line with AutoCAD's aliases and coordinate input:
//   L line, PL polyline, REC rectangle, C circle, A arc (3 points), DLI linear dimension, DRA radius dimension,
//   T text, E erase, M move, CO copy, MI mirror, O offset (lines, circles), Z zoom extents, U undo, REDO,
//   VIEWS (the design in three views and an isometric, hidden lines dashed, on an A3 sheet with a title block),
//   FLAT (the flat pattern of the selected bent plate, with bend lines), SKETCH (the largest closed outline to the
//   3D sketcher: pad it), DXFOUT / DXFIN (AutoCAD R12 DXF), PNGOUT.
//   Points: click, or type x,y (absolute), @dx,dy (relative), @len<angle (polar), or a single length along the
//   cursor's direction. F3 object snap (end, mid, centre), F8 ortho, F9 grid snap. Enter repeats or ends a command,
//   Esc cancels, the wheel zooms, a middle drag pans, a click picks an entity (Shift adds), Delete erases the picked.

struct DEnt { int type = 0, layer = 1; std::vector<double> p; std::string text; double h = 3.5; bool closed = false; };
enum { DE_LINE, DE_CIRCLE, DE_ARC, DE_PLINE, DE_TEXT, DE_DIMLIN, DE_DIMRAD };
struct DLayer { std::string name; int aci; int ltype; };  // ltype 0 continuous, 1 hidden, 2 centre
static std::vector<DLayer> g_dl = {{"0", 7, 0}, {"OUTLINE", 7, 0}, {"HIDDEN", 4, 1}, {"CENTER", 1, 2}, {"DIM", 3, 0}, {"TEXT", 2, 0}, {"BORDER", 8, 0}};
static std::vector<DEnt> g_de;
static std::vector<std::vector<DEnt>> g_de_undo, g_de_redo;
static std::set<int> g_dsel;
static HWND g_dr, g_dr_canvas, g_dr_cmd, g_dr_layer, g_dr_prompt;
static double g_dcx = 210, g_dcy = 148, g_dscale = 2.2;  // view centre (mm) and pixels per mm
static bool g_dortho = false, g_dsnap = false, g_dosnap = true, g_dpan = false;
static int g_dlayer = 1, g_dmx = 0, g_dmy = 0;
static double g_dwx = 0, g_dwy = 0;  // the cursor in model space
static int g_dpx = 0, g_dpy = 0;  // the pan's last mouse position
static std::string g_dtool, g_dlast_tool, g_dmsg = "Type a command (L, PL, REC, C, A, DLI, DRA, T, E, M, CO, MI, O, Z, VIEWS, FLAT, DXFIN, DXFOUT) or pick a tool on the left.";
static std::vector<geo::P2> g_dpts;  // the points of the command in progress
static bool g_dsnap_hit = false; static int g_dsnap_kind = 0; static double g_dsnx = 0, g_dsny = 0;
static std::string g_dfile;  // the DXF last opened or saved
enum { ID_DR_CMD = 8900, ID_DR_LAYER, ID_DR_T0 = 8920 };
static const char* DR_TOOLS[][3] = {
    {"LINE", "L", "Line"}, {"PLINE", "PL", "Polyline"}, {"RECTANG", "REC", "Rectangle"}, {"CIRCLE", "C", "Circle"}, {"ARC", "A", "Arc (3 points)"},
    {"DIMLINEAR", "DLI", "Linear dimension"}, {"DIMRADIUS", "DRA", "Radius dimension"}, {"TEXT", "T", "Text"},
    {"ERASE", "E", "Erase"}, {"MOVE", "M", "Move"}, {"COPY", "CO", "Copy"}, {"MIRROR", "MI", "Mirror"}, {"OFFSET", "O", "Offset"},
    {"ZOOM", "Z", "Zoom extents"}, {"UNDO", "U", "Undo"}, {"REDO", "REDO", "Redo"},
    {"VIEWS", "VIEWS", "3 views of the design"}, {"FLAT", "FLAT", "Sheet-metal flat pattern"}, {"SKETCH", "SKETCH", "Outline to the 3D sketcher"},
    {"NEW", "NEW", "New drawing"}, {"DXFIN", "DXFIN", "Open DXF..."}, {"DXFOUT", "DXFOUT", "Save DXF..."}, {"SVGOUT", "SVGOUT", "Save SVG (vector)..."},
    {"PDFOUT", "PDFOUT", "Save PDF (vector)..."}, {"PNGOUT", "PNGOUT", "Save picture (3x)..."}};
static const int NDRT = (int)(sizeof(DR_TOOLS) / sizeof(DR_TOOLS[0]));

static gp::Color aci_col(int aci) {
    switch (aci) { case 1: return gcol(255, 70, 70); case 2: return gcol(255, 230, 60); case 3: return gcol(70, 230, 110); case 4: return gcol(70, 220, 240);
                   case 5: return gcol(90, 130, 255); case 6: return gcol(240, 90, 240); case 8: return gcol(140, 150, 165); default: return gcol(235, 240, 248); }
}
static void dr_push_undo() { g_de_undo.push_back(g_de); if (g_de_undo.size() > 60) g_de_undo.erase(g_de_undo.begin()); g_de_redo.clear(); }
static int g_dr_fw2 = 0, g_dr_fh2 = 0;
static gp::PointF dr_scr(double x, double y) {
    RECT rc; GetClientRect(g_dr_canvas, &rc); if (g_dr_fw2 > 0) { rc.right = g_dr_fw2; rc.bottom = g_dr_fh2; }
    return gp::PointF((float)(rc.right / 2.0 + (x - g_dcx) * g_dscale), (float)(rc.bottom / 2.0 - (y - g_dcy) * g_dscale));
}
static void dr_world(int sx, int sy, double& x, double& y) {
    RECT rc; GetClientRect(g_dr_canvas, &rc); if (g_dr_fw2 > 0) { rc.right = g_dr_fw2; rc.bottom = g_dr_fh2; }
    x = g_dcx + (sx - rc.right / 2.0) / g_dscale; y = g_dcy - (sy - rc.bottom / 2.0) / g_dscale;
}
static void dr_add(int type, std::vector<double> p, int layer = -1, const std::string& text = "", double h = 3.5, bool closed = false) {
    DEnt e; e.type = type; e.p = p; e.layer = layer < 0 ? g_dlayer : layer; e.text = text; e.h = h; e.closed = closed; g_de.push_back(e);
}
static int dr_layer_of(const std::string& n) {
    for (size_t i = 0; i < g_dl.size(); ++i) if (_stricmp(g_dl[i].name.c_str(), n.c_str()) == 0) return (int)i;
    g_dl.push_back(DLayer{n, 7, 0});
    if (g_dr_layer) SendMessageA(g_dr_layer, CB_ADDSTRING, 0, (LPARAM)n.c_str());
    return (int)g_dl.size() - 1;
}

// ---------------------------------------------------------------- geometry of the entities
static void dr_bounds(const DEnt& e, double& x0, double& y0, double& x1, double& y1) {
    auto put = [&](double x, double y) { x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y); };
    switch (e.type) {
        case DE_CIRCLE: case DE_ARC: put(e.p[0] - e.p[2], e.p[1] - e.p[2]); put(e.p[0] + e.p[2], e.p[1] + e.p[2]); break;
        case DE_TEXT: put(e.p[0], e.p[1]); put(e.p[0] + 0.7 * e.h * e.text.size(), e.p[1] + e.h); break;
        default: for (size_t i = 0; i + 1 < e.p.size(); i += 2) put(e.p[i], e.p[i + 1]);
    }
}
static double seg_dist(double px, double py, double ax, double ay, double bx, double by) {
    double dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy, t = L2 > 0 ? std::max(0.0, std::min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2)) : 0;
    return std::hypot(px - ax - t * dx, py - ay - t * dy);
}
static bool ang_in(double a, double a0, double a1) { auto n = [](double v) { v = std::fmod(v, 360.0); return v < 0 ? v + 360 : v; }; a = n(a - a0); return a <= n(a1 - a0) + 1e-9; }
static double dr_dist(const DEnt& e, double x, double y) {
    switch (e.type) {
        case DE_LINE: case DE_DIMLIN: return seg_dist(x, y, e.p[0], e.p[1], e.p[2], e.p[3]);
        case DE_CIRCLE: return std::fabs(std::hypot(x - e.p[0], y - e.p[1]) - e.p[2]);
        case DE_ARC: { double a = std::atan2(y - e.p[1], x - e.p[0]) * 180 / geo::PI; if (!ang_in(a, e.p[3], e.p[4])) return 1e9; return std::fabs(std::hypot(x - e.p[0], y - e.p[1]) - e.p[2]); }
        case DE_DIMRAD: return seg_dist(x, y, e.p[0], e.p[1], e.p[0] + e.p[2] * std::cos(e.p[3]), e.p[1] + e.p[2] * std::sin(e.p[3]));
        case DE_TEXT: { double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300; dr_bounds(e, x0, y0, x1, y1); return (x >= x0 && x <= x1 && y >= y0 && y <= y1) ? 0 : 1e9; }
        case DE_PLINE: {
            double d = 1e300; size_t n = e.p.size() / 2;
            for (size_t i = 0; i + 1 < n; ++i) d = std::min(d, seg_dist(x, y, e.p[2 * i], e.p[2 * i + 1], e.p[2 * i + 2], e.p[2 * i + 3]));
            if (e.closed && n > 2) d = std::min(d, seg_dist(x, y, e.p[2 * n - 2], e.p[2 * n - 1], e.p[0], e.p[1]));
            return d;
        }
    }
    return 1e9;
}
static int dr_pick(double x, double y) {
    int best = -1; double bd = 7 / g_dscale;
    for (size_t i = 0; i < g_de.size(); ++i) { double d = dr_dist(g_de[i], x, y); if (d < bd) { bd = d; best = (int)i; } }
    return best;
}
static void dr_transform(DEnt& e, double ox, double oy, bool mirror, double mx0 = 0, double my0 = 0, double mx1 = 0, double my1 = 0) {
    auto mir = [&](double& x, double& y) {
        double dx = mx1 - mx0, dy = my1 - my0, L2 = dx * dx + dy * dy; if (L2 <= 0) return;
        double t = ((x - mx0) * dx + (y - my0) * dy) / L2, fx = mx0 + t * dx, fy = my0 + t * dy; x = 2 * fx - x; y = 2 * fy - y;
    };
    if (!mirror) {
        if (e.type == DE_CIRCLE || e.type == DE_ARC || e.type == DE_DIMRAD || e.type == DE_TEXT) { e.p[0] += ox; e.p[1] += oy; }
        else for (size_t i = 0; i + 1 < e.p.size(); i += 2) { e.p[i] += ox; e.p[i + 1] += oy; }
        return;
    }
    if (e.type == DE_ARC) {
        double a0 = e.p[3] * geo::PI / 180, a1 = e.p[4] * geo::PI / 180;
        double sx = e.p[0] + e.p[2] * std::cos(a0), sy = e.p[1] + e.p[2] * std::sin(a0), ex = e.p[0] + e.p[2] * std::cos(a1), ey = e.p[1] + e.p[2] * std::sin(a1);
        mir(e.p[0], e.p[1]); mir(sx, sy); mir(ex, ey);
        e.p[3] = std::atan2(ey - e.p[1], ex - e.p[0]) * 180 / geo::PI; e.p[4] = std::atan2(sy - e.p[1], sx - e.p[0]) * 180 / geo::PI;  // reversed sense
    } else if (e.type == DE_CIRCLE || e.type == DE_TEXT) mir(e.p[0], e.p[1]);
    else if (e.type == DE_DIMRAD) { double ex = e.p[0] + std::cos(e.p[3]), ey = e.p[1] + std::sin(e.p[3]); mir(e.p[0], e.p[1]); mir(ex, ey); e.p[3] = std::atan2(ey - e.p[1], ex - e.p[0]); }
    else for (size_t i = 0; i + 1 < e.p.size(); i += 2) mir(e.p[i], e.p[i + 1]);
}

// object snap: ends, midpoints and centres near the cursor
static bool dr_osnap(double x, double y, double& sx, double& sy, int& kind) {
    double bd = 11 / g_dscale; bool hit = false;
    auto test = [&](double px, double py, int k) { double d = std::hypot(px - x, py - y); if (d < bd) { bd = d; sx = px; sy = py; kind = k; hit = true; } };
    for (const DEnt& e : g_de) {
        switch (e.type) {
            case DE_LINE: test(e.p[0], e.p[1], 0); test(e.p[2], e.p[3], 0); test(0.5 * (e.p[0] + e.p[2]), 0.5 * (e.p[1] + e.p[3]), 1); break;
            case DE_CIRCLE: test(e.p[0], e.p[1], 2); for (int q = 0; q < 4; ++q) test(e.p[0] + e.p[2] * std::cos(q * geo::PI / 2), e.p[1] + e.p[2] * std::sin(q * geo::PI / 2), 3); break;
            case DE_ARC: test(e.p[0], e.p[1], 2); test(e.p[0] + e.p[2] * std::cos(e.p[3] * geo::PI / 180), e.p[1] + e.p[2] * std::sin(e.p[3] * geo::PI / 180), 0); test(e.p[0] + e.p[2] * std::cos(e.p[4] * geo::PI / 180), e.p[1] + e.p[2] * std::sin(e.p[4] * geo::PI / 180), 0); break;
            case DE_PLINE: {
                size_t n = e.p.size() / 2;
                for (size_t i = 0; i < n; ++i) { test(e.p[2 * i], e.p[2 * i + 1], 0); size_t j = (i + 1) % n; if (i + 1 < n || e.closed) test(0.5 * (e.p[2 * i] + e.p[2 * j]), 0.5 * (e.p[2 * i + 1] + e.p[2 * j + 1]), 1); }
                break;
            }
            default: break;
        }
    }
    for (const geo::P2& q : g_dpts) test(q.x, q.y, 0);
    return hit;
}

// the point the cursor gives now: object snap, grid snap and ortho applied
static void dr_cursor_point(double& x, double& y) {
    x = g_dwx; y = g_dwy;
    if (g_dosnap && g_dsnap_hit) { x = g_dsnx; y = g_dsny; return; }
    if (g_dsnap) { double s = g_dscale > 4 ? 1 : g_dscale > 1 ? 5 : 10; x = std::round(x / s) * s; y = std::round(y / s) * s; }
    if (g_dortho && !g_dpts.empty()) { const geo::P2& a = g_dpts.back(); if (std::fabs(x - a.x) > std::fabs(y - a.y)) y = a.y; else x = a.x; }
}

// ---------------------------------------------------------------- painting
static void dr_dash(gp::Pen& pen, int lt) {
    if (lt == 1) { gp::REAL d[2] = {6, 4}; pen.SetDashPattern(d, 2); }
    else if (lt == 2) { gp::REAL d[4] = {14, 3, 3, 3}; pen.SetDashPattern(d, 4); }
}
static void dr_arrow(gp::Graphics& g, gp::Brush& b, gp::PointF tip, double ang, float s) {
    gp::PointF P[3] = {tip, gp::PointF(tip.X - s * (float)std::cos(ang - 0.3), tip.Y + s * (float)std::sin(ang - 0.3)), gp::PointF(tip.X - s * (float)std::cos(ang + 0.3), tip.Y + s * (float)std::sin(ang + 0.3))};
    g.FillPolygon(&b, P, 3);
}
static void dr_text(gp::Graphics& g, const std::string& t, double x, double y, double h, double rot_deg, gp::Color c, bool centre = false) {
    float px = (float)std::max(6.0, h * g_dscale * 1.25);
    gp::FontFamily ff(L"Consolas"); gp::Font f(&ff, px, gp::FontStyleRegular, gp::UnitPixel);
    gp::SolidBrush b(c);
    gp::PointF p = dr_scr(x, y);
    gp::GraphicsState st = g.Save();
    g.TranslateTransform(p.X, p.Y); g.RotateTransform((gp::REAL)-rot_deg);
    std::wstring w = wide(t);
    gp::StringFormat sf; if (centre) sf.SetAlignment(gp::StringAlignmentCenter);
    g.DrawString(w.c_str(), (INT)w.size(), &f, gp::PointF(0, -px * 1.05f), &sf, &b);
    g.Restore(st);
}
static void dr_dim_geom(const DEnt& e, double& ax, double& ay, double& bx, double& by, double& nx, double& ny) {
    // a linear dimension: measured points (p0, p1), and the dimension line through p2 parallel to them (aligned)
    double dx = e.p[2] - e.p[0], dy = e.p[3] - e.p[1], L = std::max(1e-12, std::hypot(dx, dy));
    nx = -dy / L; ny = dx / L;
    double off = (e.p.size() >= 6) ? (e.p[4] - e.p[0]) * nx + (e.p[5] - e.p[1]) * ny : 10;
    ax = e.p[0] + nx * off; ay = e.p[1] + ny * off; bx = e.p[2] + nx * off; by = e.p[3] + ny * off;
}
static void dr_draw_ent(gp::Graphics& g, const DEnt& e, bool sel, gp::Color* force = nullptr) {
    const DLayer& L = g_dl[(size_t)std::max(0, std::min((int)g_dl.size() - 1, e.layer))];
    gp::Color c = force ? *force : sel ? gcol(80, 170, 255) : aci_col(L.aci);
    gp::Pen pen(c, sel ? 2.2f : (e.layer == 1 ? 1.6f : 1.1f)); dr_dash(pen, L.ltype);
    gp::SolidBrush br(c);
    switch (e.type) {
        case DE_LINE: g.DrawLine(&pen, dr_scr(e.p[0], e.p[1]), dr_scr(e.p[2], e.p[3])); break;
        case DE_CIRCLE: { gp::PointF c0 = dr_scr(e.p[0], e.p[1]); float r = (float)(e.p[2] * g_dscale); g.DrawEllipse(&pen, c0.X - r, c0.Y - r, 2 * r, 2 * r); break; }
        case DE_ARC: {
            gp::PointF c0 = dr_scr(e.p[0], e.p[1]); float r = (float)(e.p[2] * g_dscale);
            double sw = std::fmod(e.p[4] - e.p[3] + 720.0, 360.0); if (sw < 1e-9) sw = 360;
            if (r > 0.5f) g.DrawArc(&pen, c0.X - r, c0.Y - r, 2 * r, 2 * r, (gp::REAL)-e.p[3], (gp::REAL)-sw);
            break;
        }
        case DE_PLINE: {
            std::vector<gp::PointF> P; for (size_t i = 0; i + 1 < e.p.size(); i += 2) P.push_back(dr_scr(e.p[i], e.p[i + 1]));
            if (P.size() >= 2) { if (e.closed) g.DrawPolygon(&pen, P.data(), (INT)P.size()); else g.DrawLines(&pen, P.data(), (INT)P.size()); }
            break;
        }
        case DE_TEXT: dr_text(g, e.text, e.p[0], e.p[1], e.h, e.p.size() > 2 ? e.p[2] : 0, c); break;
        case DE_DIMLIN: {
            double ax, ay, bx, by, nx, ny; dr_dim_geom(e, ax, ay, bx, by, nx, ny);
            gp::Pen thin(c, 1.0f);
            double ex = 2.0;  // extension past the dimension line, mm
            double sgn = ((ax - e.p[0]) * nx + (ay - e.p[1]) * ny) >= 0 ? 1 : -1;
            g.DrawLine(&thin, dr_scr(e.p[0] + nx * sgn * 1.0, e.p[1] + ny * sgn * 1.0), dr_scr(ax + nx * sgn * ex, ay + ny * sgn * ex));
            g.DrawLine(&thin, dr_scr(e.p[2] + nx * sgn * 1.0, e.p[3] + ny * sgn * 1.0), dr_scr(bx + nx * sgn * ex, by + ny * sgn * ex));
            g.DrawLine(&thin, dr_scr(ax, ay), dr_scr(bx, by));
            gp::PointF A = dr_scr(ax, ay), B = dr_scr(bx, by);
            double an = std::atan2(-(B.Y - A.Y), B.X - A.X);
            float s = (float)std::max(5.0, 2.5 * g_dscale);
            dr_arrow(g, br, B, an, s); dr_arrow(g, br, A, an + geo::PI, s);
            double ang = std::atan2(by - ay, bx - ax) * 180 / geo::PI; if (ang > 90.01 || ang < -89.99) ang += 180;
            double mx = 0.5 * (ax + bx) + nx * sgn * 0.8, my = 0.5 * (ay + by) + ny * sgn * 0.8;
            std::string t = e.text.empty() ? fmtd(std::round(std::hypot(e.p[2] - e.p[0], e.p[3] - e.p[1]) * 100) / 100) : e.text;
            dr_text(g, t, mx, my, e.h, ang, c, true);
            break;
        }
        case DE_DIMRAD: {
            double ex = e.p[0] + e.p[2] * std::cos(e.p[3]), ey = e.p[1] + e.p[2] * std::sin(e.p[3]);
            gp::Pen thin(c, 1.0f); g.DrawLine(&thin, dr_scr(e.p[0], e.p[1]), dr_scr(ex, ey));
            gp::PointF A = dr_scr(e.p[0], e.p[1]), B = dr_scr(ex, ey);
            dr_arrow(g, br, B, std::atan2(-(B.Y - A.Y), B.X - A.X), (float)std::max(5.0, 2.5 * g_dscale));
            std::string t = e.text.empty() ? "R" + fmtd(std::round(e.p[2] * 100) / 100) : e.text;
            dr_text(g, t, ex + 1.5 * std::cos(e.p[3]), ey + 1.5 * std::sin(e.p[3]), e.h, 0, c);
            break;
        }
    }
}

static void dr_rubber(gp::Graphics& g) {  // what the command in progress would add
    if (g_dtool.empty() || g_dpts.empty()) return;
    double x, y; dr_cursor_point(x, y);
    gp::Color c = gcol(255, 200, 80);
    const geo::P2& a = g_dpts.back();
    DEnt e; e.layer = g_dlayer;
    if (g_dtool == "LINE" || g_dtool == "PLINE") { e.type = DE_LINE; e.p = {a.x, a.y, x, y}; }
    else if (g_dtool == "RECTANG") { e.type = DE_PLINE; e.closed = true; e.p = {a.x, a.y, x, a.y, x, y, a.x, y}; }
    else if (g_dtool == "CIRCLE") { e.type = DE_CIRCLE; e.p = {a.x, a.y, std::hypot(x - a.x, y - a.y)}; }
    else if (g_dtool == "ARC" && g_dpts.size() == 1) { e.type = DE_LINE; e.p = {a.x, a.y, x, y}; }
    else if (g_dtool == "DIMLINEAR") { e.type = g_dpts.size() == 1 ? DE_LINE : DE_DIMLIN; e.p = g_dpts.size() == 1 ? std::vector<double>{a.x, a.y, x, y} : std::vector<double>{g_dpts[0].x, g_dpts[0].y, g_dpts[1].x, g_dpts[1].y, x, y}; e.layer = 4; }
    else if (g_dtool == "MOVE" || g_dtool == "COPY") {
        for (int i : g_dsel) if (i < (int)g_de.size()) { DEnt q = g_de[(size_t)i]; dr_transform(q, x - a.x, y - a.y, false); dr_draw_ent(g, q, false, &c); }
        e.type = DE_LINE; e.p = {a.x, a.y, x, y};
    } else if (g_dtool == "MIRROR") {
        for (int i : g_dsel) if (i < (int)g_de.size()) { DEnt q = g_de[(size_t)i]; dr_transform(q, 0, 0, true, a.x, a.y, x, y); dr_draw_ent(g, q, false, &c); }
        e.type = DE_LINE; e.p = {a.x, a.y, x, y};
    } else return;
    if (g_dtool == "PLINE" && g_dpts.size() >= 2) { std::vector<gp::PointF> P; for (auto& q : g_dpts) P.push_back(dr_scr(q.x, q.y)); gp::Pen pp(c, 1.4f); g.DrawLines(&pp, P.data(), (INT)P.size()); }
    dr_draw_ent(g, e, false, &c);
    if (e.type == DE_LINE) {  // length and angle beside the cursor, as AutoCAD's dynamic input
        char b[80]; snprintf(b, sizeof b, "%.2f < %.1f", std::hypot(x - a.x, y - a.y), std::atan2(y - a.y, x - a.x) * 180 / geo::PI);
        gp::PointF p = dr_scr(x, y); gtext(g, b, gp::RectF(p.X + 14, p.Y + 10, 150, 18), gcol(255, 220, 120), 11.0f);
    }
}

static int g_dr_fw = 0, g_dr_fh = 0;  // a forced size: the picture export
static void paint_draft(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    int W = std::max(1L, rc.right), H = std::max(1L, rc.bottom);
    if (g_dr_fw > 0) { W = g_dr_fw; H = g_dr_fh; }
    gp::Bitmap buf(W, H, PixelFormat32bppARGB); gp::Graphics g(&buf); quality(g);
    gp::SolidBrush bg(gcol(33, 40, 48)); g.FillRectangle(&bg, 0, 0, W, H);
    {   // the grid: dots every 10 mm (or 100 mm when zoomed out), a stronger line every fifth
        double step = g_dscale > 1.5 ? 10 : g_dscale > 0.15 ? 100 : 1000;
        double x0, y0, x1, y1; dr_world(0, H, x0, y0); dr_world(W, 0, x1, y1);
        gp::Pen gl(gcol(60, 70, 82), 1.0f), gl5(gcol(74, 86, 100), 1.0f);
        if ((x1 - x0) / step < 300) {
            for (double x = std::floor(x0 / step) * step; x <= x1; x += step) { gp::PointF a = dr_scr(x, y0); bool k = std::fmod(std::fabs(x), 5 * step) < 1e-6; g.DrawLine(k ? &gl5 : &gl, a.X, 0.0f, a.X, (float)H); }
            for (double y = std::floor(y0 / step) * step; y <= y1; y += step) { gp::PointF a = dr_scr(x0, y); bool k = std::fmod(std::fabs(y), 5 * step) < 1e-6; g.DrawLine(k ? &gl5 : &gl, 0.0f, a.Y, (float)W, a.Y); }
        }
        gp::PointF o = dr_scr(0, 0);  // the origin's axes (the UCS icon)
        gp::Pen xr(gcol(230, 80, 80), 2.0f), yg(gcol(80, 220, 110), 2.0f);
        g.DrawLine(&xr, o.X, o.Y, o.X + 40, o.Y); g.DrawLine(&yg, o.X, o.Y, o.X, o.Y - 40);
        gtext(g, "X", gp::RectF(o.X + 42, o.Y - 9, 20, 18), gcol(230, 80, 80), 11.0f, gp::FontStyleBold);
        gtext(g, "Y", gp::RectF(o.X - 5, o.Y - 60, 20, 18), gcol(80, 220, 110), 11.0f, gp::FontStyleBold);
    }
    for (size_t i = 0; i < g_de.size(); ++i) dr_draw_ent(g, g_de[i], g_dsel.count((int)i) > 0);
    dr_rubber(g);
    if (g_dosnap && g_dsnap_hit) {  // the snap marker: square end, triangle middle, circle centre, diamond quadrant
        gp::PointF p = dr_scr(g_dsnx, g_dsny); gp::Pen sp(gcol(80, 255, 120), 2.0f); float s = 7;
        if (g_dsnap_kind == 0) g.DrawRectangle(&sp, p.X - s, p.Y - s, 2 * s, 2 * s);
        else if (g_dsnap_kind == 1) { gp::PointF T[3] = {gp::PointF(p.X, p.Y - s), gp::PointF(p.X + s, p.Y + s), gp::PointF(p.X - s, p.Y + s)}; g.DrawPolygon(&sp, T, 3); }
        else if (g_dsnap_kind == 2) g.DrawEllipse(&sp, p.X - s, p.Y - s, 2 * s, 2 * s);
        else { gp::PointF T[4] = {gp::PointF(p.X, p.Y - s), gp::PointF(p.X + s, p.Y), gp::PointF(p.X, p.Y + s), gp::PointF(p.X - s, p.Y)}; g.DrawPolygon(&sp, T, 4); }
        static const char* SN[4] = {"Endpoint", "Midpoint", "Center", "Quadrant"};
        gtext(g, SN[g_dsnap_kind], gp::RectF(p.X + 10, p.Y - 24, 90, 16), gcol(80, 255, 120), 10.5f);
    }
    if (g_dr_fw > 0) { gp::Graphics out(dc); out.DrawImage(&buf, 0, 0); return; }  // the picture: no cursor, no status bar
    {   // the crosshair over the whole canvas, as in AutoCAD
        gp::Pen ch(gcol(220, 225, 235, 150), 1.0f);
        g.DrawLine(&ch, 0.0f, (float)g_dmy, (float)W, (float)g_dmy); g.DrawLine(&ch, (float)g_dmx, 0.0f, (float)g_dmx, (float)H);
        gp::Pen bx(gcol(240, 240, 250), 1.0f); g.DrawRectangle(&bx, g_dmx - 5, g_dmy - 5, 10, 10);
    }
    {   // the status bar: the cursor, the drawing aids and the entity count
        gp::SolidBrush sb(gcol(20, 26, 33)); g.FillRectangle(&sb, 0, H - 24, W, 24);
        char b[260];
        snprintf(b, sizeof b, "%.2f, %.2f mm   |   %s  %s  %s   |   layer %s   |   %d entities%s%s", g_dwx, g_dwy, g_dosnap ? "OSNAP" : "osnap", g_dortho ? "ORTHO" : "ortho", g_dsnap ? "SNAP" : "snap",
                 g_dl[(size_t)g_dlayer].name.c_str(), (int)g_de.size(), g_dsel.empty() ? "" : (", " + tostr((int)g_dsel.size()) + " picked").c_str(), g_dfile.empty() ? "" : ("   |   " + g_dfile).c_str());
        gtext(g, b, gp::RectF(10, (float)H - 23, (float)W - 20, 22), gcol(150, 200, 255), 11.0f);
        gp::LinearGradientBrush acc(gp::Rect(0, H - 26, W, 2), gcol(236, 72, 153), gcol(34, 211, 238), gp::LinearGradientModeHorizontal);
        g.FillRectangle(&acc, 0, H - 26, W, 2);
    }
    gp::Graphics out(dc); out.DrawImage(&buf, 0, 0);
}

// ---------------------------------------------------------------- zoom, prompts
static void dr_zoom_extents() {
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (const DEnt& e : g_de) dr_bounds(e, x0, y0, x1, y1);
    if (x1 < x0) { x0 = 0; y0 = 0; x1 = 420; y1 = 297; }
    RECT rc; GetClientRect(g_dr_canvas, &rc);
    g_dcx = 0.5 * (x0 + x1); g_dcy = 0.5 * (y0 + y1);
    g_dscale = std::min((rc.right - 60) / std::max(1e-6, x1 - x0), (rc.bottom - 80) / std::max(1e-6, y1 - y0));
    g_dscale = std::max(1e-4, std::min(1e4, g_dscale));
}
static void dr_prompt(const std::string& s) { g_dmsg = s; if (g_dr_prompt) SetWindowTextA(g_dr_prompt, s.c_str()); }
static void dr_refresh() { if (g_dr_canvas) InvalidateRect(g_dr_canvas, nullptr, FALSE); }

// ---------------------------------------------------------------- DXF (AutoCAD R12, ASCII)
static void dxf_pair(std::string& s, int code, const std::string& v) { s += tostr(code) + "\n" + v + "\n"; }
static std::string dxf_num(double v) { char b[40]; snprintf(b, sizeof b, "%.6f", v); return b; }
static void dxf_line(std::string& s, const std::string& L, double x1, double y1, double x2, double y2) {
    dxf_pair(s, 0, "LINE"); dxf_pair(s, 8, L); dxf_pair(s, 10, dxf_num(x1)); dxf_pair(s, 20, dxf_num(y1)); dxf_pair(s, 30, "0.0"); dxf_pair(s, 11, dxf_num(x2)); dxf_pair(s, 21, dxf_num(y2)); dxf_pair(s, 31, "0.0");
}
static void dxf_text(std::string& s, const std::string& L, double x, double y, double h, double rot, const std::string& t, int halign = 0) {
    dxf_pair(s, 0, "TEXT"); dxf_pair(s, 8, L); dxf_pair(s, 10, dxf_num(x)); dxf_pair(s, 20, dxf_num(y)); dxf_pair(s, 30, "0.0"); dxf_pair(s, 40, dxf_num(h)); dxf_pair(s, 1, t);
    if (rot != 0) dxf_pair(s, 50, dxf_num(rot));
    if (halign) { dxf_pair(s, 72, tostr(halign)); dxf_pair(s, 11, dxf_num(x)); dxf_pair(s, 21, dxf_num(y)); dxf_pair(s, 31, "0.0"); }
}
static std::string dxf_write() {
    std::string s;
    dxf_pair(s, 0, "SECTION"); dxf_pair(s, 2, "HEADER"); dxf_pair(s, 9, "$ACADVER"); dxf_pair(s, 1, "AC1009"); dxf_pair(s, 9, "$INSUNITS"); dxf_pair(s, 70, "4");
    dxf_pair(s, 9, "$LTSCALE"); dxf_pair(s, 40, "1.0"); dxf_pair(s, 0, "ENDSEC");
    dxf_pair(s, 0, "SECTION"); dxf_pair(s, 2, "TABLES");
    dxf_pair(s, 0, "TABLE"); dxf_pair(s, 2, "LTYPE"); dxf_pair(s, 70, "3");
    dxf_pair(s, 0, "LTYPE"); dxf_pair(s, 2, "CONTINUOUS"); dxf_pair(s, 70, "0"); dxf_pair(s, 3, "Solid line"); dxf_pair(s, 72, "65"); dxf_pair(s, 73, "0"); dxf_pair(s, 40, "0.0");
    dxf_pair(s, 0, "LTYPE"); dxf_pair(s, 2, "HIDDEN"); dxf_pair(s, 70, "0"); dxf_pair(s, 3, "Hidden __ __ __"); dxf_pair(s, 72, "65"); dxf_pair(s, 73, "2"); dxf_pair(s, 40, "9.0"); dxf_pair(s, 49, "6.0"); dxf_pair(s, 49, "-3.0");
    dxf_pair(s, 0, "LTYPE"); dxf_pair(s, 2, "CENTER"); dxf_pair(s, 70, "0"); dxf_pair(s, 3, "Center ____ _ ____"); dxf_pair(s, 72, "65"); dxf_pair(s, 73, "4"); dxf_pair(s, 40, "30.0");
    dxf_pair(s, 49, "20.0"); dxf_pair(s, 49, "-4.0"); dxf_pair(s, 49, "2.0"); dxf_pair(s, 49, "-4.0");
    dxf_pair(s, 0, "ENDTAB");
    dxf_pair(s, 0, "TABLE"); dxf_pair(s, 2, "LAYER"); dxf_pair(s, 70, tostr((int)g_dl.size()));
    static const char* LT[3] = {"CONTINUOUS", "HIDDEN", "CENTER"};
    for (const DLayer& L : g_dl) { dxf_pair(s, 0, "LAYER"); dxf_pair(s, 2, L.name); dxf_pair(s, 70, "0"); dxf_pair(s, 62, tostr(L.aci)); dxf_pair(s, 6, LT[std::max(0, std::min(2, L.ltype))]); }
    dxf_pair(s, 0, "ENDTAB"); dxf_pair(s, 0, "ENDSEC");
    dxf_pair(s, 0, "SECTION"); dxf_pair(s, 2, "ENTITIES");
    for (const DEnt& e : g_de) {
        std::string L = g_dl[(size_t)std::max(0, std::min((int)g_dl.size() - 1, e.layer))].name;
        switch (e.type) {
            case DE_LINE: dxf_line(s, L, e.p[0], e.p[1], e.p[2], e.p[3]); break;
            case DE_CIRCLE: dxf_pair(s, 0, "CIRCLE"); dxf_pair(s, 8, L); dxf_pair(s, 10, dxf_num(e.p[0])); dxf_pair(s, 20, dxf_num(e.p[1])); dxf_pair(s, 30, "0.0"); dxf_pair(s, 40, dxf_num(e.p[2])); break;
            case DE_ARC: dxf_pair(s, 0, "ARC"); dxf_pair(s, 8, L); dxf_pair(s, 10, dxf_num(e.p[0])); dxf_pair(s, 20, dxf_num(e.p[1])); dxf_pair(s, 30, "0.0"); dxf_pair(s, 40, dxf_num(e.p[2])); dxf_pair(s, 50, dxf_num(e.p[3])); dxf_pair(s, 51, dxf_num(e.p[4])); break;
            case DE_PLINE:
                dxf_pair(s, 0, "POLYLINE"); dxf_pair(s, 8, L); dxf_pair(s, 66, "1"); dxf_pair(s, 70, e.closed ? "1" : "0"); dxf_pair(s, 10, "0.0"); dxf_pair(s, 20, "0.0"); dxf_pair(s, 30, "0.0");
                for (size_t i = 0; i + 1 < e.p.size(); i += 2) { dxf_pair(s, 0, "VERTEX"); dxf_pair(s, 8, L); dxf_pair(s, 10, dxf_num(e.p[i])); dxf_pair(s, 20, dxf_num(e.p[i + 1])); dxf_pair(s, 30, "0.0"); }
                dxf_pair(s, 0, "SEQEND"); dxf_pair(s, 8, L);
                break;
            case DE_TEXT: dxf_text(s, L, e.p[0], e.p[1], e.h, e.p.size() > 2 ? e.p[2] : 0, e.text); break;
            case DE_DIMLIN: {  // exploded: extension lines, the dimension line, arrows as short lines, the value
                double ax, ay, bx, by, nx, ny; dr_dim_geom(e, ax, ay, bx, by, nx, ny);
                double sgn = ((ax - e.p[0]) * nx + (ay - e.p[1]) * ny) >= 0 ? 1 : -1;
                dxf_line(s, L, e.p[0] + nx * sgn, e.p[1] + ny * sgn, ax + nx * sgn * 2, ay + ny * sgn * 2);
                dxf_line(s, L, e.p[2] + nx * sgn, e.p[3] + ny * sgn, bx + nx * sgn * 2, by + ny * sgn * 2);
                dxf_line(s, L, ax, ay, bx, by);
                double ux = bx - ax, uy = by - ay, ul = std::max(1e-12, std::hypot(ux, uy)); ux /= ul; uy /= ul;
                for (int k = 0; k < 2; ++k) {
                    double tx = k ? bx : ax, ty = k ? by : ay, d = k ? -1 : 1;
                    dxf_line(s, L, tx, ty, tx + d * 2.5 * ux + 0.8 * nx, ty + d * 2.5 * uy + 0.8 * ny);
                    dxf_line(s, L, tx, ty, tx + d * 2.5 * ux - 0.8 * nx, ty + d * 2.5 * uy - 0.8 * ny);
                }
                double ang = std::atan2(by - ay, bx - ax) * 180 / geo::PI; if (ang > 90.01 || ang < -89.99) ang += 180;
                std::string t = e.text.empty() ? fmtd(std::round(std::hypot(e.p[2] - e.p[0], e.p[3] - e.p[1]) * 100) / 100) : e.text;
                dxf_text(s, L, 0.5 * (ax + bx) + nx * sgn * 0.8, 0.5 * (ay + by) + ny * sgn * 0.8, e.h, ang, t, 1);
                break;
            }
            case DE_DIMRAD: {
                double ex = e.p[0] + e.p[2] * std::cos(e.p[3]), ey = e.p[1] + e.p[2] * std::sin(e.p[3]);
                dxf_line(s, L, e.p[0], e.p[1], ex, ey);
                dxf_text(s, L, ex + 1.5 * std::cos(e.p[3]), ey + 1.5 * std::sin(e.p[3]), e.h, 0, e.text.empty() ? "R" + fmtd(std::round(e.p[2] * 100) / 100) : e.text);
                break;
            }
        }
    }
    dxf_pair(s, 0, "ENDSEC"); dxf_pair(s, 0, "EOF");
    return s;
}
static int dxf_read(const std::string& txt, std::string& err) {  // LINE, CIRCLE, ARC, LWPOLYLINE, POLYLINE, TEXT, MTEXT, POINT
    std::vector<std::pair<int, std::string>> P;
    {
        size_t pos = 0; std::vector<std::string> lines;
        while (pos <= txt.size()) { size_t e = txt.find('\n', pos); if (e == std::string::npos) e = txt.size(); std::string l = txt.substr(pos, e - pos); while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back(); size_t a = l.find_first_not_of(' '); lines.push_back(a == std::string::npos ? "" : l.substr(a)); pos = e + 1; if (e == txt.size()) break; }
        for (size_t i = 0; i + 1 < lines.size(); i += 2) P.push_back(std::make_pair(atoi(lines[i].c_str()), lines[i + 1]));
    }
    if (P.empty()) { err = "not a DXF file"; return 0; }
    int n0 = (int)g_de.size();
    bool in_ent = false;
    size_t i = 0;
    while (i < P.size()) {
        if (P[i].first == 0 && P[i].second == "SECTION" && i + 1 < P.size() && P[i + 1].second == "ENTITIES") { in_ent = true; i += 2; continue; }
        if (P[i].first == 0 && P[i].second == "ENDSEC") { in_ent = false; ++i; continue; }
        if (!in_ent || P[i].first != 0) { ++i; continue; }
        std::string ty = P[i].second; ++i;
        std::map<int, std::vector<std::string>> G;
        while (i < P.size() && P[i].first != 0) { G[P[i].first].push_back(P[i].second); ++i; }
        auto num = [&](int c, size_t k = 0) { return G.count(c) && G[c].size() > k ? atof(G[c][k].c_str()) : 0.0; };
        int layer = G.count(8) ? dr_layer_of(G[8][0]) : 0;
        if (ty == "LINE") dr_add(DE_LINE, {num(10), num(20), num(11), num(21)}, layer);
        else if (ty == "CIRCLE") dr_add(DE_CIRCLE, {num(10), num(20), num(40)}, layer);
        else if (ty == "ARC") dr_add(DE_ARC, {num(10), num(20), num(40), num(50), num(51)}, layer);
        else if (ty == "TEXT" || ty == "MTEXT") { std::string t = G.count(1) ? G[1][0] : ""; dr_add(DE_TEXT, {num(10), num(20), num(50)}, layer, t, G.count(40) ? num(40) : 2.5); }
        else if (ty == "LWPOLYLINE") {
            std::vector<double> p; size_t n = G.count(10) ? G[10].size() : 0;
            for (size_t k = 0; k < n; ++k) { p.push_back(num(10, k)); p.push_back(num(20, k)); }
            if (n >= 2) dr_add(DE_PLINE, p, layer, "", 3.5, ((int)num(70) & 1) != 0);
        } else if (ty == "POLYLINE") {
            bool closed = ((int)num(70) & 1) != 0; std::vector<double> p;
            while (i < P.size() && P[i].first == 0 && P[i].second == "VERTEX") {
                ++i; double x = 0, y = 0;
                while (i < P.size() && P[i].first != 0) { if (P[i].first == 10) x = atof(P[i].second.c_str()); if (P[i].first == 20) y = atof(P[i].second.c_str()); ++i; }
                p.push_back(x); p.push_back(y);
            }
            if (i < P.size() && P[i].second == "SEQEND") { ++i; while (i < P.size() && P[i].first != 0) ++i; }
            if (p.size() >= 4) dr_add(DE_PLINE, p, layer, "", 3.5, closed);
        }
    }
    return (int)g_de.size() - n0;
}

// ---------------------------------------------------------------- the design in three views (hidden lines dashed)
struct DTri { V3 a, b, c; };
static void dr_design_tris(std::vector<DTri>& T) {
    auto add_mesh = [&](const geo::Mesh& M, const std::function<V3(const V3&)>& f) {
        for (size_t t = 0; t + 2 < M.t.size(); t += 3) T.push_back(DTri{f(M.v[(size_t)M.t[t]]), f(M.v[(size_t)M.t[t + 1]]), f(M.v[(size_t)M.t[t + 2]])});
    };
    if (g_asm_on) {
        for (const AComp& C : g_ac) {
            if (C.hidden) continue;
            for (const APiece& pc : C.pieces) if (pc.mesh) { const Pose P = C.pose, Lc = pc.local; add_mesh(pc.mesh->m, [&](const V3& v) { return P.apply(Lc.apply(v)); }); }
        }
    } else add_mesh(g_mesh, [](const V3& v) { return v; });
}
// the visible and hidden feature edges of the triangles seen from direction e (towards the viewer), in (u, v) metres
static void dr_project(const std::vector<DTri>& T, V3 e, V3 up, std::vector<double>& vis, std::vector<double>& hid, double& u0, double& v0, double& u1, double& v1) {
    e = e / geo::len(e);
    V3 r = geo::cross(up, e); if (geo::len(r) < 1e-9) r = geo::cross(V3(0, 1, 0), e); r = r / geo::len(r);
    V3 vv = geo::cross(e, r);
    // weld the vertices so that neighbouring triangles share edges
    std::map<std::tuple<long long, long long, long long>, int> id;
    std::vector<V3> P; std::vector<int> tri;
    double q = 1e5;  // 10 micrometre welding
    auto key = [&](const V3& p) { return std::make_tuple((long long)std::floor(p.x * q + 0.5), (long long)std::floor(p.y * q + 0.5), (long long)std::floor(p.z * q + 0.5)); };
    for (const DTri& t : T) for (const V3* p : {&t.a, &t.b, &t.c}) { auto k = key(*p); auto it = id.find(k); int j; if (it == id.end()) { j = (int)P.size(); id[k] = j; P.push_back(*p); } else j = it->second; tri.push_back(j); }
    size_t nt = tri.size() / 3;
    std::vector<V3> N(nt);
    for (size_t t = 0; t < nt; ++t) { V3 n = geo::cross(P[(size_t)tri[3 * t + 1]] - P[(size_t)tri[3 * t]], P[(size_t)tri[3 * t + 2]] - P[(size_t)tri[3 * t]]); double l = geo::len(n); N[t] = l > 0 ? n / l : V3(); }
    std::map<std::pair<int, int>, std::vector<int>> E;
    for (size_t t = 0; t < nt; ++t) for (int k = 0; k < 3; ++k) { int a = tri[3 * t + (size_t)k], b = tri[3 * t + (size_t)((k + 1) % 3)]; E[std::make_pair(std::min(a, b), std::max(a, b))].push_back((int)t); }
    std::vector<V3> S(P.size());  // (u, v, w)
    u0 = v0 = 1e300; u1 = v1 = -1e300; double w0 = 1e300, w1 = -1e300;
    for (size_t i = 0; i < P.size(); ++i) { S[i] = V3(geo::dot(P[i], r), geo::dot(P[i], vv), geo::dot(P[i], e)); u0 = std::min(u0, S[i].x); u1 = std::max(u1, S[i].x); v0 = std::min(v0, S[i].y); v1 = std::max(v1, S[i].y); w0 = std::min(w0, S[i].z); w1 = std::max(w1, S[i].z); }
    if (P.empty()) return;
    // the depth buffer (nearest w) at about 1600 pixels on the longer side
    double span = std::max(u1 - u0, v1 - v0), res = span / 1600.0;
    int W = (int)((u1 - u0) / res) + 3, H = (int)((v1 - v0) / res) + 3;
    std::vector<float> Z((size_t)W * H, -1e30f);
    for (size_t t = 0; t < nt; ++t) {
        const V3 &A = S[(size_t)tri[3 * t]], &B = S[(size_t)tri[3 * t + 1]], &C = S[(size_t)tri[3 * t + 2]];
        double ax = (A.x - u0) / res, ay = (A.y - v0) / res, bx = (B.x - u0) / res, by = (B.y - v0) / res, cx = (C.x - u0) / res, cy = (C.y - v0) / res;
        double den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy); if (std::fabs(den) < 1e-12) continue;
        int x0 = std::max(0, (int)std::floor(std::min(ax, std::min(bx, cx)))), x1 = std::min(W - 1, (int)std::ceil(std::max(ax, std::max(bx, cx))));
        int y0 = std::max(0, (int)std::floor(std::min(ay, std::min(by, cy)))), y1 = std::min(H - 1, (int)std::ceil(std::max(ay, std::max(by, cy))));
        for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) {
            double l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / den, l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / den, l3 = 1 - l1 - l2;
            if (l1 < -1e-6 || l2 < -1e-6 || l3 < -1e-6) continue;
            float w = (float)(l1 * A.z + l2 * B.z + l3 * C.z); float& z = Z[(size_t)y * W + x]; if (w > z) z = w;
        }
    }
    double eps = 0.004 * (w1 - w0) + 2.5 * res;
    auto visible = [&](double u, double v, double w) {
        int x = (int)std::round((u - u0) / res), y = (int)std::round((v - v0) / res); float zb = -1e30f;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) { int xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < W && yy < H) zb = std::max(zb, Z[(size_t)yy * W + xx]); }
        return w >= zb - eps;
    };
    double cos_crease = std::cos(35 * geo::PI / 180);
    for (auto& kv : E) {
        const std::vector<int>& ts = kv.second;
        bool crease = ts.size() == 1, sil = false;
        if (ts.size() == 2) {
            double d = geo::dot(N[(size_t)ts[0]], N[(size_t)ts[1]]);
            double s0 = geo::dot(N[(size_t)ts[0]], e), s1 = geo::dot(N[(size_t)ts[1]], e);
            crease = d < cos_crease; sil = (s0 > 0) != (s1 > 0);
        }
        if (!crease && !sil) continue;
        const V3 &A = S[(size_t)kv.first.first], &B = S[(size_t)kv.first.second];
        double L = std::hypot(B.x - A.x, B.y - A.y); if (L < 1e-12) continue;
        int ns = std::max(2, (int)(L / (0.7 * res)));
        std::vector<char> st((size_t)ns + 1);
        for (int k = 0; k <= ns; ++k) { double t = (double)k / ns; st[(size_t)k] = visible(A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)) ? 1 : 0; }
        // runs shorter than about 4 pixels take their neighbour's state: no flicker where the depth test is marginal
        int minrun = 6;
        for (int pass = 0; pass < 2; ++pass) {
            int k = 0;
            while (k <= ns) {
                int j = k; while (j + 1 <= ns && st[(size_t)j + 1] == st[(size_t)k]) ++j;
                if (j - k + 1 < minrun && (k > 0 || j < ns)) { char v = k > 0 ? st[(size_t)k - 1] : st[(size_t)j + 1]; for (int q = k; q <= j; ++q) st[(size_t)q] = v; }
                k = j + 1;
            }
        }
        int k = 0;
        while (k <= ns) {
            int j = k; while (j + 1 <= ns && st[(size_t)j + 1] == st[(size_t)k]) ++j;
            double ta = (double)k / ns, tb = (double)std::min(ns, j + 1) / ns;
            if (st[(size_t)k] || crease) {  // a hidden silhouette is not drawn, as in a drawing office
                std::vector<double>& out = st[(size_t)k] ? vis : hid;
                out.push_back(A.x + ta * (B.x - A.x)); out.push_back(A.y + ta * (B.y - A.y)); out.push_back(A.x + tb * (B.x - A.x)); out.push_back(A.y + tb * (B.y - A.y));
            }
            k = j + 1;
        }
    }
}
static void dr_title_block(double sc_num, double sc_den, const std::string& title, const std::string& sub) {
    dr_add(DE_PLINE, {0, 0, 420, 0, 420, 297, 0, 297}, 6, "", 3.5, true);
    dr_add(DE_PLINE, {10, 10, 410, 10, 410, 287, 10, 287}, 6, "", 3.5, true);
    double x0 = 240, y0 = 10, x1 = 410, y1 = 50;
    dr_add(DE_PLINE, {x0, y0, x1, y0, x1, y1, x0, y1}, 6, "", 3.5, true);
    dr_add(DE_LINE, {x0, 30, x1, 30}, 6); dr_add(DE_LINE, {x0, 20, x1, 20}, 6); dr_add(DE_LINE, {330, 10, 330, 30}, 6);
    SYSTEMTIME t; GetLocalTime(&t); char d[32]; snprintf(d, sizeof d, "%04d-%02d-%02d", t.wYear, t.wMonth, t.wDay);
    char s[48]; snprintf(s, sizeof s, "SCALE %g:%g", sc_num, sc_den);
    dr_add(DE_TEXT, {x0 + 3, 40}, 5, title, 5.0); dr_add(DE_TEXT, {x0 + 3, 33}, 5, sub, 2.5);
    dr_add(DE_TEXT, {x0 + 3, 23}, 5, s, 3.0); dr_add(DE_TEXT, {333, 23}, 5, "UNITS mm   A3", 3.0);
    dr_add(DE_TEXT, {x0 + 3, 13}, 5, std::string("AeroSynapse Studio  ") + d, 2.5); dr_add(DE_TEXT, {333, 13}, 5, "FIRST-ANGLE PROJECTION", 2.5);
}
static void dr_views() {
    std::vector<DTri> T; dr_design_tris(T);
    if (T.empty()) { MessageBoxA(g_dr ? g_dr : g_main, "There is no model to draw: open or build a part or an assembly first.", "2D drawing", MB_ICONINFORMATION); return; }
    struct View { const char* name; V3 e, up; std::vector<double> vis, hid; double u0, v0, u1, v1; };
    View V[4] = {{"FRONT", V3(0, -1, 0), V3(0, 0, 1), {}, {}, 0, 0, 0, 0}, {"TOP", V3(0, 0, 1), V3(0, 1, 0), {}, {}, 0, 0, 0, 0},
                 {"LEFT", V3(-1, 0, 0), V3(0, 0, 1), {}, {}, 0, 0, 0, 0}, {"ISOMETRIC", V3(1, -1, 1), V3(0, 0, 1), {}, {}, 0, 0, 0, 0}};
    for (View& v : V) dr_project(T, v.e, v.up, v.vis, v.hid, v.u0, v.v0, v.u1, v.v1);
    // the scale: the largest standard one at which front + left side fit across and front + top fit down the sheet
    double needW = (V[0].u1 - V[0].u0) + (V[2].u1 - V[2].u0), needH = (V[0].v1 - V[0].v0) + (V[1].v1 - V[1].v0);  // metres
    static const double SC[][2] = {{10, 1}, {5, 1}, {2, 1}, {1, 1}, {1, 2}, {1, 5}, {1, 10}, {1, 20}, {1, 50}, {1, 100}, {1, 200}, {1, 500}, {1, 1000}};
    double sn = 1, sd = 1000;
    for (auto& s : SC) { double k = 1000 * s[0] / s[1]; if (needW * k <= 300 && needH * k <= 215) { sn = s[0]; sd = s[1]; break; } }
    double k = 1000 * sn / sd;  // mm on paper per metre
    dr_push_undo(); g_de.clear(); g_dsel.clear();
    std::string nm = g_path.empty() ? std::string("design") : g_path.substr(g_path.find_last_of("\\/") + 1);
    dr_title_block(sn, sd, nm, std::string(g_asm_on ? "assembly" : "part") + ": three views and an isometric, hidden lines dashed");
    double gap = 22, fx = 30, fy = 287 - 20 - (V[0].v1 - V[0].v0) * k;  // front view at the top left
    double ox[4] = {fx, fx, fx + (V[0].u1 - V[0].u0) * k + gap, 0}, oy[4] = {fy, fy - gap - (V[1].v1 - V[1].v0) * k, fy, 0};
    {   // the isometric in the space above the title block, at the same scale or smaller
        double iw = (V[3].u1 - V[3].u0) * k, ih = (V[3].v1 - V[3].v0) * k, f = std::min(1.0, std::min(150 / std::max(1e-9, iw), 95 / std::max(1e-9, ih)));
        ox[3] = 250 + (150 - iw * f) / 2; oy[3] = 60 + (95 - ih * f) / 2;
        V[3].u1 = V[3].u0 + (V[3].u1 - V[3].u0) * f; V[3].v1 = V[3].v0 + (V[3].v1 - V[3].v0) * f;
        for (size_t i = 0; i < V[3].vis.size(); i += 2) { V[3].vis[i] = V[3].u0 + (V[3].vis[i] - V[3].u0) * f; V[3].vis[i + 1] = V[3].v0 + (V[3].vis[i + 1] - V[3].v0) * f; }
        V[3].hid.clear();  // an isometric shows only what is seen
    }
    for (int a = 0; a < 4; ++a) {
        View& v = V[a];
        auto X = [&](double u) { return ox[a] + (u - v.u0) * k; }; auto Y = [&](double w) { return oy[a] + (w - v.v0) * k; };
        for (size_t i = 0; i + 3 < v.hid.size(); i += 4) dr_add(DE_LINE, {X(v.hid[i]), Y(v.hid[i + 1]), X(v.hid[i + 2]), Y(v.hid[i + 3])}, 2);
        for (size_t i = 0; i + 3 < v.vis.size(); i += 4) dr_add(DE_LINE, {X(v.vis[i]), Y(v.vis[i + 1]), X(v.vis[i + 2]), Y(v.vis[i + 3])}, 1);
        double w = (v.u1 - v.u0) * k, h = (v.v1 - v.v0) * k;
        dr_add(DE_TEXT, {ox[a], oy[a] + h + 4}, 5, v.name, 3.5);
        if (a < 3) {  // overall sizes, measured in model millimetres
            DEnt d; d.type = DE_DIMLIN; d.layer = 4; d.h = 2.5;
            d.p = {ox[a], oy[a], ox[a] + w, oy[a], ox[a], oy[a] - 8}; d.text = fmtd(std::round((v.u1 - v.u0) * 1e4) / 10); g_de.push_back(d);
            if (a != 1) { d.p = {ox[a] + w, oy[a], ox[a] + w, oy[a] + h, ox[a] + w + 8, oy[a]}; d.text = fmtd(std::round((v.v1 - v.v0) * 1e4) / 10); g_de.push_back(d); }
            else { d.p = {ox[a], oy[a] + h, ox[a], oy[a], ox[a] - 8, oy[a]}; d.text = fmtd(std::round((v.v1 - v.v0) * 1e4) / 10); g_de.push_back(d); }
            // centre lines through the view's middle
            dr_add(DE_LINE, {ox[a] - 4, oy[a] + h / 2, ox[a] + w + 4, oy[a] + h / 2}, 3);
            dr_add(DE_LINE, {ox[a] + w / 2, oy[a] - 4, ox[a] + w / 2, oy[a] + h + 4}, 3);
        }
    }
    int nv = 0, nh = 0; for (const DEnt& e : g_de) { if (e.layer == 1) ++nv; if (e.layer == 2) ++nh; }
    dr_zoom_extents(); g_dfile.clear();
    dr_prompt("VIEWS: " + tostr(nv) + " visible and " + tostr(nh) + " hidden edge pieces at scale " + fmtd(sn) + ":" + fmtd(sd) + " (A3, first-angle). DXFOUT saves it for AutoCAD.");
    dr_refresh();
}

// ---------------------------------------------------------------- the flat pattern of a bent plate
static void dr_flat() {
    int sel = -1;
    if (g_sel >= 0 && g_sel < (int)features().size() && features().at((size_t)g_sel).str("type", "") == "bentplate") sel = g_sel;
    for (size_t q = 0; q < features().size() && sel < 0; ++q) if (features().at(q).str("type", "") == "bentplate") sel = (int)q;
    if (sel < 0) { MessageBoxA(g_dr ? g_dr : g_main, "There is no bent plate in the design. Make one with Sheet & Composite (Bent plate, a preset, or Sheet from sketch), then FLAT.", "Flat pattern", MB_ICONINFORMATION); return; }
    const js::Value& f = features().at((size_t)sel);
    std::vector<double> xs, zs;
    if (const js::Value* a = f.find("profile")) for (size_t i = 0; i < a->size(); ++i) if (a->at(i).size() >= 2) { xs.push_back(a->at(i).at(0).n); zs.push_back(a->at(i).at(1).n); }
    double t = f.num("thickness", 0.003), r = f.num("bend_radius", 0.01), wid = f.num("width", 0.3), K = 0.44, rc = r + t / 2;
    size_t n = xs.size(); if (n < 2) return;
    std::vector<double> th(n, 0.0), sb(n, 0.0), turn(n, 0.0);
    for (size_t i = 1; i + 1 < n; ++i) {
        double ax = xs[i] - xs[i - 1], az = zs[i] - zs[i - 1], bx = xs[i + 1] - xs[i], bz = zs[i + 1] - zs[i];
        th[i] = std::acos(std::max(-1.0, std::min(1.0, (ax * bx + az * bz) / std::max(1e-12, std::hypot(ax, az) * std::hypot(bx, bz)))));
        sb[i] = rc * std::tan(th[i] / 2); turn[i] = ax * bz - az * bx;
    }
    dr_push_undo(); g_de.clear(); g_dsel.clear();
    double x = 20, y0 = 20, W = wid * 1000;
    std::vector<double> bl;  // bend-zone edges along the blank, mm
    for (size_t i = 0; i + 1 < n; ++i) {
        double seg = std::hypot(xs[i + 1] - xs[i], zs[i + 1] - zs[i]) - sb[i] - sb[i + 1];
        x += 1000 * seg;
        if (i + 1 < n - 1) { double ba = 1000 * th[i + 1] * (r + K * t); bl.push_back(x); bl.push_back(x + ba); x += ba; }
    }
    double L = x - 20;
    dr_add(DE_PLINE, {20, y0, 20 + L, y0, 20 + L, y0 + W, 20, y0 + W}, 1, "", 3.5, true);
    for (size_t b = 0; b + 1 < bl.size(); b += 2) {
        double cxm = 0.5 * (bl[b] + bl[b + 1]);
        dr_add(DE_LINE, {bl[b], y0, bl[b], y0 + W}, 2); dr_add(DE_LINE, {bl[b + 1], y0, bl[b + 1], y0 + W}, 2);
        dr_add(DE_LINE, {cxm, y0 - 6, cxm, y0 + W + 6}, 3);
        size_t i = b / 2 + 1;
        char s[96]; snprintf(s, sizeof s, "BEND %s %.0f deg  R%.1f", turn[i] > 0 ? "UP" : "DOWN", th[i] * 180 / geo::PI, r * 1000);
        dr_add(DE_TEXT, {cxm + 1.5, y0 + W + 8, 90}, 5, s, 2.5);
    }
    DEnt d; d.type = DE_DIMLIN; d.layer = 4; d.h = 3;
    d.p = {20, y0, 20 + L, y0, 20, y0 - 12}; d.text = fmtd(std::round(L * 10) / 10); g_de.push_back(d);
    d.p = {20 + L, y0, 20 + L, y0 + W, 20 + L + 12, y0}; d.text = fmtd(std::round(W * 10) / 10); g_de.push_back(d);
    char s[200]; snprintf(s, sizeof s, "FLAT PATTERN of '%s': t = %.1f mm, inner R = %.1f mm, K = %.2f; dashed = bend zones, dash-dot = bend lines",
                          f.str("id", "").c_str(), t * 1000, r * 1000, K);
    dr_add(DE_TEXT, {20, y0 + W + 40}, 5, s, 3.5);
    snprintf(s, sizeof s, "blank %.1f x %.1f mm (the Unfold check: %.1f mm)", L, W, 1000 * flat_length(f));
    dr_add(DE_TEXT, {20, y0 + W + 33}, 5, s, 3.0);
    dr_zoom_extents(); g_dfile.clear();
    dr_prompt(std::string("FLAT: ") + s);
    dr_refresh();
}

// the largest closed outline to the 3D sketcher (mm to m), as a polyline to pad or pocket
static void dr_to_sketch() {
    int best = -1; double ba = 0;
    for (size_t i = 0; i < g_de.size(); ++i) {
        const DEnt& e = g_de[i];
        if (!(g_dsel.empty() || g_dsel.count((int)i))) continue;
        double a = 0;
        if (e.type == DE_PLINE && e.closed) { size_t n = e.p.size() / 2; for (size_t k = 0; k < n; ++k) { size_t j = (k + 1) % n; a += e.p[2 * k] * e.p[2 * j + 1] - e.p[2 * j] * e.p[2 * k + 1]; } a = std::fabs(a) / 2; }
        else if (e.type == DE_CIRCLE) a = geo::PI * e.p[2] * e.p[2];
        if (a > ba) { ba = a; best = (int)i; }
    }
    if (best < 0) { MessageBoxA(g_dr, "Draw (or pick) a closed polyline, a rectangle or a circle first.", "To the sketcher", MB_ICONINFORMATION); return; }
    const DEnt& e = g_de[(size_t)best];
    std::vector<geo::P2> pts;
    if (e.type == DE_CIRCLE) for (int k = 0; k < 48; ++k) pts.push_back(geo::P2((e.p[0] + e.p[2] * std::cos(2 * geo::PI * k / 48)) / 1000, (e.p[1] + e.p[2] * std::sin(2 * geo::PI * k / 48)) / 1000));
    else for (size_t i = 0; i + 1 < e.p.size(); i += 2) pts.push_back(geo::P2(e.p[i] / 1000, e.p[i + 1] / 1000));
    open_sketch(SK_EXTRUDE, -1, OP_PAD, SH_POLY);
    g_sk_pts = pts; g_sk_smooth = false;
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300; for (auto& q : pts) { x0 = std::min(x0, q.x); y0 = std::min(y0, q.y); x1 = std::max(x1, q.x); y1 = std::max(y1, q.y); }
    g_sk_cx = 0.5 * (x0 + x1); g_sk_cy = 0.5 * (y0 + y1);
    RECT rc; GetClientRect(g_sk_canvas, &rc); g_sk_scale = std::min((rc.right - 80) / std::max(1e-6, x1 - x0), (rc.bottom - 100) / std::max(1e-6, y1 - y0));
    InvalidateRect(g_sk_canvas, nullptr, FALSE);
    status("the drawing's outline is in the sketcher (" + tostr((int)pts.size()) + " points, mm to m): set the depth in Size and press Finish to pad it");
}

// ---------------------------------------------------------------- commands
static bool dr_parse_point(const std::string& s0, double& x, double& y) {  // x,y  @dx,dy  @len<ang  or a length along the cursor
    std::string s = s0; s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    if (s.empty()) return false;
    bool rel = s[0] == '@'; if (rel) s = s.substr(1);
    geo::P2 b = g_dpts.empty() ? geo::P2(0, 0) : g_dpts.back();
    size_t lt = s.find('<'), cm = s.find(',');
    char* end = nullptr;
    if (lt != std::string::npos) { double L = strtod(s.c_str(), &end), a = atof(s.c_str() + lt + 1) * geo::PI / 180; x = b.x + L * std::cos(a); y = b.y + L * std::sin(a); return true; }
    if (cm != std::string::npos) { double a = strtod(s.c_str(), &end), c = atof(s.c_str() + cm + 1); if (rel) { x = b.x + a; y = b.y + c; } else { x = a; y = c; } return true; }
    double L = strtod(s.c_str(), &end); if (end == s.c_str()) return false;
    if (g_dpts.empty()) return false;
    double cx, cy; dr_cursor_point(cx, cy); double d = std::hypot(cx - b.x, cy - b.y);
    if (d < 1e-12) { x = b.x + L; y = b.y; } else { x = b.x + L * (cx - b.x) / d; y = b.y + L * (cy - b.y) / d; }
    return true;
}
static void dr_end_tool() {
    if (g_dtool == "PLINE" && g_dpts.size() >= 2) { dr_push_undo(); std::vector<double> p; for (auto& q : g_dpts) { p.push_back(q.x); p.push_back(q.y); } dr_add(DE_PLINE, p); }
    if (!g_dtool.empty()) g_dlast_tool = g_dtool;
    g_dtool.clear(); g_dpts.clear();
    dr_prompt("Command:");
    dr_refresh();
}
static std::string dr_tool_prompt() {
    const std::string& t = g_dtool; size_t n = g_dpts.size();
    if (t == "LINE") return n ? "LINE  Specify next point (or type @dx,dy / @len<ang / a length), Enter to end:" : "LINE  Specify first point:";
    if (t == "PLINE") return n ? "PLINE  Specify next point, C to close, Enter to end:" : "PLINE  Specify start point:";
    if (t == "RECTANG") return n ? "RECTANG  Specify other corner (or @width,height):" : "RECTANG  Specify first corner:";
    if (t == "CIRCLE") return n ? "CIRCLE  Specify radius (a point or a number):" : "CIRCLE  Specify center point:";
    if (t == "ARC") return n == 0 ? "ARC  Specify start point:" : n == 1 ? "ARC  Specify second point:" : "ARC  Specify end point:";
    if (t == "DIMLINEAR") return n == 0 ? "DIMLINEAR  Specify first extension line origin:" : n == 1 ? "DIMLINEAR  Specify second extension line origin:" : "DIMLINEAR  Specify dimension line location:";
    if (t == "DIMRADIUS") return "DIMRADIUS  Select arc or circle:";
    if (t == "TEXT") return n ? "TEXT  Type the text and press Enter:" : "TEXT  Specify start point of text:";
    if (t == "ERASE") return "ERASE  Select objects (click), Enter to erase the picked:";
    if (t == "MOVE" || t == "COPY") return n ? t + "  Specify second point:" : t + "  Specify base point (picked: " + tostr((int)g_dsel.size()) + "):";
    if (t == "MIRROR") return n ? "MIRROR  Specify second point of mirror line:" : "MIRROR  Specify first point of mirror line (picked: " + tostr((int)g_dsel.size()) + "):";
    if (t == "OFFSET") return "OFFSET  Specify offset distance, then click the line or circle and the side:";
    return "Command:";
}
static double g_doffset = 10;
static void dr_point(double x, double y) {  // a point given to the command in progress
    const std::string& t = g_dtool;
    if (t.empty()) return;
    if (t == "LINE") { if (!g_dpts.empty()) { dr_push_undo(); dr_add(DE_LINE, {g_dpts.back().x, g_dpts.back().y, x, y}); } g_dpts.push_back(geo::P2(x, y)); }
    else if (t == "PLINE") g_dpts.push_back(geo::P2(x, y));
    else if (t == "RECTANG") { if (g_dpts.empty()) g_dpts.push_back(geo::P2(x, y)); else { const geo::P2 a = g_dpts[0]; dr_push_undo(); dr_add(DE_PLINE, {a.x, a.y, x, a.y, x, y, a.x, y}, -1, "", 3.5, true); g_dpts.clear(); g_dlast_tool = t; g_dtool.clear(); } }
    else if (t == "CIRCLE") { if (g_dpts.empty()) g_dpts.push_back(geo::P2(x, y)); else { const geo::P2 a = g_dpts[0]; dr_push_undo(); dr_add(DE_CIRCLE, {a.x, a.y, std::hypot(x - a.x, y - a.y)}); g_dpts.clear(); g_dlast_tool = t; g_dtool.clear(); } }
    else if (t == "ARC") {
        g_dpts.push_back(geo::P2(x, y));
        if (g_dpts.size() == 3) {  // the circle through three points
            const geo::P2 &a = g_dpts[0], &b = g_dpts[1], &c = g_dpts[2];
            double d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
            if (std::fabs(d) > 1e-12) {
                double ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y) + (b.x * b.x + b.y * b.y) * (c.y - a.y) + (c.x * c.x + c.y * c.y) * (a.y - b.y)) / d;
                double uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x) + (b.x * b.x + b.y * b.y) * (a.x - c.x) + (c.x * c.x + c.y * c.y) * (b.x - a.x)) / d;
                double R = std::hypot(a.x - ux, a.y - uy), a0 = std::atan2(a.y - uy, a.x - ux) * 180 / geo::PI, a2 = std::atan2(c.y - uy, c.x - ux) * 180 / geo::PI;
                bool ccw = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0;
                dr_push_undo(); dr_add(DE_ARC, ccw ? std::vector<double>{ux, uy, R, a0, a2} : std::vector<double>{ux, uy, R, a2, a0});
            }
            g_dpts.clear(); g_dlast_tool = t; g_dtool.clear();
        }
    } else if (t == "DIMLINEAR") {
        g_dpts.push_back(geo::P2(x, y));
        if (g_dpts.size() == 3) {
            DEnt d; d.type = DE_DIMLIN; d.layer = 4; d.h = 3.0;
            geo::P2 a = g_dpts[0], b = g_dpts[1];
            // horizontal or vertical as AutoCAD does by where the line is placed, aligned when typed as such
            if (std::fabs(x - 0.5 * (a.x + b.x)) > std::fabs(y - 0.5 * (a.y + b.y))) b = geo::P2(a.x, b.y); else b = geo::P2(b.x, a.y);
            if (std::hypot(b.x - a.x, b.y - a.y) < 1e-9) b = g_dpts[1];
            d.p = {a.x, a.y, b.x, b.y, x, y};
            dr_push_undo(); g_de.push_back(d); g_dpts.clear(); g_dlast_tool = t; g_dtool.clear();
        }
    } else if (t == "DIMRADIUS") {
        int i = dr_pick(x, y);
        if (i >= 0 && (g_de[(size_t)i].type == DE_CIRCLE || g_de[(size_t)i].type == DE_ARC)) {
            const DEnt& c = g_de[(size_t)i]; DEnt d; d.type = DE_DIMRAD; d.layer = 4; d.h = 3.0;
            d.p = {c.p[0], c.p[1], c.p[2], std::atan2(y - c.p[1], x - c.p[0])};
            dr_push_undo(); g_de.push_back(d); g_dlast_tool = t; g_dtool.clear();
        }
    } else if (t == "TEXT") { g_dpts.clear(); g_dpts.push_back(geo::P2(x, y)); }
    else if (t == "ERASE") { int i = dr_pick(x, y); if (i >= 0) { if (g_dsel.count(i)) g_dsel.erase(i); else g_dsel.insert(i); } }
    else if (t == "MOVE" || t == "COPY" || t == "MIRROR") {
        if (g_dsel.empty()) { int i = dr_pick(x, y); if (i >= 0) g_dsel.insert(i); dr_prompt(dr_tool_prompt()); dr_refresh(); return; }
        if (g_dpts.empty()) g_dpts.push_back(geo::P2(x, y));
        else {
            const geo::P2 a = g_dpts[0]; dr_push_undo();
            std::vector<DEnt> add;
            for (int i : g_dsel) { if (i >= (int)g_de.size()) continue; DEnt q = g_de[(size_t)i]; if (t == "MIRROR") dr_transform(q, 0, 0, true, a.x, a.y, x, y); else dr_transform(q, x - a.x, y - a.y, false); if (t == "MOVE") g_de[(size_t)i] = q; else add.push_back(q); }
            g_de.insert(g_de.end(), add.begin(), add.end());
            g_dpts.clear(); g_dlast_tool = t; g_dtool.clear();
        }
    } else if (t == "OFFSET") {
        int i = dr_pick(x, y);
        if (i >= 0) {
            DEnt q = g_de[(size_t)i];
            if (q.type == DE_LINE) {
                double dx = q.p[2] - q.p[0], dy = q.p[3] - q.p[1], L = std::max(1e-12, std::hypot(dx, dy)), nx = -dy / L, ny = dx / L;
                double sgn = ((x - q.p[0]) * nx + (y - q.p[1]) * ny) >= 0 ? 1 : -1;
                dr_transform(q, sgn * g_doffset * nx, sgn * g_doffset * ny, false); dr_push_undo(); g_de.push_back(q);
            } else if (q.type == DE_CIRCLE || q.type == DE_ARC) {
                bool out = std::hypot(x - q.p[0], y - q.p[1]) > q.p[2]; q.p[2] = std::max(1e-6, q.p[2] + (out ? g_doffset : -g_doffset)); dr_push_undo(); g_de.push_back(q);
            }
        }
    }
    dr_prompt(dr_tool_prompt());
    dr_refresh();
}
static void dr_save_png();
static void dr_vector_out(bool pdf);
static void dr_command(std::string c) {  // a typed command or an alias
    for (char& ch : c) ch = (char)toupper((unsigned char)ch);
    for (int k = 0; k < NDRT; ++k) if (c == DR_TOOLS[k][1]) { c = DR_TOOLS[k][0]; break; }
    if (c == "ZOOM" || c == "ZE") { dr_zoom_extents(); dr_refresh(); return; }
    if (c == "UNDO") { if (!g_de_undo.empty()) { g_de_redo.push_back(g_de); g_de = g_de_undo.back(); g_de_undo.pop_back(); g_dsel.clear(); } dr_refresh(); return; }
    if (c == "REDO") { if (!g_de_redo.empty()) { g_de_undo.push_back(g_de); g_de = g_de_redo.back(); g_de_redo.pop_back(); g_dsel.clear(); } dr_refresh(); return; }
    if (c == "VIEWS") { dr_views(); return; }
    if (c == "FLAT") { dr_flat(); return; }
    if (c == "SKETCH") { dr_to_sketch(); return; }
    if (c == "NEW") { dr_push_undo(); g_de.clear(); g_dsel.clear(); g_dfile.clear(); dr_zoom_extents(); dr_prompt("A new, empty drawing (UNDO brings the old one back)."); dr_refresh(); return; }
    if (c == "PNGOUT") { dr_save_png(); return; }
    if (c == "SVGOUT" || c == "PDFOUT") { dr_vector_out(c == "PDFOUT"); return; }
    if (c == "DXFOUT") {
        std::string p = g_dfile.empty() ? std::string("drawing.dxf") : g_dfile;
        if (!ask_path(true, "AutoCAD DXF (*.dxf)\0*.dxf\0", "dxf", p)) return;
        try { js::write_file(p, dxf_write()); g_dfile = p; dr_prompt("DXFOUT: " + tostr((int)g_de.size()) + " entities written to " + p + " (AutoCAD R12 DXF, millimetres)"); }
        catch (std::exception& e) { MessageBoxA(g_dr, e.what(), "DXF", MB_ICONERROR); }
        dr_refresh(); return;
    }
    if (c == "DXFIN") {
        std::string p;
        if (!ask_path(false, "AutoCAD DXF (*.dxf)\0*.dxf\0All files\0*.*\0", "dxf", p)) return;
        std::string txt; try { txt = js::read_file(p); } catch (...) { MessageBoxA(g_dr, ("Could not read " + p).c_str(), "DXF", MB_ICONERROR); return; }
        dr_push_undo(); g_de.clear(); g_dsel.clear();
        std::string err; int n = dxf_read(txt, err);
        if (!err.empty()) { MessageBoxA(g_dr, err.c_str(), "DXF", MB_ICONERROR); return; }
        g_dfile = p; dr_zoom_extents();
        dr_prompt("DXFIN: " + tostr(n) + " entities read from " + p);
        dr_refresh(); return;
    }
    if (c == "OFFSET") {
        std::string v = fmtd(g_doffset);
        if (!ask_text(g_dr, "Offset", "Offset distance (mm):", v)) return;
        g_doffset = std::max(1e-6, atof(v.c_str()));
    }
    if (c == "ERASE" && !g_dsel.empty()) {  // erase what is picked already
        dr_push_undo(); std::vector<DEnt> keep; for (size_t i = 0; i < g_de.size(); ++i) if (!g_dsel.count((int)i)) keep.push_back(g_de[i]); g_de.swap(keep); g_dsel.clear(); dr_refresh(); return;
    }
    for (int k = 0; k < NDRT; ++k) if (c == DR_TOOLS[k][0]) { if (g_dtool == "PLINE") dr_end_tool(); g_dtool = c; g_dpts.clear(); dr_prompt(dr_tool_prompt()); dr_refresh(); return; }
    dr_prompt("Unknown command \"" + c + "\". Try L, PL, REC, C, A, DLI, DRA, T, E, M, CO, MI, O, Z, U, VIEWS, FLAT, SKETCH, DXFIN, DXFOUT.");
}
static void dr_enter(const std::string& in) {  // the command line's Enter
    std::string s = in; while (!s.empty() && s.back() == ' ') s.pop_back();
    if (g_dtool == "TEXT" && !g_dpts.empty()) {
        if (!s.empty()) { dr_push_undo(); dr_add(DE_TEXT, {g_dpts[0].x, g_dpts[0].y, 0}, 5, s, 3.5); }
        g_dlast_tool = "TEXT"; g_dtool.clear(); g_dpts.clear(); dr_prompt("Command:"); dr_refresh(); return;
    }
    if (s.empty()) {
        if (g_dtool == "ERASE") { dr_command("ERASE"); g_dtool.clear(); return; }
        if (!g_dtool.empty()) { dr_end_tool(); return; }
        if (!g_dlast_tool.empty()) dr_command(g_dlast_tool);  // Enter repeats the last command, as in AutoCAD
        return;
    }
    if (g_dtool == "PLINE" && (s == "C" || s == "c") && g_dpts.size() >= 3) {
        dr_push_undo(); std::vector<double> p; for (auto& q : g_dpts) { p.push_back(q.x); p.push_back(q.y); } dr_add(DE_PLINE, p, -1, "", 3.5, true);
        g_dpts.clear(); g_dlast_tool = "PLINE"; g_dtool.clear(); dr_prompt("Command:"); dr_refresh(); return;
    }
    if (!g_dtool.empty()) {
        if (g_dtool == "CIRCLE" && g_dpts.size() == 1 && s.find(',') == std::string::npos && s.find('<') == std::string::npos && s[0] != '@') {
            double r = atof(s.c_str()); if (r > 0) { dr_push_undo(); dr_add(DE_CIRCLE, {g_dpts[0].x, g_dpts[0].y, r}); g_dpts.clear(); g_dlast_tool = "CIRCLE"; g_dtool.clear(); dr_prompt("Command:"); dr_refresh(); return; }
        }
        double x, y;
        if (dr_parse_point(s, x, y)) { dr_point(x, y); return; }
    }
    dr_command(s);
}

// ---------------------------------------------------------------- windows
static WNDPROC g_dr_cmd_old = nullptr;
static LRESULT CALLBACK dr_cmd_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_KEYDOWN && w == VK_RETURN) { std::string s = read_edit(h); SetWindowTextA(h, ""); dr_enter(s); return 0; }
    if (m == WM_KEYDOWN && w == VK_ESCAPE) { SetWindowTextA(h, ""); g_dtool.clear(); g_dpts.clear(); g_dsel.clear(); dr_prompt("*Cancel*   Command:"); dr_refresh(); return 0; }
    if (m == WM_KEYDOWN && (w == VK_F3 || w == VK_F8 || w == VK_F9)) { SendMessageA(g_dr_canvas, WM_KEYDOWN, w, l); return 0; }
    if (m == WM_CHAR && (w == VK_RETURN || w == VK_ESCAPE)) return 0;
    if (m == WM_CHAR && w == ' ' && GetWindowTextLengthA(h) > 0 && g_dtool != "TEXT") { std::string s = read_edit(h); SetWindowTextA(h, ""); dr_enter(s); return 0; }  // space acts as Enter
    return CallWindowProcA(g_dr_cmd_old, h, m, w, l);
}
static LRESULT CALLBACK dr_canvas_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_PAINT: { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_draft(h, dc); EndPaint(h, &ps); return 0; }
        case WM_ERASEBKGND: return 1;
        case WM_SETCURSOR: if (LOWORD(l) == HTCLIENT) { SetCursor(nullptr); return TRUE; } break;  // the crosshair is drawn
        case WM_MOUSEMOVE: {
            g_dmx = (short)LOWORD(l); g_dmy = (short)HIWORD(l);
            if (g_dpan) { double ox, oy, nx, ny; dr_world(g_dpx, g_dpy, ox, oy); dr_world(g_dmx, g_dmy, nx, ny); g_dcx -= nx - ox; g_dcy -= ny - oy; g_dpx = g_dmx; g_dpy = g_dmy; }
            dr_world(g_dmx, g_dmy, g_dwx, g_dwy);
            g_dsnap_hit = g_dosnap && !g_dtool.empty() && dr_osnap(g_dwx, g_dwy, g_dsnx, g_dsny, g_dsnap_kind);
            TRACKMOUSEEVENT te = {sizeof(TRACKMOUSEEVENT), TME_LEAVE, h, 0}; TrackMouseEvent(&te);
            dr_refresh(); return 0;
        }
        case WM_MOUSELEAVE: g_dmx = g_dmy = -100; dr_refresh(); return 0;
        case WM_MOUSEWHEEL: {
            POINT p = {(short)LOWORD(l), (short)HIWORD(l)}; ScreenToClient(h, &p);
            double bx, by; dr_world(p.x, p.y, bx, by);
            g_dscale *= GET_WHEEL_DELTA_WPARAM(w) > 0 ? 1.2 : 1 / 1.2; g_dscale = std::max(1e-4, std::min(1e4, g_dscale));
            double ax, ay; dr_world(p.x, p.y, ax, ay); g_dcx += bx - ax; g_dcy += by - ay;
            dr_refresh(); return 0;
        }
        case WM_MBUTTONDOWN: g_dpan = true; g_dpx = (short)LOWORD(l); g_dpy = (short)HIWORD(l); SetCapture(h); return 0;
        case WM_MBUTTONUP: g_dpan = false; ReleaseCapture(); return 0;
        case WM_MBUTTONDBLCLK: dr_zoom_extents(); dr_refresh(); return 0;
        case WM_LBUTTONDOWN: {
            SetFocus(g_dr_cmd);
            double x, y; dr_cursor_point(x, y);
            if (g_dtool.empty()) {  // picking
                int i = dr_pick(g_dwx, g_dwy);
                if (!(w & MK_SHIFT)) g_dsel.clear();
                if (i >= 0) g_dsel.insert(i);
                dr_refresh(); return 0;
            }
            dr_point(x, y); return 0;
        }
        case WM_RBUTTONDOWN: dr_enter(""); return 0;  // right click = Enter
        case WM_KEYDOWN:
            if (w == VK_F3) { g_dosnap = !g_dosnap; dr_prompt(g_dosnap ? "<Osnap on>" : "<Osnap off>"); }
            else if (w == VK_F8) { g_dortho = !g_dortho; dr_prompt(g_dortho ? "<Ortho on>" : "<Ortho off>"); }
            else if (w == VK_F9) { g_dsnap = !g_dsnap; dr_prompt(g_dsnap ? "<Snap on>" : "<Snap off>"); }
            else if (w == VK_DELETE && !g_dsel.empty()) dr_command("ERASE");
            dr_refresh(); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}
static void dr_layout() {
    RECT rc; GetClientRect(g_dr, &rc);
    int W = rc.right, H = rc.bottom, lw = 168;
    for (int k = 0; k < NDRT; ++k) MoveWindow(GetDlgItem(g_dr, ID_DR_T0 + k), 6, 40 + k * 27, lw - 12, 25, TRUE);
    MoveWindow(g_dr_layer, 6, 6, lw - 12, 300, TRUE);
    MoveWindow(g_dr_canvas, lw, 0, W - lw, H - 56, TRUE);
    MoveWindow(g_dr_prompt, lw + 6, H - 54, W - lw - 12, 22, TRUE);
    MoveWindow(g_dr_cmd, lw + 6, H - 30, W - lw - 12, 24, TRUE);
}
static LRESULT CALLBACK dr_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            HINSTANCE hi = GetModuleHandleA(nullptr);
            g_dr_canvas = CreateWindowExA(0, "AeroSynapseDraftCanvas", "", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            for (int k = 0; k < NDRT; ++k) CreateWindowExA(0, "BUTTON", (std::string(DR_TOOLS[k][2]) + "  (" + DR_TOOLS[k][1] + ")").c_str(), WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)(INT_PTR)(ID_DR_T0 + k), hi, nullptr);
            g_dr_layer = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL, 0, 0, 0, 0, h, (HMENU)ID_DR_LAYER, hi, nullptr);
            for (const DLayer& L : g_dl) SendMessageA(g_dr_layer, CB_ADDSTRING, 0, (LPARAM)L.name.c_str());
            SendMessageA(g_dr_layer, CB_SETCURSEL, g_dlayer, 0);
            g_dr_prompt = CreateWindowExA(0, "STATIC", g_dmsg.c_str(), WS_CHILD | WS_VISIBLE | SS_LEFTNOWORDWRAP, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_dr_cmd = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL, 0, 0, 0, 0, h, (HMENU)ID_DR_CMD, hi, nullptr);
            g_dr_cmd_old = (WNDPROC)SetWindowLongPtrA(g_dr_cmd, GWLP_WNDPROC, (LONG_PTR)dr_cmd_proc);
            static HFONT mono = CreateFontA(-14, 0, 0, 0, FW_NORMAL, 0, 0, 0, ANSI_CHARSET, 0, 0, CLEARTYPE_QUALITY, FIXED_PITCH, "Consolas");
            SendMessageA(g_dr_cmd, WM_SETFONT, (WPARAM)mono, TRUE); SendMessageA(g_dr_prompt, WM_SETFONT, (WPARAM)mono, TRUE); SendMessageA(g_dr_layer, WM_SETFONT, (WPARAM)g_font, TRUE);
            return 0;
        }
        case WM_SIZE: dr_layout(); return 0;
        case WM_SETFOCUS: SetFocus(g_dr_cmd); return 0;
        case WM_COMMAND:
            if (LOWORD(w) >= ID_DR_T0 && LOWORD(w) < ID_DR_T0 + NDRT) { dr_command(DR_TOOLS[LOWORD(w) - ID_DR_T0][0]); SetFocus(g_dr_cmd); }
            if (LOWORD(w) == ID_DR_LAYER && HIWORD(w) == CBN_SELCHANGE) {
                g_dlayer = (int)SendMessageA(g_dr_layer, CB_GETCURSEL, 0, 0);
                if (!g_dsel.empty()) { dr_push_undo(); for (int i : g_dsel) if (i < (int)g_de.size()) g_de[(size_t)i].layer = g_dlayer; }  // the picked entities move to the layer
                SetFocus(g_dr_cmd); dr_refresh();
            }
            return 0;
        case WM_DRAWITEM: {
            DRAWITEMSTRUCT* d = (DRAWITEMSTRUCT*)l;
            int k = (int)d->CtlID - ID_DR_T0;
            bool active = k >= 0 && k < NDRT && g_dtool == DR_TOOLS[k][0];
            draw_button(d, active ? gcol(250, 160, 30) : k >= 16 ? gcol(99, 102, 241) : k >= 8 ? gcol(20, 140, 170) : gcol(34, 120, 200));
            return TRUE;
        }
        case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkColor(dc, RGB(26, 32, 40)); SetTextColor(dc, RGB(250, 210, 120)); static HBRUSH b = CreateSolidBrush(RGB(26, 32, 40)); return (LRESULT)b; }
        case WM_CTLCOLOREDIT: { HDC dc = (HDC)w; SetBkColor(dc, RGB(16, 20, 26)); SetTextColor(dc, RGB(235, 240, 248)); static HBRUSH b = CreateSolidBrush(RGB(16, 20, 26)); return (LRESULT)b; }
        case WM_CLOSE: ShowWindow(h, SW_HIDE); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}
static void dr_save_png() {  // the drawing as a picture at three times the window, any picture format
    std::string p = g_dfile.empty() ? std::string("drawing.png") : g_dfile.substr(0, g_dfile.find_last_of('.')) + ".png";
    if (!ask_path(true, IMG_FILTER, "png", p)) return;
    RECT rc; GetClientRect(g_dr_canvas, &rc);
    int k = 3, W = std::max(1L, rc.right) * k, H = std::max(1L, rc.bottom) * k;
    gp::Bitmap bm(W, H, PixelFormat32bppARGB);
    double keep = g_dscale; g_dscale *= k; g_dr_fw = g_dr_fw2 = W; g_dr_fh = g_dr_fh2 = H;
    {
        gp::Graphics g(&bm); HDC dc = g.GetHDC(); paint_draft(g_dr_canvas, dc); g.ReleaseHDC(dc);
    }
    g_dscale = keep; g_dr_fw = g_dr_fw2 = g_dr_fh = g_dr_fh2 = 0;
    if (save_image_as(&bm, p)) dr_prompt("PNGOUT: " + p + " (" + tostr(W) + " x " + tostr(H) + ")"); else dr_prompt("PNGOUT failed: " + p);
}
static void open_draft(const char* cmd = nullptr) {
    if (!g_dr) {
        g_dr = CreateWindowExA(0, "AeroSynapseDraft", "2D drawing (AutoCAD-style drafting, DXF)", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1440, 860, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
        dr_layout(); dr_zoom_extents();
    }
    ShowWindow(g_dr, SW_SHOW); SetForegroundWindow(g_dr); SetFocus(g_dr_cmd);
    if (cmd) dr_command(cmd); else { dr_prompt(g_dmsg); dr_refresh(); }
}
static void register_draft_classes(HINSTANCE hi, HICON icon) {
    WNDCLASSA a; memset(&a, 0, sizeof a);
    a.lpfnWndProc = dr_proc; a.hInstance = hi; a.hCursor = LoadCursor(nullptr, IDC_ARROW); a.hbrBackground = CreateSolidBrush(RGB(26, 32, 40)); a.lpszClassName = "AeroSynapseDraft"; a.hIcon = icon;
    RegisterClassA(&a);
    a.lpfnWndProc = dr_canvas_proc; a.style = CS_DBLCLKS; a.hbrBackground = nullptr; a.lpszClassName = "AeroSynapseDraftCanvas";
    RegisterClassA(&a);
}

// ---------------------------------------------------------------- the sketcher's DXF import / export and its link to the drawing
static int dr_largest_outline(const std::vector<DEnt>& E) {
    int best = -1; double ba = -1;
    for (size_t i = 0; i < E.size(); ++i) {
        const DEnt& e = E[i]; double a = -1;
        if (e.type == DE_PLINE && e.p.size() >= 6) { a = 0; size_t n = e.p.size() / 2; for (size_t k = 0; k < n; ++k) { size_t j = (k + 1) % n; a += e.p[2 * k] * e.p[2 * j + 1] - e.p[2 * j] * e.p[2 * k + 1]; } a = std::fabs(a) / 2; }
        else if (e.type == DE_CIRCLE) a = geo::PI * e.p[2] * e.p[2];
        if (a > ba) { ba = a; best = (int)i; }
    }
    return best;
}
static DEnt sk_as_polyline() {  // the sketch's curve in millimetres
    DEnt e; e.type = DE_PLINE; e.layer = 1; e.closed = !sk_revolving() && g_sk_shape != SH_SHEET;
    for (const geo::P2& q : sk_curve()) { e.p.push_back(q.x * 1000); e.p.push_back(q.y * 1000); }
    if (e.closed && e.p.size() >= 6) { e.p.pop_back(); e.p.pop_back(); }  // the closed curve repeats its first point
    return e;
}
static void sk_dxf_import() {
    std::string pth;
    if (!ask_path(false, "AutoCAD DXF (*.dxf)\0*.dxf\0All files\0*.*\0", "dxf", pth)) return;
    std::string txt, err; try { txt = js::read_file(pth); } catch (...) { return; }
    std::vector<DEnt> keep = g_de; g_de.clear(); dxf_read(txt, err);
    int best = dr_largest_outline(g_de);
    DEnt e; if (best >= 0) e = g_de[(size_t)best];
    g_de = keep;
    if (best < 0) { MessageBoxA(g_sk, "The DXF file has no polyline or circle to use as an outline.", "Import DXF", MB_ICONINFORMATION); return; }
    g_sk_shape = SH_POLY; g_sk_smooth = false; g_sk_pts.clear();
    if (g_sk_shape_cb) SendMessageA(g_sk_shape_cb, CB_SETCURSEL, SH_POLY, 0);
    if (e.type == DE_CIRCLE) for (int k = 0; k < 48; ++k) g_sk_pts.push_back(geo::P2((e.p[0] + e.p[2] * std::cos(2 * geo::PI * k / 48)) / 1000, (e.p[1] + e.p[2] * std::sin(2 * geo::PI * k / 48)) / 1000));
    else for (size_t i = 0; i + 1 < e.p.size(); i += 2) g_sk_pts.push_back(geo::P2(e.p[i] / 1000, e.p[i + 1] / 1000));
    status("outline imported from " + pth + " (" + tostr((int)g_sk_pts.size()) + " points, mm to m)");
    InvalidateRect(g_sk_canvas, nullptr, FALSE);
}
static void sk_dxf_export() {
    if (g_sk_pts.size() < 2) { MessageBoxA(g_sk, "Draw a shape first.", "Sketch", MB_ICONINFORMATION); return; }
    std::vector<DEnt> keep = g_de; g_de.clear(); g_de.push_back(sk_as_polyline());
    std::string pth = "sketch.dxf";
    if (ask_path(true, "AutoCAD DXF (*.dxf)\0*.dxf\0", "dxf", pth)) { try { js::write_file(pth, dxf_write()); status("sketch written to " + pth + " (AutoCAD R12 DXF, millimetres)"); } catch (...) {} }
    g_de = keep;
}
static void sk_to_draft() {
    if (g_sk_pts.size() < 2) { MessageBoxA(g_sk, "Draw a shape first.", "Sketch", MB_ICONINFORMATION); return; }
    open_draft(nullptr);
    dr_push_undo(); g_de.push_back(sk_as_polyline()); dr_zoom_extents();
    dr_prompt("the sketch is in the drawing as a polyline (mm): dimension it (DLI), add views (VIEWS) or save it (DXFOUT)");
    dr_refresh();
}

// ---------------------------------------------------------------- vector output: SVG and PDF of the drawing (on white paper)
struct DPrim { int kind = 0; double a = 0, b = 0, c = 0, d = 0, e = 0; int layer = 1; std::string text; double h = 3.5, rot = 0; bool centre = false; };  // 0 line, 1 circle, 2 arc (deg), 3 text
static void dr_explode(std::vector<DPrim>& P) {
    auto line = [&](double x1, double y1, double x2, double y2, int L) { DPrim q; q.kind = 0; q.a = x1; q.b = y1; q.c = x2; q.d = y2; q.layer = L; P.push_back(q); };
    auto text = [&](double x, double y, double h, double rot, const std::string& t, int L, bool c) { DPrim q; q.kind = 3; q.a = x; q.b = y; q.h = h; q.rot = rot; q.text = t; q.layer = L; q.centre = c; P.push_back(q); };
    for (const DEnt& e : g_de) {
        switch (e.type) {
            case DE_LINE: line(e.p[0], e.p[1], e.p[2], e.p[3], e.layer); break;
            case DE_CIRCLE: { DPrim q; q.kind = 1; q.a = e.p[0]; q.b = e.p[1]; q.c = e.p[2]; q.layer = e.layer; P.push_back(q); break; }
            case DE_ARC: { DPrim q; q.kind = 2; q.a = e.p[0]; q.b = e.p[1]; q.c = e.p[2]; q.d = e.p[3]; q.e = e.p[4]; q.layer = e.layer; P.push_back(q); break; }
            case DE_PLINE: { size_t n = e.p.size() / 2; for (size_t i = 0; i + 1 < n; ++i) line(e.p[2 * i], e.p[2 * i + 1], e.p[2 * i + 2], e.p[2 * i + 3], e.layer); if (e.closed && n > 2) line(e.p[2 * n - 2], e.p[2 * n - 1], e.p[0], e.p[1], e.layer); break; }
            case DE_TEXT: text(e.p[0], e.p[1], e.h, e.p.size() > 2 ? e.p[2] : 0, e.text, e.layer, false); break;
            case DE_DIMLIN: {
                double ax, ay, bx, by, nx, ny; dr_dim_geom(e, ax, ay, bx, by, nx, ny);
                double sgn = ((ax - e.p[0]) * nx + (ay - e.p[1]) * ny) >= 0 ? 1 : -1;
                line(e.p[0] + nx * sgn, e.p[1] + ny * sgn, ax + nx * sgn * 2, ay + ny * sgn * 2, e.layer);
                line(e.p[2] + nx * sgn, e.p[3] + ny * sgn, bx + nx * sgn * 2, by + ny * sgn * 2, e.layer);
                line(ax, ay, bx, by, e.layer);
                double ux = bx - ax, uy = by - ay, ul = std::max(1e-12, std::hypot(ux, uy)); ux /= ul; uy /= ul;
                for (int k = 0; k < 2; ++k) { double tx = k ? bx : ax, ty = k ? by : ay, d = k ? -1 : 1; line(tx, ty, tx + d * 2.5 * ux + 0.8 * nx, ty + d * 2.5 * uy + 0.8 * ny, e.layer); line(tx, ty, tx + d * 2.5 * ux - 0.8 * nx, ty + d * 2.5 * uy - 0.8 * ny, e.layer); }
                double ang = std::atan2(by - ay, bx - ax) * 180 / geo::PI; if (ang > 90.01 || ang < -89.99) ang += 180;
                std::string t = e.text.empty() ? fmtd(std::round(std::hypot(e.p[2] - e.p[0], e.p[3] - e.p[1]) * 100) / 100) : e.text;
                text(0.5 * (ax + bx) + nx * sgn * 0.8, 0.5 * (ay + by) + ny * sgn * 0.8, e.h, ang, t, e.layer, true);
                break;
            }
            case DE_DIMRAD: {
                double ex = e.p[0] + e.p[2] * std::cos(e.p[3]), ey = e.p[1] + e.p[2] * std::sin(e.p[3]);
                line(e.p[0], e.p[1], ex, ey, e.layer);
                text(ex + 1.5 * std::cos(e.p[3]), ey + 1.5 * std::sin(e.p[3]), e.h, 0, e.text.empty() ? "R" + fmtd(std::round(e.p[2] * 100) / 100) : e.text, e.layer, false);
                break;
            }
        }
    }
}
static void dr_paper_style(int layer, int& r, int& g, int& b, double& w, int& lt) {  // pen colours and widths on white paper (mm)
    const DLayer& L = g_dl[(size_t)std::max(0, std::min((int)g_dl.size() - 1, layer))];
    lt = L.ltype; w = layer == 1 ? 0.5 : 0.25;
    switch (L.aci) { case 1: r = 200; g = 30; b = 30; break; case 2: r = 150; g = 110; b = 0; break; case 3: r = 0; g = 130; b = 60; break; case 4: r = 0; g = 90; b = 170; break;
                     case 5: r = 40; g = 60; b = 200; break; case 6: r = 160; g = 30; b = 160; break; case 8: r = 120; g = 125; b = 135; break; default: r = g = b = 0; }
}
static std::string xml_esc(const std::string& t) { std::string o; for (char c : t) { if (c == '<') o += "&lt;"; else if (c == '>') o += "&gt;"; else if (c == '&') o += "&amp;"; else o += c; } return o; }
static std::string pdf_esc(const std::string& t) { std::string o; for (char c : t) { if (c == '(' || c == ')' || c == '\\') o += '\\'; o += c; } return o; }
static void dr_vector_out(bool pdf) {
    if (g_de.empty()) { dr_prompt("nothing to save: the drawing is empty"); return; }
    std::vector<DPrim> P; dr_explode(P);
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (const DEnt& e : g_de) dr_bounds(e, x0, y0, x1, y1);
    x0 -= 10; y0 -= 10; x1 += 10; y1 += 10;
    double Wm = x1 - x0, Hm = y1 - y0;
    std::string p = g_dfile.empty() ? std::string(pdf ? "drawing.pdf" : "drawing.svg") : g_dfile.substr(0, g_dfile.find_last_of('.')) + (pdf ? ".pdf" : ".svg");
    if (!ask_path(true, pdf ? "PDF document (*.pdf)\0*.pdf\0" : "SVG drawing (*.svg)\0*.svg\0", pdf ? "pdf" : "svg", p)) return;
    std::string out; char b[600];
    if (!pdf) {
        snprintf(b, sizeof b, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.2fmm\" height=\"%.2fmm\" viewBox=\"0 0 %.3f %.3f\">\n<rect width=\"100%%\" height=\"100%%\" fill=\"white\"/>\n", Wm, Hm, Wm, Hm);
        out += b;
        for (const DPrim& q : P) {
            int r, g, bb, lt; double w; dr_paper_style(q.layer, r, g, bb, w, lt);
            std::string dash = lt == 1 ? " stroke-dasharray=\"3 1.5\"" : lt == 2 ? " stroke-dasharray=\"8 1.5 1.5 1.5\"" : "";
            auto X = [&](double x) { return x - x0; }; auto Y = [&](double y) { return y1 - y; };
            if (q.kind == 0) snprintf(b, sizeof b, "<line x1=\"%.4f\" y1=\"%.4f\" x2=\"%.4f\" y2=\"%.4f\" stroke=\"rgb(%d,%d,%d)\" stroke-width=\"%.2f\" stroke-linecap=\"round\"%s/>\n", X(q.a), Y(q.b), X(q.c), Y(q.d), r, g, bb, w, dash.c_str());
            else if (q.kind == 1) snprintf(b, sizeof b, "<circle cx=\"%.4f\" cy=\"%.4f\" r=\"%.4f\" fill=\"none\" stroke=\"rgb(%d,%d,%d)\" stroke-width=\"%.2f\"%s/>\n", X(q.a), Y(q.b), q.c, r, g, bb, w, dash.c_str());
            else if (q.kind == 2) {
                double a0 = q.d * geo::PI / 180, a1 = q.e * geo::PI / 180, sw = std::fmod(q.e - q.d + 720.0, 360.0);
                snprintf(b, sizeof b, "<path d=\"M %.4f %.4f A %.4f %.4f 0 %d 0 %.4f %.4f\" fill=\"none\" stroke=\"rgb(%d,%d,%d)\" stroke-width=\"%.2f\"%s/>\n", X(q.a + q.c * std::cos(a0)), Y(q.b + q.c * std::sin(a0)), q.c, q.c, sw > 180 ? 1 : 0,
                         X(q.a + q.c * std::cos(a1)), Y(q.b + q.c * std::sin(a1)), r, g, bb, w, dash.c_str());
            } else snprintf(b, sizeof b, "<text x=\"%.4f\" y=\"%.4f\" font-family=\"Consolas, monospace\" font-size=\"%.3f\" fill=\"rgb(%d,%d,%d)\"%s transform=\"rotate(%.3f %.4f %.4f)\">%s</text>\n", X(q.a), Y(q.b), q.h * 1.2, r, g, bb,
                            q.centre ? " text-anchor=\"middle\"" : "", -q.rot, X(q.a), Y(q.b), xml_esc(q.text).c_str());
            out += b;
        }
        out += "</svg>\n";
    } else {
        double k = 72.0 / 25.4;  // mm to points
        std::string cs;
        for (const DPrim& q : P) {
            int r, g, bb, lt; double w; dr_paper_style(q.layer, r, g, bb, w, lt);
            auto X = [&](double x) { return (x - x0) * k; }; auto Y = [&](double y) { return (y - y0) * k; };
            snprintf(b, sizeof b, "%.3f %.3f %.3f RG %.3f %.3f %.3f rg %.3f w %s 1 J\n", r / 255.0, g / 255.0, bb / 255.0, r / 255.0, g / 255.0, bb / 255.0, w * k, lt == 1 ? "[8.5 4.25] 0 d" : lt == 2 ? "[22.7 4.25 4.25 4.25] 0 d" : "[] 0 d");
            cs += b;
            if (q.kind == 0) { snprintf(b, sizeof b, "%.3f %.3f m %.3f %.3f l S\n", X(q.a), Y(q.b), X(q.c), Y(q.d)); cs += b; }
            else if (q.kind == 1 || q.kind == 2) {
                double a0 = q.kind == 1 ? 0 : q.d, sw = q.kind == 1 ? 360 : std::fmod(q.e - q.d + 720.0, 360.0); if (sw < 1e-9) sw = 360;
                int n = std::max(8, (int)(sw / 3));
                for (int i = 0; i <= n; ++i) { double a = (a0 + sw * i / n) * geo::PI / 180; snprintf(b, sizeof b, "%.3f %.3f %s\n", X(q.a + q.c * std::cos(a)), Y(q.b + q.c * std::sin(a)), i ? "l" : "m"); cs += b; }
                cs += "S\n";
            } else {
                double a = q.rot * geo::PI / 180, fs = q.h * 1.2 * k, ca = std::cos(a), sa = std::sin(a), x = X(q.a), y = Y(q.b);
                if (q.centre) { double tw = 0.6 * fs * q.text.size(); x -= 0.5 * tw * ca; y -= 0.5 * tw * sa; }
                snprintf(b, sizeof b, "BT /F1 %.3f Tf %.5f %.5f %.5f %.5f %.3f %.3f Tm (", fs, ca, sa, -sa, ca, x, y);
                cs += b; cs += pdf_esc(q.text); cs += ") Tj ET\n";
            }
        }
        std::vector<size_t> off;
        out = "%PDF-1.4\n";
        auto obj = [&](const std::string& body) { off.push_back(out.size()); out += tostr((int)off.size()) + " 0 obj\n" + body + "\nendobj\n"; };
        obj("<< /Type /Catalog /Pages 2 0 R >>");
        obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
        snprintf(b, sizeof b, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>", Wm * k, Hm * k); obj(b);
        obj("<< /Length " + tostr((int)cs.size()) + " >>\nstream\n" + cs + "endstream");
        obj("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>");
        size_t xref = out.size();
        out += "xref\n0 " + tostr((int)off.size() + 1) + "\n0000000000 65535 f \n";
        for (size_t o : off) { snprintf(b, sizeof b, "%010u 00000 n \n", (unsigned)o); out += b; }
        out += "trailer\n<< /Size " + tostr((int)off.size() + 1) + " /Root 1 0 R >>\nstartxref\n" + tostr((long)xref) + "\n%%EOF\n";
    }
    FILE* f = fopen(p.c_str(), "wb");
    if (!f) { dr_prompt("could not write " + p); return; }
    fwrite(out.data(), 1, out.size(), f); fclose(f);
    dr_prompt(std::string(pdf ? "PDFOUT" : "SVGOUT") + ": " + tostr((int)P.size()) + " vector pieces, " + fmtd(std::round(Wm)) + " x " + fmtd(std::round(Hm)) + " mm, written to " + p);
}
