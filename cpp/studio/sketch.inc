// Sketcher (2D sketching for part design): pick a sketch plane, draw, apply an operation.
//   Plane      XY (top view, normal z), XZ (side view, normal y) or YZ (front view, normal x), at an offset
//   Shape      polyline / spline (click points), rectangle, circle and slot (drag)
//   Operation  Pad (extrude, add), Pocket (extrude, cut), Shaft (revolve about the sketch's horizontal axis, add),
//              Groove (revolve, cut), Loft (the section through three stations along the plane's normal)
// Pads and pockets are symmetric about the sketch plane (CATIA's "mirrored extent"), depth = Size.
// Shapes also: ellipse, regular polygon, rounded rectangle, star, involute gear, arc slot and triangle (Sides sets the
// polygon's sides, the star's points and the gear's teeth). Edit: mirror, offset, fillet the corners, rotate, scale,
// exact sizes typed in, points typed as coordinates.
// Mouse: left click adds a point (or drags a rectangle, circle or slot), drag moves a point, right click deletes one,
// wheel zooms, middle drag pans. Enter finishes, Backspace removes the last point, Esc closes.

static const int SK_REVOLVE = 0, SK_EXTRUDE = 1, SK_LOFT = 2;
static const int OP_PAD = 0, OP_POCKET = 1, OP_SHAFT = 2, OP_GROOVE = 3, OP_LOFT = 4;
static const int SH_POLY = 0, SH_RECT = 1, SH_CIRCLE = 2, SH_SLOT = 3, SH_ELLIPSE = 4, SH_POLYGON = 5, SH_RRECT = 6, SH_STAR = 7, SH_GEAR = 8, SH_ARCSLOT = 9, SH_TRIANGLE = 10, SH_NACA = 11, SH_LSEC = 12, SH_TSEC = 13, SH_ISEC = 14, SH_KEYHOLE = 15, SH_CROSS = 16, SH_CHANNEL = 17, SH_ZSEC = 18, SH_HAT = 19, SH_SHEET = 20;
static const int NSHAPE = 21;
static int g_sk_pat = 0, g_sk_pat_n = 1; static double g_sk_pat_dx = 0, g_sk_pat_dy = 0;  // pattern on finish: 0 none, 1 linear, 2 circular
static HWND g_sk, g_sk_canvas, g_sk_size, g_sk_btn_snap, g_sk_btn_smooth, g_sk_size_lbl, g_sk_plane_cb, g_sk_shape_cb, g_sk_op_cb, g_sk_off, g_sk_off_lbl, g_sk_sides, g_sk_sides_lbl, g_sk_btn_edit;
static int g_sk_plane = 1, g_sk_shape = SH_POLY, g_sk_op = OP_SHAFT, g_sk_drag = -1, g_sk_hover = -1, g_sk_count = 0;
static std::vector<geo::P2> g_sk_pts;
static bool sk_clicked() { return g_sk_shape == SH_POLY || g_sk_shape == SH_SHEET; }  // shapes made by clicking points (the others are dragged)
static double g_sk_cx = 8, g_sk_cy = 1, g_sk_scale = 50;
static bool g_sk_snap = true, g_sk_smooth = true, g_sk_pan = false, g_sk_dragshape = false;
static int g_sk_mx = 0, g_sk_my = 0;
static double g_sk_wx = 0, g_sk_wy = 0, g_sk_ax = 0, g_sk_ay = 0;
enum { ID_SK_FINISH = 8600, ID_SK_UNDO, ID_SK_CLEAR, ID_SK_SNAP, ID_SK_SMOOTH, ID_SK_SIZE, ID_SK_PLANE, ID_SK_SHAPE, ID_SK_OP, ID_SK_OFF, ID_SK_SIDES, ID_SK_EDIT,
       ID_SKE_MIRROR_V = 8640, ID_SKE_MIRROR_H, ID_SKE_OFFSET, ID_SKE_FILLET, ID_SKE_ROTATE, ID_SKE_SCALE, ID_SKE_SIZE, ID_SKE_POINTS, ID_SKE_PATTERN, ID_SKE_CHAMFER, ID_SKE_REVERSE, ID_SKE_DXFIN, ID_SKE_DXFOUT, ID_SKE_DRAFT };
static const char* PLANE_NAMES[3] = {"XY plane (top view)", "XZ plane (side view)", "YZ plane (front view)"};
static const char* AXIS_U[3] = {"x", "x", "y"};
static const char* AXIS_V[3] = {"y", "z", "z"};

static bool sk_revolving() { return g_sk_op == OP_SHAFT || g_sk_op == OP_GROOVE; }

static void sk_to_world(int sx, int sy, double& wx, double& wy) {
    RECT rc; GetClientRect(g_sk_canvas, &rc);
    wx = g_sk_cx + (sx - rc.right / 2.0) / g_sk_scale;
    wy = g_sk_cy - (sy - rc.bottom / 2.0) / g_sk_scale;
}
static gp::PointF sk_to_screen(double wx, double wy) {
    RECT rc; GetClientRect(g_sk_canvas, &rc);
    return gp::PointF((float)(rc.right / 2.0 + (wx - g_sk_cx) * g_sk_scale), (float)(rc.bottom / 2.0 - (wy - g_sk_cy) * g_sk_scale));
}

static std::vector<geo::P2> sk_curve() {
    if (g_sk_pts.size() < 2) return g_sk_pts;
    bool closed = !sk_revolving() && g_sk_shape != SH_SHEET;
    if (g_sk_smooth && g_sk_shape == SH_POLY && g_sk_pts.size() >= 3) return geo::spline2(g_sk_pts, closed, 10);
    std::vector<geo::P2> v = g_sk_pts;
    if (closed) v.push_back(v.front());
    return v;
}

static void neon(gp::Graphics& g, const std::vector<gp::PointF>& pts, gp::Color c, float w) {
    if (pts.size() < 2) return;
    for (int k = 3; k >= 1; --k) { gp::Pen p(gcol(c.GetR(), c.GetG(), c.GetB(), 35), w + 4.0f * k); p.SetLineJoin(gp::LineJoinRound); g.DrawLines(&p, pts.data(), (INT)pts.size()); }
    gp::Pen core(gshade(c, 0.45f), w); core.SetLineJoin(gp::LineJoinRound);
    g.DrawLines(&core, pts.data(), (INT)pts.size());
}

static int sk_sides() { int n = g_sk_sides ? atoi(read_edit(g_sk_sides).c_str()) : 6; return std::max(3, std::min(200, n)); }
static void sk_shape_points(double ax, double ay, double bx, double by) {  // every dragged shape, from its two drag points
    g_sk_pts.clear();
    const double PI = geo::PI;
    double dx = bx - ax, dy = by - ay, L = std::hypot(dx, dy), th0 = std::atan2(dy, dx);
    int n = sk_sides();
    switch (g_sk_shape) {
        case SH_RECT: g_sk_pts = {geo::P2(ax, ay), geo::P2(bx, ay), geo::P2(bx, by), geo::P2(ax, by)}; break;
        case SH_CIRCLE: for (int k = 0; k < 36; ++k) { double t = 2 * PI * k / 36; g_sk_pts.push_back(geo::P2(ax + L * std::cos(t), ay + L * std::sin(t))); } break;
        case SH_SLOT: {
            double r = std::max(0.02, 0.18 * L);
            if (L < 1e-6) return;
            double ux = dx / L, uy = dy / L, nx = -uy, ny = ux, base = std::atan2(ny, nx);
            for (int k = 0; k <= 12; ++k) { double t = base + PI * k / 12; g_sk_pts.push_back(geo::P2(bx + r * std::cos(t), by + r * std::sin(t))); }
            for (int k = 0; k <= 12; ++k) { double t = base + PI + PI * k / 12; g_sk_pts.push_back(geo::P2(ax + r * std::cos(t), ay + r * std::sin(t))); }
            break;
        }
        case SH_ELLIPSE: {  // in the dragged box
            double cx = 0.5 * (ax + bx), cy = 0.5 * (ay + by), ra = 0.5 * std::fabs(dx), rb = 0.5 * std::fabs(dy);
            for (int k = 0; k < 48; ++k) { double t = 2 * PI * k / 48; g_sk_pts.push_back(geo::P2(cx + ra * std::cos(t), cy + rb * std::sin(t))); }
            break;
        }
        case SH_POLYGON: for (int k = 0; k < n; ++k) { double t = th0 + 2 * PI * k / n; g_sk_pts.push_back(geo::P2(ax + L * std::cos(t), ay + L * std::sin(t))); } break;
        case SH_STAR: for (int k = 0; k < 2 * n; ++k) { double t = th0 + PI * k / n, r = k % 2 ? 0.45 * L : L; g_sk_pts.push_back(geo::P2(ax + r * std::cos(t), ay + r * std::sin(t))); } break;
        case SH_RRECT: {  // corners rounded with a fifth of the shorter side
            double x0 = std::min(ax, bx), x1 = std::max(ax, bx), y0 = std::min(ay, by), y1 = std::max(ay, by), r = 0.2 * std::min(x1 - x0, y1 - y0);
            double cxs[4] = {x1 - r, x0 + r, x0 + r, x1 - r}, cys[4] = {y0 + r, y0 + r, y1 - r, y1 - r}, st[4] = {-0.5 * PI, PI, 0.5 * PI, 0};
            int order[4] = {0, 3, 2, 1};
            for (int q : order) for (int k = 0; k <= 6; ++k) { double t = st[q] + 0.5 * PI * k / 6; g_sk_pts.push_back(geo::P2(cxs[q] + r * std::cos(t), cys[q] + r * std::sin(t))); }
            break;
        }
        case SH_GEAR: {  // an involute spur gear of n teeth whose tips reach the drag (module 2 R / (z + 2))
            if (L < 1e-6) return;
            double m = 2 * L / (n + 2);
            std::vector<geo::P2> tooth = geo::gear_tooth(m, std::max(6, n), 20);
            for (int k = 0; k < std::max(6, n); ++k) {
                double t = th0 + 2 * PI * k / std::max(6, n), c = std::cos(t), sn = std::sin(t);
                for (size_t i = 0; i + 1 < tooth.size(); ++i) g_sk_pts.push_back(geo::P2(ax + tooth[i].x * c - tooth[i].y * sn, ay + tooth[i].x * sn + tooth[i].y * c));
            }
            break;
        }
        case SH_ARCSLOT: {  // round the first point, from the +u direction to the drag, a quarter of its radius wide
            if (L < 1e-6) return;
            double w = 0.125 * L, sweep = th0 <= 1e-3 ? th0 + 2 * PI : th0;
            sweep = std::min(sweep, 1.9 * PI);
            int m = 32;
            for (int k = 0; k <= m; ++k) { double t = sweep * k / m; g_sk_pts.push_back(geo::P2(ax + (L + w) * std::cos(t), ay + (L + w) * std::sin(t))); }
            for (int k = 1; k < 8; ++k) { double t = PI * k / 8; double ex = ax + L * std::cos(sweep), ey = ay + L * std::sin(sweep); g_sk_pts.push_back(geo::P2(ex + w * std::cos(sweep + t), ey + w * std::sin(sweep + t))); }
            for (int k = m; k >= 0; --k) { double t = sweep * k / m; g_sk_pts.push_back(geo::P2(ax + (L - w) * std::cos(t), ay + (L - w) * std::sin(t))); }
            for (int k = 1; k < 8; ++k) { double t = PI + PI * k / 8; g_sk_pts.push_back(geo::P2(ax + L + w * std::cos(t), ay + w * std::sin(t))); }
            break;
        }
        case SH_NACA: {  // a NACA 4-digit airfoil: the code in Sides (e.g. 2412), the chord along the drag
            if (L < 1e-6) return;
            int code = g_sk_sides ? atoi(read_edit(g_sk_sides).c_str()) : 2412; if (code < 1 || code > 9999) code = 2412;
            double mc = (code / 1000) / 100.0, pc = ((code / 100) % 10) / 10.0, tc = (code % 100) / 100.0; if (tc <= 0) tc = 0.12;
            auto yt = [&](double x) { return 5 * tc * (0.2969 * std::sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1036 * x * x * x * x); };
            auto yc = [&](double x) { if (mc <= 0 || pc <= 0) return 0.0; return x < pc ? mc / (pc * pc) * (2 * pc * x - x * x) : mc / ((1 - pc) * (1 - pc)) * ((1 - 2 * pc) + 2 * pc * x - x * x); };
            std::vector<geo::P2> up, lo;
            for (int k = 0; k <= 36; ++k) { double x = 0.5 * (1 - std::cos(PI * k / 36)); up.push_back(geo::P2(x, yc(x) + yt(x))); lo.push_back(geo::P2(x, yc(x) - yt(x))); }
            double ca = std::cos(th0), sa = std::sin(th0);
            auto put = [&](const geo::P2& q) { g_sk_pts.push_back(geo::P2(ax + L * (q.x * ca - q.y * sa), ay + L * (q.x * sa + q.y * ca))); };
            for (int k = 36; k >= 0; --k) put(up[(size_t)k]);
            for (int k = 1; k < 36; ++k) put(lo[(size_t)k]);
            break;
        }
        case SH_LSEC: case SH_TSEC: case SH_ISEC: {  // structural sections in the dragged box, walls a quarter (L) or a fifth of the smaller side
            double x0 = std::min(ax, bx), x1 = std::max(ax, bx), y0 = std::min(ay, by), y1 = std::max(ay, by), xm = 0.5 * (x0 + x1);
            double th = (g_sk_shape == SH_LSEC ? 0.25 : 0.2) * std::min(x1 - x0, y1 - y0);
            if (th <= 0) return;
            if (g_sk_shape == SH_LSEC) g_sk_pts = {geo::P2(x0, y0), geo::P2(x1, y0), geo::P2(x1, y0 + th), geo::P2(x0 + th, y0 + th), geo::P2(x0 + th, y1), geo::P2(x0, y1)};
            else if (g_sk_shape == SH_TSEC) g_sk_pts = {geo::P2(xm - th / 2, y0), geo::P2(xm + th / 2, y0), geo::P2(xm + th / 2, y1 - th), geo::P2(x1, y1 - th), geo::P2(x1, y1), geo::P2(x0, y1), geo::P2(x0, y1 - th), geo::P2(xm - th / 2, y1 - th)};
            else g_sk_pts = {geo::P2(x0, y0), geo::P2(x1, y0), geo::P2(x1, y0 + th), geo::P2(xm + th / 2, y0 + th), geo::P2(xm + th / 2, y1 - th), geo::P2(x1, y1 - th), geo::P2(x1, y1), geo::P2(x0, y1), geo::P2(x0, y1 - th), geo::P2(xm - th / 2, y1 - th), geo::P2(xm - th / 2, y0 + th), geo::P2(x0, y0 + th)};
            break;
        }
        case SH_KEYHOLE: {  // a round hole at the first point and a slot to the drag
            if (L < 1e-6) return;
            double r = 0.35 * L, w = 0.45 * r, ph = std::asin(std::min(0.99, w / r));
            double ca = std::cos(th0 + PI / 2), sa = std::sin(th0 + PI / 2);  // local -y along the drag
            auto put = [&](double x, double y) { g_sk_pts.push_back(geo::P2(ax + x * ca - y * sa, ay + x * sa + y * ca)); };
            for (int k = 0; k <= 30; ++k) { double t = -PI / 2 + ph + (2 * PI - 2 * ph) * k / 30; put(r * std::cos(t), r * std::sin(t)); }
            put(-w, -L); put(w, -L);
            break;
        }
        case SH_CROSS: {  // a plus: arms to the drag, a third of their length wide
            if (L < 1e-6) return;
            double h = L / 6, ca = std::cos(th0), sa = std::sin(th0);
            double P[12][2] = {{L, -h}, {L, h}, {h, h}, {h, L}, {-h, L}, {-h, h}, {-L, h}, {-L, -h}, {-h, -h}, {-h, -L}, {h, -L}, {h, -h}};
            for (auto& q : P) g_sk_pts.push_back(geo::P2(ax + q[0] * ca - q[1] * sa, ay + q[0] * sa + q[1] * ca));
            break;
        }
        case SH_CHANNEL: case SH_ZSEC: case SH_HAT: {  // C (channel), Z and hat sections in the dragged box, walls a sixth of the smaller side
            double x0 = std::min(ax, bx), x1 = std::max(ax, bx), y0 = std::min(ay, by), y1 = std::max(ay, by), th = std::min(x1 - x0, y1 - y0) / 6;
            if (th <= 0) return;
            if (g_sk_shape == SH_CHANNEL) g_sk_pts = {geo::P2(x0, y0), geo::P2(x1, y0), geo::P2(x1, y0 + th), geo::P2(x0 + th, y0 + th), geo::P2(x0 + th, y1 - th), geo::P2(x1, y1 - th), geo::P2(x1, y1), geo::P2(x0, y1)};
            else if (g_sk_shape == SH_ZSEC) { double xm = 0.5 * (x0 + x1); g_sk_pts = {geo::P2(xm - th / 2, y0), geo::P2(x1, y0), geo::P2(x1, y0 + th), geo::P2(xm + th / 2, y0 + th), geo::P2(xm + th / 2, y1), geo::P2(x0, y1), geo::P2(x0, y1 - th), geo::P2(xm - th / 2, y1 - th)}; }
            else {  // a top-hat stiffener: two flanges at the bottom, the crown at the top
                double w = x1 - x0, a = x0 + 0.25 * w, b = x1 - 0.25 * w;
                g_sk_pts = {geo::P2(x0, y0), geo::P2(a + th, y0), geo::P2(a + th, y1 - th), geo::P2(b - th, y1 - th), geo::P2(b - th, y0), geo::P2(x1, y0), geo::P2(x1, y0 + th), geo::P2(b, y0 + th), geo::P2(b, y1), geo::P2(a, y1), geo::P2(a, y0 + th), geo::P2(x0, y0 + th)};
            }
            break;
        }
        case SH_TRIANGLE: {  // equilateral on the dragged base
            if (L < 1e-6) return;
            double h = 0.8660254 * L, mx = 0.5 * (ax + bx), my = 0.5 * (ay + by);
            g_sk_pts = {geo::P2(ax, ay), geo::P2(bx, by), geo::P2(mx - dy / L * h, my + dx / L * h)};
            break;
        }
    }
}
// ---------------------------------------------------------------- editing the drawn points
static geo::P2 sk_centroid() { double x = 0, y = 0; for (auto& q : g_sk_pts) { x += q.x; y += q.y; } double n = std::max<size_t>(1, g_sk_pts.size()); return geo::P2(x / n, y / n); }
static void sk_offset(double d) {  // outwards by d (a closed outline), mitred corners
    size_t n = g_sk_pts.size(); if (n < 3) return;
    double A = 0; for (size_t i = 0; i < n; ++i) { const geo::P2 &a = g_sk_pts[i], &b = g_sk_pts[(i + 1) % n]; A += a.x * b.y - b.x * a.y; }
    double sg = A > 0 ? 1 : -1;  // counter-clockwise: the outward normal of an edge (dx, dy) is (dy, -dx)
    std::vector<geo::P2> out;
    for (size_t i = 0; i < n; ++i) {
        const geo::P2 &a = g_sk_pts[(i + n - 1) % n], &b = g_sk_pts[i], &c = g_sk_pts[(i + 1) % n];
        double e1x = b.x - a.x, e1y = b.y - a.y, l1 = std::hypot(e1x, e1y), e2x = c.x - b.x, e2y = c.y - b.y, l2 = std::hypot(e2x, e2y);
        if (l1 < 1e-12 || l2 < 1e-12) { out.push_back(b); continue; }
        double n1x = sg * e1y / l1, n1y = -sg * e1x / l1, n2x = sg * e2y / l2, n2y = -sg * e2x / l2, mx = n1x + n2x, my = n1y + n2y, ml = std::hypot(mx, my);
        if (ml < 1e-9) { out.push_back(geo::P2(b.x + d * n1x, b.y + d * n1y)); continue; }
        mx /= ml; my /= ml;
        double cosh = mx * n1x + my * n1y, k = d / std::max(0.25, cosh);
        out.push_back(geo::P2(b.x + k * mx, b.y + k * my));
    }
    g_sk_pts = out;
}
static void sk_fillet(double r) {  // every corner rounded with radius r (as far as the edges allow)
    size_t n = g_sk_pts.size(); bool closed = !sk_revolving(); if (n < 3) return;
    std::vector<geo::P2> out;
    for (size_t i = 0; i < n; ++i) {
        const geo::P2& b = g_sk_pts[i];
        if (!closed && (i == 0 || i == n - 1)) { out.push_back(b); continue; }
        const geo::P2 &a = g_sk_pts[(i + n - 1) % n], &c = g_sk_pts[(i + 1) % n];
        double u1x = a.x - b.x, u1y = a.y - b.y, l1 = std::hypot(u1x, u1y), u2x = c.x - b.x, u2y = c.y - b.y, l2 = std::hypot(u2x, u2y);
        if (l1 < 1e-12 || l2 < 1e-12) { out.push_back(b); continue; }
        u1x /= l1; u1y /= l1; u2x /= l2; u2y /= l2;
        double cs = geo::clampd(u1x * u2x + u1y * u2y, -1, 1), th = std::acos(cs);
        if (th > geo::PI - 1e-3 || th < 1e-3) { out.push_back(b); continue; }
        double t = r / std::tan(0.5 * th); t = std::min(t, 0.45 * std::min(l1, l2));
        double re = t * std::tan(0.5 * th), bx = u1x + u2x, by = u1y + u2y, bl = std::hypot(bx, by); bx /= bl; by /= bl;
        double h = re / std::sin(0.5 * th), cx = b.x + bx * h, cy = b.y + by * h;
        double p1x = b.x + u1x * t, p1y = b.y + u1y * t, p2x = b.x + u2x * t, p2y = b.y + u2y * t;
        double a1 = std::atan2(p1y - cy, p1x - cx), a2 = std::atan2(p2y - cy, p2x - cx), da = std::remainder(a2 - a1, 2 * geo::PI);
        for (int k = 0; k <= 8; ++k) { double aa = a1 + da * k / 8; out.push_back(geo::P2(cx + re * std::cos(aa), cy + re * std::sin(aa))); }
    }
    g_sk_pts = out;
}
static void sk_chamfer(double d) {  // every corner cut back by d along both edges
    size_t n = g_sk_pts.size(); bool closed = !sk_revolving() && g_sk_shape != SH_SHEET; if (n < 3) return;
    std::vector<geo::P2> out;
    for (size_t i = 0; i < n; ++i) {
        const geo::P2& b = g_sk_pts[i];
        if (!closed && (i == 0 || i == n - 1)) { out.push_back(b); continue; }
        const geo::P2 &a = g_sk_pts[(i + n - 1) % n], &c = g_sk_pts[(i + 1) % n];
        double l1 = std::hypot(a.x - b.x, a.y - b.y), l2 = std::hypot(c.x - b.x, c.y - b.y);
        if (l1 < 1e-12 || l2 < 1e-12) { out.push_back(b); continue; }
        double t1 = std::min(d, 0.45 * l1), t2 = std::min(d, 0.45 * l2);
        out.push_back(geo::P2(b.x + (a.x - b.x) / l1 * t1, b.y + (a.y - b.y) / l1 * t1));
        out.push_back(geo::P2(b.x + (c.x - b.x) / l2 * t2, b.y + (c.y - b.y) / l2 * t2));
    }
    g_sk_pts = out;
}
static void sk_dxf_import();  // in draft.inc: through the 2D drawing's DXF reader and writer
static void sk_dxf_export();
static void sk_to_draft();
static double flat_length(const js::Value& f);
static void sk_edit_menu() {
    HMENU m = CreatePopupMenu();
    AppendMenuA(m, MF_STRING, ID_SKE_MIRROR_V, "Mirror about the vertical axis (a symmetric shape)");
    AppendMenuA(m, MF_STRING, ID_SKE_MIRROR_H, "Mirror about the horizontal axis");
    AppendMenuA(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(m, MF_STRING, ID_SKE_OFFSET, "Offset...");
    AppendMenuA(m, MF_STRING, ID_SKE_FILLET, "Fillet the corners...");
    AppendMenuA(m, MF_STRING, ID_SKE_CHAMFER, "Chamfer the corners...");
    AppendMenuA(m, MF_STRING, ID_SKE_REVERSE, "Reverse the point order");
    AppendMenuA(m, MF_STRING, ID_SKE_ROTATE, "Rotate...");
    AppendMenuA(m, MF_STRING, ID_SKE_SCALE, "Scale...");
    AppendMenuA(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(m, MF_STRING, ID_SKE_SIZE, "Exact size...");
    AppendMenuA(m, MF_STRING, ID_SKE_POINTS, "Type points...");
    AppendMenuA(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(m, MF_STRING | (g_sk_pat ? MF_CHECKED : 0), ID_SKE_PATTERN, "Pattern on finish (copies)...");
    AppendMenuA(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(m, MF_STRING, ID_SKE_DXFIN, "Import an outline from DXF (AutoCAD)...");
    AppendMenuA(m, MF_STRING, ID_SKE_DXFOUT, "Export the sketch as DXF...");
    AppendMenuA(m, MF_STRING, ID_SKE_DRAFT, "Send the sketch to the 2D drawing");
    RECT r; GetWindowRect(g_sk_btn_edit, &r);
    TrackPopupMenu(m, TPM_LEFTALIGN | TPM_BOTTOMALIGN, r.left, r.top, 0, g_sk, nullptr);
    DestroyMenu(m);
}
static void sk_edit(int id) {
    std::string v;
    geo::P2 c = sk_centroid();
    switch (id) {
        case ID_SKE_MIRROR_V: case ID_SKE_MIRROR_H: {
            bool vert = id == ID_SKE_MIRROR_V;
            if (g_sk_shape == SH_POLY && g_sk_pts.size() >= 2) {  // a half profile drawn on one side: add its mirror image, reversed
                std::vector<geo::P2> m2;
                for (size_t i = g_sk_pts.size(); i-- > 0;) { geo::P2 q = g_sk_pts[i]; if (vert) q.x = -q.x; else q.y = -q.y; m2.push_back(q); }
                if (!m2.empty() && std::hypot(m2.front().x - g_sk_pts.back().x, m2.front().y - g_sk_pts.back().y) < 1e-9) m2.erase(m2.begin());
                g_sk_pts.insert(g_sk_pts.end(), m2.begin(), m2.end());
            } else for (auto& q : g_sk_pts) { if (vert) q.x = -q.x; else q.y = -q.y; }
            break;
        }
        case ID_SKE_OFFSET: v = "0.05"; if (ask_text(g_sk, "Offset", "Distance outwards (m; negative inwards):", v)) sk_offset(atof(v.c_str())); break;
        case ID_SKE_FILLET: v = "0.1"; if (ask_text(g_sk, "Fillet", "Corner radius (m):", v)) { sk_fillet(std::max(0.0, atof(v.c_str()))); g_sk_smooth = false; } break;
        case ID_SKE_CHAMFER: v = "0.05"; if (ask_text(g_sk, "Chamfer", "Distance back from each corner (m):", v)) { sk_chamfer(std::max(0.0, atof(v.c_str()))); g_sk_smooth = false; } break;
        case ID_SKE_REVERSE: std::reverse(g_sk_pts.begin(), g_sk_pts.end()); break;
        case ID_SKE_DXFIN: sk_dxf_import(); break;
        case ID_SKE_DXFOUT: sk_dxf_export(); break;
        case ID_SKE_DRAFT: sk_to_draft(); break;
        case ID_SKE_ROTATE: v = "90"; if (ask_text(g_sk, "Rotate", "Angle (degrees, anticlockwise, about the shape's centre):", v)) { double a = atof(v.c_str()) * geo::PI / 180, ca = std::cos(a), sa = std::sin(a); for (auto& q : g_sk_pts) { double x = q.x - c.x, y = q.y - c.y; q = geo::P2(c.x + x * ca - y * sa, c.y + x * sa + y * ca); } } break;
        case ID_SKE_SCALE: v = "2"; if (ask_text(g_sk, "Scale", "Factor (about the shape's centre):", v)) { double f = atof(v.c_str()); if (f > 0) for (auto& q : g_sk_pts) q = geo::P2(c.x + (q.x - c.x) * f, c.y + (q.y - c.y) * f); } break;
        case ID_SKE_SIZE: {  // the shape again from typed sizes, about its centre (or the origin)
            static const char* help[NSHAPE] = {"", "Width and height (m), e.g. 1 0.5:", "Radius (m):", "Length and width (m):", "Width and height (m):", "Radius to the corners (m):", "Width and height (m):", "Outer radius (m):", "Tip radius (m):", "Radius (m) and sweep (degrees):", "Side (m):",
                                              "Chord (m):", "Width and height (m):", "Width and height (m):", "Width and height (m):", "Length from the hole to the end of the slot (m):", "Arm length from the centre (m):", "Width and height (m):", "Width and height (m):", "Width and height (m):", ""};
            if (sk_clicked()) { MessageBoxA(g_sk, "Exact size works on a rectangle, circle, slot, ellipse, polygon, rounded rectangle, star, gear, arc slot or triangle.", "Sketch", MB_ICONINFORMATION); break; }
            bool box = g_sk_shape == SH_RECT || g_sk_shape == SH_ELLIPSE || g_sk_shape == SH_RRECT || g_sk_shape == SH_LSEC || g_sk_shape == SH_TSEC || g_sk_shape == SH_ISEC || g_sk_shape == SH_CHANNEL || g_sk_shape == SH_ZSEC || g_sk_shape == SH_HAT;
            v = box ? "1 0.5" : g_sk_shape == SH_SLOT ? "1 0.36" : g_sk_shape == SH_ARCSLOT ? "0.5 120" : "0.5";
            if (!ask_text(g_sk, "Exact size", help[g_sk_shape], v)) break;
            double a = 0, b = 0; sscanf(v.c_str(), "%lf %lf", &a, &b);
            if (a <= 0) break;
            if (g_sk_pts.empty()) c = geo::P2(0, 0);
            if (box) { if (b <= 0) b = a; sk_shape_points(c.x - a / 2, c.y - b / 2, c.x + a / 2, c.y + b / 2); }
            else if (g_sk_shape == SH_SLOT) sk_shape_points(c.x - a / 2, c.y, c.x + a / 2, c.y);
            else if (g_sk_shape == SH_ARCSLOT) { double sw = (b > 0 ? b : 120) * geo::PI / 180; sk_shape_points(c.x, c.y, c.x + a * std::cos(sw), c.y + a * std::sin(sw)); }
            else if (g_sk_shape == SH_TRIANGLE) sk_shape_points(c.x - a / 2, c.y - 0.2887 * a, c.x + a / 2, c.y - 0.2887 * a);
            else if (g_sk_shape == SH_NACA) sk_shape_points(c.x - a / 2, c.y, c.x + a / 2, c.y);
            else if (g_sk_shape == SH_KEYHOLE) sk_shape_points(c.x, c.y, c.x, c.y - a);
            else sk_shape_points(c.x, c.y, c.x + a, c.y);
            break;
        }
        case ID_SKE_PATTERN: {
            v = g_sk_pat == 2 ? "circular " + tostr(g_sk_pat_n) : g_sk_pat == 1 ? "linear " + tostr(g_sk_pat_n) + " " + fmtd(g_sk_pat_dx) + " " + fmtd(g_sk_pat_dy) : "linear 4 0.5 0";
            if (!ask_text(g_sk, "Pattern on finish", "Copies when the pad or pocket is made: linear N DX DY (m, in the sketch), circular N (about the sketch's origin), or none:", v)) break;
            char kind[32] = {0}; int n = 1; double dx = 0, dy = 0;
            sscanf(v.c_str(), "%31s %d %lf %lf", kind, &n, &dx, &dy);
            std::string k = kind; g_sk_pat_n = std::max(1, std::min(100, n)); g_sk_pat_dx = dx; g_sk_pat_dy = dy;
            g_sk_pat = k == "linear" ? 1 : k == "circular" ? 2 : 0;
            if (g_sk_pat_n <= 1) g_sk_pat = 0;
            break;
        }
        case ID_SKE_POINTS: {
            v = "0 0, 1 0, 1 0.5";
            if (!ask_text(g_sk, "Type points", "Points as x y pairs separated by commas (m); they are added to the polyline:", v)) break;
            if (g_sk_shape != SH_POLY) { g_sk_shape = SH_POLY; g_sk_pts.clear(); }
            std::string t = v; for (char& ch : t) if (ch == ';') ch = ',';
            size_t pos = 0;
            while (pos < t.size()) { size_t e = t.find(',', pos); std::string one = t.substr(pos, e == std::string::npos ? std::string::npos : e - pos); double x, y; if (sscanf(one.c_str(), "%lf %lf", &x, &y) == 2) g_sk_pts.push_back(geo::P2(x, y)); if (e == std::string::npos) break; pos = e + 1; }
            break;
        }
    }
    InvalidateRect(g_sk_canvas, nullptr, FALSE);
}

static void paint_sketch(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    int W = std::max(1L, rc.right), H = std::max(1L, rc.bottom);
    gp::Bitmap buf(W, H, PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    gp::LinearGradientBrush bg(gp::Rect(0, 0, W, H), gcol(5, 14, 30), gcol(12, 34, 66), gp::LinearGradientModeVertical);
    g.FillRectangle(&bg, 0, 0, W, H);
    double x0, y0, x1, y1;
    sk_to_world(0, H, x0, y0); sk_to_world(W, 0, x1, y1);
    double minor = g_sk_scale > 60 ? 0.1 : 0.25;
    for (int pass = 0; pass < 2; ++pass) {
        double st = pass ? 1.0 : minor;
        gp::Pen pen(gcol(80, 170, 255, pass ? 60 : 22), pass ? 1.0f : 0.7f);
        for (double x = std::floor(x0 / st) * st; x <= x1; x += st) g.DrawLine(&pen, sk_to_screen(x, y0), sk_to_screen(x, y1));
        for (double y = std::floor(y0 / st) * st; y <= y1; y += st) g.DrawLine(&pen, sk_to_screen(x0, y), sk_to_screen(x1, y));
    }
    gp::Pen axis(gcol(0, 200, 255, 160), 1.6f);
    g.DrawLine(&axis, sk_to_screen(x0, 0), sk_to_screen(x1, 0));
    g.DrawLine(&axis, sk_to_screen(0, y0), sk_to_screen(0, y1));
    gtext(g, AXIS_U[g_sk_plane], gp::RectF((float)W - 28, sk_to_screen(0, 0).Y - 22, 20, 18), gcol(0, 210, 255), 14.0f, gp::FontStyleBold);
    gtext(g, AXIS_V[g_sk_plane], gp::RectF(sk_to_screen(0, 0).X + 6, 36, 20, 18), gcol(0, 210, 255), 14.0f, gp::FontStyleBold);
    if (sk_revolving()) {
        gp::PointF a = sk_to_screen(x0, 0);
        gp::SolidBrush below(gcol(0, 0, 0, 70)); g.FillRectangle(&below, 0.0f, a.Y, (float)W, (float)H - a.Y);
        gtext(g, std::string("axis of revolution (") + AXIS_U[g_sk_plane] + ")", gp::RectF(10, a.Y + 4, 260, 18), gcol(0, 200, 255), 11.5f);
    }
    for (int k = 0; k <= 60; ++k) {
        double x = std::floor(x0) + k;
        if (x > x1) break;
        gp::PointF p = sk_to_screen(x, 0);
        gtext(g, tostr((long)x), gp::RectF(p.X + 2, p.Y + 2, 30, 14), gcol(120, 190, 255, 160), 10.0f);
    }
    std::vector<geo::P2> cv = sk_curve();
    std::vector<gp::PointF> sp;
    for (auto& q : cv) sp.push_back(sk_to_screen(q.x, q.y));
    gp::Color main = g_sk_op == OP_POCKET || g_sk_op == OP_GROOVE ? gcol(244, 114, 182) : gcol(0, 210, 255);
    if (sk_revolving() && sp.size() >= 2) {
        std::vector<gp::PointF> mir;
        for (auto& q : cv) mir.push_back(sk_to_screen(q.x, -q.y));
        gp::GraphicsPath body; body.AddLines(sp.data(), (INT)sp.size());
        std::vector<gp::PointF> back(mir.rbegin(), mir.rend()); body.AddLines(back.data(), (INT)back.size()); body.CloseFigure();
        gp::SolidBrush fillb(gcol(main.GetR(), main.GetG(), main.GetB(), 40)); g.FillPath(&fillb, &body);
        neon(g, mir, gshade(main, -0.35f), 1.4f);
    } else if (sp.size() >= 3) {
        gp::GraphicsPath shape; shape.AddLines(sp.data(), (INT)sp.size()); shape.CloseFigure();
        gp::SolidBrush fillb(gcol(main.GetR(), main.GetG(), main.GetB(), 45)); g.FillPath(&fillb, &shape);
        if (g_sk_op == OP_LOFT)
            for (double s : {0.35, 0.55}) {
                std::vector<gp::PointF> sc; for (auto& q : cv) sc.push_back(sk_to_screen(q.x * s, q.y * s));
                gp::Pen pn(gcol(167, 139, 250, 150), 1.2f); pn.SetDashStyle(gp::DashStyleDash); g.DrawLines(&pn, sc.data(), (INT)sc.size());
            }
    }
    neon(g, sp, main, 2.4f);
    for (size_t i = 0; i < g_sk_pts.size(); ++i) {
        if (!sk_clicked() && i % 3) continue;  // fewer handles on generated shapes
        gp::PointF p = sk_to_screen(g_sk_pts[i].x, g_sk_pts[i].y);
        bool hv = (int)i == g_sk_hover || (int)i == g_sk_drag;
        float r = hv ? 7.0f : 4.5f;
        gp::SolidBrush glow(gcol(main.GetR(), main.GetG(), main.GetB(), 60)); g.FillEllipse(&glow, p.X - r - 4, p.Y - r - 4, 2 * r + 8, 2 * r + 8);
        gp::SolidBrush dot(hv ? gcol(255, 255, 255) : gshade(main, 0.5f)); g.FillEllipse(&dot, p.X - r, p.Y - r, 2 * r, 2 * r);
    }
    // live dimensions of the last segment (length and angle), the sketch's constraints at a glance
    if (sk_clicked() && g_sk_pts.size() >= 2) {
        const geo::P2& a = g_sk_pts[g_sk_pts.size() - 2]; const geo::P2& b = g_sk_pts.back();
        double L = std::hypot(b.x - a.x, b.y - a.y), ang = std::atan2(b.y - a.y, b.x - a.x) * 180 / geo::PI;
        bool hz = std::fabs(b.y - a.y) < 1e-9, vt = std::fabs(b.x - a.x) < 1e-9;
        char t[96]; snprintf(t, sizeof t, "%.3f m  %.1f deg%s", L, ang, hz ? "  [horizontal]" : vt ? "  [vertical]" : "");
        gp::PointF m = sk_to_screen(0.5 * (a.x + b.x), 0.5 * (a.y + b.y));
        gtext(g, t, gp::RectF(m.X + 8, m.Y - 22, 260, 18), gcol(251, 191, 36), 11.5f, gp::FontStyleBold);
    }
    static const char* opn[5] = {"Pad (extrude, add)", "Pocket (extrude, cut)", "Shaft (revolve, add)", "Groove (revolve, cut)", "Loft"};
    static const char* shn[NSHAPE] = {"click points (spline or lines)", "drag a rectangle", "drag from the centre", "drag the slot's length", "drag the ellipse's box", "drag from the centre (Sides)", "drag a rounded rectangle",
                                      "drag from the centre (Sides = points)", "drag from the centre to the tips (Sides = teeth)", "drag from the centre: radius and sweep", "drag the base",
                                      "drag the chord, leading edge first (Sides = NACA code, e.g. 2412)", "drag the L section's box", "drag the T section's box", "drag the I section's box", "drag from the hole to the slot's end", "drag from the centre to an arm's end",
                                      "drag the channel's box", "drag the Z section's box", "drag the hat section's box", "click the sheet's bend line (mid-surface), Enter: thickness, bend radius"};
    gp::SolidBrush hud(gcol(0, 0, 0, 110)); g.FillRectangle(&hud, 0, 0, W, 30);
    gtext(g, std::string(PLANE_NAMES[g_sk_plane]) + "  |  " + opn[g_sk_op] + "  |  " + shn[g_sk_shape], gp::RectF(12, 4, (float)W - 24, 22), gcol(255, 255, 255), 13.5f, gp::FontStyleBold);
    char b[220];
    snprintf(b, sizeof b, "%s %.2f m   %s %.2f m   |   %d points   |   right click delete, wheel zoom, middle drag pan, Enter finish, Esc close",
             AXIS_U[g_sk_plane], g_sk_wx, AXIS_V[g_sk_plane], g_sk_wy, (int)g_sk_pts.size());
    g.FillRectangle(&hud, 0, H - 24, W, 24);
    gtext(g, b, gp::RectF(10, (float)H - 23, (float)W - 20, 22), gcol(150, 220, 255), 11.5f);
    gp::Graphics out(dc);
    out.DrawImage(&buf, 0, 0);
}

static int sk_near(int sx, int sy, double px = 9) {
    int best = -1; double bd = px * px;
    for (size_t i = 0; i < g_sk_pts.size(); ++i) {
        gp::PointF p = sk_to_screen(g_sk_pts[i].x, g_sk_pts[i].y);
        double d = (p.X - sx) * (p.X - sx) + (p.Y - sy) * (p.Y - sy);
        if (d < bd) { bd = d; best = (int)i; }
    }
    return best;
}

static void sk_snap(double& x, double& y) {
    if (g_sk_snap) { double s = g_sk_scale > 60 ? 0.05 : 0.1; x = std::round(x / s) * s; y = std::round(y / s) * s; }
    if (sk_revolving()) y = std::max(0.0, y);
}

static js::Value v3j(double a, double b, double c) { js::Value p = js::Value::array(); p.push(js::Value(a)); p.push(js::Value(b)); p.push(js::Value(c)); return p; }

static void sk_finish_sheet() {
    if (g_sk_pts.size() < 2) { MessageBoxA(g_sk, "Click at least two points of the sheet's bend line (its mid-surface).", "Sheet metal", MB_ICONINFORMATION); return; }
    double width = atof(read_edit(g_sk_size).c_str()), off = atof(read_edit(g_sk_off).c_str());
    std::string v = "0.003 0.006 " + fmtd(width > 0 ? width : 0.3);
    if (!ask_text(g_sk, "Sheet metal", "Thickness, inner bend radius and width (m), e.g. 0.003 0.006 0.3:", v)) return;
    double t = 0.003, r = 0.006, w = 0.3; sscanf(v.c_str(), "%lf %lf %lf", &t, &r, &w);
    t = std::max(1e-4, t); r = std::max(0.0, r); w = std::max(1e-3, w);
    js::Value f = js::Value::object(), pr = js::Value::array();
    char id[48]; snprintf(id, sizeof id, "sheet_%d", ++g_sk_count);
    for (const geo::P2& q : g_sk_pts) { js::Value a = js::Value::array(); a.push(js::Value(q.x)); a.push(js::Value(q.y)); pr.push(a); }
    f["id"] = id; f["type"] = "bentplate"; f["op"] = "add"; f["mirror"] = false; f["profile"] = pr; f["thickness"] = t; f["bend_radius"] = r; f["width"] = w;
    if (g_sk_plane == 1) f["translate"] = v3j(0, off, 0);                                   // the profile is drawn in x-z
    else if (g_sk_plane == 0) { f["rotate"] = v3j(-90, 0, 0); f["translate"] = v3j(0, 0, off); }  // x-y: the width along z
    else { f["rotate"] = v3j(0, 0, 90); f["translate"] = v3j(off, 0, 0); }                    // y-z: the width along x
    insert(f);
    status(std::string("sheet-metal part '") + id + "': " + fmtd(t * 1000) + " mm thick, bends R" + fmtd(r * 1000) + " mm, flat blank " + fmtd(std::round(flat_length(f) * 10000) / 10) + " mm (Sheet > Flat pattern draws it)");
    ShowWindow(g_sk, SW_HIDE);
}
static void sk_finish() {
    if (g_sk_shape == SH_SHEET) { sk_finish_sheet(); return; }
    size_t need = sk_revolving() ? 2 : 3;
    if (g_sk_pts.size() < need) { MessageBoxA(g_sk, sk_revolving() ? "Draw at least two profile points." : "Draw a closed shape with at least three points.", "Sketch", MB_ICONINFORMATION); return; }
    double size = atof(read_edit(g_sk_size).c_str()), off = atof(read_edit(g_sk_off).c_str());
    js::Value f = js::Value::object();
    char id[48];
    static const char* opid[5] = {"pad", "pocket", "shaft", "groove", "loft"};
    snprintf(id, sizeof id, "%s_%d", opid[g_sk_op], ++g_sk_count);
    f["id"] = id; f["smooth"] = g_sk_smooth && g_sk_shape == SH_POLY; f["mirror"] = false;
    f["op"] = (g_sk_op == OP_POCKET || g_sk_op == OP_GROOVE) ? "cut" : "add";
    auto pt = [](double a, double b) { js::Value p = js::Value::array(); p.push(js::Value(a)); p.push(js::Value(b)); return p; };
    if (sk_revolving()) {
        std::vector<geo::P2> v = g_sk_pts;
        if (v.front().y > 1e-6) v.insert(v.begin(), geo::P2(v.front().x, 0.0));
        if (v.back().y > 1e-6) v.push_back(geo::P2(v.back().x, 0.0));
        js::Value pr = js::Value::array(); for (auto& q : v) pr.push(pt(q.x, q.y));
        f["type"] = "revolve"; f["profile"] = pr;
        if (g_sk_plane == 0) f["translate"] = v3j(0, 0, off);
        else if (g_sk_plane == 1) f["translate"] = v3j(0, off, 0);
        else { f["rotate"] = v3j(0, 0, 90); f["translate"] = v3j(off, 0, 0); }
    } else if (g_sk_op == OP_LOFT) {
        double L = size > 0 ? size : 10.0, xs[3] = {0, 0.45 * L, L}, sc[3] = {0.35, 1.0, 0.55};
        js::Value ss = js::Value::array();
        for (int k = 0; k < 3; ++k) {
            js::Value sec = js::Value::object(); sec["x"] = xs[k];
            js::Value pts = js::Value::array();
            for (auto& q : g_sk_pts) {
                double u = q.x * sc[k], v = q.y * sc[k];
                if (g_sk_plane == 2) pts.push(pt(u, v));          // local (y, z) = (u, v)
                else if (g_sk_plane == 0) pts.push(pt(v, -u));    // loft along z: local y = world y, local z = -world x
                else pts.push(pt(-u, v));                         // loft along y: local y = -world x, local z = world z
            }
            sec["points"] = pts; ss.push(sec);
        }
        f["type"] = "loft"; f["sections"] = ss;
        if (g_sk_plane == 2) f["translate"] = v3j(off, 0, 0);
        else if (g_sk_plane == 0) { f["rotate"] = v3j(0, -90, 0); f["translate"] = v3j(0, 0, off); }
        else { f["rotate"] = v3j(0, 0, 90); f["translate"] = v3j(0, off, 0); }
    } else {
        js::Value ol = js::Value::array(); for (auto& q : g_sk_pts) ol.push(pt(q.x, q.y));
        f["type"] = "extrude"; f["outline"] = ol; f["height"] = size > 0 ? size : 0.1;
        if (g_sk_plane == 0) f["translate"] = v3j(0, 0, off);
        else if (g_sk_plane == 1) { f["rotate"] = v3j(90, 0, 0); f["translate"] = v3j(0, off, 0); }
        else { f["rotate"] = v3j(90, 0, 90); f["translate"] = v3j(off, 0, 0); }
    }
    insert(f);
    int copies = 0;
    if (g_sk_pat && f.str("type", "") == "extrude") {  // the pattern: more features, moved along the sketch or turned about its origin
        for (int k = 1; k < g_sk_pat_n; ++k) {
            js::Value c = f.clone();
            c["id"] = std::string(id) + "_" + tostr(k + 1);
            if (g_sk_pat == 1) {
                double du = g_sk_pat_dx * k, dv = g_sk_pat_dy * k, t[3] = {0, 0, 0};
                if (const js::Value* tr = f.find("translate")) for (int a = 0; a < 3 && a < (int)tr->size(); ++a) t[a] = tr->at((size_t)a).n;
                if (g_sk_plane == 0) { t[0] += du; t[1] += dv; } else if (g_sk_plane == 1) { t[0] += du; t[2] += dv; } else { t[1] += du; t[2] += dv; }
                c["translate"] = v3j(t[0], t[1], t[2]);
            } else {
                double a = 2 * geo::PI * k / g_sk_pat_n, ca = std::cos(a), sa = std::sin(a);
                js::Value ol = js::Value::array(); for (auto& q : g_sk_pts) ol.push(pt(q.x * ca - q.y * sa, q.x * sa + q.y * ca));
                c["outline"] = ol;
            }
            insert(c); ++copies;
        }
    }
    status(std::string("sketch on the ") + PLANE_NAMES[g_sk_plane] + " became '" + id + "'" + (copies ? " and " + tostr(copies) + " patterned copies" : std::string()));
    ShowWindow(g_sk, SW_HIDE);
}

static LRESULT CALLBACK sk_canvas_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_PAINT: { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_sketch(h, dc); EndPaint(h, &ps); return 0; }
        case WM_ERASEBKGND: return 1;
        case WM_LBUTTONDOWN: {
            SetFocus(h); SetCapture(h);
            int x = (short)LOWORD(l), y = (short)HIWORD(l);
            double wx, wy; sk_to_world(x, y, wx, wy); sk_snap(wx, wy);
            if (!sk_clicked()) { g_sk_dragshape = true; g_sk_ax = wx; g_sk_ay = wy; g_sk_pts.clear(); }
            else {
                int k = sk_near(x, y);
                if (k >= 0) g_sk_drag = k;
                else {
                    if (GetKeyState(VK_SHIFT) < 0 && !g_sk_pts.empty()) {  // Shift: horizontal or vertical from the last point
                        const geo::P2& a = g_sk_pts.back();
                        if (std::fabs(wx - a.x) > std::fabs(wy - a.y)) wy = a.y; else wx = a.x;
                    }
                    g_sk_pts.push_back(geo::P2(wx, wy)); g_sk_drag = (int)g_sk_pts.size() - 1;
                }
            }
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        }
        case WM_LBUTTONUP: case WM_MBUTTONUP: ReleaseCapture(); g_sk_drag = -1; g_sk_pan = false; g_sk_dragshape = false; InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_MBUTTONDOWN: SetCapture(h); g_sk_pan = true; g_sk_mx = (short)LOWORD(l); g_sk_my = (short)HIWORD(l); return 0;
        case WM_RBUTTONDOWN: { int k = sk_near((short)LOWORD(l), (short)HIWORD(l), 12); if (k >= 0) { g_sk_pts.erase(g_sk_pts.begin() + k); InvalidateRect(h, nullptr, FALSE); } return 0; }
        case WM_MOUSEMOVE: {
            int x = (short)LOWORD(l), y = (short)HIWORD(l);
            sk_to_world(x, y, g_sk_wx, g_sk_wy);
            if (g_sk_pan) { g_sk_cx -= (x - g_sk_mx) / g_sk_scale; g_sk_cy += (y - g_sk_my) / g_sk_scale; g_sk_mx = x; g_sk_my = y; }
            else if (g_sk_dragshape) { double wx = g_sk_wx, wy = g_sk_wy; sk_snap(wx, wy); sk_shape_points(g_sk_ax, g_sk_ay, wx, wy); }
            else if (g_sk_drag >= 0) { double wx = g_sk_wx, wy = g_sk_wy; sk_snap(wx, wy); g_sk_pts[(size_t)g_sk_drag] = geo::P2(wx, wy); }
            else g_sk_hover = sk_near(x, y);
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        }
        case WM_MOUSEWHEEL: {
            POINT p = {(short)LOWORD(l), (short)HIWORD(l)}; ScreenToClient(h, &p);
            double wx, wy; sk_to_world(p.x, p.y, wx, wy);
            g_sk_scale = std::max(8.0, std::min(800.0, g_sk_scale * std::pow(1.15, (short)HIWORD(w) / 120.0)));
            double nx, ny; sk_to_world(p.x, p.y, nx, ny);
            g_sk_cx += wx - nx; g_sk_cy += wy - ny;
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        }
        case WM_KEYDOWN:
            if (w == VK_RETURN) sk_finish();
            else if (w == VK_BACK && !g_sk_pts.empty()) { g_sk_pts.pop_back(); InvalidateRect(h, nullptr, FALSE); }
            else if (w == VK_ESCAPE) ShowWindow(g_sk, SW_HIDE);
            return 0;
        case WM_GETDLGCODE: return DLGC_WANTALLKEYS;
    }
    return DefWindowProcA(h, m, w, l);
}

static void sk_layout() {
    RECT rc; GetClientRect(g_sk, &rc);
    int W = rc.right, H = rc.bottom, bh = 46;
    MoveWindow(g_sk_canvas, 0, 0, W, H - bh, TRUE);
    int x = 8, y = H - bh + 9;
    HWND bt[] = {GetDlgItem(g_sk, ID_SK_FINISH), GetDlgItem(g_sk, ID_SK_UNDO), GetDlgItem(g_sk, ID_SK_CLEAR), g_sk_btn_snap, g_sk_btn_smooth};
    int bw[] = {116, 70, 64, 86, 86};
    for (int i = 0; i < 5; ++i) { MoveWindow(bt[i], x, y, bw[i], 30, TRUE); x += bw[i] + 6; }
    x += 6;
    MoveWindow(g_sk_plane_cb, x, y + 3, 150, 200, TRUE); x += 156;
    MoveWindow(g_sk_shape_cb, x, y + 3, 134, 200, TRUE); x += 140;
    MoveWindow(g_sk_op_cb, x, y + 3, 140, 200, TRUE); x += 148;
    MoveWindow(g_sk_off_lbl, x, y + 7, 56, 20, TRUE); MoveWindow(g_sk_off, x + 56, y + 3, 50, 24, TRUE); x += 112;
    MoveWindow(g_sk_size_lbl, x, y + 7, 70, 20, TRUE); MoveWindow(g_sk_size, x + 70, y + 3, 50, 24, TRUE); x += 128;
    MoveWindow(g_sk_sides_lbl, x, y + 7, 36, 20, TRUE); MoveWindow(g_sk_sides, x + 38, y + 3, 40, 24, TRUE); x += 86;
    MoveWindow(g_sk_btn_edit, x, y, 76, 30, TRUE);
}

static void sk_labels() {
    SetWindowTextA(g_sk_btn_snap, g_sk_snap ? "Snap: on" : "Snap: off");
    SetWindowTextA(g_sk_btn_smooth, g_sk_smooth ? "Spline" : "Lines");
    SetWindowTextA(g_sk_size_lbl, g_sk_op == OP_LOFT ? "Length, m" : sk_revolving() ? "" : "Depth, m");
    ShowWindow(g_sk_size, sk_revolving() ? SW_HIDE : SW_SHOW);
    SendMessageA(g_sk_plane_cb, CB_SETCURSEL, g_sk_plane, 0);
    SendMessageA(g_sk_shape_cb, CB_SETCURSEL, g_sk_shape, 0);
    SendMessageA(g_sk_op_cb, CB_SETCURSEL, g_sk_op, 0);
}

static LRESULT CALLBACK sk_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            HINSTANCE hi = GetModuleHandleA(nullptr);
            g_sk_canvas = CreateWindowExA(0, "AeroSynapseSketchCanvas", "", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            const char* names[] = {"Finish  (Enter)", "Undo", "Clear"};
            int ids[] = {ID_SK_FINISH, ID_SK_UNDO, ID_SK_CLEAR};
            for (int i = 0; i < 3; ++i) CreateWindowExA(0, "BUTTON", names[i], WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)(INT_PTR)ids[i], hi, nullptr);
            g_sk_btn_snap = CreateWindowExA(0, "BUTTON", "", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)ID_SK_SNAP, hi, nullptr);
            g_sk_btn_smooth = CreateWindowExA(0, "BUTTON", "", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)ID_SK_SMOOTH, hi, nullptr);
            g_sk_plane_cb = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST, 0, 0, 0, 0, h, (HMENU)ID_SK_PLANE, hi, nullptr);
            for (const char* s : PLANE_NAMES) SendMessageA(g_sk_plane_cb, CB_ADDSTRING, 0, (LPARAM)s);
            g_sk_shape_cb = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST, 0, 0, 0, 0, h, (HMENU)ID_SK_SHAPE, hi, nullptr);
            for (const char* s : {"Polyline / spline", "Rectangle", "Circle", "Slot", "Ellipse", "Polygon", "Rounded rectangle", "Star", "Gear (involute)", "Arc slot", "Triangle", "NACA airfoil", "L section", "T section", "I section", "Keyhole", "Cross", "Channel (C)", "Z section", "Hat section", "Sheet metal (bend line)"}) SendMessageA(g_sk_shape_cb, CB_ADDSTRING, 0, (LPARAM)s);
            g_sk_op_cb = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST, 0, 0, 0, 0, h, (HMENU)ID_SK_OP, hi, nullptr);
            for (const char* s : {"Pad (add)", "Pocket (cut)", "Shaft (revolve)", "Groove (revolve cut)", "Loft"}) SendMessageA(g_sk_op_cb, CB_ADDSTRING, 0, (LPARAM)s);
            g_sk_off_lbl = CreateWindowExA(0, "STATIC", "Offset, m", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_sk_off = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "0", WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL, 0, 0, 0, 0, h, (HMENU)ID_SK_OFF, hi, nullptr);
            g_sk_size_lbl = CreateWindowExA(0, "STATIC", "", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_sk_size = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "0.1", WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL, 0, 0, 0, 0, h, (HMENU)ID_SK_SIZE, hi, nullptr);
            g_sk_sides_lbl = CreateWindowExA(0, "STATIC", "Sides", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_sk_sides = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "6", WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL, 0, 0, 0, 0, h, (HMENU)ID_SK_SIDES, hi, nullptr);
            g_sk_btn_edit = CreateWindowExA(0, "BUTTON", "Edit...", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)ID_SK_EDIT, hi, nullptr);
            HWND all[] = {g_sk_size_lbl, g_sk_size, g_sk_plane_cb, g_sk_shape_cb, g_sk_op_cb, g_sk_off_lbl, g_sk_off, g_sk_sides_lbl, g_sk_sides};
            for (HWND x : all) SendMessageA(x, WM_SETFONT, (WPARAM)g_font, TRUE);
            return 0;
        }
        case WM_SIZE: sk_layout(); return 0;
        case WM_COMMAND:
            switch (LOWORD(w)) {
                case ID_SK_FINISH: sk_finish(); break;
                case ID_SK_UNDO: if (!g_sk_pts.empty()) g_sk_pts.pop_back(); break;
                case ID_SK_CLEAR: g_sk_pts.clear(); break;
                case ID_SK_SNAP: g_sk_snap = !g_sk_snap; sk_labels(); break;
                case ID_SK_SMOOTH: g_sk_smooth = !g_sk_smooth; sk_labels(); break;
                case ID_SK_PLANE: if (HIWORD(w) == CBN_SELCHANGE) { g_sk_plane = (int)SendMessageA(g_sk_plane_cb, CB_GETCURSEL, 0, 0); sk_labels(); } break;
                case ID_SK_SHAPE: if (HIWORD(w) == CBN_SELCHANGE) { g_sk_shape = (int)SendMessageA(g_sk_shape_cb, CB_GETCURSEL, 0, 0); g_sk_pts.clear(); sk_labels(); } break;
                case ID_SK_EDIT: sk_edit_menu(); break;
                case ID_SKE_MIRROR_V: case ID_SKE_MIRROR_H: case ID_SKE_OFFSET: case ID_SKE_FILLET: case ID_SKE_ROTATE: case ID_SKE_SCALE: case ID_SKE_SIZE: case ID_SKE_POINTS: case ID_SKE_PATTERN: sk_edit(LOWORD(w)); break;
                case ID_SK_SIDES: return 0;
                case ID_SK_OP: if (HIWORD(w) == CBN_SELCHANGE) { g_sk_op = (int)SendMessageA(g_sk_op_cb, CB_GETCURSEL, 0, 0); sk_labels(); } break;
                default: return 0;
            }
            InvalidateRect(g_sk_canvas, nullptr, FALSE);
            if (HIWORD(w) != CBN_SELCHANGE) SetFocus(g_sk_canvas);
            return 0;
        case WM_DRAWITEM: {
            DRAWITEMSTRUCT* d = (DRAWITEMSTRUCT*)l;
            draw_button(d, d->CtlID == ID_SK_FINISH ? gcol(0, 160, 215) : gcol(22, 50, 92));
            return TRUE;
        }
        case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkColor(dc, C_PANEL); SetTextColor(dc, C_INK); return (LRESULT)g_br_panel; }
        case WM_CLOSE: ShowWindow(h, SW_HIDE); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

// mode: SK_REVOLVE / SK_EXTRUDE / SK_LOFT presets, or -1 to keep; plane and op override when >= 0
static void open_sketch(int mode, int plane = -1, int op = -1, int shape = -1) {
    if (!g_sk) {
        g_sk = CreateWindowExA(WS_EX_TOOLWINDOW, "AeroSynapseSketch", "Sketcher", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1420, 720, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
    }
    if (mode == SK_REVOLVE) { g_sk_plane = 1; g_sk_op = OP_SHAFT; g_sk_shape = SH_POLY; }
    else if (mode == SK_EXTRUDE) { g_sk_plane = 0; g_sk_op = OP_PAD; g_sk_shape = SH_POLY; }
    else if (mode == SK_LOFT) { g_sk_plane = 2; g_sk_op = OP_LOFT; g_sk_shape = SH_POLY; }
    if (plane >= 0) g_sk_plane = plane;
    if (op >= 0) g_sk_op = op;
    if (shape >= 0) g_sk_shape = shape;
    g_sk_pts.clear(); g_sk_drag = g_sk_hover = -1;
    if (g_sk_plane == 2) { g_sk_cx = 0; g_sk_cy = 0; g_sk_scale = 110; } else { g_sk_cx = 8; g_sk_cy = sk_revolving() ? 1.2 : 0.5; g_sk_scale = 55; }
    SetWindowTextA(g_sk_size, g_sk_op == OP_LOFT ? "10" : "0.1");
    SetWindowTextA(g_sk_off, "0");
    SetWindowTextA(g_sk, (std::string("Sketcher: ") + PLANE_NAMES[g_sk_plane]).c_str());
    sk_labels(); sk_layout();
    ShowWindow(g_sk, SW_SHOW); SetForegroundWindow(g_sk); SetFocus(g_sk_canvas);
    InvalidateRect(g_sk_canvas, nullptr, FALSE);
}

static void register_sketch_classes(HINSTANCE hi, HICON icon) {
    WNDCLASSA a; memset(&a, 0, sizeof a);
    a.lpfnWndProc = sk_proc; a.hInstance = hi; a.hCursor = LoadCursor(nullptr, IDC_ARROW); a.hbrBackground = g_br_panel; a.lpszClassName = "AeroSynapseSketch"; a.hIcon = icon;
    RegisterClassA(&a);
    a.lpfnWndProc = sk_canvas_proc; a.hCursor = LoadCursor(nullptr, IDC_CROSS); a.hbrBackground = nullptr; a.lpszClassName = "AeroSynapseSketchCanvas";
    RegisterClassA(&a);
}
