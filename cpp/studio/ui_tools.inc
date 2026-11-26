// Scheme bar (top) and tool palette (left) of AeroSynapse Studio.
// The scheme bar shows the logo and one tab per scheme (Sketch & Body, Curves & Math, Wing & Aero, Transform, View,
// Simulate, Optimise, Workflow, Export), each with its own colour and icon. The palette lists the active scheme's
// tools as icon tiles, two per row, like a paint program's tool box. Clicking a tool inserts a feature or runs a command.

static const int SCHEME_H = 122, PALETTE_W = 128, PAL_HANDLE = 14;  // SCHEME_H: the ribbon (ribbon.inc)
static int g_pal_w = PALETTE_W, g_pal_target = PALETTE_W;  // the tool bar's current and target width (it slides)
static bool g_pal_open = false;
static void palette_hover(bool in);
static LRESULT CALLBACK ribbon_proc(HWND h, UINT m, WPARAM w, LPARAM l);
static void toggle_panel(int which);
static void show_tip(int tool, int screen_x, int screen_y);
static const char* PANEL_ICONS[3] = {"panel_tree", "panel_props", "full_view"};
static const char* PANEL_TIPS[3] = {"Feature tree (F2)", "Properties (F3)", "Full 3D view (F11)"};
static int g_hover_panel = -1;
static const UINT WM_TOOL_INSERT = WM_APP + 3;
static HWND g_schemebar, g_palette;
static int g_scheme = 0, g_hover_scheme = -1, g_hover_tool = -1, g_press_tool = -1, g_pal_scroll = 0;

static gp::Color scheme_col(int k) { return gcol(SCHEMES[k].r, SCHEMES[k].g, SCHEMES[k].b); }

static std::vector<int> scheme_tools(int k) {
    std::vector<int> v;
    for (int i = 0; i < NTOOL; ++i) if (std::string(TOOLS[i].scheme) == SCHEMES[k].key) v.push_back(i);
    return v;
}

// ---------------------------------------------------------------- scheme bar
static int tab_w(int W) { return std::max(54, std::min(132, (W - 330 - 130) / NSCHEME)); }
static RECT panel_rect(int k, int W) { RECT r = {W - 124 + k * 40, 14, W - 124 + k * 40 + 36, 50}; return r; }

static RECT tab_rect(int k, int W) {
    int w = tab_w(W), x0 = 318;
    RECT r = {x0 + k * w, 6, x0 + (k + 1) * w - 4, SCHEME_H - 8};
    return r;
}

static void paint_schemebar(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    int W = rc.right, H = rc.bottom;
    gp::Bitmap buf(std::max(1, W), std::max(1, H), PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    gp::Color acc = scheme_col(g_scheme);
    gp::LinearGradientBrush bg(gp::Rect(0, 0, W, H), gcol(6, 14, 30), gmix(gcol(14, 40, 80), acc, 0.25f), gp::LinearGradientModeHorizontal);
    g.FillRectangle(&bg, 0, 0, W, H);
    gp::LinearGradientBrush sheen(gp::Rect(0, 0, W, H / 2 + 1), gcol(255, 255, 255, 22), gcol(255, 255, 255, 0), gp::LinearGradientModeVertical);
    g.FillRectangle(&sheen, 0, 0, W, H / 2);
    // accent line with glow
    for (int k = 0; k < 4; ++k) { gp::SolidBrush gb(gcol(acc.GetR(), acc.GetG(), acc.GetB(), 40 - k * 9)); g.FillRectangle(&gb, 0, H - 3 - k * 2, W, 2); }
    gp::SolidBrush ab(acc); g.FillRectangle(&ab, 0, H - 3, W, 3);
    if (gp::Bitmap* logo = load_png(g_root + "\\docs\\assets\\aerosynapse_logo.png")) g.DrawImage(logo, 12, 8, 46, 46);
    gtext(g, "AeroSynapse Studio", gp::RectF(66, 8, 250, 26), gcol(255, 255, 255), 19.0f, gp::FontStyleBold);
    gtext(g, "design  |  simulate  |  optimise  |  export", gp::RectF(67, 33, 250, 18), gcol(120, 210, 255), 11.5f);
    for (int k = 0; k < NSCHEME; ++k) {
        RECT t = tab_rect(k, W);
        gp::RectF r((float)t.left, (float)t.top, (float)(t.right - t.left), (float)(t.bottom - t.top));
        gp::Color c = scheme_col(k);
        if (k == g_scheme) {
            soft_shadow(g, r, 9, 5, 90);
            fill_round(g, r, 9, gshade(c, 0.25f), gshade(c, -0.35f));
            gloss(g, r, 9, 80);
            gp::GraphicsPath pth; round_path(pth, r, 9);
            gp::Pen pen(gshade(c, 0.55f), 1.2f); g.DrawPath(&pen, &pth);
        } else if (k == g_hover_scheme) {
            gp::GraphicsPath pth; round_path(pth, r, 9);
            gp::SolidBrush hb(gcol(255, 255, 255, 26)); g.FillPath(&hb, &pth);
            gp::Pen pen(gcol(c.GetR(), c.GetG(), c.GetB(), 160), 1.2f); g.DrawPath(&pen, &pth);
        }
        bool wide = r.Width >= 118;
        float isz = wide ? 30.0f : 24.0f;
        if (gp::Bitmap* ic = icon_png(SCHEMES[k].icon)) {
            float ix = wide ? r.X + 6 : r.X + (r.Width - isz) / 2, iy = wide ? r.Y + (r.Height - isz) / 2 : r.Y + 3;
            g.DrawImage(ic, ix, iy, isz, isz);
        }
        if (wide) gtext(g, SCHEMES[k].title, gp::RectF(r.X + 40, r.Y, r.Width - 42, r.Height), gcol(255, 255, 255, k == g_scheme ? 255 : 210), 12.5f,
                        k == g_scheme ? gp::FontStyleBold : gp::FontStyleRegular, gp::StringAlignmentNear, gp::StringAlignmentCenter, true);
        else gtext(g, SCHEMES[k].title, gp::RectF(r.X - 2, r.Y + 28, r.Width + 4, r.Height - 28), gcol(255, 255, 255, k == g_scheme ? 255 : 215), 10.0f,
                   k == g_scheme ? gp::FontStyleBold : gp::FontStyleRegular, gp::StringAlignmentCenter, gp::StringAlignmentNear, false);
    }
    for (int k = 0; k < 3; ++k) {  // panel switches: feature tree, properties, full 3D view
        RECT t = panel_rect(k, W);
        gp::RectF r((float)t.left, (float)t.top, 36, 36);
        if (k == g_hover_panel) { gp::GraphicsPath pth; round_path(pth, r, 7); gp::SolidBrush hb(gcol(255, 255, 255, 40)); g.FillPath(&hb, &pth); }
        if (gp::Bitmap* ic = icon_png(PANEL_ICONS[k])) g.DrawImage(ic, r.X + 3, r.Y + 3, 30.0f, 30.0f);
    }
    gp::Graphics out(dc);
    out.DrawImage(&buf, 0, 0);
}

static LRESULT CALLBACK schemebar_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_PAINT: { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_schemebar(h, dc); EndPaint(h, &ps); return 0; }
        case WM_ERASEBKGND: return 1;
        case WM_MOUSEMOVE: {
            RECT rc; GetClientRect(h, &rc);
            int x = (short)LOWORD(l), y = (short)HIWORD(l), hit = -1, ph = -1;
            for (int k = 0; k < NSCHEME; ++k) { RECT t = tab_rect(k, rc.right); if (x >= t.left && x < t.right && y >= t.top && y < t.bottom) hit = k; }
            for (int k = 0; k < 3; ++k) { RECT t = panel_rect(k, rc.right); if (x >= t.left && x < t.right && y >= t.top && y < t.bottom) ph = k; }
            if (ph != g_hover_panel) { g_hover_panel = ph; InvalidateRect(h, nullptr, FALSE); if (ph >= 0) status(PANEL_TIPS[ph]); }
            if (hit != g_hover_scheme) {
                g_hover_scheme = hit; InvalidateRect(h, nullptr, FALSE);
                if (hit >= 0) status(std::string("Workbench: ") + SCHEMES[hit].title + (hit == g_scheme ? "  (click again to hide or show its tools)" : ""));
            }
            TRACKMOUSEEVENT te = {sizeof te, TME_LEAVE, h, 0}; TrackMouseEvent(&te);
            return 0;
        }
        case WM_MOUSELEAVE: g_hover_scheme = -1; g_hover_panel = -1; InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_LBUTTONDOWN:
            if (g_hover_panel >= 0) { toggle_panel(g_hover_panel + 1); return 0; }
            if (g_hover_scheme >= 0 && g_hover_scheme == g_scheme) { toggle_panel(0); return 0; }  // the active workbench: hide or show its tools
            if (g_hover_scheme >= 0 && g_hover_scheme != g_scheme) {
                if (!g_show_palette) toggle_panel(0);
                g_scheme = g_hover_scheme; g_pal_scroll = 0; g_hover_tool = -1;
                InvalidateRect(h, nullptr, FALSE); InvalidateRect(g_palette, nullptr, FALSE);
                status(std::string(SCHEMES[g_scheme].title) + ": " + tostr((long)scheme_tools(g_scheme).size()) + " tools on the left");
            }
            return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

// ---------------------------------------------------------------- tool palette
static const int CELL_W = 58, CELL_H = 82, PAL_HEAD = 46;

static RECT tool_rect(int idx) {
    int col = idx % 2, row = idx / 2;
    RECT r = {6 + col * (CELL_W + 4), PAL_HEAD + row * CELL_H - g_pal_scroll, 6 + col * (CELL_W + 4) + CELL_W, PAL_HEAD + row * CELL_H - g_pal_scroll + CELL_H - 4};
    return r;
}

static void paint_palette(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    int W = rc.right, H = rc.bottom;
    gp::Bitmap buf(std::max(1, W), std::max(1, H), PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    gp::Color c = scheme_col(g_scheme);
    gp::LinearGradientBrush bg(gp::Rect(0, 0, W, H), gmix(gcol(10, 22, 44), c, 0.32f), gcol(7, 14, 28), gp::LinearGradientModeVertical);
    g.FillRectangle(&bg, 0, 0, W, H);
    if (W < 48) {  // the slid-in tool bar: a handle with an arrow
        gp::SolidBrush hb(gcol(c.GetR(), c.GetG(), c.GetB(), 200)); g.FillRectangle(&hb, W - 3, 0, 3, H);
        gp::Pen ar(gcol(255, 255, 255, 220), 2.0f);
        for (int k = 0; k < 3; ++k) { float yy = H / 2.0f - 30 + k * 30; g.DrawLine(&ar, 3.0f, yy - 6, W - 6.0f, yy); g.DrawLine(&ar, W - 6.0f, yy, 3.0f, yy + 6); }
        gp::Graphics out(dc); out.DrawImage(&buf, 0, 0);
        return;
    }
    gp::SolidBrush edge(gcol(c.GetR(), c.GetG(), c.GetB(), 140)); g.FillRectangle(&edge, W - 2, 0, 2, H);
    std::vector<int> tl = scheme_tools(g_scheme);
    for (size_t i = 0; i < tl.size(); ++i) {
        RECT t = tool_rect((int)i);
        if (t.bottom < PAL_HEAD - 10 || t.top > H) continue;
        gp::RectF r((float)t.left, (float)t.top, (float)(t.right - t.left), (float)(t.bottom - t.top));
        if ((int)i == g_hover_tool) {
            gp::GraphicsPath p; round_path(p, r, 8);
            gp::LinearGradientBrush hb(r, gcol(255, 255, 255, (int)i == g_press_tool ? 70 : 40), gcol(c.GetR(), c.GetG(), c.GetB(), 60), gp::LinearGradientModeVertical);
            g.FillPath(&hb, &p);
            gp::Pen pen(gshade(c, 0.4f), 1.3f); g.DrawPath(&pen, &p);
        }
        const ToolDef& td = TOOLS[tl[i]];
        float isz = 44, ix = r.X + (r.Width - isz) / 2, iy = r.Y + 3 + ((int)i == g_press_tool ? 1 : 0);
        if (gp::Bitmap* ic = icon_png(td.icon)) g.DrawImage(ic, ix, iy, isz, isz);
        gtext(g, td.label, gp::RectF(r.X - 3, r.Y + 47, r.Width + 6, 32), gcol(232, 242, 255), 10.5f, gp::FontStyleRegular, gp::StringAlignmentCenter,
              gp::StringAlignmentNear, true);
    }
    // header drawn last so scrolled tiles pass under it
    gp::LinearGradientBrush hb(gp::Rect(0, 0, W, PAL_HEAD), gmix(gcol(10, 22, 44), c, 0.45f), gmix(gcol(10, 22, 44), c, 0.3f), gp::LinearGradientModeVertical);
    g.FillRectangle(&hb, 0, 0, W - 2, PAL_HEAD - 4);
    gtext(g, SCHEMES[g_scheme].title, gp::RectF(8, 4, W - 14, 22), gcol(255, 255, 255), 13.5f, gp::FontStyleBold);
    gtext(g, tostr((long)tl.size()) + " tools", gp::RectF(8, 23, W - 14, 16), gshade(c, 0.55f), 11.0f);
    gp::SolidBrush ln(c); g.FillRectangle(&ln, 8, PAL_HEAD - 6, W - 18, 2);
    gp::Graphics out(dc);
    out.DrawImage(&buf, 0, 0);
}

static int palette_hit(int x, int y) {
    std::vector<int> tl = scheme_tools(g_scheme);
    if (y < PAL_HEAD - 4) return -1;
    for (size_t i = 0; i < tl.size(); ++i) { RECT t = tool_rect((int)i); if (x >= t.left && x < t.right && y >= t.top && y < t.bottom) return (int)i; }
    return -1;
}

static void run_tool(int ti) {
    const ToolDef& td = TOOLS[ti];
    if (td.kind) PostMessageA(g_main, WM_TOOL_INSERT, (WPARAM)ti, 0);
    else if (td.cmd) PostMessageA(g_main, WM_COMMAND, (WPARAM)td.cmd, 0);
}

static LRESULT CALLBACK palette_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_PAINT: { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_palette(h, dc); EndPaint(h, &ps); return 0; }
        case WM_ERASEBKGND: return 1;
        case WM_MOUSEMOVE: {
            palette_hover(true);
            int hit = palette_hit((short)LOWORD(l), (short)HIWORD(l));
            if (hit != g_hover_tool) {
                g_hover_tool = hit; InvalidateRect(h, nullptr, FALSE);
                if (hit >= 0) {
                    int ti = scheme_tools(g_scheme)[(size_t)hit];
                    const ToolDef& td = TOOLS[ti]; status(std::string(td.label) + ":  " + td.tip);
                    RECT t = tool_rect(hit); POINT p = {PALETTE_W + 6, t.top}; ClientToScreen(h, &p);
                    show_tip(ti, p.x, p.y);
                } else show_tip(-1, 0, 0);
            }
            TRACKMOUSEEVENT te = {sizeof te, TME_LEAVE, h, 0}; TrackMouseEvent(&te);
            return 0;
        }
        case WM_MOUSELEAVE: g_hover_tool = -1; g_press_tool = -1; show_tip(-1, 0, 0); palette_hover(false); InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_LBUTTONDOWN: g_press_tool = g_hover_tool; SetCapture(h); InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_LBUTTONUP: {
            ReleaseCapture();
            int hit = palette_hit((short)LOWORD(l), (short)HIWORD(l));
            if (hit >= 0 && hit == g_press_tool) { show_tip(-1, 0, 0); run_tool(scheme_tools(g_scheme)[(size_t)hit]); }
            g_press_tool = -1; InvalidateRect(h, nullptr, FALSE);
            return 0;
        }
        case WM_MOUSEWHEEL: {
            RECT rc; GetClientRect(h, &rc);
            int rows = ((int)scheme_tools(g_scheme).size() + 1) / 2, content = PAL_HEAD + rows * CELL_H + 8;
            g_pal_scroll = std::max(0, std::min(std::max(0, content - (int)rc.bottom), g_pal_scroll - (short)HIWORD(w) / 2));
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        }
    }
    return DefWindowProcA(h, m, w, l);
}

static void register_ui_classes(HINSTANCE hi) {
    WNDCLASSA a; memset(&a, 0, sizeof a);
    a.lpfnWndProc = ribbon_proc; a.hInstance = hi; a.hCursor = LoadCursor(nullptr, IDC_ARROW); a.lpszClassName = "AeroSynapseSchemes";
    RegisterClassA(&a);
    a.lpfnWndProc = palette_proc; a.lpszClassName = "AeroSynapsePalette";
    RegisterClassA(&a);
}
