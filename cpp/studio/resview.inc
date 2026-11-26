// ---------------------------------------------------------------- results viewer
// Every result folder opens here. It lists the folder's figures (PNG), its data tables (CSV, drawn as charts here, one
// panel per column), interactive 3D pages (HTML, opened in the browser), flow movies (MP4) and the summary, so the
// window always has something to show. Thumbnails along the bottom pick a figure; "Pop out" opens the entry in a window
// of its own (several can be open at once), where the wheel zooms, a drag pans and a double click fits it again.
static HWND g_sv, g_sv_list, g_sv_text, g_sv_img;
static std::string g_sv_dir, g_sv_title;
static std::vector<std::string> g_sv_files;  // the entries' file names (the summary is "summary.txt")
static std::vector<int> g_sv_kind;           // 0 figure, 1 movie, 2 chart (csv), 3 interactive 3D page (html), 4 summary
enum { ID_SV_LIST = 8700, ID_SV_FOLDER, ID_SV_SAVE, ID_SV_POP };
static const int SVK_PNG = 0, SVK_MP4 = 1, SVK_CSV = 2, SVK_HTML = 3, SVK_SUM = 4;
// A study that names its figures <STUDY>_<SETTING>_... and lists its settings in settings.json (the fin PINN does) gets a
// row of buttons: Everything, Overview (the figures of every setting together) and one per optimisation setting.
struct SvSetting { std::string key, label; gp::Color col; };
static std::vector<SvSetting> g_sv_set;
static int g_sv_filter = -2;  // -2 everything, -1 overview (_ALL_), k a setting
static std::vector<std::string> g_sv_all_files; static std::vector<int> g_sv_all_kind;
static std::vector<std::pair<gp::RectF, int>> g_sv_chips;
static void sv_fill_list();

struct CsvTable { std::vector<std::string> head; std::vector<std::vector<double>> col; };
static std::map<std::string, CsvTable> g_csv_cache;
static const CsvTable& csv_load(const std::string& path) {
    auto it = g_csv_cache.find(path);
    if (it != g_csv_cache.end()) return it->second;
    CsvTable& T = g_csv_cache[path];
    std::string s;
    try { s = js::read_file(path); } catch (...) { return T; }
    size_t pos = 0; bool first = true;
    while (pos < s.size()) {
        size_t e = s.find('\n', pos); if (e == std::string::npos) e = s.size();
        std::string line = s.substr(pos, e - pos); pos = e + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::vector<std::string> cells; size_t a = 0;
        while (true) { size_t c = line.find(',', a); cells.push_back(line.substr(a, c == std::string::npos ? std::string::npos : c - a)); if (c == std::string::npos) break; a = c + 1; }
        if (first) { T.head = cells; T.col.assign(cells.size(), std::vector<double>()); first = false; continue; }
        for (size_t k = 0; k < T.col.size(); ++k) { char* end = nullptr; double v = k < cells.size() ? strtod(cells[k].c_str(), &end) : NAN; if (k < cells.size() && end == cells[k].c_str()) v = NAN; T.col[k].push_back(v); }
    }
    return T;
}

static std::string sv_pretty(std::string n) { for (char& c : n) if (c == '_') c = ' '; return n; }

// a data table as small multiples: every column against the first, on dark cards
static void paint_csv_chart(gp::Graphics& g, const std::string& path, gp::RectF area) {
    const CsvTable& T = csv_load(path);
    if (T.head.size() < 2 || T.col.empty() || T.col[0].size() < 2) { gtext(g, "This table has no numbers to draw.", area, gcol(200, 210, 230), 13.0f); return; }
    int n = std::min<int>(16, (int)T.head.size() - 1);
    int nc = n <= 1 ? 1 : n <= 4 ? 2 : n <= 9 ? 3 : 4, nr = (n + nc - 1) / nc;
    float gap = 12, cw = (area.Width - gap * (nc - 1)) / nc, ch = (area.Height - gap * (nr - 1)) / nr;
    static const int PAL[8][3] = {{34, 211, 238}, {236, 72, 153}, {250, 204, 21}, {52, 211, 153}, {167, 139, 250}, {251, 146, 60}, {96, 165, 250}, {244, 114, 182}};
    const std::vector<double>& X = T.col[0];
    double x0 = 1e300, x1 = -1e300; for (double v : X) if (v == v) { x0 = std::min(x0, v); x1 = std::max(x1, v); }
    if (!(x1 > x0)) { x0 = 0; x1 = (double)X.size(); }
    for (int p = 0; p < n; ++p) {
        int r = p / nc, c = p % nc;
        gp::RectF card(area.X + c * (cw + gap), area.Y + r * (ch + gap), cw, ch);
        soft_shadow(g, card, 8, 5, 90);
        gp::GraphicsPath cp; round_path(cp, card, 8);
        gp::LinearGradientBrush cb(card, gcol(18, 28, 52), gcol(10, 18, 36), gp::LinearGradientModeVertical);
        g.FillPath(&cb, &cp);
        const int* pc = PAL[p % 8];
        gp::Pen edge(gcol(pc[0], pc[1], pc[2], 90), 1.2f); g.DrawPath(&edge, &cp);
        const std::vector<double>& Y = T.col[(size_t)p + 1];
        double y0 = 1e300, y1 = -1e300; for (double v : Y) if (v == v) { y0 = std::min(y0, v); y1 = std::max(y1, v); }
        gtext(g, sv_pretty(T.head[(size_t)p + 1]), gp::RectF(card.X + 10, card.Y + 4, card.Width - 20, 18), gcol(pc[0], pc[1], pc[2]), 11.5f, gp::FontStyleBold);
        if (!(y1 >= y0)) continue;
        if (y1 - y0 < 1e-300) { y0 -= 0.5; y1 += 0.5; }
        gp::RectF pl(card.X + 52, card.Y + 26, card.Width - 64, card.Height - 50);
        gp::Pen grid(gcol(255, 255, 255, 22), 1.0f);
        for (int k = 0; k <= 4; ++k) {
            float yy = pl.Y + pl.Height * k / 4.0f; g.DrawLine(&grid, pl.X, yy, pl.X + pl.Width, yy);
            char b[32]; snprintf(b, sizeof b, "%.3g", y1 - (y1 - y0) * k / 4.0);
            gtext(g, b, gp::RectF(card.X + 2, yy - 8, 48, 16), gcol(150, 165, 190), 9.0f);
        }
        char bx[64]; snprintf(bx, sizeof bx, "%.4g", x0); gtext(g, bx, gp::RectF(pl.X - 4, pl.Y + pl.Height + 2, 80, 16), gcol(150, 165, 190), 9.0f);
        snprintf(bx, sizeof bx, "%.4g", x1); gtext(g, bx, gp::RectF(pl.X + pl.Width - 60, pl.Y + pl.Height + 2, 64, 16), gcol(150, 165, 190), 9.0f);
        gtext(g, sv_pretty(T.head[0]), gp::RectF(pl.X + pl.Width / 2 - 60, pl.Y + pl.Height + 2, 120, 16), gcol(120, 135, 160), 9.0f);
        std::vector<gp::PointF> pts;
        size_t m = std::min(X.size(), Y.size()), stride = std::max<size_t>(1, m / 1500);
        for (size_t i = 0; i < m; i += stride) {
            if (!(Y[i] == Y[i])) continue;
            double xv = X[i] == X[i] ? X[i] : (double)i;
            pts.push_back(gp::PointF(pl.X + (float)((xv - x0) / (x1 - x0)) * pl.Width, pl.Y + (float)((y1 - Y[i]) / (y1 - y0)) * pl.Height));
        }
        if (pts.size() >= 2) {
            std::vector<gp::PointF> fill = pts; fill.push_back(gp::PointF(pts.back().X, pl.Y + pl.Height)); fill.push_back(gp::PointF(pts.front().X, pl.Y + pl.Height));
            gp::LinearGradientBrush fb(pl, gcol(pc[0], pc[1], pc[2], 70), gcol(pc[0], pc[1], pc[2], 0), gp::LinearGradientModeVertical);
            g.FillPolygon(&fb, fill.data(), (INT)fill.size());
            gp::Pen glow(gcol(pc[0], pc[1], pc[2], 60), 5.0f); glow.SetLineJoin(gp::LineJoinRound); g.DrawLines(&glow, pts.data(), (INT)pts.size());
            gp::Pen line(gcol(pc[0], pc[1], pc[2]), 1.8f); line.SetLineJoin(gp::LineJoinRound); g.DrawLines(&line, pts.data(), (INT)pts.size());
        }
    }
}

static void paint_summary_card(gp::Graphics& g, const std::string& dir, gp::RectF area) {
    std::string t; try { t = js::read_file(dir + "\\summary.txt"); } catch (...) { t = "(no summary.txt in " + dir + ")"; }
    gp::GraphicsPath cp; round_path(cp, area, 10);
    gp::SolidBrush bg(gcol(10, 16, 32, 235)); g.FillPath(&bg, &cp);
    gp::Pen edge(gcol(34, 211, 238, 110), 1.4f); g.DrawPath(&edge, &cp);
    gp::FontFamily ff(L"Consolas"); gp::Font f(&ff, 11.0f, gp::FontStyleRegular, gp::UnitPixel);
    gp::SolidBrush ink(gcol(205, 225, 245));
    float y = area.Y + 12; size_t pos = 0;
    while (pos < t.size() && y < area.Y + area.Height - 16) {
        size_t e = t.find('\n', pos); if (e == std::string::npos) e = t.size();
        std::string line = t.substr(pos, e - pos); pos = e + 1;
        bool head = !line.empty() && line.size() > 3 && std::all_of(line.begin(), line.begin() + std::min<size_t>(line.size(), 6), [](char c) { return std::isupper((unsigned char)c) || c == ' ' || std::isdigit((unsigned char)c); });
        gp::SolidBrush hb(gcol(250, 204, 21));
        std::wstring w = wide(line);
        g.DrawString(w.c_str(), (INT)w.size(), &f, gp::PointF(area.X + 14, y), head ? &hb : &ink);
        y += 15;
    }
}

// one entry drawn into a rectangle (the viewer's image area or a pop-out window), with zoom z and pan (px, py)
static void paint_entry(gp::Graphics& g, const std::string& dir, const std::string& file, int kind, gp::RectF area, float z = 1, float px = 0, float py = 0) {
    if (kind == SVK_CSV) { paint_csv_chart(g, dir + "\\" + file, area); return; }
    if (kind == SVK_SUM) { paint_summary_card(g, dir, area); return; }
    std::string f = file;
    if (kind == SVK_MP4 || kind == SVK_HTML) {
        f = f.substr(0, f.find_last_of('.')) + ".png";
        gtext(g, kind == SVK_MP4 ? "Movie: it plays in your video player (double-click the entry or Pop out to play it again)."
                                 : "Interactive 3D page: double-click the entry or press Pop out to open it in the browser (drag to turn it, scroll to zoom).",
              gp::RectF(area.X, area.Y, area.Width, 20), gcol(200, 220, 240), 11.0f);
        area.Y += 26; area.Height -= 26;
        if (!file_exists(dir + "\\" + f)) {  // no still picture: a big card that says what it is
            gp::GraphicsPath cp; round_path(cp, area, 14);
            gp::LinearGradientBrush b(area, gcol(99, 102, 241, 120), gcol(236, 72, 153, 120), gp::LinearGradientModeForwardDiagonal);
            g.FillPath(&b, &cp);
            gtext(g, kind == SVK_MP4 ? "MOVIE" : "INTERACTIVE 3D", gp::RectF(area.X, area.Y + area.Height / 2 - 40, area.Width, 40), gcol(255, 255, 255), 30.0f, gp::FontStyleBold);
            gtext(g, sv_pretty(file), gp::RectF(area.X, area.Y + area.Height / 2 + 6, area.Width, 24), gcol(230, 240, 255), 14.0f);
            return;
        }
    }
    if (gp::Bitmap* im = load_png(dir + "\\" + f)) {
        float iw = (float)im->GetWidth(), ih = (float)im->GetHeight();
        float s = std::min(area.Width / iw, area.Height / ih) * z;
        float w = iw * s, hh = ih * s, x = area.X + (area.Width - w) / 2 + px, y = area.Y + (area.Height - hh) / 2 + py;
        gp::RectF fr(x - 6, y - 6, w + 12, hh + 12);
        soft_shadow(g, fr, 10, 8, 120);
        gp::GraphicsPath p; round_path(p, fr, 10);
        gp::SolidBrush wb(gcol(255, 255, 255, 18)); g.FillPath(&wb, &p);
        gp::Pen glow(gcol(34, 211, 238, 120), 2.0f); g.DrawPath(&glow, &p);
        g.DrawImage(im, x, y, w, hh);
    } else gtext(g, "The picture could not be read: " + f, area, gcol(250, 160, 160), 13.0f);
}

static std::vector<std::pair<gp::RectF, int>> g_sv_thumbs;  // the thumbnails' rectangles and entries
static void paint_simimage(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    int W = std::max(1L, rc.right), H = std::max(1L, rc.bottom);
    gp::Bitmap buf(W, H, PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    gp::LinearGradientBrush bg(gp::Rect(0, 0, W, H), gcol(6, 14, 30), gcol(16, 30, 56), gp::LinearGradientModeVertical);
    g.FillRectangle(&bg, 0, 0, W, H);
    // the header: a gradient band, the title and chips counting what the folder holds
    gp::LinearGradientBrush band(gp::Rect(0, 0, W, 46), gcol(236, 72, 153), gcol(99, 102, 241), gp::LinearGradientModeHorizontal);
    g.FillRectangle(&band, 0, 0, W, 46);
    gloss(g, gp::RectF(0, 0, (float)W, 46), 0, 60);
    gtext(g, g_sv_title, gp::RectF(16, 0, (float)W - 380, 46), gcol(255, 255, 255), 18.0f, gp::FontStyleBold);
    int cnt[5] = {0, 0, 0, 0, 0}; for (int k : g_sv_kind) cnt[k]++;
    const char* cn[4] = {"figures", "movies", "charts", "3D pages"}; const char* c1[4] = {"figure", "movie", "chart", "3D page"};
    float cx = (float)W - 12;
    for (int k = 3; k >= 0; --k) {
        if (!cnt[k]) continue;
        std::string t = tostr(cnt[k]) + " " + (cnt[k] == 1 ? c1[k] : cn[k]);
        float w = 14 + 7.0f * t.size(); cx -= w + 8;
        gp::RectF chip(cx, 11, w, 24); gp::GraphicsPath cp; round_path(cp, chip, 12);
        gp::SolidBrush cb(gcol(255, 255, 255, 50)); g.FillPath(&cb, &cp);
        gtext(g, t, chip, gcol(255, 255, 255), 11.0f, gp::FontStyleBold, gp::StringAlignmentCenter);
    }
    float top = 56;
    g_sv_chips.clear();
    if (!g_sv_set.empty()) {  // the optimisation settings' buttons
        float x = 20, y = 54, h = 26;
        auto chip = [&](const std::string& t, gp::Color c, int id) {
            float w = 18 + 7.4f * t.size();
            if (x + w > W - 10) { x = 20; y += h + 6; }
            gp::RectF r(x, y, w, h); gp::GraphicsPath cp; round_path(cp, r, h / 2);
            bool on = g_sv_filter == id;
            gp::SolidBrush b(on ? c : gcol(c.GetR(), c.GetG(), c.GetB(), 45)); g.FillPath(&b, &cp);
            gp::Pen pe(c, on ? 2.0f : 1.2f); g.DrawPath(&pe, &cp);
            gtext(g, t, r, on ? gcol(255, 255, 255) : gshade(c, 0.55f), 11.0f, gp::FontStyleBold, gp::StringAlignmentCenter);
            g_sv_chips.push_back(std::make_pair(r, id));
            x += w + 8;
        };
        chip("Everything", gcol(148, 163, 184), -2);
        chip("Overview (all settings)", gcol(250, 204, 21), -1);
        for (size_t k = 0; k < g_sv_set.size(); ++k) chip(g_sv_set[k].label + "  " + g_sv_set[k].key, g_sv_set[k].col, (int)k);
        top = y + h + 10;
    }
    int sel = (int)SendMessageA(g_sv_list, LB_GETCURSEL, 0, 0);
    // thumbnails of the figures along the bottom
    g_sv_thumbs.clear();
    float thH = 0;
    std::vector<int> figs; for (size_t i = 0; i < g_sv_kind.size(); ++i) if (g_sv_kind[i] == SVK_PNG || g_sv_kind[i] == SVK_CSV) figs.push_back((int)i);
    if (figs.size() >= 2 && H > 360) {
        thH = 86;
        float tw = 112, x = 14, y = (float)H - thH + 6;
        gp::SolidBrush strip(gcol(0, 0, 0, 90)); g.FillRectangle(&strip, 0.0f, (float)H - thH, (float)W, thH);
        for (int i : figs) {
            if (x + tw > W - 10) break;
            gp::RectF r(x, y, tw, thH - 14);
            gp::GraphicsPath p; round_path(p, r, 6);
            gp::SolidBrush tb(gcol(20, 32, 60)); g.FillPath(&tb, &p);
            if (g_sv_kind[(size_t)i] == SVK_PNG) {
                if (gp::Bitmap* im = load_png(g_sv_dir + "\\" + g_sv_files[(size_t)i])) {
                    float iw = (float)im->GetWidth(), ih = (float)im->GetHeight(), s = std::min((tw - 8) / iw, (thH - 22) / ih);
                    g.DrawImage(im, r.X + (tw - iw * s) / 2, r.Y + (r.Height - ih * s) / 2, iw * s, ih * s);
                }
            } else gtext(g, "chart", r, gcol(34, 211, 238), 12.0f, gp::FontStyleBold, gp::StringAlignmentCenter);
            gp::Pen pe(i == sel ? gcol(250, 204, 21) : gcol(255, 255, 255, 50), i == sel ? 2.4f : 1.0f); g.DrawPath(&pe, &p);
            g_sv_thumbs.push_back(std::make_pair(r, i));
            x += tw + 8;
        }
    }
    gp::RectF area(20, top, (float)W - 40, (float)H - top - 14 - thH);
    if (sel >= 0 && sel < (int)g_sv_files.size()) paint_entry(g, g_sv_dir, g_sv_files[(size_t)sel], g_sv_kind[(size_t)sel], area);
    else gtext(g, "Pick an entry on the left.", gp::RectF(20, 60, (float)W - 40, 40), gcol(160, 200, 240), 14.0f);
    gp::Graphics out(dc);
    out.DrawImage(&buf, 0, 0);
}

static void sv_select(int i);
static LRESULT CALLBACK simimage_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_PAINT) { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_simimage(h, dc); EndPaint(h, &ps); return 0; }
    if (m == WM_ERASEBKGND) return 1;
    if (m == WM_LBUTTONDOWN) {
        float x = (float)(short)LOWORD(l), y = (float)(short)HIWORD(l);
        for (auto& t : g_sv_thumbs) if (t.first.Contains(x, y)) { sv_select(t.second); return 0; }
        for (auto& c : g_sv_chips) if (c.first.Contains(x, y)) { g_sv_filter = c.second; sv_fill_list(); InvalidateRect(h, nullptr, FALSE); return 0; }
    }
    return DefWindowProcA(h, m, w, l);
}

// ---------------------------------------------------------------- pop-out figure windows
struct FigWin { std::string dir, file; int kind = 0; float z = 1, px = 0, py = 0; int mx = 0, my = 0; bool drag = false; };
static LRESULT CALLBACK fig_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    FigWin* F = (FigWin*)GetWindowLongPtrA(h, GWLP_USERDATA);
    switch (m) {
        case WM_CREATE: { CREATESTRUCTA* cs = (CREATESTRUCTA*)l; SetWindowLongPtrA(h, GWLP_USERDATA, (LONG_PTR)cs->lpCreateParams); return 0; }
        case WM_ERASEBKGND: return 1;
        case WM_PAINT: {
            PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps);
            RECT rc; GetClientRect(h, &rc);
            int W = std::max(1L, rc.right), H = std::max(1L, rc.bottom);
            gp::Bitmap buf(W, H, PixelFormat32bppARGB); gp::Graphics g(&buf); quality(g);
            gp::LinearGradientBrush bg(gp::Rect(0, 0, W, H), gcol(8, 16, 34), gcol(18, 30, 58), gp::LinearGradientModeVertical);
            g.FillRectangle(&bg, 0, 0, W, H);
            if (F) paint_entry(g, F->dir, F->file, F->kind, gp::RectF(14, 12, (float)W - 28, (float)H - 40), F->z, F->px, F->py);
            gtext(g, "wheel zoom, drag pan, double-click fit, S save a copy", gp::RectF(14, (float)H - 24, (float)W - 28, 20), gcol(130, 160, 200), 10.5f);
            gp::Graphics o(dc); o.DrawImage(&buf, 0, 0);
            EndPaint(h, &ps); return 0;
        }
        case WM_MOUSEWHEEL: if (F) { float f = GET_WHEEL_DELTA_WPARAM(w) > 0 ? 1.2f : 1 / 1.2f; F->z = std::max(0.2f, std::min(20.0f, F->z * f)); F->px *= f; F->py *= f; InvalidateRect(h, nullptr, FALSE); } return 0;
        case WM_LBUTTONDOWN: if (F) { F->drag = true; F->mx = (short)LOWORD(l); F->my = (short)HIWORD(l); SetCapture(h); } return 0;
        case WM_MOUSEMOVE: if (F && F->drag) { int x = (short)LOWORD(l), y = (short)HIWORD(l); F->px += x - F->mx; F->py += y - F->my; F->mx = x; F->my = y; InvalidateRect(h, nullptr, FALSE); } return 0;
        case WM_LBUTTONUP: if (F) F->drag = false; ReleaseCapture(); return 0;
        case WM_LBUTTONDBLCLK: if (F) { F->z = 1; F->px = F->py = 0; InvalidateRect(h, nullptr, FALSE); } return 0;
        case WM_KEYDOWN:
            if (F && w == 'S' && F->kind == SVK_PNG) { std::string p = F->file; if (ask_path(true, "PNG image (*.png)\0*.png\0", "png", p)) CopyFileA((F->dir + "\\" + F->file).c_str(), p.c_str(), FALSE); }
            if (w == VK_ESCAPE) DestroyWindow(h);
            return 0;
        case WM_SIZE: InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_NCDESTROY: delete F; SetWindowLongPtrA(h, GWLP_USERDATA, 0); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}
static void pop_out(const std::string& dir, const std::string& file, int kind, const std::string& title) {
    if (kind == SVK_HTML || kind == SVK_MP4) { ShellExecuteA(g_main, "open", (dir + "\\" + file).c_str(), nullptr, nullptr, SW_SHOWNORMAL); return; }
    FigWin* F = new FigWin(); F->dir = dir; F->file = file; F->kind = kind;
    static int k = 0; int off = 30 * (k++ % 8);
    HWND w = CreateWindowExA(0, "AeroSynapseFigure", (title + ": " + sv_pretty(file.substr(0, file.find_last_of('.')))).c_str(), WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                             120 + off, 90 + off, 1100, 760, nullptr, nullptr, GetModuleHandleA(nullptr), F);
    if (!w) delete F;
}

static void sv_layout() {
    RECT rc; GetClientRect(g_sv, &rc);
    int W = rc.right, H = rc.bottom, lw = 300;
    MoveWindow(g_sv_list, 8, 8, lw - 16, 210, TRUE);
    int bw = (lw - 32) / 3;
    MoveWindow(GetDlgItem(g_sv, ID_SV_POP), 8, 224, bw, 30, TRUE);
    MoveWindow(GetDlgItem(g_sv, ID_SV_FOLDER), 16 + bw, 224, bw, 30, TRUE);
    MoveWindow(GetDlgItem(g_sv, ID_SV_SAVE), 24 + 2 * bw, 224, bw, 30, TRUE);
    MoveWindow(g_sv_text, 8, 262, lw - 16, H - 270, TRUE);
    MoveWindow(g_sv_img, lw, 0, W - lw, H, TRUE);
}

static void sv_select(int i) {
    if (i < 0 || i >= (int)g_sv_files.size()) return;
    SendMessageA(g_sv_list, LB_SETCURSEL, i, 0);
    InvalidateRect(g_sv_img, nullptr, FALSE);
}

static LRESULT CALLBACK sv_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_CREATE: {
            HINSTANCE hi = GetModuleHandleA(nullptr);
            g_sv_list = CreateWindowExA(WS_EX_CLIENTEDGE, "LISTBOX", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | LBS_NOTIFY | LBS_NOINTEGRALHEIGHT, 0, 0, 0, 0, h, (HMENU)ID_SV_LIST, hi, nullptr);
            CreateWindowExA(0, "BUTTON", "Pop out", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)ID_SV_POP, hi, nullptr);
            CreateWindowExA(0, "BUTTON", "Folder", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)ID_SV_FOLDER, hi, nullptr);
            CreateWindowExA(0, "BUTTON", "Save...", WS_CHILD | WS_VISIBLE | BS_OWNERDRAW, 0, 0, 0, 0, h, (HMENU)ID_SV_SAVE, hi, nullptr);
            g_sv_text = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "", WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL | ES_MULTILINE | ES_READONLY, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            g_sv_img = CreateWindowExA(0, "AeroSynapseSimImage", "", WS_CHILD | WS_VISIBLE, 0, 0, 0, 0, h, nullptr, hi, nullptr);
            static HFONT mono = CreateFontA(-12, 0, 0, 0, FW_NORMAL, 0, 0, 0, ANSI_CHARSET, 0, 0, CLEARTYPE_QUALITY, FIXED_PITCH, "Consolas");
            SendMessageA(g_sv_list, WM_SETFONT, (WPARAM)g_font, TRUE); SendMessageA(g_sv_text, WM_SETFONT, (WPARAM)mono, TRUE);
            return 0;
        }
        case WM_SIZE: sv_layout(); return 0;
        case WM_COMMAND: {
            int sel = (int)SendMessageA(g_sv_list, LB_GETCURSEL, 0, 0);
            bool ok = sel >= 0 && sel < (int)g_sv_files.size();
            if (LOWORD(w) == ID_SV_LIST && (HIWORD(w) == LBN_SELCHANGE || HIWORD(w) == LBN_DBLCLK)) {
                InvalidateRect(g_sv_img, nullptr, FALSE);
                if (ok && g_sv_kind[(size_t)sel] == SVK_MP4) ShellExecuteA(h, "open", (g_sv_dir + "\\" + g_sv_files[(size_t)sel]).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
                if (ok && HIWORD(w) == LBN_DBLCLK && g_sv_kind[(size_t)sel] != SVK_MP4) pop_out(g_sv_dir, g_sv_files[(size_t)sel], g_sv_kind[(size_t)sel], g_sv_title);
            }
            if (LOWORD(w) == ID_SV_POP && ok) pop_out(g_sv_dir, g_sv_files[(size_t)sel], g_sv_kind[(size_t)sel], g_sv_title);
            if (LOWORD(w) == ID_SV_FOLDER) ShellExecuteA(h, "open", g_sv_dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            if (LOWORD(w) == ID_SV_SAVE && ok) {
                std::string p = g_sv_files[(size_t)sel], ext = p.substr(p.find_last_of('.') + 1);
                if (ext == "png") {  // a figure: PNG, JPEG, TIFF, BMP or PDF, by the name's extension
                    if (ask_path(true, IMG_FILTER, "png", p)) {
                        std::string pe = p.substr(p.find_last_of('.') + 1); for (char& c : pe) c = (char)tolower((unsigned char)c);
                        if (pe == "png") CopyFileA((g_sv_dir + "\\" + g_sv_files[(size_t)sel]).c_str(), p.c_str(), FALSE);
                        else if (gp::Bitmap* im = load_png(g_sv_dir + "\\" + g_sv_files[(size_t)sel])) save_image_as(im, p);
                    }
                } else {
                    std::string filt = ext == "csv" ? std::string("CSV table (*.csv)\0*.csv\0", 24) : std::string("All files\0*.*\0", 14);
                    if (ask_path(true, filt.c_str(), ext.c_str(), p)) CopyFileA((g_sv_dir + "\\" + g_sv_files[(size_t)sel]).c_str(), p.c_str(), FALSE);
                }
            }
            return 0;
        }
        case WM_DRAWITEM: { DRAWITEMSTRUCT* d = (DRAWITEMSTRUCT*)l; draw_button(d, d->CtlID == ID_SV_POP ? gcol(99, 102, 241) : gcol(236, 72, 153)); return TRUE; }
        case WM_CTLCOLOREDIT: case WM_CTLCOLORLISTBOX: case WM_CTLCOLORSTATIC: { HDC dc = (HDC)w; SetBkColor(dc, C_FIELD); SetTextColor(dc, C_INK); return (LRESULT)g_br_field; }
        case WM_CLOSE: ShowWindow(h, SW_HIDE); return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

static void sv_add(const std::string& dir, const char* pat, int kind, std::vector<std::pair<std::string, int>>& out) {
    WIN32_FIND_DATAA fd; std::vector<std::string> v;
    HANDLE fh = FindFirstFileA((dir + "\\" + pat).c_str(), &fd);
    if (fh != INVALID_HANDLE_VALUE) { do { v.push_back(fd.cFileName); } while (FindNextFileA(fh, &fd)); FindClose(fh); }
    std::sort(v.begin(), v.end());
    for (auto& f : v) out.push_back(std::make_pair(f, kind));
}

static void sv_fill_list() {  // the entries the chosen button shows
    g_sv_files.clear(); g_sv_kind.clear();
    SendMessageA(g_sv_list, LB_RESETCONTENT, 0, 0);
    static const char* pre[5] = {"", "movie: ", "chart: ", "3D: ", ""};
    for (size_t i = 0; i < g_sv_all_files.size(); ++i) {
        const std::string& f = g_sv_all_files[i]; int k = g_sv_all_kind[i];
        if (g_sv_filter == -1 && k != SVK_SUM && f.find("_ALL_") == std::string::npos) continue;
        if (g_sv_filter >= 0 && g_sv_filter < (int)g_sv_set.size() && k != SVK_SUM && f.find("_" + g_sv_set[(size_t)g_sv_filter].key + "_") == std::string::npos) continue;
        g_sv_files.push_back(f); g_sv_kind.push_back(k);
        std::string n = k == SVK_SUM ? std::string("summary (the report)") : pre[k] + sv_pretty(f.substr(0, f.find_last_of('.')));
        SendMessageA(g_sv_list, LB_ADDSTRING, 0, (LPARAM)n.c_str());
    }
    SendMessageA(g_sv_list, LB_SETCURSEL, 0, 0);
    if (g_sv_img) InvalidateRect(g_sv_img, nullptr, FALSE);
}
static void open_simview(const std::string& dir, const std::string& title) {
    if (!g_sv) g_sv = CreateWindowExA(WS_EX_TOOLWINDOW, "AeroSynapseSimView", "Simulation results", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1320, 800, g_main, nullptr, GetModuleHandleA(nullptr), nullptr);
    g_sv_dir = dir; g_sv_title = title; g_sv_files.clear(); g_sv_kind.clear();
    SendMessageA(g_sv_list, LB_RESETCONTENT, 0, 0);
    std::vector<std::pair<std::string, int>> e;
    sv_add(dir, "*.png", SVK_PNG, e);
    sv_add(dir, "*.csv", SVK_CSV, e);
    sv_add(dir, "*.html", SVK_HTML, e);
    sv_add(dir, "*.mp4", SVK_MP4, e);
    {   // a still picture that belongs to a movie or a 3D page is shown with it, not twice
        std::set<std::string> stems; for (auto& p : e) if (p.second == SVK_MP4 || p.second == SVK_HTML) stems.insert(p.first.substr(0, p.first.find_last_of('.')));
        std::vector<std::pair<std::string, int>> keep;
        for (auto& p : e) if (!(p.second == SVK_PNG && stems.count(p.first.substr(0, p.first.size() - 4)))) keep.push_back(p);
        e.swap(keep);
    }
    if (file_exists(dir + "\\summary.txt")) e.push_back(std::make_pair(std::string("summary.txt"), SVK_SUM));
    g_csv_cache.clear();
    g_sv_all_files.clear(); g_sv_all_kind.clear();
    for (auto& p : e) { g_sv_all_files.push_back(p.first); g_sv_all_kind.push_back(p.second); if (p.second == SVK_PNG) forget_png(dir + "\\" + p.first); }
    g_sv_set.clear(); g_sv_filter = -2;
    try {  // the study's optimisation settings, for the buttons
        js::Value m = js::parse(js::read_file(dir + "\\settings.json"));
        if (const js::Value* a = m.find("settings"))
            for (size_t i = 0; i < a->size(); ++i) {
                const js::Value& v = a->at(i); std::string c = v.str("colour", "#888888"); unsigned rgb = 0x888888; sscanf(c.c_str() + (c[0] == '#'), "%x", &rgb);
                g_sv_set.push_back(SvSetting{v.str("key", ""), v.str("label", ""), gcol((rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255)});
            }
    } catch (...) {}
    sv_fill_list();
    std::string t;
    try { t = js::read_file(dir + "\\summary.txt"); } catch (...) { t = "(no summary.txt in " + dir + ")"; }
    SetWindowTextA(g_sv_text, crlf(t).c_str());
    SetWindowTextA(g_sv, ("Simulation results: " + title).c_str());
    sv_layout();
    ShowWindow(g_sv, SW_SHOW); SetForegroundWindow(g_sv);
    InvalidateRect(g_sv_img, nullptr, FALSE);
}
