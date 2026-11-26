// GDI+ drawing helpers for AeroSynapse Studio: PNG icons (assets/icons), gradients, rounded tiles with soft shadows,
// anti-aliased text and glossy gradient buttons. Included by studio.cpp.
#include <algorithm>
namespace Gdiplus { using std::min; using std::max; }
#include <gdiplus.h>
namespace gp = Gdiplus;

static ULONG_PTR g_gdip_token = 0;
static std::map<std::string, gp::Bitmap*> g_png_cache;

static void gdip_start() { gp::GdiplusStartupInput in; gp::GdiplusStartup(&g_gdip_token, &in, nullptr); }

static std::wstring wide(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 1 ? (size_t)(n - 1) : 0, L' ');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

static gp::Bitmap* load_png(const std::string& path) {
    auto it = g_png_cache.find(path);
    if (it != g_png_cache.end()) return it->second;
    gp::Bitmap* b = gp::Bitmap::FromFile(wide(path).c_str());
    if (b && b->GetLastStatus() != gp::Ok) { delete b; b = nullptr; }
    g_png_cache[path] = b;
    return b;
}
static bool save_png_bitmap(gp::Bitmap* bm, const std::string& path) {  // through GDI+'s PNG encoder
    UINT n = 0, sz = 0; gp::GetImageEncodersSize(&n, &sz);
    if (!sz) return false;
    std::vector<char> buf(sz); gp::ImageCodecInfo* ci = (gp::ImageCodecInfo*)buf.data();
    gp::GetImageEncoders(n, sz, ci);
    for (UINT i = 0; i < n; ++i) if (wcscmp(ci[i].MimeType, L"image/png") == 0) return bm->Save(wide(path).c_str(), &ci[i].Clsid, nullptr) == gp::Ok;
    return false;
}
static gp::Bitmap* icon_png(const std::string& name) { return load_png(g_root + "\\assets\\icons\\" + name + ".png"); }
static void forget_png(const std::string& path) {
    auto it = g_png_cache.find(path);
    if (it != g_png_cache.end()) { delete it->second; g_png_cache.erase(it); }
}

static gp::Color gcol(int r, int g, int b, int a = 255) { return gp::Color((BYTE)a, (BYTE)r, (BYTE)g, (BYTE)b); }
static gp::Color gmix(gp::Color a, gp::Color b, float t) {
    auto m = [t](BYTE x, BYTE y) { return (BYTE)std::max(0.0f, std::min(255.0f, x + (y - x) * t)); };
    return gp::Color(m(a.GetA(), b.GetA()), m(a.GetR(), b.GetR()), m(a.GetG(), b.GetG()), m(a.GetB(), b.GetB()));
}
static gp::Color gshade(gp::Color c, float f) { return f > 0 ? gmix(c, gcol(255, 255, 255, c.GetA()), f) : gmix(c, gcol(0, 0, 0, c.GetA()), -f); }
static gp::Color gfrom(COLORREF c, int a = 255) { return gcol(GetRValue(c), GetGValue(c), GetBValue(c), a); }

static void round_path(gp::GraphicsPath& p, gp::RectF r, float rad) {
    float d = std::min(rad * 2, std::min(r.Width, r.Height));
    if (d < 1) { p.AddRectangle(r); return; }
    p.AddArc(r.X, r.Y, d, d, 180, 90);
    p.AddArc(r.X + r.Width - d, r.Y, d, d, 270, 90);
    p.AddArc(r.X + r.Width - d, r.Y + r.Height - d, d, d, 0, 90);
    p.AddArc(r.X, r.Y + r.Height - d, d, d, 90, 90);
    p.CloseFigure();
}

static void fill_round(gp::Graphics& g, gp::RectF r, float rad, gp::Color top, gp::Color bottom) {
    gp::GraphicsPath p;
    round_path(p, r, rad);
    gp::RectF rr(r.X, r.Y - 1, r.Width, r.Height + 2);
    gp::LinearGradientBrush br(rr, top, bottom, gp::LinearGradientModeVertical);
    g.FillPath(&br, &p);
}

static void soft_shadow(gp::Graphics& g, gp::RectF r, float rad, int depth = 6, int alpha = 60) {
    for (int k = depth; k >= 1; --k) {
        gp::GraphicsPath p;
        round_path(p, gp::RectF(r.X - k * 0.5f + 1, r.Y + k * 0.8f, r.Width + k, r.Height + k * 0.6f), rad + k);
        gp::SolidBrush b(gcol(0, 0, 0, alpha / depth));
        g.FillPath(&b, &p);
    }
}

static void gloss(gp::Graphics& g, gp::RectF r, float rad, int alpha = 70) {
    gp::GraphicsPath p;
    round_path(p, gp::RectF(r.X + 1, r.Y + 1, r.Width - 2, r.Height * 0.48f), rad);
    gp::LinearGradientBrush br(gp::RectF(r.X, r.Y, r.Width, r.Height * 0.5f + 1), gcol(255, 255, 255, alpha), gcol(255, 255, 255, 4), gp::LinearGradientModeVertical);
    g.FillPath(&br, &p);
}

static void gtext(gp::Graphics& g, const std::string& s, gp::RectF r, gp::Color c, float px, int style = gp::FontStyleRegular,
                  gp::StringAlignment ha = gp::StringAlignmentNear, gp::StringAlignment va = gp::StringAlignmentCenter, bool wrap = false) {
    gp::FontFamily ff(L"Segoe UI");
    gp::Font f(&ff, px, style, gp::UnitPixel);
    gp::StringFormat sf;
    sf.SetAlignment(ha); sf.SetLineAlignment(va);
    if (!wrap) sf.SetFormatFlags(gp::StringFormatFlagsNoWrap);
    sf.SetTrimming(gp::StringTrimmingEllipsisCharacter);
    gp::SolidBrush b(c);
    std::wstring w = wide(s);
    g.DrawString(w.c_str(), -1, &f, r, &sf, &b);
}

static void quality(gp::Graphics& g) {
    g.SetSmoothingMode(gp::SmoothingModeAntiAlias);
    g.SetTextRenderingHint(gp::TextRenderingHintClearTypeGridFit);
    g.SetInterpolationMode(gp::InterpolationModeHighQualityBicubic);
    g.SetPixelOffsetMode(gp::PixelOffsetModeHalf);
}

// A glossy gradient push button with depth, for every owner-drawn button in Studio.
static void draw_button_light(DRAWITEMSTRUCT* d, bool danger) {  // a flat light button, as in a CAD program's panels
    RECT rc = d->rcItem;
    int W = rc.right - rc.left, H = rc.bottom - rc.top;
    gp::Bitmap buf(W, H, PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    g.Clear(gfrom(C_PANEL));
    bool down = (d->itemState & ODS_SELECTED) != 0;
    gp::RectF r(0.5f, 0.5f, W - 2.0f, H - 2.0f);
    gp::LinearGradientBrush bg(r, down ? gcol(214, 228, 245) : gcol(255, 255, 255), down ? gcol(200, 218, 240) : gcol(236, 238, 241), gp::LinearGradientModeVertical);
    g.FillRectangle(&bg, r);
    gp::Pen pen(down ? gcol(120, 160, 210) : gcol(185, 190, 198), 1.0f); g.DrawRectangle(&pen, r);
    char t[96]; GetWindowTextA(d->hwndItem, t, sizeof t);
    gtext(g, t, r, danger ? gcol(170, 30, 40) : gcol(30, 34, 40), 12.0f, gp::FontStyleRegular, gp::StringAlignmentCenter);
    gp::Graphics out(d->hDC);
    out.DrawImage(&buf, (INT)rc.left, (INT)rc.top);
}
static void draw_button(DRAWITEMSTRUCT* d, gp::Color base) {
    RECT rc = d->rcItem;
    int W = rc.right - rc.left, H = rc.bottom - rc.top;
    gp::Bitmap buf(W, H, PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    HWND par = GetParent(d->hwndItem);
    (void)par;
    g.Clear(gfrom(C_PANEL));
    bool down = (d->itemState & ODS_SELECTED) != 0, dis = (d->itemState & ODS_DISABLED) != 0;
    gp::RectF r(1.5f, 1.0f + (down ? 1 : 0), W - 3.5f, H - 3.5f);
    if (!down) soft_shadow(g, r, 6, 3, 50);
    gp::Color top = gshade(base, down ? -0.05f : 0.22f), bot = gshade(base, down ? -0.35f : -0.25f);
    if (dis) { top = gcol(170, 180, 195); bot = gcol(140, 150, 165); }
    fill_round(g, r, 6, top, bot);
    gloss(g, r, 6, down ? 30 : 75);
    gp::GraphicsPath p; round_path(p, r, 6);
    gp::Pen pen(gshade(base, -0.45f), 1.0f); g.DrawPath(&pen, &p);
    char t[96]; GetWindowTextA(d->hwndItem, t, sizeof t);
    gtext(g, t, gp::RectF(r.X + 4, r.Y, r.Width - 8, r.Height), gcol(255, 255, 255), 13.0f, gp::FontStyleBold, gp::StringAlignmentCenter);
    gp::Graphics out(d->hDC);
    out.DrawImage(&buf, (INT)rc.left, (INT)rc.top);
}
