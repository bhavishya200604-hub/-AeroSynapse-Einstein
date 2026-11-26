// Ultra-clear images of the 3D view. The view is drawn again off screen (an OpenGL framebuffer object) at up to 6K,
// with 2 x 2 supersampling (four samples a pixel, averaged): edges, text and lines come out sharp instead of an
// enlarged screen grab. Text and line widths grow with the picture, so the layout matches the screen. The picture can
// carry the AeroSynapse logo and the design's name, and is saved as PNG, JPEG, TIFF, BMP or PDF under a clean name:
// AeroSynapse_<design>_<view>_<width>x<height>_<date-time>.<format>. The result folders' 3D pictures use it too.

typedef void (APIENTRY* PFN_GenFB)(GLsizei, GLuint*);
typedef void (APIENTRY* PFN_BindFB)(GLenum, GLuint);
typedef void (APIENTRY* PFN_RBStorage)(GLenum, GLenum, GLsizei, GLsizei);
typedef void (APIENTRY* PFN_FBRenderbuffer)(GLenum, GLenum, GLenum, GLuint);
typedef GLenum (APIENTRY* PFN_CheckFB)(GLenum);
static PFN_GenFB p_glGenFramebuffers, p_glGenRenderbuffers, p_glDeleteFramebuffers, p_glDeleteRenderbuffers;
static PFN_BindFB p_glBindFramebuffer, p_glBindRenderbuffer;
static PFN_RBStorage p_glRenderbufferStorage;
static PFN_FBRenderbuffer p_glFramebufferRenderbuffer;
static PFN_CheckFB p_glCheckFramebufferStatus;
static const GLenum HQ_FB = 0x8D40, HQ_RB = 0x8D41, HQ_COL0 = 0x8CE0, HQ_DEPTH = 0x8D00, HQ_STENCIL = 0x8D20, HQ_D24S8 = 0x88F0, HQ_RGBA8 = 0x8058, HQ_DONE = 0x8CD5, HQ_MAXRB = 0x84E8;

static bool hq_load() {
    static int ok = -1;
    if (ok >= 0) return ok == 1;
    auto get = [](const char* a, const char* b) { PROC p = wglGetProcAddress(a); if (!p) p = wglGetProcAddress(b); return p; };
    p_glGenFramebuffers = (PFN_GenFB)get("glGenFramebuffersEXT", "glGenFramebuffers");
    p_glGenRenderbuffers = (PFN_GenFB)get("glGenRenderbuffersEXT", "glGenRenderbuffers");
    p_glDeleteFramebuffers = (PFN_GenFB)get("glDeleteFramebuffersEXT", "glDeleteFramebuffers");
    p_glDeleteRenderbuffers = (PFN_GenFB)get("glDeleteRenderbuffersEXT", "glDeleteRenderbuffers");
    p_glBindFramebuffer = (PFN_BindFB)get("glBindFramebufferEXT", "glBindFramebuffer");
    p_glBindRenderbuffer = (PFN_BindFB)get("glBindRenderbufferEXT", "glBindRenderbuffer");
    p_glRenderbufferStorage = (PFN_RBStorage)get("glRenderbufferStorageEXT", "glRenderbufferStorage");
    p_glFramebufferRenderbuffer = (PFN_FBRenderbuffer)get("glFramebufferRenderbufferEXT", "glFramebufferRenderbuffer");
    p_glCheckFramebufferStatus = (PFN_CheckFB)get("glCheckFramebufferStatusEXT", "glCheckFramebufferStatus");
    ok = p_glGenFramebuffers && p_glGenRenderbuffers && p_glDeleteFramebuffers && p_glDeleteRenderbuffers && p_glBindFramebuffer && p_glBindRenderbuffer &&
         p_glRenderbufferStorage && p_glFramebufferRenderbuffer && p_glCheckFramebufferStatus ? 1 : 0;
    return ok == 1;
}

// the view drawn at k times the window (k may be fractional), with ss x ss supersampling; the result is
// (window * k) pixels, RGB top row first. Returns false (and an explanation) when the graphics card cannot.
static bool hq_capture(double k, int ss, int& OW, int& OH, std::vector<unsigned char>& out, std::string& why) {
    if (!g_view || !g_hdc || !g_rc) { why = "there is no 3D view"; return false; }
    wglMakeCurrent(g_hdc, g_rc);
    if (!hq_load()) { why = "the graphics driver has no off-screen framebuffers (OpenGL framebuffer objects)"; return false; }
    RECT rc; GetClientRect(g_view, &rc);
    int lw = std::max(1L, rc.right), lh = std::max(1L, rc.bottom);
    GLint maxrb = 4096; glGetIntegerv(HQ_MAXRB, &maxrb); maxrb = std::min(maxrb, (GLint)16384);
    OW = (int)std::lround(lw * k); OH = (int)std::lround(lh * k);
    while (ss > 1 && (OW * ss > maxrb || OH * ss > maxrb || (double)OW * OH * ss * ss > 1.4e8)) --ss;  // the card's limit and about 560 MB
    if (OW > maxrb || OH > maxrb) { double f = std::min((double)maxrb / OW, (double)maxrb / OH); OW = (int)(OW * f); OH = (int)(OH * f); }
    int RW = OW * ss, RH = OH * ss;
    GLuint fb = 0, rb[2] = {0, 0};
    p_glGenFramebuffers(1, &fb); p_glGenRenderbuffers(2, rb);
    p_glBindFramebuffer(HQ_FB, fb);
    p_glBindRenderbuffer(HQ_RB, rb[0]); p_glRenderbufferStorage(HQ_RB, HQ_RGBA8, RW, RH);
    p_glFramebufferRenderbuffer(HQ_FB, HQ_COL0, HQ_RB, rb[0]);
    p_glBindRenderbuffer(HQ_RB, rb[1]); p_glRenderbufferStorage(HQ_RB, HQ_D24S8, RW, RH);
    p_glFramebufferRenderbuffer(HQ_FB, HQ_DEPTH, HQ_RB, rb[1]); p_glFramebufferRenderbuffer(HQ_FB, HQ_STENCIL, HQ_RB, rb[1]);
    bool okfb = p_glCheckFramebufferStatus(HQ_FB) == HQ_DONE;
    if (okfb) {
        // text at the picture's size: the window's font, scaled
        double sc = (double)RW / lw;
        GLuint base_screen = g_font_base, base_hr = glGenLists(96);
        LOGFONTA lf; GetObjectA(g_font, sizeof lf, &lf); lf.lfHeight = (LONG)std::lround(lf.lfHeight * sc); lf.lfQuality = ANTIALIASED_QUALITY;
        HFONT big = CreateFontIndirectA(&lf); HGDIOBJ old = SelectObject(g_hdc, big);
        wglUseFontBitmapsA(g_hdc, 32, 96, base_hr);
        SelectObject(g_hdc, old);
        g_font_base = base_hr; g_gl_lscale = (float)sc;
        g_hr_on = true; g_hr_W = RW; g_hr_H = RH; g_hr_lW = lw; g_hr_lH = lh;
        try { render(); } catch (...) {}
        g_hr_on = false; g_gl_lscale = 1; g_font_base = base_screen;
        glDeleteLists(base_hr, 96); DeleteObject(big);
        std::vector<unsigned char> px((size_t)RW * RH * 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, RW, RH, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        out.assign((size_t)OW * OH * 3, 0);
        for (int y = 0; y < OH; ++y)  // the ss x ss block average, the rows turned top first
            for (int x = 0; x < OW; ++x) {
                unsigned s[3] = {0, 0, 0};
                for (int b = 0; b < ss; ++b) for (int a = 0; a < ss; ++a) { const unsigned char* p = &px[((size_t)(y * ss + b) * RW + (size_t)(x * ss + a)) * 4]; s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; }
                unsigned char* o = &out[((size_t)(OH - 1 - y) * OW + x) * 3];
                for (int c = 0; c < 3; ++c) o[c] = (unsigned char)(s[c] / (ss * ss));
            }
    } else why = "the graphics card refused a " + tostr(RW) + " x " + tostr(RH) + " framebuffer";
    p_glBindFramebuffer(HQ_FB, 0);
    p_glDeleteRenderbuffers(2, rb); p_glDeleteFramebuffers(1, &fb);
    glViewport(0, 0, lw, lh);
    InvalidateRect(g_view, nullptr, FALSE);
    return okfb;
}

static std::string hq_design_name() {
    std::string n = g_path.empty() ? std::string("design") : g_path.substr(g_path.find_last_of("\\/") + 1);
    size_t d = n.find_last_of('.'); if (d != std::string::npos) n = n.substr(0, d);
    std::string o; for (char c : n) o += (isalnum((unsigned char)c) || c == '-') ? c : '-';
    return o.empty() ? "design" : o;
}
static std::string hq_clean(std::string s) { std::string o; for (char c : s) if (isalnum((unsigned char)c)) o += c; else if (!o.empty() && o.back() != '-') o += '-'; while (!o.empty() && o.back() == '-') o.pop_back(); return o.empty() ? "view" : o; }

// the picture as a GDI+ bitmap, with the logo and the name in a pill at the lower right when asked
static gp::Bitmap* hq_bitmap(int W, int H, const std::vector<unsigned char>& rgb, bool stamp, const std::string& what) {
    gp::Bitmap* bm = new gp::Bitmap(W, H, PixelFormat24bppRGB);
    gp::Rect r(0, 0, W, H); gp::BitmapData bd;
    if (bm->LockBits(&r, gp::ImageLockModeWrite, PixelFormat24bppRGB, &bd) == gp::Ok) {
        for (int y = 0; y < H; ++y) { unsigned char* d = (unsigned char*)bd.Scan0 + (size_t)y * bd.Stride; const unsigned char* s = &rgb[(size_t)y * W * 3]; for (int x = 0; x < W; ++x) { d[3 * x] = s[3 * x + 2]; d[3 * x + 1] = s[3 * x + 1]; d[3 * x + 2] = s[3 * x]; } }
        bm->UnlockBits(&bd);
    }
    if (stamp) {
        gp::Graphics g(bm); quality(g);
        float u = std::max(1.0f, W / 1400.0f), ph = 34 * u, lg = 26 * u;
        SYSTEMTIME t; GetLocalTime(&t); char d[24]; snprintf(d, sizeof d, "%04d-%02d-%02d", t.wYear, t.wMonth, t.wDay);
        std::string txt = "AeroSynapse Studio  |  " + what + "  |  " + d;
        float tw = 7.2f * u * txt.size() + lg + 30 * u;
        gp::RectF pill((float)W - tw - 18 * u, (float)H - ph - 16 * u, tw, ph);
        gp::GraphicsPath p; round_path(p, pill, ph / 2);
        gp::SolidBrush bg(gcol(255, 255, 255, 215)); g.FillPath(&bg, &p);
        gp::Pen pe(gcol(99, 102, 241, 160), 1.2f * u); g.DrawPath(&pe, &p);
        if (gp::Bitmap* logo = load_png(g_root + "\\assets\\logo.png")) g.DrawImage(logo, pill.X + 6 * u, pill.Y + (ph - lg) / 2, lg, lg);
        gtext(g, txt, gp::RectF(pill.X + lg + 14 * u, pill.Y, tw - lg - 18 * u, ph), gcol(30, 41, 59), 11.5f * u, gp::FontStyleBold);
    }
    return bm;
}

// a saved picture for the result folders: twice the view, supersampled, with the logo
static bool hq_save_png(const std::string& path, double k, int ss, bool stamp) {
    int W = 0, H = 0; std::vector<unsigned char> rgb; std::string why;
    if (!hq_capture(k, ss, W, H, rgb, why)) return false;
    gp::Bitmap* bm = hq_bitmap(W, H, rgb, stamp, hq_design_name() + "  |  " + view_name());
    bool ok = save_image_as(bm, path);
    delete bm; forget_png(path);
    return ok;
}

// the tool: Ultra-clear image...
static void hq_image_tool() {
    std::vector<FField> f;
    f.push_back({"Resolution", {"2x the view", "3x the view", "4K (3840 px wide)", "6K (5760 px wide)", "8K (7680 px wide)"}, "4K (3840 px wide)"});
    f.push_back({"Smoothing", {"2 x 2 supersampling (sharpest)", "none (faster)"}, "2 x 2 supersampling (sharpest)"});
    f.push_back({"Format", {"PNG (lossless)", "JPEG (quality 95)", "TIFF", "BMP", "PDF (the picture on a page)"}, "PNG (lossless)"});
    f.push_back({"Logo and name", {"yes", "no"}, "yes"});
    if (!form_dialog("Ultra-clear image", "The 3D view drawn again off screen at the size you pick, with four samples a pixel: sharp edges, text and lines, the same layout as on the screen.", f)) return;
    RECT rc; GetClientRect(g_view, &rc); double lw = std::max(1L, rc.right);
    double k = f[0].value[0] == '2' ? 2 : f[0].value[0] == '3' ? 3 : f[0].value[0] == '4' ? 3840 / lw : f[0].value[0] == '6' ? 5760 / lw : 7680 / lw;
    int ss = f[1].value[0] == '2' ? 2 : 1;
    std::string ext = f[2].value[0] == 'J' ? "jpg" : f[2].value[0] == 'T' ? "tif" : f[2].value[0] == 'B' ? "bmp" : f[2].value[1] == 'D' ? "pdf" : "png";
    int W = 0, H = 0; std::vector<unsigned char> rgb; std::string why;
    status("drawing the ultra-clear image...");
    if (!hq_capture(k, ss, W, H, rgb, why)) { MessageBoxA(g_main, ("The ultra-clear image could not be drawn: " + why + ".").c_str(), "Ultra-clear image", MB_ICONWARNING); return; }
    SYSTEMTIME t; GetLocalTime(&t); char d[32]; snprintf(d, sizeof d, "%04d%02d%02d-%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute);
    std::string name = "AeroSynapse_" + hq_design_name() + "_" + hq_clean(view_name()) + "_" + tostr(W) + "x" + tostr(H) + "_" + d + "." + ext;
    std::string p = work_dir("images") + "\\" + name;
    const char* filt = ext == "png" ? "PNG image (*.png)\0*.png\0" : ext == "jpg" ? "JPEG image (*.jpg)\0*.jpg\0" : ext == "tif" ? "TIFF image (*.tif)\0*.tif\0" : ext == "bmp" ? "Bitmap (*.bmp)\0*.bmp\0" : "PDF document (*.pdf)\0*.pdf\0";
    if (!ask_path(true, filt, ext.c_str(), p)) return;
    gp::Bitmap* bm = hq_bitmap(W, H, rgb, f[3].value == "yes", hq_design_name() + "  |  " + view_name());
    bool ok = save_image_as(bm, p);
    delete bm;
    status(ok ? "ultra-clear image saved: " + p + " (" + tostr(W) + " x " + tostr(H) + (ss > 1 ? ", 2 x 2 supersampled)" : ")") : "the image could not be saved: " + p);
    if (ok) ShellExecuteA(g_main, "open", p.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}
