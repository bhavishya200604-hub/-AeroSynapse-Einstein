// Appearances: every part drawn in its own colour, from its material or its appearance, like the parts of an
// assembly in a CAD program. A feature may carry "appearance" (one of APPEAR below) and "material" (a structural
// material of the analyses: al7075, ti64, cfrp_im7, ...). The display mode picks the rule:
//   by part      - explicit appearance, else the material's colour, else a distinct colour per part (default)
//   by material  - the material's colour (unset material: aluminium)
//   single       - one neutral grey
// The sampled mesh is coloured vertex by vertex by the part that owns the surface there (the feature with the lowest
// signed distance), so cut faces take the colour of the part they cut into.

struct Appear { const char* key; const char* label; float r, g, b; };
static const Appear APPEAR[] = {
    {"aluminium", "Aluminium", 0.68f, 0.70f, 0.73f}, {"steel", "Steel", 0.60f, 0.62f, 0.65f}, {"titanium", "Titanium", 0.50f, 0.51f, 0.54f},
    {"carbon", "Carbon fibre", 0.24f, 0.25f, 0.28f}, {"glass_fibre", "Glass fibre", 0.60f, 0.76f, 0.50f}, {"ceramic", "Ceramic", 0.86f, 0.82f, 0.72f},
    {"purple", "Purple", 0.42f, 0.22f, 0.64f}, {"violet", "Violet", 0.30f, 0.17f, 0.56f}, {"teal", "Teal", 0.24f, 0.60f, 0.62f},
    {"green", "Green", 0.52f, 0.76f, 0.24f}, {"magenta", "Magenta", 0.64f, 0.16f, 0.62f}, {"blue", "Blue", 0.22f, 0.42f, 0.78f},
    {"orange", "Orange", 0.92f, 0.54f, 0.16f}, {"red", "Red", 0.78f, 0.18f, 0.20f}, {"gold", "Gold", 0.84f, 0.68f, 0.26f},
    {"white", "White", 0.93f, 0.93f, 0.94f}, {"glass", "Glass (canopy)", 0.55f, 0.72f, 0.86f},
};
static const int NAPPEAR = (int)(sizeof(APPEAR) / sizeof(APPEAR[0]));
struct MatDef { const char* key; const char* label; const char* appear; };
static const MatDef MATERIALS[] = {  // the structural materials of the analyses, with the appearance each one shows
    {"al7075", "Aluminium 7075", "aluminium"}, {"al2024", "Aluminium 2024", "aluminium"}, {"al_li", "Aluminium-lithium", "aluminium"},
    {"ti64", "Titanium Ti-6Al-4V", "titanium"}, {"ti6242", "Titanium Ti-6242", "titanium"}, {"in718", "Inconel 718", "steel"},
    {"steel_300m", "Steel 300M", "steel"}, {"cfrp_im7", "Carbon/epoxy IM7", "carbon"}, {"cfrp_bmi", "Carbon/BMI", "carbon"},
    {"glass_epoxy", "Glass/epoxy", "glass_fibre"}, {"cmc_csic", "Ceramic matrix C/SiC", "ceramic"},
};
static const int NMATERIAL = (int)(sizeof(MATERIALS) / sizeof(MATERIALS[0]));
static const char* PART_CYCLE[] = {"aluminium", "purple", "teal", "green", "violet", "steel", "magenta", "blue", "gold", "orange"};
static int g_appear_mode = 0;  // 0 by part, 1 by material, 2 single
static const char* APPEAR_MODE_NAME[3] = {"by part", "by material", "single colour"};

static const Appear& appear_of(const std::string& key) {
    for (int i = 0; i < NAPPEAR; ++i) if (key == APPEAR[i].key) return APPEAR[i];
    return APPEAR[0];
}
static const MatDef* material_of(const std::string& key) {
    for (int i = 0; i < NMATERIAL; ++i) if (key == MATERIALS[i].key) return &MATERIALS[i];
    return nullptr;
}
// the colour of feature k of the design
static void part_colour(int k, float c[3]) {
    const Appear* a = &APPEAR[0];
    const js::Value& fl = features();
    if (g_appear_mode == 2 || k < 0 || k >= (int)fl.size()) { c[0] = 0.74f; c[1] = 0.76f; c[2] = 0.79f; return; }
    const js::Value& f = fl.at((size_t)k);
    const MatDef* m = material_of(f.str("material", ""));
    std::string ap = f.str("appearance", "");
    if (g_appear_mode == 1) a = &appear_of(m ? m->appear : "aluminium");
    else if (!ap.empty()) a = &appear_of(ap);
    else if (m) a = &appear_of(m->appear);
    else {
        std::string id = f.str("id", ""), ty = f.str("type", "");
        if (id.find("canopy") != std::string::npos || id.find("glass") != std::string::npos) a = &appear_of("glass");
        else a = &appear_of(PART_CYCLE[k % (int)(sizeof(PART_CYCLE) / sizeof(PART_CYCLE[0]))]);
    }
    c[0] = a->r; c[1] = a->g; c[2] = a->b;
}
static std::string part_appearance_name(int k) {
    const js::Value& fl = features();
    if (k < 0 || k >= (int)fl.size()) return "";
    const js::Value& f = fl.at((size_t)k);
    std::string ap = f.str("appearance", "");
    const MatDef* m = material_of(f.str("material", ""));
    return (m ? std::string(m->label) : std::string("no material")) + (ap.empty() ? "" : ", " + std::string(appear_of(ap).label));
}

// which part owns each vertex of a sampled mesh: the added feature with the lowest signed distance there
static std::vector<int> mesh_owners(const geo::Model& m, const geo::Mesh& M) {
    std::vector<int> o(M.v.size(), -1);
    if (m.feats.empty()) return o;
    par::parallel_for((int)M.v.size(), [&](int i) {
        const V3& p = M.v[(size_t)i];
        double best = 1e30; int bi = -1;
        for (size_t k = 0; k < m.feats.size(); ++k) {
            const geo::Feature& f = m.feats[k];
            if (f.op != geo::ADD) continue;
            if (m.boxes[k].dist(p) > best) continue;  // the box distance is a lower bound of the feature's distance
            double v = f.eval(p);
            if (v < best) { best = v; bi = f.src; }
        }
        o[(size_t)i] = bi;
    });
    return o;
}
// per-vertex colours of a sampled mesh from its owners (shade < 1 darkens: the caps of a section)
static const std::vector<float>& owner_colours(const std::vector<int>& own, float shade = 1.0f) {
    static std::vector<float> c;
    static std::vector<float> lut;
    const js::Value& fl = features();
    lut.assign((fl.size() + 1) * 3, 0.75f);
    for (size_t k = 0; k <= fl.size(); ++k) { float q[3]; part_colour(k < fl.size() ? (int)k : -1, q); for (int j = 0; j < 3; ++j) lut[k * 3 + (size_t)j] = q[j] * shade; }
    c.resize(own.size() * 3);
    for (size_t i = 0; i < own.size(); ++i) {
        size_t k = own[i] >= 0 && own[i] < (int)fl.size() ? (size_t)own[i] : fl.size();
        c[3 * i] = lut[3 * k]; c[3 * i + 1] = lut[3 * k + 1]; c[3 * i + 2] = lut[3 * k + 2];
    }
    return c;
}

// ---------------------------------------------------------------- the model tree, drawn like a CAD tree
static const char* type_icon(const std::string& ty) {
    static const char* known[] = {"fuselage", "wing", "canard", "tail", "fin", "strake", "blade", "sphere", "ellipsoid", "box", "cylinder", "cone", "torus",
                                  "capsule", "revolve", "extrude", "loft", "sheet", "duct", "implicit", "superellipsoid", "sweep", "lattice", "halfspace", "bentplate"};
    for (const char* k : known) if (ty == k) return k;
    return "box";
}
static const int TREE_HEAD = 26;
static void tree_draw_item(DRAWITEMSTRUCT* d) {
    if (g_asm_on) { asm_tree_draw(d); return; }
    RECT rc = d->rcItem;
    int W = rc.right - rc.left, H = rc.bottom - rc.top;
    if (W <= 0 || H <= 0) return;
    gp::Bitmap buf(W, H, PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    bool sel = (d->itemState & ODS_SELECTED) != 0;
    g.Clear(sel ? gcol(204, 224, 247) : gcol(255, 255, 255));
    if (sel) { gp::Pen pn(gcol(120, 165, 220), 1.0f); g.DrawRectangle(&pn, 0, 0, W - 1, H - 1); }
    const js::Value& fl = features();
    int k = (int)d->itemID;
    if (k >= 0 && k < (int)fl.size()) {
        const js::Value& f = fl.at((size_t)k);
        bool on = f.flag("enabled", true);
        std::string op = f.str("op", "add"), ty = f.str("type", "?"), id = f.str("id", "?");
        gp::Pen exp(gcol(150, 155, 162), 1.0f);  // the tree's expander mark
        g.DrawLine(&exp, 8.0f, H / 2.0f, 13.0f, H / 2.0f);
        if (gp::Bitmap* ic = icon_flat(type_icon(ty))) g.DrawImage(ic, 16.0f, (H - 16) / 2.0f, 16.0f, 16.0f);
        float c[3]; part_colour(k, c);
        gp::SolidBrush sw(gcol((int)(c[0] * 255), (int)(c[1] * 255), (int)(c[2] * 255)));
        g.FillRectangle(&sw, 36.0f, (H - 10) / 2.0f, 10.0f, 10.0f);
        gp::Pen swp(gcol(90, 95, 105), 1.0f); g.DrawRectangle(&swp, 36.0f, (H - 10) / 2.0f, 10.0f, 10.0f);
        std::string tag = op == "cut" ? "  (cut)" : op == "intersect" ? "  (intersect)" : "";
        if (f.flag("mirror", false)) tag += "  mirrored";
        gp::Color ink = on ? gcol(25, 28, 34) : gcol(160, 165, 172);
        float tw = text_w(g, id, 12.5f, sel ? gp::FontStyleBold : gp::FontStyleRegular);
        gtext(g, id, gp::RectF(52, 0, (float)W - 54, (float)H), ink, 12.5f, sel ? gp::FontStyleBold : gp::FontStyleRegular);
        gtext(g, (on ? "" : "hidden  ") + ty + tag, gp::RectF(56 + tw, 0, std::max(0.0f, W - 58 - tw), (float)H), gcol(120, 126, 136), 11.5f);
    }
    gp::Graphics out(d->hDC);
    out.DrawImage(&buf, (INT)rc.left, (INT)rc.top);
}
static void tree_paint_header(HDC dc, int x, int y, int w) {
    gp::Graphics g(dc);
    quality(g);
    gp::SolidBrush bg(gcol(243, 244, 246)); g.FillRectangle(&bg, x, y, w, TREE_HEAD);
    std::string name = g_path.empty() ? std::string("untitled") : g_path.substr(g_path.find_last_of("\\/") + 1);
    gtext(g, "Model Tree", gp::RectF((float)x + 8, (float)y, 120, (float)TREE_HEAD), gcol(30, 34, 40), 12.5f, gp::FontStyleBold);
    gtext(g, name, gp::RectF((float)x + 100, (float)y, (float)w - 106, (float)TREE_HEAD), gcol(110, 116, 126), 11.5f, gp::FontStyleRegular, gp::StringAlignmentFar);
    gp::Pen rule(gcol(205, 208, 213), 1.0f); g.DrawLine(&rule, x, y + TREE_HEAD - 1, x + w, y + TREE_HEAD - 1);
}

// right-click on a part (tree or 3D view): edit, dimensions, appearance, material, show/hide, delete
enum { ID_CTX_EDIT = 15100, ID_CTX_DIMS, ID_CTX_HIDE, ID_CTX_DEL, ID_CTX_DUP, ID_CTX_ISOLATE, ID_CTX_APPEAR0 = 15120, ID_CTX_MAT0 = 15160, ID_CTX_MODE0 = 15190 };
static void part_context_menu(int sx, int sy) {
    const js::Value& fl = features();
    if (g_sel < 0 || g_sel >= (int)fl.size()) return;
    HMENU m = CreatePopupMenu(), ap = CreatePopupMenu(), mt = CreatePopupMenu(), md = CreatePopupMenu();
    std::string cur_ap = fl.at((size_t)g_sel).str("appearance", ""), cur_mt = fl.at((size_t)g_sel).str("material", "");
    for (int i = 0; i < NAPPEAR; ++i) AppendMenuA(ap, MF_STRING | (cur_ap == APPEAR[i].key ? MF_CHECKED : 0), ID_CTX_APPEAR0 + i, APPEAR[i].label);
    AppendMenuA(ap, MF_SEPARATOR, 0, nullptr); AppendMenuA(ap, MF_STRING, ID_CTX_APPEAR0 + NAPPEAR, "Automatic (from the material or the part)");
    for (int i = 0; i < NMATERIAL; ++i) AppendMenuA(mt, MF_STRING | (cur_mt == MATERIALS[i].key ? MF_CHECKED : 0), ID_CTX_MAT0 + i, MATERIALS[i].label);
    AppendMenuA(mt, MF_SEPARATOR, 0, nullptr); AppendMenuA(mt, MF_STRING, ID_CTX_MAT0 + NMATERIAL, "None");
    for (int i = 0; i < 3; ++i) AppendMenuA(md, MF_STRING | (g_appear_mode == i ? MF_CHECKED : 0), ID_CTX_MODE0 + i, APPEAR_MODE_NAME[i]);
    std::string id = fl.at((size_t)g_sel).str("id", "part");
    AppendMenuA(m, MF_STRING | MF_GRAYED, 0, id.c_str());
    AppendMenuA(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(m, MF_STRING, ID_CTX_EDIT, "Edit definition (live preview)...");
    AppendMenuA(m, MF_STRING, ID_CTX_DIMS, "Show dimensions (click one to change it)");
    AppendMenuA(m, MF_POPUP, (UINT_PTR)mt, "Material");
    AppendMenuA(m, MF_POPUP, (UINT_PTR)ap, "Appearance");
    AppendMenuA(m, MF_POPUP, (UINT_PTR)md, "Colour all parts");
    AppendMenuA(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(m, MF_STRING, ID_CTX_HIDE, fl.at((size_t)g_sel).flag("enabled", true) ? "Hide" : "Show");
    AppendMenuA(m, MF_STRING, ID_CTX_DUP, "Duplicate");
    AppendMenuA(m, MF_STRING, ID_CTX_DEL, "Delete");
    TrackPopupMenu(m, TPM_LEFTALIGN | TPM_TOPALIGN | TPM_RIGHTBUTTON, sx, sy, 0, g_main, nullptr);
    DestroyMenu(m);
}
