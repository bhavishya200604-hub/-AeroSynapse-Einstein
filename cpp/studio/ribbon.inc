// The ribbon of AeroSynapse Studio, in the manner of Creo and other CAD programs: a row of text tabs (one per
// workbench, with File at the left holding every menu), and below it the active workbench's tools in named groups.
// The first tool of a group is a large button (icon over its label); the others are small buttons (icon beside the
// label) stacked three to a column. Group names sit under each group, and thin rules separate the groups.
// It replaces the dark scheme bar and the tool palette; the tools themselves (tools.inc) are unchanged.

struct RGroup { const char* scheme; const char* title; const char* icons; };  // icons: comma list; '*' = large button
static const RGroup RGROUPS[] = {
    {"start", "File", "*start_page,new_blank,wf_new,wf_open,save"},
    {"start", "Settings", "*prefs,*wf_window"},
    {"sketch", "Sketch plane", "*plane_xy,plane_xz,plane_yz"},
    {"sketch", "Draw", "sketch_line,sketch_rect,sketch_circle,sketch_slot"},
    {"sketch", "Make solid", "*sketch_revolve,sketch_extrude,sketch_loft"},
    {"sketch", "AutoCAD 2D", "*draft,draft_views,draft_dxf"},
    {"part", "Shapes", "*pad,pocket,shaft,groove,rib,cut_slot"},
    {"part", "Engineering", "*cut_hole,blend,chamfer,shell,draft"},
    {"part", "Pattern", "mirror,pattern"},
    {"part", "Primitives", "*fuselage,sphere,ellipsoid,box,cylinder,cone,torus,capsule"},
    {"curves", "Shapes", "*revolve,extrude,loft,sheet,duct"},
    {"curves", "Math surfaces", "*implicit,superellipsoid,sweep,lattice,halfspace"},
    {"curves", "Surface quality", "zebra,curvature"},
    {"aero", "Lifting surfaces", "*wing,canard,tail,fin,strake,blade"},
    {"aero", "Sections", "*airfoil"},
    {"sheet", "Sheet metal", "*bentplate,*sheet_sketch,sheet_flat,unfold"},
    {"sheet", "Presets", "*sheet_channel,sheet_z,sheet_hat,sheet_l"},
    {"sheet", "Composites", "*laminate,infill,shell"},
    {"asm", "Assembly", "*asm_new,asm_from_part,asm_insert,asm_demo"},
    {"asm", "Fasteners", "*asm_bolt,asm_screw,asm_nut,asm_washer,asm_pin,asm_shaft"},
    {"asm", "Mates", "*asm_mate,*asm_insert_hole"},
    {"asm", "Motion", "*asm_play,asm_motor,asm_key,asm_trace,asm_reset,*asm_explode"},
    {"asm", "More", "clash,reserve,kinematics,pivot,pipe,harness"},
    {"mech", "Gears", "*asm_gear,asm_gear_helical,asm_gear_bevel,asm_rack,asm_worm,asm_worm_wheel,*asm_gear_mate"},
    {"mech", "Drive parts", "*asm_bearing,asm_bushing,asm_keyshaft,asm_key,asm_coupling,asm_roller,*asm_roll_mate"},
    {"mech", "Cams", "*asm_cam,asm_follower,*asm_cam_mate"},
    {"mech", "Belts & chains", "*asm_pulley,asm_sprocket,*asm_belt,*asm_chain"},
    {"mech", "Analysis", "*asm_check,*asm_power,env_setup"},
    {"mech", "Examples", "*asm_drive_demo,asm_belt_demo,asm_worm_demo,asm_cam_demo,asm_gear_demo,asm_gear_demo2,asm_flow_cyl,asm_flow_tank"},
    {"xform", "Edit", "*undo,redo,duplicate,delete,booleans"},
    {"xform", "Deform", "twist,taper,bend,shear,stretch"},
    {"xform", "Position", "mirror,pattern,pivot,clear"},
    {"view", "Orientation", "*view_iso,view_front,view_side,view_top,view_fit,zoom_in,zoom_out"},
    {"view", "Display", "view_wire,full_view,panel_tree,panel_props,resolution"},
    {"view", "Section", "*cut_x,cut_y,cut_z"},
    {"view", "Inspect", "*dims_part,*appearance,*hq_image,annotate,measure,check,zebra,curvature"},
    {"sim", "CFD", "*cfd_cruise,cfd_transonic,cfd_super,cfd_alpha,cfd_sideslip,cfd_gust,*sim_setup,cfd_quality,live"},
    {"sim", "Flow display", "*flow_field,particles,cfd_flowlines,streamlines,vortex,slice,cfd_sweep,cfd_record"},
    {"sim", "Parts in flow", "*asm_flow,asm_flow_stop,asm_flow_view,asm_flow_mode,asm_flow_walls,asm_flow_refine,asm_flow_surf,asm_flow_splash,asm_flow_gears,asm_flow_belt"},
    {"sim", "Results", "*results,colour_cp,colour_cf,colour_t,clear_results"},
    {"sim", "Physics", "*quick_aero,env_setup,analytic,turbulence,structure,verify"},
    {"sim", "Thermal and PINN", "*pinn,*fin_pinn,pinn_lab,pinn_2d,pinn_inverse"},
    {"sim", "Labs (inside Studio)", "*labs,*labs_airfrans,labs_app,labs_designer,labs_wb"},
    {"opt", "Wing", "*optimise,*opt_wing_apply"},
    {"opt", "Airfoil and shape", "*opt_airfoil,opt_xfoil,opt_topology"},
    {"opt", "Studies", "*opt_fighter,opt_aircraft,opt_structure,opt_surrogate"},
    {"opt", "Centre", "*opt_centre,*labs_designer"},
    {"know", "Rules", "*rules,check"},
    {"know", "Top-down", "*skeleton,rflp"},
    {"wf", "Project", "*wf_window,wf_new,wf_open,wf_run"},
    {"wf", "Stages", "stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8"},
    {"export", "CAD", "*package,*hq_image,stl,obj,svg,save_as"},
    {"export", "2D drawing (AutoCAD)", "*draft,draft_views,draft_dxf,sheet_flat"},
    {"export", "Reports", "*report_pdf,report_docx,report_txt,report_html"},
};
static const int NRGROUP = (int)(sizeof(RGROUPS) / sizeof(RGROUPS[0]));

static const int RIB_TAB_H = 27, RIB_BODY_H = 94;
struct RItem { int tool; bool large; RECT r; int app = -1; };  // app >= 0: a group of the Applications tab
struct RGroupBox { std::string title; RECT r; std::vector<RItem> items; };
static std::vector<RGroupBox> g_rgroups;
static std::vector<RECT> g_rtabs;  // 0 = File, k + 1 = scheme k
static RECT g_rfile_rect, g_rquick[6], g_rpanel[3];
static const int RQUICK_W = 172;  // the quick-access group at the right end of the ribbon
static int g_rhover_tab = -1, g_rhover_item = -1, g_rpress_item = -1, g_rhover_quick = -1;
static HMENU g_menu_all = nullptr;  // every menu of the old menu bar, under the File tab
static const int RQUICK_CMD[6] = {ID_FILE_NEW, ID_FILE_OPEN, ID_FILE_SAVE, ID_EDIT_UNDO, ID_EDIT_REDO, ID_VIEW_FIT};
static const char* RQUICK_ICON[6] = {"new_blank", "wf_open", "save", "undo", "redo", "view_fit"};
static const char* RQUICK_TIP[6] = {"New (Ctrl+N)", "Open (Ctrl+O)", "Save (Ctrl+S)", "Undo (Ctrl+Z)", "Redo (Ctrl+Y)", "Fit the model (F)"};

static std::string app_short(const std::string& g) {  // a catalogue group's name on its ribbon button
    std::string p = "Thermal PINN research: ";
    if (g.compare(0, p.size(), p) == 0) return "PINN " + g.substr(p.size());
    return g;
}
static gp::Bitmap* icon_flat(const std::string& name) {
    if (gp::Bitmap* b = load_png(g_root + "\\assets\\icons_flat\\" + name + ".png")) return b;
    return icon_png(name);
}
static int find_tool(const std::string& scheme, const std::string& icon) {
    for (int i = 0; i < NTOOL; ++i) if (scheme == TOOLS[i].scheme && icon == TOOLS[i].icon) return i;
    for (int i = 0; i < NTOOL; ++i) if (icon == TOOLS[i].icon) return i;  // a tool shown on another tab too
    return -1;
}
static float text_w(gp::Graphics& g, const std::string& s, float px, int style = gp::FontStyleRegular) {
    gp::FontFamily ff(L"Segoe UI"); gp::Font f(&ff, px, style, gp::UnitPixel);
    gp::RectF box; std::wstring w = wide(s);
    g.MeasureString(w.c_str(), -1, &f, gp::PointF(0, 0), &box);
    return box.Width;
}
// the two lines of a large button's label
static void split2(const std::string& s, std::string& a, std::string& b) {
    a = s; b.clear();
    if (s.size() < 9) return;
    size_t m = s.size() / 2, best = std::string::npos;
    for (size_t i = 0; i < s.size(); ++i) if (s[i] == ' ' && (best == std::string::npos || (i > m ? i - m : m - i) < (best > m ? best - m : m - best))) best = i;
    if (best == std::string::npos) return;
    a = s.substr(0, best); b = s.substr(best + 1);
}

static void ribbon_layout(gp::Graphics& g, int W) {
    // tabs
    g_rtabs.clear();
    int x = 4;
    g_rfile_rect = {x, 2, x + 52, RIB_TAB_H};
    x += 56;
    for (int k = 0; k < NSCHEME; ++k) {
        int w = (int)text_w(g, SCHEMES[k].title, 12.5f) + 20;
        g_rtabs.push_back({x, 3, x + w, RIB_TAB_H + 1});
        x += w;
    }
    for (int q = 0; q < 6; ++q) { int qx = W - RQUICK_W + 6 + q * 27; g_rquick[q] = {qx, RIB_TAB_H + 8, qx + 25, RIB_TAB_H + 33}; }
    for (int k = 0; k < 3; ++k) { int qx = W - RQUICK_W + 6 + k * 27; g_rpanel[k] = {qx, RIB_TAB_H + 38, qx + 25, RIB_TAB_H + 63}; }
    // groups of the active workbench
    std::string key = SCHEMES[g_scheme].key;
    for (int pass = 0; pass < 3; ++pass) {  // 0 as designed, 1 all small, 2 shorter labels
        g_rgroups.clear();
        x = 6;
        std::vector<bool> used(NTOOL, false);
        auto add_group = [&](const std::string& title, const std::vector<std::pair<int, bool>>& items) {
            RGroupBox gb; gb.title = title;
            int gx = x, top = RIB_TAB_H + 4, bottom = RIB_TAB_H + RIB_BODY_H - 20;
            int col_x = x, row = 0, col_w = 0;
            auto flush = [&]() { if (row > 0) { col_x += col_w + 4; row = 0; col_w = 0; } };
            for (auto& it : items) {
                const ToolDef& td = TOOLS[it.first];
                std::string lab = td.label;
                if (pass == 2 && lab.size() > 11) lab = lab.substr(0, 10) + ".";
                if (it.second && pass == 0) {
                    flush();
                    std::string a, b; split2(lab, a, b);
                    int w = std::max(46, (int)std::max(text_w(g, a, 11.5f), text_w(g, b, 11.5f)) + 10);
                    gb.items.push_back({it.first, true, {col_x, top, col_x + w, bottom}});
                    col_x += w + 4;
                } else {
                    int w = 24 + (int)text_w(g, lab, 11.5f) + 6;
                    int y = top + 2 + row * 23;
                    gb.items.push_back({it.first, false, {col_x, y, col_x + w, y + 22}});
                    col_w = std::max(col_w, w);
                    for (auto& o : gb.items) if (!o.large && o.r.left == col_x) o.r.right = col_x + col_w;
                    if (++row == 3) flush();
                }
            }
            flush();
            int gw = std::max(col_x - gx, (int)text_w(g, title, 11.0f) + 12);
            gb.r = {gx, top - 2, gx + gw + 4, RIB_TAB_H + RIB_BODY_H};
            g_rgroups.push_back(gb);
            x = gx + gw + 14;
        };
        if (key == "apps") {  // the Applications tab: one large button per catalogue group
            int top = RIB_TAB_H + 4, bottom = RIB_TAB_H + RIB_BODY_H - 20;
            RGroupBox gb; gb.title = tostr((long)g_apps.size()) + " applications of AeroSynapse-Einstein";
            int gx = x;
            for (size_t gi = 0; gi < g_app_groups.size(); ++gi) {
                std::string a, b; split2(app_short(g_app_groups[gi]), a, b);
                int w = std::max(56, (int)std::max(text_w(g, a, 11.5f), text_w(g, b, 11.5f)) + 12);
                RItem it; it.tool = -1; it.large = true; it.r = {x, top, x + w, bottom}; it.app = (int)gi;
                gb.items.push_back(it);
                x += w + 2;
            }
            gb.r = {gx, top - 2, x, RIB_TAB_H + RIB_BODY_H};
            g_rgroups.push_back(gb);
            break;
        }
        for (int k = 0; k < NRGROUP; ++k) {
            if (key != RGROUPS[k].scheme) continue;
            std::vector<std::pair<int, bool>> items;
            std::string list = RGROUPS[k].icons;
            size_t p = 0;
            while (p <= list.size()) {
                size_t q = list.find(',', p); if (q == std::string::npos) q = list.size();
                std::string ic = list.substr(p, q - p);
                bool large = !ic.empty() && ic[0] == '*'; if (large) ic = ic.substr(1);
                int t = find_tool(key, ic);
                if (t >= 0) { items.push_back({t, large}); used[(size_t)t] = true; }
                p = q + 1;
            }
            if (!items.empty()) add_group(RGROUPS[k].title, items);
        }
        std::vector<std::pair<int, bool>> rest;  // tools of this workbench that no group names
        for (int i = 0; i < NTOOL; ++i) if (key == TOOLS[i].scheme && !used[(size_t)i]) rest.push_back({i, false});
        if (!rest.empty()) add_group("More", rest);
        if (x < W - RQUICK_W - 8) break;
    }
}

static void paint_ribbon(HWND h, HDC dc) {
    RECT rc; GetClientRect(h, &rc);
    int W = rc.right, H = rc.bottom;
    gp::Bitmap buf(std::max(1, W), std::max(1, H), PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    ribbon_layout(g, W);
    // tab row and body
    gp::SolidBrush tabbg(gcol(232, 234, 237)); g.FillRectangle(&tabbg, 0, 0, W, RIB_TAB_H);
    gp::LinearGradientBrush body(gp::Rect(0, RIB_TAB_H, W, RIB_BODY_H), gcol(252, 252, 253), gcol(240, 241, 243), gp::LinearGradientModeVertical);
    g.FillRectangle(&body, 0, RIB_TAB_H, W, H - RIB_TAB_H);
    gp::Pen rule(gcol(200, 203, 208), 1.0f);
    g.DrawLine(&rule, 0, RIB_TAB_H, W, RIB_TAB_H);
    g.DrawLine(&rule, 0, H - 1, W, H - 1);
    // File tab: the application menu
    {
        gp::RectF r((float)g_rfile_rect.left, (float)g_rfile_rect.top, (float)(g_rfile_rect.right - g_rfile_rect.left), (float)(g_rfile_rect.bottom - g_rfile_rect.top));
        gp::SolidBrush fb(g_rhover_tab == 0 ? gcol(38, 110, 190) : gcol(28, 92, 168)); g.FillRectangle(&fb, r);
        gtext(g, "File", r, gcol(255, 255, 255), 12.5f, gp::FontStyleRegular, gp::StringAlignmentCenter);
    }
    for (int k = 0; k < NSCHEME; ++k) {
        const RECT& t = g_rtabs[(size_t)k];
        gp::RectF r((float)t.left, (float)t.top, (float)(t.right - t.left), (float)(t.bottom - t.top));
        if (k == g_scheme) {
            gp::SolidBrush wb(gcol(252, 252, 253)); g.FillRectangle(&wb, r);
            g.DrawLine(&rule, r.X, r.Y, r.X, r.Y + r.Height); g.DrawLine(&rule, r.X + r.Width, r.Y, r.X + r.Width, r.Y + r.Height);
            g.DrawLine(&rule, r.X, r.Y, r.X + r.Width, r.Y);
            gp::LinearGradientBrush acc(gp::RectF(r.X, r.Y, r.Width, 3.0f), scheme_col(k), gshade(scheme_col(k), 0.45f), gp::LinearGradientModeHorizontal);
            g.FillRectangle(&acc, r.X + 1, r.Y, r.Width - 1, 3.0f);
            // a soft glow of the workbench's colour under the tab bar, fading out across the ribbon
            gp::LinearGradientBrush glow(gp::RectF(0, (float)RIB_TAB_H + RIB_BODY_H - 3, (float)W, 3.0f), gshade(scheme_col(k), 0.1f), gcol(255, 255, 255, 0), gp::LinearGradientModeHorizontal);
            g.FillRectangle(&glow, 0.0f, (float)RIB_TAB_H + RIB_BODY_H - 3, (float)W, 3.0f);
        } else if (k + 1 == g_rhover_tab) {
            gp::LinearGradientBrush hb(r, gcol(232, 238, 246), gmix(gcol(232, 238, 246), scheme_col(k), 0.18f), gp::LinearGradientModeVertical); g.FillRectangle(&hb, r);
        }
        gtext(g, SCHEMES[k].title, r, k == g_scheme ? gshade(scheme_col(k), -0.55f) : gcol(55, 60, 68), 12.5f, gp::FontStyleRegular, gp::StringAlignmentCenter);
    }
    // quick access (new, open, save, undo, redo, fit) and the panel switches, at the right end of the ribbon
    g.DrawLine(&rule, (float)(W - RQUICK_W), (float)(RIB_TAB_H + 6), (float)(W - RQUICK_W), (float)(RIB_TAB_H + RIB_BODY_H - 6));
    for (int q = 0; q < 6; ++q) {
        gp::RectF r((float)g_rquick[q].left, (float)g_rquick[q].top, 25, 25);
        if (q == g_rhover_quick) { gp::SolidBrush hb(gcol(220, 233, 248)); g.FillRectangle(&hb, r); gp::Pen pn(gcol(135, 175, 222), 1.0f); g.DrawRectangle(&pn, r); }
        if (gp::Bitmap* ic = icon_flat(RQUICK_ICON[q])) g.DrawImage(ic, r.X + 3, r.Y + 3, 19.0f, 19.0f);
    }
    for (int k = 0; k < 3; ++k) {
        gp::RectF r((float)g_rpanel[k].left, (float)g_rpanel[k].top, 25, 25);
        bool on = k == 0 ? g_show_tree : k == 1 ? g_show_props : false;
        if (k == g_hover_panel || on) { gp::SolidBrush hb(on ? gcol(206, 225, 246) : gcol(220, 233, 248)); g.FillRectangle(&hb, r); gp::Pen pn(gcol(135, 175, 222), 1.0f); g.DrawRectangle(&pn, r); }
        if (gp::Bitmap* ic = icon_flat(PANEL_ICONS[k])) g.DrawImage(ic, r.X + 3, r.Y + 3, 19.0f, 19.0f);
    }
    gtext(g, "Quick access", gp::RectF((float)(W - RQUICK_W), (float)(RIB_TAB_H + RIB_BODY_H - 19), (float)RQUICK_W, 17), gcol(80, 86, 96), 11.0f, gp::FontStyleRegular, gp::StringAlignmentCenter);
    // groups and their buttons
    int idx = 0;
    for (const RGroupBox& gb : g_rgroups) {
        for (const RItem& it : gb.items) {
            gp::RectF r((float)it.r.left, (float)it.r.top, (float)(it.r.right - it.r.left), (float)(it.r.bottom - it.r.top));
            if (it.app >= 0) {  // an Applications group: icon, two-line name and a drop-down mark
                if (idx == g_rhover_item) { gp::SolidBrush hb(gcol(220, 233, 248)); g.FillRectangle(&hb, r); gp::Pen pn(gcol(135, 175, 222), 1.0f); g.DrawRectangle(&pn, r.X, r.Y, r.Width - 1, r.Height - 1); }
                const std::string& gt = g_app_groups[(size_t)it.app];
                if (gp::Bitmap* ic = icon_flat(app_group_icon(gt))) g.DrawImage(ic, r.X + (r.Width - 32) / 2, r.Y + 3, 32.0f, 32.0f);
                std::string a, b; split2(app_short(gt), a, b);
                gtext(g, a, gp::RectF(r.X - 2, r.Y + 37, r.Width + 4, 15), gcol(30, 34, 40), 11.5f, gp::FontStyleRegular, gp::StringAlignmentCenter);
                gtext(g, b + " \xE2\x96\xBE", gp::RectF(r.X - 2, r.Y + 51, r.Width + 4, 15), gcol(30, 34, 40), 11.5f, gp::FontStyleRegular, gp::StringAlignmentCenter);
                ++idx;
                continue;
            }
            const ToolDef& td = TOOLS[it.tool];
            if (idx == g_rhover_item) {
                gp::SolidBrush hb(idx == g_rpress_item ? gcol(188, 213, 242) : gcol(220, 233, 248)); g.FillRectangle(&hb, r);
                gp::Pen pn(gcol(135, 175, 222), 1.0f); g.DrawRectangle(&pn, r.X, r.Y, r.Width - 1, r.Height - 1);
            }
            std::string lab = td.label;
            if (it.large) {
                if (gp::Bitmap* ic = icon_flat(td.icon)) g.DrawImage(ic, r.X + (r.Width - 32) / 2, r.Y + 3, 32.0f, 32.0f);
                std::string a, b; split2(lab, a, b);
                gtext(g, a, gp::RectF(r.X - 2, r.Y + 37, r.Width + 4, 15), gcol(30, 34, 40), 11.5f, gp::FontStyleRegular, gp::StringAlignmentCenter);
                if (!b.empty()) gtext(g, b, gp::RectF(r.X - 2, r.Y + 51, r.Width + 4, 15), gcol(30, 34, 40), 11.5f, gp::FontStyleRegular, gp::StringAlignmentCenter);
            } else {
                if (gp::Bitmap* ic = icon_flat(td.icon)) g.DrawImage(ic, r.X + 3, r.Y + 2, 18.0f, 18.0f);
                gtext(g, lab, gp::RectF(r.X + 24, r.Y, r.Width - 25, r.Height), gcol(30, 34, 40), 11.5f);
            }
            ++idx;
        }
        gp::RectF tr((float)gb.r.left, (float)(RIB_TAB_H + RIB_BODY_H - 19), (float)(gb.r.right - gb.r.left), 17);
        gtext(g, gb.title, tr, gcol(80, 86, 96), 11.0f, gp::FontStyleRegular, gp::StringAlignmentCenter);
        g.DrawLine(&rule, (float)gb.r.right + 4, (float)(RIB_TAB_H + 6), (float)gb.r.right + 4, (float)(RIB_TAB_H + RIB_BODY_H - 6));
    }
    gp::Graphics out(dc);
    out.DrawImage(&buf, 0, 0);
}

static int ribbon_item_at(int x, int y) {
    int idx = 0;
    for (const RGroupBox& gb : g_rgroups)
        for (const RItem& it : gb.items) { if (x >= it.r.left && x < it.r.right && y >= it.r.top && y < it.r.bottom) return idx; ++idx; }
    return -1;
}
static const RItem* ribbon_item(int k) {
    int idx = 0;
    for (const RGroupBox& gb : g_rgroups) for (const RItem& it : gb.items) { if (idx == k) return &it; ++idx; }
    return nullptr;
}
static bool in_rect(const RECT& r, int x, int y) { return x >= r.left && x < r.right && y >= r.top && y < r.bottom; }

static void ribbon_file_menu(HWND h) {
    if (!g_menu_all) return;
    POINT p = {g_rfile_rect.left, g_rfile_rect.bottom}; ClientToScreen(h, &p);
    TrackPopupMenu(g_menu_all, TPM_LEFTALIGN | TPM_TOPALIGN, p.x, p.y, 0, g_main, nullptr);
}

static LRESULT CALLBACK ribbon_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
        case WM_PAINT: { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); paint_ribbon(h, dc); EndPaint(h, &ps); return 0; }
        case WM_ERASEBKGND: return 1;
        case WM_MOUSEMOVE: {
            RECT rc; GetClientRect(h, &rc);
            int x = (short)LOWORD(l), y = (short)HIWORD(l);
            int tab = in_rect(g_rfile_rect, x, y) ? 0 : -1;
            for (size_t k = 0; k < g_rtabs.size(); ++k) if (in_rect(g_rtabs[k], x, y)) tab = (int)k + 1;
            int item = y > RIB_TAB_H && x < rc.right - RQUICK_W ? ribbon_item_at(x, y) : -1;
            int quick = -1; for (int q = 0; q < 6; ++q) if (in_rect(g_rquick[q], x, y)) quick = q;
            int ph = -1; for (int k = 0; k < 3; ++k) if (in_rect(g_rpanel[k], x, y)) ph = k;
            bool changed = tab != g_rhover_tab || item != g_rhover_item || quick != g_rhover_quick || ph != g_hover_panel;
            g_rhover_tab = tab; g_rhover_quick = quick; g_hover_panel = ph;
            if (item != g_rhover_item) {
                g_rhover_item = item;
                if (const RItem* it = ribbon_item(item)) {
                    if (it->app >= 0) { status(g_app_groups[(size_t)it->app] + ": " + tostr((long)app_group_count(g_app_groups[(size_t)it->app])) + " entries (click to list them)"); show_tip(-1, 0, 0); }
                    else {
                    const ToolDef& td = TOOLS[it->tool]; status(std::string(td.label) + ":  " + td.tip);
                    POINT p = {it->r.left, RIB_TAB_H + RIB_BODY_H + 4}; ClientToScreen(h, &p);
                    show_tip(it->tool, p.x, p.y);
                    }
                } else show_tip(-1, 0, 0);
            }
            if (quick >= 0) status(RQUICK_TIP[quick]);
            if (ph >= 0) status(PANEL_TIPS[ph]);
            if (tab > 0) status(std::string("Workbench: ") + SCHEMES[tab - 1].title);
            if (changed) InvalidateRect(h, nullptr, FALSE);
            TRACKMOUSEEVENT te = {sizeof te, TME_LEAVE, h, 0}; TrackMouseEvent(&te);
            return 0;
        }
        case WM_MOUSELEAVE: g_rhover_tab = g_rhover_item = g_rpress_item = g_rhover_quick = g_hover_panel = -1; show_tip(-1, 0, 0); InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_LBUTTONDOWN: {
            SendMessageA(h, WM_MOUSEMOVE, w, l);  // hit-test where the click is (the hover state may be stale)
            if (g_rhover_tab == 0) { ribbon_file_menu(h); return 0; }
            if (g_rhover_tab > 0) {
                g_scheme = g_rhover_tab - 1; g_rhover_item = -1;
                InvalidateRect(h, nullptr, FALSE);
                status(std::string(SCHEMES[g_scheme].title) + ": " + (std::string(SCHEMES[g_scheme].key) == "apps" ? tostr((long)g_apps.size()) + " applications in " + tostr((long)g_app_groups.size()) + " groups (click a group)" : tostr((long)scheme_tools(g_scheme).size()) + " tools"));
                return 0;
            }
            if (g_rhover_quick >= 0) { PostMessageA(g_main, WM_COMMAND, RQUICK_CMD[g_rhover_quick], 0); return 0; }
            if (g_hover_panel >= 0) { toggle_panel(g_hover_panel + 1); return 0; }
            g_rpress_item = g_rhover_item; SetCapture(h); InvalidateRect(h, nullptr, FALSE);
            return 0;
        }
        case WM_LBUTTONUP: {
            ReleaseCapture();
            int item = ribbon_item_at((short)LOWORD(l), (short)HIWORD(l));
            if (item >= 0 && item == g_rpress_item) if (const RItem* it = ribbon_item(item)) {
                show_tip(-1, 0, 0);
                if (it->app >= 0) { POINT p = {it->r.left, it->r.bottom + 20}; ClientToScreen(h, &p); g_rpress_item = -1; app_menu(it->app, p.x, p.y); }
                else run_tool(it->tool);
            }
            g_rpress_item = -1; InvalidateRect(h, nullptr, FALSE);
            return 0;
        }
        case WM_MOUSEWHEEL:  // the wheel walks through the workbench tabs, as in Office and Creo ribbons
            g_scheme = (g_scheme + ((short)HIWORD(w) < 0 ? 1 : NSCHEME - 1)) % NSCHEME;
            InvalidateRect(h, nullptr, FALSE);
            return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

// every menu of the classic menu bar, as one popup under the File tab
static void ribbon_set_menu(HMENU bar) {
    g_menu_all = CreatePopupMenu();
    int n = GetMenuItemCount(bar);
    for (int i = 0; i < n; ++i) {
        char t[128] = ""; GetMenuStringA(bar, (UINT)i, t, sizeof t, MF_BYPOSITION);
        HMENU sub = GetSubMenu(bar, i);
        if (sub) AppendMenuA(g_menu_all, MF_POPUP, (UINT_PTR)sub, t);
    }
}
