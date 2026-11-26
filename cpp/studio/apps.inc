// The Applications tab: every runnable part of AeroSynapse-Einstein (assets/app_catalog.json, written by
// scripts/make_app_catalog.py from the command line, the research scripts, the thermal PINN experiments and the
// local web apps). One ribbon button per group; a click lists the group's entries; picking one opens it in the task
// window with its command and description shown first (nothing runs until Run is pressed), then the live log.

struct AppEntry { std::string group, label, args, icon, about; bool web = false; };
static std::vector<AppEntry> g_apps;
static std::vector<std::string> g_app_groups;

static void apps_load() {
    g_apps.clear(); g_app_groups.clear();
    try {
        js::Value c = js::parse(js::read_file(g_root + "\\assets\\app_catalog.json"));
        const js::Value* gs = c.find("groups");
        for (size_t i = 0; gs && i < gs->size(); ++i) {
            const js::Value& g = gs->at(i);
            std::string title = g.str("title", "");
            g_app_groups.push_back(title);
            const js::Value* it = g.find("items");
            for (size_t k = 0; it && k < it->size(); ++k) {
                const js::Value& e = it->at(k);
                AppEntry a; a.group = title; a.label = e.str("label", ""); a.args = e.str("args", ""); a.icon = e.str("icon", "opt_centre");
                a.about = e.str("about", ""); a.web = e.flag("web", false);
                g_apps.push_back(a);
            }
        }
    } catch (...) {}
}
static std::string app_group_icon(const std::string& g) {
    for (const AppEntry& a : g_apps) if (a.group == g) return a.icon;
    return "opt_centre";
}
static int app_group_count(const std::string& g) { int n = 0; for (const AppEntry& a : g_apps) if (a.group == g) ++n; return n; }

static void app_menu(int group, int sx, int sy) {
    if (group < 0 || group >= (int)g_app_groups.size()) return;
    HMENU m = CreatePopupMenu();
    for (size_t i = 0; i < g_apps.size(); ++i)
        if (g_apps[i].group == g_app_groups[(size_t)group]) AppendMenuA(m, MF_STRING, ID_APP0 + (UINT)i, g_apps[i].label.c_str());
    TrackPopupMenu(m, TPM_LEFTALIGN | TPM_TOPALIGN, sx, sy, 0, g_main, nullptr);
    DestroyMenu(m);
}
static void jobs_show_custom(const std::string& args, const std::string& about);
static void app_open(int i) {
    if (i < 0 || i >= (int)g_apps.size()) return;
    const AppEntry& a = g_apps[(size_t)i];
    std::string about = a.group + ": " + a.label + (a.about.empty() ? "" : "\r\n\r\n" + a.about);
    if (a.args.find("{design}") != std::string::npos) about += "\r\n\r\nUses the design open in Studio.";
    { std::string app = labs_app_of(a.args); if (!app.empty()) { open_labs(app, ""); status(a.label + ": inside Studio (Labs window)"); return; } }
    jobs_show_custom(a.args, about);
    status(a.label + ": check the command and press Run");
}
