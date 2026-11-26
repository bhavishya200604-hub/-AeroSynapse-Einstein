// Self-test: `AeroSynapseStudio --selftest report.json design.json` runs every tool of every workbench on the design,
// one after another, and writes what each one did: whether the design changed, whether it rebuilt without error, the
// windows it opened, the messages it showed (recorded, not shown), the background jobs it would start (their
// commands are recorded and a placeholder process stands in), and whether it left the design as it found it after
// Cancel. Nothing is saved and no file dialog opens.

struct StRec { int tool; std::string scheme, label, kind; bool changed = false; int nf0 = 0, nf1 = 0; double v0 = 0, v1 = 0; std::string error; std::vector<std::string> log, windows; };
static std::string g_st_out, g_st_base, g_st_last_error;
static std::vector<StRec> g_st_recs;
static int g_st_i = -1, g_st_phase = 0;
static DWORD g_st_t0 = 0;
static std::vector<int> g_st_order;

static std::vector<std::string> st_windows() {
    std::vector<std::string> w;
    struct { HWND h; const char* n; } ws[] = {{g_pad, "pad"}, {g_setup, "setup"}, {g_prefs, "preferences"}, {g_sk, "sketcher"}, {g_wf, "workflow"}, {g_jobs, "jobs"},
                                             {g_start, "start page"}, {g_sv, "results"}, {g_par, "skeleton"}};
    for (auto& x : ws) if (x.h && IsWindowVisible(x.h)) w.push_back(x.n);
    return w;
}
static void st_cleanup() {
    if (g_pad && IsWindowVisible(g_pad)) pad_close(false);
    HWND hs[] = {g_setup, g_prefs, g_sk, g_wf, g_jobs, g_start, g_sv, g_par};
    for (HWND h : hs) if (h) ShowWindow(h, SW_HIDE);
    if (g_job.hProcess) { TerminateProcess(g_job.hProcess, 0); CloseHandle(g_job.hProcess); CloseHandle(g_job.hThread); memset(&g_job, 0, sizeof g_job); KillTimer(g_main, 1); }
    if (g_wf_proc) { TerminateProcess(g_wf_proc, 0); CloseHandle(g_wf_proc); g_wf_proc = nullptr; }
    if (g_kin_on) kin_toggle();
    g_live = false; KillTimer(g_main, 8);
    g_view_mode = 0; g_annot = false; g_cut = false; g_wire = false; g_pick_mode = 0;
    g_colors.clear(); g_result_label.clear(); viz_clear(); g_field_on = g_particles_on = false; KillTimer(g_main, 7);
    g_sl_on = g_sweep_on = g_rec_on = false; KillTimer(g_main, 41);  // flow animations, the assembly's motion timer
    g_show_palette = g_show_tree = g_show_props = true; palette_target(); layout();
}
static void st_write() {
    js::Value out = js::Value::array();
    for (auto& r : g_st_recs) {
        js::Value o = js::Value::object();
        o["tool"] = r.tool; o["scheme"] = r.scheme; o["label"] = r.label; o["kind"] = r.kind; o["changed"] = r.changed;
        o["features_before"] = r.nf0; o["features_after"] = r.nf1; o["volume_before"] = r.v0; o["volume_after"] = r.v1; o["build_error"] = r.error;
        js::Value l = js::Value::array(); for (auto& s : r.log) l.push(s); o["log"] = l;
        js::Value w = js::Value::array(); for (auto& s : r.windows) w.push(s); o["windows"] = w;
        out.push(o);
    }
    js::write_file(g_st_out, js::dump(out, 1));
}
static void selftest_tick() {  // timer 13, every 50 ms
    if (g_building || g_dirty_build) { if (GetTickCount() - g_st_t0 < 60000) return; }
    if (g_st_i < 0) {  // the design is built: remember it and list every tool
        g_st_base = js::dump(g_doc, 0);
        for (int k = 0; k < NSCHEME; ++k) for (int t : scheme_tools(k)) g_st_order.push_back(t);
        g_st_i = 0; g_st_phase = 0; return;
    }
    if (g_st_i >= (int)g_st_order.size()) { KillTimer(g_main, 13); st_write(); g_selftest = false; g_modified = false; DestroyWindow(g_main); return; }
    int ti = g_st_order[(size_t)g_st_i];
    const ToolDef& td = TOOLS[ti];
    if (g_st_phase == 0) {  // run the tool on the design with the wing selected
        g_doc = js::parse(g_st_base); g_undo.clear(); g_redo.clear();
        g_sel = 0;
        for (size_t i = 0; i < features().size(); ++i) if (features().at(i).str("id", "") == "wing") g_sel = (int)i;
        for (int k = 0; k < NSCHEME; ++k) if (std::string(td.scheme) == SCHEMES[k].key) g_scheme = k;
        fill_list(); fill_props();
        StRec r; r.tool = ti; r.scheme = td.scheme; r.label = td.label; r.kind = td.kind ? td.kind : ""; r.nf0 = (int)features().size(); r.v0 = g_props.volume;
        g_st_recs.push_back(r);
        g_st_log.clear(); g_st_last_error.clear();
        if (td.kind) SendMessageA(g_main, WM_TOOL_INSERT, (WPARAM)ti, 0);
        else if (td.cmd) SendMessageA(g_main, WM_COMMAND, (WPARAM)td.cmd, 0);
        g_st_t0 = GetTickCount(); g_st_phase = 1; return;
    }
    if (g_st_phase == 1) {  // let the pad preview and the rebuild finish
        if (GetTickCount() - g_st_t0 < 700) return;
        if ((g_building || g_dirty_build) && GetTickCount() - g_st_t0 < 60000) return;
        StRec& r = g_st_recs.back();
        r.changed = js::dump(g_doc, 0) != g_st_base; r.nf1 = (int)features().size(); r.v1 = g_props.volume; r.error = g_st_last_error;
        r.windows = st_windows();
        r.log = g_st_log;
        st_cleanup();
        g_st_phase = 2; g_st_t0 = GetTickCount(); return;
    }
    if (g_st_phase == 2) {  // after Cancel the design must be the original again
        if (GetTickCount() - g_st_t0 < 100) return;
        StRec& r = g_st_recs.back();
        if (js::dump(g_doc, 0) != g_st_base && !r.windows.empty() && std::find(r.windows.begin(), r.windows.end(), "pad") != r.windows.end())
            r.log.push_back("CANCEL did not restore the design");
        g_st_log.clear();
        ++g_st_i; g_st_phase = 0;
        if (g_st_i % 10 == 0) st_write();
    }
}
