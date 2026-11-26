// The environment a part or an assembly works in, and what it does there: the medium (air at an altitude, fresh or
// sea water, engine oil, vacuum), its temperature, a current, the liquid's surface, gravity and lubrication; then for
// every part its weight and buoyancy (and the floating waterline), hydrostatic pressure, drag, heat transfer by
// convection and radiation (heat flow, cooling time, steady temperature under a heat load), thermal expansion and
// the fits it changes, the friction of every bearing mate, the torque and power each motor needs (bearing friction,
// fluid drag on turning parts, gear losses), and corrosion and temperature warnings.
//
// Every number is an engineering estimate, and the report says which correlation gave it: submerged volume and centre
// of buoyancy exact on the mesh (divergence theorem on the triangles clipped at the surface); convection from sphere
// correlations on the part's equivalent diameter 6 V / A (Whitaker 1972 forced, Churchill 1983 natural, mixed by
// Churchill's cube rule); grey-body radiation; drag from the sphere law of Clift and Gauvin (1971) on the projected
// area; turning parts as discs (von Karman: Cm = 3.87 Re^-1/2 laminar, 0.146 Re^-1/5 turbulent); galvanic couples
// by anodic index (MIL-STD-889: 0.15 V in sea water, 0.25 V in fresh water).

struct Fluid {
    std::string name;
    double rho = 0, mu = 1, k = 0, cp = 1, beta = 0, Pr = 1, p = 101325, Tfreeze = -1e9, Tboil = 1e9;
    bool gas = false, liquid = false, vacuum = false;
};
enum { MED_AIR, MED_WATER, MED_SEA, MED_OIL, MED_VACUUM, MED_HYD, MED_GEAROIL, NMED };
static const char* MED_NAME[NMED] = {"Air", "Fresh water", "Sea water", "Engine oil SAE 30", "Vacuum", "Hydraulic oil ISO VG 32", "Gear oil ISO VG 220"};
static bool is_oil(int med) { return med == MED_OIL || med == MED_HYD || med == MED_GEAROIL; }
static const char* FLOW_DIR[6] = {"+x", "-x", "+y", "-y", "+z (up)", "-z (down)"};
static const char* LUBE_NAME[3] = {"dry", "greased", "oil-lubricated"};

static Fluid fluid_props(int med, double Tc, double alt) {
    Fluid f; f.name = MED_NAME[med];
    double T = Tc + 273.15;
    if (med == MED_AIR) {  // ideal gas at the ISA pressure of the altitude, the given temperature
        double h = std::max(0.0, std::min(alt, 20000.0)), Ti = h < 11000 ? 288.15 - 0.0065 * h : 216.65;
        f.p = h < 11000 ? 101325 * std::pow(Ti / 288.15, 5.2559) : 22632 * std::exp(-(h - 11000) / 6341.6);
        f.rho = f.p / (287.05 * T); f.mu = 1.458e-6 * std::pow(T, 1.5) / (T + 110.4); f.k = 0.0241 * std::pow(T / 273.15, 0.85);
        f.cp = 1006; f.beta = 1 / T; f.gas = true;
    } else if (med == MED_WATER || med == MED_SEA) {  // Jones-Harris density, Vogel viscosity; sea water 35 g/kg
        double t = Tc;
        f.rho = 1000 * (1 - (t + 288.9414) / (508929.2 * (t + 68.12963)) * (t - 3.9863) * (t - 3.9863));
        f.mu = 2.414e-5 * std::pow(10.0, 247.8 / (T - 140));
        f.k = 0.5706 + 0.00175 * t - 6.46e-6 * t * t; f.cp = 4182; f.beta = std::max(1e-6, 1e-6 * (-68 + 17.2 * t - 0.1 * t * t));
        f.Tfreeze = 0; f.Tboil = 100;
        if (med == MED_SEA) { f.rho += 26.5; f.mu *= 1.08; f.k *= 0.99; f.cp = 3993; f.beta *= 1.1; f.Tfreeze = -1.9; f.Tboil = 100.6; }
        f.liquid = true;
    } else if (med == MED_OIL) {  // SAE 30: 0.1 Pa s at 40 C, 0.011 Pa s at 100 C (Vogel fit)
        f.rho = 890 - 0.6 * (Tc - 15); f.mu = 2.72e-5 * std::exp(1339.0 / (T - 150)); f.k = 0.145; f.cp = 1900; f.beta = 7e-4;
        f.Tfreeze = -25; f.Tboil = 230; f.liquid = true;
    } else if (med == MED_HYD) {  // ISO VG 32: 32 mm2/s at 40 C, 5.4 mm2/s at 100 C (Vogel fit)
        f.rho = 870 - 0.62 * (Tc - 15); f.mu = 3.28e-5 * std::exp(1098.0 / (T - 150)); f.k = 0.135; f.cp = 1880; f.beta = 7.5e-4;
        f.Tfreeze = -30; f.Tboil = 200; f.liquid = true;
    } else if (med == MED_GEAROIL) {  // ISO VG 220: 220 mm2/s at 40 C, 19 mm2/s at 100 C (Vogel fit)
        f.rho = 895 - 0.6 * (Tc - 15); f.mu = 1.85e-5 * std::exp(1510.0 / (T - 150)); f.k = 0.14; f.cp = 1950; f.beta = 6.8e-4;
        f.Tfreeze = -15; f.Tboil = 250; f.liquid = true;
    } else { f.vacuum = true; f.p = 0; f.rho = 0; }
    f.Pr = f.vacuum ? 1 : f.mu * f.cp / std::max(f.k, 1e-9);
    return f;
}

struct Env {
    bool on = false;
    int medium = MED_AIR, dir = 0, lube = 0;
    double Tc = 15, U = 0, alt = 0, surface = 1e30, Tpart = 60, heat = 0, g = 9.81;
    bool find_float = false;
};
static Env g_env;
struct EBody {
    std::string name, matkey;
    const EnvMat* M = nullptr;
    int comp = -1;
    double V = 0, A = 0, Vsub = 0, Asub = 0, Aproj = 0, zmin = 0, zmax = 0, Lmax = 0, mass = 0, W = 0, B = 0;
    V3 c, cb;
    double Lc = 0, h_liq = 0, h_gas = 0, h_rad = 0, Q = 0, tau = 0, Bi = 0, Ts = 0, Re = 0, Cd = 0, drag = 0, dL = 0;
};
static std::vector<EBody> g_eb;
static std::string g_env_summary;
static bool g_env_floats = false;  // the last analysis found a floating waterline

// ---------------------------------------------------------------- a small modal form
struct FField { std::string label; std::vector<std::string> options; std::string value; HWND h = nullptr; };
static bool g_form_done = false, g_form_ok = false;
static LRESULT CALLBACK form_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_COMMAND && (LOWORD(w) == IDOK || LOWORD(w) == IDCANCEL)) { g_form_ok = LOWORD(w) == IDOK; g_form_done = true; return 0; }
    if (m == WM_CLOSE) { g_form_ok = false; g_form_done = true; return 0; }
    if (m == WM_CTLCOLORSTATIC) { HDC dc = (HDC)w; SetBkMode(dc, TRANSPARENT); return (LRESULT)GetSysColorBrush(COLOR_WINDOW); }
    return DefWindowProcA(h, m, w, l);
}
static bool form_dialog(const char* title, const std::string& intro, std::vector<FField>& f) {
    if (g_selftest) { g_st_log.push_back(std::string("DIALOG form: ") + title + " (defaults taken)"); return true; }
    static bool reg = false;
    HINSTANCE hi = GetModuleHandleA(nullptr);
    if (!reg) {
        WNDCLASSA c; memset(&c, 0, sizeof c);
        c.lpfnWndProc = form_proc; c.hInstance = hi; c.hCursor = LoadCursor(nullptr, IDC_ARROW); c.hbrBackground = GetSysColorBrush(COLOR_WINDOW); c.lpszClassName = "AeroSynapseForm";
        RegisterClassA(&c); reg = true;
    }
    int n = (int)f.size(), H = 96 + 32 * n + 56, Wd = 560;
    RECT mr; GetWindowRect(g_main, &mr);
    HWND d = CreateWindowExA(WS_EX_DLGMODALFRAME, "AeroSynapseForm", title, WS_POPUP | WS_CAPTION | WS_SYSMENU, mr.left + (mr.right - mr.left - Wd) / 2, mr.top + 90, Wd, H, g_main, nullptr, hi, nullptr);
    HWND it = CreateWindowExA(0, "STATIC", intro.c_str(), WS_CHILD | WS_VISIBLE, 16, 12, Wd - 40, 52, d, nullptr, hi, nullptr);
    SendMessageA(it, WM_SETFONT, (WPARAM)g_font, TRUE);
    for (int i = 0; i < n; ++i) {
        int y = 72 + 32 * i;
        HWND lb = CreateWindowExA(0, "STATIC", f[(size_t)i].label.c_str(), WS_CHILD | WS_VISIBLE, 16, y + 4, 270, 22, d, nullptr, hi, nullptr);
        SendMessageA(lb, WM_SETFONT, (WPARAM)g_font, TRUE);
        if (f[(size_t)i].options.empty()) f[(size_t)i].h = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", f[(size_t)i].value.c_str(), WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL, 290, y, 240, 24, d, nullptr, hi, nullptr);
        else {
            f[(size_t)i].h = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | WS_TABSTOP | CBS_DROPDOWNLIST | WS_VSCROLL, 290, y, 240, 300, d, nullptr, hi, nullptr);
            int sel = 0;
            for (size_t k = 0; k < f[(size_t)i].options.size(); ++k) { SendMessageA(f[(size_t)i].h, CB_ADDSTRING, 0, (LPARAM)f[(size_t)i].options[k].c_str()); if (f[(size_t)i].options[k] == f[(size_t)i].value) sel = (int)k; }
            SendMessageA(f[(size_t)i].h, CB_SETCURSEL, sel, 0);
        }
        SendMessageA(f[(size_t)i].h, WM_SETFONT, (WPARAM)g_font, TRUE);
    }
    HWND ok = CreateWindowExA(0, "BUTTON", "Analyse", WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON, Wd - 230, H - 82, 100, 30, d, (HMENU)(INT_PTR)IDOK, hi, nullptr);
    HWND ca = CreateWindowExA(0, "BUTTON", "Cancel", WS_CHILD | WS_VISIBLE | WS_TABSTOP, Wd - 122, H - 82, 90, 30, d, (HMENU)(INT_PTR)IDCANCEL, hi, nullptr);
    SendMessageA(ok, WM_SETFONT, (WPARAM)g_font, TRUE); SendMessageA(ca, WM_SETFONT, (WPARAM)g_font, TRUE);
    EnableWindow(g_main, FALSE);
    ShowWindow(d, SW_SHOW);
    g_form_done = false; g_form_ok = false;
    MSG msg;
    while (!g_form_done && GetMessageA(&msg, nullptr, 0, 0)) {
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_RETURN) { g_form_ok = true; g_form_done = true; continue; }
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_ESCAPE) { g_form_done = true; continue; }
        if (!IsDialogMessageA(d, &msg)) { TranslateMessage(&msg); DispatchMessageA(&msg); }
    }
    if (g_form_ok)
        for (auto& x : f) {
            if (x.options.empty()) { char b[256]; GetWindowTextA(x.h, b, sizeof b); x.value = b; }
            else { int s = (int)SendMessageA(x.h, CB_GETCURSEL, 0, 0); if (s >= 0 && s < (int)x.options.size()) x.value = x.options[(size_t)s]; }
        }
    EnableWindow(g_main, TRUE);
    DestroyWindow(d);
    SetForegroundWindow(g_main);
    return g_form_ok;
}

// ---------------------------------------------------------------- the parts as bodies
static std::string body_material(const js::Value& part, const std::string& over) {
    if (!over.empty() && envmat_of(over)) return over;
    std::string m = part.str("material", "");
    if (envmat_of(m)) return m;
    if (const js::Value* fl = part.find("features"))
        for (size_t i = 0; i < fl->size(); ++i) { std::string mm = fl->at(i).str("material", ""); if (envmat_of(mm)) return mm; }
    if (part.find("fastener") || part.find("gear")) return "steel_mild";
    return "";  // none named: the caller's default
}
// geometry of a triangle soup in world coordinates, clipped below the liquid surface zs (divergence theorem: the cap on
// the surface carries no flux of F = (0, 0, z - zs) and its first moments)
struct GeoSum { double V = 0, Vx = 0, Vy = 0, Vz = 0, A = 0, Aproj = 0; };
static void tri_clip_add(V3 p0, V3 p1, V3 p2, double zs, const V3& u, GeoSum& s) {
    V3 in[3] = {p0, p1, p2}, out[4]; int n = 0;
    for (int i = 0; i < 3; ++i) {  // Sutherland-Hodgman against z <= zs
        const V3 &a = in[i], &b = in[(i + 1) % 3];
        bool ia = a.z <= zs, ib = b.z <= zs;
        if (ia) out[n++] = a;
        if (ia != ib) { double t = (zs - a.z) / (b.z - a.z); out[n++] = a + (b - a) * t; }
    }
    for (int k = 1; k + 1 < n; ++k) {
        const V3 &a = out[0], &b = out[k], &c = out[k + 1];
        V3 N = geo::cross(b - a, c - a) * 0.5;
        double Az = N.z;
        V3 m[3] = {(a + b) * 0.5, (b + c) * 0.5, (c + a) * 0.5};  // edge midpoints: exact for quadratics
        double fz = 0, fx = 0, fy = 0, fzz = 0, zr = zs < 1e29 ? zs : 0.0;  // no surface: any reference height gives the same volume
        for (auto& q : m) { double dz = q.z - zr; fz += dz; fx += q.x * dz; fy += q.y * dz; fzz += 0.5 * (q.z * q.z - zr * zr); }
        s.V += Az * fz / 3; s.Vx += Az * fx / 3; s.Vy += Az * fy / 3; s.Vz += Az * fzz / 3;
        s.A += geo::len(N); s.Aproj += 0.5 * std::fabs(geo::dot(N, u));
    }
}
static void body_geometry(const geo::Mesh& M, const Pose& P, double zs, const V3& u, GeoSum& s) {
    for (size_t q = 0; q + 2 < M.t.size(); q += 3)
        tri_clip_add(P.apply(M.v[(size_t)M.t[q]]), P.apply(M.v[(size_t)M.t[q + 1]]), P.apply(M.v[(size_t)M.t[q + 2]]), zs, u, s);
}
static V3 flow_dir() { static const V3 D[6] = {V3(1, 0, 0), V3(-1, 0, 0), V3(0, 1, 0), V3(0, -1, 0), V3(0, 0, 1), V3(0, 0, -1)}; return D[std::max(0, std::min(5, g_env.dir))]; }

// the parts to analyse: an assembly's components, or the open design as one body
struct BodyRef { std::string name; js::Value part; std::string mat; std::vector<std::pair<const geo::Mesh*, Pose>> meshes; int comp = -1; };
static std::vector<BodyRef> env_bodies() {
    std::vector<BodyRef> out;
    if (g_asm_on) {
        for (size_t c = 0; c < g_ac.size(); ++c) {
            const AComp& C = g_ac[c];
            if (C.hidden || !comp_ready(C)) continue;
            BodyRef b; b.name = C.id; b.part = C.part; b.comp = (int)c; b.mat = body_material(C.part, C.material);
            if (b.mat.empty()) b.mat = "al6061";
            for (const APiece& pc : C.pieces) { Pose P; mat_mul(C.pose.R, pc.local.R, P.R); P.t = C.pose.apply(pc.local.t); b.meshes.push_back({&pc.mesh->m, P}); }
            out.push_back(b);
        }
    } else if (!g_mesh.t.empty()) {
        static geo::Mesh mesh; static std::string meshed;
        std::string key = js::dump(g_doc, 0);
        if (key != meshed) {
            mesh = geo::Mesh();
            try {
                geo::Model m; m.load(g_doc);
                if (m.bounds.valid()) {
                    V3 e = m.bounds.hi - m.bounds.lo;
                    double h = std::max(geo::len(e) / 160.0, std::cbrt((e.x + 1e-3) * (e.y + 1e-3) * (e.z + 1e-3) / 4e6));
                    geo::Grid gr = geo::sample(m, h);
                    mesh = geo::surface_nets(m, gr, 1);
                }
            } catch (...) {}
            meshed = key;
        }
        BodyRef b; b.name = g_doc.str("name", "design"); b.part = g_doc; b.mat = body_material(g_doc, "");
        if (b.mat.empty()) b.mat = "al7075";  // an aircraft structure unless a material says otherwise
        b.meshes.push_back({mesh.t.empty() ? &g_mesh : &mesh, Pose()});
        out.push_back(b);
    }
    return out;
}

// ---------------------------------------------------------------- heat transfer and drag correlations
static double nu_forced_sphere(double Re, double Pr) { return 2 + (0.4 * std::sqrt(Re) + 0.06 * std::pow(Re, 2.0 / 3)) * std::pow(Pr, 0.4); }
static double nu_natural_sphere(double Ra, double Pr) { return 2 + 0.589 * std::pow(std::max(Ra, 0.0), 0.25) / std::pow(1 + std::pow(0.469 / Pr, 9.0 / 16), 4.0 / 9); }
static double cd_sphere(double Re) { if (Re < 1e-9) return 0; return 24 / Re * (1 + 0.15 * std::pow(Re, 0.687)) + 0.42 / (1 + 42500 * std::pow(Re, -1.16)); }
static double h_conv(const Fluid& f, double L, double U, double dT, double g) {  // W/m2K
    if (f.vacuum || f.rho <= 0) return 0;
    double nu = f.mu / f.rho, al = f.k / (f.rho * f.cp);
    double Re = U * L / nu, Ra = g * f.beta * std::fabs(dT) * L * L * L / (nu * al);
    double Nf = U > 0 ? nu_forced_sphere(Re, f.Pr) : 0, Nn = nu_natural_sphere(Ra, f.Pr);
    double Nu = std::cbrt(Nf * Nf * Nf + Nn * Nn * Nn);
    return Nu * f.k / L;
}
// Friction of material pairs: static and kinetic, dry and lubricated (handbook values: Marks' Standard Handbook for
// Mechanical Engineers and the CRC Handbook as compiled in the usual engineering tables; -1 where no lubricated value
// is listed, then 0.12 x dry with a floor of 0.04). Materials are grouped into families first.
static std::string fric_family(const EnvMat* m) {
    if (!m) return "";
    std::string k = m->key;
    if (k.rfind("al", 0) == 0) return "al";
    if (k == "steel_mild" || k == "steel_300m") return "steel";
    if (k.rfind("ti", 0) == 0) return "ti";
    if (k.rfind("cfrp", 0) == 0) return "cfrp";
    return k;
}
struct FricPair { const char *a, *b; double ms, mk, lms, lmk; const char* src; };
static const FricPair FRIC_PAIRS[] = {
    {"steel", "steel", 0.74, 0.57, 0.11, 0.08, "handbook"},
    {"steel", "stainless_316", 0.74, 0.57, 0.11, 0.08, "handbook (steel on steel)"},
    {"stainless_316", "stainless_316", 0.74, 0.57, 0.11, 0.08, "handbook (steel on steel; stainless galls more)"},
    {"steel", "al", 0.61, 0.47, -1, -1, "handbook"},
    {"steel", "brass", 0.51, 0.44, -1, -1, "handbook"},
    {"steel", "copper", 0.53, 0.36, 0.18, -1, "handbook"},
    {"steel", "cast_iron", 0.40, 0.23, 0.21, 0.133, "handbook"},
    {"cast_iron", "cast_iron", 1.10, 0.15, -1, 0.070, "handbook"},
    {"steel", "bronze", 0.35, 0.30, 0.16, 0.08, "handbook (phosphor bronze on steel)"},
    {"steel", "ptfe", 0.04, 0.04, 0.04, 0.04, "handbook"},
    {"ptfe", "ptfe", 0.04, 0.04, 0.04, 0.04, "handbook"},
    {"steel", "nylon", 0.40, 0.35, 0.10, 0.08, "estimate"},
    {"nylon", "nylon", 0.25, 0.20, 0.10, 0.08, "handbook (0.15 to 0.25)"},
    {"al", "al", 1.05, 1.05, 0.30, 0.30, "handbook (galls when dry)"},
    {"copper", "copper", 1.00, 0.80, 0.08, 0.08, "handbook (static); kinetic estimate"},
    {"ti", "ti", 0.36, 0.30, -1, -1, "handbook (Ti-6Al-4V)"},
    {"oak", "oak", 0.62, 0.48, -1, -1, "handbook (along the grain)"},
    {"glass_soda", "glass_soda", 0.94, 0.40, 0.35, 0.10, "handbook"},
    {"steel", "glass_soda", 0.60, 0.50, -1, -1, "estimate"},
    {"rubber", "steel", 0.80, 0.70, -1, -1, "estimate (rubber on metal 0.6 to 1)"},
    {"rubber", "al", 0.80, 0.70, -1, -1, "estimate (rubber on metal 0.6 to 1)"},
    {"rubber", "cast_iron", 0.80, 0.70, -1, -1, "estimate (rubber on metal 0.6 to 1)"},
};
// static and kinetic friction of a pair at the given lubrication and medium; src says where the numbers came from
static void friction_pair(const EnvMat* a, const EnvMat* b, int lube, int med, double& ms, double& mk, std::string& src) {
    std::string fa = fric_family(a), fb = fric_family(b);
    const FricPair* hit = nullptr;
    for (const FricPair& p : FRIC_PAIRS) if ((fa == p.a && fb == p.b) || (fa == p.b && fb == p.a)) hit = &p;
    bool oily = is_oil(med) || lube == 2, greased = lube == 1;
    if (hit) {
        src = hit->src;
        if (oily || greased) {
            ms = hit->lms > 0 ? hit->lms : std::max(0.04, 0.12 * hit->ms);
            mk = hit->lmk > 0 ? hit->lmk : std::max(0.04, 0.12 * hit->mk);
            if (greased) { ms *= 1.3; mk *= 1.3; }  // grease: a little above an oil film at low speed
            if (hit->lmk <= 0) src += "; lubricated value estimated";
        } else { ms = hit->ms; mk = hit->mk; }
        if (!oily && !greased && (med == MED_WATER || med == MED_SEA)) { bool rub = fa == "rubber" || fb == "rubber"; mk = rub ? 0.3 : 0.75 * mk; ms = rub ? 0.4 : 0.75 * ms; src += "; wet"; }
    } else {
        mk = 0.3; ms = 0.38; src = "estimate";
    }
}

static double friction_mu(const EnvMat* a, const EnvMat* b, int lube, int med, std::string* note) {
    if (!a || !b) return 0.3;
    { double ms, mk; std::string src; friction_pair(a, b, lube, med, ms, mk, src); if (src != "estimate") { if (a == b && a->mu > 0.9 && note && lube == 0 && !is_oil(med)) *note = std::string(a->label) + " on itself galls when dry"; return mk; } }
    double mu;
    if (a->cls == 1 || b->cls == 1) mu = a->cls == 1 && b->cls == 1 ? std::max(a->mu, b->mu) : (a->cls == 1 ? a->mu : b->mu);
    else if (a->cls == 0 && b->cls == 0) {
        mu = a == b ? a->mu : 0.5 * (a->mu + b->mu) * 0.8;
        if (a == b && a->mu > 0.9 && note) *note = std::string(a->label) + " on itself galls when dry";
    } else mu = 0.35;
    if (is_oil(med) || lube == 2) mu = (a->cls == 1 || b->cls == 1) ? std::min(mu, 0.08) : std::max(0.04, 0.12 * mu);
    else if (lube == 1) mu = std::max(0.05, 0.2 * mu);
    else if (med == MED_WATER || med == MED_SEA) mu = (a->key == std::string("rubber") || b->key == std::string("rubber")) ? 0.3 : mu * 0.75;
    return mu;
}

// ---------------------------------------------------------------- the analysis
static void env_analyse(bool report = true) {
    std::vector<BodyRef> bodies = env_bodies();
    g_eb.clear();
    if (bodies.empty()) { if (!report) return; MessageBoxA(g_main, "Nothing to analyse yet: open a design or an assembly (its parts must have finished meshing).", "Environment", MB_ICONINFORMATION); return; }
    Fluid F = fluid_props(g_env.medium, g_env.Tc, g_env.alt), AIR = fluid_props(MED_AIR, g_env.Tc, 0);
    V3 u = flow_dir();
    const double SIG = 5.670374e-8, g = g_env.g;
    double T_inf = g_env.Tc + 273.15;
    // the liquid's surface: given, or above everything (fully under), or where the whole floats
    double zlo = 1e30, zhi = -1e30;
    for (auto& b : bodies) for (auto& m : b.meshes) for (const V3& v : m.first->v) { double z = m.second.apply(v).z; zlo = std::min(zlo, z); zhi = std::max(zhi, z); }
    double zs = F.liquid ? (g_env.surface < 1e29 ? g_env.surface : zhi + 1.0) : 1e30;
    std::vector<double> vol(bodies.size()), mass(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i) {
        GeoSum s; for (auto& m : bodies[i].meshes) body_geometry(*m.first, m.second, 1e30, u, s);
        vol[i] = std::fabs(s.V); const EnvMat* M = envmat_of(bodies[i].mat); mass[i] = (M ? M->rho : 2700) * vol[i];
    }
    std::string float_note;
    if (F.liquid && g_env.find_float) {  // bisect the waterline so that buoyancy carries the weight (no trim or heel)
        double Wt = 0; for (double m : mass) Wt += m * g;
        auto Bat = [&](double z) { double B = 0; for (auto& b : bodies) { GeoSum s; for (auto& m : b.meshes) body_geometry(*m.first, m.second, z, u, s); B += F.rho * g * std::fabs(s.V); } return B; };
        if (Bat(zhi + 1e-6) < Wt) { zs = zhi + 1.0; float_note = "it does not float: fully under water it still weighs more than the water it displaces"; }
        else {
            double a = zlo, bb = zhi;
            for (int it = 0; it < 60; ++it) { double mid = 0.5 * (a + bb); if (Bat(mid) < Wt) a = mid; else bb = mid; }
            zs = 0.5 * (a + bb);
            float_note = "it floats with its waterline at z = " + fmtd(std::round(zs * 1e4) / 1e4) + " m, draft " + fmtd(std::round((zs - zlo) * 1e4) / 10) + " mm (no trim or heel assumed)";
        }
    }
    for (size_t i = 0; i < bodies.size(); ++i) {
        const BodyRef& br = bodies[i];
        EBody e; e.name = br.name; e.matkey = br.mat; e.M = envmat_of(br.mat); e.comp = br.comp;
        GeoSum all, sub;
        for (auto& m : br.meshes) { body_geometry(*m.first, m.second, 1e30, u, all); body_geometry(*m.first, m.second, zs, u, sub); }
        e.V = std::fabs(all.V); e.A = all.A;
        e.c = e.V > 0 ? V3(all.Vx, all.Vy, all.Vz) / all.V : V3();
        e.Vsub = F.liquid ? std::fabs(sub.V) : e.V; e.Asub = F.liquid ? sub.A : e.A;
        e.cb = std::fabs(sub.V) > 1e-15 ? V3(sub.Vx, sub.Vy, sub.Vz) / sub.V : e.c;
        geo::Box bx; for (auto& m : br.meshes) for (const V3& v : m.first->v) bx.add(m.second.apply(v));
        e.zmin = bx.lo.z; e.zmax = bx.hi.z; e.Lmax = std::max(bx.hi.x - bx.lo.x, std::max(bx.hi.y - bx.lo.y, bx.hi.z - bx.lo.z));
        const EnvMat* M = e.M;
        e.mass = (M ? M->rho : 2700) * e.V; e.W = e.mass * g;
        e.B = (F.liquid ? F.rho * e.Vsub + AIR.rho * (e.V - e.Vsub) : F.rho * e.V) * g;
        // drag: the wetted projection in the liquid (or the whole in a gas)
        e.Lc = e.A > 0 ? 6 * e.V / e.A : 0;
        if (g_env.U > 0 && !F.vacuum && e.Lc > 0) {
            double Ap = F.liquid ? sub.Aproj : all.Aproj;
            e.Re = F.rho * g_env.U * e.Lc / F.mu; e.Cd = cd_sphere(e.Re);
            e.drag = e.Cd * 0.5 * F.rho * g_env.U * g_env.U * Ap;
        }
        // heat: the part at Tpart (or warmed by its share of the heat load), the wetted part in the liquid, the rest in air
        double share = 0; { double vt = 0; for (double v : vol) vt += v; share = vt > 0 ? e.V / vt : 0; }
        double P = g_env.heat * share, Ts = g_env.Tpart + 273.15;
        double Adry = std::max(0.0, e.A - (F.liquid ? e.Asub : 0)), Awet = F.liquid ? e.Asub : 0;
        const Fluid& gasF = F.liquid ? AIR : F;
        for (int it = 0; it < (P > 0 ? 40 : 1); ++it) {
            e.h_liq = F.liquid ? h_conv(F, e.Lc, g_env.U, Ts - T_inf, g) : 0;
            e.h_gas = (F.liquid && Adry <= 0) ? 0 : h_conv(gasF, e.Lc, F.liquid ? 0 : g_env.U, Ts - T_inf, g);
            e.h_rad = (M ? M->eps : 0.5) * SIG * (Ts * Ts + T_inf * T_inf) * (Ts + T_inf);
            double G = e.h_liq * Awet + (e.h_gas + e.h_rad) * (F.liquid ? Adry : e.A);  // W/K
            if (P > 0 && G > 0) Ts = 0.5 * Ts + 0.5 * (T_inf + P / G);
            e.Q = G * (Ts - T_inf);
            double heff = e.A > 0 ? G / e.A : 0;
            e.tau = (heff > 0 && M) ? M->rho * M->cp * e.V / (heff * e.A) : 1e30;
            e.Bi = (M && e.A > 0) ? heff * (e.V / e.A) / M->k : 0;
        }
        if (P > 0 && e.Q <= 0) Ts = 1e9;  // no way to shed the heat (vacuum without radiation)
        e.Ts = Ts - 273.15;
        e.dL = (M ? M->alpha : 0) * e.Lmax * (e.Ts - 20);
        g_eb.push_back(e);
    }
    // ---------------------------------------------------------------- the report
    char b[512];
    std::string r;
    auto line = [&](const std::string& s) { r += s; r += "\n"; };
    snprintf(b, sizeof b, "ENVIRONMENT: %s at %.1f C%s, gravity %.2f m/s2", F.name.c_str(), g_env.Tc, g_env.U > 0 ? (", current " + fmtd(g_env.U) + " m/s along " + FLOW_DIR[g_env.dir]).c_str() : ", still", g); line(b);
    if (F.gas) { snprintf(b, sizeof b, "  air at %.0f m: p %.0f Pa, rho %.4g kg/m3, mu %.3g Pa s, k %.3g W/mK, Pr %.2f", g_env.alt, F.p, F.rho, F.mu, F.k, F.Pr); line(b); }
    else if (F.liquid) {
        snprintf(b, sizeof b, "  rho %.1f kg/m3, mu %.3g Pa s, k %.3g W/mK, cp %.0f J/kgK, Pr %.3g; surface at z = %.4g m", F.rho, F.mu, F.k, F.cp, F.Pr, zs); line(b);
        if (zhi < zs) { snprintf(b, sizeof b, "  fully submerged: top %.3f m below the surface, bottom %.3f m; pressure at the bottom %.4g bar", zs - zhi, zs - zlo, (101325 + F.rho * g * (zs - zlo)) / 1e5); line(b); }
        else if (zlo < zs) { snprintf(b, sizeof b, "  partly submerged; pressure at the bottom %.4g bar", (101325 + F.rho * g * (zs - zlo)) / 1e5); line(b); }
        if (!float_note.empty()) line("  " + float_note);
    } else line("  no convection and no buoyancy: heat leaves only by radiation (surroundings at the given temperature)");
    snprintf(b, sizeof b, "  parts at %.1f C%s; lubrication %s", g_env.Tpart, g_env.heat > 0 ? (" (or warmed by " + fmtd(g_env.heat) + " W shared by volume)").c_str() : "", LUBE_NAME[g_env.lube]); line(b);
    line("");
    line("PART                MATERIAL              MASS kg   WEIGHT N  BUOYANCY N     NET N  UNDER %   DRAG N   h W/m2K  HEAT W     T C   COOL 3tau   Bi   GROWS mm");
    double Wt = 0, Bt = 0, Dt = 0, Qt = 0, Mt = 0;
    for (auto& e : g_eb) {
        double hshow = F.liquid ? e.h_liq : e.h_gas + e.h_rad;
        std::string cool = e.tau > 1e29 ? "never" : e.tau * 3 > 7200 ? fmtd(std::round(e.tau * 3 / 360) / 10) + " h" : e.tau * 3 > 120 ? fmtd(std::round(e.tau * 3 / 6) / 10) + " min" : fmtd(std::round(e.tau * 30) / 10) + " s";
        snprintf(b, sizeof b, "%-18.18s  %-20.20s %8.4g %10.4g %11.4g %9.4g %8.1f %8.3g %9.4g %7.4g %7.1f %11s %6.3f %9.4g",
                 e.name.c_str(), e.M ? e.M->label : e.matkey.c_str(), e.mass, e.W, e.B, e.B - e.W, e.V > 0 ? 100 * e.Vsub / e.V : 0, e.drag, hshow, e.Q, e.Ts, cool.c_str(), e.Bi, e.dL * 1000);
        line(b);
        Wt += e.W; Bt += e.B; Dt += e.drag; Qt += e.Q; Mt += e.mass;
    }
    snprintf(b, sizeof b, "TOTAL                                    %8.4g %10.4g %11.4g %9.4g          %8.3g          %7.4g", Mt, Wt, Bt, Bt - Wt, Dt, Qt); line(b);
    bool floated = F.liquid && g_env.find_float && float_note.find("floats") != std::string::npos && float_note.find("does not") == std::string::npos;
    g_env_floats = floated;
    if (floated) line("  => it FLOATS in balance at the waterline above (the small difference is air buoyancy on the part above the water)");
    else if (Bt - Wt > 1e-9) line("  => as a whole it would RISE (buoyancy exceeds weight) unless it is held down");
    else if (F.vacuum) line("  => vacuum: no buoyancy; it weighs " + fmtd(std::round(Wt * 1000) / 1000) + " N on its supports");
    else line("  => as a whole it SINKS (or rests on its supports) with an apparent weight of " + fmtd(std::round((Wt - Bt) * 1000) / 1000) + " N");
    line("");
    // fits and friction at the mates
    std::vector<std::string> warn;
    auto matc = [&](int c) -> const EnvMat* { for (auto& e : g_eb) if (e.comp == c) return e.M; return nullptr; };
    auto tempc = [&](int c) -> double { for (auto& e : g_eb) if (e.comp == c) return e.Ts; return g_env.Tpart; };
    if (g_asm_on && !g_am.empty()) {
        line("MATES                                   FRICTION mu   FIT AT 20 C      AT TEMPERATURE");
        for (const AMate& m : g_am) {
            if (!m.on || m.a.comp < 0 || m.b.comp < 0 || m.a.comp >= (int)g_ac.size() || m.b.comp >= (int)g_ac.size()) continue;
            const EnvMat *A = matc(m.a.comp), *B = matc(m.b.comp);
            std::string note;
            bool contact = m.type == MT_COINCIDENT || m.type == MT_CONCENTRIC || m.type == MT_INSERT;  // the others hold parts apart or rigid
            double mu = contact ? friction_mu(A, B, g_env.lube, g_env.medium, &note) : 0;
            std::string fit = "-", fitT = "-";
            if (m.type == MT_CONCENTRIC || m.type == MT_INSERT) {  // a shaft in a hole: the clearance on the radius
                auto rad = [&](const ARef& rf, int c) { if (rf.r > 0) return rf.r; const js::Value* fz = g_ac[(size_t)c].part.find("fastener"); return fz ? 0.5 * fz->num("d", 0) : 0.0; };
                double ra = rad(m.a, m.a.comp), rb = rad(m.b, m.b.comp);
                if (ra > 0 && rb > 0 && std::fabs(ra - rb) < 0.2 * std::max(ra, rb)) {
                    bool ahole = m.a.hole || (!m.b.hole && ra >= rb);
                    double rh = ahole ? ra : rb, rsft = ahole ? rb : ra;
                    const EnvMat *Mh = ahole ? A : B, *Msft = ahole ? B : A;
                    double Th = tempc(ahole ? m.a.comp : m.b.comp), Tsf = tempc(ahole ? m.b.comp : m.a.comp);
                    double c0 = rh - rsft, c1 = rh * (1 + (Mh ? Mh->alpha : 0) * (Th - 20)) - rsft * (1 + (Msft ? Msft->alpha : 0) * (Tsf - 20));
                    fit = fmtd(std::round(c0 * 1e7) / 1e4) + " mm"; fitT = fmtd(std::round(c1 * 1e7) / 1e4) + " mm";
                    if (c0 >= 0 && c1 < 0) warn.push_back("mate " + m.id + ": the clearance closes at temperature (" + fit + " -> " + fitT + "): the parts bind or seize");
                    else if (c0 < 0) warn.push_back("mate " + m.id + ": an interference fit (" + fit + ")");
                }
            }
            snprintf(b, sizeof b, "%-14s %-10.10s / %-10.10s %8s      %-14s   %s", MATE_LABEL[m.type], g_ac[(size_t)m.a.comp].id.c_str(), g_ac[(size_t)m.b.comp].id.c_str(), contact ? fmtd(std::round(mu * 1000) / 1000).c_str() : "-", fit.c_str(), fitT.c_str());
            line(b);
            if (!note.empty()) warn.push_back(note + " (mate " + m.id + ")");
            // galvanic couples in an electrolyte
            if ((g_env.medium == MED_SEA || g_env.medium == MED_WATER) && A && B && A->anodic >= 0 && B->anodic >= 0) {
                double dv = std::fabs(A->anodic - B->anodic), lim = g_env.medium == MED_SEA ? 0.15 : 0.25;
                if (dv > lim) {
                    const EnvMat* an = A->anodic > B->anodic ? A : B;
                    snprintf(b, sizeof b, "galvanic corrosion at mate %s: %s against %s (%.2f V apart, limit %.2f V): the %s corrodes; insulate them or choose closer metals", m.id.c_str(), A->label, B->label, dv, lim, an->label);
                    warn.push_back(b);
                }
            }
        }
        line("");
    }
    // ---------------------------------------------------------------- coefficients
    {
        Fluid Fp = F.vacuum ? AIR : F;
        line("HEAT TRANSFER COEFFICIENTS (each part in the " + F.name + "; Lc = 6 V / A)");
        line("PART                Lc mm     Re        Ra        Nu    h conv W/m2K  h rad W/m2K  h total   UA W/K   k solid W/mK  alpha mm2/s     Bi    tau s");
        for (auto& e : g_eb) {
            const Fluid& ff = F.liquid && e.Vsub > 0.5 * e.V ? F : (F.liquid ? AIR : Fp);
            double hc = F.liquid && e.Vsub > 0.5 * e.V ? e.h_liq : e.h_gas, dT = std::fabs(e.Ts - g_env.Tc);
            double nu = ff.rho > 0 ? ff.mu / ff.rho : 0, al = ff.rho > 0 ? ff.k / (ff.rho * ff.cp) : 0;
            double Ra = (nu > 0 && al > 0) ? g * ff.beta * dT * e.Lc * e.Lc * e.Lc / (nu * al) : 0, Nu = ff.k > 0 ? hc * e.Lc / ff.k : 0;
            double hrad = F.liquid && e.Vsub > 0.5 * e.V ? 0 : e.h_rad, ht = hc + hrad, UA = e.A > 0 ? e.Q / std::max(1e-9, e.Ts - g_env.Tc) : 0;
            if (std::fabs(e.Ts - g_env.Tc) < 1e-6) UA = ht * e.A;
            double ks = e.M ? e.M->k : 0, as = e.M ? e.M->k / (e.M->rho * e.M->cp) * 1e6 : 0;
            snprintf(b, sizeof b, "%-18.18s %7.3g %9.3g %9.3g %8.3g %13.4g %12.4g %9.4g %8.3g %13.4g %12.4g %7.3g %8.3g", e.name.c_str(), e.Lc * 1000, e.Re, Ra, Nu, hc, hrad, ht, UA, ks, as, e.Bi, e.tau > 1e29 ? 0.0 : e.tau);
            line(b);
        }
        line("  Re from the current, Ra from the part's temperature difference; Nu = h Lc / k of the fluid; h rad grey-body (emissivity of");
        line("  the material); UA the conductance to the surroundings; Bi under 0.1: the part is at one temperature (lumped tau valid).");
        line("");
        if (g_asm_on && !g_am.empty()) {
            line("FRICTION AND CONTACT COEFFICIENTS (each mate at the present lubrication: " + std::string(LUBE_NAME[g_env.lube]) + (is_oil(g_env.medium) ? ", in oil" : "") + ")");
            line("MATE                         CONTACT        mu static  mu kinetic  rolling    h contact W/m2K   SOURCE / NOTE");
            bool oily = is_oil(g_env.medium) || g_env.lube == 2, lub = oily || g_env.lube == 1;
            for (const AMate& m : g_am) {
                if (!m.on || m.a.comp < 0 || m.b.comp < 0 || m.a.comp >= (int)g_ac.size() || m.b.comp >= (int)g_ac.size()) continue;
                const EnvMat *A = matc(m.a.comp), *B = matc(m.b.comp);
                const js::Value &pa = g_ac[(size_t)m.a.comp].part, &pb = g_ac[(size_t)m.b.comp].part;
                std::string kind = "-", src; double ms = 0, mk = 0, roll = 0; bool touch = true, slides = true;
                friction_pair(A, B, g_env.lube, g_env.medium, ms, mk, src);
                switch (m.type) {
                    case MT_CONCENTRIC: case MT_INSERT:
                        if (pa.find("bearing") || pb.find("bearing")) { kind = "ball bearing"; ms = 0.0020; mk = 0.0015; roll = 0.0015; src = "deep groove ball bearing (SKF: mu 0.0015 on the bore radius)"; }
                        else if (pa.find("bushing") || pb.find("bushing")) { kind = "plain bushing"; src = "journal: " + src; }
                        else kind = "journal / fit";
                        break;
                    case MT_COINCIDENT: case MT_DISTANCE: kind = m.type == MT_COINCIDENT ? "sliding face" : "faces apart"; touch = m.type == MT_COINCIDENT; if (!touch) { ms = mk = 0; src = "no contact"; } break;
                    case MT_LOCK: kind = "bonded / keyed"; ms = mk = 0; slides = false; src = "rigid: no sliding (contact conductance applies)"; break;
                    case MT_GEAR:
                        if (is_worm(m.a.comp) || is_worm(m.b.comp)) { kind = "worm mesh"; mk = lub ? 0.05 : 0.15; ms = lub ? 0.08 : 0.2; src = "worm: mu from the sliding speed (AGMA; Power flow gives it at speed)"; }
                        else { kind = "gear mesh"; mk = lub ? 0.05 : 0.15; ms = lub ? 0.08 : 0.2; src = "tooth sliding, typical (mesh losses 1.5 % lubricated)"; }
                        break;
                    case MT_ROLL: {
                        bool rub = (A && std::string(A->key) == "rubber") || (B && std::string(B->key) == "rubber");
                        kind = "friction drive"; mk = rub ? (lub || F.liquid ? 0.3 : 0.6) : (lub ? 0.06 : 0.15); ms = mk * 1.2; roll = rub ? 0.015 : 0.001;
                        src = std::string("traction coefficient; rolling resistance ") + (rub ? "rubber on metal" : "steel on steel"); break;
                    }
                    case MT_BELT:
                        if (m.drive == 2) { kind = "roller chain"; ms = mk = lub ? 0.1 : 0.25; roll = 0; src = "pin-bush articulation (estimate)"; }
                        else { kind = "V-belt"; mk = lub || F.liquid ? 0.10 : 0.25; ms = mk * 1.2; src = "rubber on the groove; wedge mu' = mu / sin 19 deg = " + fmtd(std::round(mk / std::sin(19 * geo::PI / 180) * 100) / 100); }
                        break;
                    case MT_CAM: {
                        const js::Value* fm = pb.find("follower");
                        if (fm && fm->str("kind", "") == "roller") { kind = "cam roller"; roll = 0.002; ms = mk = lub ? 0.08 : 0.2; src = "rolling on the cam; mu for the roller's pin"; }
                        else { kind = "cam flat face"; src = "sliding: " + src; }
                        break;
                    }
                    default: kind = "-"; ms = mk = 0; touch = false; src = "no contact"; break;
                }
                // thermal contact conductance: Incropera Table 3.2 (aluminium on aluminium, 10 um roughness, 1 bar:
                // 1 / 2.75e-4 in air, 1 / 0.525e-4 with an oil film), scaled by the pair's harmonic-mean conductivity
                double hcon = 0;
                if (touch && A && B) {
                    double ks = 2 * A->k * B->k / std::max(1e-9, A->k + B->k), base = F.vacuum ? 1500 : (F.liquid || oily) ? 1 / 0.525e-4 : 1 / 2.75e-4;
                    hcon = base * ks / 167.0;
                }
                std::string id = std::string(MATE_LABEL[m.type]).substr(0, 10) + " " + g_ac[(size_t)m.a.comp].id + "/" + g_ac[(size_t)m.b.comp].id;
                snprintf(b, sizeof b, "%-28.28s %-14.14s %9s %11s %9s %16s   %s", id.c_str(), kind.c_str(), touch && slides ? fmtd(std::round(ms * 1000) / 1000).c_str() : "-", touch && slides ? fmtd(std::round(mk * 10000) / 10000).c_str() : "-",
                         roll > 0 ? fmtd(roll).c_str() : "-", hcon > 0 ? fmtd(std::round(hcon)).c_str() : "-", src.c_str());
                line(b);
            }
            line("  Static and kinetic coefficients: handbook pairs (Marks, CRC) where listed, otherwise estimates (marked). Contact");
            line("  conductance: Incropera Table 3.2 (Al/Al, 10 um, 1 bar) scaled by the harmonic-mean conductivity (Mikic): +-50 %.");
            line("");
        }
    }
    // motors: what it takes to turn the mechanism
    if (g_asm_on && !g_amot.empty()) {
        line("MOTORS");
        for (const AMotor& mo : g_amot) {
            if (mo.comp < 0 || mo.comp >= (int)g_ac.size()) continue;
            double w0 = mo.speed * geo::PI / 180;
            if (mo.linear) {
                const EBody* eb = nullptr; for (auto& e : g_eb) if (e.comp == mo.comp) eb = &e;
                double N = eb ? std::fabs(eb->W - eb->B) : 0, mu = 0.3;
                double Fr = mu * N + (eb ? eb->drag : 0);
                snprintf(b, sizeof b, "  linear motor on %s: %.3g m/s; friction about %.3g N, power %.3g W", g_ac[(size_t)mo.comp].id.c_str(), mo.speed, Fr, Fr * mo.speed); line(b);
                continue;
            }
            // the turning parts: locked to the motor's part, or driven through gear mates (speed ratio, mesh count)
            std::vector<double> ratio(g_ac.size(), 0); std::vector<int> meshes(g_ac.size(), 0);
            ratio[(size_t)mo.comp] = 1;
            for (int pass = 0; pass < 8; ++pass)
                for (const AMate& m : g_am) {
                    if (!m.on) continue;
                    for (int dirn = 0; dirn < 2; ++dirn) {
                        int a = dirn ? m.b.comp : m.a.comp, c = dirn ? m.a.comp : m.b.comp;
                        if (a < 0 || c < 0 || a >= (int)g_ac.size() || c >= (int)g_ac.size() || ratio[(size_t)a] == 0 || ratio[(size_t)c] != 0) continue;
                        if (m.type == MT_LOCK) { ratio[(size_t)c] = ratio[(size_t)a]; meshes[(size_t)c] = meshes[(size_t)a]; }
                        else if (m.type == MT_GEAR && m.za > 0 && m.zb > 0) { double rr = dirn ? (double)m.zb / m.za : (double)m.za / m.zb; ratio[(size_t)c] = ratio[(size_t)a] * rr; meshes[(size_t)c] = meshes[(size_t)a] + 1; }
                        else if ((m.type == MT_ROLL || m.type == MT_BELT) && m.value > 0) { double rr = dirn ? 1 / m.value : m.value; ratio[(size_t)c] = ratio[(size_t)a] * rr; meshes[(size_t)c] = meshes[(size_t)a] + 1; }
                    }
                }
            double eta1 = g_env.lube || is_oil(g_env.medium) ? 0.985 : 0.95, Ptot = 0, Tb = 0, Tf = 0, Ieff = 0, hang = 0;
            for (size_t c = 0; c < g_ac.size(); ++c) {  // the load of turning parts with no bearing of their own goes to the motor's part
                if (ratio[c] == 0 || (int)c == mo.comp) continue;
                bool own = false;
                for (const AMate& m : g_am) if (m.on && (m.type == MT_CONCENTRIC || m.type == MT_INSERT) && (m.a.comp == (int)c || m.b.comp == (int)c)) { int o = m.a.comp == (int)c ? m.b.comp : m.a.comp; if (o >= 0 && ratio[(size_t)o] == 0) own = true; }
                if (!own) for (auto& e : g_eb) if (e.comp == (int)c) hang += std::fabs(e.W - e.B);
            }
            V3 axw = g_ac[(size_t)mo.comp].pose.rot(mo.d);
            for (size_t c = 0; c < g_ac.size(); ++c) {
                if (ratio[c] == 0) continue;
                const EBody* eb = nullptr; for (auto& e : g_eb) if (e.comp == (int)c) eb = &e;
                if (!eb) continue;
                double wc = std::fabs(w0 * ratio[c]);
                // bearing friction: every concentric or insert mate to a part that does not turn
                double N = std::fabs(eb->W - eb->B) + ((int)c == mo.comp ? hang : 0), tb = 0; int nb = 0;
                for (const AMate& m : g_am) if (m.on && (m.type == MT_CONCENTRIC || m.type == MT_INSERT) && (m.a.comp == (int)c || m.b.comp == (int)c)) { int o = m.a.comp == (int)c ? m.b.comp : m.a.comp; if (o >= 0 && ratio[(size_t)o] == 0) ++nb; }
                for (const AMate& m : g_am) {
                    if (!m.on || (m.type != MT_CONCENTRIC && m.type != MT_INSERT) || (m.a.comp != (int)c && m.b.comp != (int)c)) continue;
                    int o = m.a.comp == (int)c ? m.b.comp : m.a.comp;
                    if (o < 0 || ratio[(size_t)o] != 0) continue;
                    double rr = std::max(m.a.r, m.b.r); if (rr <= 0) rr = 0.005;
                    tb += friction_mu(eb->M, matc(o), g_env.lube, g_env.medium, nullptr) * (N / std::max(1, nb)) * rr;
                }
                // fluid drag on the turning part, as a disc of its largest radius about the axis
                double R = 0; V3 ac = g_ac[c].pose.apply(V3());
                for (const APiece& pc : g_ac[c].pieces) if (pc.mesh) for (const V3& v : pc.mesh->m.v) { V3 w = g_ac[c].pose.apply(pc.local.apply(v)) - ac; w = w - axw * geo::dot(w, axw); R = std::max(R, geo::len(w)); }
                const Fluid& Fl = (F.liquid && eb->c.z < zs) ? F : (F.liquid ? AIR : F);
                double tf = 0;
                if (!Fl.vacuum && Fl.rho > 0 && wc > 0 && R > 0) { double Rw = wc * R * R * Fl.rho / Fl.mu, Cm = Rw < 3e5 ? 3.87 / std::sqrt(std::max(Rw, 1.0)) : 0.146 * std::pow(Rw, -0.2); tf = Cm * 0.5 * Fl.rho * wc * wc * std::pow(R, 5); }
                Ptot += (tb + tf) * wc / std::pow(eta1, meshes[c]);
                Tb += tb * ratio[c]; Tf += tf * ratio[c];
                Ieff += 0.5 * eb->mass * R * R * ratio[c] * ratio[c];
            }
            double Tm = w0 > 0 ? Ptot / w0 : 0, Tacc = Ieff * w0 / 0.5;
            snprintf(b, sizeof b, "  rotary motor on %s: %.0f deg/s (%.1f rpm). Steady torque %.4g N m (bearings %.3g, fluid drag %.3g, gear losses at %.1f %% per mesh), power %.4g W; to reach speed in 0.5 s add %.3g N m (inertia %.3g kg m2)",
                     g_ac[(size_t)mo.comp].id.c_str(), mo.speed, mo.speed / 6, Tm, Tb, Tf, 100 * (1 - eta1), Ptot, Tacc, Ieff);
            line(b);
        }
        line("");
    }
    // warnings: corrosion, temperatures, the medium itself
    for (auto& e : g_eb) {
        if (!e.M) continue;
        std::string k = e.M->key;
        if ((g_env.medium == MED_SEA || g_env.medium == MED_WATER) && e.Vsub > 0 && (k == "steel_mild" || k == "cast_iron" || k == "steel_300m"))
            warn.push_back(e.name + " (" + e.M->label + ") rusts in " + F.name + ": paint, galvanise or choose stainless steel or bronze");
        if (g_env.medium == MED_SEA && e.Vsub > 0 && (k == "al2024" || k == "al7075" || k == "al_li"))
            warn.push_back(e.name + " (" + e.M->label + ") pits in sea water: anodise it or use a marine alloy (5083, 6061)");
        if (e.Ts > e.M->tmax) { snprintf(b, sizeof b, "%s reaches %.0f C, above the %.0f C that %s takes in service", e.name.c_str(), e.Ts, e.M->tmax, e.M->label); warn.push_back(b); }
        if (e.Bi > 0.1) warn.push_back(e.name + ": Biot number " + fmtd(std::round(e.Bi * 100) / 100) + " > 0.1, so its inside lags its surface; the cooling time is a lower bound");
        if (e.Ts > 1e8) warn.push_back(e.name + ": nothing carries its heat away (no medium and no radiation): its temperature rises without limit");
    }
    if (F.liquid && g_env.Tc <= F.Tfreeze) warn.push_back(F.name + " freezes at " + fmtd(F.Tfreeze) + " C: the medium is ice, not liquid");
    if (F.liquid && g_env.Tc >= F.Tboil) warn.push_back(F.name + " boils at " + fmtd(F.Tboil) + " C at 1 atm");
    if (F.liquid && g_env.Tpart >= F.Tboil) warn.push_back("the parts at " + fmtd(g_env.Tpart) + " C boil the " + F.name + " on their surfaces (film boiling: real heat transfer differs from the convection estimate)");
    if (F.vacuum && g_asm_on) warn.push_back("in vacuum, clean metal-on-metal mates can cold-weld; use dry lubricants (MoS2) or coatings");
    line("WARNINGS");
    if (warn.empty()) line("  none");
    for (auto& w : warn) line("  - " + w);
    line("");
    line("METHODS (estimates): submerged volume and buoyancy centre exact on the mesh; convection from sphere correlations on the");
    line("equivalent diameter 6V/A (Whitaker 1972 forced, Churchill 1983 natural, Churchill cube mixing) plus grey-body radiation in");
    line("gases and vacuum; drag from the sphere law of Clift & Gauvin (1971) on the projected area; lumped cooling time 3 tau with");
    line("tau = rho c V / (h A); turning parts as discs (von Karman); bearing friction mu N r; gear mesh efficiency 95 % dry, 98.5 %");
    line("lubricated; galvanic limits MIL-STD-889 (0.15 V sea water, 0.25 V fresh water). Shapes far from compact (plates, long");
    line("rods) can differ by a factor of two in h and drag: use the CFD and thermal tools for design numbers.");
    g_env_summary = r;
    g_env.on = true;
    g_doc["environment"] = js::Value::object();
    {
        js::Value& o = g_doc["environment"];
        o["medium"] = MED_NAME[g_env.medium]; o["temperature_C"] = g_env.Tc; o["flow_m_s"] = g_env.U; o["flow_dir"] = FLOW_DIR[g_env.dir];
        o["altitude_m"] = g_env.alt; if (g_env.surface < 1e29) o["surface_z_m"] = g_env.surface; o["find_float"] = g_env.find_float;
        o["part_temperature_C"] = g_env.Tpart; o["heat_W"] = g_env.heat; o["lubrication"] = LUBE_NAME[g_env.lube]; o["gravity_m_s2"] = g_env.g;
    }
    g_env.surface = F.liquid ? zs : 1e30;  // shown in the view
    if (!g_selftest && report) {
        SYSTEMTIME t; GetLocalTime(&t);
        char nm[64]; snprintf(nm, sizeof nm, "env_%04d%02d%02d_%02d%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
        std::string dir = g_root + "\\results\\studio\\" + nm;
        make_dirs(dir);
        try {
            js::write_file(dir + "\\summary.txt", r);
            js::Value o = g_doc["environment"].clone(), parts = js::Value::array();
            for (auto& e : g_eb) {
                js::Value p = js::Value::object();
                p["name"] = e.name; p["material"] = e.matkey; p["volume_m3"] = e.V; p["area_m2"] = e.A; p["mass_kg"] = e.mass; p["weight_N"] = e.W; p["buoyancy_N"] = e.B;
                p["submerged_fraction"] = e.V > 0 ? e.Vsub / e.V : 0; p["drag_N"] = e.drag; p["h_W_m2K"] = F.liquid ? e.h_liq : e.h_gas + e.h_rad; p["heat_W"] = e.Q;
                p["temperature_C"] = e.Ts; p["tau_s"] = e.tau < 1e29 ? e.tau : -1; p["biot"] = e.Bi; p["growth_m"] = e.dL;
                parts.push(p);
            }
            o["parts"] = parts;
            js::Value wv = js::Value::array(); for (auto& w : warn) wv.push(js::Value(w)); o["warnings"] = wv;
            js::write_file(dir + "\\environment.json", js::dump(o, 1));
        } catch (...) {}
        snap_view_png(dir + "\\00_view_3d.png"); open_simview(dir, "Environment: " + F.name);
    }
    status("environment: " + F.name + " at " + fmtd(g_env.Tc) + " C; " + tostr((int)g_eb.size()) + " parts analysed, " + tostr((int)warn.size()) + " warnings (report in the results window)");
    InvalidateRect(g_view, nullptr, FALSE);
}

static void env_from_doc() {
    if (const js::Value* o = g_doc.find("environment")) {  // the last settings of this document
        std::string m = o->str("medium", MED_NAME[g_env.medium]);
        for (int i = 0; i < NMED; ++i) if (m == MED_NAME[i]) g_env.medium = i;
        g_env.Tc = o->num("temperature_C", g_env.Tc); g_env.U = o->num("flow_m_s", g_env.U); g_env.alt = o->num("altitude_m", g_env.alt);
        g_env.Tpart = o->num("part_temperature_C", g_env.Tpart); g_env.heat = o->num("heat_W", g_env.heat); g_env.g = o->num("gravity_m_s2", g_env.g);
        g_env.find_float = o->flag("find_float", g_env.find_float);
        if (o->find("surface_z_m")) g_env.surface = o->num("surface_z_m", 1e30);
        for (int i = 0; i < 6; ++i) if (o->str("flow_dir", "") == FLOW_DIR[i]) g_env.dir = i;
        for (int i = 0; i < 3; ++i) if (o->str("lubrication", "") == LUBE_NAME[i]) g_env.lube = i;
    }
}
static void flow_cfg_auto();
static void flow_setup();
static void env_setup() {
    env_from_doc();
    std::vector<std::string> meds(MED_NAME, MED_NAME + NMED), dirs(FLOW_DIR, FLOW_DIR + 6), lubes(LUBE_NAME, LUBE_NAME + 3);
    std::vector<FField> f = {
        {"Medium", meds, MED_NAME[g_env.medium]},
        {"Medium temperature (C)", {}, fmtd(g_env.Tc)},
        {"Current / flow speed (m/s)", {}, fmtd(g_env.U)},
        {"Flow direction", dirs, FLOW_DIR[g_env.dir]},
        {"Altitude for air (m)", {}, fmtd(g_env.alt)},
        {"Liquid surface z (m), blank: 1 m above", {}, g_env.surface < 1e29 ? fmtd(g_env.surface) : std::string()},
        {"Waterline", {"as given above", "find where it floats"}, g_env.find_float ? "find where it floats" : "as given above"},
        {"Part temperature (C)", {}, fmtd(g_env.Tpart)},
        {"Heat generated in the parts (W)", {}, fmtd(g_env.heat)},
        {"Lubrication of the mates", lubes, LUBE_NAME[g_env.lube]},
        {"Gravity (m/s2): Earth 9.81, Moon 1.62, Mars 3.71", {}, fmtd(g_env.g)},
        {"3D flow simulation (the whole volume)", {"yes: simulate the flow, heat and turbulence in the whole x-y-z volume", "no: the correlations only"}, g_selftest ? "no: the correlations only" : "yes: simulate the flow, heat and turbulence in the whole x-y-z volume"},
    };
    std::string intro = "Where does it work? The medium, its temperature and current, the liquid's surface, gravity and lubrication. Studio then works out for every part: weight and buoyancy, pressure, drag, heat transfer, expansion, friction, motor power and corrosion risks.";
    if (!form_dialog("Environment", intro, f)) return;
    for (int i = 0; i < NMED; ++i) if (f[0].value == MED_NAME[i]) g_env.medium = i;
    g_env.Tc = atof(f[1].value.c_str()); g_env.U = std::max(0.0, atof(f[2].value.c_str()));
    for (int i = 0; i < 6; ++i) if (f[3].value == FLOW_DIR[i]) g_env.dir = i;
    g_env.alt = atof(f[4].value.c_str());
    g_env.surface = f[5].value.find_first_not_of(" \t") == std::string::npos ? 1e30 : atof(f[5].value.c_str());
    g_env.find_float = f[6].value == "find where it floats";
    g_env.Tpart = atof(f[7].value.c_str()); g_env.heat = std::max(0.0, atof(f[8].value.c_str()));
    for (int i = 0; i < 3; ++i) if (f[9].value == LUBE_NAME[i]) g_env.lube = i;
    g_env.g = std::max(0.0, atof(f[10].value.c_str()));
    push_undo();
    env_analyse();
    if (f[11].value.compare(0, 3, "yes") == 0) { flow_cfg_auto(); flow_setup(); }
}
static void env_clear() { g_env.on = false; g_eb.clear(); status("environment display off"); InvalidateRect(g_view, nullptr, FALSE); }

// ---------------------------------------------------------------- in the 3D view: the liquid, the forces, the current
static void env_arrow(const V3& a, const V3& b, float r, float gg, float bb, float w) {
    glColor4f(r, gg, bb, 0.95f); glLineWidth(w);
    V3 d = b - a; double l = geo::len(d); if (l < 1e-12) return;
    V3 u = d / l, p = std::fabs(u.z) < 0.9 ? geo::cross(u, V3(0, 0, 1)) : geo::cross(u, V3(1, 0, 0)); p = p / geo::len(p);
    V3 q = geo::cross(u, p), h = b - u * (0.25 * l);
    glBegin(GL_LINES);
    glVertex3d(a.x, a.y, a.z); glVertex3d(b.x, b.y, b.z);
    for (int k = 0; k < 4; ++k) { double an = k * geo::PI / 2; V3 e = h + (p * std::cos(an) + q * std::sin(an)) * (0.09 * l); glVertex3d(b.x, b.y, b.z); glVertex3d(e.x, e.y, e.z); }
    glEnd(); glLineWidth(1.0f);
}
static void env_render() {
    if (!g_env.on || g_eb.empty()) return;
    geo::Box bx;
    if (g_asm_on) { for (const AComp& c : g_ac) comp_world_box(c, bx); } else { bx.add(g_props.lo); bx.add(g_props.hi); }
    if (!bx.valid()) return;
    double S = geo::len(bx.hi - bx.lo);
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    double Wm = 1e-30; for (auto& e : g_eb) Wm = std::max(Wm, std::max(e.W, e.B));
    glDisable(GL_DEPTH_TEST);
    for (auto& e : g_eb) {  // weight (red, down from the centre of mass) and buoyancy (blue, up from the centre of buoyancy)
        double Lw = 0.3 * S * e.W / Wm, Lb = 0.3 * S * e.B / Wm;
        if (Lw > 1e-6 * S) env_arrow(e.c, e.c - V3(0, 0, Lw), 0.85f, 0.15f, 0.12f, 2.6f);
        if (Lb > 1e-6 * S) env_arrow(e.cb, e.cb + V3(0, 0, Lb), 0.1f, 0.4f, 0.95f, 2.6f);
    }
    if (g_env.U > 0) {  // the current: arrows upstream of the parts
        V3 u = flow_dir(), mid = (bx.lo + bx.hi) * 0.5, p = std::fabs(u.z) < 0.9 ? geo::cross(u, V3(0, 0, 1)) : geo::cross(u, V3(1, 0, 0)); p = p / geo::len(p);
        V3 q = geo::cross(u, p);
        for (int a = -1; a <= 1; ++a) for (int c = -1; c <= 1; c += 2) { V3 s = mid - u * (0.75 * S) + p * (0.25 * S * a) + q * (0.15 * S * c); env_arrow(s, s + u * (0.3 * S), 0.05f, 0.6f, 0.55f, 2.0f); }
    }
    glEnable(GL_DEPTH_TEST);
    Fluid F = fluid_props(g_env.medium, g_env.Tc, g_env.alt);
    if (F.liquid) {  // the liquid: a tinted volume up to the surface, its surface a little stronger
        double pad = 0.25 * S, top = std::min(g_env.surface, bx.hi.z + pad), bot = bx.lo.z - pad;
        double x0 = bx.lo.x - pad, x1 = bx.hi.x + pad, y0 = bx.lo.y - pad, y1 = bx.hi.y + pad;
        float cr = is_oil(g_env.medium) ? 0.75f : 0.2f, cg = is_oil(g_env.medium) ? 0.55f : 0.5f, cb = is_oil(g_env.medium) ? 0.15f : 0.9f;
        glDepthMask(GL_FALSE);
        glColor4f(cr, cg, cb, 0.10f);
        glBegin(GL_QUADS);
        double X[2] = {x0, x1}, Y[2] = {y0, y1};
        for (int s = 0; s < 2; ++s) {
            glVertex3d(X[s], y0, bot); glVertex3d(X[s], y1, bot); glVertex3d(X[s], y1, top); glVertex3d(X[s], y0, top);
            glVertex3d(x0, Y[s], bot); glVertex3d(x1, Y[s], bot); glVertex3d(x1, Y[s], top); glVertex3d(x0, Y[s], top);
        }
        glVertex3d(x0, y0, bot); glVertex3d(x1, y0, bot); glVertex3d(x1, y1, bot); glVertex3d(x0, y1, bot);
        glEnd();
        if (g_env.surface <= bx.hi.z + pad) {
            glColor4f(cr, cg, cb, 0.28f);
            glBegin(GL_QUADS); glVertex3d(x0, y0, top); glVertex3d(x1, y0, top); glVertex3d(x1, y1, top); glVertex3d(x0, y1, top); glEnd();
            glColor4f(cr * 0.6f, cg * 0.6f, cb * 0.8f, 0.8f); glLineWidth(1.6f);
            glBegin(GL_LINE_LOOP); glVertex3d(x0, y0, top); glVertex3d(x1, y0, top); glVertex3d(x1, y1, top); glVertex3d(x0, y1, top); glEnd(); glLineWidth(1.0f);
        }
        glDepthMask(GL_TRUE);
    }
    glDisable(GL_LINE_SMOOTH); glDisable(GL_BLEND); glEnable(GL_LIGHTING);
}
static void env_overlay(int W, int H) {
    (void)W;
    if (!g_env.on || g_eb.empty()) return;
    double Wt = 0, Bt = 0, Q = 0; for (auto& e : g_eb) { Wt += e.W; Bt += e.B; Q += e.Q; }
    char b[256];
    snprintf(b, sizeof b, "%s %.0f C%s   weight %.4g N, buoyancy %.4g N (%s)   heat %.4g W   [red: weight, blue: buoyancy]", MED_NAME[g_env.medium], g_env.Tc,
             g_env.U > 0 ? ("  current " + fmtd(g_env.U) + " m/s " + FLOW_DIR[g_env.dir]).c_str() : "", Wt, Bt, g_env_floats ? "floats" : Bt > Wt ? "rises" : "sinks", Q);
    glColor3f(0.1f, 0.25f, 0.55f);
    gl_text(16, (float)H - 54, b);  // under the floating view toolbar
}

// --env-run: analyse with the document's saved environment once everything is meshed (tests, reports)
static bool g_env_pending = false;
static void env_run_tick() {
    if (!g_env_pending) return;
    bool ready = g_asm_on ? !g_ac.empty() : !g_mesh.t.empty();
    if (g_asm_on) for (const AComp& c : g_ac) if (!comp_ready(c)) ready = false;
    if (!ready) return;
    g_env_pending = false; KillTimer(g_main, 42);
    env_from_doc();
    env_analyse();
}
