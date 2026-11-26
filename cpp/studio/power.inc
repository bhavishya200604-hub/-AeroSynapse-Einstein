// Power flow: power given to a motor's part goes through the mechanism (locks, gears, racks, rollers) to every part
// it drives. First the speeds follow the ratios (gears by their teeth, rollers by their radii with 2 % creep, a rack
// by the pinion's pitch radius); then the power passes from part to part, less what each part loses: bearing friction
// (mu N r, with rolling bearings at mu 0.0015 and bushings at mu 0.08 lubricated), fluid drag on turning parts (as
// discs; the submerged share of a part in the liquid, the rest in air: churning in an oil bath), gear mesh losses
// and roller creep; a roller drive slips when the torque it must pass exceeds mu x preload x radius. The parts at the
// ends of the chain get what is left: their speed, torque (or force) and power, and the efficiency of the whole.

struct PFNode { double w = 0, v = 0, Pin = 0, Lb = 0, Lf = 0, Ledge = 0, Pout = 0, T = 0; int parent = -1, ptype = -1; bool reached = false, linear = false, slip = false, terminal = false; std::string note; };
static std::vector<PFNode> g_pf;
static std::string g_pf_doc;

static const EBody* eb_of(int c) { for (auto& e : g_eb) if (e.comp == c) return &e; return nullptr; }
static void comp_axis(int c, V3& p, V3& d) {  // a turning part's axis: from its gear, roller or bearing mates, else its z axis
    for (const AMate& m : g_am) {
        if (!m.on) continue;
        const ARef* r = m.a.comp == c ? &m.a : m.b.comp == c ? &m.b : nullptr;
        if (!r || r->kind != 1) continue;
        if (m.type == MT_GEAR || m.type == MT_ROLL || m.type == MT_BELT || m.type == MT_CONCENTRIC || m.type == MT_INSERT) { WRef w = world(*r); p = w.p; d = w.d; return; }
    }
    p = g_ac[(size_t)c].pose.t; d = g_ac[(size_t)c].pose.rot(V3(0, 0, 1));
}
static bool via_chain(int c) {  // the part is driven by a chain (else a belt)
    for (const AMate& m : g_am) if (m.on && m.type == MT_BELT && (m.a.comp == c || m.b.comp == c)) return m.drive == 2;
    return false;
}
static void power_flow(bool report) {
    if (!g_asm_on || g_amot.empty()) { MessageBoxA(g_main, "Power flow needs an assembly with a motor (Motion > Motor on the part that is driven).", "Power flow", MB_ICONINFORMATION); return; }
    env_from_doc();
    env_analyse(false);  // weights, buoyancy and the submerged share of every part, in the document's environment
    Fluid F = fluid_props(g_env.medium, g_env.Tc, g_env.alt), AIR = fluid_props(MED_AIR, g_env.Tc, 0);
    bool lube = g_env.lube > 0 || is_oil(g_env.medium);
    size_t n = g_ac.size();
    g_pf.assign(n, PFNode());
    std::string out; char b[600];
    auto line = [&](const std::string& t) { out += t; out += "\n"; };
    snprintf(b, sizeof b, "POWER FLOW in %s at %.0f C%s; lubrication %s; roller preload %.0f N; belt tension %.0f N", F.name.c_str(), g_env.Tc,
             F.liquid ? (", liquid surface at z = " + fmtd(std::round(g_env.surface * 1e4) / 1e4) + " m").c_str() : "", lube ? "oil or grease" : "dry", g_pf_preload, g_pf_tension);
    line(b); line("");
    double Pin_total = 0, Pout_total = 0, Ltot = 0;
    std::vector<std::string> warn;
    for (const AMotor& mo : g_amot) {
        if (mo.comp < 0 || mo.comp >= (int)n || mo.linear) continue;
        double P0 = mo.power > 0 ? mo.power : 100;
        double w0 = std::fabs(mo.speed) * geo::PI / 180;
        // speeds, breadth first from the motor's part
        std::vector<int> order{mo.comp};
        PFNode& s0 = g_pf[(size_t)mo.comp]; s0.reached = true; s0.w = w0; s0.Pin = P0;
        for (size_t q = 0; q < order.size(); ++q) {
            int c = order[q];
            for (const AMate& m : g_am) {
                if (!m.on || (m.type != MT_LOCK && m.type != MT_GEAR && m.type != MT_ROLL && m.type != MT_BELT)) continue;
                int o = m.a.comp == c ? m.b.comp : m.b.comp == c ? m.a.comp : -1;
                if (o < 0 || o >= (int)n || g_pf[(size_t)o].reached) continue;
                PFNode& t = g_pf[(size_t)o]; PFNode& f = g_pf[(size_t)c];
                bool fwd = m.a.comp == c;
                if (m.type == MT_LOCK) { t.w = f.w; t.v = f.v; t.linear = f.linear; }
                else if (m.type == MT_ROLL) { double rr = fwd ? m.value : 1 / m.value; t.w = f.w * rr * 0.98; }
                else if (m.type == MT_BELT) { double rr = fwd ? m.value : 1 / m.value; t.w = f.w * rr * (m.drive == 2 ? 1.0 : 0.985); }  // a V-belt creeps 1.5 %
                else if (m.zb == 0 && m.rp > 0) {  // rack and pinion
                    if (fwd) { t.linear = true; t.v = f.w * m.rp; } else { t.w = f.v / m.rp; }
                } else if (m.za > 0 && m.zb > 0) { double rr = fwd ? (double)m.za / m.zb : (double)m.zb / m.za; t.w = f.w * rr; }
                t.reached = true; t.parent = c; t.ptype = m.type;
                order.push_back(o);
            }
        }
        // losses at every reached part, then the power passed on to its children (shared equally)
        for (int c : order) {
            PFNode& N = g_pf[(size_t)c];
            const EBody* e = eb_of(c);
            double load = e ? std::fabs(e->W - e->B) : 0;
            for (int k : order) if (g_pf[(size_t)k].parent == c && g_pf[(size_t)k].ptype == MT_LOCK) {  // parts locked on it without bearings of their own
                bool own = false;
                for (const AMate& m : g_am) if (m.on && (m.type == MT_CONCENTRIC || m.type == MT_INSERT) && (m.a.comp == k || m.b.comp == k)) { int o = m.a.comp == k ? m.b.comp : m.a.comp; if (o >= 0 && !g_pf[(size_t)o].reached) own = true; }
                const EBody* ek = eb_of(k);
                if (!own && ek) load += std::fabs(ek->W - ek->B);
            }
            if (!N.linear) {
                // bearings: concentric or insert mates to parts that do not turn
                std::vector<std::pair<double, double>> brg;  // mu, radius
                for (const AMate& m : g_am) {
                    if (!m.on || (m.type != MT_CONCENTRIC && m.type != MT_INSERT) || (m.a.comp != c && m.b.comp != c)) continue;
                    int o = m.a.comp == c ? m.b.comp : m.a.comp;
                    if (o < 0 || g_pf[(size_t)o].reached) continue;
                    double r = std::max(mate_radius(m.a, m.a.comp), mate_radius(m.b, m.b.comp)); if (r <= 0) r = 0.005;
                    const js::Value &Pc = g_ac[(size_t)c].part, &Po = g_ac[(size_t)o].part;
                    double mu = (Pc.find("bearing") || Po.find("bearing")) ? 0.0015 : (Pc.find("bushing") || Po.find("bushing")) ? (lube ? 0.08 : 0.15)
                              : friction_mu(e ? e->M : nullptr, eb_of(o) ? eb_of(o)->M : nullptr, g_env.lube, g_env.medium, nullptr);
                    brg.push_back({mu, r});
                }
                for (auto& br : brg) N.Lb += br.first * (load / brg.size()) * br.second * N.w;
                // fluid drag: the turning part as a disc of its largest radius about its axis
                V3 ap, ad; comp_axis(c, ap, ad);
                double R = 0;
                for (const APiece& pc : g_ac[(size_t)c].pieces) if (pc.mesh) for (const V3& v : pc.mesh->m.v) { V3 w = g_ac[(size_t)c].pose.apply(pc.local.apply(v)) - ap; w = w - ad * geo::dot(w, ad); R = std::max(R, geo::len(w)); }
                double frac = (e && e->V > 0 && F.liquid) ? e->Vsub / e->V : 0;
                auto disc = [&](const Fluid& fl) { if (fl.vacuum || fl.rho <= 0 || N.w <= 0 || R <= 0) return 0.0; double Rw = N.w * R * R * fl.rho / fl.mu, Cm = Rw < 3e5 ? 3.87 / std::sqrt(std::max(Rw, 1.0)) : 0.146 * std::pow(Rw, -0.2); return Cm * 0.5 * fl.rho * N.w * N.w * std::pow(R, 5) * N.w; };
                N.Lf = F.liquid ? frac * disc(F) + (1 - frac) * disc(AIR) : disc(F);
                if (frac > 0.02) N.note += std::string(N.note.empty() ? "" : "; ") + "churns the " + F.name + " (" + fmtd(std::round(frac * 100)) + " % in)";
            } else {
                double mu = friction_mu(e ? e->M : nullptr, envmat_of("steel_mild"), g_env.lube, g_env.medium, nullptr);  // a rack on its guides
                N.Lb = mu * load * N.v + (e ? e->drag * 0 : 0);
            }
            double avail = std::max(0.0, N.Pin - N.Lb - N.Lf);
            std::vector<int> kids; for (int k : order) if (g_pf[(size_t)k].parent == c) kids.push_back(k);
            if (kids.empty()) { N.terminal = true; N.Pout = avail; }
            for (int k : kids) {
                PFNode& K = g_pf[(size_t)k];
                double share = avail / kids.size(), eta = 1;
                const AMate* via = nullptr;
                for (const AMate& m : g_am) if (m.on && ((m.a.comp == c && m.b.comp == k) || (m.b.comp == c && m.a.comp == k)) && m.type == K.ptype) via = &m;
                if (K.ptype == MT_GEAR && via && is_worm(via->a.comp)) {
                    // worm and wheel: eta = tan g / tan(g + phi') driving from the worm, tan(g - phi') / tan g back from the
                    // wheel (self-locking when g <= phi'), phi' = atan(mu / cos 20 deg); mu from the sliding speed (AGMA,
                    // hardened steel on bronze, lubricated; 0.15 dry)
                    const js::Value* wg = gear_meta(via->a.comp);
                    double gam = wg->num("lead_angle", 5) * geo::PI / 180, r1 = wg->num("pitch_radius", 0.02);
                    bool from_worm = via->a.comp == c;
                    double w1 = from_worm ? N.w : K.w, vs = w1 * r1 / std::cos(gam), fpm = vs * 196.85;
                    double mu = !lube ? 0.15 : fpm <= 1e-9 ? 0.15 : fpm <= 10 ? 0.124 * std::exp(-0.074 * std::pow(fpm, 0.645)) : 0.103 * std::exp(-0.110 * std::pow(fpm, 0.450)) + 0.012;
                    double phi = std::atan(mu / std::cos(20 * geo::PI / 180));
                    double eta_f = std::tan(gam) / std::tan(gam + phi), eta_b = gam > phi ? std::tan(gam - phi) / std::tan(gam) : 0;
                    eta = from_worm ? eta_f : eta_b;
                    snprintf(b, sizeof b, "worm: lead %.2f deg, sliding %.3g m/s, mu %.3f (friction angle %.2f deg): %.1f %% from the worm, %s", gam * 180 / geo::PI, vs, mu, phi * 180 / geo::PI, eta_f * 100,
                             eta_b > 0 ? ("back-drives at " + fmtd(std::round(eta_b * 1000) / 10) + " %").c_str() : "self-locking (the wheel cannot turn the worm)");
                    K.note = b;
                    if (!from_worm && eta_b <= 0) { K.slip = true; snprintf(b, sizeof b, "%s cannot drive the worm %s: the drive is self-locking (lead %.2f deg <= friction angle %.2f deg)", g_ac[(size_t)c].id.c_str(), g_ac[(size_t)k].id.c_str(), gam * 180 / geo::PI, phi * 180 / geo::PI); warn.push_back(b); }
                } else if (K.ptype == MT_GEAR && via) {
                    bool bevel = gear_meta(via->a.comp) && gear_meta(via->a.comp)->num("cone", 0) > 0;
                    bool rack = via->zb == 0 && via->rp > 0;
                    eta = rack ? (lube ? 0.97 : 0.94) : bevel ? (lube ? 0.97 : 0.94) : (lube ? 0.985 : 0.96);
                } else if (K.ptype == MT_BELT && via) {
                    DriveLoop D = drive_loop(*via);
                    bool from_a = via->a.comp == c;
                    double rd = from_a ? D.rA : D.rB, Tneed = N.w > 0 ? share / N.w : 0, pull = rd > 0 ? Tneed / rd : 0;
                    double wrap = D.ok ? geo::PI - 2 * std::fabs(D.b) : geo::PI;
                    if (via->drive == 2) {  // roller chain: 98 % lubricated, 96 % dry; its pull against the breaking load
                        eta = lube ? 0.98 : 0.96;
                        const js::Value* sm = g_ac[(size_t)via->a.comp].part.find("sprocket");
                        double Fu = sm ? sm->num("strength_N", 18000) : 18000, p = sm ? sm->num("pitch", 0.0127) : 0.0127;
                        int za = sm ? (int)sm->num("teeth", 12) : 12; const js::Value* sb = g_ac[(size_t)via->b.comp].part.find("sprocket"); int zb = sb ? (int)sb->num("teeth", za) : za;
                        double v = N.w * rd, ripple = 1 - std::cos(geo::PI / std::min(za, zb));
                        snprintf(b, sizeof b, "chain %s, %d links, %.3g m/s, pull %.4g N (safety %.0f on %.0f kN), speed ripple %.1f %%", sm ? sm->str("chain", "").c_str() : "", D.ok ? (int)std::round(D.L / p) : 0, v, pull, pull > 0 ? Fu / pull : 0.0, Fu / 1000, ripple * 100);
                        K.note = b;
                        if (pull > 0 && Fu / pull < 7) { snprintf(b, sizeof b, "the chain to %s pulls %.4g N: only %.1f times its breaking load (keep 7 or more); use a bigger chain", g_ac[(size_t)k].id.c_str(), pull, Fu / pull); warn.push_back(b); }
                    } else {  // V-belt: 95 % with its 1.5 % creep; the most it can pull before it slips (Euler-Eytelwein,
                              // wedge: mu' = mu / sin 19 deg, less the centrifugal tension of 0.07 kg/m at v)
                        eta = 0.95;
                        double mu = lube || F.liquid ? 0.10 : 0.25, mue = mu / std::sin(19 * geo::PI / 180), ex = std::exp(mue * wrap), v = N.w * rd;
                        double Fc = 0.07 * v * v, Fmax = std::max(0.0, 2 * (g_pf_tension - Fc) * (ex - 1) / (ex + 1));
                        snprintf(b, sizeof b, "V-belt, %.0f mm, %.3g m/s, wrap %.0f deg: pulls %.4g N of the %.4g N it can (tension %.0f N)", D.L * 1000, v, wrap * 180 / geo::PI, pull, Fmax, g_pf_tension);
                        K.note = b;
                        if (pull > Fmax) {
                            K.slip = true;
                            snprintf(b, sizeof b, "the belt to %s slips: it must pull %.4g N but %.0f N of tension over %.0f deg of wrap (mu' %.2f) holds %.4g N; raise the tension to %.0f N%s",
                                     g_ac[(size_t)k].id.c_str(), pull, g_pf_tension, wrap * 180 / geo::PI, mue, Fmax, pull / std::max(1e-9, Fmax / std::max(1e-9, g_pf_tension - Fc)) + Fc + 1, mu < 0.2 ? " and keep oil off the belt" : "");
                            warn.push_back(b);
                            share = Fmax * rd * N.w;
                        }
                    }
                } else if (K.ptype == MT_ROLL && via) {
                    eta = 0.96;  // creep and rolling resistance
                    const EBody *ea = eb_of(c), *ebk = eb_of(k);
                    bool rubber = (ea && ea->M && std::string(ea->M->key) == "rubber") || (ebk && ebk->M && std::string(ebk->M->key) == "rubber");
                    double mut = rubber ? (lube || F.liquid ? 0.3 : 0.6) : (lube ? 0.06 : 0.15);  // traction coefficient
                    double rdrv = via->a.comp == c ? via->a.r : via->b.r;
                    double Tmax = mut * g_pf_preload * rdrv, Tneed = N.w > 0 ? share / N.w : 0;
                    if (Tneed > Tmax && rdrv > 0) {
                        K.slip = true;
                        snprintf(b, sizeof b, "%s slips on %s: it must pass %.3g N m but traction (mu %.2f x %.0f N preload x %.3g m) holds %.3g N m; raise the preload to %.0f N or use rubber",
                                 g_ac[(size_t)k].id.c_str(), g_ac[(size_t)c].id.c_str(), Tneed, mut, g_pf_preload, rdrv, Tmax, Tneed / (mut * rdrv) * 1.2);
                        warn.push_back(b);
                        share = Tmax * N.w;  // what it can pass; the rest is lost slipping
                    }
                }
                K.Pin = share * eta; K.Ledge = share * (1 - eta);
            }
        }
        // the report for this motor
        snprintf(b, sizeof b, "MOTOR on %s: %.0f W at %.1f rpm (%.4g N m)", g_ac[(size_t)mo.comp].id.c_str(), P0, w0 * 30 / geo::PI, w0 > 0 ? P0 / w0 : 0); line(b);
        line("PART                VIA        SPEED rpm  (m/s)   POWER IN W  BEARINGS W   FLUID W   PASSED ON LOSS W   OUT W     TORQUE N m (N)");
        double Pout = 0, L = 0;
        for (int c : order) {
            PFNode& N = g_pf[(size_t)c];
            const char* via = N.parent < 0 ? "motor" : N.ptype == MT_LOCK ? "lock" : N.ptype == MT_ROLL ? "roller" : N.ptype == MT_GEAR ? "gear" : N.ptype == MT_BELT ? (via_chain(c) ? "chain" : "belt") : "-";
            double tq = N.terminal ? (N.linear ? (N.v > 0 ? N.Pout / N.v : 0) : (N.w > 0 ? N.Pout / N.w : 0)) : (N.w > 0 ? N.Pin / N.w : 0);
            N.T = tq;
            snprintf(b, sizeof b, "%-18.18s  %-8s %10.4g %8s %12.4g %12.4g %9.4g %18.4g %8s  %10.4g%s%s", g_ac[(size_t)c].id.c_str(), via, N.w * 30 / geo::PI, N.linear ? fmtd(std::round(N.v * 1e4) / 1e4).c_str() : "",
                     N.Pin, N.Lb, N.Lf, N.Ledge, N.terminal ? fmtd(std::round(N.Pout * 1000) / 1000).c_str() : "", tq, N.slip ? "  SLIPS" : "", N.note.empty() ? "" : ("  " + N.note).c_str());
            line(b);
            L += N.Lb + N.Lf + N.Ledge; if (N.terminal) Pout += N.Pout;
        }
        snprintf(b, sizeof b, "  => output %.4g W of %.4g W: efficiency %.1f %%; %.4g W of losses become heat%s", Pout, P0, 100 * Pout / P0, P0 - Pout,
                 F.liquid ? (" in the " + F.name).c_str() : ""); line(b); line("");
        Pin_total += P0; Pout_total += Pout; Ltot += P0 - Pout;
    }
    if (g_amot.size() > 1) { snprintf(b, sizeof b, "ALL MOTORS: %.4g W in, %.4g W out, efficiency %.1f %%", Pin_total, Pout_total, Pin_total > 0 ? 100 * Pout_total / Pin_total : 0); line(b); line(""); }
    line("WARNINGS"); if (warn.empty()) line("  none"); for (auto& w : warn) line("  - " + w);
    line("");
    line("Speeds follow the gear teeth, the roller radii (2 % creep) and the rack's pitch radius. Losses: bearings mu N r under the");
    line("parts' apparent weight (rolling bearings mu 0.0015, bushings 0.08 lubricated / 0.15 dry, otherwise the material pair);");
    line("fluid drag on turning parts as discs (von Karman), the submerged share in the liquid; gear meshes 98.5 % spur/helical,");
    line("97 % bevel and rack when lubricated (96 / 94 % dry); rollers 96 %. V-belts 95 % with 1.5 % creep; they slip above");
    line("2 (T0 - m' v^2)(e^(mu' a) - 1) / (e^(mu' a) + 1) (mu' = mu / sin 19 deg, mu 0.25 dry, 0.10 oily; a the wrap). Roller chains");
    line("98 % lubricated (96 % dry), checked against ISO 606 breaking loads. Worm drives tan g / tan(g + phi') (mu from the sliding");
    line("speed, AGMA, steel on bronze), self-locking when the lead angle is under the friction angle. Power splits equally where a");
    line("part drives several.");
    g_pf_doc = g_asm_loaded;
    if (report && !g_selftest) {
        SYSTEMTIME t; GetLocalTime(&t);
        char nm[64]; snprintf(nm, sizeof nm, "power_%04d%02d%02d_%02d%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
        std::string dir = g_root + "\\results\\studio\\" + nm;
        make_dirs(dir);
        try { js::write_file(dir + "\\summary.txt", out); } catch (...) {}
        snap_view_png(dir + "\\00_view_3d.png"); open_simview(dir, "Power flow");
    }
    status("power flow: " + fmtd(std::round(Pout_total * 100) / 100) + " W out of " + fmtd(Pin_total) + " W in (" + fmtd(std::round(Pin_total > 0 ? 1000 * Pout_total / Pin_total : 0) / 10) + " %)" + (warn.empty() ? "" : "; " + tostr((int)warn.size()) + " warnings"));
    InvalidateRect(g_view, nullptr, FALSE);
}
static void power_setup() {
    if (!g_asm_on || g_amot.empty()) { power_flow(true); return; }
    std::vector<FField> f;
    for (const AMotor& m : g_amot) {
        std::string id = m.comp >= 0 && m.comp < (int)g_ac.size() ? g_ac[(size_t)m.comp].id : "?";
        f.push_back({"Motor on " + id + ": speed (rpm)", {}, fmtd(std::round(std::fabs(m.speed) / 6 * 100) / 100)});
        f.push_back({"Motor on " + id + ": input power (W)", {}, fmtd(m.power > 0 ? m.power : 100)});
    }
    f.push_back({"Roller drives: preload between the rollers (N)", {}, fmtd(g_pf_preload)});
    f.push_back({"Belt drives: static tension in each strand (N)", {}, fmtd(g_pf_tension)});
    std::string intro = "Give the motors power and see where it goes: the speed, torque and power of every part they drive, the losses on the way and the efficiency. The medium and its oil level come from Environment.";
    if (!form_dialog("Power flow", intro, f)) return;
    push_undo();
    for (size_t k = 0; k < g_amot.size(); ++k) {
        double rpm = atof(f[2 * k].value.c_str()); g_amot[k].speed = (g_amot[k].speed < 0 ? -1 : 1) * std::fabs(rpm) * 6;
        g_amot[k].power = std::max(0.0, atof(f[2 * k + 1].value.c_str()));
    }
    g_pf_preload = std::max(0.0, atof(f[f.size() - 2].value.c_str()));
    g_pf_tension = std::max(0.0, atof(f.back().value.c_str()));
    asm_store();
    power_flow(true);
}
static void power_overlay(int W, int H) {  // each moving part's speed, torque and power beside it
    (void)W; (void)H;
    if (g_pf.empty() || g_pf_doc != g_asm_loaded || g_pf.size() != g_ac.size()) return;
    for (size_t c = 0; c < g_ac.size(); ++c) {
        const PFNode& N = g_pf[c];
        if (!N.reached) continue;
        V3 p = comp_mid(g_ac[c]) + explode_offset((int)c);
        double wx, wy, wz;
        if (!gluProject(p.x, p.y, p.z, g_mv, g_pj, g_vp, &wx, &wy, &wz) || wz < 0 || wz > 1) continue;
        char b[160];
        if (N.linear) snprintf(b, sizeof b, "%s  %.3g m/s  %.3g N  %.3g W", g_ac[c].id.c_str(), N.v, N.T, N.terminal ? N.Pout : N.Pin);
        else snprintf(b, sizeof b, "%s  %.4g rpm  %.3g N m  %.3g W%s", g_ac[c].id.c_str(), N.w * 30 / geo::PI, N.T, N.terminal ? N.Pout : N.Pin, N.slip ? "  SLIPS" : "");
        float x = (float)wx + 10, y = (float)wy + 6, wlen = 7.0f * (float)strlen(b);
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(1, 1, 1, 0.8f); glBegin(GL_QUADS); glVertex2f(x - 4, y - 5); glVertex2f(x + wlen, y - 5); glVertex2f(x + wlen, y + 13); glVertex2f(x - 4, y + 13); glEnd();
        glDisable(GL_BLEND);
        glColor3f(N.slip ? 0.8f : 0.05f, N.slip ? 0.1f : 0.25f, N.slip ? 0.1f : 0.55f);
        glBegin(GL_LINES); glVertex2f((float)wx, (float)wy); glVertex2f(x - 4, y); glEnd();
        gl_text(x, y, b);
    }
}

// --power-run / --check-run: once everything is meshed (tests, reports)
static bool g_power_pending = false, g_check_pending = false;
static void power_run_tick() {
    if (!g_power_pending && !g_check_pending) { KillTimer(g_main, 43); return; }
    if (!g_asm_on || g_ac.empty()) return;
    for (const AComp& c : g_ac) if (!comp_ready(c)) return;
    if (g_check_pending) { g_check_pending = false; asm_check(); }
    if (g_power_pending) { g_power_pending = false; power_flow(true); }
    KillTimer(g_main, 43);
}

// the s-v-a diagram of every cam mate over one turn, with the cam's present angle (the motion study moves the marker)
static void cam_overlay(int W, int H) {
    if (!g_asm_on) return;
    int row = 0;
    for (const AMate& m : g_am) {
        if (m.type != MT_CAM || !m.on || m.a.comp < 0 || m.a.comp >= (int)g_ac.size()) continue;
        const js::Value* cm = g_ac[(size_t)m.a.comp].part.find("cam");
        if (!cm) continue;
        CamDef cd = cam_def(*cm);
        double w = 0;  // the cam's speed: its motor's, through locks
        for (const AMotor& mo : g_amot) if (!mo.linear) { bool lk = mo.comp == m.a.comp; for (const AMate& q : g_am) if (q.on && q.type == MT_LOCK && ((q.a.comp == mo.comp && q.b.comp == m.a.comp) || (q.b.comp == mo.comp && q.a.comp == m.a.comp))) lk = true; if (lk) w = std::fabs(mo.speed) * geo::PI / 180; }
        const int n = 360; const double hd = 2 * geo::PI / n;
        std::vector<double> s(n), v(n), a(n);
        double vm = 1e-12, am = 1e-12;
        for (int i = 0; i < n; ++i) { double th = i * hd; s[i] = cam_lift(cd, th); v[i] = (cam_lift(cd, th + 1e-4) - cam_lift(cd, th - 1e-4)) / 2e-4; a[i] = (cam_lift(cd, th + 1e-3) - 2 * s[i] + cam_lift(cd, th - 1e-3)) / 1e-6; vm = std::max(vm, std::fabs(v[i])); am = std::max(am, std::fabs(a[i])); }
        float pw = 330, ph = 120, x0 = (float)W - pw - 24, y0 = 96 + row * (ph + 34);
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(1, 1, 1, 0.86f); glBegin(GL_QUADS); glVertex2f(x0 - 8, y0 - 8); glVertex2f(x0 + pw + 8, y0 - 8); glVertex2f(x0 + pw + 8, y0 + ph + 26); glVertex2f(x0 - 8, y0 + ph + 26); glEnd();
        glColor4f(0.6f, 0.64f, 0.7f, 0.9f); glBegin(GL_LINES); glVertex2f(x0, y0 + ph / 2); glVertex2f(x0 + pw, y0 + ph / 2); glEnd();
        auto curve = [&](const std::vector<double>& c, double sc, float r, float g, float bl, bool from_base) {
            glColor3f(r, g, bl); glLineWidth(1.8f); glBegin(GL_LINE_STRIP);
            for (int i = 0; i < n; ++i) glVertex2f(x0 + pw * i / (n - 1), from_base ? y0 + (float)(ph * c[i] / sc) : y0 + ph / 2 + (float)(0.5 * ph * c[i] / sc));
            glEnd(); glLineWidth(1.0f);
        };
        curve(s, cd.h, 0.1f, 0.35f, 0.8f, true);
        curve(v, vm, 0.1f, 0.6f, 0.3f, false);
        curve(a, am, 0.85f, 0.3f, 0.1f, false);
        V3 dl = g_ac[(size_t)m.a.comp].pose.inv_rot(m.cdir);
        double th = std::atan2(dl.x, dl.y); th -= 2 * geo::PI * std::floor(th / (2 * geo::PI));
        float xm = x0 + (float)(pw * th / (2 * geo::PI));
        glColor3f(0.1f, 0.1f, 0.12f); glBegin(GL_LINES); glVertex2f(xm, y0 - 4); glVertex2f(xm, y0 + ph + 4); glEnd();
        glDisable(GL_BLEND);
        char b[200];
        snprintf(b, sizeof b, "%s %.0f rpm: lift %.1f mm, v %.3g m/s, a %.3g m/s2", g_ac[(size_t)m.a.comp].id.c_str(), w * 30 / geo::PI, cd.h * 1000, vm * w, am * w * w);
        glColor3f(0.1f, 0.12f, 0.2f); gl_text(x0 - 4, y0 + ph + 8, b);
        snprintf(b, sizeof b, "angle %.0f deg, lift %.2f mm   (blue s, green v, red a)", th * 180 / geo::PI, cam_lift(cd, th) * 1000);
        gl_text(x0 + 4, y0 - 4 + 0.0f * H, b);
        ++row;
    }
}
