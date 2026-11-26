// Assemblies: components (part designs, each meshed once and placed by a rigid pose), mates solved as rigid-body
// constraints, a library of ISO metric fasteners, smart insertion of a fastener into a hole, dragging, an animated
// exploded view and a motion study (rotary and linear motors, key poses, trajectory traces, a timeline).
//
// Document: {"name": ..., "assembly": {"components": [{"id", "part": {design} | "file": path, "pose": {"R": [9], "t": [3]},
// "fixed", "hidden", "appearance"}], "mates": [{"id", "type", "a": ref, "b": ref, "value", "aligned", "on"}],
// "motion": {"duration", "motors": [...], "keys": [...], "traces": [...]}}}. A reference is
// {"comp", "kind": "plane"|"axis"|"point", "p": [3], "d": [3], "r", "fp": [3], "fn": [3]} in the component's own frame
// (fp, fn: the seat or entry face of an insert, a point on the axis and the face's outward normal).
//
// Mouse in an assembly: click selects a component (a second click on a face, in mate mode, picks a reference);
// left drag orbits; Shift + left drag moves the component under the mouse (the mates follow); Ctrl + left drag turns it.

#include <map>
#include <memory>

#include "envmat.inc"

// ---------------------------------------------------------------- rigid poses
struct Pose {
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    V3 t;
    V3 apply(const V3& p) const { return V3(R[0] * p.x + R[1] * p.y + R[2] * p.z, R[3] * p.x + R[4] * p.y + R[5] * p.z, R[6] * p.x + R[7] * p.y + R[8] * p.z) + t; }
    V3 rot(const V3& p) const { return V3(R[0] * p.x + R[1] * p.y + R[2] * p.z, R[3] * p.x + R[4] * p.y + R[5] * p.z, R[6] * p.x + R[7] * p.y + R[8] * p.z); }
    V3 inv_rot(const V3& p) const { return V3(R[0] * p.x + R[3] * p.y + R[6] * p.z, R[1] * p.x + R[4] * p.y + R[7] * p.z, R[2] * p.x + R[5] * p.y + R[8] * p.z); }
    V3 inv(const V3& p) const { return inv_rot(p - t); }
};
static void rot_axis(const V3& a0, double ang, double M[9]) {  // Rodrigues
    double l = geo::len(a0); V3 a = l > 1e-15 ? a0 / l : V3(0, 0, 1);
    double c = std::cos(ang), s = std::sin(ang), C = 1 - c;
    M[0] = c + a.x * a.x * C; M[1] = a.x * a.y * C - a.z * s; M[2] = a.x * a.z * C + a.y * s;
    M[3] = a.y * a.x * C + a.z * s; M[4] = c + a.y * a.y * C; M[5] = a.y * a.z * C - a.x * s;
    M[6] = a.z * a.x * C - a.y * s; M[7] = a.z * a.y * C + a.x * s; M[8] = c + a.z * a.z * C;
}
static void mat_mul(const double A[9], const double B[9], double C[9]) {
    double T[9];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) T[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
    memcpy(C, T, sizeof T);
}
static V3 mat_vec(const double M[9], const V3& p) { return V3(M[0] * p.x + M[1] * p.y + M[2] * p.z, M[3] * p.x + M[4] * p.y + M[5] * p.z, M[6] * p.x + M[7] * p.y + M[8] * p.z); }
static void orthonormalise(double R[9]) {
    V3 x(R[0], R[3], R[6]), y(R[1], R[4], R[7]);
    double l = geo::len(x); x = l > 0 ? x / l : V3(1, 0, 0);
    y = y - x * geo::dot(x, y); l = geo::len(y); y = l > 0 ? y / l : V3(0, 1, 0);
    V3 z = geo::cross(x, y);
    R[0] = x.x; R[3] = x.y; R[6] = x.z; R[1] = y.x; R[4] = y.y; R[7] = y.z; R[2] = z.x; R[5] = z.y; R[8] = z.z;
}
// turn a pose by M about the world point c
static void pose_turn(Pose& P, const double M[9], const V3& c) {
    mat_mul(M, P.R, P.R);
    P.t = mat_vec(M, P.t - c) + c;
    orthonormalise(P.R);
}
// the smallest turn that takes unit vector u to v
static void rot_between(const V3& u, const V3& v, double M[9]) {
    V3 a = geo::cross(u, v);
    double s = geo::len(a), c = geo::clampd(geo::dot(u, v), -1, 1);
    if (s < 1e-12) {
        if (c > 0) { rot_axis(V3(0, 0, 1), 0, M); return; }
        V3 p = std::fabs(u.x) < 0.9 ? geo::cross(u, V3(1, 0, 0)) : geo::cross(u, V3(0, 1, 0));  // half a turn about any normal
        rot_axis(p, geo::PI, M); return;
    }
    rot_axis(a, std::atan2(s, c), M);
}
struct Quat { double w, x, y, z; };
static Quat quat_of(const double R[9]) {
    Quat q; double tr = R[0] + R[4] + R[8];
    if (tr > 0) { double s = std::sqrt(tr + 1) * 2; q.w = 0.25 * s; q.x = (R[7] - R[5]) / s; q.y = (R[2] - R[6]) / s; q.z = (R[3] - R[1]) / s; }
    else if (R[0] > R[4] && R[0] > R[8]) { double s = std::sqrt(1 + R[0] - R[4] - R[8]) * 2; q.w = (R[7] - R[5]) / s; q.x = 0.25 * s; q.y = (R[1] + R[3]) / s; q.z = (R[2] + R[6]) / s; }
    else if (R[4] > R[8]) { double s = std::sqrt(1 + R[4] - R[0] - R[8]) * 2; q.w = (R[2] - R[6]) / s; q.x = (R[1] + R[3]) / s; q.y = 0.25 * s; q.z = (R[5] + R[7]) / s; }
    else { double s = std::sqrt(1 + R[8] - R[0] - R[4]) * 2; q.w = (R[3] - R[1]) / s; q.x = (R[2] + R[6]) / s; q.y = (R[5] + R[7]) / s; q.z = 0.25 * s; }
    return q;
}
static void mat_of(const Quat& q0, double R[9]) {
    double n = std::sqrt(q0.w * q0.w + q0.x * q0.x + q0.y * q0.y + q0.z * q0.z);
    Quat q = {q0.w / n, q0.x / n, q0.y / n, q0.z / n};
    R[0] = 1 - 2 * (q.y * q.y + q.z * q.z); R[1] = 2 * (q.x * q.y - q.z * q.w); R[2] = 2 * (q.x * q.z + q.y * q.w);
    R[3] = 2 * (q.x * q.y + q.z * q.w); R[4] = 1 - 2 * (q.x * q.x + q.z * q.z); R[5] = 2 * (q.y * q.z - q.x * q.w);
    R[6] = 2 * (q.x * q.z - q.y * q.w); R[7] = 2 * (q.y * q.z + q.x * q.w); R[8] = 1 - 2 * (q.x * q.x + q.y * q.y);
}
static Pose pose_lerp(const Pose& a, const Pose& b, double s) {
    Quat p = quat_of(a.R), q = quat_of(b.R);
    double d = p.w * q.w + p.x * q.x + p.y * q.y + p.z * q.z;
    if (d < 0) { q = {-q.w, -q.x, -q.y, -q.z}; d = -d; }
    Quat r;
    if (d > 0.9995) r = {p.w + s * (q.w - p.w), p.x + s * (q.x - p.x), p.y + s * (q.y - p.y), p.z + s * (q.z - p.z)};
    else {
        double th = std::acos(d), sa = std::sin((1 - s) * th) / std::sin(th), sb = std::sin(s * th) / std::sin(th);
        r = {sa * p.w + sb * q.w, sa * p.x + sb * q.x, sa * p.y + sb * q.y, sa * p.z + sb * q.z};
    }
    Pose o; mat_of(r, o.R); o.t = a.t + (b.t - a.t) * s;
    return o;
}
static js::Value pose_json(const Pose& P) {
    js::Value o = js::Value::object(), r = js::Value::array(), t = js::Value::array();
    for (double v : P.R) r.push(js::Value(v));
    t.push(js::Value(P.t.x)); t.push(js::Value(P.t.y)); t.push(js::Value(P.t.z));
    o["R"] = r; o["t"] = t;
    return o;
}
static Pose pose_from(const js::Value* o) {
    Pose P;
    if (!o) return P;
    std::vector<double> r = o->nums("R"), t = o->nums("t");
    if (r.size() == 9) { for (int i = 0; i < 9; ++i) P.R[i] = r[(size_t)i]; orthonormalise(P.R); }
    if (t.size() == 3) P.t = V3(t[0], t[1], t[2]);
    return P;
}
static js::Value v3json(const V3& v) { js::Value a = js::Value::array(); a.push(js::Value(v.x)); a.push(js::Value(v.y)); a.push(js::Value(v.z)); return a; }
static V3 v3of(const js::Value& o, const char* k, V3 d = V3()) { std::vector<double> v = o.nums(k); return v.size() == 3 ? V3(v[0], v[1], v[2]) : d; }

// ---------------------------------------------------------------- component meshes (one per distinct part)
struct AMesh {
    geo::Mesh m;                 // crisp: split along creases
    std::vector<float> creases;  // edge lines
    geo::Box box;
    std::vector<int> vstart, vtris;  // triangles of each vertex (for face recognition)
    double h = 0, volume = 0;
    std::string error;
};
struct APiece {  // one part of a component, placed in the component's frame
    std::string key;  // the part's JSON text (the mesh cache key)
    js::Value part;
    Pose local;
    std::shared_ptr<AMesh> mesh;
};
struct AComp {
    std::string id, file, key;
    js::Value part;  // a part design, or an assembly (a rigid sub-assembly)
    std::vector<APiece> pieces;  // a part: one piece; a sub-assembly: every part inside it, at any depth
    Pose pose, base;  // base: the pose at time 0 of the motion study
    bool fixed = false, hidden = false;
    std::string appearance, material;  // material: a key of ENVMATS (envmat.inc); empty: from the part
};
enum { MT_COINCIDENT, MT_CONCENTRIC, MT_DISTANCE, MT_ANGLE, MT_PARALLEL, MT_PERP, MT_LOCK, MT_INSERT, MT_GEAR, MT_ROLL, MT_BELT, MT_CAM, NMT };
static const char* MATE_NAME[NMT] = {"coincident", "concentric", "distance", "angle", "parallel", "perpendicular", "lock", "insert", "gear", "roll", "belt", "cam"};
static const char* MATE_LABEL[NMT] = {"Coincident", "Concentric", "Distance", "Angle", "Parallel", "Perpendicular", "Lock", "Insert", "Gear", "Roller (friction drive)", "Belt or chain drive", "Cam and follower"};
struct ARef { int comp = -1; int kind = 2; V3 p, d = V3(0, 0, 1), fp, fn; double r = 0; bool face = false, hole = false; double value_ends[2] = {0, 0}; };  // kind 0 plane, 1 axis, 2 point; ends: the face's extent along the axis
struct AMate { std::string id; int type = MT_COINCIDENT; ARef a, b; double value = 0; bool aligned = false, on = true; Pose rel, rel2; int za = 0, zb = 0; double acc[2] = {0, 0}, last[2] = {0, 0}, sense = 0, rp = 0; int drive = 0; double bacc = 0, blast = 0; bool bsync = false; V3 cdir; double cax = 0; };  // cam: cdir the follower's line (world), cax its plane along the cam axis  // belt: drive 1 V-belt, 2 roller chain; bacc the driver's unwrapped turn (for drawing the belt)  // gear: sense +1 / -1 (b turns with / against a), rp the pinion's pitch radius of a rack pair (zb = 0)  // gear: acc the unwrapped turn of each gear since rel / rel2  // gear: rel, rel2 the two poses when it was made, za, zb the teeth
struct AMotor { int comp = -1; bool linear = false; V3 p, d = V3(0, 0, 1); double speed = 90, power = 0; };  // power: input W (power flow)  // p, d in the component's frame; deg/s or m/s
struct AKey { double t = 0; std::vector<int> comps; std::vector<Pose> poses; };
struct ATrace { int comp = -1; V3 p; std::vector<V3> path; };

static std::vector<AComp> g_ac;
static std::vector<AMate> g_am;
static std::vector<AMotor> g_amot;
static std::vector<AKey> g_akeys;
static std::vector<ATrace> g_atr;
static std::map<std::string, std::shared_ptr<AMesh>> g_amesh;
static std::map<std::string, int> g_amesh_pending;
static std::string g_asm_loaded;  // the assembly JSON last loaded (undo, open)
static bool g_asm_fitted = false;  // the view fitted once the components are meshed
static int g_asm_sel = -1, g_asm_mate_sel = -1;  // selected component, selected mate
static double g_explode = 0, g_explode_to = 0;     // 0 assembled, 1 exploded
static double g_ms_t = 0, g_ms_T = 4.0;            // motion study time and duration (s)
static double g_pf_tension = 250;                  // belt drives: the static tension in each strand (N), for power flow
static double g_pf_preload = 50;                   // roller drives: the preload between the rollers (N), for power flow
static bool g_ms_play = false, g_ms_based = false;
static int g_asm_pick = 0;                         // 1-2 mate references, 3-4 insert, 5 trace point, 6 motor axis, 7-8 gear mate, 9-10 roller mate
static ARef g_pick_ref[2];
static std::string g_st_last_note;  // a remark on the last mate (gear checks)
static int g_asm_drag = 0, g_asm_drag_comp = -1;   // 1 move, 2 turn
static V3 g_asm_drag_anchor;
enum { WM_ASM_MESHED = WM_APP + 41, ASM_TIMER = 41 };

static void asm_store();
static void asm_solve(const std::vector<char>* driven = nullptr);
static void gear_resync(bool from_base);
static void asm_check();
static void power_setup();
static void flow_setup();
static void flow_stop(bool keep_view = true);
static void flow_refine();
static void flow_demo_cylinder();
static void flow_demo_tank();
static void flow_cycle_view();
static void flow_cycle_mode();
static void flow_cycle_walls();
static void flow_cycle_surface();
static void flow_demo_splash();
static void flow_demo_gears_oil();
static void flow_demo_belt_fluid();

// ---------------------------------------------------------------- components made of pieces
static std::string dir_of(const std::string& p) { size_t k = p.find_last_of("\\/"); return k == std::string::npos ? std::string(".") : p.substr(0, k); }
static std::string resolve_file(const std::string& f);
static void add_pieces(std::vector<APiece>& out, const js::Value& part, const Pose& at, const std::string& dir, int depth) {
    const js::Value* a = part.find("assembly");
    if (!a || depth > 5) {
        APiece pc; pc.part = part.clone();
        if (!pc.part.find("features")) pc.part["features"] = js::Value::array();
        pc.key = js::dump(pc.part, 0); pc.local = at;
        out.push_back(pc);
        return;
    }
    const js::Value* cs = a->find("components");
    if (!cs) return;
    for (size_t i = 0; i < cs->size(); ++i) {  // the parts of a sub-assembly where its own mates left them (rigid)
        const js::Value& o = cs->at(i);
        if (o.flag("hidden", false)) continue;
        js::Value sub; std::string f = o.str("file", ""), sdir = dir;
        if (const js::Value* pp = o.find("part")) sub = pp->clone();
        else if (!f.empty()) {
            std::string path = f;
            if (path.size() < 2 || (path[1] != ':' && path[0] != '\\' && path[0] != '/')) path = dir + "\\" + f;
            try { sub = js::parse(js::read_file(path)); } catch (...) { continue; }
            sdir = dir_of(path);
        } else continue;
        Pose lp = pose_from(o.find("pose")), comb;
        mat_mul(at.R, lp.R, comb.R); comb.t = at.apply(lp.t);
        add_pieces(out, sub, comb, sdir, depth + 1);
    }
}
static void build_pieces(AComp& c) {
    c.pieces.clear();
    std::string dir = !c.file.empty() ? dir_of(resolve_file(c.file)) : g_path.empty() ? std::string(".") : dir_of(g_path);
    add_pieces(c.pieces, c.part, Pose(), dir, 0);
    for (APiece& pc : c.pieces) { auto it = g_amesh.find(pc.key); if (it != g_amesh.end()) pc.mesh = it->second; }
}
static std::string resolve_file(const std::string& f) {  // a component file named relative to the assembly's folder
    if (f.size() >= 2 && (f[1] == ':' || f[0] == '\\' || f[0] == '/')) return f;
    return (g_path.empty() ? std::string(".") : dir_of(g_path)) + "\\" + f;
}
static bool is_sub(const AComp& c) { return c.part.find("assembly") != nullptr; }
static bool comp_ready(const AComp& c) { if (c.pieces.empty()) return false; for (const APiece& pc : c.pieces) if (!pc.mesh) return false; return true; }
static geo::Box comp_box(const AComp& c) {  // in the component's frame
    geo::Box b;
    for (const APiece& pc : c.pieces)
        if (pc.mesh && pc.mesh->box.valid())
            for (int k = 0; k < 8; ++k) b.add(pc.local.apply(V3(k & 1 ? pc.mesh->box.hi.x : pc.mesh->box.lo.x, k & 2 ? pc.mesh->box.hi.y : pc.mesh->box.lo.y, k & 4 ? pc.mesh->box.hi.z : pc.mesh->box.lo.z)));
    return b;
}
static double comp_size(const AComp& c) { geo::Box b = comp_box(c); return b.valid() ? geo::len(b.hi - b.lo) : 0.05; }
static V3 comp_mid(const AComp& c) { geo::Box b = comp_box(c); return c.pose.apply(b.valid() ? (b.lo + b.hi) * 0.5 : V3()); }
static void comp_world_box(const AComp& c, geo::Box& w) {
    geo::Box b = comp_box(c);
    if (b.valid()) for (int k = 0; k < 8; ++k) w.add(c.pose.apply(V3(k & 1 ? b.hi.x : b.lo.x, k & 2 ? b.hi.y : b.lo.y, k & 4 ? b.hi.z : b.lo.z)));
}

// part design -> crisp mesh, in a worker thread
struct AMeshJob { std::string key; };
static unsigned __stdcall asm_mesh_thread(void* p) {
    std::unique_ptr<AMeshJob> job((AMeshJob*)p);
    AMesh* r = new AMesh();
    try {
        geo::Model m; m.load(js::parse(job->key));
        if (!m.errors.empty()) r->error = m.errors[0];
        if (m.bounds.valid()) {
            V3 e = m.bounds.hi - m.bounds.lo;
            double fine = js::parse(job->key).find("gear") ? 220.0 : 170.0;  // gear teeth need a finer grid
            double diag = geo::len(e), h = std::max(diag / fine, std::cbrt((e.x + 1e-3) * (e.y + 1e-3) * (e.z + 1e-3) / 3e6));
            geo::Grid g = geo::sample(m, h);
            r->m = geo::surface_nets(m, g, 1);
            r->h = h;
            geo::Props P = geo::properties(r->m, g); r->volume = P.volume;
            r->creases = crease_lines(r->m, 38.0, false);
            geo::split_sharp(r->m, 30.0);
            for (const V3& v : r->m.v) r->box.add(v);
            size_t nv = r->m.v.size(), nt = r->m.t.size() / 3;
            r->vstart.assign(nv + 1, 0); r->vtris.resize(3 * nt);
            for (int v : r->m.t) ++r->vstart[(size_t)v + 1];
            for (size_t i = 0; i < nv; ++i) r->vstart[i + 1] += r->vstart[i];
            std::vector<int> fill(r->vstart.begin(), r->vstart.end() - 1);
            for (size_t t = 0; t < nt; ++t) for (int k = 0; k < 3; ++k) r->vtris[(size_t)fill[(size_t)r->m.t[3 * t + (size_t)k]]++] = (int)t;
        }
    } catch (std::exception& e) { r->error = e.what(); }
    std::string* key = new std::string(job->key);
    PostMessageA(g_main, WM_ASM_MESHED, (WPARAM)key, (LPARAM)r);
    return 0;
}
static void asm_request_meshes() {
    for (AComp& c : g_ac) for (APiece& pc : c.pieces) {
        if (pc.mesh) continue;
        auto it = g_amesh.find(pc.key);
        if (it != g_amesh.end()) { pc.mesh = it->second; continue; }
        if (g_amesh_pending.count(pc.key)) continue;
        g_amesh_pending[pc.key] = 1;
        AMeshJob* j = new AMeshJob{pc.key};
        HANDLE t = (HANDLE)_beginthreadex(nullptr, 0, asm_mesh_thread, j, 0, nullptr);
        if (t) CloseHandle(t);
    }
}
static void asm_meshed(WPARAM w, LPARAM l) {
    std::unique_ptr<std::string> key((std::string*)w);
    std::shared_ptr<AMesh> m((AMesh*)l);
    g_amesh_pending.erase(*key);
    g_amesh[*key] = m;
    for (AComp& c : g_ac) for (APiece& pc : c.pieces) if (pc.key == *key) pc.mesh = m;
    int left = 0; for (AComp& c : g_ac) if (!comp_ready(c)) ++left;
    if (!left && !g_asm_fitted) { g_asm_fitted = true; asm_fit(); }
    status(left ? "meshing components... " + tostr(left) + " left" : "assembly: " + tostr((int)g_ac.size()) + " components, " + tostr((int)g_am.size()) + " mates" + (m->error.empty() ? "" : "   (" + m->error + ")"), 1);
    InvalidateRect(g_view, nullptr, FALSE);
}

// ---------------------------------------------------------------- the document
static ARef ref_from(const js::Value* o) {
    ARef r;
    if (!o) return r;
    r.comp = (int)o->num("comp", -1);
    std::string k = o->str("kind", "point");
    r.kind = k == "plane" ? 0 : k == "axis" ? 1 : 2;
    r.p = v3of(*o, "p"); r.d = v3of(*o, "d", V3(0, 0, 1)); r.r = o->num("r", 0);
    if (o->find("fp")) { r.face = true; r.fp = v3of(*o, "fp"); r.fn = v3of(*o, "fn", V3(0, 0, 1)); }
    std::vector<double> e = o->nums("ends"); if (e.size() == 2) { r.value_ends[0] = e[0]; r.value_ends[1] = e[1]; }
    r.hole = o->flag("hole", false);
    return r;
}
static js::Value ref_json(const ARef& r) {
    js::Value o = js::Value::object();
    o["comp"] = (double)r.comp; o["kind"] = r.kind == 0 ? "plane" : r.kind == 1 ? "axis" : "point";
    o["p"] = v3json(r.p); o["d"] = v3json(r.d);
    if (r.r > 0) o["r"] = r.r;
    if (r.face) { o["fp"] = v3json(r.fp); o["fn"] = v3json(r.fn); }
    if (r.value_ends[1] > r.value_ends[0]) { js::Value e = js::Value::array(); e.push(js::Value(r.value_ends[0])); e.push(js::Value(r.value_ends[1])); o["ends"] = e; }
    if (r.hole) o["hole"] = true;
    return o;
}
static void asm_from_doc() {
    const js::Value* a = g_doc.find("assembly");
    g_asm_on = a != nullptr;
    if (!a) { g_asm_loaded.clear(); return; }
    std::string text = js::dump(*a, 0);
    if (text == g_asm_loaded) return;
    g_asm_loaded = text;
    g_ac.clear(); g_am.clear(); g_amot.clear(); g_akeys.clear(); g_atr.clear();
    if (const js::Value* cs = a->find("components"))
        for (size_t i = 0; i < cs->size(); ++i) {
            const js::Value& o = cs->at(i);
            AComp c;
            c.id = o.str("id", "part" + tostr((int)i + 1));
            c.file = o.str("file", "");
            if (const js::Value* p = o.find("part")) c.part = p->clone();
            else if (!c.file.empty()) { try { c.part = js::parse(js::read_file(resolve_file(c.file))); } catch (...) { c.part = js::Value::object(); status("cannot read component file " + resolve_file(c.file)); } }
            else c.part = js::Value::object();
            if (!c.part.find("features")) c.part["features"] = js::Value::array();
            c.key = js::dump(c.part, 0);
            build_pieces(c);
            c.pose = c.base = pose_from(o.find("pose"));
            c.fixed = o.flag("fixed", false); c.hidden = o.flag("hidden", false);
            c.appearance = o.str("appearance", c.part.find("fastener") ? "steel" : "");
            c.material = o.str("material", "");
            g_ac.push_back(c);
        }
    if (const js::Value* ms = a->find("mates"))
        for (size_t i = 0; i < ms->size(); ++i) {
            const js::Value& o = ms->at(i);
            AMate m; m.id = o.str("id", "mate" + tostr((int)i + 1));
            std::string ty = o.str("type", "coincident");
            for (int k = 0; k < NMT; ++k) if (ty == MATE_NAME[k]) m.type = k;
            m.a = ref_from(o.find("a")); m.b = ref_from(o.find("b"));
            m.value = o.num("value", 0); m.aligned = o.flag("aligned", false); m.on = o.flag("on", true);
            m.rel = pose_from(o.find("rel")); m.rel2 = pose_from(o.find("rel2"));
            std::vector<double> zz = o.nums("teeth"); if (zz.size() == 2) { m.za = (int)zz[0]; m.zb = (int)zz[1]; }
            m.sense = o.num("sense", 0); m.rp = o.num("rp", 0);
            if (m.type == MT_BELT) m.drive = o.str("drive", "belt") == "chain" ? 2 : 1;
            if (m.type == MT_CAM) { m.cdir = v3of(o, "dir", V3(0, 0, 1)); m.cax = o.num("axial", 0); }
            g_am.push_back(m);
        }
    if (const js::Value* mo = a->find("motion")) {
        g_ms_T = std::max(0.5, mo->num("duration", 4)); g_pf_preload = mo->num("roller_preload_N", 50); g_pf_tension = mo->num("belt_tension_N", 250);
        if (const js::Value* ms = mo->find("motors"))
            for (size_t i = 0; i < ms->size(); ++i) {
                const js::Value& o = ms->at(i); AMotor m;
                m.comp = (int)o.num("comp", -1); m.linear = o.str("kind", "rotary") == "linear";
                m.p = v3of(o, "p"); m.d = v3of(o, "d", V3(0, 0, 1)); m.speed = o.num("speed", 90); m.power = o.num("power", 0);
                g_amot.push_back(m);
            }
        if (const js::Value* ks = mo->find("keys"))
            for (size_t i = 0; i < ks->size(); ++i) {
                const js::Value& o = ks->at(i); AKey k; k.t = o.num("t", 0);
                if (const js::Value* ps = o.find("poses"))
                    for (size_t j = 0; j < ps->size(); ++j) { k.comps.push_back((int)ps->at(j).num("comp", -1)); k.poses.push_back(pose_from(&ps->at(j))); }
                g_akeys.push_back(k);
            }
        if (const js::Value* ts = mo->find("traces"))
            for (size_t i = 0; i < ts->size(); ++i) { ATrace t; t.comp = (int)ts->at(i).num("comp", -1); t.p = v3of(ts->at(i), "p"); g_atr.push_back(t); }
    }
    if (g_asm_sel >= (int)g_ac.size()) g_asm_sel = -1;
    if (g_asm_mate_sel >= (int)g_am.size()) g_asm_mate_sel = -1;
    g_ms_t = 0; g_ms_play = false; g_ms_based = false; g_explode = g_explode_to = 0;
    g_asm_fitted = false;
    gear_resync(false);
    asm_request_meshes();
}
static void asm_store() {  // the live assembly into the document (save, undo)
    if (!g_asm_on) return;
    js::Value a = js::Value::object(), cs = js::Value::array(), ms = js::Value::array();
    for (const AComp& c : g_ac) {
        js::Value o = js::Value::object();
        o["id"] = c.id;
        if (!c.file.empty()) o["file"] = c.file; else o["part"] = c.part.clone();
        o["pose"] = pose_json(g_ms_based ? c.base : c.pose);
        if (c.fixed) o["fixed"] = true;
        if (c.hidden) o["hidden"] = true;
        if (!c.appearance.empty()) o["appearance"] = c.appearance;
        if (!c.material.empty()) o["material"] = c.material;
        cs.push(o);
    }
    for (const AMate& m : g_am) {
        js::Value o = js::Value::object();
        o["id"] = m.id; o["type"] = MATE_NAME[m.type]; o["a"] = ref_json(m.a); o["b"] = ref_json(m.b);
        if (m.type == MT_DISTANCE || m.type == MT_ANGLE || m.type == MT_ROLL || m.type == MT_BELT) o["value"] = m.value;
        if (m.aligned) o["aligned"] = true;
        if (!m.on) o["on"] = false;
        if (m.type == MT_LOCK || m.type == MT_GEAR || m.type == MT_ROLL || m.type == MT_BELT) o["rel"] = pose_json(m.rel);
        if (m.type == MT_BELT) o["drive"] = m.drive == 2 ? "chain" : "belt";
        if (m.type == MT_CAM) { o["dir"] = v3json(m.cdir); o["axial"] = m.cax; o["rel2"] = pose_json(m.rel2); }
        if (m.type == MT_ROLL || m.type == MT_BELT) { o["rel2"] = pose_json(m.rel2); o["sense"] = m.sense; }
        if (m.type == MT_GEAR) { o["rel2"] = pose_json(m.rel2); o["sense"] = m.sense; if (m.rp > 0) o["rp"] = m.rp; js::Value z = js::Value::array(); z.push(js::Value((double)m.za)); z.push(js::Value((double)m.zb)); o["teeth"] = z; }
        ms.push(o);
    }
    a["components"] = cs; a["mates"] = ms;
    js::Value mo = js::Value::object(), mt = js::Value::array(), ks = js::Value::array(), ts = js::Value::array();
    mo["duration"] = g_ms_T; mo["roller_preload_N"] = g_pf_preload; mo["belt_tension_N"] = g_pf_tension;
    for (const AMotor& m : g_amot) { js::Value o = js::Value::object(); o["comp"] = (double)m.comp; o["kind"] = m.linear ? "linear" : "rotary"; o["p"] = v3json(m.p); o["d"] = v3json(m.d); o["speed"] = m.speed; if (m.power > 0) o["power"] = m.power; mt.push(o); }
    for (const AKey& k : g_akeys) {
        js::Value o = js::Value::object(), ps = js::Value::array();
        o["t"] = k.t;
        for (size_t j = 0; j < k.comps.size(); ++j) { js::Value p = pose_json(k.poses[j]); p["comp"] = (double)k.comps[j]; ps.push(p); }
        o["poses"] = ps; ks.push(o);
    }
    for (const ATrace& t : g_atr) { js::Value o = js::Value::object(); o["comp"] = (double)t.comp; o["p"] = v3json(t.p); ts.push(o); }
    mo["motors"] = mt; mo["keys"] = ks; mo["traces"] = ts;
    a["motion"] = mo;
    g_doc["assembly"] = a;
    g_asm_loaded = js::dump(a, 0);
}
static void asm_changed(bool undoable = true) {  // after an edit: undo point, document, tree, view
    if (undoable) push_undo();
    asm_store();
    g_modified = true; update_title();
    asm_request_meshes();
    fill_list();
    InvalidateRect(g_view, nullptr, FALSE);
}

// ---------------------------------------------------------------- the fastener library (ISO metric, sizes in mm)
struct Thread { const char* name; double d, p, s, k, m, dw, hw; };
static const Thread THREADS[] = {
    {"M3", 3, 0.5, 5.5, 2, 2.4, 7, 0.5}, {"M4", 4, 0.7, 7, 2.8, 3.2, 9, 0.8}, {"M5", 5, 0.8, 8, 3.5, 4.7, 10, 1},
    {"M6", 6, 1.0, 10, 4, 5.2, 12, 1.6}, {"M8", 8, 1.25, 13, 5.3, 6.8, 16, 1.6}, {"M10", 10, 1.5, 16, 6.4, 8.4, 20, 2},
    {"M12", 12, 1.75, 18, 7.5, 10.8, 24, 2.5}, {"M16", 16, 2, 24, 10, 14.8, 30, 3}, {"M20", 20, 2.5, 30, 12.5, 18, 37, 3},
};
static const int NTHREAD = (int)(sizeof(THREADS) / sizeof(THREADS[0]));
static js::Value hex_outline(double s) {  // a hexagon, across flats s
    js::Value o = js::Value::array();
    double R = s / std::sqrt(3.0);
    for (int i = 0; i < 6; ++i) { js::Value p = js::Value::array(); double a = geo::PI / 6 + i * geo::PI / 3; p.push(js::Value(R * std::cos(a))); p.push(js::Value(R * std::sin(a))); o.push(p); }
    return o;
}
static js::Value prim(const char* type, const char* op, std::initializer_list<std::pair<const char*, double>> nums, V3 at) {
    js::Value f = js::Value::object();
    f["id"] = type; f["type"] = type; f["op"] = op; f["mirror"] = false;
    for (auto& kv : nums) f[kv.first] = kv.second;
    f["translate"] = v3json(at);
    return f;
}
// every fastener: axis local z, the seat (the face that bears on the joined part) at z = 0 with outward normal -z;
// what goes into a hole lies along -z, the head, nut or washer along +z
static js::Value fastener_part(const std::string& kind, const Thread& T, double Lmm) {
    const double u = 0.001;
    double d = T.d * u, L = Lmm * u;
    js::Value part = js::Value::object(), fl = js::Value::array();
    std::string name;
    if (kind == "bolt") {  // ISO 4017 hex head screw
        js::Value head = prim("extrude", "add", {{"height", T.k * u}}, V3(0, 0, 0.5 * T.k * u)); head["outline"] = hex_outline(T.s * u); head["smooth"] = false; head["id"] = "head";
        js::Value cham = prim("cone", "intersect", {{"r1", 0.5 * T.s * u * 1.12 + T.k * u}, {"r2", 0.5 * T.s * u * 1.12 - 0.15 * T.k * u}, {"length", T.k * u * 1.15}}, V3(0, 0, 0.5 * T.k * u)); cham["axis"] = "z"; cham["id"] = "head_chamfer";
        js::Value shank = prim("cylinder", "add", {{"r", 0.5 * d}, {"length", L}}, V3(0, 0, -0.5 * L)); shank["axis"] = "z"; shank["id"] = "shank";
        fl.push(head); fl.push(shank);
        (void)cham;
        name = std::string(T.name) + "x" + fmtd(Lmm) + " hex bolt";
    } else if (kind == "screw") {  // ISO 4762 socket head cap screw
        double dk = 1.5 * T.d * u, k = T.d * u, sk = 0.75 * T.d * u;
        js::Value head = prim("cylinder", "add", {{"r", 0.5 * dk}, {"length", k}}, V3(0, 0, 0.5 * k)); head["axis"] = "z"; head["id"] = "head";
        js::Value sock = prim("extrude", "cut", {{"height", 1.2 * k}}, V3(0, 0, k)); sock["outline"] = hex_outline(sk); sock["smooth"] = false; sock["id"] = "socket";
        js::Value shank = prim("cylinder", "add", {{"r", 0.5 * d}, {"length", L}}, V3(0, 0, -0.5 * L)); shank["axis"] = "z"; shank["id"] = "shank";
        fl.push(head); fl.push(shank); fl.push(sock);
        name = std::string(T.name) + "x" + fmtd(Lmm) + " socket head cap screw";
    } else if (kind == "nut") {  // ISO 4032
        js::Value body = prim("extrude", "add", {{"height", T.m * u}}, V3(0, 0, 0.5 * T.m * u)); body["outline"] = hex_outline(T.s * u); body["smooth"] = false; body["id"] = "body";
        js::Value hole = prim("cylinder", "cut", {{"r", 0.5 * d}, {"length", 3 * T.m * u}}, V3(0, 0, 0.5 * T.m * u)); hole["axis"] = "z"; hole["id"] = "thread";
        fl.push(body); fl.push(hole);
        name = std::string(T.name) + " hex nut";
    } else if (kind == "washer") {  // ISO 7089
        js::Value body = prim("cylinder", "add", {{"r", 0.5 * T.dw * u}, {"length", T.hw * u}}, V3(0, 0, 0.5 * T.hw * u)); body["axis"] = "z"; body["id"] = "body";
        js::Value hole = prim("cylinder", "cut", {{"r", 0.54 * d}, {"length", 4 * T.hw * u}}, V3(0, 0, 0.5 * T.hw * u)); hole["axis"] = "z"; hole["id"] = "bore";
        fl.push(body); fl.push(hole);
        name = std::string(T.name) + " washer";
    } else if (kind == "pin") {  // dowel pin: 40 % stands out of the hole
        js::Value body = prim("cylinder", "add", {{"r", 0.5 * d}, {"length", L}}, V3(0, 0, -0.1 * L)); body["axis"] = "z"; body["id"] = "pin";
        fl.push(body);
        name = std::string(T.name) + "x" + fmtd(Lmm) + " dowel pin";
    } else {  // shaft
        js::Value body = prim("cylinder", "add", {{"r", 0.5 * d}, {"length", L}}, V3(0, 0, -0.5 * L)); body["axis"] = "z"; body["id"] = "shaft";
        fl.push(body);
        name = "shaft " + fmtd(T.d) + " x " + fmtd(Lmm);
    }
    part["name"] = name; part["units"] = "m"; part["features"] = fl;
    js::Value fm = js::Value::object();
    fm["kind"] = kind; fm["size"] = T.name; fm["d"] = d; fm["pitch"] = T.p * u; fm["length"] = L;
    fm["thread_from"] = kind == "bolt" || kind == "screw" ? -L : 0.0;
    fm["thread_to"] = kind == "bolt" || kind == "screw" ? -L + std::min(L, 2 * d + 0.006) : 0.0;
    part["fastener"] = fm;
    return part;
}

// ---------------------------------------------------------------- involute spur gears (20 deg pressure angle)
// axis local z, the gear centred on z = 0, tooth 0 on +x. Sizes in metres.
// helix in degrees (+ right hand, - left hand), cone the pitch cone angle of a bevel gear (degrees, 0 for none); for a
// helical gear the module is the normal module; for a bevel gear it is taken at the large end (local z = -width / 2)
static js::Value gear_part(double mod, int z, double width, double bore, double helix = 0, double cone = 0) {
    double mt = mod / std::cos(helix * geo::PI / 180), r = 0.5 * mt * z, ra = r + mod;
    js::Value part = js::Value::object(), fl = js::Value::array();
    js::Value body = js::Value::object();
    body["id"] = "teeth"; body["type"] = "gear"; body["op"] = "add"; body["mirror"] = false;
    body["module"] = mod; body["teeth"] = (double)z; body["width"] = width; body["pressure_angle"] = 20.0; body["translate"] = v3json(V3());
    if (helix != 0) body["helix"] = helix;
    if (cone > 0) body["cone"] = cone;
    fl.push(body);
    if (bore > 0) { js::Value b = prim("cylinder", "cut", {{"r", 0.5 * bore}, {"length", 3 * width}}, V3()); b["axis"] = "z"; b["id"] = "bore"; fl.push(b); }
    std::string kind = cone > 0 ? "bevel" : helix != 0 ? "helical" : "spur";
    part["name"] = kind + " gear m" + fmtd(mod * 1000) + " z" + tostr(z) + (helix != 0 ? " " + fmtd(std::fabs(helix)) + (helix > 0 ? " deg RH" : " deg LH") : "") + (cone > 0 ? " cone " + fmtd(cone) + " deg" : "");
    part["units"] = "m"; part["features"] = fl;
    js::Value gm = js::Value::object();
    gm["kind"] = kind; gm["module"] = mod; gm["teeth"] = (double)z; gm["pressure_angle"] = 20.0; gm["helix"] = helix; gm["cone"] = cone;
    gm["width"] = width; gm["pitch_radius"] = r; gm["tip_radius"] = ra; gm["bore"] = bore;
    part["gear"] = gm;
    return part;
}
// a straight rack: teeth along local x (centred on x = 0), tips towards +y, the pitch line at y = 0, the face width
// along z (centred); body: the bar's depth under the roots
static js::Value rack_part(double mod, int n, double width, double body) {
    js::Value part = js::Value::object(), fl = js::Value::array();
    js::Value f = js::Value::object();
    f["id"] = "rack"; f["type"] = "rack"; f["op"] = "add"; f["mirror"] = false;
    f["module"] = mod; f["teeth"] = (double)n; f["width"] = width; f["height"] = body; f["pressure_angle"] = 20.0; f["translate"] = v3json(V3());
    fl.push(f);
    part["name"] = "rack m" + fmtd(mod * 1000) + " " + tostr(n) + " teeth"; part["units"] = "m"; part["features"] = fl;
    js::Value gm = js::Value::object();
    gm["kind"] = "rack"; gm["module"] = mod; gm["teeth"] = (double)n; gm["pitch"] = geo::PI * mod; gm["length"] = n * geo::PI * mod; gm["width"] = width; gm["height"] = body;
    part["gear"] = gm;
    return part;
}
// ---------------------------------------------------------------- drive parts (axis local z, centred on z = 0)
struct BearingDef { const char* name; double d, D, B; };  // ISO 15 deep groove ball bearings, mm
static const BearingDef BEARINGS[] = {
    {"6000", 10, 26, 8}, {"6001", 12, 28, 8}, {"6002", 15, 32, 9}, {"6003", 17, 35, 10}, {"6004", 20, 42, 12}, {"6005", 25, 47, 12},
    {"6006", 30, 55, 13}, {"6007", 35, 62, 14}, {"6008", 40, 68, 15}, {"6010", 50, 80, 16},
    {"6200", 10, 30, 9}, {"6201", 12, 32, 10}, {"6202", 15, 35, 11}, {"6203", 17, 40, 12}, {"6204", 20, 47, 14}, {"6205", 25, 52, 15}, {"6206", 30, 62, 16},
};
static const int NBEARING = (int)(sizeof(BEARINGS) / sizeof(BEARINGS[0]));
struct KeyDef { double dmin, dmax, b, h, t1; };  // DIN 6885 parallel keys, mm
static const KeyDef KEYS[] = {{6, 8, 2, 2, 1.2}, {8, 10, 3, 3, 1.8}, {10, 12, 4, 4, 2.5}, {12, 17, 5, 5, 3}, {17, 22, 6, 6, 3.5}, {22, 30, 8, 7, 4}, {30, 38, 10, 8, 5}, {38, 44, 12, 8, 5}, {44, 50, 14, 9, 5.5}, {50, 58, 16, 10, 6}};
static const KeyDef& key_for(double dmm) { for (const KeyDef& k : KEYS) if (dmm > k.dmin - 1e-9 && dmm <= k.dmax + 1e-9) return k; return dmm < 6 ? KEYS[0] : KEYS[sizeof(KEYS) / sizeof(KEYS[0]) - 1]; }
static js::Value cyl_f(const char* id, const char* op, double r, double L, V3 at, const char* axis = "z") { js::Value f = prim("cylinder", op, {{"r", r}, {"length", L}}, at); f["axis"] = axis; f["id"] = id; return f; }
static js::Value box_f(const char* id, const char* op, V3 size, V3 at) { js::Value f = js::Value::object(); f["id"] = id; f["type"] = "box"; f["op"] = op; f["size"] = v3json(size); f["round"] = 0.0; f["translate"] = v3json(at); f["mirror"] = false; return f; }
static js::Value make_part(const std::string& name, js::Value fl, const char* meta, js::Value md, const char* material) {
    js::Value p = js::Value::object(); p["name"] = name; p["units"] = "m"; p["features"] = fl; p[meta] = md; if (material) p["material"] = material; return p;
}
// a deep groove ball bearing: outer ring, inner ring, a ring of balls between them
static js::Value bearing_part(const BearingDef& B) {
    const double u = 0.001;
    double d = B.d * u, D = B.D * u, w = B.B * u, h = 0.5 * (D - d), roi = 0.5 * D - 0.3 * h, rii = 0.5 * d + 0.3 * h, rb = 0.3 * h, rp = 0.5 * (roi + rii);
    js::Value fl = js::Value::array();
    fl.push(cyl_f("outer", "add", 0.5 * D, w, V3()));
    fl.push(cyl_f("outer_bore", "cut", roi, 1.2 * w, V3()));
    fl.push(cyl_f("inner", "add", rii, w, V3()));
    int n = std::max(6, (int)(geo::PI * rp / (1.18 * rb)));
    for (int k = 0; k < n; ++k) { double a = 2 * geo::PI * k / n; js::Value s = prim("sphere", "add", {{"r", rb * 1.02}}, V3(rp * std::cos(a), rp * std::sin(a), 0)); s["id"] = "ball" + tostr(k + 1); fl.push(s); }
    fl.push(cyl_f("bore", "cut", 0.5 * d, 1.5 * w, V3()));
    js::Value md = js::Value::object(); md["designation"] = B.name; md["d"] = d; md["D"] = D; md["B"] = w; md["balls"] = (double)n;
    return make_part(std::string("ball bearing ") + B.name + " (" + fmtd(B.d) + " x " + fmtd(B.D) + " x " + fmtd(B.B) + ")", fl, "bearing", md, "steel_300m");
}
// a flanged plain bushing (sleeve bearing), bronze
static js::Value bushing_part(double dmm, double Lmm) {
    const double u = 0.001;
    double d = dmm * u, L = Lmm * u, D = (1.25 * dmm + 2) * u, F = (1.6 * dmm + 4) * u, t = (0.12 * dmm + 1) * u;
    js::Value fl = js::Value::array();
    fl.push(cyl_f("sleeve", "add", 0.5 * D, L, V3()));
    fl.push(cyl_f("flange", "add", 0.5 * F, t, V3(0, 0, 0.5 * L - 0.5 * t)));
    fl.push(cyl_f("bore", "cut", 0.5 * d, 1.5 * L, V3()));
    js::Value md = js::Value::object(); md["d"] = d; md["D"] = D; md["L"] = L; md["flange"] = F;
    return make_part("flanged bushing " + fmtd(dmm) + " x " + fmtd(Lmm), fl, "bushing", md, "bronze");
}
// a shaft with a DIN 6885 keyway on +x, the keyway centred along its length
static js::Value keyshaft_part(double dmm, double Lmm) {
    const double u = 0.001;
    const KeyDef& K = key_for(dmm);
    double d = dmm * u, L = Lmm * u, lk = std::min(0.6 * L, 3 * d);
    js::Value fl = js::Value::array();
    fl.push(cyl_f("shaft", "add", 0.5 * d, L, V3()));
    fl.push(box_f("keyway", "cut", V3(2 * K.t1 * u, K.b * u, lk), V3(0.5 * d, 0, 0)));
    js::Value md = js::Value::object(); md["kind"] = "shaft"; md["d"] = d; md["L"] = L; md["key_b"] = K.b * u; md["key_h"] = K.h * u; md["key_t1"] = K.t1 * u; md["key_len"] = lk;
    return make_part("keyed shaft " + fmtd(dmm) + " x " + fmtd(Lmm), fl, "shaft", md, "steel_mild");
}
// the parallel key for that shaft: it sits in the keyway when placed at the shaft's pose
static js::Value key_part(double dmm, double lmm) {
    const double u = 0.001;
    const KeyDef& K = key_for(dmm);
    double d = dmm * u, lk = lmm * u;
    js::Value fl = js::Value::array();
    js::Value k = box_f("key", "add", V3(K.h * u, K.b * u, lk), V3(0.5 * d - K.t1 * u + 0.5 * K.h * u, 0, 0)); k["round"] = 0.15 * K.b * u; fl.push(k);
    js::Value md = js::Value::object(); md["b"] = K.b * u; md["h"] = K.h * u; md["length"] = lk; md["shaft_d"] = d;
    return make_part("key " + fmtd(K.b) + " x " + fmtd(K.h) + " x " + fmtd(lmm) + " (DIN 6885)", fl, "key", md, "steel_mild");
}
// a rigid sleeve coupling joining two shafts of diameter d, keyed
static js::Value coupling_part(double dmm) {
    const double u = 0.001;
    const KeyDef& K = key_for(dmm);
    double d = dmm * u, D = 2 * d, L = 3 * d;
    js::Value fl = js::Value::array();
    fl.push(cyl_f("sleeve", "add", 0.5 * D, L, V3()));
    fl.push(cyl_f("bore", "cut", 0.5 * d, 1.2 * L, V3()));
    fl.push(box_f("keyway", "cut", V3(2 * (K.h - K.t1) * u, K.b * u, 1.2 * L), V3(0.5 * d, 0, 0)));
    js::Value md = js::Value::object(); md["d"] = d; md["D"] = D; md["L"] = L;
    return make_part("sleeve coupling " + fmtd(dmm), fl, "coupling", md, "steel_mild");
}
// a roller (an idler, a friction wheel, a conveyor roller) with a bore
static js::Value roller_part(double Dmm, double Lmm, double dmm) {
    const double u = 0.001;
    js::Value fl = js::Value::array();
    fl.push(cyl_f("roller", "add", 0.5 * Dmm * u, Lmm * u, V3()));
    if (dmm > 0) fl.push(cyl_f("bore", "cut", 0.5 * dmm * u, 1.5 * Lmm * u, V3()));
    js::Value md = js::Value::object(); md["D"] = Dmm * u; md["L"] = Lmm * u; md["bore"] = dmm * u;
    return make_part("roller " + fmtd(Dmm) + " x " + fmtd(Lmm), fl, "roller", md, "steel_mild");
}
// a V-belt pulley, SPZ grooves (9.7 mm wide at the rim, 38 deg, 11 mm deep, the pitch line 2 mm under the rim, 12 mm
// apart), on a hub; axis local z, the grooves centred on z = 0. Dp the pitch diameter, d the bore (mm)
static js::Value pulley_part(double Dpmm, double dmm, int grooves) {
    const double u = 0.001;
    grooves = std::max(1, std::min(6, grooves));
    double Dp = Dpmm * u, ro = 0.5 * Dp + 2.0 * u, bt = 9.7 * u, hg = 11.0 * u, e = 12.0 * u, f = 8.0 * u, W = 2 * f + (grooves - 1) * e;
    double tn = std::tan(19 * geo::PI / 180), rr = ro - hg, hb = std::max(0.3 * u, 0.5 * bt - hg * tn);
    js::Value prof = js::Value::array();
    auto pt = [&](double x, double r) { js::Value q = js::Value::array(); q.push(js::Value(x)); q.push(js::Value(r)); prof.push(q); };
    pt(-0.5 * W, ro);
    for (int g = 0; g < grooves; ++g) { double xc = -0.5 * W + f + g * e; pt(xc - 0.5 * bt, ro); pt(xc - hb, rr); pt(xc + hb, rr); pt(xc + 0.5 * bt, ro); }
    pt(0.5 * W, ro);
    js::Value body = js::Value::object(); body["id"] = "rim"; body["type"] = "revolve"; body["op"] = "add"; body["mirror"] = false; body["smooth"] = false;
    body["profile"] = prof; body["translate"] = v3json(V3()); body["rotate"] = v3json(V3(0, -90, 0));  // the revolve axis (x) along z
    js::Value fl = js::Value::array(); fl.push(body);
    double d = dmm * u, rh = std::max(0.9 * d, 0.5 * d + 6 * u), rw = rr - 5 * u;
    if (rw > rh + 4 * u) {  // a web between the hub and the rim: the faces recessed
        fl.push(cyl_f("recess_front", "cut", rw, 0.6 * W, V3(0, 0, 0.5 * W)));
        fl.push(cyl_f("recess_back", "cut", rw, 0.6 * W, V3(0, 0, -0.5 * W)));
        fl.push(cyl_f("hub", "add", rh, W, V3()));
    }
    if (d > 0) fl.push(cyl_f("bore", "cut", 0.5 * d, 2 * W, V3()));
    js::Value md = js::Value::object(); md["belt"] = "SPZ"; md["pitch_d"] = Dp; md["outer_d"] = 2 * ro; md["width"] = W; md["bore"] = d; md["grooves"] = (double)grooves;
    return make_part("V-belt pulley SPZ " + fmtd(Dpmm) + (grooves > 1 ? " x" + tostr(grooves) : std::string()) + ", bore " + fmtd(dmm), fl, "pulley", md, "al6061");
}
// ISO 606 roller chains, B series (mm): pitch, roller diameter d1, inner width b1, plate height, plate thickness, and
// the minimum tensile strength (kN)
struct ChainDef { const char* name; double p, d1, b1, h, t, Fu; };
static const ChainDef CHAINS[] = {{"06B", 9.525, 6.35, 5.72, 8.26, 1.3, 9.0}, {"08B", 12.7, 8.51, 7.75, 11.81, 1.6, 18.0}, {"10B", 15.875, 10.16, 9.65, 14.73, 1.7, 22.4},
                                  {"12B", 19.05, 12.07, 11.68, 16.13, 1.85, 29.0}, {"16B", 25.4, 15.88, 17.02, 21.08, 3.1, 60.0}};
static const int NCHAIN = (int)(sizeof(CHAINS) / sizeof(CHAINS[0]));
static const ChainDef* chain_of(const std::string& n) { for (const ChainDef& c : CHAINS) if (n == c.name) return &c; return nullptr; }
// a sprocket for that chain with z teeth (ISO 606 tooth form, simplified): the roller seats (radius 0.505 d1 +
// 0.069 d1^(1/3)) on the pitch circle Dp = p / sin(180 deg / z), flanks widening to the tip, a hub on +z, a bore d (mm)
static js::Value sprocket_part(const ChainDef& C, int z, double dmm) {
    const double u = 0.001;
    z = std::max(8, std::min(120, z));
    double p = C.p * u, d1 = C.d1 * u, Dp = p / std::sin(geo::PI / z), ri = (0.505 * C.d1 + 0.069 * std::cbrt(C.d1)) * u;
    double da = Dp + 0.5 * (1.25 * p + p * (1 - 1.6 / z)) - d1, df = Dp - d1, bf = (C.p <= 12.7 ? 0.93 : 0.95) * C.b1 * u;
    js::Value fl = js::Value::array();
    fl.push(cyl_f("blank", "add", 0.5 * da, bf, V3()));
    for (int k = 0; k < z; ++k) {
        double a = 2 * geo::PI * k / z, c = std::cos(a), s = std::sin(a);
        fl.push(cyl_f(("seat" + tostr(k + 1)).c_str(), "cut", ri, 2 * bf, V3(0.5 * Dp * c, 0.5 * Dp * s, 0)));
        fl.push(cyl_f(("flank" + tostr(k + 1)).c_str(), "cut", 0.7 * d1, 2 * bf, V3((0.5 * Dp + 0.6 * d1) * c, (0.5 * Dp + 0.6 * d1) * s, 0)));
    }
    double d = dmm * u, rh = std::min(0.5 * df - 0.6 * d1, std::max(0.85 * d, 0.5 * d + 5 * u)), Lh = std::max(bf, 0.8 * d);
    if (rh > 0.5 * d + 2 * u) fl.push(cyl_f("hub", "add", rh, Lh + 0.5 * bf, V3(0, 0, 0.5 * Lh + 0.25 * bf)));
    if (d > 0) fl.push(cyl_f("bore", "cut", 0.5 * d, 3 * (bf + Lh), V3()));
    js::Value md = js::Value::object(); md["chain"] = C.name; md["pitch"] = p; md["teeth"] = (double)z; md["pitch_d"] = Dp; md["tip_d"] = da; md["root_d"] = df;
    md["roller_d"] = d1; md["width"] = bf; md["inner_width"] = C.b1 * u; md["plate_h"] = C.h * u; md["plate_t"] = C.t * u; md["bore"] = d; md["strength_N"] = C.Fu * 1000;
    return make_part(std::string("sprocket ") + C.name + " z" + tostr(z) + ", bore " + fmtd(dmm), fl, "sprocket", md, "steel_mild");
}
// a worm (ZA, straight-sided threads in the axial section): axial module, starts, diameter factor q (pitch diameter
// q x module), length, bore, hand (+1 right, -1 left); axis local z, centred on z = 0, a thread on +x at z = 0
static js::Value worm_part(double mod, int starts, double q, double length, double bore, int hand) {
    double d1 = q * mod, lead = starts * geo::PI * mod, gam = std::atan(starts / q) * 180 / geo::PI;
    js::Value fl = js::Value::array(), w = js::Value::object();
    w["id"] = "thread"; w["type"] = "worm"; w["op"] = "add"; w["mirror"] = false; w["module"] = mod; w["starts"] = (double)starts; w["diameter"] = d1;
    w["length"] = length; w["pressure_angle"] = 20.0; w["hand"] = (double)hand; w["translate"] = v3json(V3());
    fl.push(w);
    if (bore > 0) fl.push(cyl_f("bore", "cut", 0.5 * bore, 1.5 * length, V3()));
    js::Value gm = js::Value::object();
    gm["kind"] = "worm"; gm["module"] = mod; gm["teeth"] = (double)starts; gm["starts"] = (double)starts; gm["q"] = q; gm["pitch_radius"] = 0.5 * d1; gm["tip_radius"] = 0.5 * d1 + mod;
    gm["lead"] = lead; gm["lead_angle"] = gam; gm["hand"] = (double)hand; gm["width"] = length; gm["bore"] = bore; gm["pressure_angle"] = 20.0;
    js::Value p = make_part("worm m" + fmtd(mod * 1000) + " z" + tostr(starts) + " q" + fmtd(q) + (hand < 0 ? " LH" : " RH") + " (lead " + fmtd(std::round(gam * 100) / 100) + " deg)", fl, "gear", gm, "steel_300m");
    return p;
}
// its worm wheel: a helical gear of the worm's lead angle (same hand; transverse module = the worm's axial module)
// with a raised blank, throated by the torus the worm sweeps round the wheel (the tips clear the worm's root by 0.25 m)
static js::Value worm_wheel_part(double mod, int z2, int starts, double q, double width, double bore, int hand) {
    double gam = std::atan(starts / q), mn = mod * std::cos(gam), r2 = 0.5 * mod * z2, r1 = 0.5 * q * mod;
    js::Value p = gear_part(mn, z2, width, bore, hand * gam * 180 / geo::PI);
    const js::Value& fl = p["features"];
    js::Value nf = js::Value::array();
    for (size_t i = 0; i < fl.size(); ++i) { js::Value f = fl.at(i).clone(); if (i == 0) f["addendum"] = 1.5; nf.push(f); }
    js::Value t = prim("torus", "cut", {{"R", r1 + r2}, {"r", r1 - mod}}, V3()); t["id"] = "throat"; t["rotate"] = v3json(V3(0, 90, 0));  // the torus's axis (x) along z
    nf.push(t);
    p["features"] = nf;
    js::Value& gm = p["gear"];
    gm["kind"] = "worm_wheel"; gm["module"] = mod; gm["normal_module"] = mn; gm["pitch_radius"] = r2; gm["tip_radius"] = r2 + mod; gm["worm_starts"] = (double)starts; gm["worm_q"] = q; gm["hand"] = (double)hand;
    p["name"] = "worm wheel m" + fmtd(mod * 1000) + " z" + tostr(z2) + " (for worm z" + tostr(starts) + " q" + fmtd(q) + (hand < 0 ? " LH)" : " RH)");
    p["material"] = "bronze";
    return p;
}
static bool is_worm(int c);
// ---------------------------------------------------------------- cams: a disc cam and its translating follower
// The follower's lift s(theta) over one turn: a rise, a dwell, a return and a dwell, each move by a standard law:
// cycloidal (zero acceleration at both ends: no jerk spikes), harmonic (simple; acceleration jumps at the ends),
// 3-4-5 polynomial, or uniform velocity (infinite acceleration at the ends: only for slow cams).
struct CamDef { int law = 0, flat = 0; double rb = 0.025, h = 0.015, rise = 2.0944, dwell = 1.0472, ret = 2.0944, rr = 0.008, e = 0; };
static const char* CAM_LAWS[4] = {"cycloidal", "harmonic", "3-4-5 polynomial", "uniform velocity"};
static CamDef cam_def(const js::Value& c) {
    CamDef d; std::string l = c.str("law", "cycloidal");
    d.law = l == "harmonic" ? 1 : l == "345" || l == "3-4-5 polynomial" ? 2 : l == "uniform" || l == "uniform velocity" ? 3 : 0;
    d.rb = c.num("base_r", 0.025); d.h = c.num("lift", 0.015); d.rise = c.num("rise_deg", 120) * geo::PI / 180; d.dwell = c.num("dwell_deg", 60) * geo::PI / 180;
    d.ret = c.num("return_deg", 120) * geo::PI / 180; d.rr = c.num("roller_r", 0.008); d.e = c.num("offset", 0); d.flat = c.str("follower", "roller") == "flat" ? 1 : 0;
    return d;
}
static double cam_lift(const CamDef& c, double th) {
    th -= 2 * geo::PI * std::floor(th / (2 * geo::PI));
    auto f = [&](double x) {
        x = geo::clampd(x, 0, 1);
        switch (c.law) {
            case 1: return 0.5 * (1 - std::cos(geo::PI * x));
            case 2: return x * x * x * (10 - 15 * x + 6 * x * x);
            case 3: return x;
            default: return x - std::sin(2 * geo::PI * x) / (2 * geo::PI);
        }
    };
    if (th < c.rise) return c.h * f(th / c.rise);
    th -= c.rise; if (th < c.dwell) return c.h;
    th -= c.dwell; if (th < c.ret) return c.h * (1 - f(th / c.ret));
    return 0;
}
static double cam_s0(const CamDef& c) { return c.flat ? c.rb : std::sqrt(std::max(1e-12, (c.rb + c.rr) * (c.rb + c.rr) - c.e * c.e)); }
// the cam's outline (cam frame, the follower along +y at x = e when the cam is at 0) and its checks: the largest
// pressure angle (roller), the smallest radius of curvature, undercut
struct CamCheck { double pmax = 0, pmax_at = 0, rho_min = 1e9, rho_at = 0, rmax = 0, face_w = 0; bool undercut = false; };
static std::vector<std::pair<double, double>> cam_outline(const CamDef& c, int n, CamCheck& k) {
    std::vector<std::pair<double, double>> out;
    double s0 = cam_s0(c), dt = 2 * geo::PI / n, hd = 1e-4;
    auto pitch = [&](double th) { double s = cam_lift(c, th), ct = std::cos(th), st = std::sin(th), X = c.flat ? 0 : c.e, Y = s0 + s; return std::make_pair(X * ct + Y * st, -X * st + Y * ct); };  // Rot(-th)(X, Y)
    for (int i = 0; i < n; ++i) {
        double th = i * dt, s = cam_lift(c, th), sp = (cam_lift(c, th + hd) - cam_lift(c, th - hd)) / (2 * hd), spp = (cam_lift(c, th + hd) - 2 * s + cam_lift(c, th - hd)) / (hd * hd);
        double ct = std::cos(th), st = std::sin(th), x, y;
        if (c.flat) {  // the face touches at (s', rb + s) in the follower's frame
            x = sp * ct + (c.rb + s) * st; y = -sp * st + (c.rb + s) * ct;
            double rho = c.rb + s + spp;
            if (rho < k.rho_min) { k.rho_min = rho; k.rho_at = th; }
            k.face_w = std::max(k.face_w, std::fabs(sp));
        } else {  // the pitch curve, offset inwards by the roller
            auto p0 = pitch(th - hd), p1 = pitch(th), p2 = pitch(th + hd);
            double tx = (p2.first - p0.first) / (2 * hd), ty = (p2.second - p0.second) / (2 * hd), tl = std::sqrt(tx * tx + ty * ty);
            double nx = -ty / tl, ny = tx / tl;
            if (nx * p1.first + ny * p1.second < 0) { nx = -nx; ny = -ny; }
            x = p1.first - c.rr * nx; y = p1.second - c.rr * ny;
            double ax = (p2.first - 2 * p1.first + p0.first) / (hd * hd), ay = (p2.second - 2 * p1.second + p0.second) / (hd * hd);
            double cr = std::fabs(tx * ay - ty * ax), rho = cr > 1e-15 ? tl * tl * tl / cr : 1e9;
            bool convex = (tx * ay - ty * ax) * (tx * p1.second - ty * p1.first) > 0;  // curving towards the centre
            if (convex && rho < k.rho_min) { k.rho_min = rho; k.rho_at = th; }
            if (convex && rho < c.rr) k.undercut = true;
            double pa = std::atan((sp - c.e) / (s0 + s));
            if (std::fabs(pa) > std::fabs(k.pmax)) { k.pmax = pa; k.pmax_at = th; }
        }
        k.rmax = std::max(k.rmax, std::sqrt(x * x + y * y));
        out.push_back({x, y});
    }
    if (c.flat && k.rho_min < 0) k.undercut = true;
    return out;
}
static std::string cam_report(const CamDef& c, const CamCheck& k) {
    std::string r = std::string(CAM_LAWS[c.law]) + ", lift " + fmtd(std::round(c.h * 1e4) / 10) + " mm";
    if (!c.flat) r += ", largest pressure angle " + fmtd(std::round(std::fabs(k.pmax) * 1800 / geo::PI) / 10) + " deg at " + fmtd(std::round(k.pmax_at * 180 / geo::PI)) + " deg" + (std::fabs(k.pmax) > 30 * geo::PI / 180 ? " (over 30 deg: the follower jams; enlarge the base circle or slow the rise)" : "");
    r += ", smallest radius of curvature " + fmtd(std::round(k.rho_min * 1e4) / 10) + " mm";
    if (k.undercut) r += c.flat ? "; warning: the face would need a cusp (radius of curvature below 0): enlarge the base circle" : "; warning: undercut, the pitch curve bends tighter than the roller: enlarge the base circle or use a smaller roller";
    return r;
}
// the cam: the outline extruded (thickness t), a bore; law, base radius rb, lift h, angles in degrees, the roller
// radius rr (0 for a flat-faced follower) and offset e, lengths in metres
static js::Value cam_part(int law, double rb, double h, double rise, double dwell, double ret, double rr, double e, double t, double bore) {
    js::Value cm = js::Value::object();
    cm["law"] = law == 1 ? "harmonic" : law == 2 ? "345" : law == 3 ? "uniform" : "cycloidal"; cm["base_r"] = rb; cm["lift"] = h; cm["rise_deg"] = rise; cm["dwell_deg"] = dwell; cm["return_deg"] = ret;
    cm["roller_r"] = rr; cm["offset"] = rr > 0 ? e : 0.0; cm["follower"] = rr > 0 ? "roller" : "flat"; cm["thickness"] = t; cm["bore"] = bore;
    CamDef c = cam_def(cm); CamCheck k;
    std::vector<std::pair<double, double>> ol = cam_outline(c, 360, k);
    js::Value o = js::Value::array();
    for (auto& q : ol) { js::Value pt = js::Value::array(); pt.push(js::Value(q.first)); pt.push(js::Value(q.second)); o.push(pt); }
    js::Value f = js::Value::object(); f["id"] = "cam"; f["type"] = "extrude"; f["op"] = "add"; f["mirror"] = false; f["smooth"] = false; f["outline"] = o; f["height"] = t; f["translate"] = v3json(V3());
    js::Value fl = js::Value::array(); fl.push(f);
    if (bore > 0) fl.push(cyl_f("bore", "cut", 0.5 * bore, 2 * t, V3()));
    cm["max_pressure_deg"] = std::fabs(k.pmax) * 180 / geo::PI; cm["min_curvature_r"] = k.rho_min; cm["undercut"] = k.undercut; cm["max_r"] = k.rmax; cm["face_halfwidth"] = k.face_w;
    return make_part("disc cam " + std::string(CAM_LAWS[c.law]) + ", base " + fmtd(rb * 2000) + " mm, lift " + fmtd(h * 1000) + " mm", fl, "cam", cm, "steel_300m");
}
// a translating follower: a roller in a fork (rr > 0) or a flat face (half-width fw), on a stem of diameter ds and
// length L along local +y; the roller's centre (or the face) at the origin, the roller's axis local z
static js::Value follower_part(double rr, double fw, double t, double L, double ds) {
    js::Value fl = js::Value::array(), md = js::Value::object();
    double top;
    if (rr > 0) {
        double g = 0.0005, w = 0.7 * t, pt = std::max(0.002, 0.14 * t);
        fl.push(cyl_f("roller", "add", rr, w, V3()));
        fl.push(cyl_f("pin", "add", 0.3 * rr, w + 2 * (g + pt), V3()));
        for (int s = -1; s <= 1; s += 2) fl.push(box_f(s < 0 ? "fork_a" : "fork_b", "add", V3(1.3 * rr, 1.5 * rr, pt), V3(0, 0.6 * rr, s * (0.5 * w + g + 0.5 * pt))));
        fl.push(box_f("bridge", "add", V3(1.3 * rr, 0.3 * rr, w + 2 * (g + pt)), V3(0, 1.35 * rr + 0.15 * rr, 0)));
        top = 1.5 * rr;
        md["kind"] = "roller"; md["roller_r"] = rr;
    } else {
        double tf = std::max(0.003, 0.2 * t);
        fl.push(box_f("face", "add", V3(2 * fw, tf, t), V3(0, 0.5 * tf, 0)));
        top = tf;
        md["kind"] = "flat"; md["face_halfwidth"] = fw;
    }
    fl.push(cyl_f("stem", "add", 0.5 * ds, L, V3(0, top + 0.5 * L - 0.0005, 0), "y"));
    md["stem_d"] = ds; md["length"] = L; md["top"] = top + L;
    return make_part(rr > 0 ? "roller follower " + fmtd(rr * 2000) + " mm, stem " + fmtd(ds * 1000) + " x " + fmtd(L * 1000) : "flat-faced follower " + fmtd(fw * 2000) + " mm, stem " + fmtd(ds * 1000) + " x " + fmtd(L * 1000), fl, "follower", md, "steel_mild");
}
// a gear's hub: the faces recessed between the hub and the teeth, and a splined bore (n straight-sided splines)
static void gear_hub(js::Value& part, double bore, int splines) {
    const js::Value* g = part.find("gear");
    if (!g || g->num("cone", 0) > 0) return;
    double w = g->num("width", 0), m = g->num("module", 0), rf = g->num("pitch_radius", 0) - 1.25 * m;
    js::Value fl = js::Value::array();
    const js::Value& old = part["features"];
    fl.push(old.at(0).clone());  // the teeth
    double rr = 0.78 * rf, dep = 0.22 * w, rh = std::min(0.62 * rf, std::max(0.8 * bore, 0.5 * bore + 3 * m));
    if (rr > rh + m) {
        fl.push(cyl_f("recess_front", "cut", rr, 2 * dep, V3(0, 0, 0.5 * w)));
        fl.push(cyl_f("recess_back", "cut", rr, 2 * dep, V3(0, 0, -0.5 * w)));
        fl.push(cyl_f("hub", "add", rh, w, V3()));
    }
    if (bore > 0 && splines >= 4) {  // straight-sided splines, the outer diameter 1.2 x the bore
        double ri = 0.5 * bore, ro = 0.6 * bore, half = geo::PI / (2 * splines);
        js::Value ol = js::Value::array();
        auto pt = [&](double r, double a) { js::Value q = js::Value::array(); q.push(js::Value(r * std::cos(a))); q.push(js::Value(r * std::sin(a))); ol.push(q); };
        for (int k = 0; k < splines; ++k) {
            double c = 2 * geo::PI * k / splines, ao = std::asin(std::min(1.0, ri * std::sin(half) / ro));  // straight flanks: constant width
            pt(ri, c - half); pt(ro, c - ao); pt(ro, c + ao); pt(ri, c + half);
            pt(ri, c + 2 * half);  // the land between splines, on the bore circle
        }
        js::Value e = js::Value::object(); e["id"] = "spline"; e["type"] = "extrude"; e["op"] = "cut"; e["mirror"] = false; e["smooth"] = false;
        e["outline"] = ol; e["height"] = 3 * w; e["translate"] = v3json(V3());
        fl.push(e);
        part["name"] = part.str("name", "gear") + ", " + tostr(splines) + "-spline bore";
    } else if (bore > 0) fl.push(cyl_f("bore", "cut", 0.5 * bore, 3 * w, V3()));
    part["features"] = fl;
    js::Value& gm = part["gear"]; gm["hub"] = true; gm["splines"] = (double)splines;
}

static const js::Value* gear_meta(int c) { return c >= 0 && c < (int)g_ac.size() ? g_ac[(size_t)c].part.find("gear") : nullptr; }
static bool is_rack(int c) { const js::Value* g = gear_meta(c); return g && g->str("kind", "") == "rack"; }
static bool is_worm(int c) { const js::Value* g = gear_meta(c); return g && g->str("kind", "") == "worm"; }
static V3 gear_centre(int c) {  // the centre of the pitch circle (a bevel gear's at its large end)
    const js::Value* g = gear_meta(c);
    double cone = g ? g->num("cone", 0) : 0;
    return g_ac[(size_t)c].pose.apply(V3(0, 0, cone > 0 ? -0.5 * g->num("width", 0) : 0));
}
static double axis_angle(const V3& n, const V3& from, const V3& v) { V3 w = v - n * geo::dot(v, n); return std::atan2(geo::dot(geo::cross(from, w), n), geo::dot(from, w)); }

// ---------------------------------------------------------------- face recognition on a picked triangle
// the smooth face around the picked triangle (split_sharp cut the mesh along its edges): a plane, a cylinder (axis and
// radius) or, failing both, the picked point
static ARef recognise_mesh(int comp, const AMesh& M, int tri, const V3& hitl, const V3& nl) {
    ARef r; r.comp = comp; r.kind = 2; r.p = hitl; r.d = nl;
    size_t nt = M.m.t.size() / 3;
    if (tri < 0 || (size_t)tri >= nt) return r;
    std::vector<char> seen(nt, 0);
    std::vector<int> st{tri}, reg;
    seen[(size_t)tri] = 1;
    while (!st.empty() && reg.size() < 400000) {
        int t = st.back(); st.pop_back(); reg.push_back(t);
        for (int k = 0; k < 3; ++k) {
            int v = M.m.t[3 * (size_t)t + (size_t)k];
            for (int q = M.vstart[(size_t)v]; q < M.vstart[(size_t)v + 1]; ++q) { int u = M.vtris[(size_t)q]; if (!seen[(size_t)u]) { seen[(size_t)u] = 1; st.push_back(u); } }
        }
    }
    double C[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, A = 0;
    V3 ns;
    std::vector<V3> cen(reg.size()), nrm(reg.size());
    std::vector<double> area(reg.size());
    for (size_t i = 0; i < reg.size(); ++i) {
        const V3 &a = M.m.v[(size_t)M.m.t[3 * (size_t)reg[i]]], &b = M.m.v[(size_t)M.m.t[3 * (size_t)reg[i] + 1]], &c = M.m.v[(size_t)M.m.t[3 * (size_t)reg[i] + 2]];
        V3 n = geo::cross(b - a, c - a); double l = geo::len(n);
        area[i] = 0.5 * l; nrm[i] = l > 0 ? n / l : V3(); cen[i] = (a + b + c) / 3.0;
        A += area[i]; ns = ns + nrm[i] * area[i];
        for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q) C[p][q] += area[i] * nrm[i][p] * nrm[i][q];
    }
    if (A <= 0) return r;
    V3 mn = ns / std::max(geo::len(ns), 1e-30);
    double flat_area = 0; for (size_t i = 0; i < reg.size(); ++i) if (geo::dot(nrm[i], mn) > 0.9986) flat_area += area[i];  // within 3 deg
    if (flat_area > 0.97 * A) {  // flat (a few slivers along its edges aside)
        r.kind = 0; r.d = mn;
        V3 c; for (size_t i = 0; i < reg.size(); ++i) c = c + cen[i] * area[i]; c = c / A;
        r.p = hitl - mn * geo::dot(hitl - c, mn);  // on the plane, under the pick
        return r;
    }
    double V[3][3], w[3];
    geo::eig_sym3(C, V, w);
    int im = 0; for (int k = 1; k < 3; ++k) if (w[k] < w[im]) im = k;
    int ix = 0; for (int k = 1; k < 3; ++k) if (w[k] > w[ix]) ix = k;
    double wmid = w[0] + w[1] + w[2] - w[im] - w[ix];
    V3 ax(V[0][im], V[1][im], V[2][im]);
    double off = 0; for (size_t i = 0; i < reg.size(); ++i) off += area[i] * std::fabs(geo::dot(nrm[i], ax)); off /= A;
    geo::Box rb; for (const V3& c : cen) rb.add(c);
    double extent = geo::len(rb.hi - rb.lo);
    if (off < 0.06 && reg.size() >= 6 && wmid > 0.08 * w[ix]) {  // the normals fan round one axis: a cylinder (hole or shaft)
        V3 u = std::fabs(ax.x) < 0.9 ? geo::cross(ax, V3(1, 0, 0)) : geo::cross(ax, V3(0, 1, 0)); u = u / geo::len(u);
        V3 v = geo::cross(ax, u);
        // Kasa circle fit in the plane across the axis: x^2 + y^2 + D x + E y + F = 0
        double S[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, b[3] = {0, 0, 0};
        double smin = 1e300, smax = -1e300;
        for (size_t i = 0; i < reg.size(); ++i) {
            double x = geo::dot(cen[i], u), y = geo::dot(cen[i], v), z = -(x * x + y * y), row[3] = {x, y, 1};
            for (int p = 0; p < 3; ++p) { for (int q = 0; q < 3; ++q) S[p][q] += area[i] * row[p] * row[q]; b[p] += area[i] * row[p] * z; }
            double s = geo::dot(cen[i], ax); smin = std::min(smin, s); smax = std::max(smax, s);
        }
        double det = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1]) - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0]) + S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);
        if (std::fabs(det) > 1e-30) {
            auto solve = [&](int col) { double T[3][3]; memcpy(T, S, sizeof T); for (int p = 0; p < 3; ++p) T[p][col] = b[p];
                return (T[0][0] * (T[1][1] * T[2][2] - T[1][2] * T[2][1]) - T[0][1] * (T[1][0] * T[2][2] - T[1][2] * T[2][0]) + T[0][2] * (T[1][0] * T[2][1] - T[1][1] * T[2][0])) / det; };
            double D = solve(0), E = solve(1), F = solve(2), cx = -D / 2, cy = -E / 2, rr = cx * cx + cy * cy - F;
            if (rr > 0 && std::sqrt(rr) < extent) {
                r.kind = 1; r.r = std::sqrt(rr); r.d = ax;
                double sh = geo::dot(hitl, ax);
                r.p = u * cx + v * cy + ax * sh;
                // a hole (normals towards the axis) or a shaft: record both ends of the face for insertion
                V3 axisPt = u * cx + v * cy;
                double inward = 0; for (size_t i = 0; i < reg.size(); ++i) { V3 rad = cen[i] - (axisPt + ax * geo::dot(cen[i], ax)); inward += area[i] * geo::dot(nrm[i], rad); }
                r.face = true;
                r.fp = axisPt + ax * smin; r.fn = ax * -1.0;  // provisional: the -axis end; insert picks the end it needs
                r.value_ends[0] = smin; r.value_ends[1] = smax; r.hole = inward < 0;
                return r;
            }
        }
    }
    return r;
}

// a reference on one piece, in its component's frame
static ARef recognise(int comp, int piece, int tri, const V3& hitp, const V3& np) {
    const APiece& P = g_ac[(size_t)comp].pieces[(size_t)piece];
    ARef r = recognise_mesh(comp, *P.mesh, tri, hitp, np);
    const Pose& L = P.local;
    r.p = L.apply(r.p); r.d = L.rot(r.d); r.fp = L.apply(r.fp); r.fn = L.rot(r.fn);
    double sh = geo::dot(L.t, r.d); r.value_ends[0] += sh; r.value_ends[1] += sh;
    return r;
}

// ---------------------------------------------------------------- world references and the solver
struct WRef { V3 p, d, fp, fn; };
static WRef world(const ARef& r) {
    const Pose& P = g_ac[(size_t)r.comp].pose;
    WRef w; w.p = P.apply(r.p); w.d = P.rot(r.d); w.fp = P.apply(r.fp); w.fn = P.rot(r.fn);
    return w;
}
static double unit_angle(const V3& a, const V3& b) { return std::acos(geo::clampd(geo::dot(a, b), -1, 1)); }
static double twist_about(const double R[9], const double R0[9], const V3& u) {  // the turn from R0 to R about u (radians)
    double R0t[9] = {R0[0], R0[3], R0[6], R0[1], R0[4], R0[7], R0[2], R0[5], R0[8]}, D[9];
    mat_mul(R, R0t, D);
    Quat q = quat_of(D);
    return 2 * std::atan2(q.x * u.x + q.y * u.y + q.z * u.z, q.w);
}
// one projection step of a mate: moves component `mv` (the b side, or the a side with the roles swapped)
static double mate_step(AMate& m, bool swap) {
    const ARef& A = swap ? m.b : m.a;
    const ARef& B = swap ? m.a : m.b;
    Pose& P = g_ac[(size_t)B.comp].pose;
    WRef a = world(A), b = world(B);
    double res = 0;
    double M[9];
    auto align = [&](const V3& from, const V3& to) {  // turn the moving part so `from` points along `to`, about b's point
        res += unit_angle(from, to);
        rot_between(from, to, M); pose_turn(P, M, b.p); b = world(B);
    };
    auto shift = [&](const V3& dv) { res += geo::len(dv); P.t = P.t + dv; b = world(B); };
    switch (m.type) {
        case MT_COINCIDENT: case MT_DISTANCE: {
            if (A.kind == 0 && B.kind == 0) {  // planes face each other (or the same way, if aligned)
                align(b.d, m.aligned ? a.d : a.d * -1.0);
                double gap = geo::dot(b.p - a.p, a.d) - (m.type == MT_DISTANCE ? m.value : 0);
                shift(a.d * -gap);
            } else if (A.kind == 1 && B.kind == 1) {  // collinear axes
                align(b.d, geo::dot(b.d, a.d) >= 0 ? a.d : a.d * -1.0);
                V3 e = b.p - a.p; e = e - a.d * geo::dot(e, a.d); shift(e * -1.0);
            } else if (A.kind == 0) {  // a point or an axis on a plane
                double gap = geo::dot(b.p - a.p, a.d) - (m.type == MT_DISTANCE ? m.value : 0);
                if (B.kind == 1) align(b.d, b.d - a.d * geo::dot(b.d, a.d));  // the axis lies in the plane
                shift(a.d * -gap);
            } else if (B.kind == 0) {
                double gap = geo::dot(a.p - b.p, b.d) - (m.type == MT_DISTANCE ? m.value : 0);
                shift(b.d * gap);
            } else if (A.kind == 1) {  // a point on an axis
                V3 e = b.p - a.p; e = e - a.d * geo::dot(e, a.d); shift(e * -1.0);
            } else shift(a.p - b.p);
            break;
        }
        case MT_CONCENTRIC: {
            if (A.kind == 1 && B.kind == 1) {
                align(b.d, (geo::dot(b.d, a.d) >= 0) != m.aligned ? a.d : a.d * -1.0);
                if (m.aligned) {}  // aligned flips the sense
            }
            V3 e = b.p - a.p; e = e - a.d * geo::dot(e, a.d); shift(e * -1.0);
            break;
        }
        case MT_INSERT: {  // axes collinear, the seat (b) on the entry face (a), facing it
            align(b.fn, a.fn * -1.0);
            V3 e = b.p - a.p; e = e - a.d * geo::dot(e, a.d); shift(e * -1.0);
            double gap = geo::dot(b.fp - a.fp, a.fn); shift(a.fn * -gap);
            break;
        }
        case MT_ROLL:
        case MT_BELT:
        case MT_GEAR: {
            if (m.type == MT_GEAR && m.zb == 0 && m.rp > 0) {  // rack and pinion: a is the pinion, b the rack; the rack moves rp per radian
                const AComp &GA = g_ac[(size_t)m.a.comp], &GB = g_ac[(size_t)m.b.comp];
                double sense = m.sense != 0 ? m.sense : 1;
                if (!swap) {  // the rack follows the pinion
                    double ta = twist_about(GA.pose.R, m.rel.R, GA.pose.rot(m.a.d));
                    m.acc[0] += std::remainder(ta - m.last[0], 2 * geo::PI); m.last[0] = ta;
                    V3 xb = P.rot(V3(1, 0, 0));
                    double want = sense * m.rp * m.acc[0], cur = geo::dot(P.t - m.rel2.t, xb), dv = want - cur;
                    if (std::fabs(dv) > 1e-12) { res += std::fabs(dv); P.t = P.t + xb * dv; b = world(B); }
                } else {      // the pinion follows the rack
                    V3 xb = GB.pose.rot(V3(1, 0, 0));
                    double sl = geo::dot(GB.pose.t - m.rel2.t, xb), want = sl / (sense * m.rp);
                    double ta = twist_about(P.R, m.rel.R, b.d), dlt = std::remainder(want - ta, 2 * geo::PI);
                    m.acc[0] = want; m.last[0] = ta + dlt;
                    if (std::fabs(dlt) > 1e-12) { res += std::fabs(dlt); double M2[9]; rot_axis(b.d, dlt, M2); pose_turn(P, M2, b.p); b = world(B); }
                }
                break;
            }
            // two gears: the turn of b about its own axis follows a's by the tooth ratio (any axes: parallel, crossed)
            double ratio = m.type == MT_ROLL || m.type == MT_BELT ? m.value : (m.za > 0 && m.zb > 0 ? (double)m.za / m.zb : 0);  // b's turn per turn of a
            if (ratio <= 0) break;
            if (swap) ratio = 1 / ratio;
            const Pose& PA = g_ac[(size_t)A.comp].pose;
            const Pose& A0 = swap ? m.rel2 : m.rel;
            const Pose& B0 = swap ? m.rel : m.rel2;
            int ia = swap ? 1 : 0;
            double ta = twist_about(PA.R, A0.R, a.d), tb = twist_about(P.R, B0.R, b.d);
            m.acc[ia] += std::remainder(ta - m.last[ia], 2 * geo::PI); m.last[ia] = ta;  // unwrapped: many turns
            double sense = m.sense != 0 ? m.sense : (geo::dot(a.d, b.d) >= 0 ? -1.0 : 1.0);  // from the contact (older files: outside spur gears)
            double want = sense * m.acc[ia] * ratio;
            double dlt = std::remainder(want - tb, 2 * geo::PI);
            if (std::fabs(dlt) > 1e-12) { res += std::fabs(dlt); double M2[9]; rot_axis(b.d, dlt, M2); pose_turn(P, M2, b.p); b = world(B); }
            break;
        }
        case MT_CAM: {  // the follower slides along its line by the cam's lift at the cam's angle
            if (swap) break;  // the cam drives the follower, never the other way
            const AComp& CA = g_ac[(size_t)A.comp];
            const js::Value* cm = CA.part.find("cam");
            if (!cm) break;
            CamDef cd = cam_def(*cm);
            V3 nA = CA.pose.rot(V3(0, 0, 1)), dl = CA.pose.inv_rot(m.cdir), xw = geo::cross(m.cdir, nA);
            double s = cam_lift(cd, std::atan2(dl.x, dl.y));
            V3 want = CA.pose.t + nA * m.cax + m.cdir * (cam_s0(cd) + s) + (cd.flat ? V3() : xw * cd.e);
            res += geo::len(want - P.t);
            for (int i = 0; i < 9; ++i) res += std::fabs(m.rel2.R[i] - P.R[i]);
            P.t = want; for (int i = 0; i < 9; ++i) P.R[i] = m.rel2.R[i];
            b = world(B);
            break;
        }
        case MT_PARALLEL: align(b.d, geo::dot(b.d, a.d) >= 0 ? a.d : a.d * -1.0); break;
        case MT_PERP: { V3 t = b.d - a.d * geo::dot(b.d, a.d); double l = geo::len(t); if (l > 1e-9) align(b.d, t / l); break; }
        case MT_ANGLE: {
            V3 ax = geo::cross(a.d, b.d); double l = geo::len(ax);
            if (l < 1e-9) ax = std::fabs(a.d.x) < 0.9 ? geo::cross(a.d, V3(1, 0, 0)) : geo::cross(a.d, V3(0, 1, 0));
            double R[9]; rot_axis(ax, m.value * geo::PI / 180, R);
            align(b.d, mat_vec(R, a.d));
            break;
        }
        case MT_LOCK: {  // keep the pose relative to a: Pb = Pa * rel
            const Pose& Pa = g_ac[(size_t)A.comp].pose;
            Pose want; mat_mul(Pa.R, m.rel.R, want.R); want.t = Pa.apply(m.rel.t);
            if (swap) {  // rel is b in a's frame; with roles swapped the moving part is a: Pa = Pb * rel^-1
                const Pose& Pb = g_ac[(size_t)A.comp].pose;  // here A is the original b
                double Rt[9] = {m.rel.R[0], m.rel.R[3], m.rel.R[6], m.rel.R[1], m.rel.R[4], m.rel.R[7], m.rel.R[2], m.rel.R[5], m.rel.R[8]};
                mat_mul(Pb.R, Rt, want.R); want.t = Pb.t - mat_vec(want.R, m.rel.t);
            }
            res += geo::len(want.t - P.t);
            for (int i = 0; i < 9; ++i) res += std::fabs(want.R[i] - P.R[i]);
            P = want;
            break;
        }
    }
    return res;
}
static void asm_solve(const std::vector<char>* driven) {
    auto moves = [&](int c) { return c >= 0 && c < (int)g_ac.size() && !g_ac[(size_t)c].fixed && !(driven && (*driven)[(size_t)c]); };
    for (int it = 0; it < 200; ++it) {
        double worst = 0;
        for (AMate& m : g_am) {
            if (!m.on || m.a.comp < 0 || m.b.comp < 0 || m.a.comp >= (int)g_ac.size() || m.b.comp >= (int)g_ac.size() || m.a.comp == m.b.comp) continue;
            bool mb = moves(m.b.comp), ma = moves(m.a.comp);
            if (!mb && !ma) continue;
            worst = std::max(worst, mate_step(m, !mb));
        }
        if (worst < 1e-9) break;
    }
}

// the gear mates' turn counters from the poses (at time 0 of the motion study, or now)
static void gear_resync(bool from_base) {
    for (AMate& m : g_am) {
        if ((m.type != MT_GEAR && m.type != MT_ROLL && m.type != MT_BELT) || m.a.comp < 0 || m.b.comp < 0 || m.a.comp >= (int)g_ac.size() || m.b.comp >= (int)g_ac.size()) continue;
        const Pose& A = from_base ? g_ac[(size_t)m.a.comp].base : g_ac[(size_t)m.a.comp].pose;
        const Pose& B = from_base ? g_ac[(size_t)m.b.comp].base : g_ac[(size_t)m.b.comp].pose;
        m.acc[0] = m.last[0] = twist_about(A.R, m.rel.R, A.rot(m.a.d));
        m.acc[1] = m.last[1] = twist_about(B.R, m.rel2.R, B.rot(m.b.d));
    }
}
static double g_ms_prev_t = 0;

// ---------------------------------------------------------------- picking
static bool asm_ray(int mx, int my, V3& o, V3& d) {
    double x0, y0, z0, x1, y1, z1, wy = g_vp[3] - my;
    if (!gluUnProject(mx, wy, 0.0, g_mv, g_pj, g_vp, &x0, &y0, &z0) || !gluUnProject(mx, wy, 1.0, g_mv, g_pj, g_vp, &x1, &y1, &z1)) return false;
    o = V3(x0, y0, z0); d = V3(x1 - x0, y1 - y0, z1 - z0); d = d / geo::len(d);
    return true;
}
static V3 explode_offset(int c);
// the part under the mouse: its component, the piece and triangle hit, the hit in the component's frame (hitl, nl)
// and in the piece's own frame (hitp, np)
static int asm_pick_full(int mx, int my, V3& hitw, V3& hitl, V3& nl, int& tri, int& piece, V3& hitp, V3& np) {
    V3 o, d;
    if (!asm_ray(mx, my, o, d)) return -1;
    double best = 1e300; int bc = -1;
    for (size_t c = 0; c < g_ac.size(); ++c) {
        const AComp& C = g_ac[c];
        if (C.hidden) continue;
        Pose P = C.pose; P.t = P.t + explode_offset((int)c);
        V3 oc = P.inv(o), dc = P.inv_rot(d);
        for (size_t k = 0; k < C.pieces.size(); ++k) {
            const APiece& pc = C.pieces[k];
            if (!pc.mesh || pc.mesh->m.t.empty()) continue;
            V3 ol = pc.local.inv(oc), dl = pc.local.inv_rot(dc);
            const geo::Mesh& M = pc.mesh->m;
            double t0 = 0, t1 = 1e300; bool miss = false;  // a quick box test first
            for (int a = 0; a < 3 && !miss; ++a) {
                if (std::fabs(dl[a]) < 1e-15) { if (ol[a] < pc.mesh->box.lo[a] || ol[a] > pc.mesh->box.hi[a]) miss = true; continue; }
                double ta = (pc.mesh->box.lo[a] - ol[a]) / dl[a], tb = (pc.mesh->box.hi[a] - ol[a]) / dl[a];
                if (ta > tb) std::swap(ta, tb);
                t0 = std::max(t0, ta); t1 = std::min(t1, tb); if (t0 > t1) miss = true;
            }
            if (miss || t0 > best) continue;
            for (size_t q = 0; q + 2 < M.t.size(); q += 3) {
                const V3 &a = M.v[(size_t)M.t[q]], &b = M.v[(size_t)M.t[q + 1]], &cc = M.v[(size_t)M.t[q + 2]];
                V3 e1 = b - a, e2 = cc - a, pv = geo::cross(dl, e2);
                double det = geo::dot(e1, pv);
                if (std::fabs(det) < 1e-18) continue;
                double inv = 1 / det; V3 tv = ol - a;
                double u = geo::dot(tv, pv) * inv; if (u < 0 || u > 1) continue;
                V3 qv = geo::cross(tv, e1);
                double v = geo::dot(dl, qv) * inv; if (v < 0 || u + v > 1) continue;
                double t = geo::dot(e2, qv) * inv;
                if (t > 1e-9 && t < best) {
                    best = t; bc = (int)c; tri = (int)(q / 3); piece = (int)k;
                    hitp = ol + dl * t; V3 n = geo::cross(e1, e2); np = n / geo::len(n);
                    if (geo::dot(np, dl) > 0) np = np * -1.0;
                    hitl = pc.local.apply(hitp); nl = pc.local.rot(np);
                    hitw = o + d * t;
                }
            }
        }
    }
    return bc;
}
static int asm_pick_at(int mx, int my, V3& hitw, V3& hitl, V3& nl, int& tri) { int pc = -1; V3 hp, np; return asm_pick_full(mx, my, hitw, hitl, nl, tri, pc, hp, np); }

// ---------------------------------------------------------------- exploded view
static V3 asm_centre() { geo::Box b; for (const AComp& c : g_ac) comp_world_box(c, b); return b.valid() ? (b.lo + b.hi) * 0.5 : V3(); }
static V3 explode_offset(int c) {
    if (g_explode <= 1e-6 || c < 0 || c >= (int)g_ac.size() || g_ac[(size_t)c].fixed) return V3();
    const AComp& C = g_ac[(size_t)c];
    double size = comp_size(C);
    double s = g_explode * g_explode * (3 - 2 * g_explode);  // ease in and out
    if (C.part.find("fastener")) {  // out of its hole, along its own axis (+z: away from the seat)
        V3 ax = C.pose.rot(V3(0, 0, 1));
        int order = 1; for (int k = 0; k < c; ++k) if (g_ac[(size_t)k].part.find("fastener") && geo::dot(g_ac[(size_t)k].pose.rot(V3(0, 0, 1)), ax) > 0.99 &&
                                                       geo::len((g_ac[(size_t)k].pose.t - C.pose.t) - ax * geo::dot(g_ac[(size_t)k].pose.t - C.pose.t, ax)) < 1e-3) ++order;
        return ax * (s * 1.6 * size * order);
    }
    V3 ctr = asm_centre(), mid = comp_mid(C), dir = mid - ctr;
    for (const AMate& m : g_am) {  // a part on a shaft or pin slides off along that axis, away from the middle
        if (!m.on || (m.type != MT_CONCENTRIC && m.type != MT_INSERT) || (m.a.comp != c && m.b.comp != c)) continue;
        const ARef& r = m.a.comp == c ? m.a : m.b;
        if (r.kind != 1) continue;
        V3 ax = g_ac[(size_t)r.comp].pose.rot(r.d);
        dir = geo::dot(ax, mid - ctr) >= 0 ? ax : ax * -1.0;
        if (std::fabs(geo::dot(ax, mid - ctr)) < 1e-9) dir = ax.z >= 0 ? ax : ax * -1.0;
        return dir * (s * 0.8 * size);
    }
    double l = geo::len(dir); dir = l > 1e-9 ? dir / l : V3(0, 0, 1);
    return dir * (s * 0.8 * size);
}

// ---------------------------------------------------------------- motion study
static void ms_base() { if (!g_ms_based) { for (AComp& c : g_ac) c.base = c.pose; g_ms_based = true; for (ATrace& t : g_atr) t.path.clear(); gear_resync(true); g_ms_prev_t = 0; } }
static void ms_reset() { if (g_ms_based) { for (AComp& c : g_ac) c.pose = c.base; g_ms_based = false; } g_ms_t = 0; g_ms_play = false; KillTimer(g_main, ASM_TIMER); gear_resync(false); g_ms_prev_t = 0; }
static void ms_eval(double t) {  // the poses at time t: key poses, then motors, then the mates
    ms_base();
    if (t < g_ms_prev_t - 1e-9) gear_resync(true);  // back in time: the gears count their turns from time 0 again
    g_ms_prev_t = t;
    std::vector<char> driven(g_ac.size(), 0);
    for (size_t c = 0; c < g_ac.size(); ++c) g_ac[c].pose = g_ac[c].base;
    for (size_t c = 0; c < g_ac.size(); ++c) {  // key poses: between the two keys around t that hold this component
        const AKey *k0 = nullptr, *k1 = nullptr; Pose p0, p1;
        Pose b = g_ac[c].base;
        for (const AKey& k : g_akeys)
            for (size_t j = 0; j < k.comps.size(); ++j) if (k.comps[j] == (int)c) {
                if (k.t <= t && (!k0 || k.t >= k0->t)) { k0 = &k; p0 = k.poses[j]; }
                if (k.t >= t && (!k1 || k.t <= k1->t)) { k1 = &k; p1 = k.poses[j]; }
            }
        if (!k0 && !k1) continue;
        if (!k0) { k0 = k1; p0 = b; double s = k1->t > 0 ? t / k1->t : 1; g_ac[c].pose = pose_lerp(b, p1, geo::clampd(s, 0, 1)); driven[c] = 1; continue; }
        if (!k1 || k1 == k0) { g_ac[c].pose = p0; driven[c] = 1; continue; }
        double s = (t - k0->t) / std::max(1e-9, k1->t - k0->t);
        g_ac[c].pose = pose_lerp(p0, p1, s * s * (3 - 2 * s)); driven[c] = 1;
    }
    for (const AMotor& m : g_amot) {
        if (m.comp < 0 || m.comp >= (int)g_ac.size()) continue;
        Pose& P = g_ac[(size_t)m.comp].pose;
        V3 pw = g_ac[(size_t)m.comp].base.apply(m.p), dw = g_ac[(size_t)m.comp].base.rot(m.d);
        if (m.linear) P.t = P.t + dw * (m.speed * t);
        else { double M[9]; rot_axis(dw, m.speed * t * geo::PI / 180, M); pose_turn(P, M, pw); }
        driven[(size_t)m.comp] = 1;
    }
    asm_solve(&driven);
    for (ATrace& tr : g_atr) if (tr.comp >= 0 && tr.comp < (int)g_ac.size()) tr.path.push_back(g_ac[(size_t)tr.comp].pose.apply(tr.p));
}
static void ms_play() {
    if (g_ms_play) { g_ms_play = false; KillTimer(g_main, ASM_TIMER); status("motion study paused at " + fmtd(g_ms_t) + " s"); return; }
    if (g_amot.empty() && g_akeys.empty()) { MessageBoxA(g_main, "Nothing moves yet. Add a motor (Assembly > Motor: pick a hole or shaft face on the part that turns) or key poses (move parts, then Key pose).", "Motion study", MB_ICONINFORMATION); return; }
    if (g_ms_t >= g_ms_T - 1e-9) { g_ms_t = 0; for (ATrace& t : g_atr) t.path.clear(); }
    g_ms_play = true; SetTimer(g_main, ASM_TIMER, 33, nullptr);
    status("motion study playing (Space pauses)");
}
static void ms_tick() {
    if (g_explode_to != g_explode) {  // the exploded view moves smoothly
        g_explode += g_explode_to > g_explode ? 0.05 : -0.05;
        if (std::fabs(g_explode - g_explode_to) < 0.051) g_explode = g_explode_to;
        if (!g_ms_play && g_explode == g_explode_to) KillTimer(g_main, ASM_TIMER);
    }
    if (g_ms_play) {
        g_ms_t = std::min(g_ms_T, g_ms_t + 0.033);
        ms_eval(g_ms_t);
        if (g_ms_t >= g_ms_T - 1e-9) { g_ms_play = false; if (g_explode == g_explode_to) KillTimer(g_main, ASM_TIMER); status("motion study finished (" + fmtd(g_ms_T) + " s); traces stay until the next run"); }
    }
    InvalidateRect(g_view, nullptr, FALSE);
}

// ---------------------------------------------------------------- belts and chains
// a wheel of a belt or chain drive: its pitch radius (a chain's on the pitch polygon's mean, z p / 2 pi, so the
// rollers fall in the seats), teeth and pitch for a sprocket; drive 1 pulley, 2 sprocket
static bool wheel_info(int c, double& r, int& z, double& p, int& drive) {
    if (c < 0 || c >= (int)g_ac.size()) return false;
    const AComp& C = g_ac[(size_t)c];
    if (const js::Value* s = C.part.find("sprocket")) { z = (int)s->num("teeth", 0); p = s->num("pitch", 0); r = z * p / (2 * geo::PI); drive = 2; return z > 0 && p > 0; }
    if (const js::Value* q = C.part.find("pulley")) { r = 0.5 * q->num("pitch_d", 0); z = 0; p = 0; drive = 1; return r > 0; }
    return false;
}
// the path of the belt or chain: round a (from the side away from b), along the lower span, round b, back along the
// upper span; counter-clockwise about a's axis n = u x v, u from a's centre towards b's
struct DriveLoop { bool ok = false; V3 cA, cB, n, u, v, TA1, TA2, TB1, TB2; double rA = 0, rB = 0, b = 0, LA = 0, S1 = 0, LB = 0, S2 = 0, L = 0, d = 0, h = 0, axang = 0; };
static DriveLoop drive_loop(const AMate& m) {
    DriveLoop D; int za, zb, da, db; double pa, pb;
    if (!wheel_info(m.a.comp, D.rA, za, pa, da) || !wheel_info(m.b.comp, D.rB, zb, pb, db)) return D;
    const Pose &PA = g_ac[(size_t)m.a.comp].pose, &PB = g_ac[(size_t)m.b.comp].pose;
    D.cA = PA.apply(V3()); D.cB = PB.apply(V3()); D.n = PA.rot(V3(0, 0, 1));
    D.axang = std::acos(geo::clampd(std::fabs(geo::dot(D.n, PB.rot(V3(0, 0, 1)))), 0, 1)) * 180 / geo::PI;
    V3 w = D.cB - D.cA; D.h = geo::dot(w, D.n); w = w - D.n * D.h;
    D.d = geo::len(w);
    if (D.d < 1e-6 || D.d <= std::fabs(D.rB - D.rA) * 1.0001) return D;
    D.u = w / D.d; D.v = geo::cross(D.n, D.u);
    double sb = (D.rB - D.rA) / D.d; D.b = std::asin(sb); double cb = std::cos(D.b);
    V3 w1 = D.u * -sb + D.v * cb, w2 = D.u * -sb - D.v * cb;
    D.TA1 = D.cA + w1 * D.rA; D.TA2 = D.cA + w2 * D.rA; D.TB1 = D.cB + w1 * D.rB; D.TB2 = D.cB + w2 * D.rB;
    D.LA = D.rA * (geo::PI - 2 * D.b); D.S1 = geo::len(D.TB2 - D.TA2); D.LB = D.rB * (geo::PI + 2 * D.b); D.S2 = geo::len(D.TA1 - D.TB1);
    D.L = D.LA + D.S1 + D.LB + D.S2; D.ok = true;
    return D;
}
static V3 drive_at(const DriveLoop& D, double s, V3& t) {  // the point at s along the loop and the direction it runs
    s -= D.L * std::floor(s / D.L);
    auto arc = [&](const V3& c, double r, double th) { t = D.u * -std::sin(th) + D.v * std::cos(th); return c + (D.u * std::cos(th) + D.v * std::sin(th)) * r; };
    if (s < D.LA) return arc(D.cA, D.rA, 0.5 * geo::PI + D.b + s / D.rA);
    s -= D.LA;
    if (s < D.S1) { t = (D.TB2 - D.TA2) / D.S1; return D.TA2 + t * s; }
    s -= D.S1;
    if (s < D.LB) return arc(D.cB, D.rB, -0.5 * geo::PI - D.b + s / D.rB);
    s -= D.LB;
    t = (D.TA1 - D.TB1) / std::max(D.S2, 1e-12); return D.TB1 + t * std::min(s, D.S2);
}
// where the chain's rollers are: one in a seat of the driving sprocket (seats every 2 pi / z from its local x), the
// rest every L / N along the loop
static double chain_phase(const AMate& m, const DriveLoop& D) {
    V3 xa = g_ac[(size_t)m.a.comp].pose.rot(V3(1, 0, 0));
    double al = std::atan2(geo::dot(xa, D.v), geo::dot(xa, D.u));
    return (al - (0.5 * geo::PI + D.b)) * D.rA;
}
static void material_shine(int cls);
// a closed outline (x along e, y along o) extruded along n from z0 to z1: both faces and the sides, lit
static void slab(const std::vector<std::pair<double, double>>& ol, const V3& P, const V3& e, const V3& o, const V3& n, double z0, double z1) {
    auto at = [&](double x, double y, double z) { V3 q = P + e * x + o * y + n * z; glVertex3d(q.x, q.y, q.z); };
    for (int f = 0; f < 2; ++f) {
        double z = f ? z1 : z0; V3 nn = f ? n : n * -1.0;
        glNormal3d(nn.x, nn.y, nn.z); glBegin(GL_POLYGON);
        if (f) for (auto& q : ol) at(q.first, q.second, z); else for (size_t i = ol.size(); i-- > 0;) at(ol[i].first, ol[i].second, z);
        glEnd();
    }
    glBegin(GL_QUADS);
    for (size_t i = 0; i < ol.size(); ++i) {
        const auto &a = ol[i], &b = ol[(i + 1) % ol.size()];
        double dx = b.first - a.first, dy = b.second - a.second, l = std::sqrt(dx * dx + dy * dy); if (l < 1e-12) continue;
        V3 nn = e * (dy / l) + o * (-dx / l); glNormal3d(nn.x, nn.y, nn.z);
        at(a.first, a.second, z0); at(b.first, b.second, z0); at(b.first, b.second, z1); at(a.first, a.second, z1);
    }
    glEnd();
}
static void pin_cyl(const V3& P, const V3& n, const V3& e, double r, double z0, double z1, int seg) {  // a short cylinder about n
    V3 o = geo::cross(n, e);
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= seg; ++i) { double a = 2 * geo::PI * i / seg; V3 rd = e * std::cos(a) + o * std::sin(a); V3 p0 = P + rd * r + n * z0, p1 = P + rd * r + n * z1; glNormal3d(rd.x, rd.y, rd.z); glVertex3d(p0.x, p0.y, p0.z); glVertex3d(p1.x, p1.y, p1.z); }
    glEnd();
    for (int f = 0; f < 2; ++f) {
        double z = f ? z1 : z0; V3 nn = f ? n : n * -1.0; glNormal3d(nn.x, nn.y, nn.z);
        glBegin(GL_POLYGON);
        for (int i = 0; i < seg; ++i) { double a = 2 * geo::PI * (f ? i : seg - i) / seg; V3 q = P + (e * std::cos(a) + o * std::sin(a)) * r + n * z; glVertex3d(q.x, q.y, q.z); }
        glEnd();
    }
}
static void drive_render() {  // the belts and chains, running with their wheels
    if (g_explode > 1e-6) return;
    for (AMate& m : g_am) {
        if (!m.on || m.type != MT_BELT || m.a.comp < 0 || m.b.comp < 0 || m.a.comp >= (int)g_ac.size() || m.b.comp >= (int)g_ac.size()) continue;
        const AComp &A = g_ac[(size_t)m.a.comp], &B = g_ac[(size_t)m.b.comp];
        if (A.hidden || B.hidden || !comp_ready(A) || !comp_ready(B)) continue;
        DriveLoop D = drive_loop(m);
        if (!D.ok) continue;
        double ta = twist_about(A.pose.R, m.rel.R, D.n);
        if (!m.bsync) { m.bacc = 0; m.blast = ta; m.bsync = true; }
        m.bacc += std::remainder(ta - m.blast, 2 * geo::PI); m.blast = ta;
        V3 t;
        if (m.drive != 2) {  // a V-belt: SPZ section, 9.7 mm across the top (2 mm over the pitch line), 8 mm deep, 38 deg
            int ns = std::max(96, std::min(4000, (int)(D.L / 0.002)));
            const double C[4][2] = {{0.002, -0.0047}, {0.002, 0.0047}, {-0.006, 0.0020}, {-0.006, -0.0020}};  // (out, along n)
            glColor3f(0.12f, 0.12f, 0.13f); material_shine(1);
            for (int f = 0; f < 4; ++f) {
                const double *c1 = C[f], *c2 = C[(f + 1) % 4];
                double dh = c2[0] - c1[0], dw = c2[1] - c1[1], nh = dw, nw = -dh;  // the side's outward normal in (out, n)
                double mh = 0.5 * (c1[0] + c2[0]) + 0.002, mw = 0.5 * (c1[1] + c2[1]);
                if (nh * mh + nw * mw < 0) { nh = -nh; nw = -nw; }
                double l = std::sqrt(nh * nh + nw * nw); nh /= l; nw /= l;
                glBegin(GL_QUAD_STRIP);
                for (int i = 0; i <= ns; ++i) {
                    V3 p = drive_at(D, D.L * i / ns, t), o = geo::cross(t, D.n);
                    V3 nn = o * nh + D.n * nw; glNormal3d(nn.x, nn.y, nn.z);
                    V3 q1 = p + o * c1[0] + D.n * c1[1], q2 = p + o * c2[0] + D.n * c2[1];
                    glVertex3d(q1.x, q1.y, q1.z); glVertex3d(q2.x, q2.y, q2.z);
                }
                glEnd();
            }
            // printed marks on the back of the belt, moving with it
            int nm = std::max(6, (int)std::round(D.L / 0.03));
            double shift = D.rA * m.bacc;
            glColor3f(0.62f, 0.62f, 0.58f);
            glBegin(GL_QUADS);
            for (int k = 0; k < nm; ++k) {
                double s = shift + D.L * k / nm;
                V3 p0 = drive_at(D, s, t); V3 o = geo::cross(t, D.n); V3 p1 = p0 + t * 0.005;
                glNormal3d(o.x, o.y, o.z);
                V3 q[4] = {p0 + o * 0.00205 + D.n * -0.003, p1 + o * 0.00205 + D.n * -0.003, p1 + o * 0.00205 + D.n * 0.003, p0 + o * 0.00205 + D.n * 0.003};
                for (V3& v : q) glVertex3d(v.x, v.y, v.z);
            }
            glEnd();
        } else {  // a roller chain: rollers, inner and outer link plates, pins
            const js::Value* sm = A.part.find("sprocket");
            double p = sm->num("pitch", 0.0127), d1 = sm->num("roller_d", 0.0085), b1 = sm->num("inner_width", 0.00775), hp = sm->num("plate_h", 0.0118), tp = sm->num("plate_t", 0.0016);
            int N = std::max(4, (int)std::round(D.L / p));
            double sp = D.L / N, s0 = chain_phase(m, D);
            std::vector<V3> R(N);
            for (int k = 0; k < N; ++k) R[k] = drive_at(D, s0 + sp * k, t);
            material_shine(0);
            for (int k = 0; k < N; ++k) {  // the plates of link k, from roller k to roller k + 1
                V3 P0 = R[k], P1 = R[(k + 1) % N], e = P1 - P0; double le = geo::len(e); if (le < 1e-9) continue; e = e / le;
                V3 o = geo::cross(D.n, e);
                std::vector<std::pair<double, double>> ol;
                double hh = 0.5 * hp;
                for (int i = 0; i <= 8; ++i) { double a = -0.5 * geo::PI + geo::PI * i / 8; ol.push_back({le + hh * std::cos(a), hh * std::sin(a)}); }
                for (int i = 0; i <= 8; ++i) { double a = 0.5 * geo::PI + geo::PI * i / 8; ol.push_back({hh * std::cos(a), hh * std::sin(a)}); }
                bool outer = k % 2 == 1;
                double z0 = 0.5 * b1 + (outer ? tp : 0.0);
                if (outer) glColor3f(0.40f, 0.42f, 0.46f); else glColor3f(0.50f, 0.52f, 0.56f);
                slab(ol, P0, e, o, D.n, z0, z0 + tp);
                slab(ol, P0, e, o, D.n, -z0 - tp, -z0);
            }
            for (int k = 0; k < N; ++k) {  // rollers, and the pins through them
                V3 e = R[(k + 1) % N] - R[k]; double le = geo::len(e); e = le > 1e-9 ? e / le : D.u;
                glColor3f(0.58f, 0.60f, 0.64f); pin_cyl(R[k], D.n, e, 0.5 * d1, -0.5 * b1 + 0.0001, 0.5 * b1 - 0.0001, 14);
                glColor3f(0.56f, 0.58f, 0.62f); pin_cyl(R[k], D.n, e, 0.26 * d1, -0.5 * b1 - 2 * tp - 0.0006, 0.5 * b1 + 2 * tp + 0.0006, 10);
            }
        }
    }
    material_shine(-1);
}
// a belt or chain mate's tree label: the wheels' sizes
static std::string drive_label(const AMate& m) {
    auto sz = [&](int c) { if (c < 0 || c >= (int)g_ac.size()) return std::string("?"); const AComp& C = g_ac[(size_t)c];
        if (const js::Value* s = C.part.find("sprocket")) return "z" + tostr((int)s->num("teeth", 0));
        if (const js::Value* q = C.part.find("pulley")) return "D" + fmtd(std::round(q->num("pitch_d", 0) * 1e4) / 10);
        return std::string("?"); };
    std::string ch; if (m.drive == 2 && m.a.comp >= 0 && m.a.comp < (int)g_ac.size()) if (const js::Value* s = g_ac[(size_t)m.a.comp].part.find("sprocket")) ch = " " + s->str("chain", "");
    return std::string(m.drive == 2 ? "Chain drive" : "V-belt drive") + ch + " " + sz(m.a.comp) + "/" + sz(m.b.comp);
}

// ---------------------------------------------------------------- drawing
// a part's colour: its material's (steel blue-grey, aluminium silver, brass gold, rubber black...), else a palette colour
static int g_draw_cls = -1;  // the material class of the colour last chosen (0 metal, 1 polymer, 2 composite; -1 none)
static bool mat_colour(const js::Value& part, const std::string& over, const std::string& appearance, float* col) {
    std::string key = appearance == "steel" ? std::string("steel_mild") : part_material_key(part, over);
    const EnvMat* m = envmat_of(key);
    if (!m) { g_draw_cls = -1; return false; }
    col[0] = m->r; col[1] = m->g; col[2] = m->b; g_draw_cls = m->cls; return true;
}
static void asm_colour(int c, float* col) {
    static const float PAL[][3] = {{0.62f, 0.70f, 0.80f}, {0.86f, 0.58f, 0.32f}, {0.45f, 0.70f, 0.52f}, {0.72f, 0.56f, 0.82f}, {0.88f, 0.78f, 0.40f}, {0.48f, 0.66f, 0.78f}};
    const AComp& C = g_ac[(size_t)c];
    if (mat_colour(C.part, C.material, C.appearance, col)) return;
    int k = 0; for (int q = 0; q < c; ++q) if (!envmat_of(part_material_key(g_ac[(size_t)q].part, g_ac[(size_t)q].material))) ++k;
    for (int i = 0; i < 3; ++i) col[i] = PAL[k % 6][i];
}
static void piece_colour(int c, int k, float* col) {  // a sub-assembly's parts each get their own colour
    const AComp& C = g_ac[(size_t)c];
    if (!is_sub(C) || k < 0 || k >= (int)C.pieces.size()) { asm_colour(c, col); return; }
    static const float PAL[][3] = {{0.55f, 0.72f, 0.62f}, {0.80f, 0.66f, 0.45f}, {0.60f, 0.64f, 0.82f}, {0.82f, 0.55f, 0.55f}};
    if (mat_colour(C.pieces[(size_t)k].part, "", "", col)) return;
    for (int i = 0; i < 3; ++i) col[i] = PAL[k % 4][i];
}
static void material_shine(int cls) {  // metals glossy, plastics and rubber matte, composites and wood satin
    float s = cls == 0 ? 0.50f : cls == 1 ? 0.10f : cls == 2 ? 0.24f : 0.42f, sh = cls == 0 ? 72.f : cls == 1 ? 10.f : cls == 2 ? 24.f : 48.f;
    GLfloat ms[4] = {s, s, s, 1}; glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, ms); glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, sh);
}
static void gl_pose(const Pose& P, const V3& off) {
    double m[16] = {P.R[0], P.R[3], P.R[6], 0, P.R[1], P.R[4], P.R[7], 0, P.R[2], P.R[5], P.R[8], 0, P.t.x + off.x, P.t.y + off.y, P.t.z + off.z, 1};
    glMultMatrixd(m);
}
static void draw_thread(const js::Value& part) {  // the cosmetic thread of a bolt or screw: a helix on the shank
    const js::Value* f = part.find("fastener");
    if (!f) return;
    double z0 = f->num("thread_from", 0), z1 = f->num("thread_to", 0), p = f->num("pitch", 0), r = 0.5 * f->num("d", 0) * 1.004;
    if (z1 <= z0 || p <= 0 || r <= 0) return;
    int n = (int)std::min(6000.0, (z1 - z0) / p * 36);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= n; ++i) { double z = z0 + (z1 - z0) * i / n, a = 2 * geo::PI * (z - z0) / p; glVertex3d(r * std::cos(a), r * std::sin(a), z); }
    glEnd();
}
static void asm_render() {
    glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(1, 1);
    for (size_t c = 0; c < g_ac.size(); ++c) {
        const AComp& C = g_ac[c];
        if (C.hidden) continue;
        glPushMatrix(); gl_pose(C.pose, explode_offset((int)c));
        bool sel = (int)c == g_asm_sel;
        for (size_t k = 0; k < C.pieces.size(); ++k) {
            const APiece& pc = C.pieces[k];
            if (!pc.mesh) continue;  // still meshing
            float col[3]; piece_colour((int)c, (int)k, col);
            material_shine(g_draw_cls);
            if (sel) { col[0] = col[0] * 0.8f + 0.25f * 0.2f; col[1] = col[1] * 0.8f + 0.6f * 0.2f; col[2] = col[2] * 0.8f + 1.0f * 0.2f; }
            glColor3fv(col);
            glPushMatrix(); gl_pose(pc.local, V3());
            const geo::Mesh& M = pc.mesh->m;
            glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_NORMAL_ARRAY);
            static std::vector<float> vb, nb;
            vb.resize(M.v.size() * 3); nb.resize(M.v.size() * 3);
            for (size_t i = 0; i < M.v.size(); ++i) { vb[3 * i] = (float)M.v[i].x; vb[3 * i + 1] = (float)M.v[i].y; vb[3 * i + 2] = (float)M.v[i].z; nb[3 * i] = (float)M.n[i].x; nb[3 * i + 1] = (float)M.n[i].y; nb[3 * i + 2] = (float)M.n[i].z; }
            glVertexPointer(3, GL_FLOAT, 0, vb.data()); glNormalPointer(GL_FLOAT, 0, nb.data());
            glDrawElements(GL_TRIANGLES, (GLsizei)M.t.size(), GL_UNSIGNED_INT, M.t.data());
            glDisableClientState(GL_NORMAL_ARRAY); glDisableClientState(GL_VERTEX_ARRAY);
            glPopMatrix();
        }
        glPopMatrix();
    }
    material_shine(-1);
    glDisable(GL_POLYGON_OFFSET_FILL);
    drive_render();
    // edges and cosmetic threads
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    for (size_t c = 0; c < g_ac.size(); ++c) {
        const AComp& C = g_ac[c];
        if (C.hidden) continue;
        glPushMatrix(); gl_pose(C.pose, explode_offset((int)c));
        bool sel = (int)c == g_asm_sel;
        for (const APiece& pc : C.pieces) {
            if (!pc.mesh) continue;
            glPushMatrix(); gl_pose(pc.local, V3());
            glLineWidth(sel ? 1.8f : 1.3f);
            if (sel) glColor4f(0.05f, 0.35f, 0.85f, 0.95f); else glColor4f(0.07f, 0.11f, 0.2f, 0.85f);
            const std::vector<float>& cr = pc.mesh->creases;
            if (!cr.empty()) { glEnableClientState(GL_VERTEX_ARRAY); glVertexPointer(3, GL_FLOAT, 0, cr.data()); glDrawArrays(GL_LINES, 0, (GLsizei)(cr.size() / 3)); glDisableClientState(GL_VERTEX_ARRAY); }
            glLineWidth(1.0f); glColor4f(0.18f, 0.2f, 0.25f, 0.75f);
            draw_thread(pc.part);
            glPopMatrix();
        }
        glPopMatrix();
    }
    // explode lines: dashed, from each part's place to where it is drawn
    if (g_explode > 1e-6) {
        glEnable(GL_LINE_STIPPLE); glLineStipple(2, 0x00FF); glColor4f(0.85f, 0.35f, 0.1f, 0.9f); glLineWidth(1.4f);
        glBegin(GL_LINES);
        for (size_t c = 0; c < g_ac.size(); ++c) {
            V3 off = explode_offset((int)c);
            if (geo::len(off) < 1e-9 || !comp_ready(g_ac[c])) continue;
            V3 mid = comp_mid(g_ac[c]);
            glVertex3d(mid.x, mid.y, mid.z); glVertex3d(mid.x + off.x, mid.y + off.y, mid.z + off.z);
        }
        glEnd(); glDisable(GL_LINE_STIPPLE); glLineWidth(1.0f);
    }
    // traces: the path of each traced point
    for (const ATrace& tr : g_atr) {
        glLineWidth(2.2f); glColor4f(0.95f, 0.45f, 0.05f, 0.95f);
        glBegin(GL_LINE_STRIP); for (const V3& p : tr.path) glVertex3d(p.x, p.y, p.z); glEnd();
        if (tr.comp >= 0 && tr.comp < (int)g_ac.size()) {
            V3 p = g_ac[(size_t)tr.comp].pose.apply(tr.p) + explode_offset(tr.comp);
            glPointSize(7.0f); glBegin(GL_POINTS); glVertex3d(p.x, p.y, p.z); glEnd(); glPointSize(1.0f);
        }
    }
    glLineWidth(1.0f);
    // the selected mate: its two references
    auto draw_ref = [&](const ARef& r, float cr, float cg, float cb) {
        if (r.comp < 0 || r.comp >= (int)g_ac.size()) return;
        WRef w = world(r); V3 off = explode_offset(r.comp); w.p = w.p + off;
        const AComp& C = g_ac[(size_t)r.comp];
        double s = 0.25 * comp_size(C);
        glColor4f(cr, cg, cb, 0.95f); glLineWidth(2.4f);
        if (r.kind == 1) { glBegin(GL_LINES); V3 a = w.p - w.d * (2 * s), b = w.p + w.d * (2 * s); glVertex3d(a.x, a.y, a.z); glVertex3d(b.x, b.y, b.z); glEnd(); }
        else if (r.kind == 0) {
            V3 u = std::fabs(w.d.x) < 0.9 ? geo::cross(w.d, V3(1, 0, 0)) : geo::cross(w.d, V3(0, 1, 0)); u = u / geo::len(u); V3 v = geo::cross(w.d, u);
            glBegin(GL_LINE_LOOP); for (int k = 0; k < 4; ++k) { V3 q = w.p + u * (s * (k == 0 || k == 3 ? -1 : 1)) + v * (s * (k < 2 ? -1 : 1)); glVertex3d(q.x, q.y, q.z); } glEnd();
            glBegin(GL_LINES); glVertex3d(w.p.x, w.p.y, w.p.z); V3 q = w.p + w.d * s; glVertex3d(q.x, q.y, q.z); glEnd();
        }
        glPointSize(8.0f); glBegin(GL_POINTS); glVertex3d(w.p.x, w.p.y, w.p.z); glEnd(); glPointSize(1.0f);
        glLineWidth(1.0f);
    };
    glDisable(GL_DEPTH_TEST);
    if (g_asm_mate_sel >= 0 && g_asm_mate_sel < (int)g_am.size()) { draw_ref(g_am[(size_t)g_asm_mate_sel].a, 0.1f, 0.6f, 0.2f); draw_ref(g_am[(size_t)g_asm_mate_sel].b, 0.9f, 0.3f, 0.1f); }
    if (g_asm_pick == 2 || g_asm_pick == 4) draw_ref(g_pick_ref[0], 0.1f, 0.45f, 0.95f);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_LINE_SMOOTH); glDisable(GL_BLEND); glEnable(GL_LIGHTING);
}
static void asm_info(char* buf, size_t n) {  // the line under the view in an assembly
    double vol = 0; int tris = 0, ready = 0, fast = 0;
    for (const AComp& c : g_ac) { if (c.part.find("fastener")) ++fast; if (!comp_ready(c)) continue; ++ready; for (const APiece& pc : c.pieces) { vol += pc.mesh->volume; tris += (int)(pc.mesh->m.t.size() / 3); } }
    snprintf(buf, n, "assembly: %d components (%d fasteners), %d mates   volume %.4g m3   %d triangles%s%s", (int)g_ac.size(), fast, (int)g_am.size(), vol, tris,
             ready < (int)g_ac.size() ? "   meshing..." : "", g_explode > 1e-6 ? "   [exploded]" : "");
}
// the timeline along the bottom of the view (y up from the bottom, as in the overlay)
static const int TL_Y = 30, TL_H = 16;
static void asm_overlay(int W, int H) {
    (void)H;
    if (!g_asm_on) return;
    bool motion = !g_amot.empty() || !g_akeys.empty() || g_ms_based;
    if (!motion) return;
    float x0 = 96, x1 = (float)W - 24;
    glColor3f(0.86f, 0.88f, 0.92f); glBegin(GL_QUADS); glVertex2f(x0, TL_Y); glVertex2f(x1, TL_Y); glVertex2f(x1, TL_Y + TL_H); glVertex2f(x0, TL_Y + TL_H); glEnd();
    glColor3f(0.55f, 0.6f, 0.68f); glBegin(GL_LINES);
    for (int k = 0; k <= (int)(g_ms_T * 4); ++k) { float x = x0 + (x1 - x0) * (float)(k / 4.0 / g_ms_T); glVertex2f(x, TL_Y); glVertex2f(x, TL_Y + (k % 4 == 0 ? TL_H : 5)); }
    glEnd();
    float xt = x0 + (x1 - x0) * (float)(g_ms_t / g_ms_T);
    glColor3f(0.2f, 0.5f, 0.95f); glBegin(GL_QUADS); glVertex2f(x0, TL_Y + 6); glVertex2f(xt, TL_Y + 6); glVertex2f(xt, TL_Y + 10); glVertex2f(x0, TL_Y + 10); glEnd();
    glColor3f(0.9f, 0.4f, 0.05f);
    for (const AKey& k : g_akeys) { float x = x0 + (x1 - x0) * (float)(k.t / g_ms_T); glBegin(GL_QUADS); glVertex2f(x, TL_Y + 2); glVertex2f(x + 5, TL_Y + 8); glVertex2f(x, TL_Y + 14); glVertex2f(x - 5, TL_Y + 8); glEnd(); }
    glColor3f(0.05f, 0.25f, 0.7f); glLineWidth(2.0f); glBegin(GL_LINES); glVertex2f(xt, TL_Y - 3); glVertex2f(xt, TL_Y + TL_H + 3); glEnd(); glLineWidth(1.0f);
    char b[160]; snprintf(b, sizeof b, "%s  t = %.2f / %.1f s   motors %d   keys %d   traces %d", g_ms_play ? "PLAYING" : "motion", g_ms_t, g_ms_T, (int)g_amot.size(), (int)g_akeys.size(), (int)g_atr.size());
    glColor3f(0.2f, 0.24f, 0.3f); gl_text(x0, TL_Y + TL_H + 5, b);
}
static bool asm_timeline_hit(int x, int ywin) {  // ywin: window y (down); click or drag on the timeline scrubs
    if (!g_asm_on || (g_amot.empty() && g_akeys.empty() && !g_ms_based)) return false;
    RECT rc; GetClientRect(g_view, &rc);
    int y = rc.bottom - ywin;
    if (y < TL_Y - 4 || y > TL_Y + TL_H + 4 || x < 90 || x > rc.right - 18) return false;
    double s = geo::clampd((x - 96.0) / std::max(1.0, rc.right - 24.0 - 96.0), 0, 1);
    g_ms_play = false; g_ms_t = s * g_ms_T;
    for (ATrace& t : g_atr) t.path.clear();
    int n = std::max(1, (int)(g_ms_t / 0.04));  // step there from time 0: traces drawn, gears counting every turn
    for (int k = 0; k <= n; ++k) ms_eval(g_ms_t * k / n);
    InvalidateRect(g_view, nullptr, FALSE);
    return true;
}

// ---------------------------------------------------------------- the model tree in an assembly
static int asm_rows() { return (int)g_ac.size() + (g_am.empty() ? 0 : 1 + (int)g_am.size()) + (g_amot.empty() ? 0 : 1 + (int)g_amot.size()); }
static void asm_fill_list() {
    SendMessageA(g_list, LB_RESETCONTENT, 0, 0);
    for (int i = 0; i < asm_rows(); ++i) SendMessageA(g_list, LB_ADDSTRING, 0, (LPARAM)"row");
    int sel = g_asm_mate_sel >= 0 ? (int)g_ac.size() + 1 + g_asm_mate_sel : g_asm_sel;
    SendMessageA(g_list, LB_SETCURSEL, sel, 0);
}
static void asm_tree_draw(DRAWITEMSTRUCT* d) {
    RECT rc = d->rcItem;
    int W = rc.right - rc.left, H = rc.bottom - rc.top;
    if (W <= 0 || H <= 0) return;
    gp::Bitmap buf(W, H, PixelFormat32bppARGB);
    gp::Graphics g(&buf);
    quality(g);
    bool sel = (d->itemState & ODS_SELECTED) != 0;
    g.Clear(sel ? gcol(204, 224, 247) : gcol(255, 255, 255));
    int k = (int)d->itemID, nc = (int)g_ac.size();
    std::string icon, id, kind;
    gp::Color ink = gcol(25, 28, 34);
    bool header = false;
    if (k >= 0 && k < nc) {
        const AComp& C = g_ac[(size_t)k];
        const js::Value* f = C.part.find("fastener");
        icon = f ? (f->str("kind", "") == "nut" || f->str("kind", "") == "washer" ? "torus" : "cylinder") : "box";
        id = C.id; kind = (C.fixed ? "(fixed)  " : "") + (envmat_of(C.material) ? std::string(envmat_of(C.material)->label) + "  " : std::string()) + C.part.str("name", f ? f->str("kind", "part") : "part") + (is_sub(C) ? "  [sub-assembly, " + tostr((int)C.pieces.size()) + " parts]" : "");
        if (is_sub(C)) icon = "asm_new"; else if (C.part.find("gear")) icon = "asm_gear";
        if (C.hidden) ink = gcol(160, 165, 172);
        float c[3]; asm_colour(k, c);
        gp::SolidBrush sw(gcol((int)(c[0] * 255), (int)(c[1] * 255), (int)(c[2] * 255)));
        g.FillRectangle(&sw, 36.0f, (H - 10) / 2.0f, 10.0f, 10.0f);
    } else if (k == nc && !g_am.empty()) { header = true; id = "Mates"; kind = tostr((int)g_am.size()); }
    else if (k > nc && k <= nc + (int)g_am.size()) {
        const AMate& m = g_am[(size_t)(k - nc - 1)];
        icon = m.type == MT_CAM ? "asm_cam" : m.type == MT_BELT ? (m.drive == 2 ? "asm_chain" : "asm_belt") : m.type == MT_ROLL ? "asm_roller" : m.type == MT_GEAR ? "asm_gear_mate" : m.type == MT_CONCENTRIC || m.type == MT_INSERT ? "cut_hole" : m.type == MT_ANGLE ? "pivot" : "plane_xy";
        auto nm = [&](int c) { return c >= 0 && c < nc ? g_ac[(size_t)c].id : std::string("?"); };
        id = std::string(MATE_LABEL[m.type]) + (m.type == MT_DISTANCE ? " " + fmtd(m.value * 1000) + " mm" : m.type == MT_ANGLE ? " " + fmtd(m.value) + " deg" : m.type == MT_GEAR ? (m.zb == 0 ? std::string(" rack & pinion") : " " + tostr(m.za) + ":" + tostr(m.zb)) : m.type == MT_ROLL ? " " + fmtd(std::round(m.value * 1000) / 1000) + ":1" : "");
        if (m.type == MT_BELT) id = drive_label(m);
        if (m.type == MT_CAM && m.a.comp >= 0 && m.a.comp < nc) if (const js::Value* cm = g_ac[(size_t)m.a.comp].part.find("cam")) id = "Cam (" + cm->str("law", "") + ", " + fmtd(cm->num("lift", 0) * 1000) + " mm lift)";
        kind = nm(m.a.comp) + " / " + nm(m.b.comp) + (m.on ? "" : "  (off)");
        if (!m.on) ink = gcol(160, 165, 172);
    } else {
        int base = nc + (g_am.empty() ? 0 : 1 + (int)g_am.size());
        if (k == base) { header = true; id = "Motion study"; kind = fmtd(g_ms_T) + " s"; }
        else if (k > base && k - base - 1 < (int)g_amot.size()) {
            const AMotor& m = g_amot[(size_t)(k - base - 1)];
            icon = "kinematics"; id = m.linear ? "Linear motor" : "Rotary motor";
            kind = (m.comp >= 0 && m.comp < nc ? g_ac[(size_t)m.comp].id : "?") + "  " + fmtd(m.speed) + (m.linear ? " m/s" : " deg/s");
        }
    }
    if (header) {
        gtext(g, id, gp::RectF(8, 0, (float)W - 16, (float)H), gcol(30, 34, 40), 12.0f, gp::FontStyleBold);
        gtext(g, kind, gp::RectF(8, 0, (float)W - 16, (float)H), gcol(120, 126, 136), 11.5f, gp::FontStyleRegular, gp::StringAlignmentFar);
    } else if (!id.empty()) {
        gp::Pen exp(gcol(150, 155, 162), 1.0f); g.DrawLine(&exp, 8.0f, H / 2.0f, 13.0f, H / 2.0f);
        if (gp::Bitmap* ic = icon_flat(icon)) g.DrawImage(ic, 16.0f, (H - 16) / 2.0f, 16.0f, 16.0f);
        float x = k < nc ? 52.0f : 38.0f;
        float tw = text_w(g, id, 12.5f, sel ? gp::FontStyleBold : gp::FontStyleRegular);
        gtext(g, id, gp::RectF(x, 0, (float)W - x - 2, (float)H), ink, 12.5f, sel ? gp::FontStyleBold : gp::FontStyleRegular);
        gtext(g, kind, gp::RectF(x + 4 + tw, 0, std::max(0.0f, W - x - 6 - tw), (float)H), gcol(120, 126, 136), 11.5f);
    }
    gp::Graphics out(d->hDC);
    out.DrawImage(&buf, (INT)rc.left, (INT)rc.top);
}
static void asm_tree_select(int row) {
    int nc = (int)g_ac.size();
    g_asm_sel = -1; g_asm_mate_sel = -1;
    if (row >= 0 && row < nc) { g_asm_sel = row; const AComp& C = g_ac[(size_t)row]; status(C.id + ": " + C.part.str("name", "part") + (C.fixed ? " (fixed)" : "") + "   Shift+drag moves it, Ctrl+drag turns it; right-click for more"); }
    else if (row > nc && row <= nc + (int)g_am.size()) { g_asm_mate_sel = row - nc - 1; status(std::string("mate: ") + MATE_LABEL[g_am[(size_t)g_asm_mate_sel].type] + " (green: first reference, orange: second); right-click to edit, switch off or delete"); }
    InvalidateRect(g_view, nullptr, FALSE);
}

// ---------------------------------------------------------------- commands
enum { ID_ASMC_FIX = 16400, ID_ASMC_FLOAT, ID_ASMC_HIDE, ID_ASMC_DEL, ID_ASMC_DUP, ID_ASMC_OPEN, ID_ASMC_MATE_ONOFF, ID_ASMC_MATE_DEL, ID_ASMC_MATE_VALUE, ID_ASMC_MATE_FLIP,
       ID_ASMC_MOTOR_SPEED, ID_ASMC_MOTOR_DEL, ID_ASMC_MATE0 = 16450, ID_ASMC_MAT0 = 16600 };
static int asm_unique(const std::string& base) {
    for (int k = 1;; ++k) { std::string id = base + tostr(k); bool used = false; for (const AComp& c : g_ac) if (c.id == id) used = true; if (!used) return k; }
}
static void asm_add_component(js::Value part, const std::string& base, const std::string& file, const Pose* at) {
    AComp c;
    c.id = base + tostr(asm_unique(base));
    c.part = part; if (!c.part.find("features")) c.part["features"] = js::Value::array();
    c.file = file; c.key = js::dump(c.part, 0);
    build_pieces(c);
    c.fixed = g_ac.empty();  // the first component is the ground, as in other CAD programs
    if (at) c.pose = *at;
    else if (!g_ac.empty()) {  // beside the others
        geo::Box b; for (const AComp& o : g_ac) comp_world_box(o, b);
        geo::Model m; try { m.load(c.part); } catch (...) {}
        double w = m.bounds.valid() ? m.bounds.hi.x - m.bounds.lo.x : 0.1;
        if (b.valid()) c.pose.t = V3(b.hi.x + 0.6 * w + 0.02 - (m.bounds.valid() ? m.bounds.lo.x : 0), 0.5 * (b.lo.y + b.hi.y), b.hi.z);
    }
    c.base = c.pose;
    g_ac.push_back(c);
    g_asm_sel = (int)g_ac.size() - 1; g_asm_mate_sel = -1;
    asm_changed();
    status("inserted " + c.id + (is_sub(c) ? " (a rigid sub-assembly of " + tostr((int)c.pieces.size()) + " parts)" : "") + (c.fixed ? " (fixed: the ground of the assembly)" : "") + ". Mate it: Assembly > Mate (pick two faces), or Insert into hole for fasteners");
}
static void asm_new() {
    if (g_modified && MessageBoxA(g_main, "Start a new assembly? Unsaved changes to the current design are lost (save first if you need them).", "New assembly", MB_OKCANCEL | MB_ICONQUESTION) != IDOK) return;
    js::Value d = js::Value::object(); d["name"] = "assembly"; d["units"] = "m"; d["features"] = js::Value::array();
    js::Value a = js::Value::object(); a["components"] = js::Value::array(); a["mates"] = js::Value::array(); d["assembly"] = a;
    g_asm_loaded.clear();
    set_doc(d, "");
    status("new assembly: insert parts (Assembly > Insert part: a design file, or the current design) and fasteners");
}
static void asm_insert_part() {
    std::string p;
    if (!ask_path(false, "Part design (*.json)\0*.json\0All files\0*.*\0", "json", p)) return;
    try {
        js::Value part = js::parse(js::read_file(p));
        std::string base = p.substr(p.find_last_of("\\/") + 1); base = base.substr(0, base.find('.'));
        asm_add_component(part, base, p, nullptr);
    } catch (std::exception& e) { MessageBoxA(g_main, e.what(), "Insert part", MB_ICONERROR); }
}
static void asm_insert_fastener(const std::string& kind) {
    std::string spec = kind == "nut" || kind == "washer" ? "M8" : kind == "shaft" ? "M20x120" : "M8x30";
    std::string prompt = "ISO metric size" + std::string(kind == "nut" || kind == "washer" ? " (M3 ... M20)" : " and length in mm, e.g. M8x30 (M3 ... M20)") + ":";
    if (!g_selftest && !ask_text(g_main, ("Insert " + kind).c_str(), prompt.c_str(), spec)) return;
    std::string s = spec; for (char& ch : s) ch = (char)toupper((unsigned char)ch);
    size_t x = s.find('X');
    std::string sz = s.substr(0, x);
    double L = x == std::string::npos ? 30 : atof(s.substr(x + 1).c_str());
    int ti = -1; for (int k = 0; k < NTHREAD; ++k) if (sz == THREADS[k].name) ti = k;
    if (ti < 0) { MessageBoxA(g_main, ("Unknown size '" + spec + "'. Use M3, M4, M5, M6, M8, M10, M12, M16 or M20.").c_str(), "Fastener", MB_ICONWARNING); return; }
    if (L <= 0) L = 30;
    asm_add_component(fastener_part(kind, THREADS[ti], L), kind, "", nullptr);
}
static void asm_from_design() {  // the design open in the editor becomes the first component of a new assembly
    js::Value part = g_doc.clone();
    if (part.find("assembly")) { status("already an assembly"); return; }
    std::string base = g_path.empty() ? std::string("part") : g_path.substr(g_path.find_last_of("\\/") + 1);
    base = base.substr(0, base.find('.'));
    std::string file = g_path;
    asm_new();
    if (!g_asm_on) return;
    asm_add_component(part, base, file, nullptr);
}
static void asm_mate_menu(int sx, int sy) {  // after two picks: the mates that fit the two references
    const ARef &a = g_pick_ref[0], &b = g_pick_ref[1];
    HMENU m = CreatePopupMenu();
    bool ax = a.kind == 1 && b.kind == 1, pl = a.kind == 0 && b.kind == 0;
    auto add = [&](int t, bool ok) { if (ok) AppendMenuA(m, MF_STRING, ID_ASMC_MATE0 + t, MATE_LABEL[t]); };
    std::string head = std::string("Mate ") + (a.kind == 0 ? "plane" : a.kind == 1 ? "axis (r " + fmtd(a.r * 1000) + " mm)" : "point") + " to " + (b.kind == 0 ? "plane" : b.kind == 1 ? "axis (r " + fmtd(b.r * 1000) + " mm)" : "point");
    AppendMenuA(m, MF_STRING | MF_GRAYED, 0, head.c_str()); AppendMenuA(m, MF_SEPARATOR, 0, nullptr);
    add(MT_INSERT, ax); add(MT_CONCENTRIC, ax); add(MT_ROLL, ax && a.r > 0 && b.r > 0); add(MT_COINCIDENT, true); add(MT_DISTANCE, pl || (a.kind == 0) != (b.kind == 0));
    add(MT_PARALLEL, a.kind != 2 && b.kind != 2); add(MT_PERP, a.kind != 2 && b.kind != 2); add(MT_ANGLE, a.kind != 2 && b.kind != 2); add(MT_LOCK, true);
    TrackPopupMenu(m, TPM_LEFTALIGN | TPM_TOPALIGN | TPM_RIGHTBUTTON, sx, sy, 0, g_main, nullptr);
    DestroyMenu(m);
}
// the end of a cylindrical face nearer to a point: the entry face of a hole (or the end of a shaft)
static void axis_end(ARef& r, const V3& towards_local) {
    double s0 = r.value_ends[0], s1 = r.value_ends[1], s = geo::dot(towards_local, r.d);
    V3 axisPt = r.p - r.d * geo::dot(r.p, r.d);
    bool hi = std::fabs(s - s1) < std::fabs(s - s0);
    r.fp = axisPt + r.d * (hi ? s1 : s0); r.fn = hi ? r.d : r.d * -1.0;
    r.face = true;
}
static void asm_make_mate(int type) {
    AMate m; m.type = type; m.a = g_pick_ref[0]; m.b = g_pick_ref[1];
    m.id = std::string(MATE_NAME[type]) + tostr((int)g_am.size() + 1);
    if (g_ac[(size_t)m.a.comp].fixed == false && g_ac[(size_t)m.b.comp].fixed) std::swap(m.a, m.b);  // the fixed part stays put
    if (type == MT_INSERT) {
        // the target's entry face: the end of the hole nearer the moving part; the moving part's seat: a fastener's
        // z = 0 face, otherwise the end of its cylinder nearer the target
        const AComp &CA = g_ac[(size_t)m.a.comp], &CB = g_ac[(size_t)m.b.comp];
        V3 bCentre = comp_mid(CB);
        axis_end(m.a, CA.pose.inv(bCentre));
        if (CB.part.find("fastener")) {
            // a fastener keeps the way it faces: its head (+z) comes out of the hole end that faces most nearly the same way
            V3 zl = CA.pose.inv_rot(CB.pose.rot(V3(0, 0, 1)));
            if (std::fabs(geo::dot(zl, m.a.d)) > 0.2) axis_end(m.a, m.a.p + m.a.d * (geo::dot(zl, m.a.d) > 0 ? 1e6 : -1e6));
            m.b.p = V3(); m.b.d = V3(0, 0, 1); m.b.fp = V3(); m.b.fn = V3(0, 0, -1); m.b.face = true;
        }
        else {
            V3 aEntry = CA.pose.apply(m.a.fp);
            axis_end(m.b, CB.pose.inv(aEntry));
            m.b.fn = m.b.fn * -1.0;  // the seat faces the target
        }
    }
    if (type == MT_DISTANCE || type == MT_ANGLE) {
        WRef a = world(m.a), b = world(m.b);
        double cur = type == MT_DISTANCE ? std::fabs(geo::dot(b.p - a.p, a.d)) * 1000 : unit_angle(a.d, b.d) * 180 / geo::PI;
        std::string v = fmtd(std::round(cur * 100) / 100);
        if (!g_selftest && !ask_text(g_main, MATE_LABEL[type], type == MT_DISTANCE ? "Distance between the faces (mm):" : "Angle between the faces (degrees):", v)) { g_asm_pick = 0; return; }
        m.value = type == MT_DISTANCE ? atof(v.c_str()) / 1000 : atof(v.c_str());
    }
    if (type == MT_GEAR) {
        if (is_rack(m.a.comp) && !is_rack(m.b.comp)) std::swap(m.a, m.b);  // the pinion is a, the rack b
        if (is_worm(m.b.comp) && !is_worm(m.a.comp)) std::swap(m.a, m.b);  // the worm is a
        if (is_worm(m.a.comp) && is_worm(m.b.comp)) { status("two worms cannot mesh: pick a worm and its worm wheel"); g_asm_pick = 0; return; }
        if (is_rack(m.a.comp)) { status("two racks cannot mesh: pick a gear and a rack"); g_asm_pick = 0; return; }
        m.a.p = V3(); m.a.d = V3(0, 0, 1); m.b.p = V3(); m.b.d = V3(0, 0, 1);
        const js::Value *ga = gear_meta(m.a.comp), *gb = gear_meta(m.b.comp);
        AComp &GA = g_ac[(size_t)m.a.comp], &GB = g_ac[(size_t)m.b.comp];
        m.za = (int)ga->num("teeth", 0);
        double ma = ga->num("module", 0), mb = gb->num("module", 0), rA = ga->num("pitch_radius", 0);
        V3 nA = GA.pose.rot(V3(0, 0, 1)), cA = gear_centre(m.a.comp);
        std::string note;
        if (is_rack(m.b.comp)) {
            m.zb = 0; m.rp = rA; m.b.d = V3(1, 0, 0);
            V3 xb = GB.pose.rot(V3(1, 0, 0)), yb = GB.pose.rot(V3(0, 1, 0));
            V3 uA = yb * -1.0; uA = uA - nA * geo::dot(uA, nA);
            double lu = geo::len(uA); uA = lu > 1e-9 ? uA / lu : V3(1, 0, 0);
            V3 C = cA + uA * rA;
            m.sense = geo::dot(geo::cross(nA, C - cA), xb) >= 0 ? 1 : -1;
            double dA = m.za > 0 ? std::remainder(axis_angle(nA, uA, GA.pose.rot(V3(1, 0, 0))), 2 * geo::PI / m.za) : 0;
            double p = gb->num("pitch", geo::PI * mb), xc = geo::dot(C - GB.pose.t, xb);
            double t = std::remainder(xc + m.sense * dA * rA - 0.5 * p, p);  // a rack gap under the pinion's nearest tooth
            GB.pose.t = GB.pose.t + xb * t;
            double off = geo::dot(cA - GB.pose.t, yb);
            if (std::fabs(off - rA) > 0.02 * rA) note = "; note: the pinion's centre is " + fmtd(std::round(off * 1e5) / 100) + " mm from the rack's pitch line, it should be " + fmtd(std::round(rA * 1e5) / 100) + " mm";
        } else if (is_worm(m.a.comp)) {  // worm and wheel: crossed axes; the thread pushes the wheel's pitch point along the worm's axis
            m.zb = (int)gb->num("teeth", 0);
            double rB = gb->num("pitch_radius", 0), hnd = ga->num("hand", 1) < 0 ? -1 : 1, Ld = ga->num("lead", 0), px = geo::PI * ma, r1 = ga->num("pitch_radius", 0);
            if (gb->str("kind", "") != "worm_wheel") note = "; warning: a worm needs a worm wheel (Gears > Worm wheel), this is a " + gb->str("kind", "") + " gear";
            V3 nB = GB.pose.rot(V3(0, 0, 1)), cB = GB.pose.t;
            V3 P = cA + nA * geo::dot(cB - cA, nA), uB = P - cB; uB = uB - nB * geo::dot(uB, nB);
            double dist = geo::len(uB);
            if (dist > 1e-9 && rB > 0 && m.zb > 0) {
                uB = uB / dist;
                V3 C = cB + uB * rB, tB = geo::cross(nB, uB);
                double s = geo::dot(tB, nA) >= 0 ? 1 : -1;
                m.sense = -hnd * s;  // a right-hand thread turning + about its axis moves the contact along -axis
                V3 Cl = GA.pose.inv(C);
                double Dl = std::remainder(hnd * Ld * std::atan2(Cl.y, Cl.x) / (2 * geo::PI) + 0.5 * px - Cl.z, px);  // the nearest thread gap, along the axis
                double dl = std::remainder(axis_angle(nB, uB, GB.pose.rot(V3(1, 0, 0))), 2 * geo::PI / m.zb);       // the wheel's nearest tooth
                double psi = std::remainder(Dl * s / rB - dl, 2 * geo::PI / m.zb);
                double M2[9]; rot_axis(nB, psi, M2); pose_turn(GB.pose, M2, GB.pose.t);
            }
            double axang = std::acos(geo::clampd(std::fabs(geo::dot(nA, nB)), 0, 1)) * 180 / geo::PI;
            if (std::fabs(axang - 90) > 1) note += "; note: a worm and its wheel need axes at 90 deg, these are " + fmtd(std::round(axang * 10) / 10);
            else if (std::fabs(dist - (r1 + rB)) > 0.02 * (r1 + rB)) note += "; note: the centre distance is " + fmtd(std::round(dist * 1e5) / 100) + " mm, it should be " + fmtd(std::round((r1 + rB) * 1e5) / 100) + " mm";
            if (gb->num("worm_starts", m.za) != m.za || std::fabs(gb->num("worm_q", ga->num("q", 10)) - ga->num("q", 10)) > 1e-6) note += "; warning: this wheel was cut for another worm (starts or diameter factor differ)";
            if (gb->num("hand", hnd) * hnd < 0) note += "; warning: the worm and the wheel are of opposite hands";
            note += "; lead angle " + fmtd(std::round(ga->num("lead_angle", 0) * 100) / 100) + " deg, ratio " + tostr(m.zb) + ":" + tostr(m.za);
        } else {
            m.zb = (int)gb->num("teeth", 0);
            double rB = gb->num("pitch_radius", 0);
            V3 nB = GB.pose.rot(V3(0, 0, 1)), cB = gear_centre(m.b.comp);
            V3 uA = cB - cA; uA = uA - nA * geo::dot(uA, nA);
            double dist = geo::len(uA);
            if (dist > 1e-9 && m.za > 0 && m.zb > 0) {
                uA = uA / dist;
                V3 C = cA + uA * rA, uB = C - cB; uB = uB - nB * geo::dot(uB, nB);
                double lb = geo::len(uB); uB = lb > 1e-9 ? uB / lb : uA * -1.0;
                m.sense = geo::dot(geo::cross(nA, C - cA), geo::cross(nB, C - cB)) >= 0 ? 1 : -1;  // b's turn that moves its pitch point the same way
                double dA = std::remainder(axis_angle(nA, uA, GA.pose.rot(V3(1, 0, 0))), 2 * geo::PI / m.za);  // A's nearest tooth off the contact
                double gap = axis_angle(nB, uB, GB.pose.rot(V3(1, 0, 0))) + geo::PI / m.zb;                       // a gap of B towards the contact
                double corr = std::remainder(m.sense * dA * m.za / m.zb - gap, 2 * geo::PI / m.zb);
                double M2[9]; rot_axis(nB, corr, M2); pose_turn(GB.pose, M2, GB.pose.t);
            }
            double ha = ga->num("helix", 0), hb = gb->num("helix", 0), ca = ga->num("cone", 0), cb = gb->num("cone", 0);
            double axang = std::acos(geo::clampd(std::fabs(geo::dot(nA, nB)), 0, 1)) * 180 / geo::PI;
            if (ca > 0 || cb > 0) {
                if (std::fabs(axang - (ca + cb)) > 1.5) note = "; note: bevel gears with cones " + fmtd(ca) + " and " + fmtd(cb) + " deg need axes " + fmtd(ca + cb) + " deg apart, these are " + fmtd(std::round(axang * 10) / 10);
            } else {
                if (axang > 1) note = "; note: these spur or helical gears' axes are not parallel";
                else if (std::fabs(ha + hb) > 0.5) note = "; note: an outside helical pair needs equal and opposite helix angles (" + fmtd(ha) + " and " + fmtd(hb) + " deg)";
                else if (std::fabs(dist - (rA + rB)) > 0.02 * (rA + rB)) note = "; note: the centre distance is " + fmtd(std::round(dist * 1e5) / 100) + " mm, it should be " + fmtd(std::round((rA + rB) * 1e5) / 100) + " mm";
            }
        }
        m.rel = GA.pose; m.rel2 = GB.pose;
        if (std::fabs(ma - mb) > 1e-9) note = "; warning: the modules differ (" + fmtd(ma * 1000) + " and " + fmtd(mb * 1000) + " mm): they cannot mesh";
        if (!note.empty()) g_st_last_note = note;
    }
    if (type == MT_ROLL) {
        WRef a = world(m.a), b = world(m.b);
        double rA = m.a.r, rB = m.b.r;
        if (m.a.kind != 1 || m.b.kind != 1 || rA <= 0 || rB <= 0) { status("a roller mate needs two round faces (pick the rollers' outer surfaces)"); g_asm_pick = 0; return; }
        m.value = rA / rB;
        V3 uA = b.p - a.p; uA = uA - a.d * geo::dot(uA, a.d);
        double dist = geo::len(uA); uA = dist > 1e-12 ? uA / dist : V3(1, 0, 0);
        V3 C = a.p + uA * rA;
        V3 rb = C - b.p; rb = rb - b.d * geo::dot(rb, b.d);
        m.sense = geo::dot(geo::cross(a.d, C - a.p), geo::cross(b.d, rb)) >= 0 ? 1 : -1;
        if (std::fabs(dist - (rA + rB)) > 0.02 * (rA + rB)) g_st_last_note = "; note: the rollers' axes are " + fmtd(std::round(dist * 1e5) / 100) + " mm apart, touching needs " + fmtd(std::round((rA + rB) * 1e5) / 100) + " mm";
        m.rel = g_ac[(size_t)m.a.comp].pose; m.rel2 = g_ac[(size_t)m.b.comp].pose;
    }
    if (type == MT_BELT) {
        double rA, rB, pa, pb; int za, zb, da, db;
        if (!wheel_info(m.a.comp, rA, za, pa, da) || !wheel_info(m.b.comp, rB, zb, pb, db) || da != db) { status("a belt drive needs two pulleys, a chain drive two sprockets"); g_asm_pick = 0; return; }
        m.drive = da;
        m.a.kind = m.b.kind = 1; m.a.p = V3(); m.a.d = V3(0, 0, 1); m.b.p = V3(); m.b.d = V3(0, 0, 1); m.a.r = rA; m.b.r = rB;
        AComp &GA = g_ac[(size_t)m.a.comp], &GB = g_ac[(size_t)m.b.comp];
        V3 nA = GA.pose.rot(V3(0, 0, 1)), nB = GB.pose.rot(V3(0, 0, 1));
        m.sense = geo::dot(nA, nB) >= 0 ? 1 : -1;  // an open belt: both turn the same way
        m.value = m.drive == 2 ? (double)za / zb : rA / rB;
        DriveLoop D = drive_loop(m);
        if (!D.ok) { status("the wheels overlap or share an axis: move them apart first"); g_asm_pick = 0; return; }
        std::string note;
        const double wrap = (geo::PI - 2 * std::fabs(D.b)) * 180 / geo::PI;
        if (m.drive == 2) {
            const js::Value *sa = GA.part.find("sprocket"), *sb = GB.part.find("sprocket");
            if (sa->str("chain", "") != sb->str("chain", "")) { status("the sprockets are for different chains (" + sa->str("chain", "") + ", " + sb->str("chain", "") + ")"); g_asm_pick = 0; return; }
            int N = (int)std::round(D.L / pa);
            // turn b so a roller sits in each of its seats
            double s0 = chain_phase(m, D), sp = D.L / N, sm = D.LA + D.S1 + 0.5 * D.LB;
            double k = std::round((sm - s0) / sp); V3 t; V3 R = drive_at(D, s0 + k * sp, t);
            double dl = std::remainder(axis_angle(nB, GB.pose.rot(V3(1, 0, 0)), R - D.cB), 2 * geo::PI / zb);
            double M2[9]; rot_axis(nB, dl, M2); pose_turn(GB.pose, M2, GB.pose.t);
            note = "; chain " + sa->str("chain", "") + ", " + tostr(N) + " links (" + fmtd(std::round(D.L * 1e4) / 10) + " mm), wrap " + fmtd(std::round(wrap)) + " deg on the small sprocket";
            if (std::fabs(D.L / pa - N) > 0.02) {  // the centre distance for a whole number of links (on the mean pitch circles)
                double lo = std::fabs(D.rB - D.rA) * 1.0001, hi = D.d * 4 + 1, want = N * pa;
                for (int it = 0; it < 80; ++it) { double c = 0.5 * (lo + hi), b = std::asin((D.rB - D.rA) / c); double L = 2 * c * std::cos(b) + D.rA * (geo::PI - 2 * b) + D.rB * (geo::PI + 2 * b); (L < want ? lo : hi) = c; }
                note += "; for exactly " + tostr(N) + " links set the centre distance to " + fmtd(std::round(lo * 1e5) / 100) + " mm (now " + fmtd(std::round(D.d * 1e5) / 100) + ")";
            }
            if (std::min(za, zb) < 17) note += "; the " + tostr(std::min(za, zb)) + "-tooth sprocket gives " + fmtd(std::round((1 - std::cos(geo::PI / std::min(za, zb))) * 1000) / 10) + " % speed ripple (17 or more teeth run smoother)";
        } else {
            note = "; belt pitch length " + fmtd(std::round(D.L * 1e4) / 10) + " mm, wrap " + fmtd(std::round(wrap)) + " deg on the small pulley";
            if (2 * std::min(rA, rB) < 0.063 - 1e-9) note += "; warning: SPZ belts need pulleys of 63 mm or more (the belt bends too hard)";
            if (wrap < 120) note += "; warning: less than 120 deg of wrap: the belt slips easily";
        }
        if (D.axang > 0.5) note += "; warning: the axes are " + fmtd(std::round(D.axang * 10) / 10) + " deg out of parallel: the " + (m.drive == 2 ? "chain" : "belt") + " would run off";
        else if (std::fabs(D.h) > 0.0005) note += "; warning: the wheels are " + fmtd(std::round(std::fabs(D.h) * 1e4) / 10) + " mm out of line";
        g_st_last_note = note;
        m.rel = GA.pose; m.rel2 = GB.pose;
    }
    if (type == MT_CAM) {
        if (!g_ac[(size_t)m.a.comp].part.find("cam") && g_ac[(size_t)m.b.comp].part.find("cam")) std::swap(m.a, m.b);
        AComp &CA = g_ac[(size_t)m.a.comp], &CB = g_ac[(size_t)m.b.comp];
        const js::Value* cm = CA.part.find("cam");
        if (!cm || !CB.part.find("follower")) { status("a cam mate needs a cam (Cams > Disc cam) and a follower (Cams > Follower)"); g_asm_pick = 0; return; }
        V3 nA = CA.pose.rot(V3(0, 0, 1)), w = CB.pose.t - CA.pose.t;
        m.cax = geo::dot(w, nA); w = w - nA * m.cax;
        if (geo::len(w) < 1e-6) { status("place the follower beside the cam (off its axis) first"); g_asm_pick = 0; return; }
        m.cdir = w / geo::len(w);
        V3 xw = geo::cross(m.cdir, nA);
        Pose& P = CB.pose;  // the follower: its stem (local y) along the line, its roller axis (local z) along the cam's axis
        P.R[0] = xw.x; P.R[1] = m.cdir.x; P.R[2] = nA.x; P.R[3] = xw.y; P.R[4] = m.cdir.y; P.R[5] = nA.y; P.R[6] = xw.z; P.R[7] = m.cdir.z; P.R[8] = nA.z;
        m.a.kind = m.b.kind = 1; m.a.p = V3(); m.a.d = V3(0, 0, 1); m.b.p = V3(); m.b.d = V3(0, 1, 0);
        m.rel = CA.pose; m.rel2 = CB.pose;
        CamDef cd = cam_def(*cm); CamCheck k; cam_outline(cd, 360, k);
        const js::Value* fm = CB.part.find("follower");
        std::string note = "; " + cam_report(cd, k);
        if (fm->str("kind", "") == "roller" && (cd.flat || std::fabs(fm->num("roller_r", 0) - cd.rr) > 1e-5)) note += "; warning: the cam was cut for a " + (cd.flat ? std::string("flat-faced follower") : "roller of " + fmtd(cd.rr * 2000) + " mm") + ", this follower differs";
        if (fm->str("kind", "") == "flat" && !cd.flat) note += "; warning: the cam was cut for a roller follower";
        if (fm->str("kind", "") == "flat" && fm->num("face_halfwidth", 0) < k.face_w) note += "; warning: the face is too narrow: the contact moves " + fmtd(std::round(k.face_w * 1e4) / 10) + " mm off centre";
        g_st_last_note = note;
    }
    if (type == MT_LOCK) {  // b in a's frame
        const Pose &Pa = g_ac[(size_t)m.a.comp].pose, &Pb = g_ac[(size_t)m.b.comp].pose;
        double Rt[9] = {Pa.R[0], Pa.R[3], Pa.R[6], Pa.R[1], Pa.R[4], Pa.R[7], Pa.R[2], Pa.R[5], Pa.R[8]};
        mat_mul(Rt, Pb.R, m.rel.R); m.rel.t = Pa.inv(Pb.t);
    }
    push_undo();
    g_am.push_back(m);
    ms_reset();
    asm_solve();
    g_asm_pick = 0; g_asm_mate_sel = (int)g_am.size() - 1; g_asm_sel = -1;
    asm_changed(false);
    status(std::string(MATE_LABEL[type]) + " mate added between " + g_ac[(size_t)m.a.comp].id + " and " + g_ac[(size_t)m.b.comp].id + g_st_last_note);
    g_st_last_note.clear();
}
static void asm_component_menu(int sx, int sy) {
    if (g_asm_sel < 0 || g_asm_sel >= (int)g_ac.size()) return;
    const AComp& C = g_ac[(size_t)g_asm_sel];
    HMENU m = CreatePopupMenu();
    AppendMenuA(m, MF_STRING | MF_GRAYED, 0, (C.id + "  " + C.part.str("name", "")).c_str()); AppendMenuA(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(m, MF_STRING | (C.fixed ? MF_CHECKED : 0), C.fixed ? ID_ASMC_FLOAT : ID_ASMC_FIX, C.fixed ? "Fixed (click to float)" : "Fix in place");
    HMENU mm = CreatePopupMenu();
    for (int i = 0; i < NENVMAT; ++i) AppendMenuA(mm, MF_STRING | (C.material == ENVMATS[i].key ? MF_CHECKED : 0), ID_ASMC_MAT0 + i, ENVMATS[i].label);
    AppendMenuA(mm, MF_SEPARATOR, 0, nullptr); AppendMenuA(mm, MF_STRING | (C.material.empty() ? MF_CHECKED : 0), ID_ASMC_MAT0 + NENVMAT, "From the part (default)");
    AppendMenuA(m, MF_POPUP, (UINT_PTR)mm, "Material (environment analysis)");
    AppendMenuA(m, MF_STRING, ID_ASMC_HIDE, C.hidden ? "Show" : "Hide");
    AppendMenuA(m, MF_STRING, ID_ASMC_DUP, "Duplicate");
    if (!C.file.empty()) AppendMenuA(m, MF_STRING, ID_ASMC_OPEN, "Open the part file (edit it)");
    AppendMenuA(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(m, MF_STRING, ID_ASMC_DEL, "Delete (and its mates)");
    TrackPopupMenu(m, TPM_LEFTALIGN | TPM_TOPALIGN | TPM_RIGHTBUTTON, sx, sy, 0, g_main, nullptr);
    DestroyMenu(m);
}
static void asm_mate_context(int sx, int sy) {
    if (g_asm_mate_sel < 0 || g_asm_mate_sel >= (int)g_am.size()) return;
    const AMate& mt = g_am[(size_t)g_asm_mate_sel];
    HMENU m = CreatePopupMenu();
    if (mt.type == MT_DISTANCE || mt.type == MT_ANGLE) AppendMenuA(m, MF_STRING, ID_ASMC_MATE_VALUE, mt.type == MT_DISTANCE ? "Change the distance..." : "Change the angle...");
    if (mt.type == MT_COINCIDENT || mt.type == MT_DISTANCE || mt.type == MT_CONCENTRIC) AppendMenuA(m, MF_STRING, ID_ASMC_MATE_FLIP, "Flip alignment");
    AppendMenuA(m, MF_STRING, ID_ASMC_MATE_ONOFF, mt.on ? "Suppress" : "Unsuppress");
    AppendMenuA(m, MF_STRING, ID_ASMC_MATE_DEL, "Delete");
    TrackPopupMenu(m, TPM_LEFTALIGN | TPM_TOPALIGN | TPM_RIGHTBUTTON, sx, sy, 0, g_main, nullptr);
    DestroyMenu(m);
}
static void asm_delete_component(int c) {
    push_undo();
    std::vector<AMate> keep;
    for (AMate m : g_am) {
        if (m.a.comp == c || m.b.comp == c) continue;
        if (m.a.comp > c) --m.a.comp;
        if (m.b.comp > c) --m.b.comp;
        keep.push_back(m);
    }
    g_am = keep;
    std::vector<AMotor> km; for (AMotor m : g_amot) { if (m.comp == c) continue; if (m.comp > c) --m.comp; km.push_back(m); } g_amot = km;
    std::vector<ATrace> kt; for (ATrace t : g_atr) { if (t.comp == c) continue; if (t.comp > c) --t.comp; kt.push_back(t); } g_atr = kt;
    for (AKey& k : g_akeys) { AKey n; n.t = k.t; for (size_t j = 0; j < k.comps.size(); ++j) { if (k.comps[j] == c) continue; n.comps.push_back(k.comps[j] > c ? k.comps[j] - 1 : k.comps[j]); n.poses.push_back(k.poses[j]); } k = n; }
    g_ac.erase(g_ac.begin() + c);
    g_asm_sel = -1; g_asm_mate_sel = -1;
    asm_changed(false);
}
static bool asm_command(int id) {
    switch (id) {
        case ID_ASMC_FIX: case ID_ASMC_FLOAT: if (g_asm_sel >= 0) { push_undo(); g_ac[(size_t)g_asm_sel].fixed = id == ID_ASMC_FIX; asm_changed(false); } return true;
        case ID_ASMC_HIDE: if (g_asm_sel >= 0) { push_undo(); g_ac[(size_t)g_asm_sel].hidden = !g_ac[(size_t)g_asm_sel].hidden; asm_changed(false); } return true;
        case ID_ASMC_DEL: if (g_asm_sel >= 0) asm_delete_component(g_asm_sel); return true;
        case ID_ASMC_DUP: if (g_asm_sel >= 0) { AComp c = g_ac[(size_t)g_asm_sel]; Pose p = c.pose; double s = comp_size(c); p.t = p.t + V3(0.6 * s, 0.6 * s, 0); std::string base = c.id; while (!base.empty() && isdigit((unsigned char)base.back())) base.pop_back(); js::Value part = c.part.clone(); asm_add_component(part, base, c.file, &p); g_ac.back().fixed = false; asm_store(); } return true;
        case ID_ASMC_OPEN: if (g_asm_sel >= 0 && !g_ac[(size_t)g_asm_sel].file.empty()) { char b[MAX_PATH]; GetModuleFileNameA(nullptr, b, MAX_PATH); std::string a = std::string(1, '"') + g_ac[(size_t)g_asm_sel].file + std::string(1, '"'); ShellExecuteA(g_main, "open", b, a.c_str(), nullptr, SW_SHOWNORMAL); } return true;
        case ID_ASMC_MATE_ONOFF: if (g_asm_mate_sel >= 0) { push_undo(); g_am[(size_t)g_asm_mate_sel].on = !g_am[(size_t)g_asm_mate_sel].on; asm_solve(); asm_changed(false); } return true;
        case ID_ASMC_MATE_DEL: if (g_asm_mate_sel >= 0) { push_undo(); g_am.erase(g_am.begin() + g_asm_mate_sel); g_asm_mate_sel = -1; asm_changed(false); } return true;
        case ID_ASMC_MATE_FLIP: if (g_asm_mate_sel >= 0) { push_undo(); g_am[(size_t)g_asm_mate_sel].aligned = !g_am[(size_t)g_asm_mate_sel].aligned; asm_solve(); asm_changed(false); } return true;
        case ID_ASMC_MATE_VALUE: if (g_asm_mate_sel >= 0) {
            AMate& m = g_am[(size_t)g_asm_mate_sel];
            std::string v = fmtd(m.type == MT_DISTANCE ? m.value * 1000 : m.value);
            if (ask_text(g_main, MATE_LABEL[m.type], m.type == MT_DISTANCE ? "Distance (mm):" : "Angle (degrees):", v)) { push_undo(); m.value = m.type == MT_DISTANCE ? atof(v.c_str()) / 1000 : atof(v.c_str()); asm_solve(); asm_changed(false); }
            return true;
        }
    }
    if (id >= ID_ASMC_MATE0 && id < ID_ASMC_MATE0 + NMT) { asm_make_mate(id - ID_ASMC_MATE0); return true; }
    if (id >= ID_ASMC_MAT0 && id <= ID_ASMC_MAT0 + NENVMAT && g_asm_sel >= 0 && g_asm_sel < (int)g_ac.size()) {
        push_undo(); g_ac[(size_t)g_asm_sel].material = id < ID_ASMC_MAT0 + NENVMAT ? ENVMATS[id - ID_ASMC_MAT0].key : ""; asm_changed(false);
        status(g_ac[(size_t)g_asm_sel].id + ": material " + (g_ac[(size_t)g_asm_sel].material.empty() ? std::string("from the part") : std::string(ENVMATS[id - ID_ASMC_MAT0].label)));
        return true;
    }
    return false;
}

// ---------------------------------------------------------------- mouse
static bool asm_mouse(UINT msg, int x, int y, WPARAM keys) {
    if (!g_asm_on) return false;
    if (msg == WM_LBUTTONDOWN && asm_timeline_hit(x, y)) { g_asm_drag = 3; SetCapture(g_view); return true; }
    if (msg == WM_LBUTTONDOWN && g_asm_pick) {
        V3 hw, hl, nl, hp, np; int tri = -1, pc = -1;
        int c = asm_pick_full(x, y, hw, hl, nl, tri, pc, hp, np);
        if (c < 0 || !comp_ready(g_ac[(size_t)c])) { status("no part under the mouse: click a face of a part (right-click cancels)"); return true; }
        if (g_asm_pick == 7 || g_asm_pick == 8) {  // gear mate: two gears
            if (!g_ac[(size_t)c].part.find("gear")) { status("click a gear (Gears > Spur gear makes one)"); return true; }
            ARef r; r.comp = c; r.kind = 1; r.p = V3(); r.d = V3(0, 0, 1);
            if (g_asm_pick == 7) { g_pick_ref[0] = r; g_asm_pick = 8; status("first gear: " + g_ac[(size_t)c].id + ". Now click the gear it meshes with"); InvalidateRect(g_view, nullptr, FALSE); return true; }
            if (c == g_pick_ref[0].comp) { status("click a different gear"); return true; }
            g_pick_ref[1] = r; asm_make_mate(MT_GEAR); return true;
        }
        if (g_asm_pick == 15 || g_asm_pick == 16) {  // cam, then its follower
            bool want_cam = g_asm_pick == 15;
            if (!g_ac[(size_t)c].part.find(want_cam ? "cam" : "follower")) { status(want_cam ? "click a cam (Cams > Disc cam makes one)" : "click a follower (Cams > Follower makes one)"); return true; }
            ARef r; r.comp = c; r.kind = 1; r.p = V3(); r.d = V3(0, 0, 1);
            if (want_cam) { g_pick_ref[0] = r; g_asm_pick = 16; status("cam: " + g_ac[(size_t)c].id + ". Now the follower (placed beside the cam, where its line should run)"); InvalidateRect(g_view, nullptr, FALSE); return true; }
            g_pick_ref[1] = r; asm_make_mate(MT_CAM); return true;
        }
        if (g_asm_pick >= 11 && g_asm_pick <= 14) {  // belt (pulleys) or chain (sprockets) drive
            bool chain = g_asm_pick >= 13;
            if (!g_ac[(size_t)c].part.find(chain ? "sprocket" : "pulley")) { status(chain ? "click a sprocket (Belts & chains > Sprocket makes one)" : "click a pulley (Belts & chains > Pulley makes one)"); return true; }
            ARef r; r.comp = c; r.kind = 1; r.p = V3(); r.d = V3(0, 0, 1);
            if (g_asm_pick == 11 || g_asm_pick == 13) { g_pick_ref[0] = r; ++g_asm_pick; status(std::string("driving ") + (chain ? "sprocket: " : "pulley: ") + g_ac[(size_t)c].id + ". Now the one it drives"); InvalidateRect(g_view, nullptr, FALSE); return true; }
            if (c == g_pick_ref[0].comp) { status(chain ? "click a different sprocket" : "click a different pulley"); return true; }
            g_pick_ref[1] = r; asm_make_mate(MT_BELT); return true;
        }
        if (g_asm_pick == 5) {  // trace point
            push_undo(); ATrace t; t.comp = c; t.p = hl; g_atr.push_back(t); g_asm_pick = 0; asm_changed(false);
            status("trace added on " + g_ac[(size_t)c].id + ": play the motion study to draw its path"); return true;
        }
        ARef r = recognise(c, pc, tri, hp, np);
        if (g_asm_pick == 9 || g_asm_pick == 10) {  // roller mate: two round faces
            if (r.kind != 1) { status("roller mate: click the round outer face of a roller"); return true; }
            if (g_asm_pick == 9) { g_pick_ref[0] = r; g_asm_pick = 10; status("first roller: " + g_ac[(size_t)c].id + ", diameter " + fmtd(std::round(r.r * 2e5) / 100) + " mm. Now the roller it drives"); InvalidateRect(g_view, nullptr, FALSE); return true; }
            if (c == g_pick_ref[0].comp) { status("click a different roller"); return true; }
            g_pick_ref[1] = r; asm_make_mate(MT_ROLL); return true;
        }
        if (g_asm_pick == 6) {  // motor axis
            if (r.kind != 1) { status("pick a round face (a hole or a shaft) for the motor's axis"); return true; }
            push_undo(); AMotor m; m.comp = c; m.p = r.p; m.d = r.d; m.speed = 90; g_amot.push_back(m); g_asm_pick = 0; ms_reset(); asm_changed(false);
            status("rotary motor on " + g_ac[(size_t)c].id + " (90 deg/s, right-click it in the tree to change): press Play"); return true;
        }
        int slot = g_asm_pick == 1 || g_asm_pick == 3 ? 0 : 1;
        if (slot == 1 && c == g_pick_ref[0].comp) { status("pick the second face on a different part"); return true; }
        if ((g_asm_pick == 3 || g_asm_pick == 4) && r.kind != 1) { status("insert: pick a round face (the bolt's shank, then the hole)"); return true; }
        g_pick_ref[slot] = r;
        std::string what = r.kind == 0 ? "plane" : r.kind == 1 ? std::string(r.hole ? "hole" : "shaft") + " axis, diameter " + fmtd(2000 * r.r) + " mm" : "point";
        if (slot == 0) { g_asm_pick = g_asm_pick == 1 ? 2 : 4; status("first: " + g_ac[(size_t)c].id + " " + what + ". Now pick the face on the other part"); InvalidateRect(g_view, nullptr, FALSE); return true; }
        if (g_asm_pick == 4) {  // insert: the first pick (the fastener) moves into the second (the hole)
            std::swap(g_pick_ref[0], g_pick_ref[1]);
            asm_make_mate(MT_INSERT); return true;
        }
        POINT p = {x, y}; ClientToScreen(g_view, &p);
        asm_mate_menu(p.x, p.y);
        g_asm_pick = 0;
        InvalidateRect(g_view, nullptr, FALSE);
        return true;
    }
    if (msg == WM_RBUTTONDOWN && g_asm_pick) { g_asm_pick = 0; status("picking cancelled"); InvalidateRect(g_view, nullptr, FALSE); return true; }
    if (msg == WM_LBUTTONDOWN && (keys & (MK_SHIFT | MK_CONTROL))) {
        V3 hw, hl, nl; int tri = -1;
        int c = asm_pick_at(x, y, hw, hl, nl, tri);
        if (c < 0) return false;
        if (g_ac[(size_t)c].fixed) { status(g_ac[(size_t)c].id + " is fixed (right-click > Float to move it)"); return true; }
        push_undo(); ms_reset();
        g_asm_drag = (keys & MK_CONTROL) ? 2 : 1; g_asm_drag_comp = c; g_asm_drag_anchor = hw;
        g_asm_sel = c; g_asm_mate_sel = -1; asm_fill_list();
        SetCapture(g_view); g_mx = x; g_my = y;
        return true;
    }
    if (msg == WM_MOUSEMOVE && g_asm_drag == 3) { asm_timeline_hit(x, y); return true; }
    if (msg == WM_MOUSEMOVE && g_asm_drag && g_asm_drag_comp >= 0) {
        int dx = x - g_mx, dy = y - g_my; g_mx = x; g_my = y;
        Pose& P = g_ac[(size_t)g_asm_drag_comp].pose;
        double yr = g_yaw * geo::PI / 180, pr = g_pitch * geo::PI / 180;
        V3 fwd(-std::cos(pr) * std::cos(yr), -std::cos(pr) * std::sin(yr), -std::sin(pr)), right = geo::cross(fwd, V3(0, 0, 1));
        right = right / std::max(1e-9, geo::len(right)); V3 up = geo::cross(right, fwd);
        if (g_asm_drag == 1) {
            double s = 2 * g_dist * std::tan(16.0 * geo::PI / 180) / std::max(1, g_vp[3]);
            V3 dv = right * (dx * s) + up * (-dy * s);
            // on a shaft or in a hole the part can only turn: turn it about that axis towards the mouse
            const ARef* axr = nullptr;
            for (const AMate& m : g_am)
                if (m.on && (m.type == MT_CONCENTRIC || m.type == MT_INSERT) && (m.a.comp == g_asm_drag_comp || m.b.comp == g_asm_drag_comp)) {
                    const ARef& r = m.a.comp == g_asm_drag_comp ? m.a : m.b;
                    if (r.kind == 1) { axr = &r; break; }
                }
            if (axr) {
                WRef w = world(*axr);
                V3 a0 = g_asm_drag_anchor - w.p, a1 = a0 + dv;
                a0 = a0 - w.d * geo::dot(a0, w.d); a1 = a1 - w.d * geo::dot(a1, w.d);
                if (geo::len(a0) > 1e-9 && geo::len(a1) > 1e-9) {
                    double ang = std::atan2(geo::dot(geo::cross(a0, a1), w.d), geo::dot(a0, a1));
                    double M[9]; rot_axis(w.d, ang, M); pose_turn(P, M, w.p);
                    g_asm_drag_anchor = mat_vec(M, g_asm_drag_anchor - w.p) + w.p;
                }
            } else { P.t = P.t + dv; g_asm_drag_anchor = g_asm_drag_anchor + dv; }
        } else {
            double M[9]; rot_axis(up, dx * 0.01, M); pose_turn(P, M, g_asm_drag_anchor);
            rot_axis(right, dy * 0.01, M); pose_turn(P, M, g_asm_drag_anchor);
        }
        std::vector<char> drv(g_ac.size(), 0); drv[(size_t)g_asm_drag_comp] = 1;
        asm_solve(&drv);  // what is mated to it follows
        asm_solve();      // then it settles back onto its own mates
        InvalidateRect(g_view, nullptr, FALSE);
        return true;
    }
    if ((msg == WM_LBUTTONUP) && g_asm_drag) { g_asm_drag = 0; g_asm_drag_comp = -1; ReleaseCapture(); asm_store(); g_modified = true; update_title(); return true; }
    return false;
}
static bool asm_click(UINT msg, int x, int y) {  // a click (no drag) in the view
    if (!g_asm_on) return false;
    V3 hw, hl, nl; int tri = -1;
    int c = asm_pick_at(x, y, hw, hl, nl, tri);
    g_asm_sel = c; g_asm_mate_sel = -1; asm_fill_list();
    if (c >= 0) asm_tree_select(c); else status("nothing under the mouse");
    if (msg == WM_RBUTTONUP && c >= 0) { POINT p = {x, y}; ClientToScreen(g_view, &p); asm_component_menu(p.x, p.y); }
    InvalidateRect(g_view, nullptr, FALSE);
    return true;
}
static void asm_fit() {
    geo::Box b;
    for (const AComp& c : g_ac) comp_world_box(c, b);
    if (!b.valid()) return;
    g_tx = 0.5 * (b.lo.x + b.hi.x); g_ty = 0.5 * (b.lo.y + b.hi.y); g_tz = 0.5 * (b.lo.z + b.hi.z);
    g_dist = 1.9 * geo::len(b.hi - b.lo);
}

// ---------------------------------------------------------------- a worked example: a bracket with a turning arm
static void asm_demo() {
    asm_new();
    if (!g_asm_on) return;
    auto box = [](const char* id, V3 size, V3 at, const char* op) { js::Value f = js::Value::object(); f["id"] = id; f["type"] = "box"; f["op"] = op; f["size"] = v3json(size); f["round"] = 0.0; f["translate"] = v3json(at); return f; };
    auto cyl = [](const char* id, double r, double L, V3 at, const char* op) { js::Value f = js::Value::object(); f["id"] = id; f["type"] = "cylinder"; f["op"] = op; f["axis"] = "z"; f["r"] = r; f["length"] = L; f["translate"] = v3json(at); return f; };
    // the base plate: 200 x 120 x 12 mm with four M8 clearance holes and a 20 mm bore for the shaft
    js::Value plate = js::Value::object(), pf = js::Value::array();
    plate["name"] = "base plate"; plate["units"] = "m";
    pf.push(box("plate", V3(0.2, 0.12, 0.012), V3(0, 0, -0.006), "add"));
    double hx[4] = {-0.08, 0.08, -0.08, 0.08}, hy[4] = {-0.04, -0.04, 0.04, 0.04};
    for (int k = 0; k < 4; ++k) pf.push(cyl(("hole" + tostr(k + 1)).c_str(), 0.0045, 0.05, V3(hx[k], hy[k], -0.006), "cut"));
    pf.push(cyl("bore", 0.0102, 0.05, V3(0, 0, -0.006), "cut"));
    plate["features"] = pf;
    asm_add_component(plate, "plate", "", nullptr);
    // the arm: 140 x 30 x 10 mm with a 20 mm bore at one end and a 10 mm hole at the other
    js::Value arm = js::Value::object(), af = js::Value::array();
    arm["name"] = "arm"; arm["units"] = "m";
    af.push(box("arm", V3(0.14, 0.03, 0.01), V3(0.055, 0, 0.005), "add"));
    af.push(cyl("hub", 0.022, 0.01, V3(0, 0, 0.005), "add"));
    af.push(cyl("bore", 0.0102, 0.05, V3(0, 0, 0.005), "cut"));
    af.push(cyl("eye", 0.005, 0.05, V3(0.11, 0, 0.005), "cut"));
    arm["features"] = af;
    Pose pa; pa.t = V3(0.0, 0.25, 0.05);
    asm_add_component(arm, "arm", "", &pa);
    Pose ps; ps.t = V3(0.25, 0.0, -0.15);  // below the plate, upside down: it goes in from underneath
    ps.R[4] = -1; ps.R[8] = -1;
    asm_add_component(fastener_part("shaft", THREADS[8], 60), "shaft", "", &ps);
    for (int k = 0; k < 4; ++k) { Pose pb; pb.t = V3(-0.1 + 0.06 * k, -0.2, 0.06); asm_add_component(fastener_part("bolt", THREADS[4], 25), "bolt", "", &pb); }
    // mates, as a user would make them: shaft into the plate's bore, arm onto the shaft, bolts into the holes
    auto comp_axis = [&](int c, V3 pl, double r, double s0, double s1, bool hole) { ARef q; q.comp = c; q.kind = 1; q.p = pl; q.d = V3(0, 0, 1); q.r = r; q.value_ends[0] = s0; q.value_ends[1] = s1; q.hole = hole; return q; };
    g_pick_ref[0] = comp_axis(0, V3(0, 0, -0.006), 0.0102, -0.012, 0.0, true);
    g_pick_ref[1] = comp_axis(2, V3(0, 0, -0.03), 0.01, -0.06, 0.0, false);
    asm_make_mate(MT_INSERT);
    // the arm's bore over the shaft, its underside 5 mm above the plate
    g_pick_ref[0] = comp_axis(2, V3(0, 0, -0.03), 0.01, -0.06, 0.0, false);
    g_pick_ref[1] = comp_axis(1, V3(0, 0, 0.005), 0.0102, 0.0, 0.01, true);
    asm_make_mate(MT_CONCENTRIC);
    ARef top; top.comp = 0; top.kind = 0; top.p = V3(0.05, 0, 0); top.d = V3(0, 0, 1);
    ARef under; under.comp = 1; under.kind = 0; under.p = V3(0.05, 0, 0); under.d = V3(0, 0, -1);
    g_pick_ref[0] = top; g_pick_ref[1] = under;
    { AMate m; m.type = MT_DISTANCE; m.a = top; m.b = under; m.value = 0.035; m.id = "distance"; g_am.push_back(m); }
    asm_solve();
    {   // the arm turns with the shaft: locked at its pose relative to the shaft once both are placed
        AMate m; m.type = MT_LOCK; m.a.comp = 2; m.b.comp = 1; m.id = "lock";
        const Pose &Pa = g_ac[2].pose, &Pb = g_ac[1].pose;
        double Rt[9] = {Pa.R[0], Pa.R[3], Pa.R[6], Pa.R[1], Pa.R[4], Pa.R[7], Pa.R[2], Pa.R[5], Pa.R[8]};
        mat_mul(Rt, Pb.R, m.rel.R); m.rel.t = Pa.inv(Pb.t);
        g_am.push_back(m);
    }
    for (int k = 0; k < 4; ++k) {
        g_pick_ref[0] = comp_axis(0, V3(hx[k], hy[k], -0.006), 0.0045, -0.012, 0.0, true);
        g_pick_ref[1] = comp_axis(3 + k, V3(0, 0, -0.01), 0.004, -0.025, 0.0, false);
        asm_make_mate(MT_INSERT);
    }
    // motion: the shaft turns at 90 deg/s; trace the arm's far eye
    AMotor mo; mo.comp = 2; mo.p = V3(); mo.d = V3(0, 0, 1); mo.speed = 90; g_amot.push_back(mo);
    ATrace tr; tr.comp = 1; tr.p = V3(0.11, 0, 0.01); g_atr.push_back(tr);
    g_ms_T = 4;
    g_asm_sel = -1; g_asm_mate_sel = -1;
    asm_changed(false);
    g_asm_fitted = false; asm_request_meshes();
    status("example assembly: plate (fixed), shaft inserted in its bore, arm on the shaft, four M8 bolts in the holes. Play the motion study (Space)");
}
static void asm_insert_gear(int kind) {  // 0 spur, 1 helical, 2 bevel, 3 rack, 4 worm, 5 worm wheel
    static const char* TITLE[6] = {"Spur gear", "Helical gear", "Bevel gear", "Rack", "Worm", "Worm wheel"};
    static const char* DEF[6] = {"m2 z20 b8 d10", "m2 z20 b10 d10 h20", "m2 z20 b6 d10 c45", "m2 n12 b10 h6", "m4 z1 q10 b50 d20 h1", "m4 z30 w1 q10 b24 d20 h1"};
    static const char* HELP[6] = {"Module (mm), teeth, face width (mm), bore (mm); r1 recesses the faces round a hub, s6 makes a 6-spline bore. E.g. m2 z20 b8 d16 r1 s6:",
                                  "Normal module (mm), teeth, face width, bore and helix angle (deg, negative for left hand), e.g. m2 z20 b10 d10 h20:",
                                  "Module at the large end (mm), teeth, face width along the axis, bore and pitch cone angle (deg; 45 for a 1:1 pair at 90 deg), e.g. m2 z20 b6 d10 c45:",
                                  "Module (mm), number of teeth, face width (mm) and bar depth under the roots (mm), e.g. m2 n12 b10 h6:",
                                  "Axial module (mm), starts z (1 to 4), diameter factor q (pitch diameter = q x module), length b, bore d (mm), hand h (1 right, -1 left), e.g. m4 z1 q10 b50 d20 h1:",
                                  "Module (mm), teeth z, the worm's starts w and diameter factor q, face width b, bore d (mm), hand h (as the worm's), e.g. m4 z30 w1 q10 b24 d20 h1:"};
    std::string spec = DEF[kind];
    if (!g_selftest && !ask_text(g_main, TITLE[kind], HELP[kind], spec)) return;
    double mod = 2, b = kind == 2 ? 6 : kind == 0 ? 8 : 10, d = 10, h = kind == 1 ? 20 : 6, c = 45, q = 10, wst = 1; int z = kind == 3 ? 12 : 20, spl = 0; bool hub = false;
    std::string t;
    for (size_t i = 0; i <= spec.size(); ++i) {
        char ch = i < spec.size() ? (char)tolower((unsigned char)spec[i]) : ' ';
        if (ch == ' ' || ch == ',') {
            if (t.size() > 1) { double v = atof(t.c_str() + 1); if (t[0] == 'm') mod = v; else if (t[0] == 'z' || t[0] == 'n') z = (int)v; else if (t[0] == 'b') b = v; else if (t[0] == 'd') d = v; else if (t[0] == 'h') h = v; else if (t[0] == 'c') c = v; else if (t[0] == 'r') hub = v > 0; else if (t[0] == 's') spl = (int)v; else if (t[0] == 'q') q = v; else if (t[0] == 'w') wst = v; }
            t.clear();
        } else t += ch;
    }
    if (kind >= 4) {
        int hand = h < 0 ? -1 : 1;
        if (mod <= 0 || b <= 0 || q < 5 || q > 25 || (kind == 4 && (z < 1 || z > 4)) || (kind == 5 && (z < 20 || z > 120 || wst < 1 || wst > 4))) {
            MessageBoxA(g_main, "Check the values: a module and width above 0; a diameter factor q of 5 to 25; a worm of 1 to 4 starts; a wheel of 20 to 120 teeth.", TITLE[kind], MB_ICONWARNING); return;
        }
        double rf = 0.5 * q * mod - 1.25 * mod;
        if (kind == 4) { if (d >= 2 * rf - 2) { MessageBoxA(g_main, ("The bore must be under " + fmtd(std::round((2 * rf - 2) * 10) / 10) + " mm for this worm.").c_str(), TITLE[kind], MB_ICONWARNING); return; }
                         asm_add_component(worm_part(mod / 1000, z, q, b / 1000, std::max(0.0, d) / 1000, hand), "worm", "", nullptr); return; }
        asm_add_component(worm_wheel_part(mod / 1000, z, (int)wst, q, b / 1000, std::max(0.0, d) / 1000, hand), "wheel", "", nullptr); return;
    }
    if (mod <= 0 || b <= 0 || (kind != 3 && (z < 8 || z > 300)) || (kind == 3 && (z < 1 || z > 500)) || (kind == 1 && std::fabs(h) > 45) || (kind == 2 && (c <= 0 || c >= 85))) {
        MessageBoxA(g_main, "Check the values: a module and width above 0; 8 to 300 teeth (a rack 1 to 500); a helix up to 45 deg; a cone between 0 and 85 deg.", TITLE[kind], MB_ICONWARNING); return;
    }
    if (kind == 3) { asm_add_component(rack_part(mod / 1000, z, b / 1000, std::max(mod, h) / 1000), "rack", "", nullptr); return; }
    js::Value g = gear_part(mod / 1000, z, b / 1000, std::max(0.0, d) / 1000, kind == 1 ? h : 0, kind == 2 ? c : 0);
    if (kind != 2 && (hub || spl >= 4)) gear_hub(g, std::max(0.0, d) / 1000, spl);
    asm_add_component(g, "gear", "", nullptr);
}
// Cams > Disc cam / Follower: sizes as words and numbers, e.g. "cycloidal R25 h15 rise120 dwell60 return120 roller8 e0 b14 d20"
static void asm_insert_cam(bool follower) {
    std::string spec = follower ? "roller8 b14 L70 s10" : "cycloidal R25 h15 rise120 dwell60 return120 roller8 e0 b14 d20";
    const char* help = follower ? "Roller follower: roller radius, cam thickness b, stem length L and diameter s (mm), e.g. roller8 b14 L70 s10; or a flat face: flat w20 b14 L70 s10 (w the face's half-width):"
                                : "Law (cycloidal, harmonic, 345 or uniform), base circle radius R, lift h, rise, dwell and return angles (deg), roller radius (0 for a flat-faced follower), offset e, thickness b, bore d (mm):";
    if (!g_selftest && !ask_text(g_main, follower ? "Follower" : "Disc cam", help, spec)) return;
    int law = 0; bool flat = false;
    double R = 25, h = 15, rise = 120, dwell = 60, ret = 120, rr = 8, e = 0, b = 14, d = 20, L = 70, s = 10, w = 20;
    std::string t;
    for (size_t i = 0; i <= spec.size(); ++i) {
        char ch = i < spec.size() ? spec[i] : ' ';
        if (ch == ' ' || ch == ',') {
            size_t k = 0; while (k < t.size() && isalpha((unsigned char)t[k])) ++k;
            std::string key = t.substr(0, k); for (char& q : key) q = (char)tolower((unsigned char)q);
            double v = k < t.size() ? atof(t.c_str() + k) : 0;
            if (t == "345") law = 2;
            else if (key == "cycloidal") law = 0; else if (key == "harmonic") law = 1; else if (key == "uniform") law = 3; else if (key == "flat") flat = true;
            else if (k < t.size()) {
                if (key == "r" && t[0] == 'R') R = v; else if (key == "h") h = v; else if (key == "rise") rise = v; else if (key == "dwell") dwell = v; else if (key == "return") ret = v;
                else if (key == "roller") rr = v; else if (key == "e") e = v; else if (key == "b") b = v; else if (key == "d") d = v; else if (key == "l") L = v; else if (key == "s") s = v; else if (key == "w") w = v;
            }
            t.clear();
        } else t += ch;
    }
    const double u = 0.001;
    if (follower) {
        if (b <= 0 || L <= 0 || s <= 0 || (!flat && rr <= 0) || (flat && w <= 0)) { MessageBoxA(g_main, "Sizes must be above 0.", "Follower", MB_ICONWARNING); return; }
        asm_add_component(follower_part(flat ? 0 : rr * u, w * u, b * u, L * u, s * u), "follower", "", nullptr); return;
    }
    if (R <= 0 || h <= 0 || b <= 0 || rise <= 0 || ret <= 0 || rise + dwell + ret > 360 || dwell < 0 || rr < 0 || std::fabs(e) >= R + rr || d >= 2 * R - 4) {
        MessageBoxA(g_main, "Check the values: rise, dwell and return within 360 deg; positive sizes; the offset under the base circle plus roller; the bore under the base circle.", "Disc cam", MB_ICONWARNING); return;
    }
    js::Value cam = cam_part(law, R * u, h * u, rise, dwell, ret, rr * u, e * u, b * u, d * u);
    std::string rep; { const js::Value* cm = cam.find("cam"); CamDef cd = cam_def(*cm); CamCheck k; cam_outline(cd, 360, k); rep = cam_report(cd, k); }
    asm_add_component(cam, "cam", "", nullptr);
    status("disc cam: " + rep);
}
// a worked example: a 20-tooth pinion driving a 40-tooth gear on pins in a plate
static void asm_demo_gears() {
    asm_new();
    if (!g_asm_on) return;
    const double mod = 0.002, w = 0.008;
    const int z1 = 20, z2 = 40;
    double a = 0.5 * mod * (z1 + z2), x1 = -0.04, x2 = x1 + a;
    js::Value plate = js::Value::object(), pf = js::Value::array();
    plate["name"] = "gear plate"; plate["units"] = "m";
    { js::Value f = js::Value::object(); f["id"] = "plate"; f["type"] = "box"; f["op"] = "add"; f["size"] = v3json(V3(0.2, 0.12, 0.01)); f["round"] = 0.0; f["translate"] = v3json(V3(0, 0, -0.005)); pf.push(f); }
    double xs[2] = {x1, x2};
    for (int k = 0; k < 2; ++k) { js::Value f = prim("cylinder", "cut", {{"r", 0.0050}, {"length", 0.05}}, V3(xs[k], 0, -0.005)); f["axis"] = "z"; f["id"] = k ? "bore2" : "bore1"; pf.push(f); }
    plate["features"] = pf;
    asm_add_component(plate, "plate", "", nullptr);
    Pose p1; p1.t = V3(x1, -0.12, 0.05); asm_add_component(gear_part(mod, z1, w, 0.0102), "pinion", "", &p1);
    Pose p2; p2.t = V3(x2, 0.15, 0.05); asm_add_component(gear_part(mod, z2, w, 0.0102), "gear", "", &p2);
    for (int k = 0; k < 2; ++k) { Pose pp; pp.t = V3(-0.1 + 0.05 * k, -0.2, 0.06); asm_add_component(fastener_part("pin", THREADS[5], 30), "pin", "", &pp); }
    auto axis = [&](int c, V3 pl, double r, double s0, double s1, bool hole) { ARef q; q.comp = c; q.kind = 1; q.p = pl; q.d = V3(0, 0, 1); q.r = r; q.value_ends[0] = s0; q.value_ends[1] = s1; q.hole = hole; return q; };
    for (int k = 0; k < 2; ++k) {  // pins into the plate, gears on the pins, 3 mm above the plate
        g_pick_ref[0] = axis(0, V3(xs[k], 0, -0.005), 0.005, -0.01, 0.0, true);
        g_pick_ref[1] = axis(3 + k, V3(0, 0, -0.01), 0.005, -0.018, 0.012, false);
        asm_make_mate(MT_INSERT);
        g_pick_ref[0] = axis(3 + k, V3(0, 0, -0.01), 0.005, -0.018, 0.012, false);
        g_pick_ref[1] = axis(1 + k, V3(), 0.0051, -0.004, 0.004, true);
        asm_make_mate(MT_CONCENTRIC);
        AMate m; m.type = MT_DISTANCE; m.id = "distance" + tostr(k + 1); m.value = 0.003 + 0.5 * w;
        m.a.comp = 0; m.a.kind = 0; m.a.p = V3(0, 0, 0); m.a.d = V3(0, 0, 1);
        m.b.comp = 1 + k; m.b.kind = 0; m.b.p = V3(0, 0, 0); m.b.d = V3(0, 0, -1);
        g_am.push_back(m);
    }
    asm_solve();
    g_pick_ref[0].comp = 1; g_pick_ref[0].kind = 1; g_pick_ref[0].p = V3(); g_pick_ref[0].d = V3(0, 0, 1);
    g_pick_ref[1] = g_pick_ref[0]; g_pick_ref[1].comp = 2;
    asm_make_mate(MT_GEAR);
    AMotor mo; mo.comp = 1; mo.p = V3(); mo.d = V3(0, 0, 1); mo.speed = 90; g_amot.push_back(mo);
    ATrace tr; tr.comp = 2; tr.p = V3(0.5 * mod * z2 + mod, 0, 0.5 * w); g_atr.push_back(tr);
    g_ms_T = 8;
    g_asm_sel = -1; g_asm_mate_sel = -1;
    asm_changed(false);
    g_asm_fitted = false; asm_request_meshes();
    status("gear example: a 20-tooth pinion (motor, 90 deg/s) drives a 40-tooth gear at half speed the other way. Play (Space)");
}
// a worked example of the other gears: a helical pair, a bevel pair at 90 deg and a rack and pinion, each with a motor
static void asm_demo_gears2() {
    asm_new();
    if (!g_asm_on) return;
    const double m = 0.002, w = 0.01, hb = 0.006;
    const double b20 = 20 * geo::PI / 180, mt = m / std::cos(b20), rh1 = 0.5 * mt * 20, rh2 = 0.5 * mt * 30;
    V3 P1(-0.12, 0.06, 0), P2(-0.12 + rh1 + rh2, 0.06, 0), PA(0.03, -0.06, 0), PR(0.07, 0.07, 0);
    // the base: a plate with bores for the pins, and an upright carrying the second bevel gear's axis
    js::Value base = js::Value::object(), bf = js::Value::array();
    base["name"] = "gear base"; base["units"] = "m";
    auto box = [](const char* id, V3 size, V3 at) { js::Value f = js::Value::object(); f["id"] = id; f["type"] = "box"; f["op"] = "add"; f["size"] = v3json(size); f["round"] = 0.0; f["translate"] = v3json(at); return f; };
    bf.push(box("plate", V3(0.34, 0.24, 0.01), V3(0, 0, -0.005)));
    bf.push(box("upright", V3(0.01, 0.05, 0.04), V3(0.058, -0.06, 0.02)));
    V3 bores[4] = {P1, P2, PA, PR};
    for (int k = 0; k < 4; ++k) { js::Value f = prim("cylinder", "cut", {{"r", 0.005}, {"length", 0.05}}, V3(bores[k].x, bores[k].y, -0.005)); f["axis"] = "z"; f["id"] = "bore" + tostr(k + 1); bf.push(f); }
    { js::Value f = prim("cylinder", "cut", {{"r", 0.0051}, {"length", 0.03}}, V3(0.058, -0.06, 0.023)); f["axis"] = "x"; f["id"] = "cross_bore"; bf.push(f); }
    base["features"] = bf;
    asm_add_component(base, "base", "", nullptr);                                                     // 0
    Pose p; p.t = V3(-0.12, -0.2, 0.05); asm_add_component(gear_part(m, 20, w, 0.0102, 20), "helical", "", &p);   // 1
    p.t = V3(-0.05, -0.2, 0.05); asm_add_component(gear_part(m, 30, w, 0.0102, -20), "helical", "", &p);         // 2
    p.t = V3(0.0, -0.25, 0.05); asm_add_component(gear_part(m, 20, hb, 0.0102, 0, 45), "bevel", "", &p);         // 3
    Pose pb; pb.R[0] = 0; pb.R[2] = -1; pb.R[6] = 1; pb.R[8] = 0; pb.t = V3(0.047, -0.06, 0.023);                // local z -> world -x
    asm_add_component(gear_part(m, 20, hb, 0.0102, 0, 45), "bevel", "", &pb);                                     // 4
    p.t = V3(0.12, 0.17, 0.05); asm_add_component(gear_part(m, 16, w, 0.0102), "pinion", "", &p);                // 5
    Pose pr; pr.t = V3(PR.x, PR.y - 0.016, 0.008); asm_add_component(rack_part(m, 20, w, 0.006), "rack", "", &pr); // 6
    for (int k = 0; k < 4; ++k) { Pose pp; pp.t = V3(-0.15 + 0.05 * k, -0.22, 0.07); asm_add_component(fastener_part("pin", THREADS[5], 30), "pin", "", &pp); }  // 7 to 10
    auto axis = [&](int c, V3 pl, V3 d, double r, double s0, double s1, bool hole) { ARef q; q.comp = c; q.kind = 1; q.p = pl; q.d = d; q.r = r; q.value_ends[0] = s0; q.value_ends[1] = s1; q.hole = hole; return q; };
    auto plane = [&](int c, V3 pl, V3 d) { ARef q; q.comp = c; q.kind = 0; q.p = pl; q.d = d; return q; };
    auto add = [&](int type, ARef a, ARef b, double v) { AMate mt2; mt2.type = type; mt2.a = a; mt2.b = b; mt2.value = v; mt2.id = std::string(MATE_NAME[type]) + tostr((int)g_am.size() + 1); g_am.push_back(mt2); };
    int gears[4] = {1, 2, 3, 5};
    double zc[4] = {0.003 + 0.5 * w, 0.003 + 0.5 * w, 0.003 + 0.5 * hb, 0.003 + 0.5 * w};
    for (int k = 0; k < 4; ++k) {  // a pin in each bore, its gear on the pin at its height
        g_pick_ref[0] = axis(0, V3(bores[k].x, bores[k].y, -0.005), V3(0, 0, 1), 0.005, -0.01, 0.0, true);
        g_pick_ref[1] = axis(7 + k, V3(0, 0, -0.01), V3(0, 0, 1), 0.005, -0.018, 0.012, false);
        asm_make_mate(MT_INSERT);
        add(MT_CONCENTRIC, axis(7 + k, V3(0, 0, -0.01), V3(0, 0, 1), 0.005, -0.018, 0.012, false), axis(gears[k], V3(), V3(0, 0, 1), 0.0051, -0.005, 0.005, true), 0);
        add(MT_DISTANCE, plane(0, V3(), V3(0, 0, 1)), plane(gears[k], V3(), V3(0, 0, -1)), zc[k]);
    }
    // the second bevel gear: in the upright's cross bore, its large end 3 mm from the upright
    add(MT_CONCENTRIC, axis(0, V3(0.058, -0.06, 0.023), V3(1, 0, 0), 0.0051, -0.015, 0.015, true), axis(4, V3(), V3(0, 0, 1), 0.0051, -0.003, 0.003, true), 0);
    add(MT_DISTANCE, plane(0, V3(0.053, -0.06, 0.023), V3(-1, 0, 0)), plane(4, V3(0, 0, -0.5 * hb), V3(0, 0, -1)), 0.003);
    // the rack: flat on two guide planes (3 mm over the plate, its back on y = pitch line - 8.5 mm): it can only slide along x
    add(MT_COINCIDENT, plane(0, V3(0, 0, 0.003), V3(0, 0, 1)), plane(6, V3(0, 0, -0.5 * w), V3(0, 0, -1)), 0);
    add(MT_COINCIDENT, plane(0, V3(0, PR.y - 0.016 - 0.0085, 0), V3(0, 1, 0)), plane(6, V3(0, -0.0085, 0), V3(0, -1, 0)), 0);
    asm_solve();
    int pairs[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    for (auto& pq : pairs) {
        g_pick_ref[0] = axis(pq[0], V3(), V3(0, 0, 1), 0, 0, 0, false);
        g_pick_ref[1] = axis(pq[1], V3(), V3(0, 0, 1), 0, 0, 0, false);
        asm_make_mate(MT_GEAR);
    }
    AMotor mo; mo.p = V3(); mo.d = V3(0, 0, 1);
    mo.comp = 1; mo.speed = 90; g_amot.push_back(mo);
    mo.comp = 3; mo.speed = 90; g_amot.push_back(mo);
    mo.comp = 5; mo.speed = 45; g_amot.push_back(mo);
    ATrace tr;
    tr.comp = 2; tr.p = V3(rh2 + m, 0, 0.5 * w); g_atr.push_back(tr);
    tr.comp = 4; tr.p = V3(0.5 * m * 20 + m, 0, -0.5 * hb); g_atr.push_back(tr);
    tr.comp = 6; tr.p = V3(0.5 * 20 * geo::PI * m, 0, 0.5 * w); g_atr.push_back(tr);
    g_ms_T = 4;
    g_asm_sel = -1; g_asm_mate_sel = -1;
    asm_changed(false);
    g_asm_fitted = false; asm_request_meshes();
    status("gear examples: a helical pair (20:30, 20 deg, opposite hands), a bevel pair at 90 deg and a rack and pinion, each driven by a motor. Play (Space)");
}
static void asm_insert_drive(int kind) {  // 0 bearing, 1 bushing, 2 keyed shaft, 3 key, 4 coupling, 5 roller, 6 pulley, 7 sprocket
    static const char* TITLE[8] = {"Ball bearing", "Flanged bushing", "Keyed shaft", "Parallel key", "Sleeve coupling", "Roller", "V-belt pulley", "Sprocket"};
    static const char* DEF[8] = {"6004", "d20 L20", "d20 L160", "d20 L32", "d20", "D60 L30 d20", "D100 d20 g1", "08B z18 d20"};
    static const char* HELP[8] = {"Deep groove ball bearing, ISO designation 6000 to 6010 or 6200 to 6206 (e.g. 6004: 20 x 42 x 12 mm):",
                                  "Bore d and length L (mm), e.g. d20 L20 (bronze):", "Diameter d and length L (mm), e.g. d20 L160 (keyway to DIN 6885):",
                                  "For a shaft of diameter d, length L (mm), e.g. d20 L32 (DIN 6885):", "Shaft diameter d (mm), e.g. d20:",
                                  "Outside diameter D, length L and bore d (mm), e.g. D60 L30 d20:",
                                  "SPZ V-belt pulley: pitch diameter D, bore d (mm) and number of grooves g, e.g. D100 d20 g1 (63 mm or more):",
                                  "Roller-chain sprocket: chain 06B, 08B, 10B, 12B or 16B (ISO 606), teeth z and bore d (mm), e.g. 08B z18 d20:"};
    std::string spec = DEF[kind];
    if (!g_selftest && !ask_text(g_main, TITLE[kind], HELP[kind], spec)) return;
    if (kind == 0) {
        std::string k = spec; while (!k.empty() && (k.back() == ' ')) k.pop_back(); while (!k.empty() && k[0] == ' ') k.erase(0, 1);
        for (int i = 0; i < NBEARING; ++i) if (k == BEARINGS[i].name) { asm_add_component(bearing_part(BEARINGS[i]), "bearing", "", nullptr); return; }
        MessageBoxA(g_main, ("No bearing " + spec + " in the table: use 6000 to 6010 or 6200 to 6206.").c_str(), "Ball bearing", MB_ICONWARNING); return;
    }
    double d = 20, L = kind == 1 ? 20 : kind == 3 ? 32 : 160, D = kind == 6 ? 100 : 60, zt = 18, ng = 1;
    std::string t, chain = "08B";
    for (size_t i = 0; i <= spec.size(); ++i) {
        char ch = i < spec.size() ? spec[i] : ' ';
        if (ch == ' ' || ch == ',') {
            if (chain_of(t)) chain = t;
            else if (t.size() > 1) { double v = atof(t.c_str() + 1); if (t[0] == 'd') d = v; else if (t[0] == 'L' || t[0] == 'l') L = v; else if (t[0] == 'D') D = v; else if (t[0] == 'z' || t[0] == 'n') zt = v; else if (t[0] == 'g') ng = v; }
            t.clear();
        } else t += (char)(kind == 7 && (ch == 'b') ? 'B' : ch);
    }
    if (kind == 6) { if (D < 30 || D > 2000 || d <= 0 || d > 0.8 * D - 20) { MessageBoxA(g_main, "Use a pitch diameter of 30 to 2000 mm and a bore that leaves a hub (under 0.8 D - 20 mm).", TITLE[kind], MB_ICONWARNING); return; }
                     asm_add_component(pulley_part(D, d, (int)ng), "pulley", "", nullptr); return; }
    if (kind == 7) { const ChainDef* C = chain_of(chain); int z = (int)std::round(zt);
                     if (!C || z < 8 || z > 120) { MessageBoxA(g_main, "Use chain 06B, 08B, 10B, 12B or 16B and 8 to 120 teeth.", TITLE[kind], MB_ICONWARNING); return; }
                     double Dp = C->p / std::sin(geo::PI / z), df = Dp - C->d1;
                     if (d <= 0 || d > df - 1.6 * C->d1) { MessageBoxA(g_main, ("The bore must be under " + fmtd(std::round((df - 1.6 * C->d1) * 10) / 10) + " mm for this sprocket.").c_str(), TITLE[kind], MB_ICONWARNING); return; }
                     asm_add_component(sprocket_part(*C, z, d), "sprocket", "", nullptr); return; }
    if (d <= 0 || L <= 0 || D <= 0) { MessageBoxA(g_main, "Sizes must be above 0.", TITLE[kind], MB_ICONWARNING); return; }
    if (kind == 1) asm_add_component(bushing_part(d, L), "bushing", "", nullptr);
    else if (kind == 2) asm_add_component(keyshaft_part(d, L), "shaft", "", nullptr);
    else if (kind == 3) asm_add_component(key_part(d, L), "key", "", nullptr);
    else if (kind == 4) asm_add_component(coupling_part(d), "coupling", "", nullptr);
    else asm_add_component(roller_part(D, kind == 5 && spec.find('L') == std::string::npos ? 30 : L, d), "roller", "", nullptr);
}
// a worked example: a gearbox and a roller drive on three shafts in six ball bearings, in an oil bath
static void asm_demo_drive() {
    asm_new();
    if (!g_asm_on) return;
    const double H = 0.075, xs[3] = {-0.03, 0.03, 0.09}, yw = 0.055;  // the output shaft clear of the big gear's teeth
    js::Value base = js::Value::object(), bf = js::Value::array();
    base["name"] = "gearbox frame"; base["units"] = "m"; base["material"] = "cast_iron";
    bf.push(box_f("plate", "add", V3(0.24, 0.17, 0.012), V3(0.025, 0, -0.006)));
    for (int s = -1; s <= 1; s += 2) bf.push(box_f(s < 0 ? "wall_front" : "wall_back", "add", V3(0.2, 0.016, 0.125), V3(0.025, s * yw, 0.0625)));
    for (int k = 0; k < 3; ++k) bf.push(cyl_f(("housing" + tostr(k + 1)).c_str(), "cut", 0.021, 0.16, V3(xs[k], 0, H), "y"));
    base["features"] = bf;
    asm_add_component(base, "frame", "", nullptr);                                                       // 0
    Pose Ry; Ry.R[0] = 1; Ry.R[1] = 0; Ry.R[2] = 0; Ry.R[3] = 0; Ry.R[4] = 0; Ry.R[5] = 1; Ry.R[6] = 0; Ry.R[7] = -1; Ry.R[8] = 0;  // local z -> world y
    const BearingDef* b6004 = nullptr; for (int i = 0; i < NBEARING; ++i) if (std::string(BEARINGS[i].name) == "6004") b6004 = &BEARINGS[i];
    for (int k = 0; k < 3; ++k) for (int s = -1; s <= 1; s += 2) { Pose p = Ry; p.t = V3(xs[k], s * yw, H); asm_add_component(bearing_part(*b6004), "bearing", "", &p); }  // 1 to 6
    for (int k = 0; k < 3; ++k) { Pose p = Ry; p.t = V3(xs[k], 0, H); asm_add_component(keyshaft_part(20, 160), "shaft", "", &p); }  // 7 to 9
    js::Value pin = gear_part(0.002, 20, 0.014, 0.0202); gear_hub(pin, 0.0202, 0);
    js::Value gr = gear_part(0.002, 40, 0.014, 0.0202); gear_hub(gr, 0.0202, 0);
    { Pose p = Ry; p.t = V3(xs[0], 0, H); asm_add_component(pin, "pinion", "", &p); }                    // 10
    { Pose p = Ry; p.t = V3(xs[1], 0, H); asm_add_component(gr, "gear", "", &p); }                       // 11
    { Pose p = Ry; p.t = V3(xs[1], -0.026, H); asm_add_component(roller_part(70, 24, 20.2), "roller", "", &p); }   // 12
    { Pose p = Ry; p.t = V3(xs[2], -0.026, H); asm_add_component(roller_part(50, 24, 20.2), "roller", "", &p); }   // 13
    g_ac[13].material = "rubber";  // a rubber-tyred follower: it grips even in oil
    auto lock = [&](int a, int b) {
        AMate m; m.type = MT_LOCK; m.a.comp = a; m.b.comp = b; m.id = "lock" + tostr((int)g_am.size() + 1);
        const Pose &Pa = g_ac[(size_t)a].pose, &Pb = g_ac[(size_t)b].pose;
        double Rt[9] = {Pa.R[0], Pa.R[3], Pa.R[6], Pa.R[1], Pa.R[4], Pa.R[7], Pa.R[2], Pa.R[5], Pa.R[8]};
        mat_mul(Rt, Pb.R, m.rel.R); m.rel.t = Pa.inv(Pb.t);
        g_am.push_back(m);
    };
    auto axis = [&](int c, V3 pl, V3 d, double r, bool hole) { ARef q; q.comp = c; q.kind = 1; q.p = pl; q.d = d; q.r = r; q.hole = hole; return q; };
    for (int b = 1; b <= 6; ++b) lock(0, b);  // the bearings sit in the frame's housings
    for (int k = 0; k < 3; ++k) {  // each shaft turns in its two bearings
        AMate m; m.type = MT_CONCENTRIC; m.id = "bearing" + tostr(2 * k + 1);
        m.a = axis(1 + 2 * k, V3(), V3(0, 0, 1), 0.010, true); m.b = axis(7 + k, V3(), V3(0, 0, 1), 0.010, false); g_am.push_back(m);
        m.id = "bearing" + tostr(2 * k + 2); m.a = axis(2 + 2 * k, V3(), V3(0, 0, 1), 0.010, true); g_am.push_back(m);
    }
    asm_solve();
    lock(7, 10);  // the pinion on the input shaft
    g_pick_ref[0] = axis(10, V3(), V3(0, 0, 1), 0, false); g_pick_ref[1] = axis(11, V3(), V3(0, 0, 1), 0, false);
    asm_make_mate(MT_GEAR);  // 20:40, the teeth phased
    lock(11, 8);  // the gear turns the middle shaft
    lock(8, 12);  // which carries the 60 mm roller
    g_pick_ref[0] = axis(12, V3(), V3(0, 0, 1), 0.035, false); g_pick_ref[1] = axis(13, V3(), V3(0, 0, 1), 0.025, false);
    asm_make_mate(MT_ROLL);  // it drives the 40 mm roller by friction
    lock(13, 9);  // on the output shaft
    AMotor mo; mo.comp = 7; mo.p = V3(); mo.d = V3(0, 0, 1); mo.speed = 360; mo.power = 20; g_amot.push_back(mo);
    g_pf_preload = 800;  // the rollers pressed together
    ATrace tr; tr.comp = 11; tr.p = V3(0.042, 0, 0.007); g_atr.push_back(tr);
    tr.comp = 13; tr.p = V3(0.025, 0, 0.012); g_atr.push_back(tr);
    g_ms_T = 4;
    js::Value env = js::Value::object();  // an oil bath up to the shafts
    env["medium"] = "Gear oil ISO VG 220"; env["temperature_C"] = 40.0; env["flow_m_s"] = 0.0; env["flow_dir"] = "+x"; env["altitude_m"] = 0.0;
    env["surface_z_m"] = H; env["find_float"] = false; env["part_temperature_C"] = 40.0; env["heat_W"] = 0.0; env["lubrication"] = "oil-lubricated"; env["gravity_m_s2"] = 9.81;
    g_doc["environment"] = env;
    g_asm_sel = -1; g_asm_mate_sel = -1;
    asm_changed(false);
    g_asm_fitted = false; asm_request_meshes();
    status("drive train: 20 W at 60 rpm into the input shaft; 20:40 gears, a 70/50 mm roller drive (rubber follower), six 6004 bearings, an oil bath. Play, Power flow, Interference");
}
// a worked example: a V-belt drive (D63 to D125) and a roller-chain drive (08B, 12 to 24 teeth) side by side on a
// cast-iron plate, each pair on stainless shafts in 6004 bearings, each driven by a motor
static void asm_demo_belts() {
    asm_new();
    if (!g_asm_on) return;
    const ChainDef* C08 = chain_of("08B");
    const double zp = 0.034, yb = 0.075, yc = -0.075, xa = -0.12;
    // the chain's centre distance for exactly 44 links on the sprockets' mean pitch circles
    double r1 = 12 * C08->p * 0.001 / (2 * geo::PI), r2 = 24 * C08->p * 0.001 / (2 * geo::PI), lo = r2 - r1 + 1e-6, hi = 1.0, want = 44 * C08->p * 0.001;
    for (int it = 0; it < 80; ++it) { double c = 0.5 * (lo + hi), b = std::asin((r2 - r1) / c), L = 2 * c * std::cos(b) + r1 * (geo::PI - 2 * b) + r2 * (geo::PI + 2 * b); (L < want ? lo : hi) = c; }
    const double Cc = lo, Cb = 0.2;
    const V3 at[4] = {V3(xa, yb, 0), V3(xa + Cb, yb, 0), V3(xa, yc, 0), V3(xa + Cc, yc, 0)};
    js::Value base = js::Value::object(), bf = js::Value::array();
    base["name"] = "drive plate"; base["units"] = "m"; base["material"] = "cast_iron";
    bf.push(box_f("plate", "add", V3(0.40, 0.30, 0.012), V3(xa + 0.5 * Cb, 0, -0.006)));
    for (int k = 0; k < 4; ++k) bf.push(cyl_f(("housing" + tostr(k + 1)).c_str(), "cut", 0.021, 0.04, V3(at[k].x, at[k].y, -0.006)));
    base["features"] = bf;
    asm_add_component(base, "plate", "", nullptr);                                                                      // 0
    g_ac[0].fixed = true;
    const BearingDef* b6004 = nullptr; for (int i = 0; i < NBEARING; ++i) if (std::string(BEARINGS[i].name) == "6004") b6004 = &BEARINGS[i];
    for (int k = 0; k < 4; ++k) { Pose p; p.t = V3(at[k].x, at[k].y, -0.006); asm_add_component(bearing_part(*b6004), "bearing", "", &p); }   // 1 to 4
    for (int k = 0; k < 4; ++k) { Pose p; p.t = V3(at[k].x, at[k].y, 0.02); asm_add_component(keyshaft_part(20, 80), "shaft", "", &p); g_ac.back().material = "stainless_316"; }  // 5 to 8
    { Pose p; p.t = V3(at[0].x, at[0].y, zp); asm_add_component(pulley_part(63, 20.2, 1), "pulley", "", &p); }          // 9
    { Pose p; p.t = V3(at[1].x, at[1].y, zp); asm_add_component(pulley_part(125, 20.2, 1), "pulley", "", &p); g_ac.back().material = "cast_iron"; }  // 10
    { Pose p; p.t = V3(at[2].x, at[2].y, zp); asm_add_component(sprocket_part(*C08, 12, 20.2), "sprocket", "", &p); g_ac.back().material = "steel_300m"; }  // 11
    { Pose p; p.t = V3(at[3].x, at[3].y, zp); asm_add_component(sprocket_part(*C08, 24, 20.2), "sprocket", "", &p); }  // 12
    auto lock = [&](int a, int b) {
        AMate m; m.type = MT_LOCK; m.a.comp = a; m.b.comp = b; m.id = "lock" + tostr((int)g_am.size() + 1);
        const Pose &Pa = g_ac[(size_t)a].pose, &Pb = g_ac[(size_t)b].pose;
        double Rt[9] = {Pa.R[0], Pa.R[3], Pa.R[6], Pa.R[1], Pa.R[4], Pa.R[7], Pa.R[2], Pa.R[5], Pa.R[8]};
        mat_mul(Rt, Pb.R, m.rel.R); m.rel.t = Pa.inv(Pb.t);
        g_am.push_back(m);
    };
    auto axis = [&](int c, double r, bool hole) { ARef q; q.comp = c; q.kind = 1; q.p = V3(); q.d = V3(0, 0, 1); q.r = r; q.hole = hole; return q; };
    for (int k = 0; k < 4; ++k) lock(0, 1 + k);  // the bearings in the plate
    for (int k = 0; k < 4; ++k) { AMate m; m.type = MT_CONCENTRIC; m.id = "bearing" + tostr(k + 1); m.a = axis(1 + k, 0.010, true); m.b = axis(5 + k, 0.010, false); g_am.push_back(m); }
    asm_solve();
    for (int k = 0; k < 4; ++k) { if (k % 2 == 0) lock(5 + k, 9 + k); else lock(9 + k, 5 + k); }  // each wheel keyed on its shaft (a driven wheel carries its shaft round)
    g_pick_ref[0] = axis(9, 0, false); g_pick_ref[1] = axis(10, 0, false); asm_make_mate(MT_BELT);
    std::string nb = g_st_last_note;
    g_pick_ref[0] = axis(11, 0, false); g_pick_ref[1] = axis(12, 0, false); asm_make_mate(MT_BELT);
    AMotor mo; mo.p = V3(); mo.d = V3(0, 0, 1);
    mo.comp = 5; mo.speed = 360; mo.power = 20; g_amot.push_back(mo);
    mo.comp = 7; mo.speed = 360; mo.power = 20; g_amot.push_back(mo);
    g_pf_tension = 250;
    ATrace tr; tr.comp = 10; tr.p = V3(0.055, 0, 0.0085); g_atr.push_back(tr);
    tr.comp = 12; tr.p = V3(0.040, 0, 0.004); g_atr.push_back(tr);
    g_ms_T = 4;
    g_asm_sel = -1; g_asm_mate_sel = -1;
    asm_changed(false);
    g_asm_fitted = false; asm_request_meshes();
    status("belt & chain: a V-belt (SPZ, D63 to D125) and a roller chain (08B, 12 to 24 teeth, 44 links), each driven at 60 rpm with 20 W. Play, Power flow");
}
// a worked example: a worm (m4, one start, q10, right hand, hardened steel) driving a 30-tooth bronze worm wheel at
// 30:1, the shafts in 6004 bearings in a cast-iron base; the motor on the worm shaft
static void asm_demo_worm() {
    asm_new();
    if (!g_asm_on) return;
    const double mod = 0.004, q = 10, H = 0.07, r1 = 0.5 * q * mod, r2 = 0.5 * 30 * mod, a = r1 + r2, xs = 0.06;
    js::Value base = js::Value::object(), bf = js::Value::array();
    base["name"] = "worm box base"; base["units"] = "m"; base["material"] = "cast_iron";
    bf.push(box_f("plate", "add", V3(0.24, 0.26, 0.012), V3(0, -0.03, -0.006)));
    for (int s = -1; s <= 1; s += 2) {
        bf.push(box_f(s < 0 ? "upright_left" : "upright_right", "add", V3(0.016, 0.06, H + 0.03), V3(s * xs, -a, 0.5 * (H + 0.03))));
        bf.push(cyl_f(s < 0 ? "housing_left" : "housing_right", "cut", 0.021, 0.03, V3(s * xs, -a, H), "x"));
    }
    bf.push(cyl_f("housing_wheel", "cut", 0.021, 0.04, V3(0, 0, -0.006)));
    base["features"] = bf;
    asm_add_component(base, "base", "", nullptr);                                                         // 0
    g_ac[0].fixed = true;
    Pose Rx; Rx.R[0] = 0; Rx.R[1] = 0; Rx.R[2] = 1; Rx.R[3] = 0; Rx.R[4] = 1; Rx.R[5] = 0; Rx.R[6] = -1; Rx.R[7] = 0; Rx.R[8] = 0;  // local z -> world x
    const BearingDef* b6004 = nullptr; for (int i = 0; i < NBEARING; ++i) if (std::string(BEARINGS[i].name) == "6004") b6004 = &BEARINGS[i];
    { Pose p; p.t = V3(0, 0, -0.006); asm_add_component(bearing_part(*b6004), "bearing", "", &p); }           // 1
    for (int s = -1; s <= 1; s += 2) { Pose p = Rx; p.t = V3(s * xs, -a, H); asm_add_component(bearing_part(*b6004), "bearing", "", &p); }  // 2, 3
    { Pose p; p.t = V3(0, 0, 0.045); asm_add_component(keyshaft_part(20, 110), "shaft", "", &p); }            // 4
    { Pose p = Rx; p.t = V3(0, -a, H); asm_add_component(keyshaft_part(20, 170), "shaft", "", &p); }          // 5
    { Pose p; p.t = V3(0, 0, H); asm_add_component(worm_wheel_part(mod, 30, 1, q, 0.024, 0.0202, 1), "wheel", "", &p); }  // 6
    { Pose p = Rx; p.t = V3(0, -a, H); asm_add_component(worm_part(mod, 1, q, 0.05, 0.0202, 1), "worm", "", &p); }      // 7
    auto lock = [&](int x, int y) {
        AMate m; m.type = MT_LOCK; m.a.comp = x; m.b.comp = y; m.id = "lock" + tostr((int)g_am.size() + 1);
        const Pose &Pa = g_ac[(size_t)x].pose, &Pb = g_ac[(size_t)y].pose;
        double Rt[9] = {Pa.R[0], Pa.R[3], Pa.R[6], Pa.R[1], Pa.R[4], Pa.R[7], Pa.R[2], Pa.R[5], Pa.R[8]};
        mat_mul(Rt, Pb.R, m.rel.R); m.rel.t = Pa.inv(Pb.t);
        g_am.push_back(m);
    };
    auto axis = [&](int c, double r, bool hole) { ARef qq; qq.comp = c; qq.kind = 1; qq.p = V3(); qq.d = V3(0, 0, 1); qq.r = r; qq.hole = hole; return qq; };
    for (int k = 1; k <= 3; ++k) lock(0, k);
    { AMate m; m.type = MT_CONCENTRIC; m.id = "bearing1"; m.a = axis(1, 0.010, true); m.b = axis(4, 0.010, false); g_am.push_back(m); }
    for (int k = 2; k <= 3; ++k) { AMate m; m.type = MT_CONCENTRIC; m.id = "bearing" + tostr(k); m.a = axis(k, 0.010, true); m.b = axis(5, 0.010, false); g_am.push_back(m); }
    asm_solve();
    lock(5, 7);
    g_pick_ref[0] = axis(7, 0, false); g_pick_ref[1] = axis(6, 0, false);
    asm_make_mate(MT_GEAR);
    lock(6, 4);  // the wheel carries its shaft round
    AMotor mo; mo.comp = 5; mo.p = V3(); mo.d = V3(0, 0, 1); mo.speed = 720; mo.power = 50; g_amot.push_back(mo);
    ATrace tr; tr.comp = 6; tr.p = V3(r2 + mod, 0, 0.012); g_atr.push_back(tr);
    g_ms_T = 6;
    js::Value env = js::Value::object();  // splash-lubricated with gear oil
    env["medium"] = "Air"; env["temperature_C"] = 40.0; env["flow_m_s"] = 0.0; env["flow_dir"] = "+x"; env["altitude_m"] = 0.0;
    env["surface_z_m"] = 0.0; env["find_float"] = false; env["part_temperature_C"] = 40.0; env["heat_W"] = 0.0; env["lubrication"] = "oil-lubricated"; env["gravity_m_s2"] = 9.81;
    g_doc["environment"] = env;
    g_asm_sel = -1; g_asm_mate_sel = -1;
    asm_changed(false);
    g_asm_fitted = false; asm_request_meshes();
    status("worm drive: a one-start worm (m4, q10, RH, 5.71 deg lead) turns a 30-tooth bronze wheel at 30:1; 50 W at 120 rpm in. Play, Power flow, Interference");
}
// a worked example: a cycloidal disc cam (base circle 50 mm, 15 mm lift: rise 120, dwell 60, return 120, dwell 60)
// on a camshaft in two bearings lifts a roller follower guided in a bridge, like an engine's valve train
static void asm_demo_cam() {
    asm_new();
    if (!g_asm_on) return;
    const double H = 0.06, xs = 0.05, zb = H + 0.08;
    js::Value base = js::Value::object(), bf = js::Value::array();
    base["name"] = "cam frame"; base["units"] = "m"; base["material"] = "cast_iron";
    bf.push(box_f("plate", "add", V3(0.16, 0.12, 0.012), V3(0, 0, -0.006)));
    for (int s = -1; s <= 1; s += 2) {
        bf.push(box_f(s < 0 ? "upright_left" : "upright_right", "add", V3(0.016, 0.05, zb + 0.015), V3(s * xs, 0, 0.5 * (zb + 0.015))));
        bf.push(cyl_f(s < 0 ? "housing_left" : "housing_right", "cut", 0.021, 0.03, V3(s * xs, 0, H), "x"));
    }
    bf.push(box_f("bridge", "add", V3(2 * xs + 0.016, 0.03, 0.015), V3(0, 0, zb + 0.0075)));
    bf.push(cyl_f("guide", "cut", 0.0051, 0.04, V3(0, 0, zb + 0.0075)));
    base["features"] = bf;
    asm_add_component(base, "frame", "", nullptr);                                                       // 0
    g_ac[0].fixed = true;
    Pose Rx; Rx.R[0] = 0; Rx.R[1] = 0; Rx.R[2] = 1; Rx.R[3] = 0; Rx.R[4] = 1; Rx.R[5] = 0; Rx.R[6] = -1; Rx.R[7] = 0; Rx.R[8] = 0;  // local z -> world x
    const BearingDef* b6004 = nullptr; for (int i = 0; i < NBEARING; ++i) if (std::string(BEARINGS[i].name) == "6004") b6004 = &BEARINGS[i];
    for (int s = -1; s <= 1; s += 2) { Pose p = Rx; p.t = V3(s * xs, 0, H); asm_add_component(bearing_part(*b6004), "bearing", "", &p); }  // 1, 2
    { Pose p = Rx; p.t = V3(0, 0, H); asm_add_component(keyshaft_part(20, 140), "camshaft", "", &p); }      // 3
    { Pose p = Rx; p.t = V3(0, 0, H); asm_add_component(cam_part(0, 0.025, 0.015, 120, 60, 120, 0.008, 0, 0.014, 0.0202), "cam", "", &p); }  // 4
    { Pose p; p.t = V3(0, 0, H + 0.04); asm_add_component(follower_part(0.008, 0, 0.014, 0.075, 0.010), "follower", "", &p); }  // 5
    auto lock = [&](int x, int y) {
        AMate m; m.type = MT_LOCK; m.a.comp = x; m.b.comp = y; m.id = "lock" + tostr((int)g_am.size() + 1);
        const Pose &Pa = g_ac[(size_t)x].pose, &Pb = g_ac[(size_t)y].pose;
        double Rt[9] = {Pa.R[0], Pa.R[3], Pa.R[6], Pa.R[1], Pa.R[4], Pa.R[7], Pa.R[2], Pa.R[5], Pa.R[8]};
        mat_mul(Rt, Pb.R, m.rel.R); m.rel.t = Pa.inv(Pb.t);
        g_am.push_back(m);
    };
    auto axis = [&](int c, V3 p, V3 d, double r, bool hole) { ARef q; q.comp = c; q.kind = 1; q.p = p; q.d = d; q.r = r; q.hole = hole; return q; };
    lock(0, 1); lock(0, 2);
    for (int k = 1; k <= 2; ++k) { AMate m; m.type = MT_CONCENTRIC; m.id = "bearing" + tostr(k); m.a = axis(k, V3(), V3(0, 0, 1), 0.010, true); m.b = axis(3, V3(), V3(0, 0, 1), 0.010, false); g_am.push_back(m); }
    asm_solve();
    lock(3, 4);
    g_pick_ref[0] = axis(4, V3(), V3(0, 0, 1), 0, false); g_pick_ref[1] = axis(5, V3(), V3(0, 1, 0), 0, false);
    asm_make_mate(MT_CAM);
    std::string note = g_st_last_note;
    { AMate m; m.type = MT_CONCENTRIC; m.id = "guide"; m.a = axis(0, V3(0, 0, zb + 0.0075), V3(0, 0, 1), 0.0051, true); m.b = axis(5, V3(0, 0.05, 0), V3(0, 1, 0), 0.005, false); g_am.push_back(m); }
    asm_solve();
    AMotor mo; mo.comp = 3; mo.p = V3(); mo.d = V3(0, 0, 1); mo.speed = 180; mo.power = 10; g_amot.push_back(mo);
    ATrace tr; tr.comp = 4; tr.p = V3(0, 0.04, 0.007); g_atr.push_back(tr);
    g_ms_T = 4;
    g_asm_sel = -1; g_asm_mate_sel = -1;
    asm_changed(false);
    g_asm_fitted = false; asm_request_meshes();
    status("cam and follower: a cycloidal disc cam at 30 rpm lifts a roller follower 15 mm" + note + ". Play (Space)");
}
static void asm_toggle_explode() {
    if (!g_asm_on) return;
    g_explode_to = g_explode_to > 0.5 ? 0 : 1;
    SetTimer(g_main, ASM_TIMER, 33, nullptr);
    status(g_explode_to > 0.5 ? "exploded view: fasteners out along their axes, parts apart (click again to collapse)" : "collapsing the exploded view");
}
static void asm_key_pose() {
    if (!g_asm_on) return;
    std::string v = fmtd(g_ms_t > 0 ? g_ms_t : g_ms_T);
    if (!g_selftest && !ask_text(g_main, "Key pose", "Time of this key pose in seconds (the moved parts keep these positions then; the motion between keys is smooth):", v)) return;
    double t = geo::clampd(atof(v.c_str()), 0, 3600);
    push_undo();
    bool base = !g_ms_based;
    AKey k; k.t = t;
    for (size_t c = 0; c < g_ac.size(); ++c) if (!g_ac[c].fixed) { k.comps.push_back((int)c); k.poses.push_back(g_ac[c].pose); }
    if (base) for (AComp& c : g_ac) c.base = c.pose;
    g_akeys.push_back(k);
    g_ms_T = std::max(g_ms_T, t);
    asm_changed(false);
    status("key pose at " + fmtd(t) + " s (" + tostr((int)k.comps.size()) + " parts). Move parts and add more keys, then Play");
}
static void asm_tool(int which) {  // the Assembly tools
    if (which != 0 && which != 1 && which != 11 && which != 20 && which != 24 && which != 33 && which != 39 && which != 42 && which != 46 && which != 49 && which != 50 && which != 52 && which != 53 && which != 55 && which != 47 && !g_asm_on) {
        if (MessageBoxA(g_main, "These tools work in an assembly. Make the open design the first part of a new assembly?", "Assembly", MB_OKCANCEL | MB_ICONQUESTION) != IDOK) return;
        asm_from_design();
        if (!g_asm_on) return;
    }
    switch (which) {
        case 0: asm_new(); break;
        case 1: asm_from_design(); break;
        case 2: asm_insert_part(); break;
        case 3: asm_insert_fastener("bolt"); break;
        case 4: asm_insert_fastener("screw"); break;
        case 5: asm_insert_fastener("nut"); break;
        case 6: asm_insert_fastener("washer"); break;
        case 7: asm_insert_fastener("pin"); break;
        case 8: g_asm_pick = 1; status("Mate: click a face on the first part (plane, hole or shaft), right-click cancels"); break;
        case 9: g_asm_pick = 3; status("Insert into hole: click the fastener's shank (or a shaft), then the hole"); break;
        case 10: asm_toggle_explode(); break;
        case 11: asm_demo(); break;
        case 12: g_asm_pick = 6; status("Motor: click a round face (hole or shaft) of the part that should turn; it turns about that axis"); break;
        case 13: asm_key_pose(); break;
        case 14: g_asm_pick = 5; status("Trace: click the point whose path you want to see"); break;
        case 15: ms_play(); break;
        case 16: ms_reset(); for (ATrace& t : g_atr) t.path.clear(); status("motion study reset to time 0"); InvalidateRect(g_view, nullptr, FALSE); break;
        case 17: asm_insert_fastener("shaft"); break;
        case 18: asm_insert_gear(0); break;
        case 21: asm_insert_gear(1); break;
        case 22: asm_insert_gear(2); break;
        case 23: asm_insert_gear(3); break;
        case 24: asm_demo_gears2(); break;
        case 25: asm_insert_drive(0); break;
        case 26: asm_insert_drive(1); break;
        case 27: asm_insert_drive(2); break;
        case 28: asm_insert_drive(3); break;
        case 29: asm_insert_drive(4); break;
        case 30: asm_insert_drive(5); break;
        case 31: asm_check(); break;
        case 32: power_setup(); break;
        case 33: asm_demo_drive(); break;
        case 34: g_asm_pick = 9; status("Roller mate: click the outer face of the driving roller, then of the roller it drives"); break;
        case 35: g_asm_pick = 11; status("Belt drive: click the driving pulley, then the pulley it drives (a V-belt runs round both)"); break;
        case 36: g_asm_pick = 13; status("Chain drive: click the driving sprocket, then the sprocket it drives (a roller chain runs round both)"); break;
        case 37: asm_insert_drive(6); break;
        case 38: asm_insert_drive(7); break;
        case 39: asm_demo_belts(); break;
        case 40: asm_insert_gear(4); break;
        case 41: asm_insert_gear(5); break;
        case 42: asm_demo_worm(); break;
        case 43: asm_insert_cam(false); break;
        case 44: asm_insert_cam(true); break;
        case 45: g_asm_pick = 15; status("Cam mate: click the cam, then its follower (place the follower beside the cam first: its line runs from the cam's axis through it)"); break;
        case 46: asm_demo_cam(); break;
        case 47: flow_setup(); break;
        case 48: flow_stop(false); status("flow simulation stopped (the report is in results/studio)"); break;
        case 49: flow_demo_cylinder(); break;
        case 50: flow_demo_tank(); break;
        case 51: flow_cycle_view(); break;
        case 52: flow_demo_gears_oil(); break;
        case 53: flow_demo_belt_fluid(); break;
        case 54: flow_cycle_mode(); break;
        case 55: flow_demo_splash(); break;
        case 56: flow_cycle_surface(); break;
        case 57: flow_cycle_walls(); break;
        case 58: flow_refine(); break;
        case 19: g_asm_pick = 7; status("Gear mate: click the first gear, then the gear it meshes with (their teeth are lined up)"); break;
        case 20: asm_demo_gears(); break;
    }
    InvalidateRect(g_view, nullptr, FALSE);
}
