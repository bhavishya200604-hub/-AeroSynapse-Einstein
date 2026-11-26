// AeroSynapse geometry kernel: a design is an ordered feature tree evaluated as a signed distance
// field (negative inside). Features are added, cut or intersected, optionally with a smooth blend,
// so cuts, holes, slots, hollow cavities, ducts and fillets never fail the way mesh booleans can.
//
// Feature types
//   fuselage  stations [[x, half_width, half_height, z_centre, superellipse_n], ...] along local x
//   wing      span, root_chord, taper, sweep (leading edge, deg), dihedral (deg, 90 = vertical), twist (tip, deg),
//             tc (root t/c), tip_tc, camber, camber_pos, profile naca|biconvex|diamond|table, section {u,t,zc}
//             (canards, tails, strakes, blades, fins and ventral fins are all wings with different settings)
//   sphere r | ellipsoid r[3] | box size[3], round | cylinder r, length, axis | cone r1, r2, length (along x)
//   torus R, r (around x) | capsule a[3], b[3], r | duct path [[x,y,z],...], radius [...] | halfspace normal[3], offset
//   revolve   profile [[x, r], ...] turned about local x (nacelles, spinners, nose cones)
//   extrude   outline [[x, y], ...] (a closed curve in the local x-y plane) with height along z (plates, blades, fins)
//   loft      sections [{"x": .., "points": [[y, z], ...]}, ...] (closed curves, same point count) blended along x
//   sheet     control [16 points, 4 x 4 Bezier net] and thickness: a curved surface (skins, fairings, winglets)
//   smooth    on revolve, extrude, loft, duct: pass the curves through Catmull-Rom splines
//   lattice   size[3], cell, wall, kind gyroid|schwarz|diamond|struts: a periodic lattice block (TPMS or struts)
//   implicit  expr "f(x, y, z)" (solid where f <= 0, in local coordinates), size[3] bounding box: any equation surface
//   superellipsoid radii[3], n1 (vertical squareness), n2 (horizontal squareness): 1 ellipsoid, 0.1 box, 2 octahedron
//   bentplate profile [[x, z], ...] (an open polyline), thickness, width (along y), bend_radius: a sheet-metal part with
//             real bends (flanges, brackets); its flat length with a K factor of 0.44 is the unfolded blank
//   sweep     x, y, z: expressions of t, t0, t1, samples, radius (number or expression of t): a tube along a parametric
//             curve (helices, spirals, knots)
// Modifiers on every feature
//   op add|cut|intersect, blend (smooth radius) or chamfer (45-degree bevel size), draft (deg, on extrude), mirror (copy across y = 0), shell (wall thickness: hollow),
//   translate[3], rotate[3] (deg about x then y then z), scale, pattern {count, step[3], rotate (deg about x)},
//   twist_rate (deg per metre about local x), enabled, name.
// Transformations (applied in this order, world to shape): twist_rate, shear [dy/dx, dz/dx], bend (curvature 1/m
//   bending the local x axis towards +y), taper_rate (cross-section scale 1 + rate * x), stretch [sx, sy, sz]
//   (non-uniform scale); pivot[3] makes rotate turn about that local point (variable sweep, folding, morphing).
//   Distances after shear, bend, taper and stretch are scaled bounds, not exact: the sign and surface are exact.
// infill {type gyroid|schwarz|diamond|struts, cell, wall} on a hollow (shell) feature fills its inside with a lattice.
#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "expr.hpp"
#include "json.hpp"

namespace geo {

const double PI = 3.14159265358979323846;
const double BIG = 1e9;

struct V3 {
    double x = 0, y = 0, z = 0;
    V3() {}
    V3(double a, double b, double c) : x(a), y(b), z(c) {}
    V3 operator+(const V3& o) const { return V3(x + o.x, y + o.y, z + o.z); }
    V3 operator-(const V3& o) const { return V3(x - o.x, y - o.y, z - o.z); }
    V3 operator*(double s) const { return V3(x * s, y * s, z * s); }
    V3 operator/(double s) const { return V3(x / s, y / s, z / s); }
    double& operator[](int i) { return i == 0 ? x : i == 1 ? y : z; }
    double operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
};
inline double dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(const V3& a, const V3& b) { return V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
inline double len(const V3& a) { return std::sqrt(dot(a, a)); }
inline V3 vmin(const V3& a, const V3& b) { return V3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)); }
inline V3 vmax(const V3& a, const V3& b) { return V3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)); }
inline double clampd(double v, double a, double b) { return v < a ? a : v > b ? b : v; }
inline double sgn(double v) { return v < 0 ? -1.0 : 1.0; }

inline double smin(double a, double b, double k) {
    if (k <= 0) return std::min(a, b);
    double h = std::max(k - std::fabs(a - b), 0.0) / k;
    return std::min(a, b) - h * h * k * 0.25;
}
inline double smax(double a, double b, double k) { return -smin(-a, -b, k); }
// chamfered union / intersection: a flat 45-degree bevel of size c where two solids meet
inline double cmin(double a, double b, double c) { return std::min(std::min(a, b), (a + b - c) * 0.70710678118654752); }
inline double cmax(double a, double b, double c) { return std::max(std::max(a, b), (a + b + c) * 0.70710678118654752); }

struct Box {
    V3 lo = V3(BIG, BIG, BIG), hi = V3(-BIG, -BIG, -BIG);
    bool valid() const { return lo.x <= hi.x; }
    void add(const V3& p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    void add(const Box& b) { if (b.valid()) { add(b.lo); add(b.hi); } }
    Box pad(double d) const { Box b = *this; b.lo = lo - V3(d, d, d); b.hi = hi + V3(d, d, d); return b; }
    double dist(const V3& p) const {  // distance from outside, 0 inside
        double dx = std::max(std::max(lo.x - p.x, p.x - hi.x), 0.0);
        double dy = std::max(std::max(lo.y - p.y, p.y - hi.y), 0.0);
        double dz = std::max(std::max(lo.z - p.z, p.z - hi.z), 0.0);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

// Rotation matrix from Euler angles in degrees, applied about x, then y, then z: R = Rz Ry Rx.
struct Xf {
    V3 t;
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double s = 1.0;
    void set_rot(double ax, double ay, double az) {
        double a = ax * PI / 180, b = ay * PI / 180, c = az * PI / 180;
        double ca = std::cos(a), sa = std::sin(a), cb = std::cos(b), sb = std::sin(b), cc = std::cos(c), sc = std::sin(c);
        double Rx[9] = {1, 0, 0, 0, ca, -sa, 0, sa, ca}, Ry[9] = {cb, 0, sb, 0, 1, 0, -sb, 0, cb}, Rz[9] = {cc, -sc, 0, sc, cc, 0, 0, 0, 1};
        double T[9];
        mul(Ry, Rx, T);
        mul(Rz, T, R);
    }
    static void mul(const double* A, const double* B, double* C) {
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
    }
    V3 to_local(const V3& p) const {
        V3 q = (p - t) / s;
        return V3(R[0] * q.x + R[3] * q.y + R[6] * q.z, R[1] * q.x + R[4] * q.y + R[7] * q.z, R[2] * q.x + R[5] * q.y + R[8] * q.z);
    }
    V3 to_world(const V3& q) const {
        return t + V3(R[0] * q.x + R[1] * q.y + R[2] * q.z, R[3] * q.x + R[4] * q.y + R[5] * q.z, R[6] * q.x + R[7] * q.y + R[8] * q.z) * s;
    }
};

// Monotone cubic interpolation (Fritsch-Carlson), so a fuselage never overshoots between stations.
struct Pchip {
    std::vector<double> x, y, m;
    void build(const std::vector<double>& xs, const std::vector<double>& ys) {
        x = xs; y = ys; size_t n = x.size(); m.assign(n, 0.0);
        if (n < 2) return;
        std::vector<double> d(n - 1);
        for (size_t i = 0; i + 1 < n; ++i) d[i] = (y[i + 1] - y[i]) / std::max(x[i + 1] - x[i], 1e-12);
        m[0] = d[0]; m[n - 1] = d[n - 2];
        for (size_t i = 1; i + 1 < n; ++i) m[i] = (d[i - 1] * d[i] <= 0) ? 0.0 : 2.0 / (1.0 / d[i - 1] + 1.0 / d[i]);  // harmonic mean
    }
    double at(double t) const {
        size_t n = x.size();
        if (n == 0) return 0.0;
        if (n == 1 || t <= x[0]) return y[0];
        if (t >= x[n - 1]) return y[n - 1];
        size_t i = (size_t)(std::upper_bound(x.begin(), x.end(), t) - x.begin()) - 1;
        double h = x[i + 1] - x[i], s = (t - x[i]) / h, s2 = s * s, s3 = s2 * s;
        return (2 * s3 - 3 * s2 + 1) * y[i] + (s3 - 2 * s2 + s) * h * m[i] + (-2 * s3 + 3 * s2) * y[i + 1] + (s3 - s2) * h * m[i + 1];
    }
};

enum Type { SPHERE, ELLIPSOID, BOX, CYLINDER, CONE, TORUS, CAPSULE, DUCT, HALFSPACE, FUSELAGE, WING, REVOLVE, EXTRUDE, LOFT, SHEET, LATTICE, IMPLICIT, SUPERELLIPSOID, BENTPLATE, GEAR, RACK, WORM, UNKNOWN };
enum Op { ADD, CUT, INTERSECT };

inline Type type_of(const std::string& s) {
    static const char* names[] = {"sphere", "ellipsoid", "box", "cylinder", "cone", "torus", "capsule", "duct", "halfspace", "fuselage"};
    for (int i = 0; i < 10; ++i) if (s == names[i]) return (Type)i;
    if (s == "wing" || s == "canard" || s == "tail" || s == "fin" || s == "strake" || s == "blade") return WING;
    if (s == "revolve") return REVOLVE;
    if (s == "extrude") return EXTRUDE;
    if (s == "loft") return LOFT;
    if (s == "lattice") return LATTICE;
    if (s == "implicit") return IMPLICIT;
    if (s == "superellipsoid") return SUPERELLIPSOID;
    if (s == "sweep") return DUCT;
    if (s == "bentplate") return BENTPLATE;
    if (s == "sheet") return SHEET;
    if (s == "gear") return GEAR;
    if (s == "rack") return RACK;
    if (s == "worm") return WORM;
    return UNKNOWN;
}

struct P2 { double x = 0, y = 0; P2() {} P2(double a, double b) : x(a), y(b) {} };

// Signed distance to a closed polygon (negative inside), after Inigo Quilez.
inline double sd_polygon(const std::vector<P2>& v, double px, double py) {
    size_t n = v.size();
    if (n < 3) return BIG;
    double d = (px - v[0].x) * (px - v[0].x) + (py - v[0].y) * (py - v[0].y), s = 1;
    for (size_t i = 0, j = n - 1; i < n; j = i, ++i) {
        double ex = v[j].x - v[i].x, ey = v[j].y - v[i].y, wx = px - v[i].x, wy = py - v[i].y;
        double t = clampd((wx * ex + wy * ey) / std::max(ex * ex + ey * ey, 1e-18), 0, 1);
        double bx = wx - ex * t, by = wy - ey * t;
        d = std::min(d, bx * bx + by * by);
        bool c1 = py >= v[i].y, c2 = py < v[j].y, c3 = ex * wy > ey * wx;
        if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) s = -s;
    }
    return s * std::sqrt(d);
}

// One tooth of an involute spur gear (module m, z teeth, pressure angle pa in degrees), from the middle of the root
// gap before it to the middle of the gap after it, at increasing angle: tooth centred on +x. The outline is
// star-shaped about the axis. Root circle 1.25 m inside the pitch circle, tip circle m outside it.
inline std::vector<P2> gear_tooth(double m, int z, double pa_deg, int nflank = 9, double ha = 1.0) {
    const double pa = pa_deg * PI / 180, pitch = 2 * PI / z;
    double r = 0.5 * m * z, rb = r * std::cos(pa), ra = r + ha * m, rf = std::max(r - 1.25 * m, 0.3 * r);
    auto inv = [](double a) { return std::tan(a) - a; };
    auto half = [&](double R) { return std::max(0.03 * pitch, PI / (2 * z) + inv(pa) - inv(std::acos(std::min(1.0, rb / std::max(R, rb))))); };  // half the tooth's angle at R (a tall tooth ends in a narrow tip)
    double rs = std::max(rb, rf);
    std::vector<P2> t;
    auto pt = [&](double R, double th) { t.push_back(P2(R * std::cos(th), R * std::sin(th))); };
    double g0 = -0.5 * pitch, a0 = -half(rs);
    for (int k = 0; k < 3; ++k) pt(rf, g0 + (a0 - g0) * k / 3.0);  // the root, from the middle of the gap
    if (rf < rb) pt(rf, -half(rb));
    for (int k = 0; k < nflank; ++k) { double R = rs + (ra - rs) * k / (nflank - 1.0); pt(R, -half(R)); }
    for (int k = 1; k < 4; ++k) pt(ra, -half(ra) + 2 * half(ra) * k / 4.0);  // the tip
    for (int k = nflank - 1; k >= 0; --k) { double R = rs + (ra - rs) * k / (nflank - 1.0); pt(R, half(R)); }
    if (rf < rb) pt(rf, half(rb));
    for (int k = 1; k <= 3; ++k) pt(rf, -a0 + (-g0 - -a0) * k / 3.0);
    return t;
}

// Catmull-Rom resampling of a polyline (open) or closed curve, k points per segment.
inline std::vector<P2> spline2(const std::vector<P2>& c, bool closed, int k = 8) {
    size_t n = c.size();
    if (n < 3) return c;
    std::vector<P2> out;
    size_t segs = closed ? n : n - 1;
    auto at = [&](long i) { if (closed) return c[(size_t)((i % (long)n + (long)n) % (long)n)]; return c[(size_t)std::max(0L, std::min((long)n - 1, i))]; };
    for (size_t s = 0; s < segs; ++s)
        for (int q = 0; q < k; ++q) {
            double t = (double)q / k, t2 = t * t, t3 = t2 * t;
            P2 p0 = at((long)s - 1), p1 = at((long)s), p2 = at((long)s + 1), p3 = at((long)s + 2);
            out.push_back(P2(0.5 * (2 * p1.x + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
                             0.5 * (2 * p1.y + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3)));
        }
    if (!closed) out.push_back(c.back());
    return out;
}

// Distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5).
inline double dist_triangle(const V3& p, const V3& a, const V3& b, const V3& c) {
    V3 ab = b - a, ac = c - a, ap = p - a;
    double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return len(ap);
    V3 bp = p - b; double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return len(bp);
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return len(p - (a + ab * (d1 / (d1 - d3))));
    V3 cp = p - c; double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return len(cp);
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return len(p - (a + ac * (d2 / (d2 - d6))));
    double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return len(p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))));
    double den = 1.0 / (va + vb + vc);
    return len(p - (a + ab * (vb * den) + ac * (vc * den)));
}

// Periodic lattices: triply periodic minimal surfaces (gyroid, Schwarz P, diamond) as walls of thickness `wall`,
// or a cubic strut lattice with struts of diameter `wall`. Cell size `cell`. Distance is a bound (|g| / max |grad g|).
inline int lattice_kind(const std::string& k) { return k == "schwarz" ? 1 : k == "diamond" ? 2 : k == "struts" ? 3 : 0; }
inline double lattice_dist(int kind, const V3& q, double cell, double wall) {
    cell = std::max(cell, 1e-4);
    if (kind == 3) {
        V3 r(q.x - cell * std::floor(q.x / cell + 0.5), q.y - cell * std::floor(q.y / cell + 0.5), q.z - cell * std::floor(q.z / cell + 0.5));
        double dx = std::sqrt(r.y * r.y + r.z * r.z), dy = std::sqrt(r.x * r.x + r.z * r.z), dz = std::sqrt(r.x * r.x + r.y * r.y);
        return std::min(dx, std::min(dy, dz)) - wall / 2;
    }
    double k = 2 * PI / cell, x = k * q.x, y = k * q.y, z = k * q.z, g, lip;
    if (kind == 1) { g = std::cos(x) + std::cos(y) + std::cos(z); lip = std::sqrt(3.0); }
    else if (kind == 2) {
        g = std::sin(x) * std::sin(y) * std::sin(z) + std::sin(x) * std::cos(y) * std::cos(z) + std::cos(x) * std::sin(y) * std::cos(z) + std::cos(x) * std::cos(y) * std::sin(z);
        lip = std::sqrt(3.0);
    } else { g = std::sin(x) * std::cos(y) + std::sin(y) * std::cos(z) + std::sin(z) * std::cos(x); lip = 1.5; }
    return std::fabs(g) / (k * lip) - wall / 2;
}

inline double naca_half(double u, double tc) {  // half thickness / chord, closed trailing edge
    u = clampd(u, 0, 1);
    return 5 * tc * (0.2969 * std::sqrt(u) - 0.1260 * u - 0.3516 * u * u + 0.2843 * u * u * u - 0.1036 * u * u * u * u);
}

struct Feature {
    std::string id, name, type_name, error;
    Type type = UNKNOWN;
    Op op = ADD;
    bool enabled = true, mirror = false;
    double blend = 0, shell = 0, twist_rate = 0, chamfer = 0, draft = 0;
    int src = -1;  // index in the design's feature list
    double min_half = 0;  // display only: a wing section is never drawn thinner than this (no sawtooth at thin trailing edges)
    double inflate = 0;  // display only: thin parts drawn at a visible minimum thickness (Model::load with display_h)
    // transformations and lattice infill
    V3 stretch = V3(1, 1, 1);
    double taper_rate = 0, bend = 0, shear_y = 0, shear_z = 0;
    int infill = -1, lat_kind = 0;  // infill: -1 none, else lattice kind
    double infill_cell = 0.15, infill_wall = 0.01, lat_cell = 0.2, lat_wall = 0.02;
    Expr fx;                                    // implicit: f(x, y, z)
    double se_n1 = 1, se_n2 = 1;                // superellipsoid exponents
    Xf xf;
    int pat_n = 1;
    V3 pat_step;
    double pat_rot = 0;
    // primitives
    double r = 0.5, length = 1, r1 = 0.5, r2 = 0.2, R = 1, round = 0, offset = 0;
    V3 rad = V3(1, 0.5, 0.5), size = V3(1, 1, 1), a, b = V3(1, 0, 0), normal = V3(0, 0, 1);
    int axis = 0;
    std::vector<V3> path;
    std::vector<double> radius;
    // fuselage
    Pchip fa, fb, fz, fn;
    double x0 = 0, x1 = 1;
    // wing
    double span = 1, cr = 1, taper = 0.5, sweep = 30, dihedral = 0, twist = 0, tc = 0.06, tip_tc = -1, camber = 0, campos = 0.4;
    int profile = 0;  // 0 naca, 1 biconvex, 2 diamond, 3 table
    Pchip tab_t, tab_z;
    double sweep_t = 0, cdi = 1, sdi = 0;
    // curve tools
    std::vector<P2> poly;                       // revolve: closed (x, r) region; extrude: outline; gear: one tooth
    double gear_pitch = 1;                      // gear: angle of one tooth (2 pi / teeth)
    std::vector<double> gear_ang;               // gear: the angle of each outline point
    double gear_tanhx = 0, gear_rp = 0, gear_apex = 0;  // helical: tan(helix) / pitch radius per metre of z; bevel: apex distance from the large end
    double rack_p = 0, rack_len = 0, rack_yb = 0;      // rack: pitch, length, bottom of the bar (pitch line at y = 0)
    double worm_px = 0, worm_lead = 0, worm_r1 = 0, worm_ra = 0, worm_rf = 0, worm_tana = 0, worm_hand = 1;  // worm: axial pitch, lead, radii, hand
    double height = 0.1, thickness = 0.02;
    std::vector<double> loft_x;                 // loft: section stations
    std::vector<std::vector<P2>> loft_s;        // loft: section curves (equal point counts)
    std::vector<V3> sheet_pts;                  // sheet: tessellated Bezier patch
    int sheet_n = 24;

    double loft_local(const V3& q) const {
        size_t n = loft_x.size();
        double xx = clampd(q.x, loft_x.front(), loft_x.back());
        size_t i = 0;
        while (i + 2 < n && xx > loft_x[i + 1]) ++i;
        double t = clampd((xx - loft_x[i]) / std::max(loft_x[i + 1] - loft_x[i], 1e-12), 0, 1);
        const std::vector<P2>& a = loft_s[i];
        const std::vector<P2>& b = loft_s[i + 1];
        std::vector<P2> c(a.size());
        for (size_t k = 0; k < a.size(); ++k) c[k] = P2(a[k].x + (b[k].x - a[k].x) * t, a[k].y + (b[k].y - a[k].y) * t);
        double f = sd_polygon(c, q.y, q.z), fe = std::max(loft_x.front() - q.x, q.x - loft_x.back());
        if (f > 0 && fe > 0) return std::sqrt(f * f + fe * fe);
        return std::max(f, fe);
    }
    double sheet_local(const V3& q) const {
        int m = sheet_n + 1;
        size_t best = 0; double bd = 1e300;
        for (size_t k = 0; k < sheet_pts.size(); ++k) { V3 d = sheet_pts[k] - q; double dd = dot(d, d); if (dd < bd) { bd = dd; best = k; } }
        int bi = (int)(best % (size_t)m), bj = (int)(best / (size_t)m);
        double d = std::sqrt(bd);
        for (int j = std::max(0, bj - 1); j < std::min(sheet_n, bj + 1); ++j)
            for (int i = std::max(0, bi - 1); i < std::min(sheet_n, bi + 1); ++i) {
                const V3 &a = sheet_pts[(size_t)(j * m + i)], &b = sheet_pts[(size_t)(j * m + i + 1)], &c = sheet_pts[(size_t)((j + 1) * m + i + 1)], &e = sheet_pts[(size_t)((j + 1) * m + i)];
                d = std::min(d, std::min(dist_triangle(q, a, b, c), dist_triangle(q, a, c, e)));
            }
        return d - 0.5 * thickness;
    }

    double half_thick(double u, double tcl) const {
        if (profile == 1) return 2 * tcl * clampd(u, 0, 1) * (1 - clampd(u, 0, 1));
        if (profile == 2) { double uu = clampd(u, 0, 1); return tcl * (uu < 0.5 ? uu : 1 - uu); }
        if (profile == 3) return 0.5 * tab_t.at(u) * (tcl / std::max(tc, 1e-9));
        return naca_half(u, tcl);
    }
    double camber_at(double u) const {
        if (profile == 3) return tab_z.at(u);
        if (camber <= 0) return 0.0;
        double m = camber, p = clampd(campos, 0.05, 0.95);
        return u < p ? m / (p * p) * (2 * p * u - u * u) : m / ((1 - p) * (1 - p)) * (1 - 2 * p + 2 * p * u - u * u);
    }

    double wing_local(const V3& q) const {
        double yl = q.y * cdi + q.z * sdi, zl = -q.y * sdi + q.z * cdi;  // undo dihedral (rotation about x)
        double eta = clampd(yl / span, 0, 1);
        double c = cr * (1 - (1 - taper) * eta);
        double xle = eta * span * sweep_t;
        double th = twist * eta * PI / 180;
        double dx = q.x - (xle + 0.25 * c), dz = zl;
        double xi = dx * std::cos(th) - dz * std::sin(th) + 0.25 * c, ze = dx * std::sin(th) + dz * std::cos(th);
        double tcl = tip_tc > 0 ? tc + (tip_tc - tc) * eta : tc;
        double u = xi / c;
        double f2;
        if (u < 0) f2 = std::sqrt(xi * xi + std::pow(ze - c * camber_at(0), 2)) - min_half;
        else if (u > 1) f2 = std::sqrt((xi - c) * (xi - c) + std::pow(ze - c * camber_at(1), 2)) - min_half;
        else f2 = std::fabs(ze - c * camber_at(u)) - std::max(c * half_thick(u, tcl), min_half);
        // inside a thin section the vertical gap underestimates the distance to the edges; that is safe
        double fs = std::max(yl - span, -yl);
        if (f2 > 0 && fs > 0) return std::sqrt(f2 * f2 + fs * fs);
        return std::max(f2, fs);
    }
    double fuselage_local(const V3& q) const {
        double xx = clampd(q.x, x0, x1);
        double A = std::max(fa.at(xx), 1e-6), B = std::max(fb.at(xx), 1e-6), Z = fz.at(xx), n = std::max(fn.at(xx), 1.0);
        double ry = std::fabs(q.y) / A, rz = std::fabs(q.z - Z) / B;
        double rn = std::pow(std::pow(ry, n) + std::pow(rz, n), 1.0 / n);
        double f = (std::min(rn, 50.0) - 1.0) * std::min(A, B);
        double fe = std::max(x0 - q.x, q.x - x1);
        if (f > 0 && fe > 0) return std::sqrt(f * f + fe * fe);
        return std::max(f, fe);
    }
    static double round_cone(const V3& p, const V3& a, const V3& b, double ra, double rb) {
        V3 ba = b - a; double l2 = dot(ba, ba);
        if (l2 < 1e-18) return len(p - a) - std::max(ra, rb);
        double rr = ra - rb;
        if (rr * rr >= l2) return std::min(len(p - a) - ra, len(p - b) - rb);
        double a2 = l2 - rr * rr, il2 = 1.0 / l2;
        V3 pa = p - a; double y = dot(pa, ba), z = y - l2;
        V3 w = pa * l2 - ba * y; double x2 = dot(w, w), y2 = y * y * l2, z2 = z * z * l2;
        double k = sgn(rr) * rr * rr * x2;
        if (sgn(z) * a2 * z2 > k) return std::sqrt(x2 + z2) * il2 - rb;
        if (sgn(y) * a2 * y2 < k) return std::sqrt(x2 + y2) * il2 - ra;
        return (std::sqrt(x2 * a2 * il2) + y * rr) * il2 - ra;
    }
    double local(V3 q) const {
        switch (type) {
            case SPHERE: return len(q) - r;
            case ELLIPSOID: {
                V3 k0v(q.x / rad.x, q.y / rad.y, q.z / rad.z), k1v(q.x / (rad.x * rad.x), q.y / (rad.y * rad.y), q.z / (rad.z * rad.z));
                double k0 = len(k0v), k1 = len(k1v);
                return k1 < 1e-12 ? -std::min(rad.x, std::min(rad.y, rad.z)) : k0 * (k0 - 1) / k1;
            }
            case BOX: {
                V3 d(std::fabs(q.x) - (size.x / 2 - round), std::fabs(q.y) - (size.y / 2 - round), std::fabs(q.z) - (size.z / 2 - round));
                V3 m = vmax(d, V3(0, 0, 0));
                return len(m) + std::min(std::max(d.x, std::max(d.y, d.z)), 0.0) - round;
            }
            case CYLINDER: {
                double ax = q[axis], rr = std::sqrt(q[(axis + 1) % 3] * q[(axis + 1) % 3] + q[(axis + 2) % 3] * q[(axis + 2) % 3]);
                double dr = rr - r, dh = std::fabs(ax) - length / 2;
                double ox = std::max(dr, 0.0), oy = std::max(dh, 0.0);
                return std::sqrt(ox * ox + oy * oy) + std::min(std::max(dr, dh), 0.0);
            }
            case CONE: {  // from x = 0 (radius r1) to x = length (radius r2)
                double hh = length / 2, qx = std::sqrt(q.y * q.y + q.z * q.z), qy = q.x - hh;
                double k1x = r2, k1y = hh, k2x = r2 - r1, k2y = 2 * hh;
                double cax = qx - std::min(qx, qy < 0 ? r1 : r2), cay = std::fabs(qy) - hh;
                double t = clampd(((k1x - qx) * k2x + (k1y - qy) * k2y) / (k2x * k2x + k2y * k2y), 0, 1);
                double cbx = qx - k1x + k2x * t, cby = qy - k1y + k2y * t;
                double s = (cbx < 0 && cay < 0) ? -1 : 1;
                return s * std::sqrt(std::min(cax * cax + cay * cay, cbx * cbx + cby * cby));
            }
            case TORUS: { double a1 = std::sqrt(q.y * q.y + q.z * q.z) - R; return std::sqrt(a1 * a1 + q.x * q.x) - r; }
            case CAPSULE: { V3 pa = q - a, ba = b - a; double h = clampd(dot(pa, ba) / std::max(dot(ba, ba), 1e-18), 0, 1); return len(pa - ba * h) - r; }
            case DUCT: {
                double d = BIG;
                for (size_t i = 0; i + 1 < path.size(); ++i) d = std::min(d, round_cone(q, path[i], path[i + 1], radius[i], radius[i + 1]));
                return d;
            }
            case HALFSPACE: return dot(q, normal) - offset;
            case FUSELAGE: return fuselage_local(q);
            case WING: return wing_local(q);
            case REVOLVE: return sd_polygon(poly, q.x, std::sqrt(q.y * q.y + q.z * q.z));
            case EXTRUDE: {
                double d2 = sd_polygon(poly, q.x, q.y), dz = std::fabs(q.z) - height / 2;
                if (draft != 0) d2 += (q.z + height / 2) * std::tan(draft * PI / 180);  // walls lean inward going up
                double ox = std::max(d2, 0.0), oy = std::max(dz, 0.0);
                return std::sqrt(ox * ox + oy * oy) + std::min(std::max(d2, dz), 0.0);
            }
            case GEAR: {  // in the point's own tooth sector: that tooth and its two neighbours
                double qx = q.x, qy = q.y, sc = 1;
                if (gear_apex > 0) {  // bevel: every section a copy of the large end's, scaled towards the apex
                    sc = std::max(0.05, (gear_apex - (q.z + height / 2)) / gear_apex);
                    qx /= sc; qy /= sc;
                }
                if (gear_tanhx != 0) {  // helical: the section turns along the face width
                    double an = -q.z * gear_tanhx, ca = std::cos(an), sa = std::sin(an), x0 = qx;
                    qx = x0 * ca - qy * sa; qy = x0 * sa + qy * ca;
                }
                double r = std::sqrt(qx * qx + qy * qy), th = std::atan2(qy, qx);
                double a = th - std::round(th / gear_pitch) * gear_pitch;
                double px = r * std::cos(a), py = r * std::sin(a), d = BIG;
                bool edge = std::fabs(a) > 0.25 * gear_pitch;  // near the gap's middle the next tooth may be nearer
                for (int sct = -1; sct <= 1; ++sct) {
                    if (sct && (!edge || (sct > 0) != (a > 0))) continue;
                    double c = std::cos(sct * gear_pitch), sn = std::sin(sct * gear_pitch), x = px * c + py * sn, y = -px * sn + py * c;
                    for (size_t i = 0; i + 1 < poly.size(); ++i) {
                        double ex = poly[i + 1].x - poly[i].x, ey = poly[i + 1].y - poly[i].y, wx = x - poly[i].x, wy = y - poly[i].y;
                        double t = clampd((wx * ex + wy * ey) / std::max(ex * ex + ey * ey, 1e-30), 0, 1), bx = wx - ex * t, by = wy - ey * t;
                        d = std::min(d, bx * bx + by * by);
                    }
                }
                d = std::sqrt(d);
                double R = poly.empty() ? 0 : std::sqrt(poly[0].x * poly[0].x + poly[0].y * poly[0].y);  // the outline's radius at angle a
                for (size_t i = 0; i + 1 < poly.size(); ++i) {
                    double t0 = gear_ang[i], t1 = gear_ang[i + 1];
                    if (t1 - t0 < 1e-12 || a < t0 || a > t1) continue;
                    double Dx = poly[i + 1].x - poly[i].x, Dy = poly[i + 1].y - poly[i].y, ca = std::cos(a), sa = std::sin(a);
                    double den = ca * Dy - sa * Dx;
                    if (std::fabs(den) > 1e-30) R = (poly[i].x * Dy - poly[i].y * Dx) / den;
                    break;
                }
                double d2 = (r < R ? -d : d) * sc, dz = std::fabs(q.z) - height / 2;
                if (gear_tanhx != 0) d2 /= std::sqrt(1 + std::pow(gear_tanhx * r, 2));  // the twist stretches distances
                double ox = std::max(d2, 0.0), oy = std::max(dz, 0.0);
                return std::sqrt(ox * ox + oy * oy) + std::min(std::max(d2, dz), 0.0);
            }
            case RACK: {  // straight 20 deg teeth along x, tips towards +y, pitch line y = 0, the bar below the roots
                double xr = q.x - std::round(q.x / rack_p) * rack_p, d = BIG, Y = poly.empty() ? 0 : poly[0].y;
                for (int sct = -1; sct <= 1; ++sct) {
                    double x = xr - sct * rack_p;
                    for (size_t i = 0; i + 1 < poly.size(); ++i) {
                        double ex = poly[i + 1].x - poly[i].x, ey = poly[i + 1].y - poly[i].y, wx = x - poly[i].x, wy = q.y - poly[i].y;
                        double t = clampd((wx * ex + wy * ey) / std::max(ex * ex + ey * ey, 1e-30), 0, 1), bx = wx - ex * t, by = wy - ey * t;
                        d = std::min(d, bx * bx + by * by);
                    }
                }
                for (size_t i = 0; i + 1 < poly.size(); ++i)  // the outline's height over xr
                    if (xr >= poly[i].x && xr <= poly[i + 1].x && poly[i + 1].x > poly[i].x) { Y = poly[i].y + (poly[i + 1].y - poly[i].y) * (xr - poly[i].x) / (poly[i + 1].x - poly[i].x); break; }
                d = std::sqrt(d);
                double dtop = q.y < Y ? -d : d;
                double d2 = std::max(dtop, std::max(rack_yb - q.y, std::fabs(q.x) - rack_len / 2)), dz = std::fabs(q.z) - height / 2;
                double ox = std::max(d2, 0.0), oy = std::max(dz, 0.0);
                return std::sqrt(ox * ox + oy * oy) + std::min(std::max(d2, dz), 0.0);
            }
            case WORM: {  // threads along z; the axial section a rack of pitch px with straight 20 deg flanks (ZA worm)
                double r = std::sqrt(q.x * q.x + q.y * q.y), th = std::atan2(q.y, q.x);
                double u = q.z - worm_hand * worm_lead * th / (2 * PI);
                u -= std::round(u / worm_px) * worm_px;
                double hw = 0.25 * worm_px - (r - worm_r1) * worm_tana;
                double ca = 1 / std::sqrt(1 + worm_tana * worm_tana), cl = 2 * PI * r / std::sqrt(4 * PI * PI * r * r + worm_lead * worm_lead);
                double thread = std::max((std::fabs(u) - hw) * ca * cl, r - worm_ra), core = r - worm_rf;
                double d2 = std::min(thread, core), dz = std::fabs(q.z) - height / 2;
                double ox = std::max(d2, 0.0), oy = std::max(dz, 0.0);
                return std::sqrt(ox * ox + oy * oy) + std::min(std::max(d2, dz), 0.0);
            }
            case LOFT: return loft_local(q);
            case SHEET: return sheet_local(q);
            case IMPLICIT: {
                V3 d(std::fabs(q.x) - size.x / 2, std::fabs(q.y) - size.y / 2, std::fabs(q.z) - size.z / 2);
                double bx = len(vmax(d, V3(0, 0, 0))) + std::min(std::max(d.x, std::max(d.y, d.z)), 0.0);
                double h = 1e-4 * std::max(size.x, std::max(size.y, size.z)), f0 = fx.eval(q.x, q.y, q.z);
                V3 g((fx.eval(q.x + h, q.y, q.z) - fx.eval(q.x - h, q.y, q.z)) / (2 * h), (fx.eval(q.x, q.y + h, q.z) - fx.eval(q.x, q.y - h, q.z)) / (2 * h),
                     (fx.eval(q.x, q.y, q.z + h) - fx.eval(q.x, q.y, q.z - h)) / (2 * h));
                double dd = f0 / std::max(len(g), 1e-6);  // first-order distance to the zero set
                if (!(dd == dd)) dd = BIG / 10;
                return std::max(bx, dd);
            }
            case BENTPLATE: {
                double dl = BIG;  // distance to the bent centre line in the x-z plane
                for (size_t i = 0; i + 1 < poly.size(); ++i) {
                    double ax = poly[i].x, az = poly[i].y, bx = poly[i + 1].x, bz = poly[i + 1].y;
                    double ex = bx - ax, ez = bz - az, l2 = ex * ex + ez * ez;
                    double t = l2 > 0 ? clampd(((q.x - ax) * ex + (q.z - az) * ez) / l2, 0, 1) : 0;
                    double dx = q.x - ax - t * ex, dz = q.z - az - t * ez;
                    dl = std::min(dl, std::sqrt(dx * dx + dz * dz));
                }
                double d2 = dl - thickness / 2, dy = std::fabs(q.y) - height / 2;
                double ox = std::max(d2, 0.0), oy = std::max(dy, 0.0);
                return std::sqrt(ox * ox + oy * oy) + std::min(std::max(d2, dy), 0.0);
            }
            case SUPERELLIPSOID: {
                double ax = std::fabs(q.x) / rad.x, ay = std::fabs(q.y) / rad.y, az = std::fabs(q.z) / rad.z;
                double e2 = std::max(se_n2, 0.05), e1 = std::max(se_n1, 0.05);
                double F = std::pow(std::pow(ax, 2 / e2) + std::pow(ay, 2 / e2), e2 / e1) + std::pow(az, 2 / e1);
                return (std::pow(F, e1 / 2) - 1) * std::min(rad.x, std::min(rad.y, rad.z));
            }
            case LATTICE: {
                V3 d(std::fabs(q.x) - size.x / 2, std::fabs(q.y) - size.y / 2, std::fabs(q.z) - size.z / 2);
                double bx = len(vmax(d, V3(0, 0, 0))) + std::min(std::max(d.x, std::max(d.y, d.z)), 0.0);
                return std::max(bx, lattice_dist(lat_kind, q, lat_cell, lat_wall));
            }
            default: return BIG;
        }
    }
    double eval_copy(V3 p, int k) const {
        if (k) {
            p = p - pat_step * k;
            if (pat_rot != 0) { double a = -pat_rot * k * PI / 180, c = std::cos(a), s = std::sin(a); p = V3(p.x, c * p.y - s * p.z, s * p.y + c * p.z); }
        }
        V3 q = xf.to_local(p);
        V3 q0 = q * xf.s;  // lattice infill is fixed to the part, in metres
        if (twist_rate != 0) { double a = -twist_rate * q.x * PI / 180, c = std::cos(a), s = std::sin(a); q = V3(q.x, c * q.y - s * q.z, s * q.y + c * q.z); }
        double kd = 1.0;
        if (shear_y != 0 || shear_z != 0) { q.y -= shear_y * q.x; q.z -= shear_z * q.x; kd /= std::sqrt(1 + shear_y * shear_y + shear_z * shear_z); }
        if (bend != 0) {  // exact circular bend: the local x axis becomes an arc of radius 1/|bend| (length preserved)
            double sg = bend > 0 ? 1.0 : -1.0, Rb = 1.0 / std::fabs(bend), X = q.x, Y = sg * q.y;
            double th = std::atan2(X, Rb - Y), rr = std::sqrt(X * X + (Rb - Y) * (Rb - Y));
            q = V3(Rb * th, sg * (Rb - rr), q.z);
        }
        if (taper_rate != 0) { double t = std::max(0.05, 1 + taper_rate * q.x); q.y /= t; q.z /= t; kd *= std::min(t, 1.0) / std::sqrt(1 + taper_rate * taper_rate); }
        if (stretch.x != 1 || stretch.y != 1 || stretch.z != 1) { q = V3(q.x / stretch.x, q.y / stretch.y, q.z / stretch.z); kd *= std::min(stretch.x, std::min(stretch.y, stretch.z)); }
        double d = local(q) * xf.s * kd;
        if (shell > 0) {
            double solid = d;
            d = std::fabs(d) - shell / 2;
            if (infill >= 0) d = std::min(d, std::max(solid, lattice_dist(infill, q0, infill_cell, infill_wall)));
        }
        return d - inflate;
    }
    double eval(V3 p) const {
        if (mirror) p.y = std::fabs(p.y);
        double d = BIG;
        for (int k = 0; k < pat_n; ++k) d = std::min(d, eval_copy(p, k));
        return d;
    }
    Box local_box() const {
        Box b;
        switch (type) {
            case SPHERE: b.add(V3(-r, -r, -r)); b.add(V3(r, r, r)); break;
            case ELLIPSOID: b.add(rad * -1); b.add(rad); break;
            case BOX: b.add(size * -0.5); b.add(size * 0.5); break;
            case CYLINDER: { V3 e(r, r, r); e[axis] = length / 2; b.add(e * -1); b.add(e); break; }
            case CONE: { double m = std::max(r1, r2); b.add(V3(0, -m, -m)); b.add(V3(length, m, m)); break; }
            case TORUS: b.add(V3(-r, -R - r, -R - r)); b.add(V3(r, R + r, R + r)); break;
            case CAPSULE: b.add(a); b.add(this->b); b = b.pad(r); break;  // this->b: the end point (the local box is also called b)
            case DUCT: for (size_t i = 0; i < path.size(); ++i) { Box s; s.add(path[i]); b.add(s.pad(radius[i])); } break;
            case HALFSPACE: b.add(V3(-BIG / 10, -BIG / 10, -BIG / 10)); b.add(V3(BIG / 10, BIG / 10, BIG / 10)); break;
            case FUSELAGE: {
                for (int i = 0; i <= 200; ++i) {
                    double xx = x0 + (x1 - x0) * i / 200.0, A = fa.at(xx), B = fb.at(xx), Z = fz.at(xx);
                    b.add(V3(xx, -A, Z - B)); b.add(V3(xx, A, Z + B));
                }
                break;
            }
            case WING: {
                double ct = cr * taper, zt = std::max(cr, ct) * std::max(tc, tip_tc) + std::max(cr, ct) * (std::fabs(camber) + std::fabs(std::sin(twist * PI / 180)));
                V3 pts[4] = {V3(0, 0, 0), V3(cr, 0, 0), V3(span * sweep_t, span, 0), V3(span * sweep_t + ct, span, 0)};
                for (auto& p : pts) for (int s = -1; s <= 1; s += 2) { double y = p.y, z = p.z + s * zt; b.add(V3(p.x, y * cdi - z * sdi, y * sdi + z * cdi)); }
                break;
            }
            case REVOLVE: { double rm = 0; for (auto& v : poly) { rm = std::max(rm, v.y); b.add(V3(v.x, 0, 0)); } b.add(V3(b.lo.x, -rm, -rm)); b.add(V3(b.hi.x, rm, rm)); break; }
            case EXTRUDE: for (auto& v : poly) { b.add(V3(v.x, v.y, -height / 2)); b.add(V3(v.x, v.y, height / 2)); } break;
            case GEAR: { double ra = 0; for (auto& v : poly) ra = std::max(ra, std::sqrt(v.x * v.x + v.y * v.y)); b.add(V3(-ra, -ra, -height / 2)); b.add(V3(ra, ra, height / 2)); break; }
            case RACK: { double top = 0; for (auto& v : poly) top = std::max(top, v.y); b.add(V3(-rack_len / 2, rack_yb, -height / 2)); b.add(V3(rack_len / 2, top, height / 2)); break; }
            case WORM: b.add(V3(-worm_ra, -worm_ra, -height / 2)); b.add(V3(worm_ra, worm_ra, height / 2)); break;
            case LOFT: for (size_t i = 0; i < loft_x.size(); ++i) for (auto& v : loft_s[i]) b.add(V3(loft_x[i], v.x, v.y)); break;
            case SHEET: for (auto& v : sheet_pts) b.add(v); b = b.pad(thickness); break;
            case LATTICE: case IMPLICIT: b.add(size * -0.5); b.add(size * 0.5); break;
            case SUPERELLIPSOID: b.add(rad * -1); b.add(rad); break;
            case BENTPLATE: for (auto& v : poly) { b.add(V3(v.x, -height / 2, v.y)); b.add(V3(v.x, height / 2, v.y)); } b = b.pad(thickness); break;
            default: break;
        }
        return b.pad(1e-6);
    }
    Box world_box() const {
        Box lb = local_box(), out;
        if (!lb.valid()) return out;
        // undo the shape-space deformations in reverse order: stretch, taper, bend, shear
        lb.lo = V3(lb.lo.x * stretch.x, lb.lo.y * stretch.y, lb.lo.z * stretch.z); lb.hi = V3(lb.hi.x * stretch.x, lb.hi.y * stretch.y, lb.hi.z * stretch.z);
        if (taper_rate != 0) {
            double t = std::max(std::max(0.05, 1 + taper_rate * lb.lo.x), std::max(0.05, 1 + taper_rate * lb.hi.x));
            lb.lo.y *= t; lb.lo.z *= t; lb.hi.y *= t; lb.hi.z *= t;
        }
        if (bend != 0) {
            double r = std::max(std::fabs(lb.lo.x), std::fabs(lb.hi.x)) + std::max(std::fabs(lb.lo.y), std::fabs(lb.hi.y));
            double g = std::min(r, 0.5 * std::fabs(bend) * r * r + 1e-6);
            lb = lb.pad(g);
        }
        if (shear_y != 0 || shear_z != 0) {
            double xm = std::max(std::fabs(lb.lo.x), std::fabs(lb.hi.x));
            lb.lo.y -= std::fabs(shear_y) * xm; lb.hi.y += std::fabs(shear_y) * xm;
            lb.lo.z -= std::fabs(shear_z) * xm; lb.hi.z += std::fabs(shear_z) * xm;
        }
        double grow = twist_rate != 0 ? std::max(std::fabs(lb.lo.y), std::fabs(lb.hi.y)) + std::max(std::fabs(lb.lo.z), std::fabs(lb.hi.z)) : 0;
        if (grow > 0) { lb.lo.y = lb.lo.z = -grow; lb.hi.y = lb.hi.z = grow; }
        for (int k = 0; k < pat_n; ++k)
            for (int c = 0; c < 8; ++c) {
                V3 q((c & 1) ? lb.hi.x : lb.lo.x, (c & 2) ? lb.hi.y : lb.lo.y, (c & 4) ? lb.hi.z : lb.lo.z);
                V3 p = xf.to_world(q);
                if (k) {
                    if (pat_rot != 0) { double a = pat_rot * k * PI / 180, cc = std::cos(a), s = std::sin(a); p = V3(p.x, cc * p.y - s * p.z, s * p.y + cc * p.z); }
                    p = p + pat_step * k;
                }
                out.add(p);
                if (mirror) out.add(V3(p.x, -p.y, p.z));
            }
        return out.pad(blend + chamfer + shell + inflate + 1e-6);
    }
};

inline V3 v3(const js::Value& o, const std::string& k, V3 def) {
    std::vector<double> v = o.nums(k);
    if (v.size() >= 3) return V3(v[0], v[1], v[2]);
    if (v.size() == 1) return V3(v[0], v[0], v[0]);
    return def;
}

inline Feature compile(const js::Value& o) {
    Feature f;
    f.id = o.str("id", "");
    f.name = o.str("name", f.id);
    f.type_name = o.str("type", "");
    f.type = type_of(f.type_name);
    if (f.type == UNKNOWN) f.error = "unknown feature type '" + f.type_name + "'";
    std::string op = o.str("op", "add");
    f.op = op == "cut" ? CUT : op == "intersect" ? INTERSECT : ADD;
    if (op != "add" && op != "cut" && op != "intersect") f.error = "op must be add, cut or intersect";
    f.enabled = o.flag("enabled", true);
    f.mirror = o.flag("mirror", f.type == WING && o.num("dihedral", 0) < 80);
    f.blend = std::max(0.0, o.num("blend", 0));
    f.chamfer = std::max(0.0, o.num("chamfer", 0));
    f.draft = o.num("draft", 0);
    f.shell = std::max(0.0, o.num("shell", 0));
    f.twist_rate = o.num("twist_rate", 0);
    f.xf.t = v3(o, "translate", V3());
    V3 rot = v3(o, "rotate", V3());
    f.xf.set_rot(rot.x, rot.y, rot.z);
    f.xf.s = std::max(1e-6, o.num("scale", 1.0));
    if (const js::Value* pv = o.find("pivot")) {  // rotate (and scale) about this local point instead of the origin
        V3 piv = v3(o, "pivot", V3());
        (void)pv;
        V3 tw = f.xf.to_world(piv);
        f.xf.t = f.xf.t * 2 + piv - tw;
    }
    {
        V3 st = v3(o, "stretch", V3(1, 1, 1));
        f.stretch = V3(std::max(1e-3, st.x), std::max(1e-3, st.y), std::max(1e-3, st.z));
    }
    f.taper_rate = o.num("taper_rate", 0);
    f.bend = o.num("bend", 0);
    {
        std::vector<double> sh = o.nums("shear");
        if (sh.size() >= 1) f.shear_y = sh[0];
        if (sh.size() >= 2) f.shear_z = sh[1];
    }
    if (const js::Value* in = o.find("infill")) {
        if (in->is_obj()) {
            f.infill = lattice_kind(in->str("type", "gyroid"));
            f.infill_cell = std::max(1e-3, in->num("cell", 0.15));
            f.infill_wall = std::max(1e-4, in->num("wall", 0.01));
        }
    }
    if (f.type == IMPLICIT) {
        f.size = v3(o, "size", V3(2, 2, 2));
        try { f.fx.compile(o.str("expr", "x^2 + y^2 + z^2 - 1")); } catch (std::exception& e) { f.error = e.what(); }
    }
    if (f.type == BENTPLATE) {  // round every corner of the polyline with the bend radius (centre line radius r + t / 2)
        std::vector<P2> c;
        if (const js::Value* a = o.find("profile")) for (size_t i = 0; i < a->size(); ++i) { const js::Value& q = a->at(i); if (q.size() >= 2) c.push_back(P2(q.at(0).n, q.at(1).n)); }
        f.thickness = std::max(1e-4, o.num("thickness", 0.003));
        f.height = std::max(1e-4, o.num("width", 0.3));
        double rc = std::max(0.0, o.num("bend_radius", 0.01)) + f.thickness / 2;
        if (c.size() < 2) f.error = "a bent plate needs at least two profile points";
        else {
            f.poly.push_back(c.front());
            for (size_t i = 1; i + 1 < c.size(); ++i) {
                double ax = c[i].x - c[i - 1].x, az = c[i].y - c[i - 1].y, bx = c[i + 1].x - c[i].x, bz = c[i + 1].y - c[i].y;
                double la = std::sqrt(ax * ax + az * az), lb = std::sqrt(bx * bx + bz * bz);
                if (la < 1e-9 || lb < 1e-9) continue;
                ax /= la; az /= la; bx /= lb; bz /= lb;
                double turn = std::acos(clampd(ax * bx + az * bz, -1, 1));
                double tl = std::min(rc * std::tan(turn / 2), 0.49 * std::min(la, lb));
                if (turn < 1e-3 || tl <= 0) { f.poly.push_back(c[i]); continue; }
                P2 s0(c[i].x - ax * tl, c[i].y - az * tl), s1(c[i].x + bx * tl, c[i].y + bz * tl);
                for (int k = 0; k <= 8; ++k) {  // quadratic Bezier through the corner: a smooth bend
                    double t = k / 8.0, u = 1 - t;
                    f.poly.push_back(P2(u * u * s0.x + 2 * u * t * c[i].x + t * t * s1.x, u * u * s0.y + 2 * u * t * c[i].y + t * t * s1.y));
                }
            }
            f.poly.push_back(c.back());
        }
    }
    if (f.type == SUPERELLIPSOID) {
        f.rad = v3(o, "radii", V3(1, 0.6, 0.5));
        f.se_n1 = o.num("n1", 0.5); f.se_n2 = o.num("n2", 0.5);
    }
    if (f.type == LATTICE) {
        f.size = v3(o, "size", V3(1, 1, 1));
        f.lat_cell = std::max(1e-3, o.num("cell", 0.2));
        f.lat_wall = std::max(1e-4, o.num("wall", 0.02));
        f.lat_kind = lattice_kind(o.str("kind", "gyroid"));
    }
    if (const js::Value* p = o.find("pattern")) {
        f.pat_n = std::max(1, (int)p->num("count", 1));
        f.pat_step = v3(*p, "step", V3());
        f.pat_rot = p->num("rotate", 0);
    }
    f.r = o.num("r", 0.5); f.length = o.num("length", 1.0); f.r1 = o.num("r1", 0.5); f.r2 = o.num("r2", 0.2);
    f.R = o.num("R", 1.0); f.round = std::max(0.0, o.num("round", 0)); f.offset = o.num("offset", 0);
    f.rad = v3(o, "radii", V3(1, 0.5, 0.5)); f.size = v3(o, "size", V3(1, 1, 1));
    f.a = v3(o, "a", V3()); f.b = v3(o, "b", V3(1, 0, 0));
    f.normal = v3(o, "normal", V3(0, 0, 1)); { double l = len(f.normal); f.normal = l > 0 ? f.normal / l : V3(0, 0, 1); }
    std::string ax = o.str("axis", "x"); f.axis = ax == "y" ? 1 : ax == "z" ? 2 : 0;
    f.round = std::min(f.round, 0.5 * std::min(f.size.x, std::min(f.size.y, f.size.z)));
    if (f.type_name == "sweep") {  // a tube along the parametric curve (x(t), y(t), z(t))
        Expr ex, ey, ez, er;
        try {
            ex.compile(o.str("x", "t")); ey.compile(o.str("y", "0")); ez.compile(o.str("z", "0"));
            const js::Value* rv = o.find("radius");
            if (rv && rv->kind == js::Value::Str) er.compile(rv->s); else er.compile(std::to_string(o.num("radius", 0.1)));
            double t0 = o.num("t0", 0), t1 = o.num("t1", 6.2832);
            int n = std::max(8, std::min(2000, (int)o.num("samples", 160)));
            for (int i = 0; i <= n; ++i) {
                double t = t0 + (t1 - t0) * i / n;
                f.path.push_back(V3(ex.eval(0, 0, 0, t), ey.eval(0, 0, 0, t), ez.eval(0, 0, 0, t)));
                f.radius.push_back(std::max(1e-4, er.eval(0, 0, 0, t)));
            }
        } catch (std::exception& e) { f.error = e.what(); }
    } else if (const js::Value* p = o.find("path")) {
        if (p->is_arr()) for (size_t i = 0; i < p->size(); ++i) { const js::Value& q = p->at(i); if (q.is_arr() && q.size() >= 3) f.path.push_back(V3(q.at(0).n, q.at(1).n, q.at(2).n)); }
        std::vector<double> rr = o.nums("radius");
        for (size_t i = 0; i < f.path.size(); ++i) f.radius.push_back(rr.empty() ? 0.3 : rr[std::min(i, rr.size() - 1)]);
        if (f.path.size() < 2) f.error = "a duct needs at least two path points";
        else if (o.flag("smooth", false) && f.path.size() >= 3) {  // spline the centre line and the radius together
            std::vector<V3> P; std::vector<double> Rr;
            size_t n = f.path.size(); int k = 6;
            for (size_t s = 0; s + 1 < n; ++s)
                for (int q = 0; q < k; ++q) {
                    double t = (double)q / k, t2 = t * t, t3 = t2 * t;
                    auto at = [&](long i) { return (size_t)std::max(0L, std::min((long)n - 1, i)); };
                    size_t i0 = at((long)s - 1), i1 = at((long)s), i2 = at((long)s + 1), i3 = at((long)s + 2);
                    auto cr = [&](double a, double b, double c, double d) { return 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3); };
                    P.push_back(V3(cr(f.path[i0].x, f.path[i1].x, f.path[i2].x, f.path[i3].x), cr(f.path[i0].y, f.path[i1].y, f.path[i2].y, f.path[i3].y), cr(f.path[i0].z, f.path[i1].z, f.path[i2].z, f.path[i3].z)));
                    Rr.push_back(std::max(1e-4, cr(f.radius[i0], f.radius[i1], f.radius[i2], f.radius[i3])));
                }
            P.push_back(f.path.back()); Rr.push_back(f.radius.back());
            f.path = P; f.radius = Rr;
        }
    } else if (f.type == DUCT) f.error = "a duct needs a path";
    auto curve2 = [&](const js::Value* a) {
        std::vector<P2> out;
        if (a && a->is_arr())
            for (size_t i = 0; i < a->size(); ++i) { const js::Value& q = a->at(i); if (q.is_arr() && q.size() >= 2) out.push_back(P2(q.at(0).n, q.at(1).n)); }
        return out;
    };
    bool smooth = o.flag("smooth", false);
    if (f.type == REVOLVE) {
        std::vector<P2> pr = curve2(o.find("profile"));
        if (pr.size() < 2) f.error = "a revolve needs a profile of at least two [x, r] points";
        else {
            if (smooth) pr = spline2(pr, false);
            for (auto& v : pr) v.y = std::max(v.y, 0.0);
            f.poly.push_back(P2(pr.front().x, 0.0));
            for (auto& v : pr) f.poly.push_back(v);
            f.poly.push_back(P2(pr.back().x, 0.0));
        }
    }
    if (f.type == EXTRUDE) {
        f.poly = curve2(o.find("outline"));
        f.height = std::max(1e-4, o.num("height", 0.1));
        if (f.poly.size() < 3) f.error = "an extrude needs an outline of at least three [x, y] points";
        else if (smooth) f.poly = spline2(f.poly, true);
    }
    if (f.type == GEAR) {
        // module (the normal module of a helical gear; at the large end of a bevel gear), teeth, width (face width; the
        // axial length of a bevel gear's teeth), pressure_angle (deg), helix (deg, + right hand, - left hand), cone
        // (pitch cone angle of a bevel gear, deg; 0 for a spur or helical gear)
        int z = (int)o.num("teeth", 20);
        double m = o.num("module", 0.002), pa = o.num("pressure_angle", 20), hx = o.num("helix", 0) * PI / 180, cone = o.num("cone", 0) * PI / 180;
        f.height = std::max(1e-5, o.num("width", o.num("height", 0.01)));
        if (z < 6 || z > 1000 || m <= 0) f.error = "a gear needs 6 to 1000 teeth and a module above 0";
        else if (std::fabs(hx) > 45 * PI / 180) f.error = "a helix angle above 45 deg is not supported";
        else if (cone < 0 || cone > 85 * PI / 180) f.error = "a bevel gear's cone angle must be 0 to 85 deg";
        else {
            double mt = m / std::cos(hx), pt = std::atan(std::tan(pa * PI / 180) / std::cos(hx)) * 180 / PI;  // transverse module and pressure angle
            f.poly = gear_tooth(mt, z, pt, 9, clampd(o.num("addendum", 1.0), 0.3, 2.0)); f.gear_pitch = 2 * PI / z;  // addendum: tip height in modules (a worm wheel's blank is taller, then throated)
            for (auto& v : f.poly) f.gear_ang.push_back(std::atan2(v.y, v.x));
            f.gear_rp = 0.5 * mt * z;
            f.gear_tanhx = std::tan(hx) / f.gear_rp;
            if (cone > 0) { f.gear_apex = f.gear_rp / std::tan(cone); if (f.height > 0.8 * f.gear_apex) f.error = "a bevel gear's teeth must stop well short of its apex (make the width smaller)"; }
        }
    }
    if (f.type == WORM) {  // module (axial), starts, diameter (pitch, m), length, pressure_angle (deg), hand (+1 right, -1 left)
        double m = o.num("module", 0.002), d1 = o.num("diameter", 10 * m), pa = o.num("pressure_angle", 20) * PI / 180;
        int z1 = (int)o.num("starts", 1);
        f.height = std::max(1e-5, o.num("length", 0.04));
        if (m <= 0 || z1 < 1 || z1 > 8 || d1 < 4 * m) f.error = "a worm needs a module above 0, 1 to 8 starts and a pitch diameter of 4 modules or more";
        else { f.worm_px = PI * m; f.worm_lead = z1 * f.worm_px; f.worm_r1 = 0.5 * d1; f.worm_ra = 0.5 * d1 + m; f.worm_rf = 0.5 * d1 - 1.25 * m; f.worm_tana = std::tan(pa); f.worm_hand = o.num("hand", 1) < 0 ? -1 : 1; }
    }
    if (f.type == RACK) {  // module, teeth (count), width (face width), height (of the bar under the roots), pressure_angle (deg)
        int n = (int)o.num("teeth", 12);
        double m = o.num("module", 0.002), pa = o.num("pressure_angle", 20) * PI / 180, body = o.num("height", 3 * m);
        f.height = std::max(1e-5, o.num("width", 0.01));
        if (n < 1 || n > 2000 || m <= 0) f.error = "a rack needs 1 to 2000 teeth and a module above 0";
        else {
            double p = PI * m, yr = -1.25 * m, yt = m;
            auto w = [&](double y) { return 0.25 * p - y * std::tan(pa); };  // half the tooth's width at height y
            f.poly = {P2(-0.5 * p, yr), P2(-w(yr), yr), P2(-w(yt), yt), P2(w(yt), yt), P2(w(yr), yr), P2(0.5 * p, yr)};
            f.rack_p = p; f.rack_len = n * p; f.rack_yb = yr - std::max(m, body);
        }
    }
    if (f.type == LOFT) {
        const js::Value* ss = o.find("sections");
        if (ss && ss->is_arr())
            for (size_t i = 0; i < ss->size(); ++i) {
                const js::Value& sec = ss->at(i);
                std::vector<P2> c = curve2(sec.find("points"));
                if (smooth && c.size() >= 3) c = spline2(c, true);
                f.loft_x.push_back(sec.num("x", 0));
                f.loft_s.push_back(c);
            }
        if (f.loft_s.size() < 2) f.error = "a loft needs at least two sections";
        for (size_t i = 1; i < f.loft_s.size() && f.error.empty(); ++i) {
            if (f.loft_s[i].size() != f.loft_s[0].size() || f.loft_s[i].size() < 3) f.error = "loft sections need the same number of points (at least three)";
            if (f.loft_x[i] <= f.loft_x[i - 1]) f.error = "loft sections must increase in x";
        }
    }
    if (f.type == SHEET) {
        const js::Value* cp = o.find("control");
        std::vector<V3> c;
        if (cp && cp->is_arr())
            for (size_t i = 0; i < cp->size(); ++i) { const js::Value& q = cp->at(i); if (q.is_arr() && q.size() >= 3) c.push_back(V3(q.at(0).n, q.at(1).n, q.at(2).n)); }
        f.thickness = std::max(1e-4, o.num("thickness", 0.02));
        if (c.size() != 16) f.error = "a sheet needs 16 control points (a 4 x 4 Bezier net)";
        else {
            auto bern = [](int i, double t) { double u = 1 - t; return i == 0 ? u * u * u : i == 1 ? 3 * t * u * u : i == 2 ? 3 * t * t * u : t * t * t; };
            int n = f.sheet_n;
            for (int j = 0; j <= n; ++j)
                for (int i = 0; i <= n; ++i) {
                    double u = (double)i / n, v = (double)j / n;
                    V3 pt;
                    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) pt = pt + c[(size_t)(b * 4 + a)] * (bern(a, u) * bern(b, v));
                    f.sheet_pts.push_back(pt);
                }
        }
    }
    if (f.type == FUSELAGE) {
        std::vector<double> X, A, B, Z, N;
        const js::Value* st = o.find("stations");
        if (st && st->is_arr())
            for (size_t i = 0; i < st->size(); ++i) {
                const js::Value& s = st->at(i);
                if (!s.is_arr() || s.size() < 3) continue;
                X.push_back(s.at(0).n); A.push_back(s.at(1).n); B.push_back(s.at(2).n);
                Z.push_back(s.size() > 3 ? s.at(3).n : 0.0); N.push_back(s.size() > 4 ? s.at(4).n : 2.0);
            }
        if (X.size() < 2) f.error = "a fuselage needs at least two stations";
        for (size_t i = 1; i < X.size(); ++i) if (X[i] <= X[i - 1]) f.error = "fuselage stations must increase in x";
        if (f.error.empty()) { f.fa.build(X, A); f.fb.build(X, B); f.fz.build(X, Z); f.fn.build(X, N); f.x0 = X.front(); f.x1 = X.back(); }
    }
    if (f.type == WING) {
        f.span = std::max(1e-4, o.num("span", 1.0)); f.cr = std::max(1e-4, o.num("root_chord", 1.0));
        f.taper = clampd(o.num("taper", 0.5), 0.0, 5.0); f.sweep = o.num("sweep", 30); f.dihedral = o.num("dihedral", 0);
        f.twist = o.num("twist", 0); f.tc = std::max(1e-4, o.num("tc", 0.06)); f.tip_tc = o.num("tip_tc", -1);
        f.camber = o.num("camber", 0); f.campos = o.num("camber_pos", 0.4);
        std::string pr = o.str("profile", f.type_name == "blade" ? "diamond" : "naca");
        f.profile = pr == "biconvex" ? 1 : pr == "diamond" ? 2 : pr == "table" ? 3 : 0;
        if (f.profile == 3) {
            const js::Value* s = o.find("section");
            std::vector<double> u = s ? s->nums("u") : std::vector<double>(), t = s ? s->nums("t") : std::vector<double>(), z = s ? s->nums("zc") : std::vector<double>();
            if (u.size() < 3 || t.size() != u.size()) f.error = "profile table needs section.u and section.t of equal length";
            else { f.tab_t.build(u, t); f.tab_z.build(u, z.size() == u.size() ? z : std::vector<double>(u.size(), 0.0)); double mx = 0; for (double v : t) mx = std::max(mx, v); f.tc = mx; }
        }
        f.sweep_t = std::tan(clampd(f.sweep, -85, 85) * PI / 180);
        f.cdi = std::cos(f.dihedral * PI / 180); f.sdi = std::sin(f.dihedral * PI / 180);
    }
    return f;
}

struct Model {
    js::Value doc;
    std::vector<Feature> feats;
    std::vector<Box> boxes;
    std::vector<std::string> errors;
    Box bounds;

    void load(const js::Value& d) {
        doc = d; feats.clear(); boxes.clear(); errors.clear(); bounds = Box();
        const js::Value* fl = d.find("features");
        if (!fl || !fl->is_arr()) { errors.push_back("the design has no 'features' list"); return; }
        for (size_t i = 0; i < fl->size(); ++i) {
            Feature f = compile(fl->at(i));
            if (!f.error.empty()) { char nm[32]; snprintf(nm, sizeof nm, "feature %d", (int)i); errors.push_back((f.id.empty() ? std::string(nm) : f.id) + ": " + f.error); continue; }
            if (!f.enabled) continue;
            f.src = (int)i;
            double dh = d.num("display_h", 0);
            if (dh > 0) {  // an editor's mesh: parts thinner than about 1.5 cells would vanish, so draw them that thick
                double t = 1e9;
                if (f.type == SHEET || f.type == BENTPLATE) t = f.thickness;
                else if (f.type == LATTICE) t = f.lat_wall;
                else if (f.type == EXTRUDE) t = f.height;
                if (f.shell > 0) t = std::min(t, f.shell);
                if (t < 1.5 * dh) f.inflate = 0.75 * dh - t / 2;
                if (f.type == WING) f.min_half = 0.8 * dh;  // thin trailing edges and tips: at least about one cell thick
            }
            Box b = f.world_box();
            feats.push_back(f); boxes.push_back(b);
            if (f.op == ADD) bounds.add(b);
        }
        if (!bounds.valid()) errors.push_back("nothing is added: the design is empty");
    }
    double eval(const V3& p) const {
        double d = BIG;
        bool any = false;
        for (size_t i = 0; i < feats.size(); ++i) {
            const Feature& f = feats[i];
            double bd = boxes[i].dist(p);
            if (f.op == ADD) {
                if (any && bd > d + f.blend + f.chamfer) { continue; }
                double v = f.eval(p);
                d = any ? (f.chamfer > 0 ? cmin(d, v, f.chamfer) : smin(d, v, f.blend)) : v;
                any = true;
            } else if (f.op == CUT) {
                if (!any) continue;
                if (bd > std::fabs(d) + f.blend + f.chamfer) { continue; }
                d = f.chamfer > 0 ? cmax(d, -f.eval(p), f.chamfer) : smax(d, -f.eval(p), f.blend);
            } else {
                if (!any) continue;
                if (bd > 0) { d = std::max(d, bd); continue; }
                d = f.chamfer > 0 ? cmax(d, f.eval(p), f.chamfer) : smax(d, f.eval(p), f.blend);
            }
        }
        return d;
    }
    // distance of a single feature, used to highlight it in an editor
    double eval_one(size_t i, const V3& p) const { return feats[i].eval(p); }
    V3 grad(const V3& p, double h) const {
        return V3(eval(p + V3(h, 0, 0)) - eval(p - V3(h, 0, 0)), eval(p + V3(0, h, 0)) - eval(p - V3(0, h, 0)), eval(p + V3(0, 0, h)) - eval(p - V3(0, 0, h))) / (2 * h);
    }
};

// Default parameters of every feature type, shared by the editor, the command line and Python.
inline js::Value template_for(const std::string& kind) {
    js::Value o = js::Value::object();
    auto arr = [](std::initializer_list<double> l) { js::Value a = js::Value::array(); for (double v : l) a.push(js::Value(v)); return a; };
    o["id"] = kind; o["type"] = kind; o["op"] = "add";
    if (kind == "fuselage") {
        js::Value st = js::Value::array();
        double s[6][5] = {{0, 0.02, 0.02, 0, 2}, {2, 0.45, 0.42, 0.05, 2}, {5, 0.8, 0.7, 0.1, 2.2}, {10, 0.9, 0.65, 0.05, 2.4}, {14, 0.65, 0.55, 0.05, 2.2}, {16, 0.45, 0.45, 0.05, 2}};
        for (auto& r : s) st.push(arr({r[0], r[1], r[2], r[3], r[4]}));
        o["stations"] = st;
    } else if (kind == "wing" || kind == "canard" || kind == "tail" || kind == "fin" || kind == "strake" || kind == "blade") {
        double span = 4.5, cr = 5, taper = 0.25, sweep = 40, di = 0, tc = 0.05;
        V3 t(6, 0.8, 0);
        if (kind == "canard") { span = 2.0; cr = 2.0; taper = 0.35; sweep = 50; t = V3(3.5, 0.7, 0.3); }
        if (kind == "tail") { span = 2.0; cr = 2.4; taper = 0.35; sweep = 40; t = V3(12.5, 0.8, 0); }
        if (kind == "fin") { span = 2.6; cr = 3.0; taper = 0.3; sweep = 45; di = 75; t = V3(11.5, 0.6, 0.4); }
        if (kind == "strake") { span = 0.9; cr = 4.5; taper = 0.05; sweep = 76; tc = 0.02; t = V3(2.5, 0.6, 0.05); }
        if (kind == "blade") { span = 0.25; cr = 0.4; taper = 0.6; sweep = 50; di = 90; tc = 0.06; t = V3(8, 2.5, 0.12); }
        o["span"] = span; o["root_chord"] = cr; o["taper"] = taper; o["sweep"] = sweep; o["dihedral"] = di;
        o["twist"] = 0.0; o["tc"] = tc; o["camber"] = 0.0; o["profile"] = kind == "blade" || kind == "strake" ? "diamond" : "naca";
        o["translate"] = arr({t.x, t.y, t.z});
        o["mirror"] = true;
    } else if (kind == "sphere") { o["r"] = 0.5; o["translate"] = arr({8, 0, 0}); }
    else if (kind == "ellipsoid") { o["radii"] = arr({1.5, 0.5, 0.4}); o["translate"] = arr({8, 0, 0}); }
    else if (kind == "box") { o["size"] = arr({1, 0.6, 0.4}); o["round"] = 0.05; o["translate"] = arr({8, 0, 0}); }
    else if (kind == "cylinder") { o["r"] = 0.3; o["length"] = 2.0; o["axis"] = "x"; o["translate"] = arr({8, 0, 0}); }
    else if (kind == "cone") { o["r1"] = 0.5; o["r2"] = 0.3; o["length"] = 1.5; o["translate"] = arr({14, 0, 0}); }
    else if (kind == "torus") { o["R"] = 0.6; o["r"] = 0.08; o["translate"] = arr({15.5, 0, 0}); }
    else if (kind == "capsule") { o["a"] = arr({4, 0, -0.4}); o["b"] = arr({9, 0, -0.4}); o["r"] = 0.25; }
    else if (kind == "duct") {
        js::Value p = js::Value::array(); p.push(arr({3.5, 1.0, -0.2})); p.push(arr({7.0, 0.6, -0.1})); p.push(arr({12.0, 0.4, 0.0}));
        o["path"] = p; o["radius"] = arr({0.35, 0.33, 0.35}); o["op"] = "cut"; o["mirror"] = true;
    } else if (kind == "halfspace") { o["normal"] = arr({0, 0, -1}); o["offset"] = -1.2; o["op"] = "cut"; }  // solid side: z > 1.2, so the cut trims the top
    else if (kind == "revolve") {  // a nacelle-like body of revolution
        js::Value p = js::Value::array();
        double pr[6][2] = {{0, 0.05}, {0.3, 0.32}, {1.0, 0.42}, {2.2, 0.42}, {2.9, 0.34}, {3.2, 0.28}};
        for (auto& r : pr) p.push(arr({r[0], r[1]}));
        o["profile"] = p; o["smooth"] = true; o["translate"] = arr({6, 2.2, -0.3}); o["mirror"] = true;
    } else if (kind == "worm") {  // a worm: module 2 mm, 1 start, 20 mm pitch diameter, 30 mm long, right hand
        o["module"] = 0.002; o["starts"] = 1.0; o["diameter"] = 0.02; o["length"] = 0.03; o["pressure_angle"] = 20.0; o["hand"] = 1.0; o["translate"] = arr({8, 0, 0}); o["mirror"] = false;
    } else if (kind == "rack") {  // a straight rack: module 2 mm, 12 teeth, 10 mm wide
        o["module"] = 0.002; o["teeth"] = 12.0; o["width"] = 0.01; o["height"] = 0.006; o["pressure_angle"] = 20.0; o["translate"] = arr({8, 0, 0}); o["mirror"] = false;
    } else if (kind == "gear") {  // an involute spur gear: module 2 mm, 20 teeth, 10 mm wide
        o["module"] = 0.002; o["teeth"] = 20.0; o["width"] = 0.01; o["pressure_angle"] = 20.0; o["translate"] = arr({8, 0, 0}); o["mirror"] = false;
    } else if (kind == "extrude") {  // a flat plate cut from a drawn outline
        js::Value p = js::Value::array();
        double pr[5][2] = {{0, 0}, {1.6, 0}, {1.2, 0.9}, {0.4, 1.0}, {0.1, 0.5}};
        for (auto& r : pr) p.push(arr({r[0], r[1]}));
        o["outline"] = p; o["height"] = 0.05; o["smooth"] = true; o["translate"] = arr({9.6, 1.3, 0.2}); o["rotate"] = arr({90, 0, 0}); o["mirror"] = true;
    } else if (kind == "loft") {  // a body through drawn cross-sections
        js::Value ss = js::Value::array();
        double xs[3] = {0, 3, 7}, sc[3] = {0.25, 0.8, 0.55};
        for (int i = 0; i < 3; ++i) {
            js::Value sec = js::Value::object(); sec["x"] = xs[i];
            js::Value pts = js::Value::array();
            for (int k = 0; k < 8; ++k) { double a = 2 * PI * k / 8; pts.push(arr({sc[i] * 1.3 * std::cos(a), sc[i] * (k < 4 ? 0.8 : 1.0) * std::sin(a)})); }
            sec["points"] = pts; ss.push(sec);
        }
        o["sections"] = ss; o["smooth"] = true; o["translate"] = arr({0, 0, 0});
    } else if (kind == "sheet") {  // a curved skin panel (bicubic Bezier surface)
        js::Value c = js::Value::array();
        for (int b = 0; b < 4; ++b) for (int a = 0; a < 4; ++a) c.push(arr({8 + a * 1.0, 1.0 + b * 0.8, 0.6 + 0.25 * std::sin(PI * a / 3) * std::sin(PI * b / 3)}));
        o["control"] = c; o["thickness"] = 0.03; o["mirror"] = true;
    } else if (kind == "bentplate") {  // a sheet-metal bracket: three flanges with 15 mm bends
        js::Value pr = js::Value::array();
        double pp[4][2] = {{0, 0}, {0.4, 0}, {0.4, 0.3}, {0.7, 0.3}};
        for (auto& r : pp) pr.push(arr({r[0], r[1]}));
        o["profile"] = pr; o["thickness"] = 0.004; o["width"] = 0.35; o["bend_radius"] = 0.015; o["translate"] = arr({7, 0.9, -0.6});
    } else if (kind == "implicit") {  // an equation surface: a pod whose radius ripples along x
        o["expr"] = "y^2 + z^2 - (0.35 + 0.1*sin(3*x))^2"; o["size"] = arr({5, 1.2, 1.2}); o["translate"] = arr({8, 0, -0.9});
    } else if (kind == "superellipsoid") {  // a rounded box between an ellipsoid and a brick
        o["radii"] = arr({1.4, 0.7, 0.5}); o["n1"] = 0.35; o["n2"] = 0.35; o["translate"] = arr({8, 0, 0});
    } else if (kind == "sweep") {  // a helix: a spring-like tube along a parametric curve
        o["x"] = "0.25*t"; o["y"] = "0.5*cos(t)"; o["z"] = "0.5*sin(t)"; o["t0"] = 0.0; o["t1"] = 25.13; o["samples"] = 240; o["radius"] = 0.06;
        o["translate"] = arr({6, 0, 1.6});
    } else if (kind == "lattice") {  // a gyroid lattice block (lightweight structure, heat exchanger core)
        o["size"] = arr({1.2, 0.6, 0.4}); o["cell"] = 0.2; o["wall"] = 0.02; o["kind"] = "gyroid"; o["translate"] = arr({8, 0, -0.2});
    }
    return o;
}

inline std::vector<std::string> feature_kinds() {
    return {"fuselage", "wing", "canard", "tail", "fin", "strake", "blade", "sphere", "ellipsoid", "box", "cylinder", "cone", "torus", "capsule", "duct", "halfspace", "revolve", "extrude", "loft", "sheet", "lattice", "implicit", "superellipsoid", "sweep", "bentplate", "gear", "rack", "worm"};
}

}  // namespace geo
