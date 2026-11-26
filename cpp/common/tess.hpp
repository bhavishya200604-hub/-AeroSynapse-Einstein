// Exact display surfaces: the main aircraft features tessellated straight from their definitions, as structured grids
// (rows and columns of points with line-to-line connectivity), with exact normals and true edge lines (leading and
// trailing edges, roots, tips, section corners). The editor draws these instead of the sampled grid mesh, which can only
// approximate sharp edges and thin sections. Features changed by later cuts, fillets into other parts, shells or
// deformations are left to the sampled mesh, so what is drawn is always the real shape.
#pragma once
#include "sdf.hpp"

namespace geo {

struct TMesh {
    std::vector<V3> v, n;
    std::vector<int> t;
    std::vector<float> edges;  // line segments, 6 floats each
    int feature = -1;          // index in the design's feature list
};

struct TBuild {
    TMesh m;
    int grid(const std::vector<V3>& P, int nu, int nv, bool wrap_u) {  // P[j * nu + i]: quads between rows; returns the first vertex
        int base = (int)m.v.size();
        for (auto& p : P) { m.v.push_back(p); m.n.push_back(V3()); }
        int iu = wrap_u ? nu : nu - 1;
        for (int j = 0; j + 1 < nv; ++j)
            for (int i = 0; i < iu; ++i) {
                int i1 = (i + 1) % nu;
                int a = base + j * nu + i, b = base + j * nu + i1, c = base + (j + 1) * nu + i1, d = base + (j + 1) * nu + i;
                m.t.insert(m.t.end(), {a, b, c, a, c, d});
            }
        return base;
    }
    void fan(const std::vector<V3>& ring, const V3& centre) {  // a flat cap
        int base = (int)m.v.size();
        m.v.push_back(centre); m.n.push_back(V3());
        for (auto& p : ring) { m.v.push_back(p); m.n.push_back(V3()); }
        int k = (int)ring.size();
        for (int i = 0; i < k; ++i) m.t.insert(m.t.end(), {base, base + 1 + i, base + 1 + (i + 1) % k});
    }
    void line(const V3& a, const V3& b) { float s[6] = {(float)a.x, (float)a.y, (float)a.z, (float)b.x, (float)b.y, (float)b.z}; m.edges.insert(m.edges.end(), s, s + 6); }
    void polyline(const std::vector<V3>& p, bool closed) { for (size_t i = 0; i + 1 < p.size(); ++i) line(p[i], p[i + 1]); if (closed && p.size() > 2) line(p.back(), p.front()); }
    void normals() {  // area-weighted vertex normals (structured grids share vertices, so they come out smooth)
        for (size_t q = 0; q + 2 < m.t.size(); q += 3) {
            const V3 &a = m.v[(size_t)m.t[q]], &b = m.v[(size_t)m.t[q + 1]], &c = m.v[(size_t)m.t[q + 2]];
            V3 fn = cross(b - a, c - a);
            for (int k = 0; k < 3; ++k) m.n[(size_t)m.t[q + (size_t)k]] = m.n[(size_t)m.t[q + (size_t)k]] + fn;
        }
        for (auto& n : m.n) { double l = len(n); n = l > 0 ? n / l : V3(0, 0, 1); }
    }
};

inline std::vector<double> cosine_spacing(int n) { std::vector<double> u((size_t)n); for (int i = 0; i < n; ++i) u[(size_t)i] = 0.5 * (1 - std::cos(PI * i / (n - 1))); return u; }

// ---------------------------------------------------------------- local surfaces
inline V3 wing_point(const Feature& f, double eta, double u, double side) {  // side +1 upper, -1 lower; inverse of Feature::wing_local
    double c = f.cr * (1 - (1 - f.taper) * eta), xle = eta * f.span * f.sweep_t, th = f.twist * eta * PI / 180;
    double tcl = f.tip_tc > 0 ? f.tc + (f.tip_tc - f.tc) * eta : f.tc;
    double xi = u * c, ze = c * f.camber_at(u) + side * c * f.half_thick(u, tcl);
    double xr = xi - 0.25 * c, dx = xr * std::cos(th) + ze * std::sin(th), dz = -xr * std::sin(th) + ze * std::cos(th);
    double yl = eta * f.span, zl = dz;
    return V3(dx + xle + 0.25 * c, yl * f.cdi - zl * f.sdi, yl * f.sdi + zl * f.cdi);
}
inline void tess_wing(const Feature& f, TBuild& B) {
    std::vector<double> us = cosine_spacing(91);
    int nu = (int)us.size(), ns = 33;
    std::vector<V3> P;
    // one closed loop per span station: trailing edge over the upper surface to the leading edge and back under
    std::vector<double> loop_u; std::vector<double> loop_s;
    for (int i = nu - 1; i >= 0; --i) { loop_u.push_back(us[(size_t)i]); loop_s.push_back(1); }
    for (int i = 1; i < nu - 1; ++i) { loop_u.push_back(us[(size_t)i]); loop_s.push_back(-1); }
    int nl = (int)loop_u.size();
    for (int j = 0; j < ns; ++j) { double eta = (double)j / (ns - 1); for (int i = 0; i < nl; ++i) P.push_back(wing_point(f, eta, loop_u[(size_t)i], loop_s[(size_t)i])); }
    B.grid(P, nl, ns, true);
    for (int e = 0; e < 2; ++e) {  // flat root and tip faces, as in the distance function
        double eta = e;
        std::vector<V3> ring; for (int i = 0; i < nl; ++i) ring.push_back(wing_point(f, eta, loop_u[(size_t)i], loop_s[(size_t)i]));
        V3 cc; for (auto& p : ring) cc = cc + p; cc = cc / (double)ring.size();
        B.fan(ring, cc);
        B.polyline(ring, true);
    }
    std::vector<V3> le, te;
    for (int j = 0; j < ns; ++j) { double eta = (double)j / (ns - 1); le.push_back(wing_point(f, eta, 0, 1)); te.push_back(wing_point(f, eta, 1, 1)); }
    B.polyline(le, false); B.polyline(te, false);
}
inline void tess_fuselage(const Feature& f, TBuild& B) {
    int nx = 161, nt = 72;
    std::vector<V3> P;
    std::vector<std::vector<V3>> ends(2);
    for (int j = 0; j < nx; ++j) {
        double s = 0.5 * (1 - std::cos(PI * j / (nx - 1))), x = f.x0 + (f.x1 - f.x0) * s;  // stations clustered at nose and tail
        double A = std::max(f.fa.at(x), 1e-6), Bv = std::max(f.fb.at(x), 1e-6), Z = f.fz.at(x), n = std::max(f.fn.at(x), 1.0);
        for (int i = 0; i < nt; ++i) {
            double ph = 2 * PI * i / nt, c = std::cos(ph), sn = std::sin(ph);
            double y = A * (c < 0 ? -1 : 1) * std::pow(std::fabs(c), 2.0 / n), z = Z + Bv * (sn < 0 ? -1 : 1) * std::pow(std::fabs(sn), 2.0 / n);
            P.push_back(V3(x, y, z));
            if (j == 0) ends[0].push_back(V3(x, y, z));
            if (j == nx - 1) ends[1].push_back(V3(x, y, z));
        }
    }
    B.grid(P, nt, nx, true);
    for (int e = 0; e < 2; ++e) {
        double A = e ? f.fa.at(f.x1) : f.fa.at(f.x0);
        if (A < 0.03) continue;  // a pointed nose: no face to cap
        V3 cc; for (auto& p : ends[(size_t)e]) cc = cc + p; cc = cc / (double)ends[(size_t)e].size();
        B.fan(ends[(size_t)e], cc);
        B.polyline(ends[(size_t)e], true);
    }
}
inline void tess_revolve(const Feature& f, TBuild& B) {
    int nt = 72;
    const std::vector<P2>& pr = f.poly;
    if (pr.size() < 2) return;
    // each profile edge off the axis is turned into a band; vertices are not shared across sharp corners
    for (size_t i = 0; i + 1 < pr.size(); ++i) {
        P2 a = pr[i], b = pr[i + 1];
        if (a.y < 1e-9 && b.y < 1e-9) continue;
        std::vector<V3> P;
        for (int k = 0; k < 2; ++k) { P2 p = k ? b : a; for (int t = 0; t < nt; ++t) { double ph = 2 * PI * t / nt; P.push_back(V3(p.x, p.y * std::cos(ph), p.y * std::sin(ph))); } }
        B.grid(P, nt, 2, true);
    }
    // ring lines where the profile turns sharply (lips, steps), and at the ends
    for (size_t i = 1; i + 1 < pr.size(); ++i) {
        P2 d0(pr[i].x - pr[i - 1].x, pr[i].y - pr[i - 1].y), d1(pr[i + 1].x - pr[i].x, pr[i + 1].y - pr[i].y);
        double l0 = std::sqrt(d0.x * d0.x + d0.y * d0.y), l1 = std::sqrt(d1.x * d1.x + d1.y * d1.y);
        if (l0 < 1e-12 || l1 < 1e-12 || pr[i].y < 1e-6) continue;
        if ((d0.x * d1.x + d0.y * d1.y) / (l0 * l1) < std::cos(30 * PI / 180)) {
            std::vector<V3> ring; for (int t = 0; t < nt; ++t) { double ph = 2 * PI * t / nt; ring.push_back(V3(pr[i].x, pr[i].y * std::cos(ph), pr[i].y * std::sin(ph))); }
            B.polyline(ring, true);
        }
    }
}
inline void tess_ellipsoid(const V3& rad, TBuild& B) {
    int nt = 72, np = 37;
    std::vector<V3> P;
    for (int j = 0; j < np; ++j) {
        double th = PI * j / (np - 1);
        for (int i = 0; i < nt; ++i) { double ph = 2 * PI * i / nt; P.push_back(V3(rad.x * std::cos(th), rad.y * std::sin(th) * std::cos(ph), rad.z * std::sin(th) * std::sin(ph))); }
    }
    B.grid(P, nt, np, true);
}
inline V3 axis_point(int axis, double along, double c1, double c2) { V3 p; p[axis] = along; p[(axis + 1) % 3] = c1; p[(axis + 2) % 3] = c2; return p; }
inline void tess_cylinder(const Feature& f, TBuild& B) {
    int nt = 72;
    std::vector<V3> P;
    std::vector<std::vector<V3>> rings(2);
    for (int k = 0; k < 2; ++k)
        for (int t = 0; t < nt; ++t) { double ph = 2 * PI * t / nt; V3 p = axis_point(f.axis, (k - 0.5) * f.length, f.r * std::cos(ph), f.r * std::sin(ph)); P.push_back(p); rings[(size_t)k].push_back(p); }
    B.grid(P, nt, 2, true);
    for (int k = 0; k < 2; ++k) { B.fan(rings[(size_t)k], axis_point(f.axis, (k - 0.5) * f.length, 0, 0)); B.polyline(rings[(size_t)k], true); }
}
inline void tess_cone(const Feature& f, TBuild& B) {
    int nt = 72;
    std::vector<V3> P;
    std::vector<std::vector<V3>> rings(2);
    for (int k = 0; k < 2; ++k) {
        double rr = k ? f.r2 : f.r1;
        for (int t = 0; t < nt; ++t) { double ph = 2 * PI * t / nt; V3 p(k * f.length, rr * std::cos(ph), rr * std::sin(ph)); P.push_back(p); rings[(size_t)k].push_back(p); }
    }
    B.grid(P, nt, 2, true);
    for (int k = 0; k < 2; ++k) {
        if ((k ? f.r2 : f.r1) < 1e-6) continue;
        B.fan(rings[(size_t)k], V3(k * f.length, 0, 0)); B.polyline(rings[(size_t)k], true);
    }
}

// ---------------------------------------------------------------- which features can be drawn exactly
inline bool exact_kind(const Feature& f) {
    return f.type == WING || f.type == FUSELAGE || f.type == REVOLVE || f.type == SPHERE || f.type == ELLIPSOID || f.type == CYLINDER || f.type == CONE;
}
// a feature can be drawn exactly when it adds material and is not deformed, hollowed or thickened for display
inline bool exact_candidate(const Feature& f) {
    if (!exact_kind(f) || f.op != ADD || f.shell > 0 || f.infill >= 0 || f.twist_rate != 0 || f.taper_rate != 0 || f.bend != 0 || f.shear_y != 0 || f.shear_z != 0) return false;
    return f.stretch.x == 1 && f.stretch.y == 1 && f.stretch.z == 1 && f.inflate <= 0;
}
// ... and when no later cut or intersection reaches its surface (those parts are left to the sampled mesh, which shows
// the cut). A cut wholly inside the part (a bay, a duct inside a body) does not count; a small cut near the surface does.
inline bool exact_ok(const Model& m, size_t i, const struct TMesh& t);

// the feature in world space: its placement, pattern copies and mirror image
inline TMesh tessellate(const Feature& f) {
    TBuild B;
    switch (f.type) {
        case WING: tess_wing(f, B); break;
        case FUSELAGE: tess_fuselage(f, B); break;
        case REVOLVE: tess_revolve(f, B); break;
        case SPHERE: tess_ellipsoid(V3(f.r, f.r, f.r), B); break;
        case ELLIPSOID: tess_ellipsoid(f.rad, B); break;
        case CYLINDER: tess_cylinder(f, B); break;
        case CONE: tess_cone(f, B); break;
        default: return TMesh();
    }
    B.normals();
    TMesh L = B.m, W;
    auto place = [&](const V3& p, int k) {
        V3 q = f.xf.to_world(p);
        if (k && f.pat_rot != 0) { double a = f.pat_rot * k * PI / 180, c = std::cos(a), s = std::sin(a); q = V3(q.x, c * q.y - s * q.z, s * q.y + c * q.z); }
        return q + f.pat_step * k;
    };
    auto turn = [&](const V3& n, int k) {
        const double* R = f.xf.R;
        V3 q(R[0] * n.x + R[1] * n.y + R[2] * n.z, R[3] * n.x + R[4] * n.y + R[5] * n.z, R[6] * n.x + R[7] * n.y + R[8] * n.z);
        if (k && f.pat_rot != 0) { double a = f.pat_rot * k * PI / 180, c = std::cos(a), s = std::sin(a); q = V3(q.x, c * q.y - s * q.z, s * q.y + c * q.z); }
        return q;
    };
    for (int k = 0; k < std::max(1, f.pat_n); ++k) {
        int base = (int)W.v.size();
        for (size_t i = 0; i < L.v.size(); ++i) { W.v.push_back(place(L.v[i], k)); W.n.push_back(turn(L.n[i], k)); }
        for (int x : L.t) W.t.push_back(base + x);
        for (size_t i = 0; i + 5 < L.edges.size(); i += 6) {
            V3 a = place(V3(L.edges[i], L.edges[i + 1], L.edges[i + 2]), k), b = place(V3(L.edges[i + 3], L.edges[i + 4], L.edges[i + 5]), k);
            float s[6] = {(float)a.x, (float)a.y, (float)a.z, (float)b.x, (float)b.y, (float)b.z}; W.edges.insert(W.edges.end(), s, s + 6);
        }
    }
    if (f.mirror) {  // the distance function keeps the y >= 0 half and reflects it
        TMesh M;
        std::vector<int> keep;
        for (size_t q = 0; q + 2 < W.t.size(); q += 3) {
            double cy = (W.v[(size_t)W.t[q]].y + W.v[(size_t)W.t[q + 1]].y + W.v[(size_t)W.t[q + 2]].y) / 3;
            if (cy >= -1e-9) keep.insert(keep.end(), {W.t[q], W.t[q + 1], W.t[q + 2]});
        }
        int nv = (int)W.v.size();
        M.v = W.v; M.n = W.n;
        for (int i = 0; i < nv; ++i) { V3 p = W.v[(size_t)i], n = W.n[(size_t)i]; M.v.push_back(V3(p.x, -p.y, p.z)); M.n.push_back(V3(n.x, -n.y, n.z)); }
        M.t = keep;
        for (size_t q = 0; q + 2 < keep.size(); q += 3) M.t.insert(M.t.end(), {keep[q] + nv, keep[q + 2] + nv, keep[q + 1] + nv});
        for (size_t i = 0; i + 5 < W.edges.size(); i += 6) {
            if (W.edges[i + 1] < -1e-6 && W.edges[i + 4] < -1e-6) continue;
            M.edges.insert(M.edges.end(), W.edges.begin() + (long)i, W.edges.begin() + (long)i + 6);
            float s[6] = {W.edges[i], -W.edges[i + 1], W.edges[i + 2], W.edges[i + 3], -W.edges[i + 4], W.edges[i + 5]};
            M.edges.insert(M.edges.end(), s, s + 6);
        }
        W = M;
    }
    return W;
}

inline bool exact_ok(const Model& m, size_t i, const TMesh& t) {
    if (!exact_candidate(m.feats[i])) return false;
    const Box& a = m.boxes[i];
    for (size_t k = i + 1; k < m.feats.size(); ++k) {
        const Feature& c = m.feats[k];
        if (c.op == ADD) continue;
        const Box& b = m.boxes[k];
        if (!(a.lo.x <= b.hi.x && b.lo.x <= a.hi.x && a.lo.y <= b.hi.y && b.lo.y <= a.hi.y && a.lo.z <= b.hi.z && b.lo.z <= a.hi.z)) continue;
        if (c.op == INTERSECT) return false;
        double tol = 1e-6 + c.blend + c.chamfer;
        auto buried = [&](const V3& p) {  // inside another part: not visible, whatever the cut does there
            for (size_t j = 0; j < m.feats.size(); ++j) {
                if (j == i || j == k || m.feats[j].op != ADD) continue;
                const Box& bj = m.boxes[j];
                if (p.x < bj.lo.x || p.x > bj.hi.x || p.y < bj.lo.y || p.y > bj.hi.y || p.z < bj.lo.z || p.z > bj.hi.z) continue;
                if (m.feats[j].eval(p) < -1e-4) return true;
            }
            return false;
        };
        for (const V3& p : t.v) if (c.eval(p) < tol && !buried(p)) return false;  // the cut removes some of this visible surface
        V3 mid = (b.lo + b.hi) * 0.5;
        double spacing = len(a.hi - a.lo) / 60;
        if (len(b.hi - b.lo) < 3 * spacing && std::fabs(m.feats[i].eval(mid)) < 0.5 * len(b.hi - b.lo) + tol) {  // a cut smaller than the point spacing, at the surface
            bool inside = true;
            for (int q = 0; q < 8 && inside; ++q) { V3 cr((q & 1) ? b.hi.x : b.lo.x, (q & 2) ? b.hi.y : b.lo.y, (q & 4) ? b.hi.z : b.lo.z); if (m.feats[i].eval(cr) > -tol) inside = false; }
            if (!inside) return false;
        }
    }
    return true;
}

}  // namespace geo

