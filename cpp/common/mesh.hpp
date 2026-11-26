// Sampling, meshing and measuring a design.
//  * sample(): distance on a regular grid, coarse-to-fine so far-away blocks cost one evaluation
//  * surface_nets(): watertight triangle mesh by dual contouring: one vertex per cell where the surface's tangent
//    planes meet, so hole rims, slot corners and other sharp edges stay sharp (Ju et al. 2002, with Lindstrom's
//    truncated solve); AEROSYNAPSE_SURFACE_NETS=1 restores plain surface nets (vertices projected onto the surface)
//  * split_sharp(): duplicate vertices along creases, so shading is crisp on both sides of an edge
//  * properties(): volume, wetted area, centroid, inertia, area distribution A(x), projected areas
//  * slice(): section contours on an axis-aligned plane (marching squares)
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "par.hpp"
#include "sdf.hpp"

namespace geo {

struct Grid {
    int nx = 0, ny = 0, nz = 0;
    double h = 0;
    V3 o;
    std::vector<float> d;
    size_t id(int i, int j, int k) const { return ((size_t)k * ny + j) * nx + i; }
    V3 at(int i, int j, int k) const { return o + V3(i * h, j * h, k * h); }
};

inline Grid sample(const Model& m, double h, const Box* region = nullptr, int feature = -1) {
    Grid g;
    Box b = region ? *region : m.bounds;
    if (feature >= 0) b = m.boxes[(size_t)feature];
    b = b.pad(2.5 * h);
    const int B = 4;
    g.h = h; g.o = b.lo;
    g.nx = ((int)std::ceil((b.hi.x - b.lo.x) / h) / B + 1) * B + 1;
    g.ny = ((int)std::ceil((b.hi.y - b.lo.y) / h) / B + 1) * B + 1;
    g.nz = ((int)std::ceil((b.hi.z - b.lo.z) / h) / B + 1) * B + 1;
    g.d.assign((size_t)g.nx * g.ny * g.nz, 0.0f);
    int cx = (g.nx - 1) / B, cy = (g.ny - 1) / B, cz = (g.nz - 1) / B;
    auto f = [&](const V3& p) { return feature >= 0 ? m.eval_one((size_t)feature, p) : m.eval(p); };
    std::vector<float> c((size_t)(cx + 1) * (cy + 1) * (cz + 1));
    par::parallel_for(cz + 1, [&](int k) {
        for (int j = 0; j <= cy; ++j)
            for (int i = 0; i <= cx; ++i) c[((size_t)k * (cy + 1) + j) * (cx + 1) + i] = (float)f(g.at(i * B, j * B, k * B));
    }, 1);
    double reach = 1.6 * B * h * std::sqrt(3.0);  // the distance is a bound, not exact: keep a margin
    par::parallel_for(cz * cy, [&](int kj) {
        int bk = kj / cy, bj = kj % cy;
        {
            for (int bi = 0; bi < cx; ++bi) {
                float cc[8]; bool close_by = false;
                for (int q = 0; q < 8; ++q) {
                    cc[q] = c[((size_t)(bk + (q >> 2 & 1)) * (cy + 1) + bj + (q >> 1 & 1)) * (cx + 1) + bi + (q & 1)];
                    if (std::fabs(cc[q]) < reach) close_by = true;
                }
                for (int k = 0; k <= B; ++k)
                    for (int j = 0; j <= B; ++j)
                        for (int i = 0; i <= B; ++i) {
                            int I = bi * B + i, J = bj * B + j, K = bk * B + k;
                            float v;
                            if (close_by) v = (float)f(g.at(I, J, K));
                            else {
                                double u = (double)i / B, w = (double)j / B, s = (double)k / B;
                                v = (float)((1 - s) * ((1 - w) * ((1 - u) * cc[0] + u * cc[1]) + w * ((1 - u) * cc[2] + u * cc[3])) +
                                            s * ((1 - w) * ((1 - u) * cc[4] + u * cc[5]) + w * ((1 - u) * cc[6] + u * cc[7])));
                            }
                            g.d[g.id(I, J, K)] = v;
                        }
            }
        }
    }, 1);
    return g;
}

struct Mesh {
    std::vector<V3> v, n;
    std::vector<int> t;  // triangles, 3 indices each
    std::vector<float> scalar;  // optional per-vertex result (Cp, temperature ...)
};

inline Mesh surface_nets_classic(const Model& m, const Grid& g, int project_iters = 2, int feature = -1) {
    Mesh M;
    int nx = g.nx, ny = g.ny, nz = g.nz;
    std::vector<int> vid((size_t)nx * ny * nz, -1);
    static const int E[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (int k = 0; k + 1 < nz; ++k)
        for (int j = 0; j + 1 < ny; ++j)
            for (int i = 0; i + 1 < nx; ++i) {
                float c[8]; int in = 0;
                for (int q = 0; q < 8; ++q) { c[q] = g.d[g.id(i + (q & 1), j + (q >> 1 & 1), k + (q >> 2 & 1))]; if (c[q] < 0) in |= 1 << q; }
                if (in == 0 || in == 255) continue;
                V3 s; int cnt = 0;
                for (auto& e : E) {
                    float a = c[e[0]], b = c[e[1]];
                    if ((a < 0) == (b < 0)) continue;
                    double t = a / (a - b);
                    V3 pa((e[0] & 1), (e[0] >> 1 & 1), (e[0] >> 2 & 1)), pb((e[1] & 1), (e[1] >> 1 & 1), (e[1] >> 2 & 1));
                    s = s + pa + (pb - pa) * t; ++cnt;
                }
                vid[g.id(i, j, k)] = (int)M.v.size();
                M.v.push_back(g.at(i, j, k) + (s / cnt) * g.h);
            }
    auto quad = [&](int a, int b, int c, int d, bool flip) {
        if (a < 0 || b < 0 || c < 0 || d < 0) return;
        if (flip) { std::swap(b, d); }
        M.t.push_back(a); M.t.push_back(b); M.t.push_back(c);
        M.t.push_back(a); M.t.push_back(c); M.t.push_back(d);
    };
    for (int k = 1; k + 1 < nz; ++k)
        for (int j = 1; j + 1 < ny; ++j)
            for (int i = 1; i + 1 < nx; ++i) {
                bool s0 = g.d[g.id(i, j, k)] < 0;
                if (s0 != (g.d[g.id(i + 1, j, k)] < 0))
                    quad(vid[g.id(i, j - 1, k - 1)], vid[g.id(i, j, k - 1)], vid[g.id(i, j, k)], vid[g.id(i, j - 1, k)], !s0);
                if (s0 != (g.d[g.id(i, j + 1, k)] < 0))
                    quad(vid[g.id(i - 1, j, k - 1)], vid[g.id(i - 1, j, k)], vid[g.id(i, j, k)], vid[g.id(i, j, k - 1)], !s0);
                if (s0 != (g.d[g.id(i, j, k + 1)] < 0))
                    quad(vid[g.id(i - 1, j - 1, k)], vid[g.id(i, j - 1, k)], vid[g.id(i, j, k)], vid[g.id(i - 1, j, k)], !s0);
            }
    M.n.resize(M.v.size());
    double eps = 0.25 * g.h;
    par::parallel_for((int)M.v.size(), [&](int q) {
        V3 p = M.v[(size_t)q];
        auto f = [&](const V3& x) { return feature >= 0 ? m.eval_one((size_t)feature, x) : m.eval(x); };
        V3 gr;
        for (int it = 0; it <= project_iters; ++it) {
            gr = V3(f(p + V3(eps, 0, 0)) - f(p - V3(eps, 0, 0)), f(p + V3(0, eps, 0)) - f(p - V3(0, eps, 0)), f(p + V3(0, 0, eps)) - f(p - V3(0, 0, eps))) / (2 * eps);
            if (it == project_iters) break;
            double g2 = dot(gr, gr);
            if (g2 < 1e-12) break;
            V3 step = gr * (f(p) / g2);
            if (len(step) > 0.5 * g.h) break;  // keep the vertex in its cell close_by sharp edges
            p = p - step;
        }
        M.v[(size_t)q] = p;
        double l = len(gr);
        M.n[(size_t)q] = l > 0 ? gr / l : V3(0, 0, 1);
    });
    return M;
}

// ---------------------------------------------------------------- dual contouring
// eigen-decomposition of a symmetric 3 x 3 matrix (Jacobi rotations): A = V diag(w) V^T, eigenvectors in V's columns
inline void eig_sym3(const double A0[3][3], double V[3][3], double w[3]) {
    double A[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { A[i][j] = A0[i][j]; V[i][j] = i == j ? 1 : 0; }
    for (int sweep = 0; sweep < 12; ++sweep) {
        double off = std::fabs(A[0][1]) + std::fabs(A[0][2]) + std::fabs(A[1][2]);
        if (off < 1e-15) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                if (std::fabs(A[p][q]) < 1e-18) continue;
                double th = 0.5 * (A[q][q] - A[p][p]) / A[p][q];
                double t = (th >= 0 ? 1.0 : -1.0) / (std::fabs(th) + std::sqrt(th * th + 1));
                double c = 1 / std::sqrt(t * t + 1), sn = t * c;
                for (int k = 0; k < 3; ++k) {  // A <- J^T A J
                    double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - sn * akq; A[k][q] = sn * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - sn * aqk; A[q][k] = sn * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    double vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = c * vkp - sn * vkq; V[k][q] = sn * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < 3; ++i) w[i] = A[i][i];
}

inline Mesh dual_contour(const Model& m, const Grid& g, int feature = -1) {
    Mesh M;
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const double h = g.h;
    auto f = [&](const V3& x) { return feature >= 0 ? m.eval_one((size_t)feature, x) : m.eval(x); };
    static const int E[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    std::vector<int> vid((size_t)nx * ny * nz, -1);
    std::vector<size_t> cells;
    for (int k = 0; k + 1 < nz; ++k)
        for (int j = 0; j + 1 < ny; ++j)
            for (int i = 0; i + 1 < nx; ++i) {
                int in = 0;
                for (int q = 0; q < 8; ++q) if (g.d[g.id(i + (q & 1), j + (q >> 1 & 1), k + (q >> 2 & 1))] < 0) in |= 1 << q;
                if (in == 0 || in == 255) continue;
                vid[g.id(i, j, k)] = (int)cells.size();
                cells.push_back(g.id(i, j, k));
            }
    M.v.resize(cells.size()); M.n.resize(cells.size());
    const double eps = 0.01 * h;
    par::parallel_for((int)cells.size(), [&](int q) {
        size_t c = cells[(size_t)q];
        int i = (int)(c % (size_t)nx), j = (int)((c / (size_t)nx) % (size_t)ny), k = (int)(c / ((size_t)nx * ny));
        V3 lo = g.at(i, j, k), hi = lo + V3(h, h, h);
        float cv[8];
        for (int r = 0; r < 8; ++r) cv[r] = g.d[g.id(i + (r & 1), j + (r >> 1 & 1), k + (r >> 2 & 1))];
        double ATA[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, Atb[3] = {0, 0, 0};
        V3 mass, nsum, en[12];
        int cnt = 0;
        for (auto& e : E) {
            double a = cv[e[0]], b = cv[e[1]];
            if ((a < 0) == (b < 0)) continue;
            V3 pa = lo + V3((e[0] & 1), (e[0] >> 1 & 1), (e[0] >> 2 & 1)) * h, pb = lo + V3((e[1] & 1), (e[1] >> 1 & 1), (e[1] >> 2 & 1)) * h;
            // the crossing on the true distance (regula falsi from the grid values)
            V3 x0 = pa, x1 = pb, p = pa + (pb - pa) * (a / (a - b));
            double f0 = a, f1 = b;
            for (int it = 0; it < 4; ++it) {
                double fp = f(p);
                if (std::fabs(fp) < 1e-7 * h) break;
                if ((fp < 0) == (f0 < 0)) { x0 = p; f0 = fp; } else { x1 = p; f1 = fp; }
                if (std::fabs(f0 - f1) < 1e-30) break;
                p = x0 + (x1 - x0) * (f0 / (f0 - f1));
            }
            V3 n(f(p + V3(eps, 0, 0)) - f(p - V3(eps, 0, 0)), f(p + V3(0, eps, 0)) - f(p - V3(0, eps, 0)), f(p + V3(0, 0, eps)) - f(p - V3(0, 0, eps)));
            double ln = len(n);
            if (ln < 1e-12) continue;
            n = n / ln;
            double d = dot(n, p);
            for (int r = 0; r < 3; ++r) { for (int s2 = 0; s2 < 3; ++s2) ATA[r][s2] += n[r] * n[s2]; Atb[r] += n[r] * d; }
            en[cnt] = n; mass = mass + p; nsum = nsum + n; ++cnt;
        }
        V3 x;
        // a smooth patch (every edge normal within 30 deg of their mean) gets the mass point projected onto the
        // surface, exact on curved faces (the tangent planes of a convex face meet slightly outside it); only a real
        // edge or corner gets the tangent-plane solve
        bool smooth = false;
        if (cnt) {
            V3 nm = nsum / std::max(len(nsum), 1e-30);
            double mind = 1;
            for (int k = 0; k < cnt; ++k) mind = std::min(mind, dot(en[k], nm));
            smooth = mind > 0.866;
        }
        if (smooth) {
            x = mass / (double)cnt;
            for (int it = 0; it < 3; ++it) {
                double fx = f(x);
                V3 gr(f(x + V3(eps, 0, 0)) - f(x - V3(eps, 0, 0)), f(x + V3(0, eps, 0)) - f(x - V3(0, eps, 0)), f(x + V3(0, 0, eps)) - f(x - V3(0, 0, eps)));
                gr = gr / (2 * eps);
                double g2 = dot(gr, gr);
                if (g2 < 1e-12) break;
                V3 st = gr * (fx / g2);
                if (len(st) > 0.5 * h) break;
                x = x - st;
                if (std::fabs(fx) < 1e-9 * h) break;
            }
        } else
        if (!cnt) x = lo + V3(0.5 * h, 0.5 * h, 0.5 * h);
        else {
            V3 mp = mass / (double)cnt;
            double rr[3];
            for (int r = 0; r < 3; ++r) rr[r] = Atb[r] - (ATA[r][0] * mp.x + ATA[r][1] * mp.y + ATA[r][2] * mp.z);
            double V[3][3], w[3];
            eig_sym3(ATA, V, w);
            double wmax = std::max(w[0], std::max(w[1], w[2]));
            x = mp;
            for (int e2 = 0; e2 < 3; ++e2) {
                if (w[e2] < 0.06 * wmax) continue;  // truncated: no sharp feature along this direction (flat or gently curved)
                double proj = (V[0][e2] * rr[0] + V[1][e2] * rr[1] + V[2][e2] * rr[2]) / w[e2];
                x = x + V3(V[0][e2], V[1][e2], V[2][e2]) * proj;
            }
            // the vertex stays in its cell (a corner just outside belongs to the neighbour)
            const double tol = 0.02 * h;
            bool out = false;
            for (int r = 0; r < 3; ++r) if (x[r] < lo[r] - 0.5 * h || x[r] > hi[r] + 0.5 * h) out = true;
            if (out) x = mp;
            x = V3(std::min(std::max(x.x, lo.x - tol), hi.x + tol), std::min(std::max(x.y, lo.y - tol), hi.y + tol), std::min(std::max(x.z, lo.z - tol), hi.z + tol));
        }
        M.v[(size_t)q] = x;
        double l = len(nsum);
        M.n[(size_t)q] = l > 0 ? nsum / l : V3(0, 0, 1);
    });
    // one quad per grid edge that crosses the surface; split along the diagonal that follows a crease
    auto tri_n = [&](int a, int b, int c) { V3 n = cross(M.v[(size_t)b] - M.v[(size_t)a], M.v[(size_t)c] - M.v[(size_t)a]); double l = len(n); return l > 0 ? n / l : V3(); };
    auto quad = [&](int a, int b, int c, int d, bool flip) {
        if (a < 0 || b < 0 || c < 0 || d < 0) return;
        if (flip) std::swap(b, d);
        double d1 = dot(tri_n(a, b, c), tri_n(a, c, d)), d2 = dot(tri_n(a, b, d), tri_n(b, c, d));
        bool first;
        if (d1 > 0.97 && d2 > 0.97) first = len(M.v[(size_t)a] - M.v[(size_t)c]) <= len(M.v[(size_t)b] - M.v[(size_t)d]);  // flat: the shorter diagonal
        else first = d1 <= d2;  // a crease runs along the diagonal whose two triangles turn the most
        if (first) { M.t.insert(M.t.end(), {a, b, c, a, c, d}); }
        else { M.t.insert(M.t.end(), {a, b, d, b, c, d}); }
    };
    for (int k = 1; k + 1 < nz; ++k)
        for (int j = 1; j + 1 < ny; ++j)
            for (int i = 1; i + 1 < nx; ++i) {
                bool s0 = g.d[g.id(i, j, k)] < 0;
                if (s0 != (g.d[g.id(i + 1, j, k)] < 0))
                    quad(vid[g.id(i, j - 1, k - 1)], vid[g.id(i, j, k - 1)], vid[g.id(i, j, k)], vid[g.id(i, j - 1, k)], !s0);
                if (s0 != (g.d[g.id(i, j + 1, k)] < 0))
                    quad(vid[g.id(i - 1, j, k - 1)], vid[g.id(i - 1, j, k)], vid[g.id(i, j, k)], vid[g.id(i, j, k - 1)], !s0);
                if (s0 != (g.d[g.id(i, j, k + 1)] < 0))
                    quad(vid[g.id(i - 1, j - 1, k)], vid[g.id(i, j - 1, k)], vid[g.id(i, j, k)], vid[g.id(i - 1, j, k)], !s0);
            }
    return M;
}

inline Mesh surface_nets(const Model& m, const Grid& g, int project_iters = 2, int feature = -1) {
    static const bool classic = std::getenv("AEROSYNAPSE_SURFACE_NETS") != nullptr;
    if (classic) return surface_nets_classic(m, g, project_iters, feature);
    return dual_contour(m, g, feature);
}

// Crisp shading: a vertex whose triangles fall into groups facing apart by more than `deg` is duplicated, one copy per
// group with the group's area-weighted normal. map[new vertex] = old vertex (for per-vertex data such as owners).
inline void split_sharp(Mesh& M, double deg, std::vector<int>* map = nullptr) {
    size_t nv = M.v.size(), nt = M.t.size() / 3;
    if (map) { map->resize(nv); for (size_t i = 0; i < nv; ++i) (*map)[i] = (int)i; }
    if (!nt) return;
    std::vector<V3> fn(nt);
    for (size_t t = 0; t < nt; ++t) fn[t] = cross(M.v[(size_t)M.t[3 * t + 1]] - M.v[(size_t)M.t[3 * t]], M.v[(size_t)M.t[3 * t + 2]] - M.v[(size_t)M.t[3 * t]]);  // area-weighted
    std::vector<int> start(nv + 1, 0), inc(3 * nt);
    for (int v : M.t) ++start[(size_t)v + 1];
    for (size_t i = 0; i < nv; ++i) start[i + 1] += start[i];
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (size_t t = 0; t < nt; ++t) for (int e = 0; e < 3; ++e) inc[(size_t)fill[(size_t)M.t[3 * t + (size_t)e]]++] = (int)t;
    double cmin = std::cos(deg * PI / 180);
    std::vector<int> grp;
    for (size_t v = 0; v < nv; ++v) {
        int b = start[v], e = start[v + 1];
        if (e - b < 2) continue;
        std::vector<V3> rep;  // unit normal of each group's first triangle
        std::vector<V3> sum;
        grp.assign((size_t)(e - b), 0);
        for (int q = b; q < e; ++q) {
            V3 n = fn[(size_t)inc[(size_t)q]]; double l = len(n); V3 u = l > 0 ? n / l : V3();
            int gi = -1;
            for (size_t r = 0; r < rep.size(); ++r) if (dot(rep[r], u) > cmin) { gi = (int)r; break; }
            if (gi < 0) { gi = (int)rep.size(); rep.push_back(u); sum.push_back(V3()); }
            grp[(size_t)(q - b)] = gi; sum[(size_t)gi] = sum[(size_t)gi] + n;
        }
        if (rep.size() < 2) continue;
        std::vector<int> idx(rep.size());
        idx[0] = (int)v;
        for (size_t r = 1; r < rep.size(); ++r) {
            idx[r] = (int)M.v.size();
            M.v.push_back(M.v[v]); M.n.push_back(V3());
            if (!M.scalar.empty()) M.scalar.push_back(M.scalar[v]);
            if (map) map->push_back((int)v);
        }
        for (size_t r = 0; r < rep.size(); ++r) { double l = len(sum[r]); M.n[(size_t)idx[r]] = l > 0 ? sum[r] / l : M.n[v]; }
        for (int q = b; q < e; ++q) {
            int t = inc[(size_t)q], gi = grp[(size_t)(q - b)];
            if (!gi) continue;
            for (int k = 0; k < 3; ++k) if (M.t[3 * (size_t)t + (size_t)k] == (int)v) M.t[3 * (size_t)t + (size_t)k] = idx[(size_t)gi];
        }
    }
}

struct Props {
    double volume = 0, area = 0, frontal = 0, planform = 0, side = 0, length = 0;
    V3 centroid, lo, hi;
    double I[6] = {0, 0, 0, 0, 0, 0};  // Ixx Iyy Izz Ixy Ixz Iyz about the centroid, unit density
    std::vector<double> ax, A;  // cross-section area distribution
    size_t triangles = 0, vertices = 0;
};

inline Props properties(const Mesh& M, const Grid& g) {
    Props P;
    double s1 = 0, sx = 0, sy = 0, sz = 0, sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0;
    for (size_t q = 0; q + 2 < M.t.size(); q += 3) {
        const V3 &a = M.v[(size_t)M.t[q]], &b = M.v[(size_t)M.t[q + 1]], &c = M.v[(size_t)M.t[q + 2]];
        V3 cr = cross(b - a, c - a);
        P.area += 0.5 * len(cr);
        double d6 = dot(a, cross(b, c));  // six times the signed volume of the tetrahedron to the origin
        s1 += d6;
        V3 sm = a + b + c;
        sx += d6 * sm.x; sy += d6 * sm.y; sz += d6 * sm.z;
        auto sq = [&](int i, int j) { return a[i] * a[j] + b[i] * b[j] + c[i] * c[j] + sm[i] * sm[j]; };
        sxx += d6 * sq(0, 0); syy += d6 * sq(1, 1); szz += d6 * sq(2, 2); sxy += d6 * sq(0, 1); sxz += d6 * sq(0, 2); syz += d6 * sq(1, 2);
    }
    P.volume = s1 / 6;
    if (std::fabs(P.volume) > 1e-12) {
        P.centroid = V3(sx, sy, sz) / (24 * P.volume);
        double xx = sxx / 120, yy = syy / 120, zz = szz / 120, xy = sxy / 120, xz = sxz / 120, yz = syz / 120;
        V3 c = P.centroid; double V = P.volume;
        double Cxx = xx - V * c.x * c.x, Cyy = yy - V * c.y * c.y, Czz = zz - V * c.z * c.z;
        P.I[0] = Cyy + Czz; P.I[1] = Cxx + Czz; P.I[2] = Cxx + Cyy;
        P.I[3] = -(xy - V * c.x * c.y); P.I[4] = -(xz - V * c.x * c.z); P.I[5] = -(yz - V * c.y * c.z);
    }
    for (auto& p : M.v) { P.lo = P.vertices ? vmin(P.lo, p) : p; P.hi = P.vertices ? vmax(P.hi, p) : p; ++P.vertices; }
    P.triangles = M.t.size() / 3;
    P.length = P.hi.x - P.lo.x;
    double h2 = g.h * g.h;
    auto frac = [&](float d) { return clampd(0.5 - d / g.h, 0, 1); };
    std::vector<char> top((size_t)g.nx * g.ny, 0), sid((size_t)g.nx * g.nz, 0);
    for (int i = 0; i < g.nx; ++i) {
        double a = 0;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j) {
                float d = g.d[g.id(i, j, k)];
                a += frac(d) * h2;
                if (d < 0) { top[(size_t)j * g.nx + i] = 1; sid[(size_t)k * g.nx + i] = 1; }
            }
        P.ax.push_back(g.o.x + i * g.h); P.A.push_back(a);
        P.frontal = std::max(P.frontal, a);
    }
    for (char c : top) P.planform += c ? h2 : 0;
    for (char c : sid) P.side += c ? h2 : 0;
    return P;
}

inline void write_stl(const Mesh& M, const std::string& path, const std::string& title) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    char head[80] = {0};
    snprintf(head, sizeof head, "AeroSynapse %s", title.c_str());
    fwrite(head, 1, 80, f);
    uint32_t n = (uint32_t)(M.t.size() / 3);
    fwrite(&n, 4, 1, f);
    for (size_t q = 0; q + 2 < M.t.size(); q += 3) {
        const V3 &a = M.v[(size_t)M.t[q]], &b = M.v[(size_t)M.t[q + 1]], &c = M.v[(size_t)M.t[q + 2]];
        V3 nn = cross(b - a, c - a); double l = len(nn); if (l > 0) nn = nn / l;
        float rec[12] = {(float)nn.x, (float)nn.y, (float)nn.z, (float)a.x, (float)a.y, (float)a.z, (float)b.x, (float)b.y, (float)b.z, (float)c.x, (float)c.y, (float)c.z};
        fwrite(rec, 4, 12, f);
        uint16_t attr = 0; fwrite(&attr, 2, 1, f);
    }
    fclose(f);
}

inline void write_obj(const Mesh& M, const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + path);
    fprintf(f, "# AeroSynapse mesh\n");
    for (auto& p : M.v) fprintf(f, "v %.6f %.6f %.6f\n", p.x, p.y, p.z);
    for (auto& p : M.n) fprintf(f, "vn %.5f %.5f %.5f\n", p.x, p.y, p.z);
    for (size_t q = 0; q + 2 < M.t.size(); q += 3) fprintf(f, "f %d//%d %d//%d %d//%d\n", M.t[q] + 1, M.t[q] + 1, M.t[q + 1] + 1, M.t[q + 1] + 1, M.t[q + 2] + 1, M.t[q + 2] + 1);
    fclose(f);
}

// Section contours on the plane x_axis = at (axis 0, 1 or 2), as line segments in that plane's (u, v).
struct Seg { double u0, v0, u1, v1; };
inline std::vector<Seg> slice(const Model& m, int axis, double at, double h, Box& extent) {
    std::vector<Seg> out;
    Box b = m.bounds.pad(2 * h);
    int ua = (axis + 1) % 3, va = (axis + 2) % 3;
    if (axis == 1) { ua = 0; va = 2; }
    if (axis == 2) { ua = 0; va = 1; }
    if (axis == 0) { ua = 1; va = 2; }
    int nu = (int)std::ceil((b.hi[ua] - b.lo[ua]) / h) + 1, nv = (int)std::ceil((b.hi[va] - b.lo[va]) / h) + 1;
    std::vector<double> d((size_t)nu * nv);
    par::parallel_for(nv, [&](int j) {
        for (int i = 0; i < nu; ++i) {
            V3 p; p[axis] = at; p[ua] = b.lo[ua] + i * h; p[va] = b.lo[va] + j * h;
            d[(size_t)j * nu + i] = m.eval(p);
        }
    });
    extent = Box(); extent.add(V3(b.lo[ua], b.lo[va], 0)); extent.add(V3(b.hi[ua], b.hi[va], 0));
    for (int j = 0; j + 1 < nv; ++j)
        for (int i = 0; i + 1 < nu; ++i) {
            double c[4] = {d[(size_t)j * nu + i], d[(size_t)j * nu + i + 1], d[(size_t)(j + 1) * nu + i + 1], d[(size_t)(j + 1) * nu + i]};
            double pu[4] = {0, 1, 1, 0}, pv[4] = {0, 0, 1, 1};
            double xs[4], ys[4]; int n = 0;
            for (int e = 0; e < 4; ++e) {
                double a = c[e], bb = c[(e + 1) % 4];
                if ((a < 0) == (bb < 0)) continue;
                double t = a / (a - bb);
                xs[n] = b.lo[ua] + (i + pu[e] + (pu[(e + 1) % 4] - pu[e]) * t) * h;
                ys[n] = b.lo[va] + (j + pv[e] + (pv[(e + 1) % 4] - pv[e]) * t) * h;
                ++n;
            }
            if (n >= 2) out.push_back({xs[0], ys[0], xs[1], ys[1]});
            if (n == 4) out.push_back({xs[2], ys[2], xs[3], ys[3]});
        }
    return out;
}

}  // namespace geo
