// aerokernel: command-line access to the AeroSynapse geometry kernel.
//
//   aerokernel check  design.json                         validate; prints errors as JSON
//   aerokernel info   design.json [--h 0.05] [--nx 120]   mass properties, areas, A(x) as JSON
//   aerokernel mesh   design.json out.stl|out.obj [--h 0.05] [--feature id]
//   aerokernel slice  design.json --axis x --at 8 [--h 0.02] out.json|out.svg
//   aerokernel eval   design.json points.bin out.bin       float64 xyz triplets -> float64 distance
//   aerokernel templates                                  default parameters of every feature type
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "../common/mesh.hpp"

using namespace geo;

static std::string arg(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
    return def;
}
static std::vector<std::string> positional(int argc, char** argv) {
    std::vector<std::string> out;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-' && argv[i][1] == '-') { ++i; continue; }
        out.push_back(argv[i]);
    }
    return out;
}

static double auto_h(const Model& m) {
    V3 e = m.bounds.hi - m.bounds.lo;
    return std::max(e.x, std::max(e.y, e.z)) / 220.0;
}

static js::Value props_json(const Props& P, const Model& m, double h, int nx, double seconds) {
    js::Value o = js::Value::object();
    auto arr3 = [](const V3& v) { js::Value a = js::Value::array(); a.push(v.x); a.push(v.y); a.push(v.z); return a; };
    o["h"] = h; o["volume"] = P.volume; o["wetted_area"] = P.area; o["frontal_area"] = P.frontal;
    o["planform_area"] = P.planform; o["side_area"] = P.side; o["length"] = P.length;
    o["centroid"] = arr3(P.centroid); o["bbox_lo"] = arr3(P.lo); o["bbox_hi"] = arr3(P.hi);
    js::Value I = js::Value::array(); for (double v : P.I) I.push(v); o["inertia_unit_density"] = I;
    o["triangles"] = (double)P.triangles; o["vertices"] = (double)P.vertices; o["seconds"] = seconds;
    o["features"] = (double)m.feats.size();
    // resample A(x) onto nx stations over the body length
    js::Value X = js::Value::array(), A = js::Value::array();
    for (int i = 0; i < nx; ++i) {
        double x = P.lo.x + (P.hi.x - P.lo.x) * i / (nx - 1);
        double a = 0;
        for (size_t k = 0; k + 1 < P.ax.size(); ++k)
            if (P.ax[k] <= x && x <= P.ax[k + 1]) { double t = (x - P.ax[k]) / (P.ax[k + 1] - P.ax[k]); a = P.A[k] * (1 - t) + P.A[k + 1] * t; break; }
        X.push(x); A.push(a);
    }
    o["area_x"] = X; o["area_A"] = A;
    return o;
}

static Model load_model(const std::string& path) {
    Model m;
    m.load(js::parse(js::read_file(path)));
    return m;
}

static int fail(const std::string& msg) {
    js::Value o = js::Value::object(); o["ok"] = false; o["error"] = msg;
    printf("%s\n", js::dump(o).c_str());
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: aerokernel check|info|mesh|slice|eval|templates ...\n"); return 1; }
    std::string cmd = argv[1];
    std::vector<std::string> pos = positional(argc - 1, argv + 1);
    try {
        if (cmd == "templates") {
            js::Value o = js::Value::object();
            for (auto& k : feature_kinds()) o[k] = template_for(k);
            printf("%s\n", js::dump(o).c_str());
            return 0;
        }
        if (pos.empty()) return fail("missing design file");
        Model m = load_model(pos[0]);
        if (cmd == "check") {
            js::Value o = js::Value::object(); js::Value e = js::Value::array();
            for (auto& s : m.errors) e.push(s);
            o["ok"] = m.errors.empty(); o["errors"] = e; o["features"] = (double)m.feats.size();
            js::Value lo = js::Value::array(), hi = js::Value::array();
            if (m.bounds.valid()) for (int i = 0; i < 3; ++i) { lo.push(m.bounds.lo[i]); hi.push(m.bounds.hi[i]); }
            o["bbox_lo"] = lo; o["bbox_hi"] = hi;
            printf("%s\n", js::dump(o).c_str());
            return m.errors.empty() ? 0 : 3;
        }
        if (!m.errors.empty()) return fail(m.errors[0]);
        double h = atof(arg(argc, argv, "--h", "0").c_str());
        if (h <= 0) h = auto_h(m);
        if (cmd == "info" || cmd == "mesh") {
            auto t0 = std::chrono::steady_clock::now();
            int feat = -1;
            std::string fid = arg(argc, argv, "--feature", "");
            if (!fid.empty()) { for (size_t i = 0; i < m.feats.size(); ++i) if (m.feats[i].id == fid) feat = (int)i; if (feat < 0) return fail("no feature " + fid); }
            Grid g = sample(m, h, nullptr, feat);
            Mesh M = surface_nets(m, g, 2, feat);
            Props P = properties(M, g);
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (cmd == "mesh") {
                if (pos.size() < 2) return fail("missing output file");
                std::string out = pos[1];
                if (out.size() > 4 && out.substr(out.size() - 4) == ".obj") write_obj(M, out); else write_stl(M, out, m.doc.str("name", "design"));
            }
            js::Value o = props_json(P, m, h, std::max(8, atoi(arg(argc, argv, "--nx", "120").c_str())), sec);
            o["ok"] = true;
            printf("%s\n", js::dump(o).c_str());
            return 0;
        }
        if (cmd == "slice") {
            std::string ax = arg(argc, argv, "--axis", "x");
            int axis = ax == "y" ? 1 : ax == "z" ? 2 : 0;
            double at = atof(arg(argc, argv, "--at", "0").c_str());
            Box ext;
            std::vector<Seg> segs = slice(m, axis, at, h, ext);
            if (pos.size() < 2) return fail("missing output file");
            std::string out = pos[1];
            if (out.size() > 4 && out.substr(out.size() - 4) == ".svg") {
                FILE* f = fopen(out.c_str(), "w");
                double W = ext.hi.x - ext.lo.x, H = ext.hi.y - ext.lo.y;
                fprintf(f, "<svg xmlns='http://www.w3.org/2000/svg' viewBox='%g %g %g %g'><g stroke='#1a5fb4' stroke-width='%g' fill='none'>\n", ext.lo.x, -ext.hi.y, W, H, std::max(W, H) / 400);
                for (auto& s : segs) fprintf(f, "<line x1='%.5f' y1='%.5f' x2='%.5f' y2='%.5f'/>\n", s.u0, -s.v0, s.u1, -s.v1);
                fprintf(f, "</g></svg>\n");
                fclose(f);
            } else {
                js::Value o = js::Value::object(), a = js::Value::array();
                for (auto& s : segs) { js::Value q = js::Value::array(); q.push(s.u0); q.push(s.v0); q.push(s.u1); q.push(s.v1); a.push(q); }
                o["axis"] = ax; o["at"] = at; o["segments"] = a;
                js::write_file(out, js::dump(o, 0));
            }
            printf("{\"ok\": true, \"segments\": %u}\n", (unsigned)segs.size());
            return 0;
        }
        if (cmd == "eval") {
            if (pos.size() < 3) return fail("eval needs points.bin and out.bin");
            std::string raw = js::read_file(pos[1]);
            size_t n = raw.size() / 24;
            const double* p = reinterpret_cast<const double*>(raw.data());
            std::vector<double> d(n);
            par::parallel_for((int)n, [&](int i) { d[(size_t)i] = m.eval(V3(p[3 * i], p[3 * i + 1], p[3 * i + 2])); });
            FILE* f = fopen(pos[2].c_str(), "wb");
            fwrite(d.data(), 8, n, f);
            fclose(f);
            printf("{\"ok\": true, \"points\": %u}\n", (unsigned)n);
            return 0;
        }
        return fail("unknown command " + cmd);
    } catch (std::exception& e) {
        return fail(e.what());
    }
}
