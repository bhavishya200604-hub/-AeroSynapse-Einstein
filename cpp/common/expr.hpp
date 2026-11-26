// Small expression compiler for math-driven features: implicit surfaces f(x, y, z) <= 0 and parametric curves x(t).
//   numbers, variables x y z t, constants pi e, + - * / ^ (right-associative), unary minus, parentheses,
//   functions sin cos tan asin acos atan exp log sqrt abs floor sign (1 argument), min max pow atan2 mod (2 arguments).
// Compiled once to a postfix program; evaluation is a short loop on a fixed stack (no allocation), so it is fast
// enough to run inside the distance field.
#pragma once
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace geo {

struct Expr {
    enum Op { NUM, VAR, ADD, SUB, MUL, DIV, POW, NEG, F1, F2 };
    struct Ins { Op op; double v; int i; };
    std::vector<Ins> prog;
    std::string src;

    static int f1_index(const std::string& n) {
        static const char* f[] = {"sin", "cos", "tan", "asin", "acos", "atan", "exp", "log", "sqrt", "abs", "floor", "sign"};
        for (int i = 0; i < 12; ++i) if (n == f[i]) return i;
        return -1;
    }
    static int f2_index(const std::string& n) {
        static const char* f[] = {"min", "max", "pow", "atan2", "mod"};
        for (int i = 0; i < 5; ++i) if (n == f[i]) return i;
        return -1;
    }

    // ---------------------------------------------------------------- parser (recursive descent)
    struct Parser {
        const std::string& s; size_t p = 0; std::vector<Ins>& out;
        Parser(const std::string& s_, std::vector<Ins>& o) : s(s_), out(o) {}
        void ws() { while (p < s.size() && std::isspace((unsigned char)s[p])) ++p; }
        bool eat(char c) { ws(); if (p < s.size() && s[p] == c) { ++p; return true; } return false; }
        [[noreturn]] void fail(const std::string& m) { throw std::runtime_error("expression: " + m + " at position " + std::to_string(p)); }
        void expr() { term(); for (;;) { if (eat('+')) { term(); out.push_back({ADD, 0, 0}); } else if (eat('-')) { term(); out.push_back({SUB, 0, 0}); } else break; } }
        void term() { unary(); for (;;) { if (eat('*')) { unary(); out.push_back({MUL, 0, 0}); } else if (eat('/')) { unary(); out.push_back({DIV, 0, 0}); } else break; } }
        void unary() { if (eat('-')) { unary(); out.push_back({NEG, 0, 0}); } else if (eat('+')) unary(); else power(); }
        void power() { atom(); if (eat('^')) { unary(); out.push_back({POW, 0, 0}); } }
        void atom() {
            ws();
            if (p >= s.size()) fail("unexpected end");
            char c = s[p];
            if (std::isdigit((unsigned char)c) || c == '.') {
                size_t q = p; while (q < s.size() && (std::isdigit((unsigned char)s[q]) || s[q] == '.' || ((s[q] == 'e' || s[q] == 'E') && q + 1 < s.size() && (std::isdigit((unsigned char)s[q + 1]) || s[q + 1] == '-' || s[q + 1] == '+')) || ((s[q] == '-' || s[q] == '+') && q > p && (s[q - 1] == 'e' || s[q - 1] == 'E')))) ++q;
                out.push_back({NUM, std::atof(s.substr(p, q - p).c_str()), 0}); p = q; return;
            }
            if (std::isalpha((unsigned char)c) || c == '_') {
                size_t q = p; while (q < s.size() && (std::isalnum((unsigned char)s[q]) || s[q] == '_')) ++q;
                std::string n = s.substr(p, q - p); p = q;
                if (n == "x") { out.push_back({VAR, 0, 0}); return; }
                if (n == "y") { out.push_back({VAR, 0, 1}); return; }
                if (n == "z") { out.push_back({VAR, 0, 2}); return; }
                if (n == "t") { out.push_back({VAR, 0, 3}); return; }
                if (n == "pi") { out.push_back({NUM, 3.14159265358979323846, 0}); return; }
                if (n == "e") { out.push_back({NUM, 2.71828182845904523536, 0}); return; }
                int k1 = f1_index(n), k2 = f2_index(n);
                if (k1 < 0 && k2 < 0) fail("unknown name '" + n + "'");
                if (!eat('(')) fail("'(' expected after " + n);
                expr();
                if (k2 >= 0) { if (!eat(',')) fail("',' expected in " + n); expr(); if (!eat(')')) fail("')' expected"); out.push_back({F2, 0, k2}); return; }
                if (!eat(')')) fail("')' expected");
                out.push_back({F1, 0, k1}); return;
            }
            if (eat('(')) { expr(); if (!eat(')')) fail("')' expected"); return; }
            fail(std::string("unexpected '") + c + "'");
        }
    };

    void compile(const std::string& text) {
        src = text; prog.clear();
        Parser ps(text, prog);
        ps.expr(); ps.ws();
        if (ps.p != text.size()) ps.fail("unexpected text");
        if (prog.size() > 4000) throw std::runtime_error("expression: too long");
    }
    bool empty() const { return prog.empty(); }

    double eval(double x, double y, double z, double t = 0) const {
        double st[256]; int sp = 0;
        const double v[4] = {x, y, z, t};
        for (const Ins& in : prog) {
            switch (in.op) {
                case NUM: st[sp++] = in.v; break;
                case VAR: st[sp++] = v[in.i]; break;
                case ADD: --sp; st[sp - 1] += st[sp]; break;
                case SUB: --sp; st[sp - 1] -= st[sp]; break;
                case MUL: --sp; st[sp - 1] *= st[sp]; break;
                case DIV: --sp; st[sp - 1] /= st[sp]; break;
                case POW: --sp; st[sp - 1] = std::pow(st[sp - 1], st[sp]); break;
                case NEG: st[sp - 1] = -st[sp - 1]; break;
                case F1: {
                    double& a = st[sp - 1];
                    switch (in.i) {
                        case 0: a = std::sin(a); break; case 1: a = std::cos(a); break; case 2: a = std::tan(a); break;
                        case 3: a = std::asin(a); break; case 4: a = std::acos(a); break; case 5: a = std::atan(a); break;
                        case 6: a = std::exp(a); break; case 7: a = std::log(a); break; case 8: a = std::sqrt(a); break;
                        case 9: a = std::fabs(a); break; case 10: a = std::floor(a); break; default: a = (a > 0) - (a < 0); break;
                    }
                    break;
                }
                case F2: {
                    --sp; double a = st[sp - 1], b = st[sp];
                    switch (in.i) {
                        case 0: st[sp - 1] = a < b ? a : b; break; case 1: st[sp - 1] = a > b ? a : b; break;
                        case 2: st[sp - 1] = std::pow(a, b); break; case 3: st[sp - 1] = std::atan2(a, b); break;
                        default: st[sp - 1] = b != 0 ? a - b * std::floor(a / b) : 0; break;
                    }
                    break;
                }
            }
            if (sp >= 250) return 0;
        }
        return sp ? st[0] : 0;
    }
};

}  // namespace geo
