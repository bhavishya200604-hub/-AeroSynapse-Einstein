// The look of the 3D view: a soft shadow on an invisible floor under the model (several jittered planar projections,
// each pass written once per pixel through the stencil buffer, so overlapping triangles never darken it twice), and
// the name of the standard view in the corner, as in other CAD programs. G toggles the shadow.

static bool g_shadow_on = true;

static void shadow_geometry() {  // the model's triangles, in world coordinates under the current matrix
    auto draw = [](const geo::Mesh& M) {
        if (M.t.empty()) return;
        static std::vector<float> vb;
        vb.resize(M.v.size() * 3);
        for (size_t i = 0; i < M.v.size(); ++i) { vb[3 * i] = (float)M.v[i].x; vb[3 * i + 1] = (float)M.v[i].y; vb[3 * i + 2] = (float)M.v[i].z; }
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, vb.data());
        glDrawElements(GL_TRIANGLES, (GLsizei)M.t.size(), GL_UNSIGNED_INT, M.t.data());
        glDisableClientState(GL_VERTEX_ARRAY);
    };
    if (g_asm_on) {
        for (size_t c = 0; c < g_ac.size(); ++c) {
            const AComp& C = g_ac[c];
            if (C.hidden) continue;
            glPushMatrix(); gl_pose(C.pose, explode_offset((int)c));
            for (const APiece& pc : C.pieces) { if (!pc.mesh) continue; glPushMatrix(); gl_pose(pc.local, V3()); draw(pc.mesh->m); glPopMatrix(); }
            glPopMatrix();
        }
    } else draw(g_mesh);
}
static void shadow_render() {
    if (!g_shadow_on || g_cut) return;
    geo::Box b;
    if (g_asm_on) { for (const AComp& c : g_ac) comp_world_box(c, b); } else if (!g_mesh.v.empty()) { b.add(g_props.lo); b.add(g_props.hi); }
    if (!b.valid()) return;
    double S = geo::len(b.hi - b.lo), zf = b.lo.z - 0.04 * S;
    if (g_env.on && fluid_props(g_env.medium, g_env.Tc, g_env.alt).liquid) return;  // under a liquid: no floor shadow
    GLint sb = 0; glGetIntegerv(GL_STENCIL_BITS, &sb);
    glDisable(GL_LIGHTING); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    if (sb > 0) glEnable(GL_STENCIL_TEST);
    const int NP = sb > 0 ? 6 : 1;
    for (int k = 0; k < NP; ++k) {  // a light above and a little behind the viewer, jittered for a soft edge
        double yr = g_yaw * geo::PI / 180;
        V3 L(0.25 * std::cos(yr) + 0.07 * std::cos(k * 1.05), 0.25 * std::sin(yr) + 0.07 * std::sin(k * 1.05), 1.0);
        double m[16] = {1, 0, 0, 0, 0, 1, 0, 0, -L.x / L.z, -L.y / L.z, 0, 0, L.x / L.z * zf, L.y / L.z * zf, zf, 1};
        if (sb > 0) { glClear(GL_STENCIL_BUFFER_BIT); glStencilFunc(GL_EQUAL, 0, 0xFF); glStencilOp(GL_KEEP, GL_KEEP, GL_INCR); }
        glColor4f(0.08f, 0.1f, 0.14f, sb > 0 ? 0.075f : 0.25f);
        glPushMatrix(); glMultMatrixd(m);
        shadow_geometry();
        glPopMatrix();
    }
    glDisable(GL_STENCIL_TEST);
    glDepthMask(GL_TRUE); glDisable(GL_BLEND); glEnable(GL_LIGHTING);
}
static std::string view_name() {  // the standard view the camera is in, starred once it has been turned away from it
    struct V { double y, p; const char* n; };
    static const V Vs[] = {{-135, 25, "Isometric"}, {180, 0, "Front"}, {0, 0, "Back"}, {-90, 0, "Right"}, {90, 0, "Left"}, {-90, 89, "Top"}, {-90, -89, "Bottom"}, {-120, 30, "Trimetric"}, {-150, 20, "Dimetric"}};
    for (const V& v : Vs) {
        double dy = std::fabs(std::remainder(g_yaw - v.y, 360.0)), dp = std::fabs(g_pitch - v.p);
        if (dy < 0.5 && dp < 0.5) return v.n;
    }
    double best = 1e9; const char* n = "Isometric";
    for (const V& v : Vs) { double d = std::fabs(std::remainder(g_yaw - v.y, 360.0)) + std::fabs(g_pitch - v.p); if (d < best) { best = d; n = v.n; } }
    return std::string("*") + n;
}
static void view_label(int H) {
    (void)H;
    glColor3f(0.18f, 0.2f, 0.25f);
    gl_text(16, 96, view_name());  // above the axis triad, clear of the motion timeline
}
