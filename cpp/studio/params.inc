// The parameter help of every feature property (unit, meaning, choices) and the user preferences shared by the panels.
static std::string g_pref_plot = "plain", g_pref_display = "calm", g_pref_palette = "fixed";
static bool g_calm = true, g_autohide = false;

// ---------------------------------------------------------------- what every feature parameter is
struct ParamHelp { const char* key; const char* unit; const char* what; const char* choices; };
static const ParamHelp PARAM_HELP[] = {
    {"id", "", "Name of the feature, used by rules, pivots, reports and the feature tree", nullptr},
    {"type", "", "Kind of feature (fixed: insert another tool for another kind)", nullptr},
    {"op", "", "add joins it to the body, cut removes its volume, intersect keeps only the overlap", "add,cut,intersect"},
    {"enabled", "", "Off hides the feature without deleting it", "true,false"},
    {"mirror", "", "Copy across the y = 0 plane (one wing makes both)", "true,false"},
    {"smooth", "", "Spline through the points instead of straight segments", "true,false"},
    {"reserve", "", "A space reservation: checked by the clash analysis, not built", "true,false"},
    {"translate", "m", "Position offset x, y, z", nullptr},
    {"rotate", "deg", "Rotation about x, then y, then z", nullptr},
    {"scale", "", "Scale factor: one number, or x, y, z", nullptr},
    {"pivot", "m", "Point that rotation and scale are taken about (folding wings, variable sweep)", nullptr},
    {"stretch", "", "Stretch factors along x, y, z", nullptr},
    {"shear", "", "Shear coefficients", nullptr},
    {"bend", "deg", "Exact circular bend of the local x axis towards +y", nullptr},
    {"twist_rate", "deg/m", "Twist about the local x axis per metre", nullptr},
    {"taper_rate", "1/m", "Cross-section scale 1 + rate * x along x", nullptr},
    {"blend", "m", "Fillet radius where this feature meets the ones before it", nullptr},
    {"chamfer", "m", "45-degree bevel where it meets the others (0: a sharp edge)", nullptr},
    {"draft", "deg", "Wall lean of an extrude (moulding draft)", nullptr},
    {"shell", "m", "Hollow it with this wall thickness (0: solid)", nullptr},
    {"infill", "", "Lattice inside a hollow feature: {type, cell, wall}", nullptr},
    {"pattern", "", "Copies: {count, step [dx, dy, dz], rotate (deg about x)}", nullptr},
    {"span", "m", "Span of one side, root to tip (the mirror makes the other side)", nullptr},
    {"root_chord", "m", "Chord at the root", nullptr},
    {"taper", "", "Tip chord / root chord", nullptr},
    {"sweep", "deg", "Leading-edge sweep", nullptr},
    {"dihedral", "deg", "Dihedral (tips up positive; 90 is a vertical fin)", nullptr},
    {"twist", "deg", "Tip twist relative to the root (washout negative)", nullptr},
    {"tc", "", "Thickness / chord of the root section", nullptr},
    {"tip_tc", "", "Thickness / chord of the tip section", nullptr},
    {"camber", "", "Maximum camber / chord", nullptr},
    {"camber_pos", "x/c", "Chordwise position of the maximum camber", nullptr},
    {"section", "", "Section table {u, t, zc} (from an airfoil file)", nullptr},
    {"section_source", "", "Airfoil file the section came from", nullptr},
    {"stations", "m", "Body stations [x, half-width, half-height, centre height, shape n (2 round, 4 boxy)]", nullptr},
    {"r", "m", "Radius (torus: the tube radius)", nullptr},
    {"R", "m", "Torus: radius of the ring", nullptr},
    {"r1", "m", "Radius at the start", nullptr},
    {"r2", "m", "Radius at the end", nullptr},
    {"radius", "m", "Tube radius (one per path point for a duct)", nullptr},
    {"radii", "m", "Semi-axes x, y, z", nullptr},
    {"size", "m", "Size x, y, z", nullptr},
    {"round", "m", "Edge rounding radius", nullptr},
    {"length", "m", "Length", nullptr},
    {"height", "m", "Height or extrude depth (bent plate: its width)", nullptr},
    {"axis", "", "Axis direction", "x,y,z"},
    {"a", "m", "Capsule: start point x, y, z", nullptr},
    {"b", "m", "Capsule: end point x, y, z", nullptr},
    {"outline", "m", "Closed 2D outline [[u, v], ...] in the sketch plane", nullptr},
    {"profile", "", "Revolve: profile [[x, r], ...]; bent plate: [[x, z], ...]; wing: section shape", "naca,biconvex,diamond,table"},
    {"path", "m", "Path points [[x, y, z], ...]", nullptr},
    {"control", "m", "Bezier net of 16 points [[x, y, z], ...] (4 x 4)", nullptr},
    {"sections", "", "Loft sections [{x, points [[y, z], ...]}, ...] with equal point counts", nullptr},
    {"width", "m", "Plate width across the bends", nullptr},
    {"thickness", "m", "Plate or sheet thickness", nullptr},
    {"bend_radius", "m", "Inner bend radius of the sheet", nullptr},
    {"expr", "", "Equation: f(x, y, z) <= 0 is solid. Use x y z, + - * / ^, sin cos tan exp log sqrt abs min max", nullptr},
    {"n1", "", "Superellipsoid exponent north-south (1 ellipsoid, small boxy, 2 diamond)", nullptr},
    {"n2", "", "Superellipsoid exponent around the axis", nullptr},
    {"x", "m", "Curve x(t) as an expression of t", nullptr},
    {"y", "m", "Curve y(t) as an expression of t", nullptr},
    {"z", "m", "Curve z(t) as an expression of t", nullptr},
    {"t0", "", "Start of the curve parameter t", nullptr},
    {"t1", "", "End of the curve parameter t", nullptr},
    {"samples", "", "Points along the curve", nullptr},
    {"cell", "m", "Lattice cell size", nullptr},
    {"wall", "m", "Lattice wall thickness", nullptr},
    {"kind", "", "Lattice type", "gyroid,schwarz,diamond,struts"},
    {"normal", "", "Normal of the plane", nullptr},
    {"offset", "m", "Plane offset along its normal", nullptr},
};
static const ParamHelp* param_help(const std::string& k) {
    for (const ParamHelp& p : PARAM_HELP) if (k == p.key) return &p;
    return nullptr;
}
static std::vector<std::string> split_csv(const char* s) {
    std::vector<std::string> v; if (!s) return v;
    std::string cur;
    for (const char* p = s;; ++p) { if (*p == ',' || !*p) { v.push_back(cur); cur.clear(); if (!*p) break; } else cur += *p; }
    return v;
}
static std::string g_param_key;  // the property row with the keyboard focus (shown on the info card)

