// Engineering materials for the environment analysis (envphys.inc) and the look of parts: density, conductivity,
// heat capacity, thermal expansion, emissivity, service temperature, anodic index (MIL-STD-889 style galvanic series,
// volts; -1 for non-conductors), a dry self-friction coefficient, and the colour the part is drawn in. Room-temperature
// handbook values (ASM, Incropera); good to a few per cent for the alloys, rougher for polymers, composites and wood.
//   cls: 0 metal (drawn glossy), 1 polymer or elastomer (matte), 2 composite, ceramic, glass or wood (satin)
struct EnvMat { const char* key; const char* label; double rho, k, cp, alpha, eps, tmax, anodic, mu; int cls; float r, g, b; };
static const EnvMat ENVMATS[] = {
    {"al6061", "Aluminium 6061", 2700, 167, 896, 23.6e-6, 0.09, 150, 0.90, 1.05, 0, 0.70f, 0.72f, 0.75f},
    {"al7075", "Aluminium 7075", 2810, 130, 960, 23.4e-6, 0.09, 120, 0.90, 1.05, 0, 0.66f, 0.68f, 0.72f},
    {"al2024", "Aluminium 2024", 2780, 121, 875, 23.2e-6, 0.09, 120, 0.90, 1.05, 0, 0.64f, 0.67f, 0.70f},
    {"al_li", "Aluminium-lithium", 2590, 88, 960, 22.0e-6, 0.09, 120, 0.90, 1.05, 0, 0.68f, 0.70f, 0.76f},
    {"steel_mild", "Carbon steel (S235)", 7850, 50, 486, 12.0e-6, 0.79, 400, 0.85, 0.60, 0, 0.47f, 0.50f, 0.55f},
    {"steel_300m", "Steel 300M (bearing steel)", 7830, 32, 470, 11.4e-6, 0.60, 300, 0.85, 0.60, 0, 0.55f, 0.57f, 0.61f},
    {"stainless_316", "Stainless steel 316", 8000, 16.3, 500, 16.0e-6, 0.30, 800, 0.50, 0.55, 0, 0.62f, 0.64f, 0.67f},
    {"cast_iron", "Grey cast iron", 7200, 52, 460, 10.5e-6, 0.80, 350, 0.85, 0.40, 0, 0.34f, 0.35f, 0.37f},
    {"brass", "Brass", 8500, 109, 380, 19.0e-6, 0.20, 200, 0.40, 0.50, 0, 0.80f, 0.64f, 0.28f},
    {"bronze", "Phosphor bronze", 8800, 70, 380, 17.8e-6, 0.20, 250, 0.35, 0.35, 0, 0.60f, 0.46f, 0.28f},
    {"copper", "Copper", 8960, 401, 385, 16.5e-6, 0.10, 200, 0.35, 1.00, 0, 0.76f, 0.42f, 0.28f},
    {"ti64", "Titanium Ti-6Al-4V", 4430, 6.7, 526, 8.6e-6, 0.30, 400, 0.15, 0.40, 0, 0.58f, 0.57f, 0.60f},
    {"ti6242", "Titanium Ti-6242", 4540, 7.1, 460, 7.7e-6, 0.30, 540, 0.15, 0.40, 0, 0.55f, 0.55f, 0.58f},
    {"in718", "Inconel 718", 8190, 11.4, 435, 13.0e-6, 0.40, 700, 0.20, 0.50, 0, 0.63f, 0.62f, 0.57f},
    {"cfrp_im7", "Carbon/epoxy IM7", 1580, 5.0, 900, 2.0e-6, 0.85, 120, 0.00, 0.30, 2, 0.17f, 0.18f, 0.20f},
    {"cfrp_bmi", "Carbon/BMI", 1600, 5.0, 900, 2.0e-6, 0.85, 200, 0.00, 0.30, 2, 0.20f, 0.20f, 0.22f},
    {"glass_epoxy", "Glass/epoxy", 1900, 0.30, 1000, 12.0e-6, 0.90, 120, -1, 0.30, 2, 0.70f, 0.80f, 0.55f},
    {"cmc_csic", "Ceramic matrix C/SiC", 2100, 15, 750, 3.0e-6, 0.85, 1500, -1, 0.40, 2, 0.30f, 0.30f, 0.32f},
    {"abs", "ABS plastic", 1050, 0.17, 1300, 90e-6, 0.90, 80, -1, 0.35, 1, 0.92f, 0.91f, 0.86f},
    {"nylon", "Nylon PA66", 1140, 0.25, 1700, 80e-6, 0.90, 100, -1, 0.25, 1, 0.94f, 0.92f, 0.82f},
    {"ptfe", "PTFE (Teflon)", 2200, 0.25, 1000, 120e-6, 0.90, 260, -1, 0.08, 1, 0.90f, 0.90f, 0.90f},
    {"rubber", "Rubber (EPDM)", 1100, 0.20, 2000, 160e-6, 0.90, 120, -1, 0.90, 1, 0.13f, 0.13f, 0.14f},
    {"oak", "Wood (oak)", 700, 0.17, 2000, 5.0e-6, 0.90, 120, -1, 0.40, 2, 0.64f, 0.46f, 0.28f},
    {"glass_soda", "Glass (soda-lime)", 2500, 1.0, 840, 9.0e-6, 0.92, 300, -1, 0.40, 2, 0.62f, 0.78f, 0.86f},
};
static const int NENVMAT = (int)(sizeof(ENVMATS) / sizeof(ENVMATS[0]));
static const EnvMat* envmat_of(const std::string& key) {
    for (int i = 0; i < NENVMAT; ++i) if (key == ENVMATS[i].key) return &ENVMATS[i];
    return nullptr;
}
// the material of a part design: a component's choice, the part's own, its features', or the usual one for its kind
static std::string part_material_key(const js::Value& part, const std::string& over) {
    if (!over.empty() && envmat_of(over)) return over;
    std::string m = part.str("material", "");
    if (envmat_of(m)) return m;
    if (const js::Value* fl = part.find("features"))
        for (size_t i = 0; i < fl->size(); ++i) { std::string mm = fl->at(i).str("material", ""); if (envmat_of(mm)) return mm; }
    if (part.find("fastener") || part.find("gear") || part.find("sprocket")) return "steel_mild";
    return "";
}
