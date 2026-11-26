// Saving a picture in any format, by the file's extension: PNG, JPEG (quality 95), TIFF, BMP and GIF through GDI+'s
// encoders, and PDF: the picture (JPEG-compressed, quality 95) on a page of its own size at 300 dots per inch.

static bool gdip_encoder(const wchar_t* mime, CLSID& id) {
    UINT n = 0, sz = 0; gp::GetImageEncodersSize(&n, &sz);
    if (!sz) return false;
    std::vector<char> buf(sz); gp::ImageCodecInfo* ci = (gp::ImageCodecInfo*)buf.data();
    gp::GetImageEncoders(n, sz, ci);
    for (UINT i = 0; i < n; ++i) if (wcscmp(ci[i].MimeType, mime) == 0) { id = ci[i].Clsid; return true; }
    return false;
}
static bool jpeg_bytes(gp::Bitmap* bm, std::string& out, ULONG quality = 95) {
    CLSID id; if (!gdip_encoder(L"image/jpeg", id)) return false;
    gp::EncoderParameters ep; ep.Count = 1; ep.Parameter[0].Guid = gp::EncoderQuality; ep.Parameter[0].Type = gp::EncoderParameterValueTypeLong; ep.Parameter[0].NumberOfValues = 1; ep.Parameter[0].Value = &quality;
    IStream* st = nullptr; if (CreateStreamOnHGlobal(nullptr, TRUE, &st) != S_OK) return false;
    bool ok = bm->Save(st, &id, &ep) == gp::Ok;
    if (ok) {
        STATSTG ss; st->Stat(&ss, STATFLAG_NONAME);
        LARGE_INTEGER z; z.QuadPart = 0; st->Seek(z, STREAM_SEEK_SET, nullptr);
        out.resize((size_t)ss.cbSize.QuadPart); ULONG got = 0; st->Read(&out[0], (ULONG)out.size(), &got); out.resize(got);
    }
    st->Release();
    return ok;
}
static bool pdf_with_image(gp::Bitmap* bm, const std::string& path) {
    std::string jpg; if (!jpeg_bytes(bm, jpg)) return false;
    int W = (int)bm->GetWidth(), H = (int)bm->GetHeight();
    double pw = W * 72.0 / 300, ph = H * 72.0 / 300;  // 300 dots per inch
    std::string pdf = "%PDF-1.4\n%\xE2\xE3\xCF\xD3\n";
    std::vector<size_t> off;
    char b[512];
    auto obj = [&](const std::string& body) { off.push_back(pdf.size()); pdf += tostr((int)off.size()) + " 0 obj\n" + body + "\nendobj\n"; };
    obj("<< /Type /Catalog /Pages 2 0 R >>");
    obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    snprintf(b, sizeof b, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>", pw, ph); obj(b);
    snprintf(b, sizeof b, "<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n", W, H, (int)jpg.size());
    obj(std::string(b) + jpg + "\nendstream");
    snprintf(b, sizeof b, "q %.2f 0 0 %.2f 0 0 cm /Im0 Do Q", pw, ph);
    std::string content = b;
    obj("<< /Length " + tostr((int)content.size()) + " >>\nstream\n" + content + "\nendstream");
    size_t xref = pdf.size();
    pdf += "xref\n0 " + tostr((int)off.size() + 1) + "\n0000000000 65535 f \n";
    for (size_t o : off) { snprintf(b, sizeof b, "%010u 00000 n \n", (unsigned)o); pdf += b; }
    pdf += "trailer\n<< /Size " + tostr((int)off.size() + 1) + " /Root 1 0 R >>\nstartxref\n" + tostr((long)xref) + "\n%%EOF\n";
    FILE* f = fopen(path.c_str(), "wb"); if (!f) return false;
    bool ok = fwrite(pdf.data(), 1, pdf.size(), f) == pdf.size(); fclose(f);
    return ok;
}
static bool save_image_as(gp::Bitmap* bm, const std::string& path) {
    std::string ext = path.substr(path.find_last_of('.') + 1);
    for (char& c : ext) c = (char)tolower((unsigned char)c);
    if (ext == "pdf") return pdf_with_image(bm, path);
    const wchar_t* mime = ext == "jpg" || ext == "jpeg" ? L"image/jpeg" : ext == "tif" || ext == "tiff" ? L"image/tiff" : ext == "bmp" ? L"image/bmp" : ext == "gif" ? L"image/gif" : L"image/png";
    CLSID id; if (!gdip_encoder(mime, id)) return false;
    if (wcscmp(mime, L"image/jpeg") == 0) {
        ULONG q = 95; gp::EncoderParameters ep; ep.Count = 1; ep.Parameter[0].Guid = gp::EncoderQuality; ep.Parameter[0].Type = gp::EncoderParameterValueTypeLong; ep.Parameter[0].NumberOfValues = 1; ep.Parameter[0].Value = &q;
        return bm->Save(wide(path).c_str(), &id, &ep) == gp::Ok;
    }
    return bm->Save(wide(path).c_str(), &id, nullptr) == gp::Ok;
}
static const char* IMG_FILTER = "PNG image (*.png)\0*.png\0JPEG image (*.jpg)\0*.jpg\0TIFF image (*.tif)\0*.tif\0Bitmap (*.bmp)\0*.bmp\0PDF document (*.pdf)\0*.pdf\0";
