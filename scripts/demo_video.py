"""Record a demonstration video of AeroSynapse Studio: a scripted session that uses the program's real features.

The script starts Studio, operates it with window messages (workbench tabs, tool tiles, pads, setup forms, mouse
orbit and zoom, picking on the model), captures every visible Studio window (main window, pads, forms, result
viewers and message boxes, composited in place) with PrintWindow, and encodes an H.264 MP4 with a caption bar.
Long computations run for real; the recording pauses while they run and says so (time-lapse).

    python scripts/demo_video.py [--out docs/demo/aerosynapse_demo.mp4] [--fps 10] [--only chapter,...]
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as W
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "bin" / "AeroSynapseStudio.exe"
STUDIO = ROOT / "cpp" / "studio"
user32, gdi32, kernel32 = ctypes.windll.user32, ctypes.windll.gdi32, ctypes.windll.kernel32
# Studio is not DPI-aware, so this script is not either: positions and captures share Studio's (scaled) coordinates

WM_SETTEXT, WM_COMMAND, WM_CLOSE, WM_KEYDOWN, WM_KEYUP, WM_CHAR = 0x000C, 0x0111, 0x0010, 0x0100, 0x0101, 0x0102
WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_RBUTTONDOWN, WM_RBUTTONUP, WM_MOUSEWHEEL = 0x0200, 0x0201, 0x0202, 0x0204, 0x0205, 0x020A
CB_SELECTSTRING, CB_GETCURSEL, BM_SETCHECK, BM_CLICK = 0x014D, 0x0147, 0x00F1, 0x00F5
CBN_SELCHANGE, BN_CLICKED = 1, 0
WM_APP = 0x8000
WM_TOOL_INSERT = WM_APP + 3


# ---------------------------------------------------------------- ids and tools from the sources
def command_ids() -> dict[str, int]:
    ids: dict[str, int] = {}
    for f in [STUDIO / "studio.cpp", *STUDIO.glob("*.inc")]:
        s = f.read_text(encoding="utf-8")
        for m in re.finditer(r"enum\s*\{([^}]*)\}", s):
            v = -1
            for part in m.group(1).split(","):
                part = re.sub(r"//.*", "", part).strip()
                if not part:
                    continue
                mm = re.match(r"(\w+)\s*(?:=\s*(.+))?$", part, re.S)
                if not mm:
                    continue
                name, val = mm.group(1), mm.group(2)
                if val is not None:
                    val = val.strip()
                    try:
                        v = int(eval(val.replace("WM_APP", str(WM_APP)), {}, dict(ids)))  # noqa: S307 (our own enum text)
                    except Exception:
                        continue
                else:
                    v += 1
                ids[name] = v
    return ids


def tool_table() -> list[tuple[str, str, str, str]]:
    s = (STUDIO / "tools.inc").read_text(encoding="utf-8")
    return re.findall(r'\{"(\w+)", "(\w+)", "([^"]+)", "(?:[^"\\]|\\.)*", (\w+(?:\s*\+\s*\d+)?), (?:nullptr|"\w+")\}', s)


IDS = command_ids()
TOOLS = tool_table()  # scheme, icon, label, command
SCHEMES = re.findall(r'\{"(\w+)", "[^"]+", \d+, \d+, \d+, "\w+"\}', (STUDIO / "tools.inc").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- Win32 helpers
EnumProc = ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)


def windows_of(pid: int, children_of: int | None = None) -> list[int]:
    out: list[int] = []

    def cb(h, _):
        p = W.DWORD()
        user32.GetWindowThreadProcessId(h, ctypes.byref(p))
        if p.value == pid:
            out.append(h)
        return True
    if children_of:
        user32.EnumChildWindows(children_of, EnumProc(cb), 0)
    else:
        user32.EnumWindows(EnumProc(cb), 0)
    return out


def cls(h: int) -> str:
    b = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(h, b, 256)
    return b.value


def text(h: int) -> str:
    b = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(h, b, 512)
    return b.value


def rect(h: int) -> tuple[int, int, int, int]:
    r = W.RECT()
    user32.GetWindowRect(h, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", W.DWORD), ("biWidth", W.LONG), ("biHeight", W.LONG), ("biPlanes", W.WORD), ("biBitCount", W.WORD), ("biCompression", W.DWORD),
                ("biSizeImage", W.DWORD), ("biXPelsPerMeter", W.LONG), ("biYPelsPerMeter", W.LONG), ("biClrUsed", W.DWORD), ("biClrImportant", W.DWORD)]


def grab(h: int) -> Image.Image | None:
    l, t, r, b = rect(h)
    w, hh = r - l, b - t
    if w < 8 or hh < 8:
        return None
    hdc = user32.GetWindowDC(h)
    mdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, hh)
    old = gdi32.SelectObject(mdc, bmp)
    user32.PrintWindow(h, mdc, 2)  # PW_RENDERFULLCONTENT: the window's own content, never the screen
    bi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -hh, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * hh * 4)
    gdi32.GetDIBits(mdc, bmp, 0, hh, buf, ctypes.byref(bi), 0)
    gdi32.SelectObject(mdc, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mdc)
    user32.ReleaseDC(h, hdc)
    return Image.frombuffer("RGB", (w, hh), buf, "raw", "BGRX", 0, 1)


def lp(x: int, y: int) -> int:
    return ((y & 0xFFFF) << 16) | (x & 0xFFFF)


# ---------------------------------------------------------------- recorder
class Recorder:
    def __init__(self, out: Path, fps: int, size: tuple[int, int]):
        import imageio.v2 as iio

        self.fps, self.size = fps, size
        self.writer = iio.get_writer(str(out), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8)
        self.chapter, self.caption, self.note = "", "", ""
        self.pid = 0
        self.main = 0
        self.paused = True
        self.stop = False
        self.frames = 0
        self.lock = threading.Lock()
        try:
            self.f_big = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 22)
            self.f_txt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 19)
            self.f_small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
        except OSError:
            self.f_big = self.f_txt = self.f_small = ImageFont.load_default()
        self.last = Image.new("RGB", size, (12, 22, 40))
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def compose(self) -> Image.Image:
        if not self.main:
            return self.last
        if not user32.IsWindow(self.main):
            raise RuntimeError("Studio's window is gone (the program closed)")
        base = grab(self.main)
        if base is None:
            raise RuntimeError(f"main window not capturable, rect {rect(self.main)}, iconic {user32.IsIconic(self.main)}")
        ml, mt, mr, mb = rect(self.main)
        tops = [h for h in windows_of(self.pid) if h != self.main and user32.IsWindowVisible(h) and cls(h) not in ("IME", "MSCTFIME UI", "AeroSynapseSplash")]
        for h in tops:  # bring windows opened elsewhere on the screen over the main window, so they are in the picture
            if cls(h) in ("AeroSynapseTip", "ComboLBox", "tooltips_class32"):
                continue
            l, t, r, b = rect(h)
            if cls(h) == "AeroSynapsePad" and l < mr - (r - l) - 40:  # pads go to the right, over the properties, so the model stays in view
                user32.MoveWindow(h, mr - (r - l) - 12, mt + 118, r - l, min(b - t, mb - mt - 180), True)
                continue
            if l < ml or t < mt or r > mr or b > mb - 40:
                w, hh = min(r - l, mr - ml), min(b - t, mb - mt - 60)
                nx = ml + (mr - ml - w) // 2 + (140 if cls(h) == "AeroSynapsePad" else 0)
                ny = mt + max(60, (mb - mt - hh) // 2)
                user32.MoveWindow(h, nx, ny, w, hh, True)
        for h in reversed(tops):  # bottom first
            im = grab(h)
            if im is None:
                continue
            l, t, _, _ = rect(h)
            base.paste(im, (l - ml, t - mt))
        frame = Image.new("RGB", self.size, (12, 22, 40))
        frame.paste(base.crop((0, 0, min(base.width, self.size[0]), min(base.height, self.size[1] - 56))), (0, 0))
        self.overlay(frame)
        return frame

    def overlay(self, f: Image.Image) -> None:
        d = ImageDraw.Draw(f)
        w, h = f.size
        d.rectangle([0, h - 56, w, h], fill=(14, 26, 46))
        d.rectangle([0, h - 56, 6, h], fill=(56, 189, 248))
        d.text((18, h - 50), self.chapter, font=self.f_big, fill=(125, 211, 252))
        cw = d.textlength(self.chapter, font=self.f_big) if self.chapter else 0
        d.text((30 + cw, h - 47), self.caption, font=self.f_txt, fill=(235, 242, 250))
        if self.note:
            tw = d.textlength(self.note, font=self.f_small)
            d.text((w - tw - 16, h - 22), self.note, font=self.f_small, fill=(148, 163, 184))

    def loop(self) -> None:
        period = 1.0 / self.fps
        nxt = time.perf_counter()
        while not self.stop:
            if self.paused:
                time.sleep(0.05)
                nxt = time.perf_counter()
                continue
            try:
                fr = self.compose()
            except Exception as e:  # keep the last frame, and say why
                msg = f"{type(e).__name__}: {e}"
                if msg != getattr(self, "_last_err", ""):
                    self._last_err = msg
                    print("capture failed:", msg, "| main:", self.main, "visible", user32.IsWindowVisible(self.main), "iconic", user32.IsIconic(self.main), flush=True)
                fr = self.last
            self.last = fr
            arr = np.asarray(fr)
            nxt += period
            n = 1
            while time.perf_counter() > nxt + period:  # a slow capture: repeat the frame so time stays true
                nxt += period
                n += 1
            with self.lock:
                for _ in range(n):
                    self.writer.append_data(arr)
                    self.frames += 1
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def card(self, title: str, lines: list[str], seconds: float) -> None:  # a title card between chapters
        f = Image.new("RGB", self.size, (10, 20, 38))
        d = ImageDraw.Draw(f)
        w, h = self.size
        try:
            ft = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 46)
        except OSError:
            ft = self.f_big
        d.text((90, h // 2 - 120), title, font=ft, fill=(240, 246, 252))
        d.rectangle([90, h // 2 - 52, 290, h // 2 - 48], fill=(56, 189, 248))
        for i, ln in enumerate(lines):
            d.text((92, h // 2 - 20 + 34 * i), ln, font=self.f_txt, fill=(186, 200, 220))
        arr = np.asarray(f)
        with self.lock:
            for _ in range(int(seconds * self.fps)):
                self.writer.append_data(arr)
                self.frames += 1

    def close(self) -> None:
        self.stop = True
        self.t.join(timeout=5)
        self.writer.close()


# ---------------------------------------------------------------- driving Studio
class Studio:
    def __init__(self, rec: Recorder):
        self.rec = rec
        self.proc: subprocess.Popen | None = None
        self.main = 0

    def start(self, *args: str, wait: float = 9.0, size: tuple[int, int] = (1440, 900)) -> None:
        self.quit()
        self.proc = subprocess.Popen([str(EXE), *args], cwd=str(ROOT))
        t0 = time.time()
        while time.time() - t0 < 30:
            hs = [h for h in windows_of(self.proc.pid) if cls(h) == "AeroSynapseStudio" and user32.IsWindowVisible(h)]
            if hs:
                self.main = hs[0]
                break
            time.sleep(0.2)
        user32.MoveWindow(self.main, 30, 20, size[0], size[1] - 56, True)
        self.rec.pid, self.rec.main = self.proc.pid, self.main
        time.sleep(wait)

    def quit(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    # windows
    def top(self, klass: str) -> int:
        for h in windows_of(self.proc.pid) if self.proc else []:
            if cls(h) == klass and user32.IsWindowVisible(h):
                return h
        return 0

    def child(self, parent: int, klass: str) -> int:
        return user32.FindWindowExW(parent, 0, klass, None)

    def wait_for(self, klass: str, timeout: float = 600) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.top(klass):
                return True
            time.sleep(0.5)
        return False

    # commands and tools
    def cmd(self, name: str, target: int | None = None) -> None:
        if target is None:
            target = self.main
        if target:  # a window that is not open gets nothing (never fall back to the main window)
            user32.PostMessageW(target, WM_COMMAND, IDS[name], 0)

    def scheme(self, key: str, pause: float = 1.2) -> None:
        bar = self.child(self.main, "AeroSynapseSchemes")
        k = SCHEMES.index(key)
        l, t, r, b = rect(bar)
        w = max(54, min(132, (r - l - 330 - 130) // len(SCHEMES)))
        x, y = 318 + k * w + (w - 4) // 2, 28
        user32.PostMessageW(bar, WM_MOUSEMOVE, 0, lp(x, y))
        user32.PostMessageW(bar, WM_LBUTTONDOWN, 1, lp(x, y))
        user32.PostMessageW(bar, WM_LBUTTONUP, 0, lp(x, y))
        time.sleep(pause)

    def tool(self, icon: str, pause: float = 1.5) -> None:  # hover the tile, then click it, as a person would
        pal = self.child(self.main, "AeroSynapsePalette")
        scheme = next(t[0] for t in TOOLS if t[1] == icon)
        tiles = [t for t in TOOLS if t[0] == scheme]
        i = next(n for n, t in enumerate(tiles) if t[1] == icon)
        x, y = 6 + (i % 2) * 62 + 29, 46 + (i // 2) * 82 + 39
        user32.PostMessageW(pal, WM_MOUSEMOVE, 0, lp(x, y))  # hover: the tile lights up
        time.sleep(0.5)
        user32.PostMessageW(pal, WM_MOUSEMOVE, 0, lp(x, y))  # and click straight after a fresh hover
        user32.PostMessageW(pal, WM_LBUTTONDOWN, 1, lp(x, y))
        user32.PostMessageW(pal, WM_LBUTTONUP, 0, lp(x, y))
        time.sleep(pause)

    def select(self, index: int) -> None:  # pick a feature in the tree
        lst = self.child(self.main, "ListBox")
        user32.SendMessageW(lst, 0x0186, index, 0)  # LB_SETCURSEL
        user32.PostMessageW(self.main, WM_COMMAND, (1 << 16) | user32.GetDlgCtrlID(lst), lst)  # LBN_SELCHANGE
        time.sleep(1.2)

    # the 3D view
    def view(self) -> int:
        return self.child(self.main, "AeroSynapseView")

    def orbit(self, dx: int, dy: int, steps: int = 30, dt: float = 0.04, button: str = "left") -> None:
        v = self.view()
        l, t, r, b = rect(v)
        cx, cy = (r - l) // 2, (b - t) // 2
        down, up, mk = (WM_LBUTTONDOWN, WM_LBUTTONUP, 1) if button == "left" else (WM_RBUTTONDOWN, WM_RBUTTONUP, 2)
        user32.PostMessageW(v, down, mk, lp(cx, cy))
        for s in range(1, steps + 1):
            user32.PostMessageW(v, WM_MOUSEMOVE, mk, lp(cx + dx * s // steps, cy + dy * s // steps))
            time.sleep(dt)
        user32.PostMessageW(v, up, 0, lp(cx + dx, cy + dy))

    def zoom(self, notches: float, steps: int = 16, dt: float = 0.05) -> None:  # wheel notches (each 12 % closer)
        v = self.view()
        for _ in range(steps):
            user32.PostMessageW(v, WM_MOUSEWHEEL, (int(120 * notches / steps) & 0xFFFF) << 16, 0)
            time.sleep(dt)

    def key(self, vk: int, target: int | None = None) -> None:
        h = target or self.view()
        user32.PostMessageW(h, WM_KEYDOWN, vk, 0)
        user32.PostMessageW(h, WM_KEYUP, vk, 0)

    def click_view(self, fx: float, fy: float) -> None:  # fractions of the view size
        v = self.view()
        l, t, r, b = rect(v)
        x, y = int((r - l) * fx), int((b - t) * fy)
        user32.PostMessageW(v, WM_MOUSEMOVE, 0, lp(x, y))
        user32.PostMessageW(v, WM_LBUTTONDOWN, 1, lp(x, y))
        user32.PostMessageW(v, WM_LBUTTONUP, 0, lp(x, y))

    # rows of a pad or a setup form: the control just after its label
    def _row(self, page: int, label: str) -> int:
        kids = []

        def cb(h, _):
            kids.append(h)
            return True
        user32.EnumChildWindows(page, EnumProc(cb), 0)
        for i, h in enumerate(kids):
            if cls(h) == "Static" and text(h).strip().lower().startswith(label.lower()) and i + 1 < len(kids):
                return kids[i + 1]
        return 0

    def set_field(self, page_class: str, label: str, value: str, typing: bool = True) -> None:
        h = 0
        for _ in range(30):  # the pad or form may still be opening
            top = self.top("AeroSynapsePad" if page_class == "AeroSynapsePadPage" else "AeroSynapseSetup" if page_class == "AeroSynapseSetupPage" else page_class)
            page = self.child(top, page_class) if top else 0
            h = self._row(page, label) if page else 0
            if h:
                break
            time.sleep(0.2)
        if not h:
            print("  (field not found:", label, ")")
            return
        k = cls(h)
        if k == "ComboBox":
            user32.SendMessageW(h, CB_SELECTSTRING, -1, ctypes.c_wchar_p(value))
            cid = user32.GetDlgCtrlID(h)
            user32.SendMessageW(page, WM_COMMAND, (CBN_SELCHANGE << 16) | cid, h)
        elif k == "Button":
            user32.SendMessageW(h, BM_SETCHECK, 1 if value in ("1", "true", "on") else 0, 0)
            cid = user32.GetDlgCtrlID(h)
            user32.SendMessageW(page, WM_COMMAND, (BN_CLICKED << 16) | cid, h)
        else:
            if typing:  # type it out, so the change is seen
                for i in range(1, len(value) + 1):
                    user32.SendMessageW(h, WM_SETTEXT, 0, ctypes.c_wchar_p(value[:i]))
                    time.sleep(0.06)
            else:
                user32.SendMessageW(h, WM_SETTEXT, 0, ctypes.c_wchar_p(value))
        time.sleep(0.3)

    def pad(self, label: str, value: str) -> None:
        self.set_field("AeroSynapsePadPage", label, value)

    def form(self, label: str, value: str) -> None:
        self.set_field("AeroSynapseSetupPage", label, value)

    def pad_ok(self) -> None:
        p = self.top("AeroSynapsePad")
        if p:
            user32.PostMessageW(p, WM_COMMAND, IDS["ID_PAD_OK"], 0)
        time.sleep(1.0)

    def pad_cancel(self) -> None:
        p = self.top("AeroSynapsePad")
        if p:
            user32.PostMessageW(p, WM_COMMAND, IDS["ID_PAD_CANCEL"], 0)
        time.sleep(1.0)

    def close(self, klass: str) -> None:
        h = self.top(klass)
        if h:
            user32.PostMessageW(h, WM_CLOSE, 0, 0)
        time.sleep(0.6)

    def dismiss_messages(self) -> None:  # message boxes: press their (first) button, as a person would
        for _ in range(20):
            boxes = [h for h in windows_of(self.proc.pid) if cls(h) == "#32770" and user32.IsWindowVisible(h)]
            if not boxes:
                return
            for h in boxes:
                btn = user32.FindWindowExW(h, 0, "Button", None)
                if btn:
                    user32.PostMessageW(btn, BM_CLICK, 0, 0)
                else:
                    user32.PostMessageW(h, WM_CLOSE, 0, 0)
            time.sleep(0.4)


# ---------------------------------------------------------------- the tour
def tour(S: Studio, R: Recorder, only: set[str] | None) -> None:
    def chapter(name: str, title: str, lines: list[str]) -> bool:
        if only and name not in only:
            return False
        R.paused = True
        R.card(title, lines, 3.0)
        R.chapter = title
        R.caption, R.note = "", ""
        return True

    def say(caption: str, secs: float = 2.5, note: str = "") -> None:
        R.caption, R.note = caption, note
        R.paused = False
        time.sleep(secs)

    def timelapse(caption: str, klass: str, timeout: float = 900) -> None:  # a real computation: the recording waits
        R.caption, R.note = caption, "running for real; the recording skips the waiting"
        R.paused = False
        time.sleep(1.5)
        R.paused = True
        S.wait_for(klass, timeout)
        time.sleep(1.5)
        R.note = ""
        R.paused = False

    jet = str(ROOT / "assets" / "templates" / "jet_conventional.json")
    falcon = str(ROOT / "projects" / "falcon" / "08_export" / "design.json")

    if chapter("start", "1  Start", ["A blank design, a new project folder, or one of 22 templates:", "jet fighters, airplanes and airfoil wings."]):
        S.start(wait=8)
        say("The start page: start from nothing, a new project in its own folder, or a template", 3.5)
        st = S.top("AeroSynapseStart")
        for _ in range(4):
            user32.PostMessageW(st, WM_MOUSEWHEEL, (-240 & 0xFFFF) << 16, 0)
            time.sleep(0.5)
        say("Airplanes and airfoil wings further down", 2.5)
        for _ in range(4):
            user32.PostMessageW(st, WM_MOUSEWHEEL, (240 & 0xFFFF) << 16, 0)
            time.sleep(0.3)
        user32.PostMessageW(st, WM_MOUSEMOVE, 0, lp(417, 399))
        say("Pick the conventional fighter", 1.5)
        user32.PostMessageW(st, WM_LBUTTONUP, 0, lp(417, 399))
        say("It opens in the editor, built from its feature tree", 4)

    if chapter("benches", "2  Workbenches", ["14 workbenches, each with its own tools:", "the CATIA and Creo functions at concept-design depth."]):
        if not S.proc:
            S.start(jet)
        names = {"start": "Start", "sketch": "Sketcher", "part": "Part Design", "curves": "Surfaces & Math", "aero": "Wing & Aero", "sheet": "Sheet & Composite",
                 "asm": "Assembly & DMU", "xform": "Transform", "view": "View & Analysis", "sim": "Simulate", "opt": "Optimise", "know": "Knowledge",
                 "wf": "Workflow", "export": "Drafting & Export"}
        for k in SCHEMES:
            S.scheme(k, 0.2)
            say(f"{names[k]}: {len([t for t in TOOLS if t[0] == k])} tools", 1.6)
        S.scheme("part", 0.3)
        say("Click the active workbench again: its tool bar slides away ...", 1.0)
        S.scheme("part", 0.3)
        say("... and again to bring it back", 1.8)
        S.scheme("part", 0.3)
        say("F2, F3 and F4 hide the tree, the properties and the tools; F11 gives the whole window to the 3D view", 1.0)
        S.cmd("ID_VIEW_FULL")
        say("Full 3D view", 2.0)
        S.orbit(260, -40, 40)
        S.cmd("ID_VIEW_FULL")
        say("And back", 1.5)

    if chapter("sketch", "3  Sketcher", ["Draw on the XY, XZ or YZ plane with an offset:", "lines and splines, rectangles, circles, slots; pad, pocket, shaft, groove, loft."]):
        if not S.proc:
            S.start(jet)
        S.scheme("part")
        S.tool("shaft", 1.5)
        sk = S.top("AeroSynapseSketch")
        canvas = S.child(sk, "AeroSynapseSketchCanvas")
        say("Shaft: a profile on the side plane, turned about its axis (an external store)", 2.0)
        off = user32.GetDlgItem(sk, IDS["ID_SK_OFF"])
        user32.SendMessageW(off, WM_SETTEXT, 0, ctypes.c_wchar_p("2.6"))
        l, t, r, b = rect(canvas)
        wc, hc = r - l, b - t
        R.caption = "Click the profile: each segment shows its length and angle; Snap and Spline are on"
        for wx, wy in [(6.2, 0.04), (6.6, 0.26), (7.5, 0.34), (8.8, 0.3), (9.6, 0.06)]:
            sx, sy = int(wc / 2 + (wx - 8) * 55), int(hc / 2 - (wy - 1.2) * 55)
            user32.PostMessageW(canvas, WM_MOUSEMOVE, 0, lp(sx, sy))
            time.sleep(0.25)
            user32.PostMessageW(canvas, WM_LBUTTONDOWN, 1, lp(sx, sy))
            user32.PostMessageW(canvas, WM_LBUTTONUP, 0, lp(sx, sy))
            time.sleep(0.45)
        say(R.caption, 1.5)
        user32.PostMessageW(sk, WM_COMMAND, IDS["ID_SK_FINISH"], 0)
        time.sleep(1.5)
        say("Finish: the store is added, and its pad opens with every parameter", 2.5)
        S.pad("Move x, y, z", "0, 0, -0.45")
        say("Typing in the pad updates the 3D view at once (a live preview)", 3.0)
        S.pad_ok()
        say("OK keeps it", 1.0)
        S.orbit(0, 140, 40)
        say("The new store under the wing", 2.5)
        S.orbit(0, -140, 30)

    if chapter("part", "4  Part Design", ["Pad, pocket, shaft, groove, rib, slot, hole, fillet, chamfer, shell, draft,", "mirror, pattern and solid primitives."]):
        if not S.proc:
            S.start(jet)
        S.scheme("part")
        S.tool("cut_hole", 2.0)
        say("Hole: diameter, through-all or depth, entry point and direction along any axis or vector", 3.0)
        S.pad("Diameter", "0.18")
        S.cmd("ID_PAD_PICK", S.top("AeroSynapsePad"))
        say("Pick the entry point on the model: the hole is drilled square into the surface", 1.5)
        S.click_view(0.42, 0.56)
        say("The point and the surface normal are filled in; the preview shows the hole", 3.0)
        S.pad_ok()
        S.zoom(14)
        say("Zoomed in: the view re-meshes what is in sight at the screen's resolution", 3.0)
        S.zoom(-14)
        S.select(0)
        S.tool("chamfer", 1.5)
        S.pad("Chamfer", "0.12")
        say("Chamfer on the selected part, with a live preview", 2.5)
        S.pad_cancel()
        say("Cancel puts the design back exactly as it was", 2.0)
        S.tool("pattern", 1.5)
        say("Pattern: copies with a step and a rotation", 2.5)
        S.pad_cancel()

    if chapter("curves", "5  Surfaces & Math", ["Revolve, extrude, loft, Bezier sheets, ducts,", "equation surfaces f(x,y,z) <= 0, superellipsoids, parametric sweeps, lattices."]):
        if not S.proc:
            S.start(jet)
        S.scheme("curves")
        S.tool("implicit", 2.0)
        say("An equation surface: any f(x, y, z) <= 0, here a pod whose radius ripples along x", 3.0)
        S.pad("expr", "y^2 + z^2 - (0.3 + 0.12*sin(4*x))^2")
        say("Change the equation: the shape follows", 3.0)
        S.pad_ok()
        S.tool("sweep", 2.0)
        say("A parametric sweep: a tube along x(t), y(t), z(t) (a helix)", 3.0)
        S.pad_ok()
        S.tool("superellipsoid", 2.0)
        say("A superellipsoid, between an ellipsoid and a brick", 2.5)
        S.pad_ok()

    if chapter("aero", "6  Wing & Aero", ["Wings, canards, tails, fins, strakes and blades,", "drawn as exact surfaces with true leading and trailing edges."]):
        if not S.proc:
            S.start(jet)
        S.scheme("aero")
        S.tool("canard", 2.0)
        say("A canard: span, chord, taper, sweep, dihedral, twist, thickness and camber", 2.5)
        S.pad("sweep", "50")
        S.pad("span", "1.6")
        say("Sweep and span typed in; the canard follows", 3.0)
        S.pad_ok()
        S.orbit(-200, 30, 40)
        say("Every surface is drawn from its exact definition", 2.5)

    if chapter("sheet", "7  Sheet & Composite", ["Bent plates and their flat blank, laminates by", "classical lamination theory, lattice infill, thin walls."]):
        if not S.proc:
            S.start(jet)
        S.scheme("sheet")
        S.tool("bentplate", 2.0)
        say("A bent plate: flanges with a bend radius", 2.5)
        S.pad_ok()
        S.tool("unfold", 2.0)
        say("Unfold: the flat blank's length and size", 3.0)
        S.dismiss_messages()
        S.tool("laminate", 2.0)
        say("Laminate setup: the ply layup and the ply material", 2.0)
        S.form("Ply layup", "[0/45/-45/90/0]s")
        S.cmd("ID_SETUP_RUN", S.top("AeroSynapseSetup"))
        timelapse("Classical lamination theory", "AeroSynapseSimView", 300)
        say("Stiffness in every direction, the ply stack, and the A, B, D matrices", 4.0)
        S.close("AeroSynapseSimView")

    if chapter("asm", "8  Assembly & DMU", ["Clash analysis, space reservations, kinematics,", "pipe and harness routing."]):
        if not S.proc:
            S.start(jet)
        S.scheme("asm")
        S.tool("reserve", 2.0)
        say("A space reservation (a volume kept free, checked but not built)", 2.5)
        S.pad_ok()
        S.tool("harness", 2.0)
        say("A harness route along a path", 2.0)
        S.pad_ok()
        S.tool("clash", 1.5)
        S.form("Analysis spacing", "0.08")
        S.cmd("ID_SETUP_RUN", S.top("AeroSynapseSetup"))
        timelapse("Clash analysis of every pair of parts", "AeroSynapseSimView", 900)
        say("Overlaps between parts, in litres, and reservation violations", 4.0)
        S.close("AeroSynapseSimView")

    if chapter("xform", "9  Transform", ["Mirror, pattern, twist, taper, bend, shear, stretch, pivot rotation,", "duplicate, add / cut, undo and redo."]):
        if not S.proc:
            S.start(jet)
        S.scheme("xform")
        S.tool("twist", 2.0)
        say("Twist the selected part about its axis", 2.5)
        S.pad_cancel()
        S.tool("bend", 2.0)
        say("Bend it (an exact circular bend)", 2.5)
        S.pad_cancel()
        say("Undo and redo keep the whole design history", 2.0)

    if chapter("view", "10  View & Analysis", ["Standard views, sections, wireframe, zebra and curvature,", "3D dimensions, mass properties and the design check."]):
        if not S.proc:
            S.start(jet)
        S.scheme("view")
        for t, c in [("view_front", "Front"), ("view_side", "Side"), ("view_top", "Top"), ("view_iso", "Isometric")]:
            S.tool(t, 0.3)
            say(c, 1.4)
        S.tool("view_wire", 0.5)
        say("Wireframe: the structured grids of the exact surfaces", 3.0)
        S.tool("view_wire", 0.5)
        S.tool("cut_x", 0.5)
        say("A section cut through the body", 3.0)
        S.tool("cut_x", 0.5)
        S.tool("zebra", 0.5)
        say("Zebra stripes show how smoothly surfaces join", 3.0)
        S.tool("zebra", 0.5)
        S.tool("curvature", 0.5)
        say("Curvature colouring", 2.5)
        S.tool("curvature", 0.5)
        S.tool("annotate", 0.5)
        say("3D dimensions", 2.5)
        S.tool("annotate", 0.3)
        S.orbit(300, -60, 50)
        S.zoom(6)
        say("Close up, the edges stay sharp", 3.0)
        S.zoom(-6)

    if chapter("prefs", "11  Preferences", ["Plain or dark charts, a calm or vivid 3D display,", "a fixed or auto-hide sliding tool bar."]):
        if not S.proc:
            S.start(jet)
        S.cmd("ID_PREFS")
        time.sleep(1.0)
        say("Preferences", 2.0)
        pr = S.top("AeroSynapsePrefs")
        user32.SendMessageW(user32.GetDlgItem(pr, IDS["ID_PREF_PALETTE"]), 0x014E, 1, 0)  # CB_SETCURSEL: auto-hide
        say("Tool bar: auto-hide", 1.5)
        user32.PostMessageW(pr, WM_COMMAND, IDS["ID_PREF_OK"], 0)
        say("The tool bar rests as a thin handle ...", 2.5)
        pal = S.child(S.main, "AeroSynapsePalette")
        user32.PostMessageW(pal, WM_MOUSEMOVE, 0, lp(6, 300))
        say("... and slides out under the mouse", 2.5)
        S.cmd("ID_PREFS")
        time.sleep(0.6)
        pr = S.top("AeroSynapsePrefs")
        user32.SendMessageW(user32.GetDlgItem(pr, IDS["ID_PREF_PALETTE"]), 0x014E, 0, 0)
        user32.PostMessageW(pr, WM_COMMAND, IDS["ID_PREF_OK"], 0)
        say("Back to fixed", 1.5)

    if chapter("sim", "12  Simulate", ["Every condition is chosen in a setup form;", "full-3D flow, turbulence and instability, analytic, PINN and structure."]):
        S.start(falcon, "--scheme", "sim")
        S.tool("sim_setup", 1.5)
        say("3D flow setup: flight condition, free-stream turbulence, gusts, model, grid and outputs", 3.5)
        S.form("Preset", "transonic")
        say("A preset fills the flight condition; every value can be changed", 2.5)
        S.form("Preset", "cruise")
        S.form("Quality", "live")
        su = S.top("AeroSynapseSetup")
        page = S.child(su, "AeroSynapseSetupPage")
        for _ in range(6):
            user32.PostMessageW(page, WM_MOUSEWHEEL, (-240 & 0xFFFF) << 16, 0)
            time.sleep(0.4)
        say("Grid, time stepping, outputs and probe points", 2.5)
        S.cmd("ID_SETUP_RUN", su)
        timelapse("Full-3D flow round the whole aircraft (live quality)", "AeroSynapseSimView", 900)
        say("The results: convergence, surface pressure, sections, slices and 3D renders", 4.0)
        S.close("AeroSynapseSimView")
        S.orbit(200, -30, 40)
        say("Streamlines all round the aircraft, Mach slices and particles", 4.0)
        S.tool("turbulence", 1.5)
        say("Turbulence and instability: transition, wake stability, spectra (from the run just made)", 3.0)
        S.form("Free-stream Tu", "0.5")
        S.cmd("ID_SETUP_RUN", S.top("AeroSynapseSetup"))
        timelapse("e^N transition, Rayleigh wake stability, turbulent energy", "AeroSynapseSimView", 600)
        say("Transition moves forward as the air gets rougher; the wake's unstable modes", 4.0)
        S.close("AeroSynapseSimView")
        S.tool("structure", 1.5)
        S.cmd("ID_SETUP_RUN", S.top("AeroSynapseSetup"))
        timelapse("Loads, wing box sizing, finite elements and flutter", "AeroSynapseSimView", 900)
        say("V-n envelope, wing box, deflection and flutter", 3.5)
        S.close("AeroSynapseSimView")
        say("The deflected wing, coloured by cover stress", 3.0)
        S.tool("pinn", 1.5)
        say("PINN skin heating: check points and the material and thickness of every skin zone", 2.5)
        S.form("wing box: material", "ti64")
        S.cmd("ID_SETUP_RUN", S.top("AeroSynapseSetup"))
        timelapse("The physics-informed network against the exact series and finite differences", "AeroSynapseSimView", 900)
        say("Errors, speed and the skin zones' peak temperatures", 3.5)
        S.close("AeroSynapseSimView")
        S.tool("analytic", 1.5)
        S.cmd("ID_SETUP_RUN", S.top("AeroSynapseSetup"))
        timelapse("Panel method, lifting line, area rule, shocks, skin heating", "AeroSynapseSimView", 600)
        say("Analytic solutions of the design", 3.0)
        S.close("AeroSynapseSimView")

    if chapter("flow3d", "13  Full-3D flow", ["A standard-quality run of the Falcon at Mach 0.85:", "331,000 cells, the whole aircraft."]):
        S.start(falcon, "--clean", "--scheme", "sim", "--field", str(ROOT / "results" / "cfd3d" / "falcon_cruise_full"), wait=14)
        say("Mach slices, particles, streamlines and vortex cores", 3.5)
        S.orbit(320, -50, 60)
        S.tool("slice", 0.5)
        say("Slice axis", 2.0)
        S.zoom(4)
        S.orbit(-200, 20, 40)
        say("All round the whole aircraft", 3.0)

    if chapter("opt", "14  Optimise", ["Airfoil shape, XFOIL check, topology, fighter and aircraft studies,", "surrogate models, wing sizing."]):
        if not S.proc:
            S.start(falcon)
        S.scheme("opt")
        S.tool("opt_centre", 2.0)
        say("The optimisation centre: every study with its description", 3.5)
        S.close("AeroSynapseJobs")

    if chapter("know", "15  Knowledge", ["Skeleton parameters, design rules, requirements traceability (RFLP)."]):
        if not S.proc:
            S.start(falcon)
        S.scheme("know")
        S.tool("skeleton", 2.0)
        say("Skeleton: the design variables drive the whole geometry", 3.0)
        S.close("AeroSynapseParams")
        S.tool("rules", 1.0)
        timelapse("Design rules checked on measured values", "AeroSynapseSimView", 300)
        say("Every rule with its measured value", 3.0)
        S.close("AeroSynapseSimView")

    if chapter("wf", "16  Design workflow", ["Basic design, design and wing and body optimisation, structure,", "PINN skin, 3D flow and export, in eight stages."]):
        S.start("--project", "falcon", wait=10)
        say("The workflow window: every stage, its status and results", 4.0)
        S.close("AeroSynapseWorkflow")

    if chapter("export", "17  Drafting & Export", ["A3 drawing, STEP, glTF, STL, OBJ, DXF,", "reports as PDF, Word and text."]):
        S.start("--simview", str(ROOT / "projects" / "falcon" / "08_export"), falcon, wait=12)  # the viewer paints its drawing first
        say("The export stage: drawing, CAD files, bill of materials, report", 4.0)

    R.paused = True
    R.card("AeroSynapse Studio", ["Design, simulate, optimise, export.", "Every scene above is the program itself, recorded as it ran."], 4.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo" / "aerosynapse_demo.mp4"))
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    R = Recorder(out, a.fps, (1440, 896))
    S = Studio(R)
    try:
        tour(S, R, set(a.only.split(",")) if a.only else None)
    finally:
        R.paused = True
        S.quit()
        R.close()
    print(f"{out}  {R.frames} frames, {R.frames / a.fps:.0f} s")


if __name__ == "__main__":
    main()
