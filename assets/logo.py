"""Draw Pandora's logo: the jar, opened, with the sparse answer rising out of it.

    python assets/logo.py        (writes the SVGs next to this file)

The jar is a pithos in line art. Inside it, faint flutes stand for the many columns of X; out of
its mouth rise a few stems, each tipped with a diamond: the few nonzeros Pandora recovers. The lid
sits lifted to one side. Ink, line weight, the Greek key and the diamonds follow the house style
of github.com/EgorKhaklin; the wordmark is set in Cinzel capitals, kept as outlines
(cinzel-caps.json, SIL Open Font License 1.1) so it renders the same everywhere.

GitHub serves these through an img tag, so the animation lives inside each SVG: it plays once,
uses opacity and stroke drawing only, and a reduced-motion rule shows the finished frame.

Writes: pandora-light.svg and pandora-dark.svg (the lockup for the README header), and
pandora-mark-light.svg and pandora-mark-dark.svg (the jar alone, square).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPS = json.loads((HERE / "cinzel-caps.json").read_text())
THEMES = {"light": ("#1F2328", "#59636E"), "dark": ("#E6EDF3", "#9198A1")}

# the jar, in its own 220 x 300 box
BODY = ("M86 114C52 120 35 148 36 180C37 220 60 258 84 276L82 286H138L136 276"
        "C160 258 183 220 184 180C185 148 168 120 134 114Z")
RIM = "M80 92H140A3 3 0 0 1 140 101H80A3 3 0 0 1 80 92ZM86 101L86 114M134 101L134 114"
LID = ("M-31 0H31A3 3 0 0 1 31 6H-31A3 3 0 0 1 -31 0ZM-25 0Q-23 -14 0 -15Q23 -14 25 0"
       "M-3.6 -19A3.6 3.6 0 1 0 3.6 -19A3.6 3.6 0 1 0 -3.6 -19Z")
STEMS = [(90, 34), (100, 60), (110, 84), (120, 46), (130, 68)]  # x, height above the rim
FLUTES = [46 + 8 * i for i in range(17)]


def f(v):
    return ("%.2f" % v).rstrip("0").rstrip(".")


def meander(x0, x1, y, tile):
    """A running Greek key, `tile` px tall, from x0 to x1, top at y."""
    s, n = tile / 30.0, int((x1 - x0) // tile)
    x0 += ((x1 - x0) - n * tile) / 2
    pts = [(0, 30), (0, 0), (24, 0), (24, 18), (12, 18), (12, 12), (18, 12), (18, 6), (6, 6), (6, 30),
           (30, 30)]
    d = []
    for i in range(n):
        for j, (px, py) in enumerate(pts):
            d.append("%s%s %s" % ("M" if i == 0 and j == 0 else "L", f(x0 + i * tile + px * s), f(y + py * s)))
    return "".join(d)


def diamond(cx, cy, r):
    return ('<rect x="%s" y="%s" width="%s" height="%s" transform="rotate(45 %s %s)"/>'
            % (f(cx - r), f(cy - r), f(2 * r), f(2 * r), f(cx), f(cy)))


def text_run(label, cap, track_em, x0, base):
    """Cinzel capitals from x0 on a baseline: (svg, width)."""
    s = cap / CAPS["cap"]
    track = track_em * CAPS["upm"] * s
    out, x = [], 0.0
    for ch in label:
        if ch == " ":
            x += CAPS["space"] * s + track
            continue
        g = CAPS["glyphs"][ch]
        out.append('<path transform="translate(%s %s) scale(%s)" d="%s"/>'
                   % (f(x0 + x), f(base), ("%.5f" % s).rstrip("0"), g["d"]))
        x += g["advance"] * s + track
    return "".join(out), x - track


STYLE = (".draw{stroke-dasharray:1 2;stroke-dashoffset:1.01;animation:draw 1.1s cubic-bezier(.45,0,.2,1) .1s 1 both}"
         ".rise{stroke-dasharray:1 2;stroke-dashoffset:1.01;animation:draw .8s cubic-bezier(.3,0,.2,1) 1s 1 both}"
         ".gem{opacity:0;animation:fade .5s ease-out 1.6s 1 both}"
         ".soft{opacity:0;animation:soft .9s ease-out .7s 1 both}"
         ".word{opacity:0;animation:fade 1s ease-out 1.2s 1 both}"
         "@keyframes draw{from{stroke-dashoffset:1.01}to{stroke-dashoffset:0}}"
         "@keyframes fade{from{opacity:0}to{opacity:1}}"
         "@keyframes soft{from{opacity:0}to{opacity:.42}}"
         "@media (prefers-reduced-motion: reduce){.draw,.rise{animation:none;stroke-dashoffset:0}"
         ".gem,.word{animation:none;opacity:1}.soft{animation:none;opacity:.42}}")


def jar(ink, ox, oy, k, uid):
    """The jar at (ox, oy), scale k. uid keeps clip ids unique per file."""
    stems = "".join('<path class="rise" pathLength="1" style="animation-delay:%ss" d="M%s 92V%s"/>'
                    % (f(1.0 + .12 * i), f(x), f(92 - h)) for i, (x, h) in enumerate(STEMS))
    gems = "".join('<g class="gem" style="animation-delay:%ss">%s</g>' % (f(1.55 + .12 * i), diamond(x, 92 - h - 4.8, 4.2))
                   for i, (x, h) in enumerate(STEMS))
    flutes = "".join("M%s 158V230" % f(x) for x in FLUTES)
    return f"""<g transform="translate({f(ox)} {f(oy)}) scale({f(k)})">
<clipPath id="in{uid}"><path d="{BODY}"/></clipPath>
<g class="soft" clip-path="url(#in{uid})" fill="none" stroke="{ink}" stroke-width="1.1">
<path d="{flutes}"/>
<path d="M20 130H200M20 236H200" stroke-width="1.2"/>
<path d="{meander(30, 190, 138, 12)}" stroke-width="1.1" stroke-linejoin="miter" stroke-linecap="square"/>
</g>
<g fill="none" stroke="{ink}" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round">
<path class="draw" pathLength="1" d="{BODY}"/>
<path class="draw" pathLength="1" d="{RIM}"/>
<g transform="translate(42 60) rotate(-22)"><path class="draw" pathLength="1" d="{LID}"/></g>
</g>
<g fill="none" stroke="{ink}" stroke-width="1.7" stroke-linecap="round">{stems}</g>
<g fill="{ink}">{gems}</g>
</g>"""


def lockup(theme):
    ink, muted = THEMES[theme]
    W, H = 1200, 360
    k = 1.1
    name_cap, name_track = 66, .2
    sub_cap, sub_track = 15, .34
    _, nw = text_run("PANDORA", name_cap, name_track, 0, 0)
    _, sw = text_run("CERTIFIED SPARSE RECOVERY", sub_cap, sub_track, 0, 0)
    gap = 44
    total = 220 * k + gap + max(nw, sw)
    x0 = (W - total) / 2
    tx = x0 + 220 * k + gap
    name, _ = text_run("PANDORA", name_cap, name_track, tx, 196)
    sub, _ = text_run("CERTIFIED SPARSE RECOVERY", sub_cap, sub_track, tx + (nw - sw) / 2, 244)
    rule_y = 218
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t">
<title id="t">Pandora: certified sparse recovery</title>
<style>{STYLE}</style>
{jar(ink, x0, 12, k, theme)}
<g class="word" fill="{ink}">{name}</g>
<g class="word" fill="{muted}">{sub}</g>
<g class="word" stroke="{muted}" stroke-width="1.1"><path d="M{f(tx)} {rule_y}H{f(tx + nw / 2 - 12)}M{f(tx + nw / 2 + 12)} {rule_y}H{f(tx + nw)}"/></g>
<g class="word" fill="{muted}">{diamond(tx + nw / 2, rule_y, 3.2)}</g>
</svg>
"""


def mark(theme):
    ink, _ = THEMES[theme]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="320" height="320" role="img" aria-labelledby="t">
<title id="t">Pandora</title>
<style>{STYLE}</style>
{jar(ink, 50, 14, 1.0, "m" + theme)}
</svg>
"""


def main():
    for theme in THEMES:
        (HERE / f"pandora-{theme}.svg").write_text(lockup(theme))
        (HERE / f"pandora-mark-{theme}.svg").write_text(mark(theme))


if __name__ == "__main__":
    main()
