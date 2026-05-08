from __future__ import annotations

from pathlib import Path

from bot_app.personality.interest_field import PerlinInterestField


class InterestFingerprintRenderer:
    """Generates a deterministic 2D SVG fingerprint from the bot's interest seed."""

    def __init__(self, field: PerlinInterestField) -> None:
        self.field = field

    def render_svg(self, path: str | Path, *, size: int = 96, cell: int = 5) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        size = max(24, int(size))
        cell = max(2, int(cell))
        width = size * cell
        height = size * cell + 56

        rects: list[str] = []
        for y in range(size):
            ny = y / max(1, size - 1) * 2 - 1
            for x in range(size):
                nx = x / max(1, size - 1) * 2 - 1
                r = int(35 + 190 * self.field.fingerprint_value(nx, ny, "fingerprint_r"))
                g = int(35 + 190 * self.field.fingerprint_value(nx + 13.7, ny - 4.2, "fingerprint_g"))
                b = int(35 + 190 * self.field.fingerprint_value(nx - 9.1, ny + 6.6, "fingerprint_b"))
                # Make the image more fingerprint-like with a soft contour overlay.
                contour = self.field.fingerprint_value(nx * 1.8, ny * 1.8, "innate_interest")
                if 0.47 < contour < 0.53:
                    r = min(255, int(r * 1.25))
                    g = min(255, int(g * 1.25))
                    b = min(255, int(b * 1.25))
                rects.append(
                    f'<rect x="{x * cell}" y="{y * cell}" width="{cell}" height="{cell}" fill="rgb({r},{g},{b})" />'
                )

        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#111" />
  <g>{''.join(rects)}</g>
  <rect x="0" y="{size * cell}" width="{width}" height="56" fill="#111" opacity="0.92" />
  <text x="12" y="{size * cell + 22}" fill="#eee" font-family="monospace" font-size="16">Byamka interest fingerprint</text>
  <text x="12" y="{size * cell + 44}" fill="#bbb" font-family="monospace" font-size="12">seed={self.field.seed}, latent_dim={self.field.latent_dim}, octaves={self.field.octaves}</text>
</svg>
'''
        path.write_text(svg, encoding="utf-8")
        return path
