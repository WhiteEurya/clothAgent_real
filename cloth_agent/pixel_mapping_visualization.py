"""RGB / robot-base XY views of a measured XYZ map (not ground truth)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


class PixelMappingView:
    """Render an RGB coordinate grid and an equal-scale, colored XY projection."""

    def __init__(self, rgb, xyz, valid, grid_mm=50.0):
        if not np.isfinite(grid_mm) or grid_mm <= 0:
            raise ValueError('Grid spacing must be finite and positive')
        self.rgb = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        self.xyz = np.asarray(xyz)
        if self.xyz.shape != (*np.asarray(rgb).shape[:2], 3):
            raise ValueError('RGB and XYZ dimensions differ')
        self.valid = np.asarray(valid, dtype=bool) & np.isfinite(self.xyz).all(axis=2)
        if not self.valid.any():
            raise ValueError('No finite points to visualize')
        self.grid_mm = grid_mm
        xy = self.xyz[self.valid, :2]
        low, high = xy.min(axis=0), xy.max(axis=0)
        self.center = (low + high) / 2
        # Equal mm/pixel on both axes; all valid points fit, without percentile clipping.
        self.span = max(float(np.max(high-low)) * 1.08, grid_mm * 2)
        self.panel_size = 640
        self.margin = 60
        self.plot_size = self.panel_size - 2*self.margin
        self.grid_rgb = self._rgb_grid()
        self.xy_image = self._xy_image()

    def xy_to_panel(self, xy):
        values = np.asarray(xy)
        normalized = (values - self.center) / self.span
        return np.stack((320 + normalized[..., 0] * self.plot_size,
                         320 - normalized[..., 1] * self.plot_size), axis=-1)

    def _rgb_grid(self):
        array = np.array(self.rgb)
        # Thin coordinate bands on measured surfaces; missing depth stays unmarked.
        width = min(1.0, self.grid_mm / 20)
        labels = []
        for axis, color, name in ((0, (0, 230, 255), 'X'), (1, (255, 200, 0), 'Y')):
            values = np.where(self.valid, self.xyz[..., axis], 0)
            distance = np.abs((values + self.grid_mm/2) % self.grid_mm - self.grid_mm/2)
            mask = self.valid & (distance < width)
            array[mask] = color
            rows, cols = np.nonzero(mask)
            levels = np.rint(values[mask] / self.grid_mm).astype(np.int64)
            # Bound label density without changing grid spacing.
            unique = np.unique(levels)
            for level in unique[::max(1, (len(unique)+9)//10)]:
                indexes = np.flatnonzero(levels == level)
                index = indexes[len(indexes)//2]
                labels.append((int(cols[index]), int(rows[index]), f'{name}={level*self.grid_mm:g}', color))
        result = Image.fromarray(array)
        draw = ImageDraw.Draw(result)
        for x, y, text, color in labels:
            draw.text((x, y), text, fill=color, stroke_width=1, stroke_fill='black')
        draw.rectangle((0, 0, min(result.width, 430), 36), fill='#17212b')
        draw.text((6, 4), f'Raw RGB | measured grid {self.grid_mm:g} mm', fill='white')
        draw.text((6, 20), 'Cyan: X constant   Yellow: Y constant', fill='white')
        return result

    def _xy_image(self):
        array = np.full((640, 640, 3), [24, 30, 39], dtype=np.uint8)
        rows, cols = np.nonzero(self.valid)
        # Keep bounded rendering cost for megapixel frames. Sort by Z so the
        # top surface wins where several sampled pixels share a projected cell.
        stride = max(1, (len(rows)+149999)//150000)
        rows, cols = rows[::stride], cols[::stride]
        order = np.argsort(self.xyz[rows, cols, 2])
        rows, cols = rows[order], cols[order]
        points = np.rint(self.xy_to_panel(self.xyz[rows, cols, :2])).astype(int)
        flat = points[:, 1] * 640 + points[:, 0]
        _, reverse_indexes = np.unique(flat[::-1], return_index=True)
        indexes = len(flat)-1-reverse_indexes
        points = points[indexes]
        array[points[:, 1], points[:, 0]] = np.asarray(self.rgb)[rows[indexes], cols[indexes]]
        result = Image.fromarray(array)
        draw = ImageDraw.Draw(result)
        # Use a coarser labeled grid if the full scene spans many metres.
        step = self.grid_mm * max(1, int(np.ceil(self.span/self.grid_mm/12)))
        for axis in (0, 1):
            lo, hi = self.center[axis]-self.span/2, self.center[axis]+self.span/2
            levels = np.arange(np.ceil(lo/step), np.floor(hi/step)+1) * step
            for value in levels:
                xy = self.center.copy()
                xy[axis] = value
                px, py = self.xy_to_panel(xy)
                if axis == 0:
                    draw.line((px, 60, px, 580), fill='#46505a')
                    draw.text((px-14, 585), f'{value:g}', fill='#00e6ff')
                else:
                    draw.line((60, py, 580, py), fill='#46505a')
                    draw.text((4, py-5), f'{value:g}', fill='#ffc800')
        draw.text((60, 10), 'ROBOT BASE XY - computed from RGB-D', fill='white')
        draw.text((60, 28), 'Prediction only; not independent physical measurement', fill='#ffc800')
        draw.text((60, 610), '+X right, +Y up | mm | equal scale on both axes', fill='white')
        return result

    def render(self, pixels=(), show_grid=True):
        left = (self.grid_rgb if show_grid else self.rgb).copy()
        right = self.xy_image.copy()
        left_draw, right_draw = ImageDraw.Draw(left), ImageDraw.Draw(right)
        for index, (u, v) in enumerate(pixels, start=1):
            if not (0 <= u < left.width and 0 <= v < left.height) or not self.valid[v, u]:
                raise ValueError(f'Invalid selected pixel {(u, v)}')
            x, y = self.xy_to_panel(self.xyz[v, u, :2])
            for draw, px, py in ((left_draw, u, v), (right_draw, x, y)):
                draw.ellipse((px-6, py-6, px+6, py+6), outline='#ff5050', width=3)
                draw.text((px+8, py-12), f'P{index}', fill='white', stroke_width=1, stroke_fill='black')
        right = right.resize((left.height, left.height), Image.Resampling.LANCZOS)
        combined = Image.new('RGB', (left.width + right.width, left.height))
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width, 0))
        return combined
