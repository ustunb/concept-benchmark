from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random 
import os
import glob
import math
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates

class SimpleHandwrittenGenerator: 
    def __init__(self, fonts_dir='./fonts'):
        """ Initialize the generator with handwritten fonts.
        Args:
            fonts_dir: Directory containing TTF font files
        """
        self.fonts_dir = fonts_dir
        self.font_files = self._load_font_files()
        if not self.font_files:
            print(f"WARNING: No fonts found in {fonts_dir}")
            print("Please download handwritten fonts and place them in the fonts directory")

    def _load_font_files(self):
        """Load all TTF font files from the fonts directory."""
        if not os.path.exists(self.fonts_dir):
            os.makedirs(self.fonts_dir)
            return []
        font_files = glob.glob(os.path.join(self.fonts_dir, '*.ttf'))
        font_files.extend(glob.glob(os.path.join(self.fonts_dir, '*.otf')))
        return font_files
    
    def generate(self, digit, size=64, color=(91,58,99), rng_seed=None):
        """
        Generate a handwritten digit image with random variations.
        Args:
            digit: Integer 0-9 representing which digit to generate
            size: Desired output size
            color: RGB tuple for the digit color
            rng_seed: Random seed for reproducible results
        Returns:
            PIL Image object (RGBA mode)
        """
        if not 0 <= digit <= 9:
            raise ValueError("Digit must be between 0 and 9")
        if not self.font_files:
            raise ValueError("No fonts available. Please add TTF files to the fonts directory")
        # Use a local RNG to avoid perturbing global state.
        rng = random.Random(rng_seed) if rng_seed is not None else random
        # Create a larger canvas for transformations
        canvas_size = size * 3
        # Random variations
        rotation_angle = rng.uniform(-15, 15)
        x_offset = rng.randint(-size//4, size//4)
        y_offset = rng.randint(-size//4, size//4)
        scale_factor = rng.uniform(0.8, 1.2)
        blur_radius = rng.choice([0, 0, 0, 0.5, 1.0])  # Most times no blur
        # Choose random font
        font_file = rng.choice(self.font_files)
        font_size = int(size * scale_factor * 1.5)
        try:
            font = ImageFont.truetype(font_file, font_size)
        except Exception as e:
            print(f"Error loading font {font_file}: {e}")
            # Fallback to default
            font = ImageFont.load_default()
        # Create temporary image for text
        temp_img = Image.new('RGBA', (canvas_size, canvas_size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(temp_img)
        # Draw text in center
        text = str(digit)
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # Center text
        x = (canvas_size - text_width) // 2 + x_offset
        y = (canvas_size - text_height) // 2 + y_offset
        # Draw text
        draw.text((x, y), text, font=font, fill=color + (255,))
        # Apply rotation
        if abs(rotation_angle) > 0.1:
            temp_img = temp_img.rotate(
                rotation_angle,
                resample=Image.BICUBIC,
                expand=False
            )
        # Apply slight blur for organic feel
        if blur_radius > 0:
            temp_img = temp_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # Crop to center and resize
        crop_box = (
            canvas_size // 2 - size,
            canvas_size // 2 - size,
            canvas_size // 2 + size,
            canvas_size // 2 + size
        )
        result_img = temp_img.crop(crop_box)
        result_img = result_img.resize((size, size), Image.LANCZOS)
        # Add random thickness variation
        if rng.random() > 0.7:
            result_img = self._vary_thickness(result_img, rng=rng)
        return result_img
    
    def _vary_thickness(self, img, *, rng=None):
        """
        Randomly vary the thickness of the digit slightly.
        Args:
            img: PIL Image
        Returns:
            Modified PIL Image
        """
        rng = rng or random
        # Convert to numpy array
        img_array = np.array(img)
        # Apply slight dilation or erosion
        if rng.random() > 0.5:
            # Dilate (thicker)
            kernel = np.ones((2, 2))
            alpha = img_array[:, :, 3]
            alpha = ndimage.maximum_filter(alpha, footprint=kernel)
            img_array[:, :, 3] = alpha
        else:
            # Erode (thinner)
            kernel = np.ones((2, 2))
            alpha = img_array[:, :, 3]
            alpha = ndimage.minimum_filter(alpha, footprint=kernel)
            img_array[:, :, 3] = alpha
        return Image.fromarray(img_array, mode='RGBA')
    
    def generate_batch(self, digit, count=10, size=64, color=(0, 0, 0), base_seed=None):
        """
        Generate multiple variations of the same digit.
        Args:
            digit: Integer 0-9
            count: Number of variations to generate
            size: Size of each image
            color: RGB color tuple
            base_seed: Base random seed (each variation will use base_seed + i)
        Returns:
            List of PIL Images
        """
        images = []
        for i in range(count):
            seed = base_seed + i if base_seed is not None else None
            img = self.generate(digit, size, color, seed)
            images.append(img)
        return images

class AdvancedHandwrittenGenerator(SimpleHandwrittenGenerator): 
    def __init__(self, fonts_dir='./fonts'):
        super().__init__(fonts_dir)

    """Extended version with more realistic distortions using OpenCV."""
    def generate(self, digit, size=64, color=(0, 0, 0), rng_seed=None):
        """Generate with elastic distortions."""
        try:
            import cv2
        except ImportError:
            print("OpenCV not available, falling back to simple generation")
            return super().generate(digit, size, color, rng_seed)
        # Get base image from parent class
        img = super().generate(digit, size * 2, color, rng_seed)
        # Convert to numpy array
        img_array = np.array(img)
        # Apply elastic distortion
        rng = random.Random(rng_seed) if rng_seed is not None else random
        if rng.random() > 0.5:
            img_array = self._elastic_transform(
                img_array,
                alpha=rng.uniform(5, 15),
                sigma=rng.uniform(2, 4)
            )
        # Convert back to PIL and resize
        img = Image.fromarray(img_array, mode='RGBA')
        img = img.resize((size, size), Image.LANCZOS)
        return img
    
    def _elastic_transform(self, image, alpha, sigma):
        """Apply elastic distortion to the image."""
        import cv2
        random_state = np.random.RandomState(None)
        shape = image.shape[:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        distorted_image = np.zeros_like(image)
        for i in range(image.shape[2]):
            distorted_image[:, :, i] = map_coordinates(
                image[:, :, i],
                indices,
                order=1,
                mode='reflect'
            ).reshape(shape)
        return distorted_image

def _build_given_mask(board: np.ndarray,
                      given_mask: np.ndarray | None,
                      starters_ratio: float | None,
                      starters_count: int | None,
                      starters_seed: int | None) -> np.ndarray:
    """
    Returns a boolean mask (shape NxN). True = starter ('given') digit.

    Priority:
      1) If given_mask is provided, use it as-is (coerced to bool).
      2) Else sample a subset of nonzero cells by count or ratio.
    """
    N = board.shape[0]
    mask = np.zeros((N, N), dtype=bool)

    if given_mask is not None:
        gm = np.asarray(given_mask)
        if gm.shape != (N, N):
            raise ValueError(f"given_mask shape {gm.shape} != {(N,N)}")
        return gm.astype(bool)

    # Auto-sample starters among currently nonzero cells (if no nonzero, sample nothing)
    coords = [(r, c) for r in range(N) for c in range(N) if int(board[r, c]) > 0]
    if not coords:
        return mask

    rng = np.random.default_rng(starters_seed)

    if starters_count is not None:
        k = max(0, min(starters_count, len(coords)))
        choose = rng.choice(len(coords), size=k, replace=False)
    else:
        # Default ratio if nothing provided
        p = 0.35 if (starters_ratio is None) else float(starters_ratio)
        p = min(max(p, 0.0), 1.0)
        k = int(round(p * len(coords)))
        choose = rng.choice(len(coords), size=k, replace=False) if k > 0 else []

    for idx in np.atleast_1d(choose):
        r, c = coords[int(idx)]
        mask[r, c] = True
    return mask

def _valid_candidates(board: np.ndarray, r: int, c: int) -> list[int]:
    """Return Sudoku-valid digits for (r,c) using row/col/block constraints."""
    N = board.shape[0]
    n = int(math.isqrt(N))
    if int(board[r, c]) != 0:
        return []
    used = set(board[r, :].astype(int)) | set(board[:, c].astype(int))
    br, bc = (r // n) * n, (c // n) * n
    used |= set(board[br:br+n, bc:bc+n].astype(int).ravel())
    return [d for d in range(1, N+1) if d not in used]

def _synthesize_prior_from_neighbors(
    board: np.ndarray,
    starters: np.ndarray,                 # bool mask: True = starter
    *,
    groups_count: int | None = None,      # exact number of groups; overrides ratio
    groups_ratio: float = 0.20,           # ~20% of eligible cells become group seeds
    group_sizes: tuple[int, ...] = (2),# group size choices (including the seed)
    allow_diagonal: bool = False,         # rook adjacencies by default
    seed: int | None = None
) -> dict[tuple[int,int], list[int]]:
    """
    Build prior_options by choosing random adjacent pairs/triples among NON-STARTER filled cells.
    Returns: {(r,c): [digits]} where list is the other cells' digits in the same group.
    """
    rng = np.random.default_rng(seed)
    N = board.shape[0]

    # eligible cells: filled AND not a starter
    elig = [(int(r), int(c)) for r in range(N) for c in range(N)
            if int(board[r, c]) > 0 and not bool(starters[r, c])]
    if not elig:
        return {}

    # decide how many seeds
    if groups_count is None:
        k = int(round(len(elig) * max(0.0, min(1.0, groups_ratio))))
        groups_count = max(0, k)

    # shuffle copy
    elig_shuf = elig.copy()
    rng.shuffle(elig_shuf)
    seeds = elig_shuf[:groups_count]

    # neighbor directions
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if allow_diagonal:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]

    prior: dict[tuple[int,int], set[int]] = {}

    for (sr, sc) in seeds:
        # collect neighbors that are eligible too
        neigh: list[tuple[int,int]] = []
        for dr, dc in dirs:
            r, c = sr + dr, sc + dc
            if 0 <= r < N and 0 <= c < N and int(board[r, c]) > 0 and not bool(starters[r, c]):
                neigh.append((int(r), int(c)))
        if not neigh:
            continue

        # choose a group size (2 or 3) but cap by available neighbors
        gsize = int(rng.choice(group_sizes))
        gsize = min(1 + len(neigh), gsize)   # 1 seed + up to len(neigh)
        if gsize <= 1:
            continue

        # pick companions by INDEX to avoid array-of-tuples issues
        idxs = rng.choice(len(neigh), size=gsize-1, replace=False)
        idxs = np.atleast_1d(idxs).tolist()
        companions = [neigh[i] for i in idxs]

        group = [(int(sr), int(sc))] + [(int(r), int(c)) for (r, c) in companions]

            # ... after you computed `group = [(sr,sc)] + companions` ...

        # Collect the group's digits (positives only)
        group_digits = {int(board[rr, cc]) for (rr, cc) in group if int(board[rr, cc]) > 0}

        # Map every position in the group to the FULL set (includes its own digit)
        for pos in group:
            s = prior.setdefault(pos, set())
            for d in group_digits:
                s.add(d)

# finalize to sorted lists
    return {pos: sorted(list(ds)) for pos, ds in prior.items()}

def _render_inline_candidates(generator, opts, color, size, seed_base=0):
    """Render all candidate digits in one tight horizontal row (RGBA, no gaps)."""
    tiles = []
    for j, d in enumerate(opts):
        if generator is not None:
            im = generator.generate(
                digit=int(d), size=size, color=color,
                rng_seed=seed_base + 17 * j + int(d) * 101
            )
        else:
            fmini = ImageFont.load_default(size=size)
            tb = ImageDraw.Draw(Image.new("RGB", (1,1))).textbbox((0,0), str(d), font=fmini)
            w, h = tb[2]-tb[0], tb[3]-tb[1]
            im = Image.new("RGBA", (w, h), (255,255,255,0))
            ImageDraw.Draw(im).text((0,0), str(d), font=fmini, fill=color + (255,))
        # make slightly bolder & then crop transparent margins
        arr = np.array(im, dtype=np.uint8)
        if arr.shape[-1] == 4:
            arr[:, :, 3] = np.clip(arr[:, :, 3].astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr, mode="RGBA")
        tiles.append(_tight_crop_rgba(im))

    # ZERO kerning between cropped tiles
    gap = 0
    total_w = sum(t.width for t in tiles) + gap * max(0, len(tiles) - 1)
    max_h  = max(t.height for t in tiles)
    strip  = Image.new("RGBA", (total_w, max_h), (255, 255, 255, 0))
    x = 0
    for t in tiles:
        strip.paste(t, (x, (max_h - t.height) // 2), t)
        x += t.width + gap
    return strip

def _tight_crop_rgba(im: Image.Image, alpha_thresh: int = 8) -> Image.Image:
    """Trim transparent margins from an RGBA image."""
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    a = np.array(im)[:, :, 3]
    ys, xs = np.where(a > alpha_thresh)
    if len(xs) == 0 or len(ys) == 0:
        return im  # nothing to crop
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return im.crop((x0, y0, x1, y1))



def _draw_wobbly_circle(draw, bbox, color, *, width=1, seed=None):
    """Fine, hand-drawn looking outline tuned for small bubbles."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bbox
    r  = (x1 - x0) / 2.0
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    passes = 2              # fewer passes → cleaner, fine-liner look
    k = 56                  # enough points for smooth circle
    wobble_amp = 0.035      # gentler wobble
    jitter_px  = 0.35       # tiny vertex jitter

    for _ in range(passes):
        pts = []
        for i in range(k + 1):
            t  = 2 * math.pi * i / k
            rr = r * (1 + rng.uniform(-wobble_amp, wobble_amp))
            x  = cx + rr * math.cos(t) + rng.normal(0, jitter_px)
            y  = cy + rr * math.sin(t) + rng.normal(0, jitter_px)
            pts.append((x, y))
        draw.line(pts, fill=color, width=width)
