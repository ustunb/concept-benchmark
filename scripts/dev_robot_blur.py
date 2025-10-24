from concept_benchmark.synthetic.helper.robot_draw import (
    draw_robot,
    image_to_numpy_and_pillow,
    blur_parts,
)
from pathlib import Path
from tempfile import NamedTemporaryFile
from IPython.display import Image as IPyImage, display

# import cProfile, pstats, io
# profiler = cProfile.Profile()
# profiler.enable()
# # blur only the body region and preview
# profiler.disable()
# s = io.StringIO()
# pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(20)
# print(s.getvalue())


hand_shape = [
    "round_circle",
    "round_oval",
    "round_oval2",
    "edgy_triangle",
    "edgy_square",
    "edgy_trapezoid",
]

feats = {
    'foot_shape': 'flat_lshaped',
}

for h in hand_shape:
    feats['hand_shape'] = h
    rbt = draw_robot(filetype='png', **feats)
    blurred = blur_parts(
        rbt,
        parts=("hands",),
        radius=8.0,
        mask_mode="uniform_rect",
        **feats
        # expand_mask_px=25,
        # feather_mask_px=3,
    )
    display(blurred)
