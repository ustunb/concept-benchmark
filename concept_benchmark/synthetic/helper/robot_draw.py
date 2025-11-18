"""
Robot feature taxonomy and drawing utilities for synthetic robot images.
"""

import cairo
import numpy as np
import pero
from PIL import Image as PILImage
from PIL import ImageFilter as PILImageFilter

from .utils import generate_color_schemes

# from itertools import combinations_with_replacement as cwr
# COLOR_LIST = [c.name for c in pero.colors]
# COLOR_SCHEMES = [(w, v) for (w, v) in cwr(COLOR_LIST, 2) if w != v]
ROBOT_TYPES = ("glorp", "drent")

ALL_ROBOT_FEATURES = {
    "foot_shape": (
        "flat_trapezoid",
        "flat_rounded",
        "flat_square",
        "flat_5sided",
        "flat_lshaped",
        "pointy_trapezoid",
        "pointy_rounded",
        "pointy_square",
        "pointy_3sided",
        "pointy_4sided",
    ),
    "body_shape": ("square", "round"),  # no subtypes (could add)
    "head_shape": ("square", "round"),  # no subtypes (could add)
    #
    'has_elbows': ('false', 'true'),  # all round
    "has_knees": ("false", "true"),
    "has_antennae": ["false", "true"],
    "ears_shape": ("square", "triangle"),
    "mouth_type": ("closed", "open"),
    "hand_shape": (
        "round_circle",
        "round_oval",
        "round_oval2",
        "edgy_triangle",
        "edgy_square",
        "edgy_trapezoid",
    ),
}

COLOR_SCHEMES = generate_color_schemes(
    shuffle=True, random_seed=123456, include_flipped=False
)


DEFAULT_ROBOT_FEATURES = {
    #
    "head_shape": "square",
    # 'head_subtype_choice': 'default',
    #
    "body_shape": "square",
    # 'body_subtype_choice': 'default',
    #
    # foot
    "foot_shape": "flat",
    "foot_subtype_choice": "default",
    #
    "has_knees": True,
    #'has_elbows': True,
    #
    "color_scheme": COLOR_SCHEMES[0],
}

FOOT_SUBTYPES = {
    "flat": ["flat_4sided", "flat_5sided", "flat_lshaped"],
    "pointy": ["pointy_3sided", "pointy_4sided", "pointy_6sided"],
}


def draw_robot(filetype="svg", col_scheme_add=0, width=600, height=600, **kwargs):
    # set unspecified features using default values
    features = dict(DEFAULT_ROBOT_FEATURES)
    features.update(kwargs)

    # canvas
    canvas = (
        pero.svg.SVGCanvas(width=width, height=height)
        if filetype == "svg"
        else pero.Image(width=width, height=height)
    )
    if filetype != "svg":
        canvas.fill(pero.colors.White)
    canvas.line_cap = pero.ROUND
    canvas.line_join = pero.ROUND
    canvas.line_color = pero.colors.Black
    canvas.line_width = 2

    # colors
    if isinstance(features["color_scheme"], int):
        color_scheme_id = np.mod(
            features["color_scheme"] + col_scheme_add, len(COLOR_SCHEMES)
        )
        color_left, color_right = COLOR_SCHEMES[color_scheme_id]
        if kwargs.get("verbose", False):
            print(np.mod(features["color_scheme"] + col_scheme_add, len(COLOR_SCHEMES)))
            print(color_left, color_right)
    elif isinstance(features["color_scheme"], tuple):
        color_left, color_right = features["color_scheme"]

    # lengths
    r = canvas.height / 12.0

    # anchor points
    y_top = 2 * r
    x_mid = 0.5 * canvas.width

    # standard elements
    line = pero.Line(line_color=pero.colors.Black, line_width=1)
    ray = pero.Ray(line_color=pero.colors.Black, line_width=1)

    # face lengths
    head_shape = features["head_shape"]
    head_height = 1.75 * r
    head_width = 1.75 * r
    y_top_face = y_top

    ############################################################################
    # arms, hands, elbows, body
    ############################################################################
    y_top += head_height

    body_width = 3.5 * r
    body_height = 4.0 * r
    x_left = x_mid - 0.5 * body_width
    x_right = x_mid + 0.5 * body_width
    y_arm = y_top + 0.33 * body_height

    arm_length = 0.75 * body_width
    hand_length = 0.25 * body_width

    # arms
    ray.draw(canvas, x=x_left, y=y_arm, length=arm_length, angle=pero.rads(180))

    ray.draw(canvas, x=x_right, y=y_arm, length=arm_length, angle=pero.rads(0))

    # elbows
    if "has_elbows" in features.keys():
        elbow_size = 0.1 * r if width < 120 or features["has_elbows"] == "true" else 0
        elbow = pero.Ellipse(y=y_arm, width=elbow_size, height=elbow_size)
        elbow.draw(canvas, x=x_left - 0.5 * arm_length, fill_color=color_left)
        elbow.draw(canvas, x=x_right + 0.5 * arm_length, fill_color=color_right)

    # hands
    hand_shape = features.get("hand_shape", "round_circle")
    hand_type, hand_subtype = hand_shape.split("_")[0], hand_shape.split("_")[1]
    hand_x_left = x_left - arm_length
    hand_x_right = x_right + arm_length
    hand_y = y_arm
    hand_size = 0.6 * r

    if hand_type == "round":
        if hand_subtype == "circle":
            # Round circle hands - centered at arm end
            hand = pero.Ellipse(width=hand_size, height=hand_size)
            hand.draw(canvas, x=hand_x_left, y=hand_y, fill_color=color_left)
            hand.draw(canvas, x=hand_x_right, y=hand_y, fill_color=color_right)

        elif hand_subtype == "oval":
            # Round oval hands (widened horizontally) - centered at arm end
            hand = pero.Ellipse(width=hand_size * 1.5, height=hand_size)
            hand.draw(canvas, x=hand_x_left, y=hand_y, fill_color=color_left)
            hand.draw(canvas, x=hand_x_right, y=hand_y, fill_color=color_right)

        elif hand_subtype == "oval2":
            # Round oval2 hands (widened vertically) - centered at arm end
            hand = pero.Ellipse(width=hand_size, height=hand_size * 1.5)
            hand.draw(canvas, x=hand_x_left, y=hand_y, fill_color=color_left)
            hand.draw(canvas, x=hand_x_right, y=hand_y, fill_color=color_right)

    elif hand_type == "edgy":
        if hand_subtype == "triangle":
            # Edgy triangle hands - tip facing outward away from body
            hand = pero.Polygon(line_color=pero.colors.Black)

            # Left triangle hand (base at arm, tip pointing left)
            p_left = (
                (hand_x_left, hand_y - hand_size / 2),  # top of base
                (hand_x_left, hand_y + hand_size / 2),  # bottom of base
                (hand_x_left - hand_size, hand_y),  # tip pointing left
            )
            hand.draw(canvas, points=p_left, fill_color=color_left)

            # Right triangle hand (base at arm, tip pointing right)
            p_right = (
                (hand_x_right, hand_y - hand_size / 2),  # top of base
                (hand_x_right, hand_y + hand_size / 2),  # bottom of base
                (hand_x_right + hand_size, hand_y),  # tip pointing right
            )
            hand.draw(canvas, points=p_right, fill_color=color_right)

        elif hand_subtype == "square":
            # Edgy square hands - centered at arm end
            hand = pero.Rect(width=hand_size, height=hand_size)
            hand.draw(
                canvas,
                x=hand_x_left - hand_size / 2,
                y=hand_y - hand_size / 2,
                fill_color=color_left,
            )
            hand.draw(
                canvas,
                x=hand_x_right - hand_size / 2,
                y=hand_y - hand_size / 2,
                fill_color=color_right,
            )

        elif hand_subtype == "trapezoid":
            # Edgy trapezoid hands - shorter base at arm end, wider base pointing outward
            hand = pero.Polygon(line_color=pero.colors.Black)

            # Left trapezoid hand (wider base pointing left)
            p_left = (
                (hand_x_left, hand_y - hand_size / 4),  # top of shorter base
                (hand_x_left, hand_y + hand_size / 4),  # bottom of shorter base
                (
                    hand_x_left - hand_size,
                    hand_y + hand_size / 2,
                ),  # bottom of wider base
                (hand_x_left - hand_size, hand_y - hand_size / 2),  # top of wider base
            )
            hand.draw(canvas, points=p_left, fill_color=color_left)

            # Right trapezoid hand (wider base pointing right)
            p_right = (
                (hand_x_right, hand_y - hand_size / 4),  # top of shorter base
                (hand_x_right, hand_y + hand_size / 4),  # bottom of shorter base
                (
                    hand_x_right + hand_size,
                    hand_y + hand_size / 2,
                ),  # bottom of wider base
                (hand_x_right + hand_size, hand_y - hand_size / 2),  # top of wider base
            )
            hand.draw(canvas, points=p_right, fill_color=color_right)

    # body
    y_body = y_top

    ############################################################################
    # legs, knees, feet
    ############################################################################
    y_top += body_height
    foot_gap = 1.0 * r
    foot_width = (body_width - 2.0 * foot_gap) / 2.0
    foot_height = foot_width
    x_left = x_mid - 0.5 * foot_gap - 0.5 * foot_width
    x_right = x_mid + 0.5 * foot_gap + 0.5 * foot_width

    # legs
    leg_height = 0.75 * (body_height - foot_height)
    line.draw(canvas, x1=x_left, x2=x_left, y1=y_top * 0.9, y2=y_top + leg_height)
    line.draw(canvas, x1=x_right, x2=x_right, y1=y_top * 0.9, y2=y_top + leg_height)

    # knees
    if "has_knees" in features.keys():# and features["has_knees"] == "true":
        #knee_size = round(0.05 * r)
        knee_size = 0.1 * r if width < 120 or features["has_knees"] == "true" else 0

        knee = pero.Ellipse(
            x=x_left, y=y_top + 0.5 * leg_height, width=knee_size, height=knee_size
        )
        knee.draw(canvas, x=x_left, fill_color=color_left)
        knee.draw(canvas, x=x_right, fill_color=color_right)

    y_top += leg_height
    foot_subtype = features["foot_shape"]
    if foot_subtype in FOOT_SUBTYPES.keys():
        if features["foot_subtype_choice"] == "default":
            foot_subtype = "%s" % FOOT_SUBTYPES[foot_subtype][0]

    # orient = str(features.get("foot_orientation", "")).lower()
    # if orient in ("vertex", "side"):
    #     if foot_subtype.startswith("flat_") and orient == "vertex":
    #         suf = foot_subtype.split("_", 1)[1]
    #         cand = f"pointy_{suf}"
    #         if cand in ALL_ROBOT_FEATURES["foot_shape"]:
    #             foot_subtype = cand
    #     elif foot_subtype.startswith("pointy_") and orient == "side":
    #         suf = foot_subtype.split("_", 1)[1]
    #         cand = f"flat_{suf}"
    #         if cand in ALL_ROBOT_FEATURES["foot_shape"]:
    #             foot_subtype = cand

    # PAIR 1: trapezoid
    if foot_subtype == "flat_trapezoid":
        # Add rounded/curved top edges like rounded shape
        p_left = (
            (x_left - 0.45 * foot_width, y_top + 0.1 * foot_height),  # Start below top with curve
            (x_left - 0.3 * foot_width, y_top),  # Curve up
            (x_left, y_top),  # Top center
            (x_left + 0.3 * foot_width, y_top),  # Curve up
            (x_left + 0.45 * foot_width, y_top + 0.1 * foot_height),  # Start below top
            (x_left + 0.15 * foot_width, y_top + foot_height),  # Narrow bottom
            (x_left + 0.03 * foot_width, y_top + foot_height),
            (x_left - 0.03 * foot_width, y_top + foot_height),
            (x_left - 0.15 * foot_width, y_top + foot_height),
        )

        p_right = (
            (x_right - 0.45 * foot_width, y_top + 0.1 * foot_height),
            (x_right - 0.3 * foot_width, y_top),
            (x_right, y_top),
            (x_right + 0.3 * foot_width, y_top),
            (x_right + 0.45 * foot_width, y_top + 0.1 * foot_height),
            (x_right + 0.15 * foot_width, y_top + foot_height),
            (x_right + 0.03 * foot_width, y_top + foot_height),
            (x_right - 0.03 * foot_width, y_top + foot_height),
            (x_right - 0.15 * foot_width, y_top + foot_height),
        )

    elif foot_subtype == "pointy_trapezoid":
        # Add same rounded top as flat_trapezoid
        p_left = (
            (x_left - 0.45 * foot_width, y_top + 0.1 * foot_height),
            (x_left - 0.3 * foot_width, y_top),
            (x_left, y_top),
            (x_left + 0.3 * foot_width, y_top),
            (x_left + 0.45 * foot_width, y_top + 0.1 * foot_height),
            (x_left + 0.18 * foot_width, y_top + 0.88 * foot_height),
            (x_left, y_top + foot_height),  # Point
            (x_left - 0.18 * foot_width, y_top + 0.88 * foot_height),
        )

        p_right = (
            (x_right - 0.45 * foot_width, y_top + 0.1 * foot_height),
            (x_right - 0.3 * foot_width, y_top),
            (x_right, y_top),
            (x_right + 0.3 * foot_width, y_top),
            (x_right + 0.45 * foot_width, y_top + 0.1 * foot_height),
            (x_right + 0.18 * foot_width, y_top + 0.88 * foot_height),
            (x_right, y_top + foot_height),
            (x_right - 0.18 * foot_width, y_top + 0.88 * foot_height),
        )

    # Make square more rounded too
    elif foot_subtype == "flat_square":
        # Add rounded top corners
        p_left = (
            (x_left - 0.5 * foot_width, y_top + 0.15 * foot_height),  # Round corner
            (x_left - 0.4 * foot_width, y_top + 0.05 * foot_height),
            (x_left - 0.2 * foot_width, y_top),
            (x_left + 0.2 * foot_width, y_top),
            (x_left + 0.4 * foot_width, y_top + 0.05 * foot_height),
            (x_left + 0.5 * foot_width, y_top + 0.15 * foot_height),  # Round corner
            (x_left + 0.5 * foot_width, y_top + 0.75 * foot_height),
            (x_left + 0.2 * foot_width, y_top + 0.92 * foot_height),
            (x_left + 0.15 * foot_width, y_top + foot_height),
            (x_left + 0.03 * foot_width, y_top + foot_height),
            (x_left - 0.03 * foot_width, y_top + foot_height),
            (x_left - 0.15 * foot_width, y_top + foot_height),
            (x_left - 0.2 * foot_width, y_top + 0.92 * foot_height),
            (x_left - 0.5 * foot_width, y_top + 0.75 * foot_height),
        )

        p_right = (
            (x_right - 0.5 * foot_width, y_top + 0.15 * foot_height),
            (x_right - 0.4 * foot_width, y_top + 0.05 * foot_height),
            (x_right - 0.2 * foot_width, y_top),
            (x_right + 0.2 * foot_width, y_top),
            (x_right + 0.4 * foot_width, y_top + 0.05 * foot_height),
            (x_right + 0.5 * foot_width, y_top + 0.15 * foot_height),
            (x_right + 0.5 * foot_width, y_top + 0.75 * foot_height),
            (x_right + 0.2 * foot_width, y_top + 0.92 * foot_height),
            (x_right + 0.15 * foot_width, y_top + foot_height),
            (x_right + 0.03 * foot_width, y_top + foot_height),
            (x_right - 0.03 * foot_width, y_top + foot_height),
            (x_right - 0.15 * foot_width, y_top + foot_height),
            (x_right - 0.2 * foot_width, y_top + 0.92 * foot_height),
            (x_right - 0.5 * foot_width, y_top + 0.75 * foot_height),
        )

    elif foot_subtype == "pointy_square":
        # Match flat_square rounded top
        p_left = (
            (x_left - 0.5 * foot_width, y_top + 0.15 * foot_height),
            (x_left - 0.4 * foot_width, y_top + 0.05 * foot_height),
            (x_left - 0.2 * foot_width, y_top),
            (x_left + 0.2 * foot_width, y_top),
            (x_left + 0.4 * foot_width, y_top + 0.05 * foot_height),
            (x_left + 0.5 * foot_width, y_top + 0.15 * foot_height),
            (x_left + 0.5 * foot_width, y_top + 0.75 * foot_height),
            (x_left + 0.2 * foot_width, y_top + 0.92 * foot_height),
            (x_left, y_top + foot_height),  # Point
            (x_left - 0.2 * foot_width, y_top + 0.92 * foot_height),
            (x_left - 0.5 * foot_width, y_top + 0.75 * foot_height),
        )

        p_right = (
            (x_right - 0.5 * foot_width, y_top + 0.15 * foot_height),
            (x_right - 0.4 * foot_width, y_top + 0.05 * foot_height),
            (x_right - 0.2 * foot_width, y_top),
            (x_right + 0.2 * foot_width, y_top),
            (x_right + 0.4 * foot_width, y_top + 0.05 * foot_height),
            (x_right + 0.5 * foot_width, y_top + 0.15 * foot_height),
            (x_right + 0.5 * foot_width, y_top + 0.75 * foot_height),
            (x_right + 0.2 * foot_width, y_top + 0.92 * foot_height),
            (x_right, y_top + foot_height),
            (x_right - 0.2 * foot_width, y_top + 0.92 * foot_height),
            (x_right - 0.5 * foot_width, y_top + 0.75 * foot_height),
        )

    # PAIR 2: rounded
    elif foot_subtype == "flat_rounded":
        # Rounded top, tapers to narrow flat bottom (trapezoid-like)
        p_left = (
            (x_left - 0.5 * foot_width, y_top + 0.5 * foot_height),
            (x_left - 0.35 * foot_width, y_top + 0.15 * foot_height),
            (x_left, y_top),
            (x_left + 0.35 * foot_width, y_top + 0.15 * foot_height),
            (x_left + 0.5 * foot_width, y_top + 0.5 * foot_height),
            (x_left + 0.18 * foot_width, y_top + 0.92 * foot_height),  # Taper in
            (x_left + 0.15 * foot_width, y_top + foot_height),  # Narrow bottom
            (x_left + 0.03 * foot_width, y_top + foot_height),  # Tiny flat
            (x_left - 0.03 * foot_width, y_top + foot_height),
            (x_left - 0.15 * foot_width, y_top + foot_height),  # Narrow bottom
            (x_left - 0.18 * foot_width, y_top + 0.92 * foot_height),  # Taper in
        )

        p_right = (
            (x_right - 0.5 * foot_width, y_top + 0.5 * foot_height),
            (x_right - 0.35 * foot_width, y_top + 0.15 * foot_height),
            (x_right, y_top),
            (x_right + 0.35 * foot_width, y_top + 0.15 * foot_height),
            (x_right + 0.5 * foot_width, y_top + 0.5 * foot_height),
            (x_right + 0.18 * foot_width, y_top + 0.92 * foot_height),
            (x_right + 0.15 * foot_width, y_top + foot_height),
            (x_right + 0.03 * foot_width, y_top + foot_height),
            (x_right - 0.03 * foot_width, y_top + foot_height),
            (x_right - 0.15 * foot_width, y_top + foot_height),
            (x_right - 0.18 * foot_width, y_top + 0.92 * foot_height),
        )

    elif foot_subtype == "pointy_rounded":
        # Keep similar structure to pointy_trapezoid
        p_left = (
            (x_left - 0.5 * foot_width, y_top + 0.3 * foot_height),
            (x_left - 0.4 * foot_width, y_top + 0.1 * foot_height),
            (x_left - 0.2 * foot_width, y_top),
            (x_left, y_top),
            (x_left + 0.2 * foot_width, y_top),
            (x_left + 0.4 * foot_width, y_top + 0.1 * foot_height),
            (x_left + 0.5 * foot_width, y_top + 0.3 * foot_height),
            (x_left + 0.18 * foot_width, y_top + 0.88 * foot_height),  # Match trapezoid angle
            (x_left, y_top + foot_height),
            (x_left - 0.18 * foot_width, y_top + 0.88 * foot_height),
        )

        p_right = (
            (x_right - 0.5 * foot_width, y_top + 0.3 * foot_height),
            (x_right - 0.4 * foot_width, y_top + 0.1 * foot_height),
            (x_right - 0.2 * foot_width, y_top),
            (x_right, y_top),
            (x_right + 0.2 * foot_width, y_top),
            (x_right + 0.4 * foot_width, y_top + 0.1 * foot_height),
            (x_right + 0.5 * foot_width, y_top + 0.3 * foot_height),
            (x_right + 0.18 * foot_width, y_top + 0.88 * foot_height),
            (x_right, y_top + foot_height),
            (x_right - 0.18 * foot_width, y_top + 0.88 * foot_height),
        )

    elif foot_subtype == "flat_5sided":
        p_left = (
            (x_left, y_top),
            (x_left - 0.5 * foot_width, y_top + 0.3 * foot_height),
            (x_left - 0.5 * foot_width, y_top + foot_height),
            (x_left + 0.5 * foot_width, y_top + foot_height),
            (x_left + 0.5 * foot_width, y_top + 0.3 * foot_height),
        )

        p_right = (
            (x_right, y_top),
            (x_right - 0.5 * foot_width, y_top + 0.3 * foot_height),
            (x_right - 0.5 * foot_width, y_top + foot_height),
            (x_right + 0.5 * foot_width, y_top + foot_height),
            (x_right + 0.5 * foot_width, y_top + 0.3 * foot_height),
        )

    elif foot_subtype == "flat_lshaped":
        p_left = (
            (x_left - 0.35 * foot_width, y_top),
            (x_left - 0.35 * foot_width, y_top + 0.35 * foot_height),
            (x_left - 1.00 * foot_width, y_top + 0.35 * foot_height),
            (x_left - 1.00 * foot_width, y_top + 1.0 * foot_height),
            (x_left + 0.35 * foot_width, y_top + 1.0 * foot_height),
            (x_left + 0.35 * foot_width, y_top),
        )

        p_right = (
            (x_right + 0.35 * foot_width, y_top),
            (x_right + 0.35 * foot_width, y_top + 0.35 * foot_height),
            (x_right + 1.00 * foot_width, y_top + 0.35 * foot_height),
            (x_right + 1.00 * foot_width, y_top + 1.0 * foot_height),
            (x_right - 0.35 * foot_width, y_top + 1.0 * foot_height),
            (x_right - 0.35 * foot_width, y_top),
        )


    elif foot_subtype == "pointy_3sided":
        p_left = (
            (x_left - 0.5 * foot_width, y_top),
            (x_left + 0.5 * foot_width, y_top),
            (x_left, y_top + foot_height),
        )

        p_right = (
            (x_right - 0.5 * foot_width, y_top),
            (x_right + 0.5 * foot_width, y_top),
            (x_right, y_top + foot_height),
        )

    elif foot_subtype == "pointy_4sided":
        p_left = (
            (x_left, y_top),
            (x_left - 0.5 * foot_width, y_top + 0.5 * foot_height),
            (x_left, y_top + foot_height),
            (x_left + 0.5 * foot_width, y_top + 0.5 * foot_height),
        )

        p_right = (
            (x_right, y_top),
            (x_right - 0.5 * foot_width, y_top + 0.5 * foot_height),
            (x_right, y_top + foot_height),
            (x_right + 0.5 * foot_width, y_top + 0.5 * foot_height),
        )

    else:
        raise TypeError("invalid foot type")

    ############################################################################
    # body (drawn last)
    ############################################################################
    body_shape = features["body_shape"]
    if body_shape == "round":
        body = pero.Arc(
            x=x_mid,
            y=y_body + 0.5 * body_height,
            radius=0.57 * body_width,
            line_color=pero.colors.Black,
        )

        body.draw(
            canvas,
            start_angle=pero.rads(90),
            end_angle=pero.rads(-90),
            fill_color=color_left,
        )
        body.draw(
            canvas,
            start_angle=pero.rads(-90),
            end_angle=pero.rads(90),
            fill_color=color_right,
        )

    elif body_shape == "square":
        body = pero.Rect(x=x_mid - 0.5 * body_width, y=y_body, height=body_height)
        body.draw(canvas, line_width=0, fill_color=color_left, width=0.5 * body_width)
        body.draw(
            canvas,
            line_width=0,
            fill_color=color_right,
            width=0.5 * body_width,
            x=x_mid,
        )
        body.draw(canvas, fill_color=None, width=body_width)

    foot = pero.Polygon(line_color=pero.colors.Black)
    foot.draw(canvas, points=p_left, fill_color=color_left)
    foot.draw(canvas, points=p_right, fill_color=color_right)

    ############################################################################
    # face, eyes, antenna
    ############################################################################

    # antennae
    if "has_antennae" in features.keys() and features["has_antennae"] == "true":
        antenna_length = 1.5 * r if head_shape == "round" else 1.75 * r
        antenna_width = max(1, int(r * 0.08))

        antenna_ray = pero.Ray(line_color=pero.colors.Black, line_width=antenna_width)
        antenna_ray.draw(
            canvas,
            x=x_mid,
            y=y_top_face + head_height / 2,
            angle=pero.rads(180 + 60),
            length=antenna_length,
        )

        antenna_ray.draw(
            canvas,
            x=x_mid,
            y=y_top_face + head_height / 2,
            angle=pero.rads(-60),
            length=antenna_length,
        )

    if head_shape == "round":
        face = pero.Arc(x=x_mid, y=y_top_face + 0.9 * r, radius=0.5 * head_width)
        face.draw(
            canvas,
            start_angle=pero.rads(90),
            end_angle=pero.rads(270),
            fill_color=color_left,
        )
        face.draw(
            canvas,
            start_angle=pero.rads(-90),
            end_angle=pero.rads(90),
            fill_color=color_right,
        )

    elif head_shape == "square":
        face = pero.Rect(y=y_top_face, height=head_height)
        face.draw(
            canvas,
            x=x_mid - 0.5 * head_width,
            width=0.5 * head_width,
            line_width=0,
            fill_color=color_left,
        )
        face.draw(
            canvas,
            x=x_mid,
            width=0.5 * head_width,
            line_width=0,
            fill_color=color_right,
        )
        face.draw(
            canvas,
            x=x_mid - 0.5 * head_width,
            width=head_width,
            height=head_height,
            fill_color=None,
        )

    # eyes
    eye = pero.Ellipse(
        fill_color=pero.colors.Black,
        y=y_top_face + (3.0 / 8.0) * head_height,
        height=0.2 * r,
        width=0.2 * r,
    )
    eye.draw(canvas, x=x_mid - 0.3 * r)
    eye.draw(canvas, x=x_mid + 0.3 * r)

    # ears
    if "ears_shape" in features:
        ear_shape = features["ears_shape"]
        ear_size = 0.4 * r
        ear_x_left = x_mid - 0.5 * head_width
        ear_x_right = x_mid + 0.5 * head_width
        ear_y = y_top_face + head_height / 2  # middle of head

        if ear_shape == "square":
            ear = pero.Rect(width=ear_size, height=ear_size)
            # Left ear (extending left from head)
            ear.draw(
                canvas,
                x=ear_x_left - ear_size,
                y=ear_y - ear_size / 2,
                fill_color=color_left,
            )
            # Right ear (extending right from head)
            ear.draw(
                canvas, x=ear_x_right, y=ear_y - ear_size / 2, fill_color=color_right
            )

        elif ear_shape == "triangle":
            ear = pero.Polygon(line_color=pero.colors.Black)

            # Left triangle ear (base at head, tip pointing left)
            p_left = (
                (ear_x_left, ear_y - ear_size / 2),  # top of base
                (ear_x_left, ear_y + ear_size / 2),  # bottom of base
                (ear_x_left - ear_size, ear_y),  # tip pointing left
            )
            ear.draw(canvas, points=p_left, fill_color=color_left)

            # Right triangle ear (base at head, tip pointing right)
            p_right = (
                (ear_x_right, ear_y - ear_size / 2),  # top of base
                (ear_x_right, ear_y + ear_size / 2),  # bottom of base
                (ear_x_right + ear_size, ear_y),  # tip pointing right
            )
            ear.draw(canvas, points=p_right, fill_color=color_right)

    if "mouth_type" in features:
        mouth_type = features["mouth_type"]
        mouth_width = 0.4 * head_width
        mouth_x = x_mid - mouth_width / 2
        mouth_y = y_top_face + (5.0 / 8.0) * head_height  # below the eyes

        if mouth_type == "closed":
            # Closed mouth - thinner filled rectangle
            mouth_height = 0.05 * r
            mouth = pero.Rect(
                x=mouth_x,
                y=mouth_y,
                width=mouth_width,
                height=mouth_height,
                fill_color=pero.colors.Black,
            )
            mouth.draw(canvas)

        elif mouth_type == "open":
            # Open mouth - taller rectangle outline with vertical grills inside
            mouth_height = 0.2 * r
            mouth_outline = pero.Rect(
                x=mouth_x,
                y=mouth_y,
                width=mouth_width,
                height=mouth_height,
                fill_color=pero.colors.White,
                line_color=pero.colors.Black,
            )
            mouth_outline.draw(canvas)

            # Add vertical grills inside the mouth
            grill_line = pero.Line(line_color=pero.colors.Black, line_width=1)
            num_grills = 4
            grill_spacing = mouth_width / (num_grills + 1)

            for i in range(1, num_grills + 1):
                grill_x = mouth_x + i * grill_spacing
                grill_line.draw(
                    canvas,
                    x1=grill_x,
                    y1=mouth_y,
                    x2=grill_x,
                    y2=mouth_y + mouth_height,
                )

    return canvas


def image_to_numpy_and_pillow(img):
    """Render a ``pero.Image`` into NumPy and Pillow representations."""

    from pero.backends.cairo.canvas import CairoCanvas

    width = int(img.width)
    height = int(img.height)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)

    canvas = CairoCanvas(context, width=width, height=height)
    img.draw(canvas)

    stride = surface.get_stride()
    buf = surface.get_data()

    array = np.frombuffer(buf, dtype=np.uint8)
    array = array.reshape((height, stride // 4, 4))
    array = array[:, :width, :]

    rgba = array[:, :, [2, 1, 0, 3]].copy()
    pil_image = PILImage.fromarray(rgba, mode="RGBA")

    return rgba, pil_image


def draw_robot_mask(
    width=600, height=600, parts=("body",), mode: str = "exact", **kwargs
):
    """Draw a binary mask (white on black) for selected robot parts.

    Currently supports: 'body', 'hands', 'feet'.
    """
    # features must match draw_robot defaults
    features = dict(DEFAULT_ROBOT_FEATURES)
    features.update(kwargs)

    canvas = pero.Image(width=width, height=height)
    canvas.fill(pero.colors.Black)
    canvas.line_width = 0

    # lengths and anchors (copy from draw_robot)
    r = canvas.height / 12.0
    y_top = 2 * r
    x_mid = 0.5 * canvas.width

    # head dims
    head_height = 1.75 * r
    head_width = 1.75 * r

    # advance past head to body
    y_top += head_height
    y_body = y_top

    # body dims
    body_width = 3.5 * r
    body_height = 4.0 * r
    x_left = x_mid - 0.5 * body_width
    x_right = x_mid + 0.5 * body_width

    if mode not in {"exact", "uniform_rect"}:
        raise ValueError("mask_mode must be 'exact' or 'uniform_rect'")

    # body mask
    if mode == "uniform_rect":
        if "body" in parts:
            half_w = 0.57 * body_width
            rect = pero.Rect(
                x=x_mid - half_w,
                y=y_body,
                width=2 * half_w,
                height=body_height,
            )
            rect.draw(canvas, fill_color=pero.colors.White, line_width=0)

        if "hands" in parts:
            y_arm = y_top + 0.33 * body_height
            arm_length = 0.75 * body_width
            hand_size = 0.6 * r
            hand_x_left = x_left - arm_length
            hand_x_right = x_right + arm_length

            margin_x = 1.1 * hand_size
            margin_y = hand_size

            # left hand rectangle
            rect_left = pero.Rect(
                x=hand_x_left - margin_x,
                y=y_arm - margin_y,
                width=margin_x + hand_size,
                height=2 * margin_y,
            )
            rect_left.draw(canvas, fill_color=pero.colors.White, line_width=0)

            # right hand rectangle
            rect_right = pero.Rect(
                x=hand_x_right - hand_size,
                y=y_arm - margin_y,
                width=margin_x + hand_size,
                height=2 * margin_y,
            )
            rect_right.draw(canvas, fill_color=pero.colors.White, line_width=0)

        if "feet" in parts:
            foot_gap = 1.0 * r
            foot_width = (body_width - 2.0 * foot_gap) / 2.0
            foot_height = foot_width
            x_left_foot = x_mid - 0.5 * foot_gap - 0.5 * foot_width
            x_right_foot = x_mid + 0.5 * foot_gap + 0.5 * foot_width
            leg_height = 0.75 * (body_height - foot_height)
            y_top_feet = y_body + body_height + leg_height

            x_min = x_left_foot - foot_width
            x_max = x_right_foot + foot_width
            y_min = y_top_feet
            y_max = y_top_feet + foot_height
            rect = pero.Rect(
                x=x_min,
                y=y_min,
                width=x_max - x_min,
                height=y_max - y_min,
            )
            rect.draw(canvas, fill_color=pero.colors.White, line_width=0)

        return canvas

    if "body" in parts:
        body_shape = features["body_shape"]
        if body_shape == "round":
            body = pero.Arc(
                x=x_mid,
                y=y_body + 0.5 * body_height,
                radius=0.57 * body_width,
                line_color=None,
            )
            # draw both halves filled white to cover full round body
            body.draw(
                canvas,
                start_angle=pero.rads(90),
                end_angle=pero.rads(-90),
                fill_color=pero.colors.White,
            )
            body.draw(
                canvas,
                start_angle=pero.rads(-90),
                end_angle=pero.rads(90),
                fill_color=pero.colors.White,
            )
        elif body_shape == "square":
            body = pero.Rect(
                x=x_mid - 0.5 * body_width,
                y=y_body,
                height=body_height,
            )
            body.draw(
                canvas,
                line_width=0,
                fill_color=pero.colors.White,
                width=body_width,
            )

    # hands mask
    if "hands" in parts:
        y_arm = y_top + 0.33 * body_height
        arm_length = 0.75 * body_width
        hand_shape = features.get("hand_shape", "round_circle")
        hand_type, hand_subtype = hand_shape.split("_")[0], hand_shape.split("_")[1]
        hand_x_left = x_left - arm_length
        hand_x_right = x_right + arm_length
        hand_y = y_arm
        hand_size = 0.6 * r

        if hand_type == "round":
            if hand_subtype == "circle":
                hand = pero.Ellipse(width=hand_size, height=hand_size)
                hand.draw(canvas, x=hand_x_left, y=hand_y, fill_color=pero.colors.White)
                hand.draw(
                    canvas, x=hand_x_right, y=hand_y, fill_color=pero.colors.White
                )
            elif hand_subtype == "oval":
                hand = pero.Ellipse(width=hand_size * 1.5, height=hand_size)
                hand.draw(canvas, x=hand_x_left, y=hand_y, fill_color=pero.colors.White)
                hand.draw(
                    canvas, x=hand_x_right, y=hand_y, fill_color=pero.colors.White
                )
            elif hand_subtype == "oval2":
                hand = pero.Ellipse(width=hand_size, height=hand_size * 1.5)
                hand.draw(canvas, x=hand_x_left, y=hand_y, fill_color=pero.colors.White)
                hand.draw(
                    canvas, x=hand_x_right, y=hand_y, fill_color=pero.colors.White
                )
        elif hand_type == "edgy":
            if hand_subtype == "triangle":
                hand = pero.Polygon(line_color=None)
                p_left = (
                    (hand_x_left, hand_y - hand_size / 2),
                    (hand_x_left, hand_y + hand_size / 2),
                    (hand_x_left - hand_size, hand_y),
                )
                hand.draw(canvas, points=p_left, fill_color=pero.colors.White)
                p_right = (
                    (hand_x_right, hand_y - hand_size / 2),
                    (hand_x_right, hand_y + hand_size / 2),
                    (hand_x_right + hand_size, hand_y),
                )
                hand.draw(canvas, points=p_right, fill_color=pero.colors.White)
            elif hand_subtype == "square":
                hand = pero.Rect(width=hand_size, height=hand_size)
                hand.draw(
                    canvas,
                    x=hand_x_left - hand_size / 2,
                    y=hand_y - hand_size / 2,
                    fill_color=pero.colors.White,
                )
                hand.draw(
                    canvas,
                    x=hand_x_right - hand_size / 2,
                    y=hand_y - hand_size / 2,
                    fill_color=pero.colors.White,
                )
            elif hand_subtype == "trapezoid":
                hand = pero.Polygon(line_color=None)
                p_left = (
                    (hand_x_left, hand_y - hand_size / 4),
                    (hand_x_left, hand_y + hand_size / 4),
                    (hand_x_left - hand_size, hand_y + hand_size / 2),
                    (hand_x_left - hand_size, hand_y - hand_size / 2),
                )
                hand.draw(canvas, points=p_left, fill_color=pero.colors.White)
                p_right = (
                    (hand_x_right, hand_y - hand_size / 4),
                    (hand_x_right, hand_y + hand_size / 4),
                    (hand_x_right + hand_size, hand_y + hand_size / 2),
                    (hand_x_right + hand_size, hand_y - hand_size / 2),
                )
                hand.draw(canvas, points=p_right, fill_color=pero.colors.White)

    # feet mask
    if "feet" in parts:
        y_top_feet = y_body + body_height
        foot_gap = 1.0 * r
        foot_width = (body_width - 2.0 * foot_gap) / 2.0
        foot_height = foot_width
        x_left_foot = x_mid - 0.5 * foot_gap - 0.5 * foot_width
        x_right_foot = x_mid + 0.5 * foot_gap + 0.5 * foot_width

        leg_height = 0.75 * (body_height - foot_height)
        y_top_feet += leg_height

        foot_subtype = features["foot_shape"]
        if foot_subtype in FOOT_SUBTYPES.keys():
            if features.get("foot_subtype_choice", "default") == "default":
                foot_subtype = "%s" % FOOT_SUBTYPES[foot_subtype][0]

        if foot_subtype == "flat_4sided":
            p_left = (
                (x_left_foot - 0.5 * foot_width, y_top_feet),
                (x_left_foot - 0.5 * foot_width, y_top_feet + foot_height),
                (x_left_foot + 0.5 * foot_width, y_top_feet + foot_height),
                (x_left_foot + 0.5 * foot_width, y_top_feet),
            )
            p_right = (
                (x_right_foot - 0.5 * foot_width, y_top_feet),
                (x_right_foot - 0.5 * foot_width, y_top_feet + foot_height),
                (x_right_foot + 0.5 * foot_width, y_top_feet + foot_height),
                (x_right_foot + 0.5 * foot_width, y_top_feet),
            )
        elif foot_subtype == "flat_5sided":
            p_left = (
                (x_left_foot, y_top_feet),
                (x_left_foot - 0.5 * foot_width, y_top_feet + 0.3 * foot_height),
                (x_left_foot - 0.5 * foot_width, y_top_feet + foot_height),
                (x_left_foot + 0.5 * foot_width, y_top_feet + foot_height),
                (x_left_foot + 0.5 * foot_width, y_top_feet + 0.3 * foot_height),
            )
            p_right = (
                (x_right_foot, y_top_feet),
                (x_right_foot - 0.5 * foot_width, y_top_feet + 0.3 * foot_height),
                (x_right_foot - 0.5 * foot_width, y_top_feet + foot_height),
                (x_right_foot + 0.5 * foot_width, y_top_feet + foot_height),
                (x_right_foot + 0.5 * foot_width, y_top_feet + 0.3 * foot_height),
            )
        elif foot_subtype == "flat_lshaped":
            p_left = (
                (x_left_foot - 0.5 * foot_width, y_top_feet),
                (x_left_foot - 0.5 * foot_width, y_top_feet + 0.5 * foot_height),
                (x_left_foot - 1.00 * foot_width, y_top_feet + 0.5 * foot_height),
                (x_left_foot - 1.00 * foot_width, y_top_feet + 1.0 * foot_height),
                (x_left_foot + 0.5 * foot_width, y_top_feet + 1.0 * foot_height),
                (x_left_foot + 0.5 * foot_width, y_top_feet),
            )
            p_right = (
                (x_right_foot + 0.5 * foot_width, y_top_feet),
                (x_right_foot + 0.5 * foot_width, y_top_feet + 0.5 * foot_height),
                (x_right_foot + 1.00 * foot_width, y_top_feet + 0.5 * foot_height),
                (x_right_foot + 1.00 * foot_width, y_top_feet + 1.0 * foot_height),
                (x_right_foot - 0.5 * foot_width, y_top_feet + 1.0 * foot_height),
                (x_right_foot - 0.5 * foot_width, y_top_feet),
            )
        elif foot_subtype == "pointy_3sided":
            p_left = (
                (x_left_foot - 0.5 * foot_width, y_top_feet),
                (x_left_foot + 0.5 * foot_width, y_top_feet),
                (x_left_foot, y_top_feet + foot_height),
            )
            p_right = (
                (x_right_foot - 0.5 * foot_width, y_top_feet),
                (x_right_foot + 0.5 * foot_width, y_top_feet),
                (x_right_foot, y_top_feet + foot_height),
            )
        elif foot_subtype == "pointy_4sided":
            p_left = (
                (x_left_foot, y_top_feet),
                (x_left_foot - 0.5 * foot_width, y_top_feet + 0.5 * foot_height),
                (x_left_foot, y_top_feet + foot_height),
                (x_left_foot + 0.5 * foot_width, y_top_feet + 0.5 * foot_height),
            )
            p_right = (
                (x_right_foot, y_top_feet),
                (x_right_foot - 0.5 * foot_width, y_top_feet + 0.5 * foot_height),
                (x_right_foot, y_top_feet + foot_height),
                (x_right_foot + 0.5 * foot_width, y_top_feet + 0.5 * foot_height),
            )
        elif foot_subtype == "pointy_6sided":
            p_left = (
                (x_left_foot, y_top_feet),
                (x_left_foot - 0.33 * foot_width, y_top_feet + 0.33 * foot_height),
                (x_left_foot - 0.33 * foot_width, y_top_feet + 0.66 * foot_height),
                (x_left_foot, y_top_feet + foot_height),
                (x_left_foot + 0.33 * foot_width, y_top_feet + 0.66 * foot_height),
                (x_left_foot + 0.33 * foot_width, y_top_feet + 0.33 * foot_height),
            )
            p_right = (
                (x_right_foot, y_top_feet),
                (x_right_foot - 0.33 * foot_width, y_top_feet + 0.33 * foot_height),
                (x_right_foot - 0.33 * foot_width, y_top_feet + 0.66 * foot_height),
                (x_right_foot, y_top_feet + foot_height),
                (x_right_foot + 0.33 * foot_width, y_top_feet + 0.66 * foot_height),
                (x_right_foot + 0.33 * foot_width, y_top_feet + 0.33 * foot_height),
            )
        else:
            p_left = p_right = None

        if p_left is not None and p_right is not None:
            foot = pero.Polygon(line_color=None)
            foot.draw(canvas, points=p_left, fill_color=pero.colors.White)
            foot.draw(canvas, points=p_right, fill_color=pero.colors.White)

    return canvas


def blur_parts(
    img,
    parts=("body",),
    radius=2.0,
    expand_mask_px=None,
    feather_mask_px=0,
    mask_mode: str = "uniform_rect",
    **kwargs,
):
    """Apply Gaussian blur to selected parts of a rendered pero.Image.

    Args:
        img: pero.Image
            The already drawn robot image (PNG path not required).
        parts: tuple[str]
            Currently supports ('body',). Extendable later.
        radius: float
            Gaussian blur radius.
        kwargs: feature overrides used to match the mask to the robot.
    Returns:
        PIL.Image: blurred composite image in RGBA.
    """
    if PILImage is None or PILImageFilter is None:
        raise ImportError("Pillow is required for blurring operations")

    # render original to PIL
    _, base_pil = image_to_numpy_and_pillow(img)

    # render mask to PIL (white = blur area)
    mask_img = draw_robot_mask(
        width=int(img.width),
        height=int(img.height),
        parts=parts,
        mode=mask_mode,
        **kwargs,
    )
    _, mask_pil = image_to_numpy_and_pillow(mask_img)
    mask_l = mask_pil.convert("L")

    # optionally expand mask to include stroke/border region
    if expand_mask_px is None:
        expand_mask_px = int(round(radius))
    if expand_mask_px and expand_mask_px > 0 and PILImageFilter is not None:
        size = max(3, 2 * int(expand_mask_px) + 1)  # odd size >= 3
        mask_l = mask_l.filter(PILImageFilter.MaxFilter(size=size))

    # optionally feather mask edge for smoother transition
    if feather_mask_px and feather_mask_px > 0 and PILImageFilter is not None:
        mask_l = mask_l.filter(
            PILImageFilter.GaussianBlur(radius=float(feather_mask_px))
        )

    # blurred version of the whole image
    blurred = base_pil.filter(PILImageFilter.GaussianBlur(radius=radius))

    # composite: where mask is white, take blurred, otherwise original
    out = PILImage.composite(blurred, base_pil, mask_l)
    return out