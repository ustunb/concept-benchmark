"""
This file contains classes to characterize a Robot and its distinct features
"""
import numpy as np
import pero
import paths
from dataclasses import dataclass, field
from typing import List, Dict
from itertools import product
from utils import generate_color_schemes

#from itertools import combinations_with_replacement as cwr
#COLOR_LIST = [c.name for c in pero.colors]
#COLOR_SCHEMES = [(w, v) for (w, v) in cwr(COLOR_LIST, 2) if w != v]
ROBOT_TYPES = ('glorp', 'drent')

ALL_ROBOT_FEATURES = {
    'foot_shape': ('flat_4sided', 'flat_5sided',  'flat_lshaped', 'pointy_3sided', 'pointy_4sided', 'pointy_6sided'),
    'body_shape': ('square', 'round'),  # no subtypes (could add)
    'head_shape': ('square', 'round'),  # no subtypes (could add)
    #
    #'has_elbows': ('true', 'false'),  # all round
    'has_knees': ('true', 'false'),
    }

# dict that assings a value to a unique comibnation of features (excluding subtypes after _)
FEATURES_NO_SUBTYPES = {k: tuple(set(val.split("_")[0] for val in v)) for k, v in ALL_ROBOT_FEATURES.items()}
ALL_COMBINATIONS = list(product(*list(FEATURES_NO_SUBTYPES.values())))


COLOR_SCHEMES = generate_color_schemes(shuffle = True, random_seed = 123456, include_flipped = False)
print('Generated %d color schemes' % len(COLOR_SCHEMES))


DEFAULT_ROBOT_FEATURES = {
    #
    'head_shape': 'square',
    # 'head_subtype_choice': 'default',
    #
    'body_shape': 'square',
    # 'body_subtype_choice': 'default',
    #
    # foot
    'foot_shape': 'flat',
    'foot_subtype_choice': 'default',
    #
    'has_knees': 'true',
    #'has_elbows': 'true',
    #
    'color_scheme': COLOR_SCHEMES[0]
    }

FOOT_SUBTYPES = {
    'flat': ['flat_4sided', 'flat_5sided', 'flat_lshaped'],
    'pointy': ['pointy_3sided', 'pointy_4sided', 'pointy_6sided'],
    }

@dataclass
class Robot:
    """
    class to represent a robot shown to users
    all fields must be JSON Serializable
    """
    features: Dict
    features_subtype: Dict
    type: str
    type_binary: int
    x: tuple
    id: int = 0
    predictions: List = field(default_factory = lambda: [])
    seen: bool = False
    image_file: str = ''
    image_url: str = ''
    kwargs: field(default_factory=dict) = None
    def __post_init__(self):
        self.x = [int(x) for x in self.x]
        resorted_values_no_subtypes = tuple([self.features[k].split("_")[0] for k in FEATURES_NO_SUBTYPES.keys()])
        self.unique_id = ALL_COMBINATIONS.index(resorted_values_no_subtypes)
        self.predictions = [int(p) for p in self.predictions]
        self.image_file = str(paths.static_dir / ('robot_%03d.png' % self.id))
        self.image_url = str(paths.image_dir.relative_to(paths.static_dir) / ('robot_%03d.png' % self.id))
        if self.kwargs != {} and self.kwargs != None:
            [object.__setattr__(self, k, v) for k, v in self.kwargs.items()]




def draw_robot(filetype= 'svg', col_scheme_add = 0, width = 600, height = 600, **kwargs):

    # set unspecified features using default values
    features = dict(DEFAULT_ROBOT_FEATURES)
    features.update(kwargs)

    # canvas
    canvas = pero.svg.SVGCanvas(width = width, height = height) if filetype == 'svg' else pero.Image(width=width, height=height)
    if filetype != 'svg': canvas.fill(pero.colors.White)
    canvas.line_cap = pero.ROUND
    canvas.line_join = pero.ROUND
    canvas.line_color = pero.colors.Black
    canvas.line_width = 2

    # colors
    if isinstance(features['color_scheme'], int):
        print(np.mod(features['color_scheme'] + col_scheme_add, len(COLOR_SCHEMES)))
        color_scheme_id = np.mod(features['color_scheme'] + col_scheme_add, len(COLOR_SCHEMES))
        color_left, color_right = COLOR_SCHEMES[color_scheme_id]
        print(color_left, color_right)
    elif isinstance(features['color_scheme'], tuple):
        color_left, color_right = features['color_scheme']

    # lengths
    r = canvas.height / 12.0

    # anchor points
    y_top = 2 * r
    x_mid = 0.5 * canvas.width

    # standard elements
    line = pero.Line(line_color = pero.colors.Black, line_width = 1)
    ray = pero.Ray(line_color = pero.colors.Black, line_width = 1)

    # face lengths
    head_shape = features['head_shape']
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
    ray.draw(canvas,
             x = x_left,
             y = y_arm,
             length = arm_length,
             angle = pero.rads(180))

    ray.draw(canvas,
             x = x_right,
             y = y_arm,
             length = arm_length,
             angle = pero.rads(0))


    # elbows
    if 'has_elbows' in features.keys() and features['has_elbows'] == 'true':
        elbow = pero.Ellipse(y = y_arm, width = 0.3 * r, height = 0.3 * r)
        elbow.draw(canvas, x = x_left - 0.5 * arm_length,
                   fill_color = color_left)
        elbow.draw(canvas, x = x_right + 0.5 * arm_length,
                   fill_color = color_right)

    # hands
    ray.draw(canvas, x = x_left - arm_length, y = y_arm, length = hand_length,
             angle = pero.rads(270))
    ray.draw(canvas, x = x_right + arm_length, y = y_arm, length = hand_length,
             angle = pero.rads(270))


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
    line.draw(canvas,
              x1 = x_left,
              x2 = x_left,
              y1 = y_top * 0.9,
              y2 = y_top + leg_height)
    line.draw(canvas,
              x1 = x_right,
              x2 = x_right,
              y1 = y_top * 0.9,
              y2 = y_top + leg_height)

    # knees
    if features['has_knees'] == 'true':
        knee = pero.Ellipse(x = x_left, y = y_top + 0.5 * leg_height,
                            width = 0.3 * r, height = 0.3 * r)
        knee.draw(canvas, x = x_left, fill_color = color_left)
        knee.draw(canvas, x = x_right, fill_color = color_right)

    y_top += leg_height
    foot_subtype = features['foot_shape']
    if foot_subtype in FOOT_SUBTYPES.keys():
        if features['foot_subtype_choice'] == 'default':
            foot_subtype = '%s' % FOOT_SUBTYPES[foot_subtype][0]

    if foot_subtype == 'flat_4sided':

        p_left = (
            (x_left - 0.5 * foot_width, y_top),
            (x_left - 0.5 * foot_width, y_top + foot_height),
            (x_left + 0.5 * foot_width, y_top + foot_height),
            (x_left + 0.5 * foot_width, y_top),
            )

        p_right = (
            (x_right - 0.5 * foot_width, y_top),
            (x_right - 0.5 * foot_width, y_top + foot_height),
            (x_right + 0.5 * foot_width, y_top + foot_height),
            (x_right + 0.5 * foot_width, y_top),
            )

    elif foot_subtype == 'flat_5sided':

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

    elif foot_subtype == 'flat_lshaped':

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

    elif foot_subtype == 'pointy_3sided':

        p_left = (
            (x_left - 0.5 * foot_width, y_top),
            (x_left + 0.5 * foot_width, y_top),
            (x_left, y_top + foot_height)
            )

        p_right = (
            (x_right - 0.5 * foot_width, y_top),
            (x_right + 0.5 * foot_width, y_top),
            (x_right, y_top + foot_height)
            )

    elif foot_subtype == 'pointy_4sided':

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

    elif foot_subtype == 'pointy_6sided':

        p_left = (
            (x_left, y_top),
            (x_left - 0.33 * foot_width, y_top + 0.33 * foot_height),
            (x_left - 0.33 * foot_width, y_top + 0.66 * foot_height),
            (x_left, y_top + foot_height),
            (x_left + 0.33 * foot_width, y_top + 0.66 * foot_height),
            (x_left + 0.33 * foot_width, y_top + 0.33 * foot_height),
            )

        p_right = (
            (x_right, y_top),
            (x_right - 0.33 * foot_width, y_top + 0.33 * foot_height),
            (x_right - 0.33 * foot_width, y_top + 0.66 * foot_height),
            (x_right, y_top + foot_height),
            (x_right + 0.33 * foot_width, y_top + 0.66 * foot_height),
            (x_right + 0.33 * foot_width, y_top + 0.33 * foot_height),
            )

    else:
        raise TypeError('invalid foot type')

    ############################################################################
    # body (drawn last)
    ############################################################################
    body_shape = features['body_shape']
    if body_shape == 'round':

        body = pero.Arc(x = x_mid,
                        y = y_body + 0.5 * body_height,
                        radius = 0.57 * body_width,
                        line_color = pero.colors.Black)

        body.draw(canvas, start_angle = pero.rads(90), end_angle = pero.rads(-90), fill_color = color_left)
        body.draw(canvas, start_angle = pero.rads(-90), end_angle = pero.rads(90), fill_color = color_right)

    elif body_shape == 'square':
        body = pero.Rect(x = x_mid - 0.5 * body_width, y = y_body, height = body_height)
        body.draw(canvas, line_width = 0, fill_color = color_left, width = 0.5 * body_width)
        body.draw(canvas, line_width = 0, fill_color = color_right, width = 0.5 * body_width, x = x_mid)
        body.draw(canvas, fill_color = None, width = body_width)

    foot = pero.Polygon(line_color = pero.colors.Black)
    foot.draw(canvas, points = p_left, fill_color = color_left)
    foot.draw(canvas, points = p_right, fill_color = color_right)

    ############################################################################
    # face, eyes, antenna
    ############################################################################

    # antennae
    antenna_length = 1.5 * r if head_shape == 'round' else 1.75 * r
    ray.draw(canvas, x = x_mid, y = y_top_face + head_height / 2,
             angle = pero.rads(180 + 60), length = antenna_length)

    ray.draw(canvas, x = x_mid, y = y_top_face + head_height / 2,
             angle = pero.rads(-60), length = antenna_length)


    if head_shape == 'round':
        face = pero.Arc(x = x_mid, y = y_top_face + 0.9*r, radius = 0.5 * head_width)
        face.draw(canvas, start_angle = pero.rads(90),
                  end_angle = pero.rads(270), fill_color = color_left)
        face.draw(canvas, start_angle = pero.rads(-90),
                  end_angle = pero.rads(90), fill_color = color_right)

    elif head_shape == 'square':
        face = pero.Rect(y = y_top_face, height = head_height)
        face.draw(canvas, x = x_mid - 0.5 * head_width,
                  width = 0.5 * head_width, line_width = 0,
                  fill_color = color_left)
        face.draw(canvas, x = x_mid, width = 0.5 * head_width, line_width = 0,
                  fill_color = color_right)
        face.draw(canvas, x = x_mid - 0.5 * head_width, width = head_width,
                  height = head_height, fill_color = None)

    # eyes
    eye = pero.Ellipse(fill_color = pero.colors.Black,
                       y = y_top_face + (3.0 / 8.0) * head_height,
                       height = 0.2 * r,
                       width = 0.2 * r)
    eye.draw(canvas, x = x_mid - 0.3 * r)
    eye.draw(canvas, x = x_mid + 0.3 * r)

    return canvas