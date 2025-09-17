
from __future__ import annotations

from typing import Dict, Iterable, Tuple

from PIL import Image, ImageDraw


def _poly(draw, pts, fill, outline, width=2):
    draw.polygon(pts, fill=fill, outline=outline)

def _rect_bbox(cx, cy, w, h):
    return (cx - w//2, cy - h//2, cx + w//2, cy + h//2)

def _circle_bbox(cx, cy, r):
    return (cx - r, cy - r, cx + r, cy + r)

def draw_robot_image(features: Dict[str, str], size: int = 256, occlude: Iterable[str] = (), bg=(255,255,255), fg=(0,0,0), left_color=(200,200,255), right_color=(255,200,200)) -> Tuple[Image.Image, Dict[str, Tuple[int,int,int,int]]]:
    W = H = size
    img = Image.new('RGB', (W, H), bg)
    d = ImageDraw.Draw(img)
    cx, cy = W//2, H//2
    bw, bh = int(0.42*W), int(0.44*H)
    head_w, head_h = int(0.32*W), int(0.22*H)
    arm_len = int(0.32*W)
    leg_len = int(0.24*H)
    limb_w = max(2, W//80)
    bbox = {}
    left_rect = (0,0,W//2,H)
    right_rect = (W//2,0,W,H)
    d.rectangle(left_rect, fill=left_color)
    d.rectangle(right_rect, fill=right_color)
    d.rectangle((0,0,W-1,H-1), outline=fg, width=2)
    bx0, by0, bx1, by1 = _rect_bbox(cx, cy+int(0.05*H), bw, bh)
    
    if features.get('body_shape','square') == 'round':
        d.ellipse((bx0,by0,bx1,by1), outline=fg, width=3)
    else:
        d.rectangle((bx0,by0,bx1,by1), outline=fg, width=3)
        
    bbox['body_shape'] = (bx0,by0,bx1,by1)
    hx0, hy0, hx1, hy1 = _rect_bbox(cx, by0-int(0.12*H), head_w, head_h)
    
    if features.get('head_shape','square') == 'round':
        d.ellipse((hx0,hy0,hx1,hy1), outline=fg, width=3)
    else:
        d.rectangle((hx0,hy0,hx1,hy1), outline=fg, width=3)
        
    bbox['head_shape'] = (hx0,hy0,hx1,hy1)
    eye_r = max(3, size//50)
    d.ellipse(_circle_bbox(cx-int(0.05*W), hy0+int(0.35*head_h), eye_r), fill=fg)
    d.ellipse(_circle_bbox(cx+int(0.05*W), hy0+int(0.35*head_h), eye_r), fill=fg)
    mouth_type = features.get('mouth_type','closed')
    mx0, my0, mx1, my1 = cx-int(0.08*W), hy1-int(0.10*head_h), cx+int(0.08*W), hy1-int(0.04*head_h)
    
    if mouth_type == 'open':
        d.rectangle((mx0,my0,mx1,my1), outline=fg, width=2)
    else:
        d.line((mx0,(my0+my1)//2,mx1,(my0+my1)//2), fill=fg, width=2)
        
    bbox['mouth_type'] = (mx0,my0,mx1,my1)
    ears_shape = features.get('ears_shape','square')
    le = (hx0-int(0.05*W), (hy0+hy1)//2)
    re = (hx1+int(0.05*W), (hy0+hy1)//2)
    ear_size = int(0.06*W)
    
    if ears_shape == 'triangle':
        _poly(d, [ (le[0], le[1]-ear_size), (le[0], le[1]+ear_size), (le[0]-ear_size, le[1]) ], fill=None, outline=fg)
        _poly(d, [ (re[0], re[1]-ear_size), (re[0], re[1]+ear_size), (re[0]+ear_size, re[1]) ], fill=None, outline=fg)
    else:
        d.rectangle(_rect_bbox(le[0], le[1], ear_size, ear_size), outline=fg, width=2)
        d.rectangle(_rect_bbox(re[0], re[1], ear_size, ear_size), outline=fg, width=2)
        
    bbox['ears_shape'] = _rect_bbox(le[0], le[1], ear_size, ear_size)
    has_antennae = str(features.get('has_antennae','false')).lower() == 'true'
    ant_y_top = hy0-int(0.06*H)
    
    if has_antennae:
        d.line((cx, hy0, cx, ant_y_top), fill=fg, width=limb_w)
        d.ellipse(_circle_bbox(cx, ant_y_top-4, 6), fill=fg, outline=fg, width=1)
        
    bbox['has_antennae'] = (cx-4, hy0- int(0.06*H)-8, cx+4, hy0)
    y_arm = by0 + int(0.33*bh)
    x_left = bx0; x_right = bx1
    d.line((x_left, y_arm, x_left-int(arm_len), y_arm), fill=fg, width=limb_w)
    d.line((x_right, y_arm, x_right+int(arm_len), y_arm), fill=fg, width=limb_w)
    elbow_radius = max(4, size//60)
    has_elbows = str(features.get('has_elbows','false')).lower() == 'true'
    
    if has_elbows:
        d.ellipse(_circle_bbox(x_left-int(arm_len*0.5), y_arm, elbow_radius), outline=fg, width=2)
        d.ellipse(_circle_bbox(x_right+int(arm_len*0.5), y_arm, elbow_radius), outline=fg, width=2)
        
    bbox['has_elbows'] = (x_left-int(arm_len), y_arm-elbow_radius-2, x_right+int(arm_len), y_arm+elbow_radius+2)
    y_leg_top = by1
    y_leg_bot = by1+int(leg_len)
    d.line((cx-int(0.18*bw), y_leg_top, cx-int(0.18*bw), y_leg_bot), fill=fg, width=limb_w)
    d.line((cx+int(0.18*bw), y_leg_top, cx+int(0.18*bw), y_leg_bot), fill=fg, width=limb_w)
    knee_radius = max(4, size//60)
    has_knees = str(features.get('has_knees','false')).lower() == 'true'
    
    if has_knees:
        d.ellipse(_circle_bbox(cx-int(0.18*bw), y_leg_top+int(0.5*leg_len), knee_radius), outline=fg, width=2)
        d.ellipse(_circle_bbox(cx+int(0.18*bw), y_leg_top+int(0.5*leg_len), knee_radius), outline=fg, width=2)
        
    bbox['has_knees'] = (cx-int(0.22*bw), y_leg_top, cx+int(0.22*bw), y_leg_bot)
    hand_shape = features.get('hand_shape','round_circle')
    left_hand_x = x_left-int(arm_len)
    right_hand_x = x_right+int(arm_len)
    hand_y = y_arm
    
    if hand_shape in ('round_circle','round_oval','round_oval2'):
        d.ellipse(_circle_bbox(left_hand_x, hand_y, 8), outline=fg, width=2)
        d.ellipse(_circle_bbox(right_hand_x, hand_y, 8), outline=fg, width=2)
    elif hand_shape in ('edgy_triangle','edgy_square','edgy_trapezoid'):
        size_h = 12
        d.rectangle(_rect_bbox(left_hand_x, hand_y, size_h, size_h), outline=fg, width=2)
        d.rectangle(_rect_bbox(right_hand_x, hand_y, size_h, size_h), outline=fg, width=2)
        
    bbox['hand_shape'] = (left_hand_x-12, hand_y-12, right_hand_x+12, hand_y+12)
    foot_shape = features.get('foot_shape','flat_4sided')
    foot_w, foot_h = int(0.18*bw), int(0.08*H)
    fl_cx = cx-int(0.18*bw); fr_cx = cx+int(0.18*bw)
    fl = _rect_bbox(fl_cx, y_leg_bot+foot_h//2, foot_w, foot_h)
    fr = _rect_bbox(fr_cx, y_leg_bot+foot_h//2, foot_w, foot_h)
    
    if foot_shape.startswith('pointy'):
        _poly(d, [ (fl[0],fl[1]),(fl[2],fl[1]),((fl[0]+fl[2])//2,fl[3]) ], fill=None, outline=fg)
        _poly(d, [ (fr[0],fr[1]),(fr[2],fr[1]),((fr[0]+fr[2])//2,fr[3]) ], fill=None, outline=fg)
    else:
        d.rectangle(fl, outline=fg, width=2)
        d.rectangle(fr, outline=fg, width=2)
        
    bbox['foot_shape'] = (min(fl[0],fr[0]), min(fl[1],fr[1]), max(fl[2],fr[2]), max(fl[3],fr[3]))
    
    for c in occlude or ():
        if c in bbox:
            d.rectangle(bbox[c], fill=bg)
            d.rectangle(bbox[c], outline=bg, width=2)
    return img, bbox
