# Functions for fitting

def plane(xy, mx, my, b):
    return mx*xy[0] + my*xy[1] + b

def parabola(xy, c0, cx1, cx2, cy1, cy2, cxy):
    return c0 + cx1*xy[0] + cy1*xy[1] + cx2*xy[0] + cy2*xy[1] + cxy*xy[0]*xy[1]

