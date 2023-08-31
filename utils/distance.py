# 自定义函数，单目测距
def object_width(D, h):
    F = 19.99
    D=float(D)
    dis = (D * h) / F
    return dis
