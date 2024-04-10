import numpy as np
import potrace

from Model import *

# Make a numpy array with a rectangle in the middle
data = cv.imread("data_cleaning_example/dac082s085-page29_SOIC_Section_0.png", cv.IMREAD_GRAYSCALE)
filename = "data_cleaning_example/dac082s085-page29_SOIC_Section_0"
data = data / 255
data = cv.threshold(data, 0.5, 1, cv.THRESH_BINARY)[1]
# exit()
# print(data.dtype)
# data = np.zeros((953, 953), np.uint32)
# print(data.dtype)
# data[8:32 - 8, 8:32 - 8] = 1
# print(data.shape)
# Create a bitmap from the array
bmp = potrace.Bitmap(data)

# Trace the bitmap to a path
path = bmp.trace(potrace.TURNPOLICY_BLACK)

# # Iterate over path curves
# for curve in path:
#     print("start_point =", curve.start_point)
#     for segment in curve:
#         print(segment)
#         end_point_x, end_point_y = segment.end_point
#         if segment.is_corner:
#             c_x, c_y = segment.c
#         else:
#             c1_x, c1_y = segment.c1
#             c2_x, c2_y = segment.c2

with open(f"{filename}.svg", "w") as fp:
    fp.write(
        f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{data.shape[1]}" height="{data.shape[0]}" viewBox="0 0 {data.shape[1]} {data.shape[1]}">''')
    parts = []
    for curve in path:
        fs = curve.start_point
        parts.append(f"M{fs[0]},{fs[1]}")
        for segment in curve.segments:
            if segment.is_corner:
                a = segment.c
                b = segment.end_point
                parts.append(f"L{a[0]},{a[1]}L{b[0]},{b[1]}")
            else:
                a = segment.c1
                b = segment.c2
                c = segment.end_point
                parts.append(f"C{a[0]},{a[1]} {b[0]},{b[1]} {c[0]},{c[1]}")
        parts.append("z")
    fp.write(f'<path stroke="none" fill="white" fill-rule="evenodd" d="{"".join(parts)}"/>')
    fp.write("</svg>")
