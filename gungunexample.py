from test_slice_image_line import analyze_connection
from utility import *

if __name__ == "__main__":
    path = "test_images/circuit50038.png"
    # min_line_length: the minimum length of the line to be considered as a line
    # local_threshold: the threshold to combine the lines in the same group
    # global_threshold: the threshold to combine the groups
    # remove_duplicate: remove the duplicate points when combining the global groups
    group_connection, img = analyze_connection(
        cv2.imread(path, cv2.IMREAD_UNCHANGED),
        min_line_length=0.005,
        local_threshold=0.09,
        global_threshold=1e-5,
        remove_duplicate=False,
        optimal_shift=True,
        strict_match=True,
        soft_match=False,
        debug=True,
        debug_shift_optimization=False,
        debug_cell=[-1, -1],
    )
    print(group_connection)
    # cv2.imwrite("tmp.png", img)
