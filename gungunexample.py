from slice import *
from test_slice_image_line import analyze_connection
from utility import *

if __name__ == "__main__":

    test_list = list(path_like_sort([x.name for x in Path("real_data/train/images").iterdir()]))
    img, processed_img = load_test_data(test_list[0], "real_data/train")
    # min_line_length: the minimum length of the line to be considered as a line
    # local_threshold: the threshold to combine the lines in the same group
    # global_threshold: the threshold to combine the groups
    # remove_duplicate: remove the duplicate points when combining the global groups
    # optimal_shift: shift the image to find the optimal position
    # boundary: the boundary of the weight matrix to guide the shift
    # strict_match: the strict match to find the connection
    # strict_match_threshold: the threshold to consider the strict connection
    # debug: debug mode
    # debug_shift_optimization: debug shift optimization
    # debug_cell: debug cell
    group_connection, result_img = analyze_connection(
        processed_img,
        img,
        min_line_length=1e-2,
        interval=5,
        local_threshold=0.07,
        global_threshold=0,
        remove_duplicate=False,
        exclude_white_points=True,
        optimal_shift=False,
        boundary=3,
        strict_match=True,
        strict_match_threshold=0.15,
        debug=True,
        debug_shift_optimization=False,
        debug_cell=[7, 15],
        debug_img_width=600,
    )
    plot_images(result_img, 800)
