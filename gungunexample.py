from test_script.test_slice_image_line import analyze_connection
from utility import *
if __name__ == "__main__":
    path = "test_images/circuit50038.png"
    group_connection, img = analyze_connection(cv2.imread(path, cv2.IMREAD_UNCHANGED), debug=False)
    cv2.imwrite("tmp.png", img)
    plot_images(img, img_width=700)
    print(group_connection)
