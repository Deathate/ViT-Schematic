from utility import *

a = [
    np.array([[1, 2, 3]]),
    np.array([[4, 5, 6], [7, 8, 9]]),
]
b = np.array(a, dtype=object)
print(b[:,0])