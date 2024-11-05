from typing import List

from utility import *


def productExceptSelf(nums: List[int]) -> List[int]:
    k = [1] * len(nums)
    r = 1
    for i in range(0, len(nums) - 1):
        r *= nums[i]
        k[i + 1] = r
    r = 1
    for i in range(len(nums) - 1, 1, -1):
        r *= nums[i]
        k[i - 1] *= r
    return k


print(productExceptSelf([1, 2, 3, 4]))  # [24, 12, 8, 6]
