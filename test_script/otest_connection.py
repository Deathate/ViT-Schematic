import json
import math
import os
from collections import Counter


# 計算兩點之間距離的輔助函數
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 用來判斷兩條線段是否近似相同
def are_segments_similar(segment1, segment2, threshold):
    return (
        distance(segment1[0], segment2[0]) < threshold
        and distance(segment1[1], segment2[1]) < threshold
    ) or (
        distance(segment1[0], segment2[1]) < threshold
        and distance(segment1[1], segment2[0]) < threshold
    )


# 去除相似的線段
def remove_similar_segments(segments, threshold):
    unique_segments = []

    for segment in segments:
        is_unique = True
        for unique_segment in unique_segments:
            if are_segments_similar(segment, unique_segment, threshold):
                is_unique = False
                break
        if is_unique:
            unique_segments.append(segment)

    return unique_segments


# 用來尋找每個線段所屬的組的根節點
def find(parent, i):
    if parent[i] == i:
        return i
    else:
        return find(parent, parent[i])


# 用來合併兩個組
def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)

    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1


# 根據兩條線段端點的距離將它們分組
def group_segments(segments, threshold=0.2):
    # 去除相似線段
    unique_segments = remove_similar_segments(segments, threshold)
    n = len(unique_segments)

    parent = list(range(n))
    rank = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            # 比較線段 i 和線段 j 之間的所有端點組合
            if (
                distance(unique_segments[i][0], unique_segments[j][0]) < threshold
                or distance(unique_segments[i][0], unique_segments[j][1]) < threshold
                or distance(unique_segments[i][1], unique_segments[j][0]) < threshold
                or distance(unique_segments[i][1], unique_segments[j][1]) < threshold
            ):
                # 如果有任一對端點距離小於閾值，則將這兩個線段分為同一組
                union(parent, rank, i, j)

    # 根據 parent 結構將線段分組
    groups = {}
    for i in range(n):
        root = find(parent, i)
        if root not in groups:
            groups[root] = []
        groups[root].append(unique_segments[i])

    return list(groups.values())


# 覆蓋原資料，使相同的點合併為同一點
def merge_and_replace_points(points, threshold):
    merged_points = points[:]
    for i in range(len(merged_points)):
        for j in range(i + 1, len(merged_points)):
            if distance(merged_points[i], merged_points[j]) < threshold:
                # 將點 j 的座標覆蓋為點 i 的座標
                merged_points[j] = merged_points[i]
    return merged_points


# 此處修改路徑
data = [
    [[0.49565523862838745, 0.8706074953079224], [0.8023309707641602, 0.8758493661880493]],
    [[0.46243903040885925, 0.326950341463089], [0.4643716812133789, 0.8831122517585754]],
    [[0.45042306184768677, 0.3148215413093567], [0.7445716857910156, 0.3184584975242615]],
    [[0.44824641942977905, 0.008173089474439621], [0.4482133686542511, 0.2943747937679291]],
]

# 使用列表生成式將最內層的列表轉換為 tuple
converted_data = [[tuple(inner_list) for inner_list in outer_list] for outer_list in data]

# 輸出結果
# print(converted_data)

# 根據距離將線段分組
grouped_segments = group_segments(converted_data, threshold=0.035)  # 閾值根據需求調整
# print(grouped_segments)

group_connection = []

for i in range(len(grouped_segments)):
    # 將所有點展平為一個列表
    all_points = [point for pair in grouped_segments[i] for point in pair]

    # 計算每個點的出現次數
    updated_points = merge_and_replace_points(all_points, threshold=0.035)
    # print(updated_points)

    point_count = Counter(updated_points)

    # 只保留出現一次的點
    unique_points = [point for point in updated_points if point_count[point] == 1]
    group_connection.append(unique_points)

print(group_connection)
