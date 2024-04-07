
# %%

from Model import *

image_set = []
dataset = []
root = Path("connect")
for path in (root / "XML").glob("*.json"):
    components = []
    nets = []
    with open(path, 'r') as f:
        data = json.load(f)
        shapes = data["shapes"]
        for shape in shapes:
            if shape["label"] != "net":
                components.append(shape)
            elif shape["label"] == "net":
                nets.append(shape)
    components_id_table = {}
    for c in components:
        components_id_table[int(c["group_id"])] = c
        # cv.rectangle(image, np.int32((c["points"][0])),
        #              np.int32((c["points"][1])), (0, 255, 0), 2)
    image = cv.imread(str(root / f"{path.stem}.jpg"))
    try:
        for net_order, net in enumerate(nets):
            for order in range(len(net["points"])):
                point_set = [net["points"][order]]
                description = list(map(lambda x: list(map(int, x.split(","))),
                                       net["description"].split(";")))
                for n in net["points"]:
                    point_set.append(n)
                del point_set[order + 1]
                dataset.append(point_set)
                # debug
                img = image.copy()
                cv.circle(img, np.int32(point_set[0]), 5, (255, 0, 0), -1)
                for point in point_set[1:]:
                    cv.circle(img, np.int32(point), 5, (0, 0, 255), -1)
                image_set.append(img)
            # start_box = components_id_table[description[0][order]]
            # cv.rectangle(image, *np.int32((start_box["points"])), (0, 0, 255), 2)
    except:
        print(path, net_order + 1)
for _ in range(10):
    for i in rng.integers(0, 1000, 1):
        plot_images(image_set[i], 400, 32)
# print(len(dataset))
