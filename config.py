import platform

machine_name = platform.node()
# for aieda lab machine
# 層數多可以收斂快一點
if machine_name == "ai-eda-server-7-MS-7D91":
    DEPTH = 6
    NUM_HEADS = 8
    EMBED_DIM = 64
    PRETRAINED_PATH = "complete/0725_11-57-10__decoder/best.pth"
    # PRETRAINED_PATH = "runs/FormalDatasetWindowed/0803_20-50-04__decoder/latest.pth"
    PRETRAINED_PATH = ""
    KEEP_OPTIMIZER = True
    KEEP_EPOCH = True
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 5000
    EPOCHS = 5000
    DATASET_SIZE = 100000
    # DATASET_SIZE = 200
    BATCH_SIZE = 32
    BATCH_STEP = 128 / BATCH_SIZE
    DEVICE_IDS = [0]
    DROPOUT = 0
    DATASET_PATH = "dataset_windowed_200000"
    PATCH_SIZE = 10
    DETERMINISTIC = False
    STAGE = 0
    NUM_RESULT = 36
    # PRETRAINED_PATH = "runs/FormalDatasetWindowed/0729_13-56-42__decoder/best.pth"
    # EVAL = True
    # DATASET_SIZE = 128
    EVAL = False
    if EVAL:
        PRETRAINED_PATH = "latest"
        DATASET_SIZE = 200
    MODEL_STYLE = "decoder"
# main
# elif machine_name == "nycu-ai-eda-5":
#     DEPTH = 3
#     NUM_HEADS = 8
#     EMBED_DIM = 50
#     PRETRAINED_PATH = "nvidia_log/best.pth"
#     KEEP_EPOCH = True
#     EVAL = True
#     LEARNING_RATE = 1e-4
#     RELATION_LEARNING_RATE = 1e-3
#     MAX_EPOCHS = 5000
#     EPOCHS = 5000
#     DATASET_SIZE = 130000
#     # DATASET_SIZE = 200
#     BATCH_SIZE = 32
#     BATCH_STEP = 128 / BATCH_SIZE
#     DEVICE_IDS = [0]
#     DROPOUT = 0
#     DATASET_PATH = "dataset"
#     PATCH_SIZE = 10
#     DETERMINISTIC = False
# main windowed
elif machine_name == "nycu-ai-eda-5":
    DEPTH = 3
    NUM_HEADS = 8
    EMBED_DIM = 50
    PRETRAINED_PATH = "complete/0725_11-57-10__decoder/best.pth"
    # PRETRAINED_PATH = "runs/FormalDatasetWindowed/0803_20-50-04__decoder/latest.pth"
    # PRETRAINED_PATH = "latest"
    KEEP_OPTIMIZER = True
    KEEP_EPOCH = True
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 5000
    EPOCHS = 5000
    DATASET_SIZE = 100000
    # DATASET_SIZE = 200
    BATCH_SIZE = 32
    BATCH_STEP = 128 / BATCH_SIZE
    DEVICE_IDS = [0]
    DROPOUT = 0
    DATASET_PATH = "dataset_windowed_200000"
    PATCH_SIZE = 10
    DETERMINISTIC = False
    STAGE = 2
    NUM_RESULT = 36
    # PRETRAINED_PATH = "runs/FormalDatasetWindowed/0729_13-56-42__decoder/best.pth"
    # EVAL = True
    # DATASET_SIZE = 128
    EVAL = True
    if EVAL:
        PRETRAINED_PATH = "latest"
        DATASET_SIZE = 200
    MODEL_STYLE = "decoder"
# for nvidia machine
else:
    PRETRAINED_PATH = "latest"
    KEEP_EPOCH = True
    EVAL = False
    TEST = False
    if EVAL:
        TEST = True
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 5000
    EPOCHS = 5000
    DATASET_SIZE = None
    BATCH_SIZE = 32
    BATCH_STEP = 128 / BATCH_SIZE
    DEVICE_IDS = [0]
    DROPOUT = 0
    DATASET_PATH = "dataset_windowed_200000"
    PATCH_SIZE = 10
    DETERMINISTIC = True
