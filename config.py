import platform

machine_name = platform.node()
# for aieda lab machine
if machine_name == "DESKTOP-S347DEH":
    DEPTH = 6
    NUM_HEADS = 8
    EMBED_DIM = 50
    # PRETRAINED_PATH = "complete/complex_detr/best.pth"
    # PRETRAINED_PATH = "runs/FindPoints/0603_10-35-06__decoder/best.pth"
    PRETRAINED_PATH = "latest"
    PRETRAINED_PATH = ""
    KEEP_EPOCH = True
    EVAL = True
    TEST = False
    if EVAL:
        TEST = True
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 5000
    EPOCHS = 5000
    DATASET_SIZE = 150000
    BATCH_SIZE = 32
    BATCH_STEP = 128 / BATCH_SIZE
    DEVICE_IDS = [0]
    DROPOUT = 0
    DATASET_PATH = "dataset_windowed_200000"
    PATCH_SIZE = 20
    DETERMINISTIC = False
elif machine_name == "nycu-ai-eda-5":
    DEPTH = 6
    NUM_HEADS = 8
    EMBED_DIM = 50
    # PRETRAINED_PATH = "complete/complex_detr/best.pth"
    # PRETRAINED_PATH = "runs/FindPoints/0603_10-35-06__decoder/best.pth"
    PRETRAINED_PATH = "latest"
    PRETRAINED_PATH = ""
    KEEP_EPOCH = True
    EVAL = False
    TEST = False
    if EVAL:
        TEST = True
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 5000
    EPOCHS = 5000
    DATASET_SIZE = 150000
    BATCH_SIZE = 32
    BATCH_STEP = 128 / BATCH_SIZE
    DEVICE_IDS = [0]
    DROPOUT = 0
    DATASET_PATH = "dataset_windowed_200000"
    PATCH_SIZE = 20
    DETERMINISTIC = False
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
