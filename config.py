import platform

machine_name = platform.node()
if machine_name == "ai-eda-server-7-MS-7D91":
    # for aieda lab machine
    DEPTH = 6
    NUM_HEADS = 4
    EMBED_DIM = 32
    PRETRAINED_PATH = "runs/FormalDatasetWindowed/0823_08-34-43/best.pth"
    PRETRAINED_PATH = "runs/FormalDatasetWindowed/0829_01-47-38/best.pth"
    # TEST = True
    # PRETRAINED_PATH = "latest"
    KEEP_OPTIMIZER = False
    KEEP_EPOCH = True
    LEARNING_RATE = 1e-4
    EPOCHS = 100000
    BATCH_SIZE = 32
    BATCH_STEP = 128 / BATCH_SIZE
    DEVICE_IDS = [0]
    DROPOUT = 0
    DATASET_PATH = [
        "data_distribution_50/w_mask_w_line",
        "data_distribution_50/wo_mask_w_line",
    ]
    IMAGE_SIZE = 50
    PATCH_SIZE = 10
    DETERMINISTIC = False
    STAGE = 2
    if STAGE == 0:
        DATASET_SIZE = [120000, 80000]
    elif STAGE == 2:
        DATASET_SIZE = [120000, 80000]
    NUM_RESULT = 19
    EVAL = False
    if EVAL:
        PRETRAINED_PATH = "runs/FormalDatasetWindowed/0829_01-47-38/best.pth"
        DATASET_SIZE = [200, 200]
    MODEL_STYLE = "encoder, decoder, moco"
    RELATION_TOKEN = False
    FREEZE_DETECTION = False
    UPLOAD = False
    if "moco" in MODEL_STYLE:
        STAGE = 0
        PRETRAINED_PATH = "runs/FormalDatasetWindowed/0901_07-45-41/latest.pth"
        # DATASET_SIZE = [100, 100]
else:
    # for nvidia machine
    DEPTH = 6
    NUM_HEADS = 4
    EMBED_DIM = 32
    PRETRAINED_PATH = "latest"
    KEEP_OPTIMIZER = True
    KEEP_EPOCH = True
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 5000
    EPOCHS = 5000
    BATCH_SIZE = 32
    BATCH_STEP = 128 / BATCH_SIZE
    DEVICE_IDS = [0]
    DROPOUT = 0
    DATASET_PATH = [
        "dataset_50x50/data_distribution_50/w_mask_w_line",
        "dataset_50x50/data_distribution_50/wo_mask_w_line",
    ]
    DATASET_SIZE = [-1, -1]
    IMAGE_SIZE = 50
    PATCH_SIZE = 10
    DETERMINISTIC = False
    STAGE = 0
    NUM_RESULT = 19
    EVAL = False
    MODEL_STYLE = "encoder, decoder"
    UPLOAD = True
