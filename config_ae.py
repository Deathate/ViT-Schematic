import platform

machine_name = platform.node()
if machine_name == "ai-eda-server-7-MS-7D91":
    # for aieda lab machine
    DEPTH = 6
    NUM_HEADS = 4
    EMBED_DIM = 32
    PRETRAINED_PATH = ""
    KEEP_OPTIMIZER = True
    KEEP_EPOCH = True
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 3000
    EPOCHS = 500
    # DATASET_SIZE = 3000
    BATCH_SIZE = 32
    BATCH_STEP = 128 / BATCH_SIZE
    DEVICE_IDS = [0]
    DROPOUT = 0
    # DATASET_PATH = "dataset_windowed_200000/data_distribution/w_mask_w_line"
    DATASET_PATH = [
        "dataset_50x50/data_distribution_50/w_mask_w_line",
        "dataset_50x50/data_distribution_50/wo_mask_w_line",
    ]
    DATASET_SIZE = [120, 30]
    IMAGE_SIZE = 50
    PATCH_SIZE = 10
    DETERMINISTIC = False
    STAGE = 0
    # NUM_RESULT = 36
    NUM_RESULT = 19
    # PRETRAINED_PATH = "runs/FormalDatasetWindowed/0729_13-56-42__decoder/best.pth"
    # EVAL = True
    # DATASET_SIZE = 128
    EVAL = False
    if EVAL:
        PRETRAINED_PATH = "latest"
        DATASET_SIZE = 200
    MODEL_STYLE = "encoder+decoder"
