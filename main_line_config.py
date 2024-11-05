import platform

machine_name = platform.node()
DEPTH = 6
NUM_HEADS = 4
EMBED_DIM = 32
# PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/0912_17-40-37/best.pth"
PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/0919_04-16-20/best.pth"
# runs/FormalDatasetWindowedLinePair/0925_01-02-38
BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/0925_11-14-32/best.pth"
BEST_MODEL_L_PATH = "runs/FormalDatasetWindowedLinePair/0925_11-13-18/best.pth"
KEEP_OPTIMIZER = False
KEEP_EPOCH = False
LEARNING_RATE = 1e-4
EPOCHS = 100000
BATCH_SIZE = 32
BATCH_STEP = 128 / BATCH_SIZE
DEVICE_IDS = [0]
DROPOUT = 0
STYLE = "mask"
if STYLE == "mask":
    DATASET_PATH = [
        "data_distribution_50/w_mask_w_line",
        "data_distribution_50/wo_mask_w_line",
    ]
    DATASET_SIZE = [-1, 0]
elif STYLE == "line":
    DATASET_PATH = [
        "data_distribution_50/w_mask_w_line",
        "data_distribution_50/wo_mask_w_line",
    ]
    DATASET_SIZE = [0, -1]
IMAGE_SIZE = 50
PATCH_SIZE = 10

# DATASET_SIZE = [1200, 800]
# DATASET_SIZE = [120000, 120000]

EVAL = False
if EVAL:
    PRETRAINED_PATH = "runs/FormalDatasetWindowed/0829_01-47-38/best.pth"
    DATASET_SIZE = [200, 200]
MODEL_STYLE = "encoder, decoder"
UPLOAD = False
