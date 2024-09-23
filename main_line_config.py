import platform

machine_name = platform.node()
# for aieda lab machine
DEPTH = 6
NUM_HEADS = 4
EMBED_DIM = 32
# PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/0912_17-40-37/best.pth"
PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/0914_22-49-35/best.pth"
KEEP_OPTIMIZER = True
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

# DATASET_SIZE = [1200, 800]
# DATASET_SIZE = [120000, 120000]
DATASET_SIZE = [0, -1]

EVAL = False
if EVAL:
    PRETRAINED_PATH = "runs/FormalDatasetWindowed/0829_01-47-38/best.pth"
    DATASET_SIZE = [200, 200]
MODEL_STYLE = "encoder, decoder"
UPLOAD = False
