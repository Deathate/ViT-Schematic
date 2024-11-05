import platform

from utility import *

machine_name = platform.node()
DEPTH = 6
NUM_HEADS = 6
EMBED_DIM = 32
KEEP_OPTIMIZER = True
KEEP_EPOCH = False
LEARNING_RATE = 1e-4
EPOCHS = 100000
BATCH_SIZE = 32
BATCH_STEP = 128 / BATCH_SIZE
DEVICE_IDS = [0]
DROPOUT = 0
DATASET_PATH = "cc_deathate_data/train"
DATASET_SIZE = 30000
EVAL_DATASET_PATH = "cc_deathate_data/val"
EVAL_DATASET_SIZE = 200
PICK = 20
FLUSH_CACHE_AFTER_STEP = 0

IMAGE_SIZE = 100
PATCH_SIZE = 5
EVAL = False
STYLE_OPTIONS = ["encoder, decoder", "cnn"]
MODEL_STYLE = STYLE_OPTIONS[1]
REAL_DATA = True
FULL_IMAGE = False
SMALL_IMAGE = True


class DatasetConfig(Enum):
    CC = auto()
    REAL = auto()


# weight version
WEIGHT_VERSION = {
    1: "runs/FormalDatasetWindowedLinePair/1027_20-00-43/best.pth",  # real data, 12 direction, cnn, 100 pick, IMAGE_SIZE = 50
    2: "runs/FormalDatasetWindowedLinePair/1030_11-02-14/best.pth",  # real data, 1 direction, cnn, 300 pick, IMAGE_SIZE = 50, pretrained from 1
    3: "runs/FormalDatasetWindowedLinePair/1031_20-36-58/best.pth",  # real data, 1 direction, cnn, 300 pick, IMAGE_SIZE = 50, no pretrained
    4: "runs/FormalDatasetWindowedLinePair/1102_10-36-16/best.pth",  # real data, 1 direction, cnn, 300 pick, IMAGE_SIZE = 100, no pretrained
    5: "runs/FormalDatasetWindowedLinePair/1104_13-44-06/best.pth",  # real data, 1 direction, cnn, 80 pick, IMAGE_SIZE = 200, no pretrained
    6: "runs/FormalDatasetWindowedLinePair/1105_12-20-29/best.pth",
}


def get_best_model_path(config):
    if config == DatasetConfig.CC:
        return "runs/FormalDatasetWindowedLinePair/1012_02-08-46/best.pth"
    elif config == DatasetConfig.REAL:
        global IMAGE_SIZE
        IMAGE_SIZE = 50
        return WEIGHT_VERSION[2]
        # IMAGE_SIZE = 200
        # return WEIGHT_VERSION[6]

    else:
        raise ValueError("Invalid config")


if not FULL_IMAGE:
    if MODEL_STYLE == STYLE_OPTIONS[0]:
        DATASET_SIZE = 30000
        PICK = 10
        PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1006_02-07-40"
        PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1006_14-30-14"
        PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1007_17-41-54"
        PRETRAINED_PATH += "/latest.pth"
        FLUSH_CACHE_AFTER_STEP = 3
        IMAGE_SIZE = 50
        PATCH_SIZE = 10
        DEPTH = 6
        NUM_HEADS = 10
        EMBED_DIM = 16
        BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1006_02-07-40/best.pth"
        RESULT_NUM = 35
        if REAL_DATA:
            PRETRAINED_PATH = ""
            DATASET_SIZE = -1
            DATASET_PATH = "real_data/train"
            TEST_DATASET_PATH = "real_data/train"
            EVAL_DATASET_PATH = None
            PICK = 30
            FLUSH_CACHE_AFTER_STEP = 0
            IMAGE_SIZE = 50
            PATCH_SIZE = 10
            DEPTH = 10
            NUM_HEADS = 10
            EMBED_DIM = 16
            RESULT_NUM = 35
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1015_20-41-17/best.pth"
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1023_17-36-46/best.pth"
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1023_17-45-14/best.pth"
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1023_19-52-36/best.pth"
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1024_11-34-11/best.pth"
    elif MODEL_STYLE == STYLE_OPTIONS[1]:
        if SMALL_IMAGE:
            DATASET_SIZE = 30000
            EVAL_DATASET_SIZE = 500
            PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1012_02-08-46/latest.pth"
            PICK = 10
            FLUSH_CACHE_AFTER_STEP = 0
            IMAGE_SIZE = 100
            PATCH_SIZE = 10
            DEPTH = 6
            NUM_HEADS = 8
            EMBED_DIM = 32
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1012_02-08-46/best.pth"
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1014_01-16-46/best.pth"
            TEST_DATASET_PATH = "cc_deathate_data/train"
            RESULT_NUM = 35
            if REAL_DATA:
                PRETRAINED_PATH = ""
                # PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1015_00-49-05/best.pth"
                # PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1015_17-20-43/best.pth"
                PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1018_02-09-44/best.pth"
                PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1027_01-09-49/best.pth"
                PRETRAINED_PATH = ""
                PRETRAINED_PATH = WEIGHT_VERSION[2]
                PICK = 300
                IMAGE_SIZE = 50
                DATASET_PATH = "real_data/train"
                TEST_DATASET_PATH = "real_data/train"
                EVAL_DATASET_PATH = None
                DATASET_SIZE = -1
                DIRECTION = 1
                # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1022_17-19-11/best.pth"
                # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1024_02-20-17/best.pth"
                # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1018_02-09-44/best.pth"
                # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1027_20-00-43/best.pth"
                # BEST_MODEL_PATH = WEIGHT_VERSION[4]

        else:
            DATASET_SIZE = 18000
            EVAL_DATASET_SIZE = 500
            PRETRAINED_PATH = ""
            PICK = 10
            FLUSH_CACHE_AFTER_STEP = 0
            IMAGE_SIZE = 100
            PATCH_SIZE = 10
            DEPTH = 6
            NUM_HEADS = 8
            EMBED_DIM = 32
            BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1015_12-05-36/best.pth"
            RESULT_NUM = 35
            TEST_DATASET_PATH = "cc_deathate_data/test"
            if REAL_DATA:
                PRETRAINED_PATH = ""
                DATASET_PATH = "real_data/train"
                TEST_DATASET_PATH = "real_data/train"
                EVAL_DATASET_PATH = None
                DATASET_SIZE = -1
                PICK = 30
                BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1018_02-09-44/best.pth"
# else:
#     if MODEL_STYLE == STYLE_OPTIONS[1]:
#         DATASET_SIZE = 20
#         PICK = 0
#         PRETRAINED_PATH = ""
#         FLUSH_CACHE_AFTER_STEP = 0
#         IMAGE_SIZE = 0
#         PADDING_SIZE = (650, 700)
#         PATCH_SIZE = 10
#         DEPTH = 6
#         NUM_HEADS = 10
#         EMBED_DIM = 16
#         # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1007_17-41-54/best.pth"
#         BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1009_12-13-44/latest.pth"
#         RESULT_NUM = 100
#     elif MODEL_STYLE == STYLE_OPTIONS[2]:
#         DATASET_SIZE = -1
#         DATASET_PATH = "real_data"
#         TEST_DATASET_PATH = "real_data"
#         PICK = 1
#         DEPTH = 6
#         NUM_HEADS = 8
#         EMBED_DIM = 32
#         RESULT_NUM = 130
UPLOAD = False
