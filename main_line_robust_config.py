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
SMALL_IMAGE = True
ZOOM = False


class DatasetConfig(Enum):
    CC = auto()
    REAL = auto()


# weight version
WEIGHT_VERSION = {
    1: "runs/FormalDatasetWindowedLinePair/1027_20-00-43/best.pth",  # real data, 12 direction, cnn, 100 pick, IMAGE_SIZE = 50
    2: "runs/FormalDatasetWindowedLinePair/1030_11-02-14/best.pth",  # real data, 1 direction, cnn, 300 pick, IMAGE_SIZE = 50, pretrained from 1
    7: "runs/FormalDatasetWindowedLinePair/1105_14-34-32/best.pth",  # real data, 1 direction, cnn, 300 pick, IMAGE_SIZE = 50, pretrained from 2
    8: "runs/FormalDatasetWindowedLinePair/1107_11-15-46/best.pth",  # real data, 4 direction, cnn, 300 pick, IMAGE_SIZE = 50, no pretrained
    9: "runs/FormalDatasetWindowedLinePair/1107_23-19-44/best.pth",  # real data, 4 direction, cnn, 150 pick, IMAGE_SIZE = 100, no pretrained
    10: "runs/FormalDatasetWindowedLinePair/1108_23-28-29/best.pth",  # real data, 4 direction, cnn, 1 pick, IMAGE_SIZE = FULL, no pretrained
    11: "runs/FormalDatasetWindowedLinePair/1112_00-59-53/best.pth",  # cc data, 1 direction, cnn, 1 pick, IMAGE_SIZE = FULL, no pretrained
}


def get_best_model_path(config, full_size):
    global IMAGE_SIZE, RESULT_NUM
    if config == DatasetConfig.CC:
        if not full_size:
            RESULT_NUM = 35
            IMAGE_SIZE = 50
            return "runs/FormalDatasetWindowedLinePair/1012_02-08-46/best.pth"
        else:
            RESULT_NUM = 125
            IMAGE_SIZE = -1
            return WEIGHT_VERSION[11]
    elif config == DatasetConfig.REAL:
        if not full_size:
            RESULT_NUM = 35
            # IMAGE_SIZE = 100
            # return WEIGHT_VERSION[9]
            IMAGE_SIZE = 50
            return "runs/FormalDatasetWindowedLinePair/1113_23-35-43/best.pth"
            return WEIGHT_VERSION[2]
        else:
            RESULT_NUM = 125
            IMAGE_SIZE = -1
            return WEIGHT_VERSION[10]

    else:
        raise ValueError("Invalid config")


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
        DATASET_SIZE = 10
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
        PRETRAINED_PATH = ""
        PICK = 10
        FLUSH_CACHE_AFTER_STEP = 0
        IMAGE_SIZE = 50
        PATCH_SIZE = 10
        DEPTH = 6
        NUM_HEADS = 8
        EMBED_DIM = 32
        # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1012_02-08-46/best.pth"
        # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1014_01-16-46/best.pth"
        TEST_DATASET_PATH = "final_test"
        RESULT_NUM = 35
        if REAL_DATA:
            PRETRAINED_PATH = ""
            # PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1015_00-49-05/best.pth"
            # PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1015_17-20-43/best.pth"
            PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1018_02-09-44/best.pth"
            PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1027_01-09-49/best.pth"
            PRETRAINED_PATH = WEIGHT_VERSION[2]
            PRETRAINED_PATH = ""
            RESULT_NUM = 10
            PICK = 1
            IMAGE_SIZE = 50
            DATASET_PATH = "real_data/train"
            TEST_DATASET_PATH = "real_data/test"
            EVAL_DATASET_PATH = None
            DATASET_SIZE = 3
            DIRECTION = 1
            # ZOOM = True
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1022_17-19-11/best.pth"
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1024_02-20-17/best.pth"
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1018_02-09-44/best.pth"
            # BEST_MODEL_PATH = "runs/FormalDatasetWindowedLinePair/1027_20-00-43/best.pth"
    else:
        DATASET_SIZE = 30000
        DATASET_PATH = "cc_deathate_data/train"
        EVAL_DATASET_PATH = None
        PRETRAINED_PATH = ""
        PICK = 1
        FLUSH_CACHE_AFTER_STEP = 0
        IMAGE_SIZE = -1
        DEPTH = 6
        NUM_HEADS = 8
        EMBED_DIM = 32
        RESULT_NUM = 125
        TEST_DATASET_PATH = "cc_deathate_data/train"
        DIRECTION = 1
        if REAL_DATA:
            PRETRAINED_PATH = WEIGHT_VERSION[10]
            RESULT_NUM = 125
            DATASET_PATH = "real_data/train"
            EVAL_DATASET_PATH = None
            TEST_DATASET_PATH = "real_data/train"
            DATASET_SIZE = -1
UPLOAD = False
