import platform

from utility import *

machine_name = platform.node()
KEEP_OPTIMIZER = True
KEEP_EPOCH = False
LEARNING_RATE = 1e-4
EPOCHS = 100000
BATCH_SIZE = 64
BATCH_STEP = 1
DEVICE_IDS = [0]
DROPOUT = 0
EVAL_DATASET_PATH = None
FLUSH_CACHE_AFTER_STEP = 0
EVAL = False
STYLE_OPTIONS = ["encoder, decoder", "cnn"]
MODEL_STYLE = STYLE_OPTIONS[1]
REAL_DATA = False
SMALL_IMAGE = True
ZOOM = False
CLASS_OUTPUT = False
UPLOAD = False


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


def get_best_model_path(config, full_size, class_output=False):
    global IMAGE_SIZE, RESULT_NUM, CLASS_OUTPUT
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
        if class_output:
            CLASS_OUTPUT = True
            if not full_size:
                RESULT_NUM = 10
                IMAGE_SIZE = 50
                return "runs/FormalDatasetWindowedLinePair/1118_17-28-26/best_train.pth"
            else:
                RESULT_NUM = 125
                IMAGE_SIZE = -1
                return WEIGHT_VERSION[10]
        else:
            CLASS_OUTPUT = False
            RESULT_NUM = 10
            IMAGE_SIZE = 50
            return "runs/FormalDatasetWindowedLinePair/1118_20-09-56/best_train.pth"

    else:
        raise ValueError("Invalid config")


if MODEL_STYLE == STYLE_OPTIONS[1]:
    if SMALL_IMAGE:
        DATASET_PATH = "cc_deathate_data/train"
        DATASET_SIZE = 300
        PRETRAINED_PATH = ""
        PICK = 1
        FLUSH_CACHE_AFTER_STEP = 0
        IMAGE_SIZE = 50
        PATCH_SIZE = 10
        DEPTH = 6
        NUM_HEADS = 8
        EMBED_DIM = 32
        RESULT_NUM = 35
        TEST_DATASET_PATH = "final_test"
        DIRECTION = 1
        if REAL_DATA:
            PRETRAINED_PATH = "runs/FormalDatasetWindowedLinePair/1117_23-54-55/latest.pth"
            PRETRAINED_PATH = ""
            RESULT_NUM = 10
            PICK = 1
            IMAGE_SIZE = 50
            DATASET_PATH = "real_data/train"
            TEST_DATASET_PATH = "real_data/test"
            EVAL_DATASET_PATH = None
            DATASET_SIZE = -1
            DIRECTION = 1
            DEPTH = 6
            NUM_HEADS = 8
            EMBED_DIM = 32
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
