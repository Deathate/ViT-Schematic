PRETRAINED_PATH = "complete/complex_detr/best.pth"
PRETRAINED_PATH = "runs/FindPoints/0603_10-35-06__decoder/best.pth"
PRETRAINED_PATH = ""
# Datasetbehaviour.RESET = False if not TEST else True
EVAL = False
TEST = False
if EVAL:
    TEST = True
LEARNING_RATE = 1e-4
DATASET_SIZE = 35000
BATCH_SIZE = 32
BATCH_STEP = 128 / BATCH_SIZE
# 使用amp後，一輪訓練時間為11.5分鐘
