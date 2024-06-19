PRETRAINED_PATH = "complete/complex_detr/best.pth"
PRETRAINED_PATH = "runs/FindPoints/0603_10-35-06__decoder/best.pth"
# Datasetbehaviour.RESET = False if not TEST else True
EVAL = False
TEST = False
if EVAL:
    TEST = True
DATA_NUM = 100000 if not TEST else 64
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
BATCH_STEP = 128 / BATCH_SIZE
# 使用amp後，一輪訓練時間為11.5分鐘
