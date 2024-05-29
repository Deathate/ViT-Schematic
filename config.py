
PRETRAINED_PATH = "complete/complex_detr/best.pth"
PRETRAINED_PATH = "complete/complex_detr_2/best.pth"
# Datasetbehaviour.RESET = False if not TEST else True
EVAL = True
TEST = False
if EVAL:
    TEST = True
DATA_NUM = 50000 if not TEST else 64
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
BATCH_STEP = 128 / BATCH_SIZE
# 使用amp後，一輪訓練時間為11.5分鐘
