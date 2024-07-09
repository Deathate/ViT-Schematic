PRETRAINED_PATH = "complete/complex_detr/best.pth"
PRETRAINED_PATH = "runs/FindPoints/0603_10-35-06__decoder/best.pth"
PRETRAINED_PATH = "latest"
KEEP_EPOCH = True
EVAL = False
TEST = False
if EVAL:
    TEST = True
LEARNING_RATE = 1e-4
EPOCHS = 50
DATASET_SIZE = 100
BATCH_SIZE = 32
BATCH_STEP = 128 / BATCH_SIZE
DEVICE_IDS = [0]
