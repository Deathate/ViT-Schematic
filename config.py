PRETRAINED_PATH = ""
PRETRAINED_PATH = "runs/FindPoints/0507_13-21-20__decoder/epoch=325.pth"
TEST = True if PRETRAINED_PATH != ""else False
DATA_NUM = 100000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
# 使用amp後，一輪訓練時間為11.5分鐘
