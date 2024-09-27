# %%
import os

os.chdir("..")
# %%
from main_line import *

Datasetbehaviour.MP = False
if isinstance(config.DATASET_SIZE, list):
    DATASET_PATH = [
        "data_distribution_50/w_mask_w_line",
        "data_distribution_50/wo_mask_w_line",
    ]
    DATASET_SIZE = [2000, 0]
    datasets = []
    for a, b in zip(DATASET_SIZE, DATASET_PATH):
        datasets.append(FormalDatasetWindowedLinePair(a, b))
    dataset_guise = datasets[0]
    for i in range(1, len(datasets)):
        dataset_guise = dataset_guise.union_dataset(datasets[i])
model = Model(
    dataset_guise,
    xtransform=xtransform,
    ytransform=ytransform,
    amp=True,
    batch_size=config.BATCH_SIZE,
    eval=config.EVAL,
)


# %%
ver_result_dir = "test_script/ver_result"
shutil.rmtree(ver_result_dir, ignore_errors=True)
Path(ver_result_dir).mkdir(parents=True, exist_ok=True)


def eval_metrics(criterion, y_hat, y):
    if model.meta.mode == "val":
        exit()
    loss = criterion(y_hat, y)
    C = torch.cdist(y_hat[:, :, 0], y[:, :, 0]) + torch.cdist(y_hat[:, :, 1], y[:, :, 1])
    for i, c in enumerate(C):
        correct = c.diag() < 0.1
        num_correct = correct.sum()
        if num_correct < len(c):
            # print(num_correct, len(c))
            file_name = Path(model.meta.data[i].meta).stem
            img = model.meta.data[i].input
            # img_bk = img.copy()
            img = cv2.resize(img, (256, 256))
            # plot_images(img)
            cv2.imwrite(f"{ver_result_dir}/{file_name}.png", img)
            for ln in y[i]:
                ln = ln.cpu().numpy()
                img = draw_line(img, [(ln[0], ln[1])], color=(0, 0, 255))
            for ln in y_hat[i]:
                ln = ln.detach().cpu().numpy()
                img = draw_line(img, [(ln[0], ln[1])], color=(255, 0, 0), thickness=2)
            cv2.imwrite(f"{ver_result_dir}/{file_name}_r.png", img)
            # if file_name.startswith("circuit2166"):
            #     print(y_hat[i][~correct])
            #     print(y[i][~correct])
            #     plot_images(img)
            #     plot_images(img_bk)
            #     exit()

    # if (c.diag()<0.05).sum() > 0:
    #     if c2 >= 0.05:
    accs = sum([(c.diag() < 0.05).sum() for c in C]) / (C.size(0) * C.size(1))
    return loss, {"acc": accs.item()}


network = create_model()
PT = "runs/FormalDatasetWindowedLinePair/0925_01-40-08/best.pth"
model.fit(
    network,
    criterion,
    optim.Adam(network.parameters(), lr=config.LEARNING_RATE),
    config.EPOCHS,
    max_epochs=config.MAX_EPOCHS if hasattr(config, "MAX_EPOCHS") else float("inf"),
    pretrained_path=PT,
    keep=True,
    backprop_freq=config.BATCH_STEP,
    device_ids=config.DEVICE_IDS,
    eval_metrics=eval_metrics,
    keep_epoch=config.KEEP_EPOCH,
    keep_optimizer=config.KEEP_OPTIMIZER,
)
# %%
