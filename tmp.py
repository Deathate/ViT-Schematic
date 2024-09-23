import random

import wandb


api = wandb.Api()
run = api.run("ho-aron0105-deathate/FormalDatasetWindowed/mfxaa542")
for artifact in run.logged_artifacts():
    if "latest" not in artifact.aliases:
        print(artifact.name, artifact.type)
    # if "latest" in artifact.name:
    #     names.append(artifact.name)
# exit()
# # if artifact.type == "dataset":
# #     artifact.delete(delete_aliases=True)
# run = api.run("ho-aron0105-deathate/FormalDatasetWindowed/6gnt4fcs")
# api = wandb.Api(overrides={"project": "FormalDatasetWindowed", "entity": "ho-aron0105-deathate"})
# names = []
# for v in api.artifact_versions("model", "run-6uwwo57l-best.pth"):
#     print(v.aliases)
# exit()
