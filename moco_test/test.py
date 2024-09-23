
import torch
import vits


model = vits.vit_conv_base()
checkpoint = torch.load("model_best.pth.tar", map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
model.load_state_dict(state_dict, strict=False)
input = torch.randn(1, 4, 50, 50)
output = model(input)
print(model)
print(output.shape)
# print(output)
