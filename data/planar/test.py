import torch

x = torch.load('E:/PycharmProjects/GD_tcad_repo/data/planar/X_train.pt')


print(x.requires_grad)
print(x.requires_grad_(True))
print(x.requires_grad)

