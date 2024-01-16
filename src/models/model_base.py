import torch


class ModelBase(torch.nn.Module):
    
    def __init__(self) -> None:
        super(ModelBase, self).__init__()
        self.name = None
        # self.loss_func = self.customed_loss()
    
    
    def forward(self, x):
        raise NotImplementedError()
    
    def customed_loss(self):
        raise NotImplementedError()

