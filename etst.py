# -*- coding: utf-8 -*-
from PIL import Image

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer=SummaryWriter("logs")
# writer.add_image()
# for i in range(100):
#     writer.add_scalar("y=x",i,i)
# writer.close()

image_path="C:\\Users\\yaoch\\Desktop\\deepleanrning\\Papers\\Low_dose\\mybaseline\\12.png"
image=Image.open(image_path)
image.convert("RGB")
print(image)
# totensor
tran_tensor=transforms.ToTensor()
image_tensor=tran_tensor(image)
writer.add_image("Tensor",image_tensor,1)

# normalize
print(image_tensor[0][0][0])
print(image_tensor.size())
tran_norm=transforms.Normalize([2,2,2],[0.5,0.5,0.5])
image_norm=tran_norm(image_tensor)
print(image_norm[0][0][0])
writer.add_image("Second",image_norm,1)
