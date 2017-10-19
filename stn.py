
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


from PIL import Image
from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')


# In[20]:


import pdb


# In[3]:


from torch.nn.parameter import Parameter


# In[4]:


class LOCNet(nn.Module):
    def __init__(self):
        super(LOCNet, self).__init__()
        self.features =  nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout()
            # Flatten()
        )
        self.linear = nn.Linear(5184, 6)
        nn.init.constant(self.linear.weight, 0)
        self.linear.bias = Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x.view(-1, 2, 3)


# In[25]:


from inceptionresnetv2 import inceptionresnetv2


# In[29]:


class TransClf(nn.Module):
    def __init__(self, num_classes=100):
        super(TransClf, self).__init__()
        self.loc_net = LOCNet()
        self.features = inceptionresnetv2(num_classes=num_classes)
    def forward(self, x):
        out = self.loc_net(x)
        grid = F.affine_grid(out, x.size())
        out = F.grid_sample(x, grid)
        out = self.features(out)
        return out
        


# In[7]:


from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


# In[8]:


trans = transforms.Compose([transforms.Scale((160, 160)), transforms.ToTensor()])


# In[9]:


faceset = ImageFolder(root='/home/wenfahu/faces/lfw-deepfunneled', transform=trans)


# In[10]:


trainloader = torch.utils.data.DataLoader(faceset, batch_size=4,
                                          shuffle=True, num_workers=2)


# In[11]:


len(faceset.classes)


# In[30]:


stn_clf = TransClf(num_classes=len(faceset.classes))


# In[13]:


from torch.autograd import Variable


# In[33]:


import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(stn_clf.parameters(), weight_decay=1e-5)


# In[36]:


for epoch in range(150):
    running_loss = 0.0
    for idx, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = stn_clf(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.data[0]
        if idx % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, idx + 1, running_loss / 20))
            running_loss = 0.0

