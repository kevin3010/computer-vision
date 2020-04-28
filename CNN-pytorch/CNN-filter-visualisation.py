import matplotlib.pyplot as plt 
import numpy as np
import cv2 

filter1 = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])
filter2 = -filter1
filter3 = filter1.T
filter4 = -filter3

filters = np.array([filter1,filter2,filter3,filter4])

import torch 
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,weight):
        
        super(Net,self).__init__()
            
        k_height , k_width = weight.shape[2:]
        
        self.conv1 = nn.Conv2d(1,4,kernel_size = (k_height,k_width),bias=False)
        self.conv1.weights = torch.nn.Parameter(weight)
        
        self.pool = nn.MaxPool2d(4,4)
        
    def forward(self,x):
        convx = self.conv1(x)
        activated_convx = F.relu(convx)
        
        pooledx = self.pool(activated_convx)
        
        return convx,activated_convx,pooledx
        
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)

model = Net(weight)


img = cv2.imread("udacity_sdc.png")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_img = gray_img.astype('float32')/255


img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)


conv,conv_active,pooled = model(img_tensor)

fig,axs1 = plt.subplots(1,4,figsize=(20,20))
axs1 = axs1.flatten()

for i,ax in enumerate(axs1):
    
    ax.imshow(conv[0][i].data.numpy(),cmap='gray')
    
    
    
fig,axs2 = plt.subplots(1,4,figsize=(20,20))
axs2 = axs2.flatten()    
    
for i,ax in enumerate(axs2):
    ax.imshow(conv_active[0][i].data.numpy(),cmap='gray')
    
    
    
fig,axs3 = plt.subplots(1,4,figsize=(20,20))
axs3 = axs3.flatten()    
    
for i,ax in enumerate(axs3):
    ax.imshow(pooled[0][i].data.numpy(),cmap='gray')
