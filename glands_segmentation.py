import numpy as np
import matplotlib.pyplot as plt

import torch

import skimage.io

import tqdm
import time

#%% Data loader

class GlandData(torch.utils.data.Dataset):
    def __init__(self, data_dir, im_id, margin_size=20):
        self.images = []
        self.labels = []
        for idx in im_id:
            self.images.append(torch.tensor(skimage.io.imread(f'{data_dir}train_{idx:03d}.png').transpose(2,0,1)).type(torch.LongTensor)/255)
            label_im = skimage.io.imread(f'{data_dir}train_{idx:03d}_anno.png')[margin_size:-margin_size, margin_size:-margin_size]/255
            self.labels.append(torch.tensor(label_im).type(torch.LongTensor))

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.images)


dirin_train = 'data/train/'

glandTrainData = GlandData(dirin_train, range(0,600))
glandValData = GlandData(dirin_train, range(600,750))

#%% Check if implementation is correct

im, lab = glandTrainData[50]

fig, ax = plt.subplots(1,2)
ax[0].imshow(im.numpy().transpose(1,2,0))
ax[1].imshow(lab)

#%%

for i, im in enumerate(glandTrainData):
    if (im[0].shape[1]==93):
        print(i)

#%%

print(len(glandTrainData))
print(len(glandValData))

#%% Model

class UNet128(torch.nn.Module):
    """Takes in patches of 128^2, returns 88^2"""
    
    def __init__(self, out_channels=2):
        super(UNet128, self).__init__()

        # learnable
        self.conv1A = torch.nn.Conv2d(3, 8, 3)  
        self.conv1B = torch.nn.Conv2d(8, 8, 3)  
        self.conv2A = torch.nn.Conv2d(8, 16, 3)  
        self.conv2B = torch.nn.Conv2d(16, 16, 3)  
        self.conv3A = torch.nn.Conv2d(16, 32, 3)  
        self.conv3B = torch.nn.Conv2d(32, 32, 3)  
        self.conv4A = torch.nn.Conv2d(32, 16, 3)  
        self.conv4B = torch.nn.Conv2d(16, 16, 3)  
        self.conv5A = torch.nn.Conv2d(16, 8, 3)  
        self.conv5B = torch.nn.Conv2d(8, 8, 3)  
        self.convfinal = torch.nn.Conv2d(8, out_channels, 1)         
        self.convtrans34 = torch.nn.ConvTranspose2d(32, 16, 2, stride=2) 
        self.convtrans45 = torch.nn.ConvTranspose2d(16, 8, 2, stride=2)
        
        # convenience
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)        
       
    def forward(self, x):
 
        # down
        l1 = self.relu(self.conv1B(self.relu(self.conv1A(x))))
        l2 = self.relu(self.conv2B(self.relu(self.conv2A(self.pool(l1)))))
        out = self.relu(self.conv3B(self.relu(self.conv3A(self.pool(l2))))) 
        
        # up 
        out = torch.cat([self.convtrans34(out), l2[:,:,4:-4,4:-4]], dim=1)
        out = self.relu(self.conv4B(self.relu(self.conv4A(out))))
        out = torch.cat([self.convtrans45(out), l1[:,:,16:-16,16:-16]], dim=1)      
        out = self.relu(self.conv5B(self.relu(self.conv5A(out))))
   
         # finishing
        out = self.convfinal(out)
  
        return out


# %% INITIALIZATION
import os
lr = 0.0001
nr_epochs = 50
outdir = 'models/'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')

trainloader = torch.utils.data.DataLoader(glandTrainData,
                                          batch_size=5,
                                          shuffle=True,
                                          drop_last=True)
model = UNet128().to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#%%
epoch_losses = []
batch_losses = []
epoch_time = []

i = 50#125 % len(glandTrainData)
im, lb = glandTrainData[i]
im = im.type(torch.cuda.FloatTensor)
fig_in, ax_in = plt.subplots(1, 2)
ax_in[0].imshow(im.cpu().numpy().transpose(1,2,0))
ax_in[1].imshow(lb)

nr_epochs_disp = nr_epochs/10
d1 = int(np.ceil(np.sqrt(nr_epochs_disp)))
d0 = int(np.ceil(nr_epochs_disp / d1))
fig_out, ax_out = plt.subplots(d0, d1, squeeze=False)
ax_out = ax_out.ravel()

fig_loss, ax_loss = plt.subplots()
ep_iter = 0
for epoch in range(nr_epochs):

    t = time.time()
    epoch_loss = 0.0
    batch_loop = tqdm.tqdm(enumerate(trainloader), total=len(trainloader))
    for i, batch in batch_loop:
        batch_loop.set_description(f'Epoch {epoch}/{nr_epochs}')

        image_batch, label_batch = batch  # unpack the data
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        logits_batch = model(image_batch)
        optimizer.zero_grad()
        loss = loss_function(logits_batch, label_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_losses.append(loss.item())

    epoch_losses.append(epoch_loss / len(trainloader))
    epoch_time.append(time.time() - t)

    torch.save({'model_statedict': model.state_dict(),
                'optimizer_statedict': optimizer.state_dict(),
                'epoch_losses': epoch_losses,
                'batch_losses': batch_losses,
                'epoch_time': epoch_time
                }, outdir + 'checkpoint.pth')
    if epoch % 10 == 9:
        # visualization
        lgt = model(im.unsqueeze(0)).detach()
        prob = torch.nn.Softmax(dim=1)(lgt)
    
        # ax_out[epoch].imshow(prob.squeeze().permute(1, 2, 0))
        ax_out[ep_iter].imshow(prob.cpu().squeeze()[1])
        ax_loss.set_title(f'epoch:{len(epoch_losses) - 1}')
    
        ax_loss.cla()
        ax_loss.plot(np.linspace(0, len(epoch_losses), len(batch_losses)), batch_losses, lw=0.5)
        ax_loss.plot(np.arange(len(epoch_losses)) + 0.5, epoch_losses, lw=2, linestyle='dashed', marker='o')
        ax_loss.set_title('epoch and batch loss')
        ax_loss.set_ylim(0, max(epoch_losses))
        plt.pause(0.00001)  # the only certian way to show the image :-(
        ep_iter += 1


#%%


i = 19#125 % len(glandTrainData)
im_val, lb_val = glandValData[i]
im_val = im_val.type(torch.cuda.FloatTensor)
fig_in, ax_in = plt.subplots(1, 2)
ax_in[0].imshow(im_val.cpu().numpy().transpose(1,2,0))
ax_in[1].imshow(lb_val)

lgt_val = model(im_val.unsqueeze(0)).detach()

prob_val = torch.nn.Softmax(dim=1)(lgt_val)



fig, ax = plt.subplots()
ax.imshow(prob_val.cpu().squeeze()[1])


