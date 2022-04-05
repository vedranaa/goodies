import numpy as np
import matplotlib.pyplot as plt

import torch
import skimage.io

import tqdm  # for nice progress bar
import yaml  # for easy handling of local settings


#%% Get local settings
settings = yaml.load(open('settings.yaml'), Loader=yaml.FullLoader)
dirin_train = settings['dirin_train']
device = settings['device']


#%% Dataset class

class GlandData(torch.utils.data.Dataset):
    '''  Dataset which loads all images for training, validation or testing'''
    def __init__(self, data_dir, im_id, margin_size=20):
        self.images = []
        self.labels = []
        for idx in im_id:
            self.images.append(torch.tensor(skimage.io.imread(
                f'{data_dir}{idx:03d}.png').transpose(2, 0, 1), 
                dtype=torch.float32)/255)
            label_im = skimage.io.imread(
                f'{data_dir}{idx:03d}_anno.png')[
                margin_size:-margin_size, margin_size:-margin_size]/255
            self.labels.append(torch.tensor(label_im, dtype=torch.int64))

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.images)


#%% Make training and validan set

glandTrainData = GlandData(dirin_train + 'train_', range(0,600))
glandValData = GlandData(dirin_train + 'train_', range(600,750))

#%% Check if implementation is Dataset works as expected
print(len(glandTrainData))
print(len(glandValData))

N = 10
fig, ax = plt.subplots(4, N)
for k in range(2):

    rand_idx = np.random.choice(len(glandTrainData), size=N, replace=False)    
    for n, idx in enumerate(rand_idx):
        
        im, lab = glandTrainData[idx]    
        ax[0+2*k, n].imshow(im.permute(1,2,0))
        ax[1+2*k, n].imshow(lab)
        

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
nr_epochs = 500
outdir = 'models/'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

if (device == 'cuda') and (torch.cuda.is_available()):
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')

#%%
trainloader = torch.utils.data.DataLoader(glandTrainData,
                                          batch_size=5,
                                          shuffle=True,
                                          drop_last=True)
valloader = torch.utils.data.DataLoader(glandValData,
                                          batch_size=20)
model = UNet128().to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#%% TRAINING

# Prepare for bookkeeping.
epoch_losses = []
batch_losses = []
validation_losses = []

# Pick some image to show result on.
i = 50
im, lb = glandTrainData[i]
fig_in, ax_in = plt.subplots(1, 2)
ax_in[0].imshow(im.permute(1,2,0))
ax_in[1].imshow(lb)

# Prepare to show results.
nr_epochs_disp = nr_epochs/10
d1 = int(np.ceil(np.sqrt(nr_epochs_disp)))
d0 = int(np.ceil(nr_epochs_disp / d1))
fig_out, ax_out = plt.subplots(d0, d1, squeeze=False)
ax_out = ax_out.ravel()


# Train.
fig_loss, ax_loss = plt.subplots()
ep_iter = 0
for epoch in range(nr_epochs):

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

    torch.save({'model_statedict': model.state_dict(),
                'optimizer_statedict': optimizer.state_dict(),
                'epoch_losses': epoch_losses,
                'batch_losses': batch_losses}, 
                outdir + 'checkpoint.pth')
    if epoch % 10 == 9: 
        #  Book-keeping every tenth iterations
        with torch.no_grad():
            lgt = model(im.unsqueeze(0).to(device))
            batch_loop = tqdm.tqdm(enumerate(valloader), total=len(valloader))
            val_loss = 0
            for i, batch in batch_loop:
                batch_loop.set_description(f'Validating {epoch}/{nr_epochs}')
                image_batch, label_batch = batch  # unpack the data
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)
                logits_batch = model(image_batch)
                loss = loss_function(logits_batch, label_batch)
                val_loss += loss.item()
            validation_losses.append(val_loss / len(valloader))
                
        prob = torch.nn.Softmax(dim=1)(lgt)
    
        ax_out[ep_iter].imshow(prob[0,1].cpu().detach())
        ax_loss.set_title(f'epoch:{len(epoch_losses) - 1}')
    
        ax_loss.cla()
        ax_loss.plot(np.linspace(0, len(epoch_losses), len(batch_losses)), 
                     batch_losses, lw=0.5)
        ax_loss.plot(np.arange(len(epoch_losses)) + 0.5, epoch_losses, 
                     lw=2, linestyle='dashed', marker='o')
        ax_loss.plot(np.linspace(9.5, len(epoch_losses)-0.5, len(validation_losses)), 
                     validation_losses, lw=1, linestyle=':', marker='.')
        ax_loss.set_title('epoch and batch loss')
        ax_loss.set_ylim(0, max(epoch_losses + validation_losses))
        plt.pause(0.00001)  # the only certian way to show the image :-(
        ep_iter += 1


#%%  Show predictions for one image from the validation set

i = 19
im_val, lb_val = glandValData[i]
with torch.no_grad():
    lgt_val = model(im_val.unsqueeze(0).to(device))
prob_val = torch.nn.Softmax(dim=1)(lgt_val)


fig, ax = plt.subplots(1, 3)
ax[0].imshow(im_val.permute(1,2,0))
ax[1].imshow(lb_val)
ax[2].imshow(prob_val[0,1].cpu().detach())


