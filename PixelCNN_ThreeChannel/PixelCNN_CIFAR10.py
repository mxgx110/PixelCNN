import time
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt

class MaskedConv2d(nn.Conv2d):
  def __init__(self, mask_type, *args, **kwargs):
    
    self.mask_type = mask_type
    assert mask_type in ('A', 'B'), "ERROR Mask Type Not Available"

    super().__init__(*args, **kwargs)
    self.register_buffer('mask', self.weight.data.clone()) #-> self.mask = self.weight.data.clone()

    out_ch, in_ch, h, w = self.weight.data.size()

    self.mask.fill_(1.)

    if mask_type == 'A':
      self.mask[:, :, h//2, w//2:] = 0.
      self.mask[:, :, h//2 + 1:, :] = 0.
    
    else:
      self.mask[:, :, h//2, w//2+1:] = 0.
      self.mask[:, :, h//2 + 1:, :] = 0.

  def forward(self, x):
    self.weight.data *= self.mask
    return super().forward(x)
    
    
class PixelCNN(nn.Module):
  
  def __init__(self, in_channel, num_layers, kernel_size, channels):
    super().__init__()

    self.conv_layers = nn.ModuleList()
    self.batch_norms = nn.ModuleList()

    self.in_channel = in_channel
    self.num_layers = num_layers
    self.kernel_size = kernel_size
    self.channels = channels
    self.activation = nn.ReLU()

    for idx in range(num_layers):
      if idx == 0:
        self.conv_layers = self.conv_layers.append(MaskedConv2d('A', in_channel, channels, kernel_size,
                                                                stride=1, padding='same', bias=False))
      else:
        self.conv_layers = self.conv_layers.append(MaskedConv2d('B', channels, channels, kernel_size,
                                                                stride=1, padding='same', bias=False))
        
      self.batch_norms = self.batch_norms.append(nn.BatchNorm2d(channels))
    
    self.conv_out = self.out = nn.Conv2d(channels, 256, 1)

  def forward(self, x, channel):

    assert channel in ['R', 'G', 'B'], 'ERROR unsupported color'
    R, G, B = x[:, 0, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1), x[:, 2, :, :].unsqueeze(1)

    if channel == 'R':

      R_aux1, R_aux2 = torch.zeros_like(R), torch.zeros_like(R)
      R_final = torch.cat((R_aux1, R_aux2, R), 1) # [128, 3, 28, 28]

      for idx in range(len(self.conv_layers)):

        R_final = self.activation(self.batch_norms[idx](self.conv_layers[idx](R_final)))

      return self.conv_out(R_final)

    elif channel == 'G':

      G_aux = torch.zeros_like(G)
      G_final = torch.cat((G_aux, R, G), 1) # [128, 3, 28, 28]

      for idx in range(len(self.conv_layers)):

        G_final = self.activation(self.batch_norms[idx](self.conv_layers[idx](G_final)))

      return self.conv_out(G_final)

    else:

      B_final = torch.cat((R, G, B), 1) # [128, 3, 28, 28]

      for idx in range(len(self.conv_layers)):

        B_final = self.activation(self.batch_norms[idx](self.conv_layers[idx](B_final)))

      return self.conv_out(B_final)
      
def sampling(model):

  img_chn = 3
  img_size = 32

  sample = torch.Tensor(1, img_chn, img_size, img_size).to(device)
  sample.fill_(0.)
  colors = ['R', 'G', 'B']

  for c in range(img_chn):  
    for i in range(img_size):
      for j in range(img_size):
          out = model(sample, colors[c])
          probs = F.softmax(out[:,:,i,j], dim=-1).data
          sample[:,c,i,j] = torch.multinomial(probs, 1).float() / 255.0

  plt.imshow(sample[0].permute(1, 2, 0).to('cpu'))
  plt.show()
  
def train(model, opt, criterion, dataloader, epochs):

  total_loss, total_loss1, total_loss2, total_loss3 = [], [], [], []
  for epoch in range(epochs):

    model.train()
    step = 0
    losses, losses1, losses2, losses3= 0, 0, 0, 0

    for idx, (img, _) in enumerate(dataloader):

      img = img.to(device).to(torch.float32)

      t_R = (img[:, 0, :, :]*255).long().to(device) #because there is no sigmoid at the end
      t_G = (img[:, 1, :, :]*255).long().to(device) #because there is no sigmoid at the end
      t_B = (img[:, 2, :, :]*255).long().to(device) #because there is no sigmoid at the end
          
      opt.zero_grad()

      R = model(img, 'R')
      G = model(img, 'G')
      B = model(img, 'B')      

      loss1 = criterion(R, t_R)
      loss2 = criterion(G, t_G)
      loss3 = criterion(B, t_B)
      loss = loss1 + loss2 + loss3
      
      losses1 += loss1
      losses2 += loss2
      losses3 += loss3
      losses += loss

      step += 1
      loss.backward()
      opt.step()
    
    avg_loss_train1 = losses1/step
    avg_loss_train2 = losses2/step
    avg_loss_train3 = losses3/step
    avg_loss_train = losses/step

    total_loss1.append(avg_loss_train1.item())
    total_loss2.append(avg_loss_train2.item())
    total_loss3.append(avg_loss_train3.item())
    total_loss.append(avg_loss_train.item())

    if epoch % 1 == 0:
      print(f"epoch: {epoch+1} -> Loss: {avg_loss_train:.3f} ---- Loss_R: {avg_loss_train1:.3f} ---- Loss_G: {avg_loss_train2:.3f} ---- Loss_B: {avg_loss_train3:.3f}")
      sampling(model)

  return total_loss, total_loss1, total_loss2, total_loss3  
 
 
  
batch_size = 128

train_data = CIFAR10(root='.\data', train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lr = 1e-3
epochs = 20
in_channel = 3
num_layers = 8
kernel_size = 7
channels = 64


net = PixelCNN(in_channel, num_layers, kernel_size, channels).to(device)#PixelCNN(in_channel, num_layers, kernel_size, channels).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


time_start = time.time()
tr_loss, tr_loss_R, tr_loss_G, tr_loss_B = train(net, optimizer, criterion, dataloader, epochs)
time_end = time.time()
print('Training Time: ', time_end-time_start)
