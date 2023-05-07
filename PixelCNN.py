import time
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

# Our customized conv2d layer + mask
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
    
# Model    
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

  def forward(self, x):

    for idx in range(len(self.conv_layers)):

      x = self.activation(self.batch_norms[idx](self.conv_layers[idx](x)))

    return self.conv_out(x) #There is no activation applied
    
# Training function    
def train(model, opt, criterion, dataloader, epochs):

  total_loss = []
  for epoch in range(epochs):

    model.train()
    step = 0
    losses= 0

    for idx, (img, _) in enumerate(dataloader):

      img = img.to(device).to(torch.float32)
      target = (img[:, 0, :, :]*255).long().to(device) #because there is no sigmoid at the end
          
      opt.zero_grad()
      output = model(img)
      loss = criterion(output, target)
      losses += loss
      step += 1
      loss.backward()
      opt.step()

    avg_loss_train = losses/step
    total_loss.append(avg_loss_train.item())

    if epoch % 1 == 0:
      print(f"epoch: {epoch+1} -> Loss: {avg_loss_train:.8f}")

  return total_loss
  
# DATA LOADING
batch_size = 128

train_data = MNIST(root='.\data', train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TRAINING
lr = 1e-3
epochs = 50
in_channel = 1
num_layers = 8
kernel_size = 7
channels = 64


net = PixelCNN(in_channel, num_layers, kernel_size, channels).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

time_start = time.time()
training_loss = train(net, optimizer, criterion, dataloader, epochs)
time_end = time.time()
print('Training Time: ', time_end-time_start)

# GENERATION
num_imgs = 1
img_chn = 1
img_size = 28

sample = torch.Tensor(num_imgs, img_chn, img_size, img_size).to(device)
sample.fill_(0.)

for i in range(img_size):
  for j in range(img_size):
    out = net(sample)
    probs = F.softmax(out[:,:,i,j], dim=-1).data
    sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0
    
idx = 0
plt.imshow(sample[idx].permute(1, 2, 0).to('cpu'), cmap='Greys_r')
plt.show()
