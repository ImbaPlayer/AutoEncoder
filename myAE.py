import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torchvision.transforms as transforms

# nodes = 387
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(387, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )
        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 387),
            nn.Sigmoid(),       # 激励函数让输出值在 (0, 1)
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    valid_loss_min = np.Inf
    train_loss_list = []
    valid_loss_list = []

    model.train()
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        # train the model
        for batch_idx, (data) in enumerate(loaders["train"]):
            # move to GPU
            if use_cuda:
                data = data.cuda()
            optimizer.zero_grad()
            output = model.forward(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        
        #validate the model
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(loaders['valid']):
                # move to GPU
                if use_cuda:
                    data = data.cuda()
                # update the average validation loss
                output = model.forward(data)
                loss = criterion(output, data)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

        train_loss_list.append(train_loss.item())
        valid_loss_list.append(valid_loss.item())

        # save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            print("Validation loss decreased ({:.6f} -> {:.6f}). Saving model ...".format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
        
    return model, train_loss_list, valid_loss_list

def detect_anomalies(unsupervised_images,decoded_outputs,quantile=0.999):
    errors = []
    for(inputing, outputing) in zip(unsupervised_images.cpu().detach().numpy()/255.0, decoded_outputs.cpu().detach().numpy()/255.0):
        mse = np.mean((inputing - outputing)**2)
        errors.append(mse)
    thresh = np.quantile(errors, quantile)
    idxs = np.where(np.array(errors) >= thresh)[0]
    print("mse threshold: {}".format(thresh))
    print("{} outliers found".format(len(idxs)))

    outlier_input_images = unsupervised_images[idxs].cpu().detach().numpy()
    outlier_output_images = decoded_outputs[idxs].cpu().detach().numpy()
    
    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=len(idxs), sharex=True, sharey=True, figsize=(25,4))

    # input images on top row, reconstructions on bottom
    for images, row in zip([outlier_input_images, outlier_output_images], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

def main():
    transform = transforms.ToTensor()

    autoencoder = AutoEncoder()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model_transfer = autoencoder.cuda()
    print(autoencoder)
    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    # train the model
    n_epochs = 10
    loaders = {}
    model_transfer, train_loss, valid_loss =  train(n_epochs, loaders, autoencoder, optimizer, criterion, use_cuda, 'Autoencoder_anomaly_adam_lr0001.pt')

def test():
    autoencoder = AutoEncoder()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model_transfer = autoencoder.cuda()
    print(autoencoder)

if __name__ == "__main__":
    test()