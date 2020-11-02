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
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import classification_report,confusion_matrix

model_path = "save_model/VAE/mice_train/VAE_adam_lr0001.pt"
# label_file = "data/dec-test.csv"
# data_file = "data/bin-test.csv"
label_file = "/data/sym/one-class-svm/data/mean_of_five/dec-feature/caida-A-50W-5-0.csv"
data_file = "/data/sym/one-class-svm/data/mean_of_five/bin-feature/caida-A-50W-5-0.csv"
# nodes = 387
thresh = 500
n_epochs = 50
init_quantile = 0.72



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(387, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            # nn.ReLU(),
            # nn.Linear(12, 3),
        )
        # 解压
        self.decoder = nn.Sequential(
            # nn.Linear(3, 12),
            # nn.ReLU(),
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 387),
            nn.Sigmoid(),       # 激励函数让输出值在 (0, 1)
        )
 
    def forward(self, x):
        batchsz = x.size(0)
        q = self.encoder(x)
        mu, sigma = q.chunk(2, dim=1)
        # reparameterize trick, eqsilon~N(0,1)
        q = mu + sigma * torch.randn_like(sigma)

        # decoder
        x_hat = self.decoder(q)
        # print("x_hat", x_hat.size())
        # KL
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) - 
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (batchsz * 387)
        return x_hat, kld

        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        # return decoded

class MNISTAnomalyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, transform=None):
        """
        Args:
            transform (callabel, optional): Optional transform to be applied
                on a sample.
        """
        # self.X = np.uint8(X)
        self.X = X
        self.transform = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        sample = self.X[idx]
        if self.transform:
            sampleX = self.transform(self.X[idx])
            sample = (sampleX)

        return torch.FloatTensor(sample)

def preprocess():
    
    label = pd.read_csv(label_file)
    data = pd.read_csv(data_file)
    
    # get label
    label = label["flowSize"]
    targets = label.copy(deep=True)
    # binary labels (0: inliers, 1: outliers) (0: mice, 1: elephant)
    targets[label > thresh] = 1
    targets[label <= thresh] = 0
    print("original mice count: ", sum(targets==0))
    print("original elephant count: ", sum(targets==1))

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=10)
  

    # data_numpy = data.values
    # targets_numpy = targets.values

    return X_train, X_test, y_train, y_test

def load_data():
   
    X_train, X_test, y_train, y_test = preprocess()
    # split train to ele and mice
    X_train_ele = X_train[y_train == 1]
    X_train_mice = X_train[y_train == 0]
    print("train ele", X_train_ele.shape[0])
    print("train mice", X_train_mice.shape[0])

    # use mice to train the model
    total_data = X_train_mice.values
    total_data[total_data=='0'] = -1
    total_data[total_data=='1'] = 1
    X_train, X_valid = train_test_split(total_data, test_size=0.2,random_state=10)

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    
    data = {}

    data['train'] = MNISTAnomalyDataset(X_train, transform=transform)
    data['valid'] = MNISTAnomalyDataset(X_valid, transform=transform)

    loaders = {}

    # batch_size=32
    batch_size = 1

    loaders['train'] = torch.utils.data.DataLoader(data['train'], batch_size=32, num_workers=0, shuffle=True)
    loaders['valid'] = torch.utils.data.DataLoader(data['valid'], batch_size=32, num_workers=0)
    return loaders

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
            
            output, kld = model.forward(data)
            loss = criterion(output, data)

            if kld is not None:
                elbo = loss + 1.0 * kld
                loss = elbo

            # output = model.forward(data)
            # print("data", data.size())
            # print("output", output.size())
            # loss = criterion(output, data)
            
            # backprop
            optimizer.zero_grad()
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
                output, kld = model.forward(data)
                loss = criterion(output, data)
                if kld is not None:
                    elbo = loss + 1.0 * kld
                    loss = elbo

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

def detect_anomalies(unsupervised_images, decoded_outputs, y_test, quantile=0.8):
    errors = []
    for(inputing, outputing) in zip(unsupervised_images.cpu().detach().numpy(), decoded_outputs.cpu().detach().numpy()):
        mse = np.mean((inputing - outputing)**2)
        errors.append(mse)
    quantile = sum(y_test==0) / unsupervised_images.shape[0]
    quantile = init_quantile
    print("quantile", quantile)
    thresh = np.quantile(errors, quantile)
    idxs = np.where(np.array(errors) >= thresh)[0]
    print("mse threshold: {}".format(thresh))
    print("{} outliers found".format(len(idxs)))
    print("test mice count: ", sum(y_test==0))
    print("test elephant count: ", sum(y_test==1))

    y_predict = np.zeros(unsupervised_images.shape[0])
    y_predict[idxs] = 1
    c_matrix = confusion_matrix(y_test, y_predict)
    print(c_matrix)
    print(classification_report(y_test, y_predict))

def test_model():
    path = model_path
    model = AutoEncoder()
    model.load_state_dict(torch.load(path))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    model.eval()
    
    X_train, X_test, y_train, y_test = preprocess()
    # test data
    total_data = X_test.values
    total_data[total_data=='0'] = -1
    total_data[total_data=='1'] = 1

    # total_data = torch.unsqueeze(torch.tensor(total_data).float(),1)
    total_data = torch.FloatTensor(total_data)
    print(total_data.shape)
    
    if use_cuda:
        total_data = total_data.cuda()

    decoded_outputs, _ = model.forward(total_data)
    y_predict = detect_anomalies(total_data, decoded_outputs, y_test, quantile=0.9)
    

def main():
    autoencoder = AutoEncoder()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = autoencoder.cuda()
    print(autoencoder)
    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    # train the model
    
    loaders = load_data()
    model, train_loss, valid_loss =  train(n_epochs, loaders, autoencoder, optimizer, criterion, use_cuda, model_path)

def test():
    autoencoder = AutoEncoder()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("use_cuda", use_cuda)
        model = autoencoder.cuda()
    print(autoencoder)
    load_data()

if __name__ == "__main__":
    main()
    test_model()
    # test()