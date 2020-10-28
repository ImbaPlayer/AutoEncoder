import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 超参数
EPOCH = 5
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = True   # 下过数据的话, 就可以设置成 False
N_TEST_IMG = 5          # 到时候显示 5张图片看效果, 如上图一

def load_data(): 
    # Mnist digits dataset
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,                                     # this is training data
        transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=DOWNLOAD_MNIST,                        # download it if you don\'t have it
    )

    # plot one example
    # print(train_data.data.size())     # (60000, 28, 28)
    # print(train_data.targets.size())   # (60000)
    # plt.imshow(train_data.data[3].numpy(), cmap='gray')
    # plt.title('%i' % train_data.targets[3])
    # plt.show()
    return train_data

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),   # 压缩成3个特征, 进行 3D 图像可视化
        )
        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # 激励函数让输出值在 (0, 1)
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train():
    autoencoder = AutoEncoder()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    train_data = load_data()
    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    # initialize figure
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5,2))
    plt.ion()   # Turn the interactive mode on, continuously plot

    # first row for viewing
    view_data = train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255
    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())
    
    for epoch in range(EPOCH):
        for step, (x, b_label) in enumerate(train_loader):
            b_x = x.view(-1, 28*28)
            b_y = x.view(-1, 28*28)

            encoded, decoded = autoencoder(b_x)
            # mean square error
            loss = loss_func(decoded, b_y)
            # clear gradients for this training step
            optimizer.zero_grad()
            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if step % 100 == 0:
                print("Epoch: ", epoch, "| train loss: %.4f" % loss.data.numpy())

                # plotting decoded image(second row)
                _, decoded_data = autoencoder(view_data)
                for i in range(N_TEST_IMG):
                    a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(())
                    a[1][i].set_yticks(())
                plt.draw()
                plt.pause(0.02)
    
    # turn the interactive mode off
    plt.ioff()
    # plt.show()
    torch.save(autoencoder, './model/autoencoder.pkl')
    # visualize(train_data, autoencoder)
    testDecoder(autoencoder)

def visualize(train_data, autoencoder):
    view_data = train_data.data[:200].view(-1, 28*28).type(torch.FloatTensor)/255
    encoded_data, _ = autoencoder(view_data)
    fig = plt.figure(2)
    ax = Axes3D(fig)
    # x, y, z
    X = encoded_data.data[:, 0].numpy()
    Y = encoded_data.data[:, 1].numpy()
    Z = encoded_data.data[:, 2].numpy()
    values = train_data.targets[:200].numpy()    # 标签值
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255*s/9))    # 上色
        ax.text(x, y, z, s, backgroundcolor=c)    # 标位子
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()

def testDecoder():
    autoencoder = AutoEncoder()
    autoencoder = torch.load('./model/autoencoder.pkl')
    code = torch.FloatTensor([[1.7, -2.5, 3.1]])
    decode = autoencoder.decoder(code)  # decode shape (1, 178)
    decode = decode.view(decode.size()[0], 28, 28)
    decode_img = decode.squeeze().data.numpy() * 255
    plt.figure()
    plt.imshow(decode_img.astype(np.uint8), cmap='gray')
    plt.show()
def testQuantile():
    errors = [1,2,3,4,5]
    quantile = 0.9
    thresh = np.quantile(errors, quantile)
    print(thresh)
if __name__ == "__main__":
    # load_data()
    # train()
    # testDecoder()
    testQuantile()