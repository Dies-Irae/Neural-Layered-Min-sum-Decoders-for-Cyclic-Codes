import dataset
import torch
import LinearCode
import numpy as np
import NeuralLayeredMS
import random
# ======Params======
datasetSize = 400000
batchSize = 2000
epochs = 10
SPANNIterTimes = 200
startSNR = 1
endSNR = 7
codeLength = 47
messageLength = 24
codeFamily = "QR"
trainWithFullZero = True
useCYC = False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# ==================

codename = codeFamily + "." + str(codeLength) + "." + str(messageLength)
pre = "./matrices/" + codename
code = LinearCode.LinearCode(codeFamily, codeLength, messageLength,
                             np.loadtxt(pre + ".G.txt"),
                             np.loadtxt(pre + ".H.txt"), d=None)
data = dataset.Trainset(code, datasetSize, (startSNR, endSNR), device, trainWithFullZero)
criterion = torch.nn.BCELoss()
if useCYC:
    H = np.loadtxt(pre + ".H.CYC.txt")
    order = list(range(codeLength))
else:
    H = code.H
    order = list(range(codeLength-messageLength))
net = NeuralLayeredMS.Layered_MS_Decoder(H, SPANNIterTimes, batchSize, device)
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)

for epoch in range(0, epochs):
    print("Epoch: ", epoch)
    for batchNum in range(datasetSize // batchSize):
        optimizer.zero_grad()
        samples = data.samples[batchNum * batchSize:(batchNum + 1) * batchSize, :]
        sigma = data.sigma[batchNum * batchSize:(batchNum + 1) * batchSize, :]
        labels = data.labels[batchNum * batchSize:(batchNum + 1) * batchSize, :]
        outputs = net(samples, order)
        loss = criterion(torch.sigmoid(-outputs[-1,:,:]), labels)
        loss.backward()
        torch.nan_to_num_(net.alphas.grad)
        optimizer.step()
        if batchNum % 50 == 0:
            print("LOSS: %.3f" % loss.item())
            print(torch.sigmoid(net.alphas))
if useCYC:
    torch.save(net.state_dict(), codename + "singleBCECYC" + str(startSNR) + str(endSNR) + str(SPANNIterTimes) + ".pth")
else:
    torch.save(net.state_dict(), codename + "singleBCE" + str(startSNR) + str(endSNR) + str(SPANNIterTimes) + ".pth")
