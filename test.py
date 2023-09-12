import MS
import torch
import math
import LinearCode
import numpy as np
import LayeredMS
import NeuralLayeredMS
import random
import dataset
# ======Params======
maxErrFrames = 100  # stop when bit errors reach bitErrNum
batchSize = 10000
codeFamily = "QR"
codeLength = 47
messageLength = 24
useCYC = False
useNeural = True
iters = 5
# ==================
codename = codeFamily + "." + str(codeLength) + "." + str(messageLength)
pre = "./matrices/" + codename
code = LinearCode.LinearCode(codeFamily, codeLength, messageLength,
                             np.loadtxt(pre + ".G.txt"),
                             np.loadtxt(pre + ".H.txt"), d=None)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.L1Loss()
if useCYC:
    H = np.loadtxt(pre + ".H.CYC.txt")
    order = list(range(codeLength))
else:
    H = code.H
    order = list(range(codeLength-messageLength))
if useNeural:
    net = NeuralLayeredMS.Layered_MS_Decoder(H, iters, batchSize, device)
    if useCYC:
        net.load_state_dict(torch.load(codename+"singleBCECYC17.pth"))
    else:
        net.load_state_dict(torch.load(codename+"singleBCE17.pth"))
else:
    net = LayeredMS.Layered_MS_Decoder(H, batchSize)

net.to(device)

for SNR in range(1, 8):
    np.random.seed(0)
    torch.manual_seed(0)
    errCnt = 0
    blockNum = 0
    badFrames = 0
    sigma = torch.Tensor([math.sqrt(1 / (2 * code.k / code.n * math.pow(10, SNR / 10)))]).to(device)
    errDistrb = torch.zeros(codeLength).to(device)
    with torch.no_grad():
        while badFrames < 100:
            testData = dataset.TestSet(code, batchSize, SNR, device)
            samples = testData.samples
            labels = testData.labels
            if useNeural:
                 _, outputs, _ = net.test(samples, order)
            else:
                outputs = net(samples, iters, order)
            diff = torch.logical_xor(labels, outputs)
            errFrames = torch.sum(diff, dim=1)
            errFrames[errFrames > 0] = 1
            badFrames += errFrames.sum()
            blockNum += batchSize
            errCnt += torch.sum(diff)
            if blockNum % 10000 == 0:
                print("Frames: ", blockNum, "\n SNR@: ", SNR, " dB", "\n Errors: ", errCnt.item(), "\n Bad Frames: ",
                      badFrames.item())
    print("===============================================")
    print("FER: %e" % (badFrames*1.0 / blockNum), "   @ ", SNR, "(dB)")
    print("BER: %e" % (errCnt*1.0 / (blockNum * codeLength)), "   @ ", SNR, "(dB)")
    print("===============================================")

