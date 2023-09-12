import dataset
import torch
import math
import LinearCode
import numpy as np
import NeuralLayeredMS
import logging
import mRRD
import time

# ======Params======
maxErrFrames = 100  # stop when bit errors reach bitErrNum
batchSize = 1000
codeFamily = "QR"
codeLength = 47
messageLength = 24
dmin = 7
width = 128
length = 5
SNRlo = 1
SNRhi = 5
step = 1
# ==================
codename = codeFamily + "." + str(codeLength) + "." + str(messageLength)
pre = "./matrices/" + codename
code = LinearCode.LinearCode(codeFamily, codeLength, messageLength,
                             np.loadtxt(pre + ".G.txt"),
                             np.loadtxt(pre + ".H.txt"), d=dmin)
logname = codename + "." + time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())+".log"
logging.basicConfig(filename="./logs/" + logname, level=logging.INFO)
logging.info("Code: " + codename)
logging.info("mRRD Size %d, %d" % (width, length))
weight = codename + "singleBCECYC17.pth"
cycH = np.loadtxt(pre+".H.CYC.txt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.L1Loss()
net = NeuralLayeredMS.Layered_MS_Decoder(cycH, length, batchSize, device)
net.load_state_dict(torch.load(weight))
logging.info("Using weight: %s" % weight)

net.to(device)
mrrd = mRRD.mRRD(code, batchSize, net.step, width, length, device)

for SNR in np.arange(SNRlo, SNRhi+step, step):
    logging.warning("Start to test at SNR = " + str(SNR))
    np.random.seed(0)
    torch.manual_seed(0)
    errCnt = 0
    blockNum = 0
    badFrames = 0
    sigma = torch.Tensor([math.sqrt(1 / (2 * code.k / code.n * math.pow(10, SNR / 10)))]).to(device)
    errDistrb = torch.zeros(codeLength).to(device)
    with torch.no_grad():
        while badFrames < maxErrFrames:
        #while blockNum < 10000:
            testData = dataset.TestSet(code, batchSize, SNR, device)
            samples = testData.samples
            labels = testData.labels
            order = list(range(codeLength))
            outputs = mrrd(samples)
            diff = torch.logical_xor(labels, outputs)
            errFrames = torch.sum(diff, dim=1)
            errFrames[errFrames > 0] = 1
            badFrames += errFrames.sum()
            blockNum += batchSize
            errCnt += torch.sum(diff[:, :messageLength])
            if blockNum % 10000 == 0:
                print("Frames: ", blockNum, "\n SNR@: ", SNR, " dB", "\n Errors: ", errCnt.item(), "\n Bad Frames: ",
                      badFrames.item())
                print("Running FER: %e" % (badFrames*1.0/blockNum))
                logging.info("Frames: " + str(blockNum) + ", Bad Frames: " + str(badFrames.item()) +
                             ", FER: %e" % (badFrames*1.0/blockNum))
                
    print("===============================================")
    print("FER: %e" % (badFrames * 1.0 / blockNum), "   @ ", SNR, "(dB)")
    print("===============================================")
    logging.critical("Frames: %d, Bad Frames: %d, FER: %e at " % (blockNum, badFrames, badFrames * 1.0 / blockNum)
                     + str(SNR) + "(dB)")
