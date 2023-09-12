import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class MSLayer(nn.Module):
    def __init__(self, shape):
        """single Sum-Product Algorithm layer"""
        super(MSLayer, self).__init__()
        self.alpha = torch.Tensor([1.0]).to(device)

    def forward(self, channelLLR, e2oLLR, maxColWeight, edgeToVar, edgeToVarMask, oddToEven, edgeToChk, rowWeight):
        """return marginalized probability and even to odd LLR"""
        # gather edge messages to var
        msgToVars = torch.gather(e2oLLR, 1, edgeToVar.view(channelLLR.shape[0], -1))
        msgToVars = msgToVars.reshape(channelLLR.shape[0], -1, maxColWeight)
        msgToVars = msgToVars * edgeToVarMask
        # add e2o to channel
        llr = channelLLR + torch.sum(msgToVars, 2)
        # send v2c messages to edges
        llr = torch.gather(llr, 1, oddToEven)
        llr = llr - e2oLLR  # remove the c2v message in last iter from the same edge

        # check nodes procession
        # gather messages from edges to nodes
        msgToChecks = torch.gather(llr, 1, edgeToChk.view(channelLLR.shape[0], -1))
        msgToChecks = msgToChecks.reshape(channelLLR.shape[0], -1, rowWeight - 1)
        min, _ = torch.min(torch.abs(msgToChecks), dim=2)
        sign = torch.sign(msgToChecks)
        sign = torch.prod(sign, 2)
        min *= self.alpha
        e2o = sign * min

        # marginalized output
        # gather edge messages to var
        out = torch.gather(e2o, 1, edgeToVar.view(channelLLR.shape[0], -1))
        out = out.reshape(channelLLR.shape[0], -1, maxColWeight)
        out = out * edgeToVarMask
        # add e2o to channel
        out = channelLLR + torch.sum(out, 2)
        return out, e2o


class MS_Decoder(torch.nn.Module):
    def __init__(self, check_matrix, n_iters, batch_size):
        super(MS_Decoder, self).__init__()
        self.H = torch.from_numpy(check_matrix).to(device)
        self.n_iters = n_iters
        self.n_e = int(np.sum(check_matrix))
        self.rowWeight = int(np.sum(check_matrix, 1)[0])
        self.maxColWeight = int(np.sum(check_matrix, 0).max())
        self.n_check = check_matrix.shape[0]
        # define map matrices
        # gather from (batchsize * n), result (batchsize * n_e)
        self.oddToEven = torch.zeros(self.n_e, device=device, dtype=torch.int64)
        # gather from (batchsize * n_e), result (batchsize * n_e * weight_row-1)
        self.evenToOdd = torch.zeros(self.n_e, device=device, dtype=torch.int64)
        # gather from (batchsize * n_e), result (batchsize * n * weightMax_col)
        self.edgeToVar = -torch.ones(check_matrix.shape[1], self.maxColWeight, device=device, dtype=torch.int64)
        self.edgeToVarMask = torch.zeros(check_matrix.shape[1], self.maxColWeight, device=device, dtype=torch.int64)
        # gather from (batchsize * n_e), result (batchsize * n * weight_row-1)
        self.edgeToChk = torch.zeros(self.n_e, self.rowWeight - 1, device=device, dtype=torch.int64)

        # record all edges' indices in parity check matrix, i.e the position of elements "1"
        self.rowPivot = dict()
        for i in range(self.n_check):
            self.rowPivot[i] = []
        self.colPivot = dict()
        for i in range(check_matrix.shape[1]):
            self.colPivot[i] = []
        self.edgeOrder = dict()
        edgeCount = 0
        for index, item in np.ndenumerate(check_matrix):
            if item == 1:
                self.rowPivot[index[0]] += [edgeCount]
                self.colPivot[index[1]] += [edgeCount]
                self.edgeOrder[edgeCount] = index
                edgeCount += 1
        self.configMask()
        # config mask
        self.edgeToVarMask = (self.edgeToVar != -1)
        self.edgeToVar[self.edgeToVar == -1] = 0

        # expand maps to gather
        self.oddToEven = self.oddToEven.expand(batch_size, -1)
        # gather from (batchsize * n_e), result (batchsize * n_e * weight_row-1)
        self.evenToOdd = self.evenToOdd.expand(batch_size, -1)
        # gather from (batchsize * n_e), result (batchsize * n * weightMax_col)
        self.edgeToVar = self.edgeToVar.expand(batch_size, -1, -1)
        self.edgeToVarMask = self.edgeToVarMask.expand(batch_size, -1, -1)
        # gather from (batchsize * n_e), result (batchsize * n * weight_row-1)
        self.edgeToChk = self.edgeToChk.expand(batch_size, -1, -1)

        self.layers = torch.nn.ModuleList([MSLayer(self.n_e) for _ in range(n_iters)])

    def configMask(self):
        """
        build map matrix
        """
        for i in range(self.n_e):
            chk = self.edgeOrder[i][0]
            edges = self.rowPivot[chk]
            j = 0
            for x in edges:
                if x == i:
                    continue
                else:
                    self.edgeToChk[i][j] = x
                j += 1

        for i in range(self.n_e):
            var = self.edgeOrder[i][1]
            edges = self.colPivot[var]
            j = 0
            for x in edges:
                self.edgeToVar[var][j] = x
                j += 1
        for i in range(self.n_e):
            self.oddToEven[i] = self.edgeOrder[i][1]
            self.evenToOdd[i] = self.edgeOrder[i][0]

    def forward(self, channelLLR, train=True):
        e2oLLR = torch.zeros(channelLLR.shape[0], self.n_e).to(device)
        if train:
            outputs = torch.zeros(self.n_iters, channelLLR.shape[0], channelLLR.shape[1]).to(device)
            for i in range(self.n_iters):
                w2, e2oLLR = self.layers[i](channelLLR, e2oLLR, self.maxColWeight, self.edgeToVar, self.edgeToVarMask,
                                            self.oddToEven, self.edgeToChk, self.rowWeight)
                outputs[i] = torch.sigmoid(-w2)
            return outputs
        else:
            for i in range(self.n_iters):
                w2, e2oLLR = self.layers[i](channelLLR, e2oLLR, self.maxColWeight, self.edgeToVar, self.edgeToVarMask,
                                            self.oddToEven, self.edgeToChk, self.rowWeight)
                c = - w2
                c[c > 0] = 1
                c[c < 0] = 0
                success = c @ self.H.t().double()
                success = torch.remainder(success, 2)
                success = success.sum(dim=1)
                if success.sum() == 0:
                    return c
            return c



