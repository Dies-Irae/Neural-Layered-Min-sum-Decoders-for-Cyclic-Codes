import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Layered_MS_Decoder(torch.nn.Module):
    """
    Layered MS
    The check matrix MUST be a ROW REGULAR matrix
    """
    def __init__(self, check_matrix, batch_size):
        super(Layered_MS_Decoder, self).__init__()
        self.H = torch.from_numpy(check_matrix).to(device)
        self.w_r = int(torch.sum(self.H[0]).item())  # assume every row has the same row weight
        # a tensor that each row represents a cn and contains associated vns
        self.H_compact = torch.zeros(self.H.shape[0], self.w_r, device=device, dtype=torch.int64)
        self.bsize = batch_size
        x = 0
        for index, item in np.ndenumerate(check_matrix):
            if item == 1:  # if it is an edge
                self.H_compact[index[0]][x] = index[1]
                x = (x + 1) % self.w_r
        """
        During Min-Sum processing, each vn receives information form vns which are connected with the same cn
        That means, given fixed row weight w_r, each vn receives w_r - 1 information(the excluded one is the vn itself)
        """
        self.mask = torch.zeros(self.w_r, self.w_r-1, device=device, dtype=torch.long)
        for i in range(self.w_r):
            num = 0
            for j in range(self.w_r-1):
                if i == j:
                    num += 1
                self.mask[i][j] = num
                num += 1
        self.mask = self.mask.expand(batch_size, -1, -1)  # expand to process a batch simultaneously

    def update(self, vn_llr, cn, pre_c2v):
        # subtract the messages from the same cn in the last iteration
        vn_llr[:, self.H_compact[cn]] -= pre_c2v[:, cn, :]

        v2c = vn_llr[:, self.H_compact[cn]]  # extract llrs of vns that are tied with the cn

        # using mask to copy vn_llr
        # information w_r times and remove the information of the information of vn itself
        # then, we can compute min-sum simply by doing "min" and "sign" along the new dimension
        """
                        [b, c, d]
        [a, b, c, d] -->[a, c, d]
                        [a, b, d]
                        [b, c, d]
        """
        v2c_expand = torch.gather(v2c, 1, self.mask.view(v2c.shape[0], -1))
        v2c_expand = v2c_expand.reshape(v2c.shape[0], self.w_r, self.w_r - 1)

        # sign * |v2c|
        amps, _ = torch.min(torch.abs(v2c_expand), dim=2)
        sign = torch.sign(v2c_expand)
        sign = torch.prod(sign, 2)
        c2v = sign * amps * 1
        c2v = torch.clamp(c2v, -20, 20)

        pre_c2v[:, cn, :] = c2v  # store c2v of this iteration, it will soon be used in the next iteration
        vn_llr[:, self.H_compact[cn]] = vn_llr[:, self.H_compact[cn]] + c2v
        return vn_llr, pre_c2v

    def forward(self, channel_llr, iters, cn_order):
        c2v = torch.zeros(self.bsize, self.H.shape[0], self.w_r, device=device)
        vn_llr = channel_llr.clone()
        for _ in range(iters):
            for cn in cn_order:
                vn_llr, c2v = self.update(vn_llr, cn, c2v)
            c = - vn_llr
            c[c > 0] = 1
            c[c < 0] = 0
            success = c @ self.H.t().double()
            success = torch.remainder(success, 2)
            success = success.sum(dim=1)
            if success.sum() == 0:  # early stop
                return c
        return c




