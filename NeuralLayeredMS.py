import torch
import torch.nn as nn
import numpy as np


class Layered_MS_Decoder(torch.nn.Module):
    """
    Layered MS(layer-wise weight)
    The check matrix MUST be a ROW REGULAR matrix
    """
    def __init__(self, check_matrix, iter_num, batch_size, device):
        super(Layered_MS_Decoder, self).__init__()
        self.iter = iter_num
        self.H = torch.from_numpy(check_matrix).to(device)
        self.w_r = int(torch.sum(self.H[0]).item())  # assume every row has the same row weight
        # a tensor that each row represents a cn and contains associated vns
        self.H_compact = torch.zeros(self.H.shape[0], self.w_r, device=device, dtype=torch.int64)
        self.bsize = batch_size
        self.device = device
        x = 0
        self.alphas = nn.Parameter(2*torch.ones(iter_num), requires_grad=True)
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

    def update(self, vn_llr, cn, pre_c2v, i_num):
        # subtract the messages from the same cn in the last iteration
        vn_llr[:, self.H_compact[cn]] = vn_llr[:, self.H_compact[cn]] - pre_c2v[:, cn, :]

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
        c2v = sign * amps * torch.sigmoid(self.alphas[i_num])
        c2v = torch.clamp(c2v, -20, 20)

        pre_c2v[:, cn, :] = c2v  # store c2v of this iteration, it will soon be used in the next iteration
        vn_llr[:, self.H_compact[cn]] = vn_llr[:, self.H_compact[cn]] + c2v
        return vn_llr, pre_c2v

    def test(self, channel_llr, cn_order):
        c2v = torch.zeros(self.bsize, self.H.shape[0], self.w_r, device=self.device)
        vn_llr = channel_llr.clone()
        for i in range(self.iter):
            for cn in cn_order:
                vn_llr, c2v = self.update(vn_llr, cn, c2v, i)
            c = - vn_llr
            c[c > 0] = 1
            c[c < 0] = 0
            success = c @ self.H.t().double()
            success = torch.remainder(success, 2)
            success = success.sum(dim=1)
            if success.sum() == 0:  # early stop
                return vn_llr, c, torch.logical_not(success)
        return vn_llr, c, torch.logical_not(success)

    def forward(self, channel_llr, cn_order):
        c2v = torch.zeros(self.bsize, self.H.shape[0], self.w_r, device=self.device)
        out = torch.zeros(self.iter, self.bsize, self.H.shape[1], device=self.device)
        vn_llr = channel_llr.clone()
        for i in range(self.iter):
            for cn in cn_order:
                vn_llr, c2v = self.update(vn_llr, cn, c2v, i)
            out[i] = vn_llr
        return out

    def step(self, llr, cn_order, step):
        c2v = torch.zeros(self.bsize, self.H.shape[0], self.w_r, device=self.device)
        vn_llr = llr.clone()
        for cn in cn_order:
            vn_llr, c2v = self.update(vn_llr, cn, c2v, step)
        c = - vn_llr
        c[c > 0] = 1
        c[c < 0] = 0
        success = c @ self.H.t().double()
        success = torch.remainder(success, 2)
        success = success.sum(dim=1)
        if success.sum() == 0:  # early stop
            return vn_llr, c, torch.logical_not(success)
        return vn_llr, c, torch.logical_not(success)


class Layered_MSEdge_Decoder(torch.nn.Module):
    """
    Layered MS(edge-wise weight)
    The check matrix MUST be a ROW REGULAR matrix
    """
    def __init__(self, check_matrix, iter_num, batch_size, device):
        super(Layered_MSEdge_Decoder, self).__init__()
        self.iter = iter_num
        self.device = device
        self.H = torch.from_numpy(check_matrix).to(device)
        self.w_r = int(torch.sum(self.H[0]).item())  # assume every row has the same row weight
        # a tensor that each row represents a cn and contains associated vns
        self.H_compact = torch.zeros(self.H.shape[0], self.w_r, device=device, dtype=torch.int64)
        self.bsize = batch_size
        x = 0
        self.alphas = nn.Parameter(torch.zeros(iter_num, self.H.shape[0], self.w_r, device=device), requires_grad=True)
        # self.betas = nn.Parameter(torch.zeros(iter_num, self.H.shape[0], self.w_r), requires_grad=True)
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

    def update(self, vn_llr, cn, pre_c2v, i_num):
        # subtract the messages from the same cn in the last iteration
        vn_llr[:, self.H_compact[cn]] = vn_llr[:, self.H_compact[cn]] - pre_c2v[:, cn, :]

        v2c = vn_llr[:, self.H_compact[cn]]  # extract llrs of vns that are tied with the cn
        # v2c = v2c * torch.sigmoid(self.betas[i_num, cn])

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
        c2v = sign * amps * torch.sigmoid(self.alphas[i_num, cn])
        c2v = torch.clamp(c2v, -20, 20)

        pre_c2v[:, cn, :] = c2v  # store c2v of this iteration, it will soon be used in the next iteration
        vn_llr[:, self.H_compact[cn]] = vn_llr[:, self.H_compact[cn]] + c2v
        return vn_llr, pre_c2v

    def test(self, channel_llr, cn_order):
        c2v = torch.zeros(self.bsize, self.H.shape[0], self.w_r, device=self.device)
        vn_llr = channel_llr.clone()
        for i in range(self.iter):
            for cn in cn_order:
                vn_llr, c2v = self.update(vn_llr, cn, c2v, i)
            c = - vn_llr
            c[c > 0] = 1
            c[c < 0] = 0
            success = c @ self.H.t().double()
            success = torch.remainder(success, 2)
            success = success.sum(dim=1)
            if success.sum() == 0:  # early stop
                return vn_llr, c, torch.logical_not(success)
        return vn_llr, c, torch.logical_not(success)

    def forward(self, channel_llr, cn_order):
        c2v = torch.zeros(self.bsize, self.H.shape[0], self.w_r, device=self.device)
        vn_llr = channel_llr.clone()
        out = torch.zeros(self.iter, self.bsize, self.H.shape[1], device=self.device)
        for i in range(self.iter):
            for cn in cn_order:
                vn_llr, c2v = self.update(vn_llr, cn, c2v, i)
            out[i] = vn_llr
        return out

# class Layered_MSTied_Decoder(torch.nn.Module):
#     """
#     Layered MS(CN-wise weight)
#     The check matrix MUST be a ROW REGULAR matrix
#     """
#     def __init__(self, check_matrix, iter_num, batch_size):
#         super(Layered_MSTied_Decoder, self).__init__()
#         self.iter = iter_num
#         self.H = torch.from_numpy(check_matrix).to(device)
#         self.w_r = int(torch.sum(self.H[0]).item())  # assume every row has the same row weight
#         # a tensor that each row represents a cn and contains associated vns
#         self.H_compact = torch.zeros(self.H.shape[0], self.w_r, device=device, dtype=torch.int64)
#         self.bsize = batch_size
#         x = 0
#         self.alphas = nn.Parameter(torch.zeros(iter_num, self.H.shape[0]), requires_grad=True)
#         for index, item in np.ndenumerate(check_matrix):
#             if item == 1:  # if it is an edge
#                 self.H_compact[index[0]][x] = index[1]
#                 x = (x + 1) % self.w_r
#         """
#         During Min-Sum processing, each vn receives information form vns which are connected with the same cn
#         That means, given fixed row weight w_r, each vn receives w_r - 1 information(the excluded one is the vn itself)
#         """
#         self.mask = torch.zeros(self.w_r, self.w_r-1, device=device, dtype=torch.long)
#         for i in range(self.w_r):
#             num = 0
#             for j in range(self.w_r-1):
#                 if i == j:
#                     num += 1
#                 self.mask[i][j] = num
#                 num += 1
#         self.mask = self.mask.expand(batch_size, -1, -1)  # expand to process a batch simultaneously
#
#     def update(self, vn_llr, cn, pre_c2v, i_num):
#         # subtract the messages from the same cn in the last iteration
#         vn_llr[:, self.H_compact[cn]] = vn_llr[:, self.H_compact[cn]] - pre_c2v[:, cn, :]
#
#         v2c = vn_llr[:, self.H_compact[cn]]  # extract llrs of vns that are tied with the cn
#
#         # using mask to copy vn_llr
#         # information w_r times and remove the information of the information of vn itself
#         # then, we can compute min-sum simply by doing "min" and "sign" along the new dimension
#         """
#                         [b, c, d]
#         [a, b, c, d] -->[a, c, d]
#                         [a, b, d]
#                         [b, c, d]
#         """
#         v2c_expand = torch.gather(v2c, 1, self.mask.view(v2c.shape[0], -1))
#         v2c_expand = v2c_expand.reshape(v2c.shape[0], self.w_r, self.w_r - 1)
#
#         # sign * |v2c|
#         amps, _ = torch.min(torch.abs(v2c_expand), dim=2)
#         sign = torch.sign(v2c_expand)
#         sign = torch.prod(sign, 2)
#         c2v = sign * amps * torch.sigmoid(self.alphas[i_num, cn])
#         c2v = torch.clamp(c2v, -20, 20)
#
#         pre_c2v[:, cn, :] = c2v  # store c2v of this iteration, it will soon be used in the next iteration
#         vn_llr[:, self.H_compact[cn]] = vn_llr[:, self.H_compact[cn]] + c2v
#         return vn_llr, pre_c2v
#
#     def test(self, channel_llr, cn_order):
#         c2v = torch.zeros(self.bsize, self.H.shape[0], self.w_r, device=device)
#         vn_llr = channel_llr.clone()
#         for i in range(self.iter):
#             for cn in cn_order:
#                 vn_llr, c2v = self.update(vn_llr, cn, c2v, i)
#             c = - vn_llr
#             c[c > 0] = 1
#             c[c < 0] = 0
#             success = c @ self.H.t().double()
#             success = torch.remainder(success, 2)
#             success = success.sum(dim=1)
#             if success.sum() == 0:  # early stop
#                 return c
#         return c
#
#     def forward(self, channel_llr, cn_order):
#         c2v = torch.zeros(self.bsize, self.H.shape[0], self.w_r, device=device)
#         out = torch.zeros(self.iter, self.bsize, self.H.shape[1], device=device)
#         vn_llr = channel_llr.clone()
#         for i in range(self.iter):
#             for cn in cn_order:
#                 vn_llr, c2v = self.update(vn_llr, cn, c2v, i)
#                 out[i] = vn_llr
#         return torch.sigmoid(-out)

