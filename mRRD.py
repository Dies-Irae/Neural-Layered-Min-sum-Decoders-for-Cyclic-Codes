import pickle
import torch
complexity_profile = True
class mRRD(torch.nn.Module):
    def __init__(self, code, b_size, decoder, width, length, device):
        """
        :param code: An object of Linear Code class
        :param decoder: Inner decoder function like bp, ms, etc.
        :param b_size: number of codewords to be decoded simultaneously
        """
        super(mRRD, self).__init__()
        self.codeLen = code.n
        self.checks = list(range(code.n))
        self.decoder = decoder
        self.batchSize = b_size
        self.d_min = code.d
        self.width = width
        self.perm_index = torch.arange(self.codeLen, device=device).repeat(self.batchSize, 1)
        with open("perms/" + code.family + str(code.n)+"_" + str(code.k) + "Permutations.list", "rb") as f:
            self.PermGrp = pickle.load(f)
        self.PermGrp = torch.LongTensor(self.PermGrp).to(device)
        self.length = length
        self.complexity = torch.zeros(b_size, device=device)
        self.device = device

    def Lambda(self, r, z, v):
        """
        correlation discrepancy between r and v
        :param r: received vector (batchSize * n)
        :param z: hard-decision of r (batchSize * n)
        :param v: codeword (batchSize * n)
        """
        d1 = torch.logical_xor(z, v)
        lambda_r_v = torch.sum(d1 * torch.abs(r), dim=1)
        return lambda_r_v

    def MLCriterion(self, r, z, v, lambda_r_v):
        """
        Early Quit Criterion for Chase Algorithm. From: L. Wang, Y. Li, T. Truong and T. Lin,
        "On Decoding of the (89, 45, 17) Quadratic Residue Code,"
        in IEEE Transactions on Communications, vol. 61, no. 3, pp. 832-841, March 2013
        """
        d1 = torch.logical_xor(z, v)
        d0 = torch.logical_not(d1)
        n_v = torch.sum(d1, dim=1)
        delta = self.d_min - n_v
        sorted_r = r * d0
        sorted_r[sorted_r == 0] = float("inf")
        delta[delta<0] = 0
        GT = torch.zeros_like(delta, device=self.device)
        if delta.sum() == 0:  # early return
            return lambda_r_v <= GT
        # faster(vectorized) but contains redundant addition
        sorted_r = torch.sort(torch.abs(sorted_r))[0]
        sums = sorted_r.cumsum(dim=1)
        GT = sums[torch.arange(delta.shape[0]), delta]
        # slower version
        # for i in range(delta.shape[0]):
        #     GT[i] = torch.sort(torch.abs(sorted_r))[0][:, :delta[i]].sum()
        return lambda_r_v <= GT

    def forward(self, r):
        self.complexity = torch.zeros_like(self.complexity, device=self.device)
        y = r.clone()
        y[y > 0] = 0
        y[y < 0] = 1
        minDist = float("inf") * torch.ones(r.shape[0], dtype=torch.double, device=self.device)
        minCodeword = torch.zeros_like(r)
        for i in range(self.width):  # k is the number of bits to be flipped in this round
            rr = r.clone()
            for j in range(self.length):
                self.perm_index = self.PermGrp[torch.randint(self.PermGrp.shape[0], (r.shape[0],))]
                rr = torch.gather(rr, 1, self.perm_index)
                rr, Y, success_flags = self.decoder(rr, self.checks, j)
                Y = Y.scatter(1, self.perm_index, Y)
                rr = rr.scatter(1, self.perm_index, rr)
                if torch.sum(success_flags) == self.batchSize:
                    break
            if torch.sum(success_flags) < self.batchSize:
                Y *= success_flags.unsqueeze(-1)  # mask all failed decoding results to 0
            distances = self.Lambda(r, y, Y)
            newMin = distances <= minDist  # find and replace old distances using newer and smaller distances
            minDist[newMin] = distances[newMin]
            minCodeword[newMin] = Y[newMin]
            MLFlags = self.MLCriterion(r, y, minCodeword, minDist)
            if complexity_profile:
                self.complexity += MLFlags
            if MLFlags.sum() == self.batchSize:
                print(5*(i + 2 - torch.mean(self.complexity)))
                return minCodeword
        print(5*(self.width + 1 - torch.mean(self.complexity)))
        return minCodeword
