# This is an implementation of GFK in Pytorch with reference to
# Gong B, Shi Y, Sha F, et al. Geodesic flow kernel for unsupervised domain adaptation. CVPR2012
# Note: The numpy version can be found in package bob.learn.linear.GFK
import torch


def degenerated_GSVD(A,B):
    """
    Implementation of the Generalized SVD algorithm where matrix A is invertible

    :param A:  matrix A with shape [n,n] 
    :param B:  matrix B with shape [b,n]
    :return: [V1,V2,V,Gam,Sig] where A = V1*Gam*V^T and B = V2*Sig*V^T are both SVD.
    """
    B_InvA = B @ torch.linalg.inv(A)
    V2,S,V1_h = torch.linalg.svd(B_InvA,full_matrices=False)
    V1 = V1_h.H
    # assert torch.dist(B_InvA, V2[:, :A.shape[1]] @ torch.diag(S) @ V1_h) < 1e-3

    Gam_i = 1 / torch.sqrt(1+S**2)
    Sig_i = S / torch.sqrt(1+S**2)

    Sig = torch.zeros_like(B)# shape [b,n]
    for i in range(len(S)):
        Sig[i,i] = Sig_i[i]
    V = A.T @ V1 @ torch.diag(1 / Gam_i)
    Gam = torch.diag(Gam_i)

    # assert torch.dist(A, V1 @ Gam @ V.H) < 1e-3
    # assert torch.dist(B, V2 @ Sig @ V.H) < 1e-3
    return V1,V2,V,Gam,Sig

def null_space(A):
    # A is colum Orthogonal matrix
    A_complement, _ = torch.linalg.qr(A,mode='complete')
    return A_complement


def sqrt_SymTensor(t):
    eigvals, eigvecs = torch.linalg.eigh(t)
    eigvals = eigvals.to(torch.complex64)
    eigvecs = eigvecs.to(torch.complex64)
    t_half = eigvecs @ torch.diag(eigvals.pow(0.5)) @ eigvecs.T
    t_half = torch.real(t_half)
    return t_half

class GFK:
    def __init__(self, dim=20):
        """
        Init func
        :param dim: dimension after GFK
        """
        self.dim = dim
        self.eps = 1e-20

    def znorm(self, data):
        """
        Z-Normaliza
        """
        mu = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        data = (data - mu) / std
        return data

    def pca_proj(self,X, threshold=0.99):
        """
        Param：
        X : torch.Tensor (n, d)
        threshold : float

        return：
        P : torch.Tensor (d, k) - Projection Matrix
        """

        X_mean = X.mean(dim=0, keepdim=True)
        X_centered = X - X_mean
        cov_matrix = torch.mm(X_centered.T, X_centered) / (X.shape[0] - 1)
        U, S, Vh = torch.linalg.svd(cov_matrix)

        explained_variance_ratio = S / S.sum()
        cumulative_variance = torch.cumsum(explained_variance_ratio, dim=0)

        k = torch.where(cumulative_variance >= threshold)[0][0].item() + 1

        P = U[:, :k]


        return P

    def fit(self, Xs, Xt, norm_inputs=None):
        """
        Obtain the kernel G
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :param norm_inputs: normalize the inputs or not
        :return: GFK kernel G
        """
        if norm_inputs:
            source = self.znorm(Xs)
            target = self.znorm(Xt)
        else:
            source = Xs
            target = Xt
        Ps = self.pca_proj(source,threshold=0.99)
        Pt = self.pca_proj(target, threshold=0.99)
        Ps_complement = null_space(Ps)[:, Ps.shape[1]:]
        Ps = torch.hstack((Ps, Ps_complement))

        # assert torch.dist(torch.eye(Ps.shape[0]), Ps.T @ Ps) < 1e-3

        Pt = Pt[:, :self.dim]
        N = Ps.shape[1]
        dim = Pt.shape[1]

        # Principal angles between subspaces
        QPt = Ps.T @ Pt

        # [V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
        A = QPt[0:dim, :]
        # assert torch.dist(torch.linalg.inv(A) @ A, torch.eye(dim)) < 1e-3 # check the orthogonality of A
        B = QPt[dim:, :]
        V1, V2, V, Gam, Sig = degenerated_GSVD(A, B) # Since A is invertible, GSVD can be degenerated in a simpler way

        V2 = -V2
        # print(torch.diag(torch.diag(Gam)))
        theta = torch.arccos(torch.diag(Gam))

        # Equation (6)
        B1 = torch.diag(0.5 * (1 + (torch.sin(2 * theta) / (2. * torch.maximum(theta, torch.tensor(self.eps))))))
        B2 = torch.diag(0.5 * ((torch.cos(2 * theta) - 1) / (2 * torch.maximum(theta, torch.tensor(self.eps)))))
        B3 = B2.clone()  # B3 = B2 
        B4 = torch.diag(0.5 * (1 - (torch.sin(2 * theta) / (2. * torch.maximum(theta, torch.tensor(self.eps))))))

        # Equation (9) of the suplementary matetial
        delta1_1 = torch.hstack((V1, torch.zeros((dim, N - dim)).cuda()))
        delta1_2 = torch.hstack((torch.zeros((N - dim, dim)).cuda(), V2))
        delta1 = torch.vstack((delta1_1, delta1_2))

        delta2_1 = torch.hstack((B1, B2, torch.zeros((dim, N - 2 * dim)).cuda()))
        delta2_2 = torch.hstack((B3, B4, torch.zeros((dim, N - 2 * dim)).cuda()))
        delta2_3 = torch.zeros((N - 2 * dim, N)).cuda()
        delta2 = torch.vstack((delta2_1, delta2_2, delta2_3))

        delta3_1 = torch.hstack((V1, torch.zeros((dim, N - dim)).cuda()))
        delta3_2 = torch.hstack((torch.zeros((N - dim, dim)).cuda(), V2))
        delta3 = torch.vstack((delta3_1, delta3_2)).T

        delta = delta1 @ delta2 @ delta3
        G = Ps @ delta @ Ps.T
        sqG = torch.real(sqrt_SymTensor(G))
        Xs_new = (sqG @ Xs.T).T
        Xt_new = (sqG @ Xt.T).T

        return G, Xs_new, Xt_new



if __name__ == "__main__":
    A = torch.randn(10, 768).cuda()
    B = torch.randn(10, 768).cuda()
    gfk = GFK(dim=20)
    G, A_new, B_new = gfk.fit(A,B,norm_inputs=True)
    print(A_new)
    print(B_new)
    print(G)
    print(G.shape, A_new.shape, B_new.shape)
