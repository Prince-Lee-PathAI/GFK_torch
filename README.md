# GFK_torch
This is an PyTorch Implementation of Geodesic Flow Kernel (CVPR2012) wrapped from
[GFK_numpy](https://www.idiap.ch/software/bob/docs/bob/bob.learn.linear/stable/_modules/bob/learn/linear/GFK.html#GFKMachine).

*Note*:In the context of GFK algorithm, the Generalized SVD of matrix pair $\mathbf(A,B)$ can be degenerated into a SVD problem of  $\mathbf{A}^{-1}B$.\

Therefore, please do not use this code to do the standard GSVD.
