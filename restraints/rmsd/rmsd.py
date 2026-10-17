import numpy as np
#import torch

def centroid(x, w):
    """
    x: atom coordinate set with shape [N,3]
    w: atom weight set with shape [N,1]
    
    returns centroid with shape [1,3] of atom set x given weights w
    """
    return (w[:, None] * x).sum(0) / w.sum()


def cross_covariance(xc, xc0, w):
    """
    xc: atom coordinate set w.r.t. centroid with shape [N, 3]
    xc0: atom coordinate set w.r.t. centroid with shape [N, 3]
    w: atom weight set with shape [N,1]

    returns the cross-covariance matrix B from the derivation of fitting cost function phi(R) = A - 2<R,B>_F 
    """
    N = len(xc)
    
    B = torch.zeros(3, 3)
    for i in range(N):
        for j in range(3):
            for k in range(3):
                B[j, k] += w[i] * xc[i, j] * xc0[i, k]
    return B


def quaternion_matrix(B):
    """
    B: cross-covariance 3x3 matrix

    returns symmetrix 4x4 matrix F from <R(q),B>_F = q^T * F * q
    """
    rows = [
            [ B[0,0] + B[1,1] + B[2,2],   B[2,1] - B[1,2],            B[0,2] - B[2,0],            B[1,0] - B[0,1]           ],
            [ B[2,1] - B[1,2],            B[0,0] - B[1,1] - B[2,2],   B[0,1] + B[1,0],            B[0,2] + B[2,0]           ],
            [ B[0,2] - B[2,0],            B[0,1] + B[1,0],           -B[0,0] + B[1,1] - B[2,2],   B[1,2] + B[2,1]           ],
            [ B[1,0] - B[0,1],            B[0,2] + B[2,0],            B[1,2] + B[2,1],           -B[0,0] - B[1,1] + B[2,2]  ],
        ]
    return torch.stack([torch.stack(r) for r in rows])


def quaternion_to_rotation(q):
    """
    q: quaternion

    returns physical rotation matrix
    """
    q0, q1, q2, q3 = q
    rows = [
            [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3),             2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3),             q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2),             2*(q2*q3 + q0*q1),             q0*q0 - q1*q1 - q2*q2 + q3*q3],
            ]
    return torch.stack([torch.stack(row) for row in r])


def fit(x, x0, w):
    c = centroid(x, w)
    c0 = centroid(x0, w)
    B = cross_covariance(x - c, x0 - c0, w)
    F = quaternion_matrix(B)
    evals, evecs = torch.linalg.eigh(F) # ascending eigenvalues
    q = evecs[:, -1]
    R = quat_to_rot(q)
    return R, c, c0, dict(B=B, F=F, evals=evals, q=q)


def restraint(xc, xc0, w, R):
    """
    flat bottom harmomic
   
    xc: atom coordinate set w.r.t. centroid OF RESTRAINING SET with shape [N, 3]
    xc0:

    """
    delta = xc - (xc0 @ R.T) # residual from cost eq., xc0*R.T = xc0T*R= R*xc0, [N, 3]
    u = (w * (delta**2.sum(1))).sum() # square the terms of delta, sum along each atom (axis=1), multiply each atom square delta sum by weight, sum
    D = torch.sqrt(u / w.sum()) 



def main():
    pass


if __name__ == "__main__":
    main()
