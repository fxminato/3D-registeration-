import numpy as np
from math import pi
from numpy.fft import fftn, ifftn
from scipy.ndimage import convolve
from skimage.transform import warp
import mrcfile
from skimage.transform import downscale_local_mean

def LDDMM(I0, I1, T = 32, maxiter = 200, lr = 1e-3, sigma = 1.0, alpha = 1.0, gamma = 1.0, eps = 2):
    r'''
    LDDMM method for image registration.

        min_v |(alpha\Delta+\gamma)v|^2 + 1/sigma^2 |I0 * Phi1 - I1|^2

    Parameters
    ==========
    I0, I1 : numpy.ndarray, two images.
    T : int, number of ticks.
    maxiter : int, maximal number of iterations.
    lr : float, learning rate.
    sigma : float, super parameter.
    alpha : float, super parameter.
    gamma : float, super parameter.

    Returns
    =======
    '''
    assert I0.shape == I1.shape
    (nx, ny, nz) = I0.shape

    def _operator_L(vt):
        r'''
        The Cauchy-Navier operator.
        '''
        kernel= np.array([[[0., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 0.]],
                      [[0., 1., 0.],
                       [1., -6.,1.],
                       [0., 1., 0.]],
                      [[0., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 0.]]],dtype = np.float64)
        laplacian = np.empty_like(vt, dtype = np.float64)
        laplacian[0] = convolve(vt[0], kernel, mode = 'wrap')
        laplacian[1] = convolve(vt[1], kernel, mode = 'wrap')
        laplacian[2] = convolve(vt[2], kernel, mode='wrap')
        return -alpha * laplacian + gamma * vt

    def _operator_K(g):
        r'''
        K = (L^T L)^{-1} = L^{-2}.
        '''
        a = 1 - np.cos(np.linspace(0, 2 * pi, nx, endpoint = False))
        b = 1 - np.cos(np.linspace(0, 2 * pi, ny, endpoint = False))
        c = 1 - np.cos(np.linspace(0, 2 * pi, ny, endpoint = False))
        A = 2*alpha * (a.reshape((nx, 1,1)) + b.reshape((1, ny,1))+c.reshape((1,1,nz))) + gamma
        G = fftn(g)
        F = G / (A ** 2)
        f = ifftn(F)
        return np.real(f)

    def _reparametrize(v):
        length = np.empty(T, dtype = np.float64)
        for t in range(T):
            length[t] = np.linalg.norm(_operator_L(v[t]))
            v[t] /= T * length[t]
        v *= np.sum(length)

    def _deform(img, phi):
        r'''
        Apply a deformation phi to an img.

        Also viewed as img \circ phi.
        '''
        result = np.empty_like(img, np.float64)
        for c in range(result.shape[0]):
            result[c] = warp(img[c], phi, mode = 'edge')
        return result

    def _forward_flow(v):
        # Phi0[t] can be viewed as a grid, also a diffeomorphism.
        # Phi0[t] as grid : (Phi[t, 0, x, y, z], Phi[t, 1, x, y, z]),Phi[t, 2, x, y, z]).
        # Phi0[t] as diffeomorphism : (x, y, z) -> (Phi0[t, 0, x, y, z], Phi0[t, 1, x, y, z],Phi[t, 2, x, y, z])).
        Phi0 = np.empty((T, 3, nx, ny, nz), dtype = np.float64)

        # Phi0[0] = id, i.e.,
        # Phi0[0, 0, x, y , z] = x, Phi0[0, 1, x, y, z] = y, Phi[0, 1, x, y, z]=z)
        # Using numpy's index trick to generate identity mapping.
        Phi0[0] = np.mgrid[:nx, :ny, :nz]
        # Calculate Phi0[t] from Phi0[t - 1].
        for t in range(1, T):
            # Calculate alpha by an iterative formula.
            alpha = np.zeros((3, nx, ny,nz), dtype = np.float64)
            for _ in range(5):
                alpha = _deform(v[t - 1], Phi0[0] - 0.5 * alpha)

            # Phi0[t] : x -> Phi0[t - 1](x - alpha).
            Phi0[t] = _deform(Phi0[t - 1], Phi0[0] - alpha)

        return Phi0

    def _backward_flow(v):
        Phi1 = np.empty((T, 3, nx, ny,nz), dtype = np.float64)
        Phi1[T - 1] = np.mgrid[:nx, :ny,:nz]
        for t in range(T - 2, -1, -1):
            alpha = np.zeros((3, nx, ny,nz), dtype = np.float64)
            for _ in range(5):
                alpha = _deform(v[t], Phi1[T - 1] + 0.5 * alpha)
            Phi1[t] = _deform(Phi1[t + 1], Phi1[T - 1] + alpha)

        return Phi1

    def _forward_deform(I0, Phi0):
        J0 = np.empty((T, 1, nx, ny,nz), dtype = np.float64)
        for t in range(T):
            J0[t] = _deform(I0, Phi0[t])
        return J0

    def _backward_deform(I1, Phi1):
        J1 = np.empty((T, 1, nx, ny,nz), dtype = np.float64)
        for t in range(T - 1, -1, -1):
            J1[t] = _deform(I1, Phi1[t])
        return J1

    def _grad(data):
        wy = np.array([[[1], [0], [-1]]],dtype = np.float64)
        wx = np.array([[[1]], [[0]], [[-1]]],dtype=np.float64)
        wz = np.array([[[1., 0., -1.]]], dtype=np.float64)
        grad = np.empty((3, nx, ny,nz), dtype = np.float64)
        grad[0] = convolve(data, wx, mode = 'wrap')
        grad[1] = convolve(data, wy, mode = 'wrap')
        grad[2] = convolve(data, wz, mode = 'wrap')
        return grad

    def _image_grad(J0):
        dJ0 = np.empty((T, 3, nx, ny,nz), dtype = np.float64)
        for t in range(T):
            dJ0[t] = _grad(J0[t, 0])
        return dJ0

    def _jacobian_derterminant(Phi1):
        detPhi1 = np.empty((T, 1, nx, ny,nz), dtype = np.float64)
        for t in range(T):
            dx = _grad(Phi1[t, 0])
            dy = _grad(Phi1[t, 1])
            dz = _grad(Phi1[t, 2])
            detPhi1[t] = dx[0] * dy[1] * dz[2]+dx[1] * dy[2]*dz[0]+dx[2]*dy[0]*dz[1]-\
                         dx[2] * dy[1] * dz[0]-dx[1] * dy[0]*dz[2]-dx[0]*dy[2]*dz[1]
        return detPhi1

    # All data, including images, velocity fields, diffeomorphisms,
    # have shape (c, nx, ny,nz), where c is the number of channels.
    I0 = I0.reshape((1, nx, ny, nz))
    I1 = I1.reshape((1, nx, ny, nz))
    # Initialize.
    v = np.zeros((T, 3, nx, ny, nz), dtype = np.float64)
    dv = np.zeros((T, 3, nx, ny, nz), dtype = np.float64)
    Phi0 = None
    Phi1 = None
    J0 = None
    J1 = None

    for round in range(1, maxiter + 1):
        # Gradient descent.
        v -= lr * dv
        if round % 10 == 0:
            _reparametrize(v)
        #print(v[0])

        # Calculate forward and backward flows.
        Phi0 = _forward_flow(v)
        Phi1 = _backward_flow(v)

        # Calculate forward and backward deformations.
        J0 = _forward_deform(I0, Phi0)
        J1 = _backward_deform(I1, Phi1)

        # Calculate image gradient.
        dJ0 = _image_grad(J0)

        # Calculate Jacobian determinant of the transformation.
        detPhi1 = _jacobian_derterminant(Phi1)

        # Calculate the gradient.
        for t in range(T):
            # Using numpy's broadcast, we can multiply a (1, nx, ny,nz)-array with a (2, nx, ny,nz)-array.
            dv[t] = 2 * v[t] - 2 / sigma ** 2 * _operator_K(detPhi1[t] * (J0[t] - J1[t]) * dJ0[t])
        if np.linalg.norm(dv) < eps:
            print("Gradient norm below threshold, stopping.")
            break

        # Calculate new energy
        E1 = sum(np.linalg.norm(_operator_L(v[t])) for t in range(T))
        E2 = 1 / sigma ** 2 * np.linalg.norm(I1[0] - J0[-1]) ** 2
        E = E1 + E2

        print(f'iteration {round:3d}, energy {E:4.2f} = {E1:4.2f} + {E2:4.2f}.')

    return (v, Phi0, Phi1, J0, J1)

if __name__ == "__main__":
    # load greyscale images
    mrc = mrcfile.open('./emd_21546.map', permissive=True, mode='r')
    i0=mrc.data
    mrc = mrcfile.open('./emd_21546_ffit_into_emd_21547.mrc', permissive=True, mode='r')
    i1=mrc.data
    i0=downscale_local_mean(i0,(2,2,2))
    i0 = downscale_local_mean(i0, (2, 2, 2))
    i1=downscale_local_mean(i1,(2,2,2))
    i1 = downscale_local_mean(i1, (2, 2, 2))
    n=i0.shape[0]
    v, Phi0, Phi1, J0, J1=LDDMM(i0, i1, T = 10, maxiter = 100, sigma = 0.1, alpha = 1, gamma = 1, lr = 0.5)
