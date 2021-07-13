import sys
import cupy as cp
from good_gradient import *
from good_laplacian import *
from kernel_wrap import *
import numpy as np
from math import pi
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
        lapla = cp.empty_like(vt, dtype = cp.float64)
        lapla[0] = laplacian(vt[0])
        lapla[1] = laplacian(vt[1])
        lapla[2] = laplacian(vt[2])
        return -alpha * lapla + gamma * vt

    def _operator_K(g):
        r'''
        K = (L^T L)^{-1} = L^{-2}.
        '''
        a = 1 - cp.cos(cp.linspace(0, 2 * pi, nx, endpoint = False))
        b = 1 - cp.cos(cp.linspace(0, 2 * pi, ny, endpoint = False))
        c = 1 - cp.cos(cp.linspace(0, 2 * pi, ny, endpoint = False))
        A = 2*alpha * (a.reshape((nx, 1,1)) + b.reshape((1, ny,1))+c.reshape((1,1,nz))) + gamma
        G = cp.fft.fftn(g)
        F = G / (A ** 2)
        f = cp.fft.ifftn(F)
        return cp.real(f)

    def _reparametrize(v):
        length = cp.empty(T, dtype = cp.float64)
        for t in range(T):
            length[t] = cp.linalg.norm(_operator_L(v[t]))
            v[t] /= T * length[t]
        v *= cp.sum(length)

    def _deform(img, phi):
        r'''
        Apply a deformation phi to an img.
        Also viewed as img \circ phi.
        '''
        result = cp.empty_like(img, cp.float64)
        for c in range(result.shape[0]):
            m0=img[c]
            m1=phi[0, :, :, :]
            m2=phi[1, :, :, :]
            m3=phi[2, :, :, :]
            result[c] = wrap(m0, m1,m2,m3)
        return result

    def _deform1(img, phi):
        r'''
        Apply a deformation phi to an img.
        Also viewed as img \circ phi.
        '''
        result = cp.empty_like(img, cp.float64)
        for c in range(result.shape[0]):
            m0=img[c].copy()
            print(m0.dtype)
            m1=phi[0, :, :, :].copy()
            m2=phi[1, :, :, :].copy()
            m3=phi[2, :, :, :].copy()
            print(f'initial{cp.linalg.norm(m0)}')
            result[c] = wrap(m0, m1,m2,m3)
            print(f'wrap result {cp.linalg.norm(result[c])}')

        return result

    def _forward_flow(v):
        # Phi0[t] can be viewed as a grid, also a diffeomorphism.
        # Phi0[t] as grid : (Phi[t, 0, x, y, z], Phi[t, 1, x, y, z]),Phi[t, 2, x, y, z]).
        # Phi0[t] as diffeomorphism : (x, y, z) -> (Phi0[t, 0, x, y, z], Phi0[t, 1, x, y, z],Phi[t, 2, x, y, z])).
        Phi0 = cp.empty((T, 3, nx, ny, nz), dtype = cp.float64)

        # Phi0[0] = id, i.e.,
        # Phi0[0, 0, x, y , z] = x, Phi0[0, 1, x, y, z] = y, Phi[0, 1, x, y, z]=z)
        # Using numpy's index trick to generate identity mapping.
        Phi0[0] = cp.mgrid[:nx, :ny, :nz]
        # Calculate Phi0[t] from Phi0[t - 1].
        for t in range(1, T):
            # Calculate alpha by an iterative formula.
            alpha = cp.zeros((3, nx, ny,nz), dtype = cp.float64)
            for _ in range(5):
                alpha = _deform(v[t - 1], Phi0[0] - 0.5 * alpha)

            # Phi0[t] : x -> Phi0[t - 1](x - alpha).
            Phi0[t] = _deform(Phi0[t - 1], Phi0[0] - alpha)

        return Phi0

    def _backward_flow(v):
        Phi1 = cp.empty((T, 3, nx, ny,nz), dtype = cp.float64)
        Phi1[T - 1] = cp.mgrid[:nx, :ny,:nz]
        for t in range(T - 2, -1, -1):
            alpha = cp.zeros((3, nx, ny,nz), dtype = cp.float64)
            for _ in range(5):
                alpha = _deform(v[t], Phi1[T - 1] + 0.5 * alpha)
            Phi1[t] = _deform(Phi1[t + 1], Phi1[T - 1] + alpha)

        return Phi1

    def _forward_deform(I0, Phi0):
        J0 = cp.empty((T, 1, nx, ny,nz), dtype = cp.float64)
        for t in range(T):
            J0[t] = _deform(I0, Phi0[t])
        return J0

    def _backward_deform(I1, Phi1):
        J1 = cp.empty((T, 1, nx, ny,nz), dtype = cp.float64)
        for t in range(T - 1, -1, -1):
            J1[t] = _deform(I1, Phi1[t])
        return J1

    def _grad(data):
        grad = cp.empty((3, nx, ny,nz), dtype = cp.float64)
        grad[0] = gradientx(data)
        grad[1] = gradienty(data)
        grad[2] = gradientz(data)
        return grad

    def _image_grad(J0):
        dJ0 = cp.empty((T, 3, nx, ny, nz), dtype = cp.float64)
        for t in range(T):
            dJ0[t] = _grad(J0[t, 0])
        return dJ0

    def _jacobian_derterminant(Phi1):
        detPhi1 = cp.empty((T, 1, nx, ny,nz), dtype = cp.float64)
        for t in range(T):
            dx = _grad(Phi1[t, 0])
            dy = _grad(Phi1[t, 1])
            dz = _grad(Phi1[t, 2])
            #detPhi1[t] = abs(dx[0] * dy[1] * dz[2]+dx[1] * dy[2]*dz[0]+dx[2]*dy[0]*dz[1]-\
                         #dx[2] * dy[1] * dz[0]-dx[1] * dy[0]*dz[2]-dx[0]*dy[2]*dz[1])
            detPhi1[t] = abs(ker_determinant(dx[0],dx[1],dx[2],dy[0],dy[1],dy[2],dz[0],dz[1],dz[2]))
        return detPhi1

    ker_determinant = cp.ElementwiseKernel(
        'float64 x0, float64 x1, float64 x2, float64 y0, float64 y1, float64 y2, float64 z0, float64 z1, float64 z2',
        'float64 z',
        'z = x0 * y1 * z2 + x1 * y2 * z0 + x2 * y0 * z1 - x2 * y1 * z0 - x1 * y0 * z2 - x0 * y2 * z1',
        'ker_determinant'
    )
    # All data, including images, velocity fields, diffeomorphisms,
    # have shape (c, nx, ny,nz), where c is the number of channels.
    I0 = I0.reshape((1, nx, ny, nz))
    I1 = I1.reshape((1, nx, ny, nz))
    # Initialize.
    v = cp.zeros((T, 3, nx, ny, nz), dtype = cp.float64)
    dv = cp.zeros((T, 3, nx, ny, nz), dtype = cp.float64)
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
        #print(cp.linalg.norm(v))
        # Calculate forward and backward flows.
        Phi0 = _forward_flow(v)
        Phi1 = _backward_flow(v)
        #print(cp.linalg.norm(I0))
        #print(cp.linalg.norm(Phi0))
        # Calculate forward and backward deformations.
        J0 = _forward_deform(I0, Phi0)
        J1 = _backward_deform(I1, Phi1)
        #print(cp.linalg.norm(J0))
        # Calculate image gradient.
        dJ0 = _image_grad(J0)
        #print(cp.linalg.norm(dJ0))
        # Calculate Jacobian determinant of the transformation.
        detPhi1 = _jacobian_derterminant(Phi1)
        #print(cp.linalg.norm(detPhi1))
        # Calculate the gradient.
        for t in range(T):
            # Using numpy's broadcast, we can multiply a (1, nx, ny,nz)-array with a (2, nx, ny,nz)-array.
            dv[t] = 2 * v[t] - 2 / sigma ** 2 * _operator_K(detPhi1[t] * (J0[t] - J1[t]) * dJ0[t])
        #print(cp.linalg.norm(dv))
        if cp.linalg.norm(dv) < eps:
            print("Gradient norm below threshold, stopping.")
            break

        # Calculate new energy
        E1 = sum(cp.linalg.norm((_operator_L(v[t])))**2 for t in range(T))
        E2 = 1 / sigma ** 2 * cp.linalg.norm(I1[0] - J0[-1]) ** 2
        E1 = E1.item()
        E2 = E2.item()
        E = E1 + E2

        print(f'iteration {round:3d}, energy {E} = {E1} + {E2}.')

    return (v, Phi0, Phi1, J0, J1)

if __name__ == "__main__":
    # load greyscale images
    mrc = mrcfile.open('/home/fanxiao/data/emd_21546.map', permissive=True, mode='r')
    i0 = cp.asarray(mrc.data, dtype = cp.float64)
    #i0 = np.array(mrc.data, dtype = np.float64)
    i0[i0 < 0.05] = 0
    #i0 = downscale_local_mean(i0, (2, 2, 2))
    #i0 = downscale_local_mean(i0, (2, 2, 2))
    #i0 = cp.asarray(i0, dtype = cp.float64)
    i1 = cp.zeros_like(i0, dtype = cp.float64)
    i1[:, 20:, :] = i1[:, :-20, :]
    #mrc = mrcfile.open('/home/fanxiao/data/emd_21546_ffit_into_emd_21547.mrc', permissive=True, mode='r')
    #i1 = np.array(mrc.data)
    #i1=cp.asarray(mrc.data)
    #i1[i1 < 0.05] = 0
    #i1 = downscale_local_mean(i1, (2, 2, 2))
    #i1 = downscale_local_mean(i1, (2, 2, 2))
    #i1 = cp.asarray(i1)
    v, Phi0, Phi1, J0, J1=LDDMM(i0, i1, T = 12, maxiter = 10, sigma = 0.1, alpha = 1, gamma = 1, lr = 0.001)
    F = J0[:, 0, :, :, :]
    J = cp.asnumpy(F)
    np.save('/home/fanxiao/data/wyt', J)
    print(cp.linalg.norm(i0 - i1))
    print(cp.linalg.norm(i1 - J0[-1, 0, :, :]))



