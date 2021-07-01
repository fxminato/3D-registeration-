import numpy as np
from math import pi
from numpy.fft import fft2, ifft2
from scipy.ndimage import convolve
from skimage.transform import warp
import matplotlib.pyplot as plt
import imageio
from PIL import Image
def save_animation(path, images):
    """
    creates an animation out of the images
    @param images:
    @return:
    """
    images = [Image.fromarray((images[t] * 255).astype('uint8')) for t in range(len(images))]
    imageio.mimsave(path, images)
def loadimg(path):
    """
    loads a greyscale image and converts it's datatype
    @param path:
    @return:
    """
    img = imageio.imread(path)
    img_grey = img[:, :, 0]
    return img_grey/ 255.
def LDDMM(I0,I00,I01,I11,I1, T = 32, maxiter = 200, lr = 5e-3, sigma = 1.0, alpha = 1.0, gamma = 1.0, eps = 1e-3):
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
    (nx, ny) = I0.shape
    T01=int(T/2)
    T00=int(T/4)
    T11=int(3*T/4)
    def _operator_L(vt):
        r'''
        The Cauchy-Navier operator.
        '''
        kernel = np.array([[0.,  1., 0.],
                           [1., -4., 1.],
                           [0.,  1., 0.]], dtype = np.float64)

        laplacian = np.empty_like(vt, dtype = np.float64)
        laplacian[0] = convolve(vt[0], kernel, mode = 'wrap')
        laplacian[1] = convolve(vt[1], kernel, mode = 'wrap')

        return -alpha * laplacian + gamma * vt

    def _operator_K(g):
        r'''
        K = (L^T L)^{-1} = L^{-2}.
        '''
        a = 1 - np.cos(np.linspace(0, 2 * pi, nx, endpoint = False))
        #a=a/((nx)**2)
        b = 1 - np.cos(np.linspace(0, 2 * pi, ny, endpoint = False))
        #b=b/((ny)**2)
        # A[i, j] = alpha * (a[i] + b[j]) + gamma.
        # Using numpy's broadcast to do this.
        A = 2*alpha * (a.reshape((nx, 1)) + b.reshape((1, ny))) + gamma
        G = fft2(g)
        F = G / (A ** 2)
        f = ifft2(F)
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
        # Phi0[t] as grid : (Phi[t, 0, x, y], Phi[t, 1, x, y]).
        # Phi0[t] as diffeomorphism : (x, y) -> (Phi0[t, 0, x, y], Phi0[t, 1, x, y]).
        Phi0 = np.empty((T, 2, nx, ny), dtype = np.float64)

        # Phi0[0] = id, i.e.,
        # Phi0[0, 0, x, y] = x, Phi0[0, 1, x, y] = y.
        # Using numpy's index trick to generate identity mapping.
        Phi0[0] = np.mgrid[:nx, :ny]

        # Calculate Phi0[t] from Phi0[t - 1].
        for t in range(1, T):
            # Calculate alpha by an iterative formula.
            alpha = np.zeros((2, nx, ny), dtype = np.float64)
            for _ in range(5):
                alpha = _deform(v[t - 1], Phi0[0] - 0.5 * alpha)

            # Phi0[t] : x -> Phi0[t - 1](x - alpha).
            Phi0[t] = _deform(Phi0[t - 1], Phi0[0] - alpha)

        return Phi0

    def _backward_flow(v):
        Phi1 = np.empty((T, 2, nx, ny), dtype = np.float64)
        Phi1[T - 1] = np.mgrid[:nx, :ny]
        for t in range(T - 2, -1, -1):
            alpha = np.zeros((2, nx, ny), dtype = np.float64)
            for _ in range(5):
                alpha = _deform(v[t], Phi1[T - 1] + 0.5 * alpha)
            Phi1[t] = _deform(Phi1[t + 1], Phi1[T - 1] + alpha)

        return Phi1

    def _backward_flow01(v):
        Phi1 = np.empty((T, 2, nx, ny), dtype = np.float64)
        Phi1[T01-1]=np.mgrid[:nx,:ny]
        for t in range(T01 - 2, -1, -1):
            alpha = np.zeros((2, nx, ny), dtype = np.float64)
            for _ in range(5):
                alpha = _deform(v[t], Phi1[T01 - 1] + 0.5 * alpha)
            Phi1[t] = _deform(Phi1[t + 1], Phi1[T01 - 1] + alpha)

        return Phi1

    def _backward_flow00(v):
        Phi1 = np.empty((T, 2, nx, ny), dtype = np.float64)
        Phi1[T00-1]=np.mgrid[:nx,:ny]
        for t in range(T00 - 2, -1, -1):
            alpha = np.zeros((2, nx, ny), dtype = np.float64)
            for _ in range(5):
                alpha = _deform(v[t], Phi1[T00 - 1] + 0.5 * alpha)
            Phi1[t] = _deform(Phi1[t + 1], Phi1[T00 - 1] + alpha)

        return Phi1

    def _backward_flow11(v):
        Phi1 = np.empty((T, 2, nx, ny), dtype = np.float64)
        Phi1[T11-1]=np.mgrid[:nx,:ny]
        for t in range(T11 - 2, -1, -1):
            alpha = np.zeros((2, nx, ny), dtype = np.float64)
            for _ in range(5):
                alpha = _deform(v[t], Phi1[T11 - 1] + 0.5 * alpha)
            Phi1[t] = _deform(Phi1[t + 1], Phi1[T11 - 1] + alpha)

        return Phi1

    def _forward_deform(I0, Phi0):
        J0 = np.empty((T, 1, nx, ny), dtype = np.float64)
        for t in range(T):
            J0[t] = _deform(I0, Phi0[t])
        return J0

    def _backward_deform(I1, Phi1):
        J1 = np.empty((T, 1, nx, ny), dtype = np.float64)
        for t in range(T - 1, -1, -1):
            J1[t] = _deform(I1, Phi1[t])
        return J1

    def _backward_deform01(I1, Phi1):
        J1 = np.empty((T, 1, nx, ny), dtype = np.float64)
        for t in range(T01 - 1, -1, -1):
            J1[t] = _deform(I1, Phi1[t])
        return J1

    def _backward_deform00(I1, Phi1):
        J1 = np.empty((T, 1, nx, ny), dtype = np.float64)
        for t in range(T00 - 1, -1, -1):
            J1[t] = _deform(I1, Phi1[t])
        return J1

    def _backward_deform11(I1, Phi1):
        J1 = np.empty((T, 1, nx, ny), dtype = np.float64)
        for t in range(T11 - 1, -1, -1):
            J1[t] = _deform(I1, Phi1[t])
        return J1

    def _grad(data):
        wtx = np.array([[1/2, 0, -1/2]], dtype = np.float64)
        wty = np.array([[1/2], [0], [-1/2]], dtype = np.float64)
        grad = np.empty((2, nx, ny), dtype = np.float64)
        grad[1] = convolve(data, wtx, mode = 'wrap')
        grad[0] = convolve(data, wty, mode = 'wrap')
        return grad

    def _image_grad(J0):
        dJ0 = np.empty((T, 2, nx, ny), dtype = np.float64)
        for t in range(T):
            dJ0[t] = _grad(J0[t, 0])
        return dJ0

    def _jacobian_derterminant(Phi1):
        detPhi1 = np.empty((T, 1, nx, ny), dtype = np.float64)
        for t in range(T):
            dx = _grad(Phi1[t, 0])
            dy = _grad(Phi1[t, 1])
            detPhi1[t] = abs(dx[0] * dy[1] - dx[1] * dy[0])
        return detPhi1

    # All data, including images, velocity fields, diffeomorphisms,
    # have shape (c, nx, ny), where c is the number of channels.
    I0 = I0.reshape((1, nx, ny))
    I1 = I1.reshape((1, nx, ny))
    I01=I01.reshape((1,nx,ny))
    I00=I00.reshape((1,nx,ny))
    I11=I11.reshape((1,nx,ny))
    # Initialize.
    v = np.zeros((T, 2, nx, ny), dtype = np.float64)
    dv = np.zeros((T, 2, nx, ny), dtype = np.float64)
    Phi0 = None
    Phi1 = None
    Phi01=None

    J0 = None
    J1 = None
    J01=None
    E0=1e9
    for round in range(1, maxiter + 1):
        # Gradient descent.
        v -= lr * dv
        if round % 10 == 0:
           _reparametrize(v)
        #print(v[0])

        # Calculate forward and backward flows.
        Phi0 = _forward_flow(v)
        Phi1 = _backward_flow(v)

        Phi01= _backward_flow01(v)
        Phi00= _backward_flow00(v)
        Phi11= _backward_flow11(v)
        # Calculate forward and backward deformations.
        J0 = _forward_deform(I0, Phi0)
        J1 = _backward_deform(I1, Phi1)
        J01= _backward_deform01(I01,Phi01)
        J00= _backward_deform00(I00,Phi00)
        J11= _backward_deform11(I11,Phi11)
        # Calculate image gradient.
        dJ0 = _image_grad(J0)

        # Calculate Jacobian determinant of the transformation.
        detPhi1 = _jacobian_derterminant(Phi1)
        detPhi01 = _jacobian_derterminant(Phi01)
        detPhi11 = _jacobian_derterminant(Phi11)
        detPhi00 = _jacobian_derterminant(Phi00)
        #if detPhi1.min() < 0:
        #    print('Injectivity violated. Stopping. Try lowering the learning rate.')
        #    break

        # Calculate the gradient.
        for t in range(T):
            # Using numpy's broadcast, we can multiply a (1, nx, ny)-array with a (2, nx, ny)-array.
            dv[t] = 2 * v[t] - 2 / sigma ** 2 * _operator_K(detPhi1[t] * (J0[t] - J1[t]) * dJ0[t])
        for t in range(T00):
            dv[t]-=2/sigma**2*_operator_K(detPhi00[t]*(J0[t]-J00[t])*dJ0[t])
        for t in range(T01):
            dv[t]-=2/sigma**2*_operator_K(detPhi01[t]*(J0[t]-J01[t])*dJ0[t])
        #if np.linalg.norm(dv) < eps:
           # print("Gradient norm below threshold, stopping.")
            #break

        # Calculate new energy
        E1 = sum(np.linalg.norm(_operator_L(v[t]))**2 for t in range(T))
        E2 = 1 / sigma ** 2 * np.linalg.norm(I1[0] - J0[-1]) ** 2
        E3 = 1 / sigma ** 2 * np.linalg.norm(I00[0] - J0[T00-1]) ** 2
        E4=1 / sigma ** 2 * np.linalg.norm(I01[0] - J0[T01-1]) ** 2

        E = E1 + E2+E3+E4
        #if(E>E0):
            #break
        #E0=E
        print(f'iteration {round:3d}, energy {E:4.2f} = {E1:4.2f} + {E2:4.2f}+{E3:4.2f}+{E4:4.2f}.')
        '''if E<E0:
            v_best=v
            Phi0_best=Phi0
            Phi1_best=Phi1
            J0_best=J0
            J1_best=J1
            E0=E'''
    return (v, Phi0, Phi1, J0, J1)
    #return (v_best, Phi0_best, Phi1_best, J0_best, J1_best)

def plot_warpgrid(warp, interval=2, show_axis=False):
    """
    plots the given warpgrid
    @param warp: array, H x W x 2, the transformation
    @param interval: int, The interval between grid-lines
    @param show_axis: Bool, should axes be included?
    @return: matplotlib plot. Show with plt.show()
    """
    if show_axis is False:
        plt.axis('off')
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal')

    for row in range(0, warp.shape[1], interval):
        plt.plot(warp[0, row, :], warp[1,row, :,], 'k')
    for col in range(0, warp.shape[2], interval):
        plt.plot(warp[0, :,col], warp[1, :,col], 'k')
    return plt

if __name__ == '__main__':
    #a =loadimg('/Users/minato/workspace/Compressed_Sensing/pythonProject/pyLDDMM/example_images/circle.png')
    #b =loadimg('/Users/minato/workspace/Compressed_Sensing/pythonProject/pyLDDMM/example_images/square.png')
    #b=np.zeros_like(a)
    #b[24:54,24:54]=0.5
    a = np.zeros([60, 60])
    b = np.zeros([60, 60])
    c = np.zeros([60, 60])
    d = np.zeros([60, 60])
    a[20:40, 20:40] = 0.5
    b[20:40, 20:40] = 0.5
    c[20:40, 20:40] = 0.5
    d[20:40, 20:40] = 0.5
    a[15:20, 21:27] = 0.5
    b[15:20, 25:31] = 0.5
    c[15:20, 29:35] = 0.5
    d[15:20, 33:39] = 0.5

    aa = np.zeros([60, 60])
    bb =np.zeros([60,60])
    aa[20:40,20:40]=0.5
    bb[25:45,20:40]=0.5
    v, Phi0, Phi1, J0, J1=LDDMM(a,b,c,d,e , T = 32, maxiter = 100, sigma =0.1, alpha = 1,lr=0.005)
    J=J0[:,0,:,:]
    plt = plot_warpgrid(Phi1[0], interval=2)
    plt.savefig('/Users/minato/desktop/wyt.png')
    save_animation('/Users/minato/desktop/out.gif', J)
    print(np.linalg.norm(a-b))
    print(np.linalg.norm(b-J0[-1,0,:,:]))