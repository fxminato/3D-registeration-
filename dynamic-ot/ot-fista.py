import numpy as np
from math import pi
from numpy.fft import fft2, ifft2,fftn,ifftn
from scipy.ndimage import convolve
from skimage.transform import warp
import matplotlib.pyplot as plt
from scipy.fft import dctn,idctn
import imageio
from PIL import Image
P=64
Q=64
T=32
d=[P,Q,T]
def loadimg(path):
    """
    loads a greyscale image and converts it's datatype
    @param path:
    @return:
    """
    img = imageio.imread(path)
    img_grey = img[:, :, 0]
    return img_grey/ 255.
def save_animation(path, images):
    """
    creates an animation out of the images
    @param images:
    @return:
    """
    images = [Image.fromarray((images[t] * 255).astype('uint8')) for t in range(len(images))]
    imageio.mimsave(path, images)
class staggered():
    #define staggered grid,
    def __init__(self,dimvect):
        self.M1=np.zeros([P+1,Q,T])
        self.M2=np.zeros([P,Q+1,T])
        self.F=np.zeros([P,Q,T+1])
        self.F[:,:,0]=i0
        self.F[:,:,-1]=i1
def interp(U):
    #V is the center grid(3*P*Q*T),the 0th and 1th dimension represents momentum,the 2th dimension represents density
    V=np.zeros([3,P,Q,T])
    V[0,:,:,:]=(U.M1[:-1,:,:]+U.M1[1:,:,:])/2
    V[1,:,:,:]=(U.M2[:,:-1,:]+U.M2[:,1:,:])/2
    V[2,:,:,:]=(U.F[:,:,1:]+U.F[:,:,:-1])/2
    return V
def interp_ad(V):
    U=staggered(dimvect=d)
    U.M1[1:-1,:,:]=(V[0,1:,:,:]+V[0,:-1,:,:])/2
    U.M2[:,1:-1,:]=(V[1,:,1:,:]+V[1,:,:-1,:])/2
    U.F[:,:,1:-1]=(V[2,:,:,1:]+V[2,:,:,:-1])/2
    return U
def grad(U):
    V = np.zeros([3, P, Q, T])
    V[0, :, :, :] = (U.M1[1:, :, :]-U.M1[:-1, :, :])*P
    V[1, :, :, :] = (U.M2[:,1:, :]-U.M2[:, -1, :])*Q
    V[2, :, :, :] = (U.F[:, :, 1:]-U.F[:, :, :-1])*T
    return V
def div(u):
    v=np.zeros([P,Q,T])
    v=P*(u.M1[1:,:,:]-u.M1[:-1,:,:])+Q*(u.M2[:,1:,:]-u.M2[:,:-1,:])+T*(u.F[:,:,1:]-u.F[:,:,:-1])
    return v
def poisson_Neumann(f):
    denom2=np.zeros([P,Q,T])
    for i in range(P):
        for j in range(Q):
            for k in range(T):
                denom2[i,j,k]=(2*np.cos(np.pi*i/P)-2)*P**2+(2*np.cos(np.pi*j/Q)-2)*Q**2+(2*np.cos(np.pi*k/T)-2)*T**2
    denom2[denom2 == 0] = 1
    fhat=dctn(f,norm='ortho')
    uhat=-(fhat)/denom2
    res=idctn(uhat,norm='ortho')
    return res
def PG(x,niter):
    tao=1
    lr=0.00005
    epsilon = 1e-8
    for i in range(niter):
        xold=x
        y=interp(xold)
        y[2,y[2,:,:,:]<epsilon]=epsilon
        #M1=np.zeros([P,Q,T])
        #M2=np.zeros([P,Q,T])
        #F=np.zeros([P,Q,T])
        #H=np.zeros([P,Q,T])
        #M1[y[2,:,:,:]!=0]=y[0,y[2,:,:,:]!=0]/y[2,y[2,:,:,:]!=0]
        #M2[y[2,:,:,:]!=0]=y[1,y[2,:,:,:]!=0]/y[2,y[2,:,:,:]!=0]
        #F[y[2,:,:,:]!=0]=(y[0,y[2,:,:,:]!=0]**2+y[1,y[2,:,:,:]!=0]**2)/(2*(y[2,y[2,:,:,:]!=0]**2))
        #H = (y[0, y[2,:,:,:]!=0] ** 2 + y[1, y[2,:,:,:]!=0] ** 2) / abs(y[2, y[2,:,:,:]!=0])
        
        M1=y[0,:,:,:]/y[2,:,:,:]
        M2=y[1,:,:,:]/y[2,:,:,:]
        F=(y[0,:,:,:]**2+y[1,:,:,:]**2)/(2*(y[2,:,:,:]**2))
        H=(y[0,:,:,:]**2+y[1,:,:,:]**2)/abs(y[2,:,:,:])
        M1[y[2,:,:,:]==epsilon]=0
        M2[y[2,:,:,:]==epsilon]=0
        F[y[2,:,:,:]==epsilon]=0
        H[y[2,:,:,:]==epsilon]=0
        
        x.M1[1:-1,:,:]=xold.M1[1:-1,:,:]-lr*(M1[1:,:,:]+M1[:-1,:,:])/2
        x.M2[:,1:-1,:]=xold.M2[:,1:-1,:]-lr*(M2[:,1:,:]+M2[:,:-1,:])/2
        x.F[:,:,1:-1]=xold.F[:,:,1:-1]+lr*(F[:,:,1:]+F[:,:,:-1])/2
        phi=poisson_Neumann(div(x))
        x.M1[1:-1,:,:]=x.M1[1:-1,:,:]+phi[1:,:,:]-phi[:-1,:,:]
        x.M2[:, 1:-1, :]=x.M2[:, 1:-1, :]+phi[:,1:,:]-phi[:,:-1,:]
        x.F[:, :, 1:-1]=x.F[:, :, 1:-1]+phi[:,:,1:]-phi[:,:,:-1]
        tao0=tao
        w=(tao0-1)/tao
        E=np.linalg.norm(H)
        x.M1[1:-1, :, :] = (1+w)*x.M1[1:-1, :, :]-w*xold.M1[1:-1, :, :]
        x.M2[:, 1:-1, :] = (1+w)*x.M2[:, 1:-1, :]-w*xold.M2[:, 1:-1, :]
        x.F[:,:,1:-1]=(1+w)*x.F[:,:,1:-1]-w*xold.F[:,:,1:-1]
        print(f'iteration {i:3d}, energy {E:4.8f} ')
    return x
i0=np.array(loadimg('/Users/minato/workspace/Compressed_Sensing/pythonProject/pyLDDMM/example_images/t0.png'))
i1=np.array(loadimg('/Users/minato/workspace/Compressed_Sensing/pythonProject/pyLDDMM/example_images/t1.png'))
a=i0.sum()
i0=i0/a
i1=i1/a
U0=staggered(dimvect=d)
F_init=np.zeros([P,Q,T+1])
for t in range(T+1):
    F_init[:,:,t]=i1*t/T+i0*(T-t)/T
U0.F=F_init
U0.M1=np.zeros([P+1,Q,T])
U0.M2=np.zeros([P,Q+1,T])
U=PG(U0,32)
F=np.zeros([T+1,P,Q])
for t in range(T+1):
    F[t,:,:]=U.F[:,:,t]*a
save_animation('/Users/minato/desktop/out.gif',F)




