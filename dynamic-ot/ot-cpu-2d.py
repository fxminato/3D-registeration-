import numpy as np
from math import pi
from numpy.fft import fft2, ifft2,fftn,ifftn
from scipy.ndimage import convolve
from skimage.transform import warp
import matplotlib.pyplot as plt
from scipy.fft import dctn,idctn
import imageio
from PIL import Image
from skimage.io import imread, imsave
P=64
Q=64
T=10
d=[P,Q,T]
epsilon=1e-8
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
    """
    Vector field on staggered grid
    """
    def __init__(self,dimvect):
        self.M1 = np.zeros([P+1, Q, T])
        self.M2 = np.zeros([P, Q+1, T])
        self.F  = np.zeros([P, Q, T+1])
        self.F[:, :, 0] = i0
        self.F[:,:,-1]=i1
def interp(U):
    V=np.zeros([3,P,Q,T])
    M1=U.M1.copy()
    M2=U.M2.copy()
    F=U.F.copy()
    V[0,:,:,:]=(M1[:-1,:,:]+M1[1:,:,:])/2
    V[1,:,:,:]=(M2[:,:-1,:]+M2[:,1:,:])/2
    V[2,:,:,:]=(F[:,:,1:]+F[:,:,:-1])/2
    return V
def interp_ad(V):
    U=staggered(dimvect=d)
    U.M1[1:-1,:,:]=(V[0,1:,:,:]+V[0,:-1,:,:])/2
    U.M2[:,1:-1,:]=(V[1,:,1:,:]+V[1,:,:-1,:])/2
    U.F[:,:,1:-1]=(V[2,:,:,1:]+V[2,:,:,:-1])/2
    U.M1[0, :, :] = 0
    U.M1[-1, :, :] = 0
    U.M2[:, 0, :] = 0
    U.M2[:, -1, :] = 0
    U.F[:, :, 0] = 0
    U.F[:, :, -1] = 0
    return U
def ProxJ(V0, gamma, epsilon):
    a = np.zeros([3, P, Q, T])
    # print(np.linalg.norm(V0))
    f0 = V0[2, :, :, :]
    m1 = V0[0, :, :, :]
    m2 = V0[1, :, :, :]
    gamma=gamma/2
    a[2,:,:,:]=poly_root(4*gamma-f0,4*gamma*gamma-4*gamma*f0,-4*gamma*gamma*f0-gamma*(m1**2+m2**2))
    a[0,:,:,:]=a[2,:,:,:]*m1/(a[2,:,:,:]+2*gamma)
    a[1,:,:,:]=a[2,:,:,:]*m2/(a[2,:,:,:]+2*gamma)
    return a
def ProxFS(y,sigma):
    return y-sigma*ProxJ(y/sigma,1/sigma,epsilon)
def div(u):
    v=np.zeros([P,Q,T])
    v=P*(u.M1[1:,:,:]-u.M1[:-1,:,:])+Q*(u.M2[:,1:,:]-u.M2[:,:-1,:])+T*(u.F[:,:,1:]-u.F[:,:,:-1])
    return v
def poisson_Neumann(f):
    x =(2 * np.cos(np.pi * np.arange(0, P) / P) - 2) * P ** 2
    y =(2 * np.cos(np.pi * np.arange(0, Q) / Q) - 2) * Q ** 2
    z =(2 * np.cos(np.pi * np.arange(0, T) / T) - 2) * T ** 2
    denom2 = x.reshape(P, 1, 1) + y.reshape(1, Q, 1) + z.reshape(1, 1, T)
    denom2[denom2 == 0] = 1
    fhat=dctn(f,norm='ortho')
    uhat=-(fhat)/denom2
    res=idctn(uhat,norm='ortho')
    return res
def ProxG(u):
    p=poisson_Neumann(-div(u))
    v=staggered(dimvect=d)
    v.M1=u.M1.copy()
    v.M2=u.M2.copy()
    v.F=u.F.copy()
    v.M1[1:-1,:,:]=v.M1[1:-1,:,:]-(p[1:,:,:]-p[:-1,:,:])*P
    v.M2[:,1:-1,:]=v.M2[:,1:-1,:]-(p[:,1:,:]-p[:,:-1,:])*Q
    v.F[:,:,1:-1]=v.F[:,:,1:-1]-(p[:,:,1:]-p[:,:,:-1])*T
    return v
def perform_primal_dual(x,niter,theta=1):
    sigma=100
    tau=0.99/sigma
    x1=staggered(dimvect=d)
    x1.M1=x.M1.copy()
    x1.M2=x.M2.copy()
    x1.F=x.F.copy()
    y=interp(x)
    for i in range(niter):
        xold=staggered(dimvect=d)
        xold.M1=x.M1.copy()
        xold.M2=x.M2.copy()
        xold.F=x.F.copy()
        y0=interp(x1)
        print(np.linalg.norm(y0))
        y=ProxFS(y+sigma*y0,sigma)
        print(np.linalg.norm(y))
        z=interp_ad(y)
        z.M1=x.M1-tau*z.M1
        z.M2=x.M2-tau*z.M2
        z.F=x.F-tau*z.F
        x=ProxG(z)
        x1.M1=x.M1+theta*(x.M1-xold.M1)
        x1.M2=x.M2+theta*(x.M2-xold.M2)
        x1.F=x.F+theta*(x.F-xold.F)
        aa=interp(x)
        E=np.sum((aa[1,:,:,:]**2+aa[0,:,:,:]**2)/np.maximum(aa[2,:,:,:]**2,1e-8))
        print(f'iteration {i:3d}, energy {E:4.8f} ')
    return (x,y)
np.lin
def poly_root(b,c,d):
    u=(9*b*c-27*d-2*b**3)/54
    u=u.astype('complex')
    delta=3*(4*c**3-b**2*c**2-18*b*c*d+27*d**2+4*b**3*d)
    v=np.zeros_like(delta,dtype='complex')
    v[delta>0]=delta[delta>0]**0.5/18
    v[delta<0]=1j*((-delta[delta<0])**0.5)/18
    m=np.zeros_like(u,dtype='complex')
    xx=abs(u+v)-abs(u-v)
    m[xx>=0]=(u+v)[xx>=0]**(1/3)
    m[xx<0]=(u-v)[xx<0]**(1/3)
    n=np.zeros_like(u,dtype='complex')
    n[m!=0]=(b[m!=0]**2-3*c[m!=0])/(9*m[m!=0])
    w=-0.5+0.5j*np.sqrt(3)
    root=np.zeros([3,P,Q,T],dtype='complex')
    root[0]=m+n-b/3
    root[1]=w*m+w*w*n-b/3
    root[2]=w*w*m+w*n-b/3
    realMask = np.abs(root.imag) < 1e-8
    root[~realMask]=0
    return np.amax(root.real, axis = 0)
i0=imread('/Users/minato/Desktop/data/ot/t1-0.png',as_gray=True)
i1=imread('/Users/minato/Desktop/data/ot/t1-1.png',as_gray=True)
'''
i0=np.zeros([256,256])
i1=np.zeros([256,256])
for i in range(64):
    for j in range(64):
        for x in range(4):
            for y in range(4):
                i0[4*i+x,4*j+y]=si0[i,j]
                i1[4*i+x,4*j+y]=si1[i,j]
'''
a=i0.sum()
b=i1.sum()
i0=i0/a
i1=i1/b
U0=staggered(dimvect=d)
F_init=np.zeros([P,Q,T+1])
for t in range(T+1):
    F_init[:,:,t]=i1*t/T+i0*(T-t)/T
U0.F=F_init
U,V=perform_primal_dual(U0,500)
F=np.zeros([T+1,P,Q])
for t in range(T+1):
    F[t, :, :] = U.F[:, :, t]
    max = np.max(F[t, :, :])
    min = np.min(F[t, :, :])
    F[t, :, :] = min + (F[t, :, :] - min) / (max - min)
'''
P=64
Q=64
F=np.zeros([T+1,P,Q])
V1=np.zeros([P,Q,T])
V2=np.zeros([P,Q,T])
for t in range(T+1):
    for i in range(P):
        for j in range(Q):
            s=0
            for x in range(4):
                for y in range(4):
                    s=s+U.F[4*i+x,4*j+y,t]
            F[t,i,j]=s/16
    max=np.max(F[t,:,:])
    min=np.min(F[t,:,:])
    F[t,:,:]=min+(F[t,:,:]-min)/(max-min)

P=256
Q=256
V=interp(U)
V11=V[0,:,:,:]/V[2,:,:,:]
V22=V[1,:,:,:]/V[2,:,:,:]
for t in range(T):
    for i in range(64):
        for j in range(64):
            V1[i,j,t]=(V11[2*i,2*j,t]+V11[2*i+1,2*j+1,t]+V11[2*i+1,2*j,t]+V11[2*i,2*j+1,t])/4
            V2[i,j,t]=(V22[2*i,2*j,t]+V22[2*i+1,2*j+1,t]+V22[2*i+1,2*j,t]+V22[2*i,2*j+1,t])/4
np.save('/Users/minato/desktop/F', F)
np.save('/Users/minato/desktop/V1',V1)
np.save('/Users/minato/desktop/V2',V2)
'''
save_animation('/Users/minato/desktop/outtest.gif',F)





