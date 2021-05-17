import numpy as np
from math import pi
from numpy.fft import fft2, ifft2,fftn,ifftn
from scipy.ndimage import convolve
from skimage.transform import warp
import matplotlib.pyplot as plt
from scipy.fft import dctn,idctn
import imageio
from PIL import Image
P=32
Q=32
T=32
d=[P,Q,T]
epsilon=1e-8
i0=np.zeros([32,32])
i1=np.zeros([32,32])
i0[8:24,8:24]=0.5
i1[8:24,8:24]=0.5
i0[4:7,8:11]=0.5
i1[4:7,13:16]=0.5
i0=i0/i0.sum()
i1=i1/i1.sum()
def save_animation(path, images):
    """
    creates an animation out of the images
    @param images:
    @return:
    """
    images = [Image.fromarray((images[t] * 255).astype('uint8')) for t in range(len(images))]
    imageio.mimsave(path, images)
class staggered():
    def __init__(self,dimvect):
        self.M1=np.zeros([P+1,Q,T])
        self.M2=np.zeros([P,Q+1,T])
        self.F=np.zeros([P,Q,T+1])

def interp(U):
    V=np.zeros([3,P,Q,T])
    U.M1[0, :, :] = 0
    U.M1[-1, :, :] = 0
    U.M2[:, 0, :] = 0
    U.M2[:, -1, :] = 0
    U.F[:, :, 0] = 0
    U.F[:, :, -1] = 0
    V[0,:,:,:]=(U.M1[:-1,:,:]+U.M1[1:,:,:])/2
    V[1,:,:,:]=(U.M2[:,:-1,:]+U.M2[:,1:,:])/2
    V[2,:,:,:]=(U.F[:,:,1:]+U.F[:,:,:-1])/2
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
    #U.M1[0, :, :] = V[0, 0, :, :] / 2
    #U.M1[-1, :, :] = V[0, -1, :, :] / 2
    #U.M2[:, 0, :] = V[1, :, 0, :] / 2
    #U.M2[:, -1, :] = V[1, :, -1, :] / 2
    #U.F[:, :, 0] = V[2, :, :, 0] / 2
    #U.F[:, :, -1] = V[2, :, :, -1] / 2
    return U
def ProxJ(V0,gamma,epsilon):
    a=np.zeros([3,P,Q,T])
    f0=V0[2,:,:,:]
    m1=V0[0,:,:,:]
    m2=V0[1,:,:,:]
    for t in range(T):
            for i in range(P):
                for j in range(Q):
                    root = np.roots([1,4*gamma-f0[i,j,t],4*gamma*gamma-4*gamma*f0[i,j,t],-4*gamma*gamma*f0[i,j,t]-gamma*(m1[i,j,t]**2+m2[i,j,t]**2)])
                    a[2,i,j,t]=np.real(np.max(root[np.isreal(root)]))
                    if(a[2,i,j,t]<epsilon):
                        a[2,i,j,t]=epsilon
                    a[0,i,j,t]=a[2,i,j,t]*m1[i,j,t]/(a[2,i,j,t]+2*gamma)
                    a[1,i,j,t]=a[2,i,j,t]*m2[i,j,t]/(a[2,i,j,t]+2*gamma)
    return a
def ProxF(V0,gamma):
    b=np.zeros([3,P,Q,T])
    b[2,:,:,0]=i0/2
    b[2,:,:,-1]=i1/2
    return ProxJ(V0+b,gamma,epsilon)-b
def ProxFS(y,sigma):
    return y-sigma*ProxF(y/sigma,1/sigma)
def div(u):
    v=np.zeros([P,Q,T])
    v=P*(u.M1[1:,:,:]-u.M1[:-1,:,:])+Q*(u.M2[:,1:,:]-u.M2[:,:-1,:])+T*(u.F[:,:,1:]-u.F[:,:,:-1])
    return v
def poisson_Neumann(f):
    dp=np.arange(P)
    dq=np.arange(Q)
    dt=np.arange(T)
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

def ProxG(u):
    p=poisson_Neumann(-div(u))
    v=u
    v.M1[1:-1,:,:]=v.M1[1:-1,:,:]-(p[1,:,:]-p[-1,:,:])*P
    v.M2[:,1:-1,:]=v.M2[:,1:-1,:]-(p[:,1:,:]-p[:,:-1,:])*Q
    v.F[:,:,1:-1]=v.F[:,:,1:-1]-(p[:,:,1:]-p[:,:,:-1])*T
    return v
def perform_primal_dual(x,niter,theta=1):
    sigma=85
    tau=0.99/sigma
    x1=x
    y=interp(x)

    for i in range(niter):
        xold=x
        y=ProxFS(y+sigma*interp(x1),sigma)
        print(np.max(y))
        z=interp_ad(y)
        z.M1=x.M1-tau*z.M1
        z.M2=x.M2-tau*z.M2
        z.F=x.F-tau*z.F
        x=ProxG(z)
        print(np.max(x.M1))
        x1.M1=x.M1+theta*(x.M1-xold.M1)
        x1.M2=x.M2+theta*(x.M2-xold.M2)
        x1.F=x.F+theta*(x.F-xold.F)
        #x1=x+theta*(x-xold)

    return (x,y)
U0=staggered(dimvect=d)
F_init=np.zeros([P,Q,T+1])
for t in range(T+1):
    F_init[:,:,t]=i1*t/T+i0*(T-t)/T

U0.F=F_init
U,V=perform_primal_dual(U0,10)
F=np.zeros([T+1,P,Q])
for t in range(T+1):
    F[t,:,:]=U.F[:,:,t]
save_animation('/Users/minato/desktop/out.gif',F)




