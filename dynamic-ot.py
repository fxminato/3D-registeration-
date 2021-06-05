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
    def __init__(self,dimvect):
        self.M1=np.zeros([P+1,Q,T])
        self.M2=np.zeros([P,Q+1,T])
        self.F=np.zeros([P,Q,T+1])
def interp(U):
    V=np.zeros([3,P,Q,T])
    M1=U.M1.copy()
    M2=U.M2.copy()
    F=U.F.copy()
    M1[0, :, :] = 0
    M1[-1, :, :] = 0
    M2[:, 0, :]= 0
    M2[:, -1, :] = 0
    F[:, :, 0] = 0
    F[:, :, -1] = 0
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
    root1,root2,root3=poly_root(4*gamma-f0,4*gamma*gamma-4*gamma*f0,-4*gamma*gamma*f0-gamma*(m1**2+m2**2))
    root1[np.isreal(root1)==0]=np.real(np.float('-inf'))
    root2[np.isreal(root2)==0] = np.real(np.float('-inf'))
    root3[np.isreal(root3) == 0] = np.real(np.float('-inf'))
    root1=root1.astype('float64')
    root2=root2.astype('float64')
    root3=root3.astype('float64')
    a[2,:,:,:]=np.maximum(root1,root2)
    a[2,:,:,:]=np.maximum(a[2,:,:,:],root3)
    a[2,a[2,:,:,:]<epsilon]=epsilon
    a[0,:,:,:]=a[2,:,:,:]*m1/(a[2,:,:,:]+2*gamma)
    a[1,:,:,:]=a[2,:,:,:]*m2/(a[2,:,:,:]+2*gamma)
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
    v=staggered(dimvect=d)
    v.M1=u.M1.copy()
    v.M2=u.M2.copy()
    v.F=u.F.copy()
    v.M1[1:-1,:,:]=v.M1[1:-1,:,:]-(p[1:,:,:]-p[:-1,:,:])*P
    v.M2[:,1:-1,:]=v.M2[:,1:-1,:]-(p[:,1:,:]-p[:,:-1,:])*Q
    v.F[:,:,1:-1]=v.F[:,:,1:-1]-(p[:,:,1:]-p[:,:,:-1])*T
    return v
def perform_primal_dual(x,niter,theta=1):
    sigma=85
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
        y=ProxFS(y+sigma*y0,sigma)
        z=interp_ad(y)
        z.M1=x.M1-tau*z.M1
        z.M2=x.M2-tau*z.M2
        z.F=x.F-tau*z.F
        x=ProxG(z)
        x1.M1=x.M1+theta*(x.M1-xold.M1)
        x1.M2=x.M2+theta*(x.M2-xold.M2)
        x1.F=x.F+theta*(x.F-xold.F)
        E=np.sum((y0[1,:,:,:]**2+y0[0,:,:,:]**2)/np.maximum(y0[2,:,:,:]**2,1e-8)/2)
        print(f'iteration {i:3d}, energy {E:4.8f} ')
    return (x,y)

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
    x1=m+n-b/3
    x2=w*m+w*w*n-b/3
    x3=w*w*m+w*n-b/3
    x1[abs(np.imag(x1))<1e-8]=np.real(x1[abs(np.imag(x1))<1e-8])
    x2[abs(np.imag(x2))<1e-8]=np.real(x2[abs(np.imag(x2))<1e-8])
    x3[abs(np.imag(x3))<1e-8]=np.real(x3[abs(np.imag(x3))<1e-8])
    return(x1,x2,x3)

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
U,V=perform_primal_dual(U0,200)
F=np.zeros([T+1,P,Q])
for t in range(T+1):
    F[t,:,:]=U.F[:,:,t]
    max=np.max(F[t,:,:])
    min=np.min(F[t,:,:])
    F[t,:,:]=min+(F[t,:,:]-min)/(max-min)

np.save('/Users/minato/desktop/aaa', F)
save_animation('/Users/minato/desktop/out.gif',F)





