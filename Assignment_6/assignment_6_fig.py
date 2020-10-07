import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


#if using termux
import subprocess
import shlex
#end if

omat = np.array([[0,1],[-1,0]]) 

# ******************************************************************* #
#line_funcs
def dir_vec(A,B):
  return B-A

def norm_vec(A,B):
  return np.matmul(omat, dir_vec(A,B))

#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB


#Intersection of two lines
def line_intersect(n1,A1,n2,A2):
  N=np.vstack((n1,n2))
  p = np.zeros(2)
  p[0] = n1@A1
  p[1] = n2@A2
  #Intersection
  P=np.linalg.inv(N)@p
  return P

#Intersection of two lines
def perp_foot(n,cn,P):
  m = omat@n
  N=np.block([[n],[m]])
  p = np.zeros(2)
  p[0] = cn
  p[1] = m@P
  #Intersection
  x_0=np.linalg.inv(N)@p
  return x_0
#line_funcs
# ************************************************************* #    

# ************************************************************* #
#Triangle_funcs
#Triangle vertices
def tri_vert(a,b,c):
  p = (a**2 + c**2-b**2 )/(2*a)
  q = np.sqrt(c**2-p**2)
  A = np.array([p,q]) 
  B = np.array([0,0]) 
  C = np.array([a,0]) 
  return  A,B,C



#Foot of the Altitude
def alt_foot(A,B,C):
  m = B-C
  n = np.matmul(omat,m) 
  N=np.vstack((m,n))
  p = np.zeros(2)
  p[0] = m@A 
  p[1] = n@B
  #Intersection
  P=np.linalg.inv(N.T)@p
  return P


#Radius and centre of the circumcircle
#of triangle ABC
def ccircle(A,B,C):
  p = np.zeros(2)
  n1 = dir_vec(B,A)
  p[0] = 0.5*(np.linalg.norm(A)**2-np.linalg.norm(B)**2)
  n2 = dir_vec(C,B)
  p[1] = 0.5*(np.linalg.norm(B)**2-np.linalg.norm(C)**2)
  #Intersection
  N=np.vstack((n1,n2))
  O=np.linalg.inv(N)@p
  r = np.linalg.norm(A -O)
  return O,r

#Radius and centre of the incircle
#of triangle ABC
def icircle(A,B,C):
  k1 = 1
  k2 = 1
  p = np.zeros(2)
  t = norm_vec(B,C)
  n1 = t/np.linalg.norm(t)
  t = norm_vec(C,A)
  n2 = t/np.linalg.norm(t)
  t = norm_vec(A,B)
  n3 = t/np.linalg.norm(t)
  p[0] = n1@B- k1*n2@C
  p[1] = n2@C- k2*n3@A
  N=np.vstack((n1-k1*n2,n2-k2*n3))
  I=np.matmul(np.linalg.inv(N),p)
  r = n1@(I-B)
  #Intersection
  return I,r 
#Triangle_funcs
# ************************************************************* #

# ************************************************************* #
#Conic_funcs
#Generating points on a circle
def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

#Generating points on an ellipse
def ellipse_gen(a,b):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_ellipse = np.zeros((2,len))
	x_ellipse[0,:] = a*np.cos(theta)
	x_ellipse[1,:] = b*np.sin(theta)
	return x_ellipse

#Generating points on a parabola
def parab_gen(y,a):
	x = y**2/a
	return x

#Generating points on a standard hyperbola 
def hyper_gen(y):
	x = np.sqrt(1+y**2)
	return x
#Conic_funcs
# *************************************************************** #

#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100
y = np.linspace(-5,5,len)

#Ellipse parameters
V = np.array(([14,-2],[-2,1]))
u = np.array(([2,3]))
f = -100
c = -LA.inv(V)@u
#Eigenvalues and eigenvectors
D_vec,P = LA.eig(V)
D = np.diag(D_vec)
pcos = np.cos(np.pi/4)
psin = np.sin(np.pi/4)
P = np.array(([pcos,-psin],[psin,pcos]))
a = np.sqrt(-f/D_vec[0])
b = np.sqrt(-f/D_vec[1])
xStandardEllipse = ellipse_gen(a,b)

#Major and Minor Axes
MajorStandard = np.array(([a,0]))
MinorStandard = np.array(([0,b]))

#Affine transform 
xActualEllipse = P@xStandardEllipse
MajorActual = P@MajorStandard
MinorActual = P@MinorStandard


#Plotting the actual ellipse
plt.plot(xActualEllipse[0,:],xActualEllipse[1,:],label='Actual ellipse')

#Labeling the coordinates
tri_coords = np.vstack((MajorActual,MinorActual)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['$a^{\prime}$','$b^{\prime}$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()
