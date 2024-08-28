#!/usr/bin/env python
# coding: utf-8

# # An Introduction to Python for Mathematics
# 
# ## a crash course
# 
# *Simon Shaw*
# 
# - <https://www.brunel.ac.uk/people/simon-shaw>
# - <https://github.com/variationalform>
# 
# Version 1

# <table>
# <tr>
# <td>
# <img src="https://www.gnu.org/graphics/heckert_gnu.transp.small.png" style="height:18px"/>
# <img src="https://www.gnu.org/graphics/heckert_gnu.transp.small.png" style="height:18px"/>
# <img src="https://www.gnu.org/graphics/heckert_gnu.transp.small.png" style="height:18px"/>
# </td>
# <td>
# 
# <p>
# This work is available under GPL 3
# 
# <p>
# Visit <a href="https://www.gnu.org/licenses/gpl-3.0.en.html">https://www.gnu.org/licenses/gpl-3.0.en.html</a> to see the terms.
# </td>
# </tr>
# </table>

# <table>
# <tr>
# <td>This document uses python</td>
# <td>
# <img src="https://www.python.org/static/community_logos/python-logo-master-v3-TM.png" style="height:30px"/>
# </td>
# <td>and also makes use of LaTeX </td>
# <td>
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/LaTeX_logo.svg/320px-LaTeX_logo.svg.png" style="height:30px"/>
# </td>
# <td>in Markdown</td> 
# <td>
# <img src="https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png" style="height:30px"/>
# </td>
# </tr>
# </table>

# ## Contents
# 
# This is a very quick run through of how to use some basic features of python in a Jupyter
# notebook. We'll cover:
# 
# 1. markdown
# 1. linear algebra
# 1. plotting
# 1. prob/stat simulations
# 1. some other stuff... if time...
# 
# There is a lot we don't touch...
# 
# There is a beamer/PDF slide show with some background to the code below.

# This material is on git at: <https://github.com/variationalform/PythonMathsPrimer>
# 
# you can run the notebook on binder using this:
# 
# <https://mybinder.org/v2/gh/variationalform/PythonMathsPrimer/HEAD>
# 
# This code creates a button - but it breaks LaTeX output
# 
# ```
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/variationalform/PythonMathsPrimer/HEAD)
# ``` 
# 

# ## Markdown
# 
# First note this cell - it is a **markdown** cell, not a 'code' cell. This allows for
# *literate programming*: we can explain our algorithm with bullets, 
# 
# - step 1
# - step 2
#   - substep 2.1
# - step 3
# 
# or with enumeration,
# 
# 1. step 1
# 1. step 2
#   1. substep 2.1
# 1. step 3
# 
# We can type maths in LaTeX, like this: find $\boldsymbol{u}\in V$ such that
# 
# $$
# a(\boldsymbol{u},\boldsymbol{v}) = \langle \daleth, \boldsymbol{v}\rangle
# \qquad\forall \boldsymbol{v} \in V.
# $$
# 
# There is no equation numbering, BiBTeX, graphics etc. AFAIK though... 
# perhaps see **bookdown**. HTML is allowed.

# ## Leontief Input-Output Models
# 
# This *input-output* problem has been set up in the accompanying slide show.
# 
# We have 
# $\boldsymbol{x} = \boldsymbol{A}\boldsymbol{x}+\boldsymbol{d}$ or, alternatively, 
# $(\boldsymbol{I} - \boldsymbol{A})\boldsymbol{x} = \boldsymbol{d}$
# 
# First some simple arithmetic as a warm up... And an introduction to `numpy`

# In[1]:


0.9*50000 - 0.5*40000


# In[2]:


-0.3*50000 + 0.8*40000


# In[3]:


print(0.9*50000 - 0.5*40000, -0.3*50000 + 0.8*40000)


# Note that
# $$
# \boldsymbol{A} = {0.1\ 0.5\choose 0.3\ 0.2}
# \Longrightarrow
# \boldsymbol{I}-\boldsymbol{A}
# = {1\ 0\choose 0\ 1}-{0.1\ 0.5\choose 0.3\ 0.2} = {\phantom{-}0.9\ \ -0.5\choose -0.3\ \ \phantom{-}0.8}
# $$
# and therefore
# $$
# (\boldsymbol{I}-\boldsymbol{A})^{-1} = \frac{1}{(0.9)(0.8) - (0.5)(0.3)}
# {0.8\ \ 0.5\choose 0.3\ \ 0.9}
# =\frac{10}{57}{8\ \ 5 \choose 3\ \ 9}
# $$
# What are these?
# $$
# \frac{10}{57}{8\ \ 5 \choose 3\ \ 9}{35000\choose 29000}
# \qquad\text{ and }\qquad
# \frac{10}{57}{8\ \ 5 \choose 3\ \ 9}{2500\choose 1900}
# $$

# In[4]:


d1=35000; d2=29000; print(10/57*(8*d1 + 5*d2), 10/57*(3*d1 + 9*d2))


# In[5]:


d1=2500; d2=1900; print(10/57*(8*d1 + 5*d2), 10/57*(3*d1 + 9*d2))


# Let's introduce `numpy` - for arrays, and hence linear algebra.
# We'll duplicate the calculation above...

# In[6]:


import numpy as np
A = np.array([[0.1, 0.5],[0.3, 0.2]]) # use tab completion on 'np.a'
Id = np.eye(2)
print(Id-A)
print(5.7*np.linalg.inv(Id-A))   # using inverse matrices is usually bad! 
d = np.array([[35000],[29000]])
print(np.linalg.solve(Id-A, d))
Dd = np.array([[2500],[1900]])
print(np.linalg.solve(Id-A, Dd))


# Back to the Leontief IO problem...

# In[7]:


print('This is the technical matrix: A = ')
A = np.array([[0.15, 0.12, 0.05, 0.03],
              [0.17, 0.16, 0.04, 0.04],
              [0.03, 0.08, 0.18, 0.22],
              [0.07, 0.18, 0.03, 0.19]])
print(A)
print('Here are the total output levels: x = ')
x = np.array([[89000], [55000], [47000], [76000]])
print(x.T)
print('Note the transpose - a tidier output...')


# In[8]:


print('This is the Leontiev matrix: I-A = ')
Id = np.eye(4)
print(Id-A)
print('The amounts available for external demand are: d = ')
print((Id-A).dot(x).T)


# In[9]:


print('The required total output is: x = ')
d = np.array([[55000],[24000],[18000],[40000]])
print(np.linalg.solve(Id-A, d).T)
print('The change in external demand: Dd = ')
Dd = np.array([[-5000], [350], [2300],[-500]])
print(Dd.T)
print(np.linalg.solve(Id-A, Dd).T)


# ## Eigensystems and SVD
# 
# While we are looking at linear algebra it's useful to look at
# eigenvalue problems and the related SVD. We'll stay with the
# technical matrix $\boldsymbol{A}$ as defined above...

# In[10]:


w, V = np.linalg.eig(A)
print('The eigenvalues are ', w) # the eigenvalues
print(f'the shape of the eigenvector matrix V is {V.shape}')
print('and the first two eigenvectors are')
print(V[:,:2])


# We can check that $\boldsymbol{A}\boldsymbol{V} = \boldsymbol{V}\boldsymbol{D}$...
# 
# See e.g. <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>

# In[11]:


D=np.diag(w)
print('D = \n', D)
# print(A@V - V@D)  # uncomment it if you like, but this is better...
print(f"Frobenius norm, ||AV-VD||F = {np.linalg.norm(A@V - V@D, ord='fro')}")


# #### The SVD ...
# 
# There are a couple of *gotchas* for the SVD, it works like this...

# In[12]:


K = np.array([[1,2,5],[5,-6,1]])   
U, S, VT = np.linalg.svd(K)
print('U is what we expect', U)
print('But S is not!', S)
print('V-transpose gets returned, not V', VT)


# We can stack `S` to get what we expect ...

# In[13]:


S = np.hstack(( np.diag(S), np.zeros((2,1)) ))
print('S is now the correct shape\n', S)

# print(K - U @ S @ VT) # again, uncomment if you like but this is better
print(f"The inf-norm ||K - U S V^T||inf = {np.linalg.norm(K - U @ S @ VT, np.inf)}")


# ### Exercises 1
# 
# Open a new cell under this and attempt these...
# 
# 1. Find the required total output for
# $\boldsymbol{d} = (51k, 26k, 15k, 48k)^T$.
# 
# 2. Find (and verify) the eigensystem for the $N\times N$ tridiagonal Laplacian matrix
# 
# $$
# \boldsymbol{A} = \frac{1}{h^2}\left(\begin{array}{rrrrrrrrr}
#          2 & -1 &  0 & \cdots          \\
#         -1 &  2 & -1 & 0 & \cdots      \\
#          0 & -1 &  2 & -1 & 0 & \cdots      \\
#     \cdots &  0 & -1 &  2 & -1 & 0 & \cdots      \\
#            &    &  &\ddots  &\ddots  &\ddots &  \\
# && \cdots  & 0 & -1 &  2 & -1 & 0      \\
# &&& \cdots         &  0 & -1 &  2 & -1      \\
# &&&& \cdots & 0 & -1 &  2 \\
# \end{array}\right)
# $$
# where $h = (N+1)^{-2}$. Hint:
# ```
# diag = np.ones(4)
# print(2*np.diag(diag)-np.diag(diag[1:],-1)-np.diag(diag[1:],1))
# ```

# ## Plotting in 2D
# 
# We introduce `matplotlib`. There are others but this seems to be the most common.
# 
# Here is a simple example

# In[14]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-1,5,0.01)
y1, y2 = np.cos(2*np.pi*x), np.exp(np.sin(4*np.pi*x))
plt.plot(x,y1, 'b-')
plt.plot(x,y2, 'r--')
plt.axis([-2, 6, -2, 4])
plt.legend(['cos', 'exp(sin)'])
plt.xlabel(r'$x_1$'); plt.ylabel('$y_1$ and $y_2$') 
plt.savefig('./gfx/my2Dplot.png', dpi=600)
plt.savefig('./gfx/my2Dplot.eps', dpi=600)


# Note that once we have imported we don't have to do it again.
# 
# However - make sure you execute the notebook from the top down!

# In[15]:


x = np.arange(-4,3,0.01)
y1, y2 = 2**(3*np.sin(3*np.pi*x)), np.log(1.2+np.sin(3*np.pi*x))
plt.plot(x,y1, 'b-.')
plt.plot(x,y2, 'r:')
plt.axis([-5, 4, -2, 10])
plt.legend(['2^(3sin)', 'log(1.2+sin)'], fontsize=15)
plt.xlabel(r'$x_1$', fontsize=20); plt.ylabel('$y_1$ and $y_2$', fontsize=20) 
plt.savefig('./gfx/my2Dplot2.png', dpi=600)
plt.savefig('./gfx/my2Dplot2.eps', dpi=600)


# ### Exercises 2
# 
# 1. Find (and verify) the SVD of 
# $$
# \boldsymbol{K} = \left(\begin{array}{rrrr}
#  2 & -5 &  7 & -9 \\
# -1 &  2 & -1 &  0 \\
# \end{array}\right)
# $$
# Find a rank one approximation to $\boldsymbol{K}$ and determine the error
# in the Frobenius norm (Hint: look up `np.linalg.norm()`).
# 
# 2. Consider `y = np.heaviside(np.sin(2*np.pi*x),0)` and plot a 
# square waves with periods $\pi$ and $\pi^2$ and amplitudes $2$ and $5$.
# (Hint: use `np.pi**2`)

# ## Anonymity
# 
# Based on 
# 
#  - https://www.johndcook.com/blog/2018/12/07/simulating-zipcode-sex-birthdate/
#  - https://techscience.org/a/2015092903/
#  

# In[16]:


from random import randrange
import matplotlib.pyplot as plt
import numpy as np

d = 365*3*2
N = 750
buckets = np.zeros(d)

for _ in range(N):
    z = randrange(d)
    buckets[z] += 1

plt.hist(buckets, range(1,10)); # note the semi-colon (try without)


# In[17]:


loners = len(buckets[buckets==1])
print('Probability that anonymous data occurs only once: ', loners/N)
print('Nearly exact probability that anonymous data occurs only once: ',
       np.exp(-N/d))
loners2 = len(buckets[buckets==2])
print('Probability that anonymous data occurs at most twice: ',
      (loners+2*loners2)/N)
loners3 = len(buckets[buckets==3])
print('Probability that anonymous data occurs at most three times: ',
      (loners+2*loners2+3*loners3)/N)
loners4 = len(buckets[buckets==4])
print('Probability that anonymous data occurs at most four times: ',
      (loners+2*loners2+3*loners3+4*loners4)/N)


# In[18]:


print( sum(buckets) )
print( len(buckets[buckets!=0]), end = ', ')
print( len(buckets[buckets==1]), end = ', ' )
print( len(buckets[buckets==2]), end = ', ' )
print( len(buckets[buckets==3]), end = ', ' )
print( len(buckets[buckets==4]), end = ', ' )
print( len(buckets[buckets>4]), end = ', ' )

# a check
print( len(buckets[buckets==1])+2*len(buckets[buckets==2])
       +3*len(buckets[buckets==3])+4*len(buckets[buckets==4]) )


# ## Discrete Fourier Transform
# 
# Here's an example from the docs: 
# - <https://docs.scipy.org/doc/scipy/reference/fft.html#module-scipy.fft>
# - <https://docs.scipy.org/doc/scipy-1.5.2/reference/tutorial/fft.html>
# 

# In[19]:


from scipy.fft import fft, fftfreq, ifft
N = 600; dt = 1.0 / 800.0   # number of sample points and sample spacing
t = np.linspace(0.0, N*dt, N)
y = np.sin(50.0 * 2.0*np.pi*t) + 0.5*np.sin(80.0 * 2.0*np.pi*t)
plt.figure(figsize=(12, 4)); plt.plot(t, y);  plt.grid(); plt.show()


# In[20]:


yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*dt), N//2) # or xf = fftfreq(N,T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2])); plt.grid(); plt.show()


# Let's noise it up a bit...

# In[21]:


#import numpy.random.normal
yn  = np.sin(50.0 * 2.0*np.pi*t) + 0.5*np.sin(80.0 * 2.0*np.pi*t)
yn += np.random.normal(0,200*dt,yn.shape)
plt.figure(figsize=(12, 4)); plt.plot(t, yn);  plt.grid(); plt.show()


# In[22]:


ynf = fft(yn)
xf = np.linspace(0.0, 1.0/(2.0*dt), N//2) # or xf = fftfreq(N,T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(ynf[0:N//2])); plt.grid(); plt.show()


# ## ODE Solver - Huen's method
# 
# Given $\dot{\boldsymbol{z}} = \boldsymbol{f}(t,\boldsymbol{z(t)})$ Heun's method, with 
# $\boldsymbol{z}(0)=\boldsymbol{z}_0$ is,
# 
# \begin{align*}
# \boldsymbol{y}_{n+1}
# & = \boldsymbol{z}_n + k \boldsymbol{f}(t_n,\boldsymbol{z}_n),
# \\
# \boldsymbol{z}_{n+1}
# & =
# \boldsymbol{z}_n + \frac{k}{2}\Big(
# \boldsymbol{f}(t_n,\boldsymbol{z}_n)
# +
# \boldsymbol{f}(t_{n+1},\boldsymbol{y}_{n+1})
# \Big),
# \end{align*}
# 
# for $n=0,1,2,3,..., N-1$ and where $k=T/N$ for the final time $T$.
# 
# Example (from Example 8.4B in Numerical Analysis, a practical approach, MJ Maron and RJ Lopez,
# Wordsworth Publishing Company, 1991):
# 
# $$
# \dot{\boldsymbol{z}}
# = \left(\begin{array}{r} \dot{x} \\ \dot{y} \end{array}\right)
# = 
# \left(\begin{array}{r} 0 \\ t \end{array}\right)
# +
# \left(\begin{array}{rr} 0 & 1 \\ 1 & 0 \end{array}\right)
# \left(\begin{array}{r} x \\ y \end{array}\right)
# = 
# \left(\begin{array}{r} 0 \\ t \end{array}\right)
# +
# \boldsymbol{B}\boldsymbol{z}
# = 
# \boldsymbol{f}(t,\boldsymbol{z})
# $$
# 
# with $\boldsymbol{z}(0) = (1,-1)^T$ has solution
# 
# $$
# \boldsymbol{z}
# = \frac{1}{2}\left(\begin{array}{r} 
# \exp(t)+\exp(-t)-2t
# \\
# \exp(t)-\exp(-t)-2
# \end{array}\right).
# $$
# 
# Here is the code... With an example definition of a function...

# In[23]:


# NOTE the use of [:,[n]] rather that [:,n] - a slice gives a column...
def Huen(N,k,B):
  # allocate solution vector and set up initial condition
  z = np.zeros((2,N+1)); z[0,0] = 1; z[1,0] = -1
  for n in range(N):
    tn = n*k
    f = np.array([[0],[tn]]) + B @ z[:,[n]]
    z[:,[n+1]] = z[:,[n]] + k*f
    f = f + np.array([[0],[tn+k]]) + B @ z[:,[n+1]]
    z[:,[n+1]] = z[:,[n]] + 0.5*k*f
  return z[:,[N]]

T = 10; N1 = 64; B = np.array([[0,1],[1,0]])
zexact  = 0.5*np.array([[np.exp(T)+np.exp(-T)-2*T],[np.exp(T)-np.exp(-T)-2]])
maxit = 8; Nvals = np.zeros(maxit); errors = np.zeros(maxit)
for i in range(maxit):
    N = N1*2**i; k = T/N; Nvals[i] = N;
    zapprox = Huen(N,k,B)
    errors[i] = np.linalg.norm(zexact - zapprox)
    print(errors[i])


# In[24]:


plt.loglog(Nvals,errors); plt.xlabel('N'); plt.ylabel('||error||');


# # Hypothesis Testing
# 
# Example taken from: <https://www.jmp.com/en_gb/statistics-knowledge-portal/t-test/one-sample-t-test.html> and with some input from <https://www.geeksforgeeks.org/how-to-find-the-t-critical-value-in-python/>. 

# In[25]:


from scipy.stats import t
x = np.linspace(-10,10,100); y = t.pdf(x, df=8); plt.figure(figsize=(3,3)); plt.plot(x,y);


# In[26]:


prot = np.array([20.70, 27.46, 22.15, 19.85, 21.29, 24.75, 20.75, 22.91, 25.34, 20.33, 21.54, 21.08,
                 22.14, 19.56, 21.10, 18.04, 24.12, 19.95, 19.72, 18.28, 16.26, 17.46, 20.53, 22.12,
                 25.06, 22.44, 19.08, 19.88, 21.39, 22.33, 25.79])
N = prot.shape[0]
null_mean = 20
print(f'There are {N} samples with mean {prot.mean()}, std dev = {prot.std(ddof=1)} and variance {prot.var()}')
print(f'Note std is unbiased: direct calc gives ', end=' ')
print(np.sqrt(np.power(prot,2).sum()/N - np.power(prot.sum()/N,2)), end=', ')
print(f'or just use ', prot.std())
diff_mean = prot.mean() - null_mean
print(f'difference from null mean: {prot.mean()} - {null_mean} = {diff_mean}') 
std_err = prot.std(ddof=1)/np.sqrt(N)
tstat = diff_mean/std_err
print(f'standard error = {std_err}, and t statistic = {tstat}')
alpha = 0.05
tcrit = t.ppf(q=1-alpha/2, df=N-1)  # q is lower tail probability: => two-sided
print(f'with alpha = {alpha}, critical t is {tcrit}')
print(f'{tstat} > {tcrit} so we reject the null hypothesis at {100*alpha}%')
print(f'the p value is {2*t.cdf(-tstat, df=N-1)}')


# In[30]:


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html#scipy.stats.ttest_1samp
from scipy.stats import ttest_1samp
#stat, pval = ttest_1samp(prot, 20, axis=0, alternative='two-sided')
stat, pval = ttest_1samp(prot, 20, axis=0)
print(f'results are stat = {stat} and p value = {pval}')
#ttest_1samp(prot, 20, axis=0, alternative='two-sided')
ttest_1samp(prot, 20, axis=0)


# # Coin tossing
# 
# Toss a coin 12 times, with a bias. A head is 1, a tail is zero.
# 

# In[31]:


# random integers from low (inclusive) to high (exclusive).
bias = 64 #  bias+1 is probability of a head.
tosses = np.random.randint(0, high=100, size=12, dtype=int)
print(tosses)
indx = tosses>bias
print(indx)
tosses[indx] = 1
tosses[~indx] = 0
print(tosses)


# ### Exercise
# 
# - test the hypothesis that the coin is fair.
# - put the test in a loop. How many times would you expect to reject the null?

# # Photo Compression
# 
# Either get your own jpeg or use one of the supplied ones...

# In[32]:


from PIL import Image
import IPython.display
# Use a jpeg photo - ffc.jpg is about 6.2MB (use your own path/filename here)
IPython.display.Image(filename='./gfx/ffc.jpg', width = 150)


# In[33]:


# that is just a display, so ... load in the FFC bear - Roy - and visually check him.
img = Image.open('./gfx/ffc.jpg')
# convert him to a numpy array for processing as a matrix
a = np.asarray(img)
im_orig = Image.fromarray(a)
plt.imshow(im_orig);


# This image is made up of pixels where each pixel has a value for RED, GREEN and BLUE. We
# can get these colour bands and show them as follows…

# In[34]:


# convert band 'bnd' to a numpy array and show them...
for bnd in range(3):
  plt.subplot(1, 3, 1+bnd)
  img_mat = np.array(list(img.getdata(bnd)), float)
  img_mat = np.matrix(img_mat)
  img_mat.shape = (img.size[1], img.size[0])
  plt.imshow(img_mat)
plt.subplots_adjust(wspace=0.5)


# In[35]:


# get the red, green and blue bands as separate objects...
rband =img.getdata(band=0)
gband =img.getdata(band=1)
bband =img.getdata(band=2)
# and convert each to a numpy arrays for maths processing
imgr_mat = np.array(list(rband), float)
imgg_mat = np.array(list(gband), float)
imgb_mat = np.array(list(bband), float)
# each of these is about 64k elements
print('sizes = ', imgr_mat.size, imgg_mat.size, imgb_mat.size)
print('shapes = ', imgr_mat.shape, imgg_mat.shape, imgb_mat.shape)


# In[36]:


# get image shape - we can assume they are all the same
imgr_mat.shape = imgg_mat.shape = imgb_mat.shape = (img.size[1], img.size[0])
print('imgr_mat.shape = ', imgr_mat.shape)
print('imgg_mat.shape = ', imgg_mat.shape)
print('imgb_mat.shape = ', imgb_mat.shape)
# convert these 1D-arrays to matrices
imgr_mat1D = np.matrix(imgr_mat)
imgg_mat1D = np.matrix(imgg_mat)
imgb_mat1D = np.matrix(imgb_mat)
print(type(imgb_mat))

Using the Singular Value Decomposition we can hope to compress these objects.

First get the SVD’s of the R, G and B layers… (takes a while)
# In[37]:


Ur, Sr, VTr = np.linalg.svd(imgr_mat)
Ug, Sg, VTg = np.linalg.svd(imgg_mat)
Ub, Sb, VTb = np.linalg.svd(imgb_mat)
print(f'RED: shapes of Ur, Sr, VTr = {Ur.shape}, {Sr.shape}, {VTr.shape}')
print(f'GREEN: shapes of Ug, Sg, VTg = {Ug.shape}, {Sg.shape}, {VTg.shape}')
print(f'BLUE: shapes of Ub, Sb, VTb = {Ub.shape}, {Sb.shape}, {VTb.shape}')


# In[38]:


# choose the number of components to use in the reconstruction
nc = 10 # 1387
rec_imgr = np.matrix(Ur[:, :nc]) * np.diag(Sr[:nc]) * np.matrix(VTr[:nc, :])
rec_imgg = np.matrix(Ug[:, :nc]) * np.diag(Sg[:nc]) * np.matrix(VTg[:nc, :])
rec_imgb = np.matrix(Ub[:, :nc]) * np.diag(Sb[:nc]) * np.matrix(VTb[:nc, :])
img_all = np.array([rec_imgr, rec_imgg, rec_imgb]).T
img_all = np.swapaxes(img_all,0,1)
PIL_image = Image.fromarray(np.uint8(img_all)).convert('RGB')
PIL_image.show()   # uncomment this to spawn an external viewer
PIL_image.save("ffc_recon.jpg")  # save the reconstruction if you like


# In[39]:


fig=plt.figure(figsize=(4, 3)); fig.suptitle('Comparison', fontsize=15)
plt.subplot(1,2,1); ax = plt.gca(); plt.subplots_adjust(wspace=0.5)
im = Image.fromarray(np.uint8(img_all)).convert('RGB'); ax.imshow(im)
ax.set_title(f'Recon, nc = {nc}', fontsize=10)
plt.subplot(1,2,2); ax = plt.gca(); ax.imshow(im_orig); ax.set_title('original', fontsize=10);


# #### Exercise
# Calculate the memory saving as a function of `nc` and plot it.

# ## Going Further
# 
# if you want to see uses of python in machine learning and data science then you can look at my 
# MA5634 binder page here:
# 
# <https://mybinder.org/v2/gh/variationalform/FML.git/HEAD>
# 

# This code creates a button - but it breaks LaTeX output
# 
# ```
# <p><a rel="noopener" href="https://mybinder.org/v2/gh/variationalform/FML.git/HEAD">https://mybinder.org/v2/gh/variationalform/FML.git/HEAD</a></p>
# <p>Or just click this button:&nbsp;<a rel="noopener" href="https://mybinder.org/badge_logo.svg"> </a><a rel="noopener" href="https://mybinder.org/v2/gh/variationalform/FML.git/HEAD"> <img src="https://mybinder.org/badge_logo.svg" width="120"> </a></p>
# <p></p>
# ```

# The raw materials are on git:
# 
# - <https://variationalform.github.io>
# - <https://github.com/variationalform/PythonMathsPrimer>

# ## Technical Notes, Production and Archiving
# 
# Ignore the material below. What follows is not relevant to the material being taught.

# #### Production Workflow
# 
# - Finalise the notebook material above
# - Clear and fresh run of entire notebook
# - Create html slide show:
#   - `jupyter nbconvert --to slides PythonMathsPrimer.ipynb `
# - Set `OUTPUTTING=1` below
# - Comment out the display of web-sourced diagrams
# - Clear and fresh run of entire notebook
# - Comment back in the display of web-sourced diagrams
# - Clear all cell output
# - Set `OUTPUTTING=0` below
# - Save
# - git add, commit and push to FML
# - copy PDF, HTML etc to web site
#   - git add, commit and push
# - rebuild binder

# In[40]:


get_ipython().run_cell_magic('bash', '', 'NBROOTNAME=PythonMathsPrimer\nOUTPUTTING=1\n\nif [ $OUTPUTTING -eq 1 ]; then\n  #jupyter nbconvert --to html $NBROOTNAME.ipynb\n  #cp $NBROOTNAME.html ./backups/$(date +"%m_%d_%Y-%H%M%S")_$NBROOTNAME.html\n  #mv -f $NBROOTNAME.html ./formats/\n\n  jupyter nbconvert --to slides $NBROOTNAME.ipynb\n  cp $NBROOTNAME.slides.html ./backups/$(date +"%m_%d_%Y-%H%M%S")_$NBROOTNAME.slides.html\n  mv -f $NBROOTNAME.slides.html ./formats/\n\n  jupyter nbconvert --to pdf $NBROOTNAME.ipynb\n  cp $NBROOTNAME.pdf ./backups/$(date +"%m_%d_%Y-%H%M%S")_$NBROOTNAME.pdf\n  mv -f $NBROOTNAME.pdf ./formats/\n\n  jupyter nbconvert --to script $NBROOTNAME.ipynb\n  cp $NBROOTNAME.py ./backups/$(date +"%m_%d_%Y-%H%M%S")_$NBROOTNAME.py\n  mv -f $NBROOTNAME.py ./formats/\nelse\n  echo \'Not Generating html, pdf and py output versions\'\nfi')


# In[ ]:




