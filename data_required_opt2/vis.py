import numpy as np
from scipy.stats import entropy
from tqdm import trange
from typing import Tuple

from nn import Tensor


### PCA ###

def squash(samples: Tensor, n: int) -> Tensor:
  mat = np.dot(np.transpose(samples), samples)
  eigvals, eigvecs = np.linalg.eigh(mat)
  ind = np.argmax(eigvals)
  res = np.expand_dims(np.dot(samples, eigvecs[ind]), axis=1) # already normalised
  eigvals[ind] = -1
  for _ in range(n - 1):
    ind = np.argmax(eigvals)
    res = np.hstack((res, np.expand_dims(np.dot(samples, eigvecs[ind]), axis=1)))
    eigvals[ind] = -1
  return res

### t-SNE ###

# the derivation form would be too clustered if following the exact definition
# in lecture. so i adopt a symmetric version here
def t_sne2(samples: Tensor, perp: float) -> Tuple[Tensor, Tensor]:
  momentum = 0.7
  step = 50
  nrounds = 1000

  if len(samples) >= 50:
    samples = squash(samples, 50)
  n = len(samples)

  P = np.empty([n, n])
  for i in trange(n): # find variances
    px = -1
    low = 0
    high = 100
    while True:
      mid = (low + high) / 2
      for j in range(n):
        dif = samples[i] - samples[j]
        P[i][j] = np.exp((-1 / 2 / mid) * np.dot(dif, dif))
      P[i][i] = 0
      P[i] = P[i] / P[i].sum()
      H = entropy(P[i], base=2)
      px = np.exp2(H)
      if np.abs(perp - px) < 0.1:
        break
      elif px < perp:
        low = mid
      else:
        high = mid

  P = (P + np.transpose(P)) / 2 / n

  xys = squash(samples, 2) # initialize with PCA
  xsys = np.transpose(xys)
  xs = xsys[0]
  ys = xsys[1]
  accx = np.empty([n])
  accy = np.empty([n])
  for i in trange(nrounds):
    xss = np.repeat(np.expand_dims(xs, axis=0), n, axis=0)
    yss = np.repeat(np.expand_dims(ys, axis=0), n, axis=0)
    Dx = np.transpose(xss) - xss
    Dy = np.transpose(yss) - yss
    D = Dx * Dx + Dy * Dy
    Q = 1 / (1 + D)
    np.fill_diagonal(Q, 0)
    Z = Q.sum()
    Q = Q / Z
    dxs = 4 * np.sum((P - Q) / (1 + D) * Dx, axis=1)
    dys = 4 * np.sum((P - Q) / (1 + D) * Dy, axis=1)
    if i == 0:
      accx = dxs
      accy = dys
    else:
      accx = momentum * accx + (1 - momentum) * dxs
      accy = momentum * accy + (1 - momentum) * dys
    xs = xs - step * accx
    ys = ys - step * accy
  return (xs, ys)

