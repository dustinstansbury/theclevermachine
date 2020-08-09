---
layout: post
title: "Efficient Matrix Power Calculation via Diagonalization"
author: Dustin Stansbury
date: 2020-08-08
tags: linear-algebra matrix-diagonalization
permalink: /matrix-power-using-diagonalization
---

Taking the power of a matrix is an important operation with applications in statistics, machine learning, and engineering. For example, solving linear ordinary differential equations, identifying the state of a Markov chain at time $$t$$, or identifying the number of paths between nodes in a graph can all be solved using powers of matrices. In this quick post we'll show how Matrix Diagonalization can be used to efficiently compute the power of a matrix.

If matrix $$M$$ is an $$m \times m$$ diagonalizable, then $$M^k$$ can be calculated directly from the diagonalization $$M = P D P^{-1}$$ as follows:

$$
\begin{align}
M^k &= M \times M \dots \times M \\
&= (P D P^{-1}) (P D P^{-1}) \dots (P D P^{-1}) \\
&= P D (P^{-1} P) D (P^{-1} P) \dots D P^{-1} \\
&= P D^k P^{-1}
\end{align}
$$

Therefore to calculate $$M^k$$, we simply need to diagonalize $$M$$ and re-matrix-multiply the diagonalization components after raising the diagonal matrix component $$D$$ to the $$k$$-th power. Since $$D$$ is a diagonal matrix, the $$k$$-th power is calculated by simply raising each element along the diagonal to the $$k$$-th power:

$$
\begin{align}
D^k &= \begin{bmatrix}
    d_{1} & & \\
    & \ddots & \\
    & & d_{m}
  \end{bmatrix}^k \\
&= \begin{bmatrix}
    d_{1}^k & & \\
    & \ddots & \\
    & & d_{m}^k
  \end{bmatrix}

\end{align}
$$

This trick allows us to calculate the matrix power by multiplying three matrices, rather than $$k$$. Thus as $$k$$ gets large, or the size of the matrix $$M$$ grows, you get more and more gains in efficiency.

To demonstrate, let's calculate the matrix power of a random matrix using **brute force**, the **matrix diagonalization** approach reviewed above, and we'll also throw in results from `numpy.linalg.matrix_power` for completeness.

```python
import numpy as np
np.random.seed(123)

# Generate a random 3 x 3 matrix
M = np.random.randn(3, 3)
k = 3  # power exponent

print('\nBrute Force:\n', eval("@".join([' M '] * k)))
# Brute Force:
#  [[-0.34077132 -0.70544947 -1.07778229]
#  [ 2.73462284 -0.71537115 -2.62514227]
#  [ 3.35955945  1.68986542 -4.1619396 ]]

# Diagonalize M via Eigenvalue Decomposition
D, P = np.linalg.eig(M)
D = np.diag(D)  # Put eigenvalues into a diagonal matrix

print('\nMatrix Diagonalization:\n', np.real(P @ D ** k @ np.linalg.inv(P)))
# Matrix Diagonalization:
#  [[-0.34077132 -0.70544947 -1.07778229]
#  [ 2.73462284 -0.71537115 -2.62514227]
#  [ 3.35955945  1.68986542 -4.1619396 ]]

print('\nnumpy.linalg.matrix_power:\n', np.linalg.matrix_power(M, k))
# numpy.linalg.matrix_power:
#  [[-0.34077132 -0.70544947 -1.07778229]
#  [ 2.73462284 -0.71537115 -2.62514227]
#  [ 3.35955945  1.68986542 -4.1619396 ]]
```

Works! 😁 
