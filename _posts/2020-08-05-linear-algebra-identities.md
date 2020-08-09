---
layout: post
title: "Common Linear Algebra Identities"
author: Dustin Stansbury
date: 2020-08-05
tags: derivation linear-algebra matrix-identities
permalink: /linear-algebra-identities
---

This post provides a convenient reference of Linear Algebra identities used in The Clever Machine Blog.

# Notation

$$

\begin{align}

\text{A Scalar:}&\;\;\;a \\
\text{A Column Vector:}&\;\;\;\mathbf{a} = [a_1, a_2, ...a_n]^T \\
\text{The } i\text{-th entry of a vector:}&\;\;\;a_i \\
\text{Vector Inner (Scalar) Product:}&\;\;\;\mathbf{a}^T\mathbf{a} \\
\text{Vector Outer (Matrix) Product:}&\;\;\;\mathbf{a}\mathbf{a}^T \\

\\
\text{A Matrix:} &\;\;\;\mathbf{A}& \\
\text{The } i,j\text{-th entry of a matrix:}&\;\;\;A_{ij} \\
\text{The Null Matrix (all zero entries):} &\;\;\;\mathbf{0}& \\
\text{The Identity Matrix:} &\;\;\;\mathbf{I}& \\
\text{A Diagonal Matrix:} &\;\;\;\mathbf{\Lambda}& \\
\text{A Positive Definite Matrix:} &\;\;\;\mathbf{\Sigma} \\
\text{Matrix of size }\mathbf{A}\text{ filled with zeros except a single 1 at } i,j &\;\;\;\mathbf{\Delta}(\mathbf{A})_{ij} \\

\\
\text{Matrix Transpose:}&\;\;\;\mathbf{A}^T \\
\text{Matrix Identity:}&\;\;\;\mathbf{A}^{-1} \\
\text{Matrix Pseudo Inverse:}&\;\;\;\mathbf{A}^+ \\
\text{Matrix Square Root:}&\;\;\;\mathbf{A}^{1/2} \\
\text{Matrix Complex Conjugate:}&\;\;\;\mathbf{A}^* \\
\text{Hermitian of a Matrix:}&\;\;\;\mathbf{A}^H \\
\text{Determinant of a Matrix:}&\;\;\;\det(\mathbf{A}) \\
\text{Trace of a Matrix:}&\;\;\;\text{tr}(\mathbf{A}) \\
\text{Diagonal Matrix:}&\;\;\;\text{diag}(\mathbf{A}),  \;\;\; (\text{diag}(\mathbf{A}))_{ij} = \delta_{ij}(A)\\
\text{Eigenvalues of a Matrix:}&\;\;\;\text{eig}(\mathbf{A}) \\
\text{Norm of a Matrix:}&\;\;\;||\mathbf{A}|| \\ 
\text{Hadamard (elementwise) product of two Matrices:}&\;\;\;\mathbf{A} \circ \mathbf{B} \\
\text{Kronecker Product of Two Matrices:}&\;\;\;\mathbf{A} \otimes \mathbf{B} \\


\end{align}
$$


# 1. Basic Properties


$$
\begin{eqnarray}
\mathbf{A} + \mathbf{B} &=& \mathbf{B} + \mathbf{A} \tag{1.1} \\
\mathbf{A} + (\mathbf{B} + \mathbf{C}) &=& (\mathbf{A} + \mathbf{B}) + \mathbf{C} \tag{1.2} \\
\mathbf{A} (\mathbf{B} + \mathbf{C}) &=& (\mathbf{AB} + \mathbf{AC}) \tag{1.3} \\
a(\mathbf{B} + \mathbf{C}) &=& (a\mathbf{B} + a\mathbf{C}) = (\mathbf{B}a + \mathbf{C}a) \tag{1.4} \\
\mathbf{AB} &\neq& \mathbf{BA} \tag{1.5} \\
\mathbf{ABC} &=& (\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC}) \tag{1.6} \\

\end{eqnarray}
$$

# 2. Transposes

$$
\begin{eqnarray}
(\mathbf{A}^T)^T &=& \mathbf{A} \tag{2.1} \\
(\mathbf{AB})^T &=& \mathbf{B}^{T}\mathbf{A}^{T}  \tag{2.2} \\
(\mathbf{ABC})^T &=& \mathbf{C}^{T}\mathbf{B}^{T}\mathbf{A}^{T} \tag{2.3} \\
(\mathbf{A} + \mathbf{B})^T &=& (\mathbf{A}^T + \mathbf{B}^T) \tag{2.4} \\
\end{eqnarray}
$$


# 3. Inverses and Identity

$$
\begin{eqnarray}
\mathbf{AI} &=& \mathbf{IA} = \mathbf{A}  \tag{3.1} \\
\mathbf{AA}^{-1} &=& \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}  \tag{3.2} \\
(\mathbf{A}^{-1})^{-1} &=& \mathbf{A} \tag{3.3} \\
(\mathbf{AB})^{-1} &=& \mathbf{B}^{-1}\mathbf{A}^{-1}  \tag{3.4} \\
(\mathbf{ABC})^{-1} &=& \mathbf{C}^{-1}\mathbf{B}^{-1}\mathbf{A}^{-1}  \tag{3.5} \\
(\mathbf{A}^T)^{-1} &=& (\mathbf{A}^{-1})^T \tag{3.6} \\
\mathbf{\Lambda}^{-1}&=& \text{diag}([1/\lambda_1, 1/\lambda_2, ... 1/\lambda_n]) \tag{3.7} \\
\end{eqnarray}
$$


# 4. Traces

$$
\begin{eqnarray}
\text{tr}(\mathbf{A}) &=& \sum_i A_{ii} \tag{4.1} \\
\text{tr}(\mathbf{A}^T) &=& \text{tr}(\mathbf{A}) \tag{4.2} \\
\text{tr}(\mathbf{AB}) &=& \text{tr}(\mathbf{BA}) \tag{if A & B are the same size,  4.3} \\
&=& \text{tr}(\mathbf{B}^T\mathbf{A}^T) \tag{if A & B are not the same size,  4.4} \\
\text{tr}(\mathbf{A} + \mathbf{B}) &=& \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B}) \tag{4.5} \\
\text{tr}(\mathbf{ABC}) &=& \text{tr}(\mathbf{BCA}) = \text{tr}(\mathbf{CAB}) \tag{4.6} \\
\mathbf{a}^T\mathbf{a} &=& \text{tr}(\mathbf{aa}^T) \tag{4.7} \\
\text{tr}(\mathbf{A}) &=& \sum_i \lambda_{i}, \;\;\; \lambda_i = \text{eig}(\mathbf{A})_i \tag{4.8} \\
\end{eqnarray}
$$


# 5. Determinants

For a square matrix $$\mathbf A$$ of dimension $$n \times n$$

$$
\begin{eqnarray}
\det(\mathbf{A}) &=& \prod_i \lambda_{i}, \;\;\; \lambda_i = \text{eig}(\mathbf{A})_i \tag{5.1} \\
\det(\mathbf{A}^T) &=& \det(\mathbf{A}) \tag{5.2} \\
\det(\mathbf{AB}) &=& \det(\mathbf{A})\det(\mathbf{B}) \tag{5.3} \\
\det(\mathbf{A}^{-1}) &=& \frac{1}{\det(\mathbf{A})} \tag{5.4} \\
\det(\mathbf{A}^n) &=& \det(\mathbf{A})^n \tag{5.5} \\
\det(c\mathbf{A}) &=& c^n \det(\mathbf{A}), \;\;\;  \text{given }  \mathbf{A} \in \mathbb{R}^{n \times n} \tag{5.6} \\

\end{eqnarray}
$$


# 6. Derivatives

####  6.1 Vector Derivatives

##### 6.1.1 Scalar-valued Objectives

For scalar function $$y = \mathbf{\beta x} = \beta_1 x_1 + \beta_2 x_2 + ... \beta_n x_n$$

$$

\begin{align}

\frac{\partial y}{\partial \mathbf{x}} &= 
\begin{bmatrix}
    \frac{\partial y}{\partial x_1} \\
    \frac{\partial y}{\partial x_2} \\
    \vdots \\
    \frac{\partial y}{\partial x_n} \\
\end{bmatrix}
= 
\frac{\partial \mathbf{\beta \mathbf{x}}}{\partial \mathbf{x}}
=
\begin{bmatrix}
    \frac{\partial \mathbf{\beta x}}{\partial x_1} \\
    \frac{\partial \mathbf{\beta x}}{\partial x_2} \\
    \vdots \\
    \frac{\partial \mathbf{\beta x}}{\partial x_n} \\
\end{bmatrix}
= 
\begin{bmatrix}
    \beta_1 \\
    \beta_2 \\
    \vdots \\
    \beta_n

\end{bmatrix} \tag{6.1.1}

\end{align}

$$

##### 6.1.2 Vector-valued Objectives

For a vector-valued function

$$ 
\begin{eqnarray}

\mathbf{y} = 
\begin{bmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_m \\
\end{bmatrix} = 

\mathbf{A x} =
\begin{bmatrix}
    a_{11}x_1 + a_{12}x_2 + ... + a_{1n}x_n \\
    a_{21}x_1 + a_{22}x_2 + ... + a_{2n}x_n \\
    \vdots \\
    a_{m1}x_1 + a_{m2}x_2 + ... + a_{mn}x_n \\
\end{bmatrix} \\ \\

\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =

\begin{bmatrix}
    \frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} & \dots  & \frac{\partial y_m}{\partial x_1} \\
    \frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2} & \dots  & \frac{\partial y_m}{\partial x_2} \\
    \vdots & \vdots  &  \ddots  &  \vdots \\
    \frac{\partial y_1}{\partial x_n} & \frac{\partial y_2}{\partial x_n} & \dots  & \frac{\partial y_m}{\partial x_n} \\
\end{bmatrix}  

= 

\frac{\partial \mathbf{Ax}}{\partial \mathbf{x}}

= 

\begin{bmatrix}
    a_{11} & a_{21} & \dots & a_{m1} \\
    a_{12} & a_{22} & \dots & a_{m2} \\
    \vdots & \vdots & \ddots &  \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn} \\
\end{bmatrix}
= \mathbf{A}^T  \tag{6.1.2}

\end{eqnarray}

$$


$$
\begin{eqnarray}
\frac{\partial \mathbf{x}^T\mathbf{A}}{\partial \mathbf{x}} &=& \mathbf{A} \tag{6.1.3} \\
\frac{\partial \mathbf{x}^T\mathbf{a}}{\partial \mathbf{x}} &=& \frac{\partial \mathbf{a}^T\mathbf{x}}{\partial \mathbf{x}} = \mathbf{a} \tag{6.1.3} \\
\frac{\partial \mathbf{y}^T \mathbf{Ax}}{\partial \mathbf{x}} &=&  \mathbf{A}^T \mathbf{y} \tag{6.1.4} \\
\frac{\partial \mathbf{y}^T \mathbf{Ax}}{\partial \mathbf{y}} &=&  \mathbf{A} \mathbf{x} \tag{6.1.5} \\
\frac{\partial \mathbf{x}^T\mathbf{x}}{\partial \mathbf{x}} &=&  2\mathbf{x} \tag{6.1.6} \\
\frac{\partial \mathbf{x}^T\mathbf{Ax}}{\partial \mathbf{x}} &=&  (\mathbf{A} + \mathbf{A}^T)\mathbf{x} \tag{6.1.7} \\
&=&2 \mathbf{Ax} \tag{if A is symmetric, 6.1.8} \\

\frac{\partial \mathbf{Ax}}{\partial \mathbf{z}} &=& \frac{\partial \mathbf{x}}{\partial \mathbf{z}} \mathbf{A}^T \tag{6.1.9} \\
\end{eqnarray}
$$

####  6.2 Matrix Derivatives

$$
\begin{eqnarray}
\frac{\partial \mathbf{X}}{\partial X_{ij}} &=& \mathbf{\Delta}(\mathbf{X})_{ij} \tag{6.2.1}  \\
\frac{\partial \mathbf{a}^T\mathbf{X} \mathbf{a}}{\partial \mathbf{X}} &=& \frac{\partial \mathbf{a}^T\mathbf{X}^T \mathbf{a}}{\partial \mathbf{X}} = \mathbf{a}\mathbf{a}^T \tag{6.2.2} \\
\frac{\partial \mathbf{a}^T\mathbf{X} \mathbf{b}}{\partial \mathbf{X}} &=& \mathbf{a}\mathbf{b}^T \tag{6.2.3} \\
\frac{\partial \mathbf{a}^T\mathbf{X}^T \mathbf{b}}{\partial \mathbf{X}} &=& \mathbf{b}\mathbf{a}^T \tag{6.2.4} \\
\frac{\partial \mathbf{X}^T \mathbf{BX}}{\partial \mathbf{X}} &=& (\mathbf{B} + \mathbf{B}^T)\mathbf{X} \tag{6.2.5}  \\

\end{eqnarray}
$$


# References
- [The Matrix Cookbook, Peterson & Pederson (2012)](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)



