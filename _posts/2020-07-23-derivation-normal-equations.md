---
layout: post
title: "Derivation: Ordinary Least Squares Solution and the Normal Equations"
author: Dustin Stansbury
date: 2020-07-23
tags: ordinary-least-squares derivation normal-equations
permalink: /derivation-normal-equations
---

In the linear regression framework, we model an output variable $$y$$ (in this case a scalar) as a linear combination of some independent input variables $$X$$ plus some independent noise $$\epsilon$$. The linear combination of the independent variables is defined by a parameter vector $$\beta$$:

$$
y = X \beta + \epsilon
$$

We also assume that the noise term $$\epsilon$$ is drawn from a Normal distribution with zero mean and a noise variance $$\sigma_{\epsilon}^2$$ (generally assumed to be equal to one):

$$
\epsilon \sim N(0,\sigma_{\epsilon}^2)
$$

For some estimate of the model parameters $$\hat \beta$$, the model’s prediction errors (a.k.a. *residuals*) $$e$$ are the difference between the model prediction and the observed ouput values:

$$
e = y - X\hat \beta
$$

The [Ordinary Least Squares (OLS) solution](http://en.wikipedia.org/wiki/Ordinary_least_squares) to the problem--i.e. determining an optimal solution for $$\hat \beta$$--requires minimizing the sum of the squared errors with respect to the model parameters $$\hat \beta$$. It turns out, the sum of squared errors [is equal to the inner product of the residuals vector with itself](https://en.wikipedia.org/wiki/Dot_product) $$\sum_i e_i^2 = e^Te$$ :

$$
\begin{align}
 e^T e &= (y - X \hat \beta)^T (y - X \hat \beta) \\  
 &= y^Ty - y^T (X \hat \beta) - (X \hat \beta)^T y + (X \hat \beta)^T (X \hat \beta) \\
 &= y^Ty - (X \hat \beta)^T y - (X \hat \beta)^T y + (X \hat \beta)^T (X \hat \beta) \\
 &= y^Ty - 2(X \hat \beta)^T y + (X \hat \beta)^T (X \hat \beta) \\
 &= y^Ty - 2\hat \beta^T X^T y + \hat \beta^T X^T X \hat \beta \text{,} \tag{1}
\end{align}
$$

where we take advantage of the matrix identity $$(AB)^T = B^TA^T$$ in steps 2-3 above. 

To determine the parameters $$\hat \beta$$ we minimize the sum of squared errors with respect to the parameters:

$$
\begin{align}
\frac{\partial}{\partial \beta} \left[ e^T e \right] &= 0 \\
\frac{\partial}{\partial \beta}  \left[ y^Ty - 2\hat \beta^T X^T y + \hat \beta^T X^T X \hat \beta \right ] &= 0 \;\; \text{, via Eq. (1)}\\
-2X^Ty + 2X^TX \hat \beta &= 0 \\ 
-X^Ty + X^TX \hat \beta &= 0 \\ 
X^TX \hat \beta&= X^Ty  \text{,} \tag{2}

\end{align}
$$

where we note to the matrix derivative identity $$\frac{\partial \mathbf{a}^T \mathbf{b}}{\partial \mathbf{a}} = \mathbf{b}$$, for vectors $$\mathbf{a}$$ and $$\mathbf{b}$$ in step 2-3 above.

The relationship in ***Equation 2*** is the matrix form of what are known as the [Normal Equations](https://mathworld.wolfram.com/NormalEquation.html). Solving for $$\hat \beta$$ gives the analytical solution to the Ordinary Least Squares problem.

$$
\hat \beta = (X^TX)^{-1}X^Ty
$$

Yay!

---
---
# Notes
This post is a refactor of content with the same title originally posted on [The Clever Machine Wordpress blog](https://theclevermachine.wordpress.com/2012/09/01/derivation-of-ols-normal-equations/).