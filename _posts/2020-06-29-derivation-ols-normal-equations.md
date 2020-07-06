---
layout: post
title: "Derivation: Ordinary Least Squares Solution and Normal Equations"
author: Dustin Stansbury
date: 2020-06-29
tags: derivation ordinary-least-squares
permalink: /derivation-ols-normal-equations
---

In a linear regression framework, we assume some output variable $$y$$ is a linear combination of some independent input variables $$X$$ plus some independent noise $$\epsilon$$. The way the independent variables are combined is defined by a parameter vector $$\beta$$:

$$\begin{array}{rcl} y &=& X \beta + \epsilon \end{array}$$

We also assume that the noise term $$\epsilon$$ is drawn from a standard Normal distribution:

$$\begin{array}{rcl}\epsilon &\sim& N(0,I)\end{array}$$

For some estimate of the model parameters $$\hat \beta$$, the model’s prediction errors $$e$$ are the difference between the model prediction and the observed ouput values

$$\begin{array}{rcl} e = y - X\hat \beta \end{array}$$

The Ordinary Least Squares (OLS) solution to the problem (i.e. determining an optimal solution for $$\hat \beta$$) involves minimizing the sum of the squared errors with respect to the model parameters, $$\hat \beta$$. The sum of squared errors is equal to the inner product of the residuals vector with itself $$\sum e_i^2 = e^Te$$ :

$$
\begin{array}{rcl} e^T e &=& (y - X \hat \beta)^T (y - X \hat \beta) \\  &=& y^Ty - y^T (X \hat \beta) - (X \hat \beta)^T y + (X \hat \beta)^T (X \hat \beta) \\  &=& y^Ty - (X \hat \beta)^T y - (X \hat \beta)^T y + (X \hat \beta)^T (X \hat \beta) \\  &=& y^Ty - 2(X \hat \beta)^T y + (X \hat \beta)^T (X \hat \beta) \\  &=& y^Ty - 2\hat \beta^T X^T y + \hat \beta^T X^T X \hat \beta \\  \end{array}
$$

To determine the parameters, $$\hat \beta$$, we minimize the sum of squared residuals with respect to the parameters.

$$
\begin{array}{rcl}
\frac{\partial}{\partial \beta} \left[ e^T e \right] &=& 0 \\  
-2X^Ty + 2X^TX \hat \beta &=& 0 \text{, and thus} \\ 
X^TX \hat \beta   &=& X^Ty
\end{array}
$$

due to the identity $$\frac{\partial \mathbf{a}^T \mathbf{b}}{\partial \mathbf{a}} = \mathbf{b}$$, for vectors $$\mathbf{a}$$ and $$\mathbf{b}$$. This relationship is matrix form of the Normal Equations. Solving for $$\hat \beta$$ gives  the analytical solution to the Ordinary Least Squares problem.

$$
\begin{array}{rcl} \hat \beta &=& (X^TX)^{-1}X^Ty \end{array}
$$