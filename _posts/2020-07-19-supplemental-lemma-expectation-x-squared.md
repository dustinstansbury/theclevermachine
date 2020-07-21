---
layout: post
title: "Supplemental Proof: The Expected Value of a Squared Random Variable"
author: Dustin Stansbury
date: 2020-07-19
tags: statistics derivation expected-value
permalink: /supplemental-lemma-expectation-x-squared
---

We want to show the following relationship:

$$
\mathbb E[X^2] = \mathbb E[(X - \mathbb E[X])^2] + \mathbb E[X]^2 \tag{1}
$$

If we expand the first expression on the right-hand side of ***Equation 1***:

$$
\begin{align}
\mathbb E [(X - \mathbb E[X])^2] &= \mathbb E[X^2-2X\mathbb E[X]+\mathbb E [X]^2] \\
&= \mathbb E[X^2] - 2\mathbb E[X] \mathbb E [\mathbb E [X]] + \mathbb E [\mathbb E [X]^2]. \tag{2}
\end{align}
$$

If we note that $$\mathbb E[X]$$ is a constant, then $$\mathbb E[\mathbb E[X]]$$ is also a constant, namely $$\mathbb E[X]$$. Taking this into account ***Equation 2*** simplifies to:

$$
\begin{align}
\mathbb E[(X - \mathbb E[X])^2] &= \mathbb E[X^2] - 2\mathbb E[X] \mathbb E[X] + \mathbb E[X]^2 \\
&= \mathbb E[X^2] - 2\mathbb E[X]^2 + \mathbb E[X]^2 \\
&= \mathbb E[X^2] - \mathbb E[X]^2 \tag{3}
\end{align}
$$

Plugging ***Equation 3*** back into the right-hand side of ***Equation 1*** gives

$$
\begin{align}
\mathbb E[(X - \mathbb E[X])^2] + \mathbb E[X]^2 &= \mathbb E[X^2] - \mathbb E[X]^2 + \mathbb E[X]^2 \\
&= \mathbb E[X^2] , \tag{4}
\end{align}
$$

thus giving the desired result.