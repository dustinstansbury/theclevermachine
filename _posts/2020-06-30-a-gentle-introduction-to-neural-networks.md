---
layout: post
title: A Gentle Introduction to Artificial Neural Networks
author: Dustin Stansbury
date: 2020-06-30
tags: neural-networks gradient-descent classification backpropagation
permalink: /a-gentle-introduction-to-neural-networks
---

# Introduction

Though many phenomena in the world can be well-modeled using basic linear regression or classification, there are also many interesting phenomena that are nonlinear in nature. In order to deal with nonlinear phenomena, there have been a diversity of nonlinear models developed.

For example parametric models assume that data follow some parameteric class of nonlinear function (e.g. polynomial, power, or exponential), then fine-tune the shape of the parametric function to fit observed data. However this approach is only helpful if data are fit nicely by the available catalog of parametric functions.

Another approach, kernel-based methods, transforms data non-linearly into an abstract space that measures distances between observations, then predicts new values or classes based on these distances. However, kernel methods generally involve constructing a kernel matrix that depends on the number of training observations and can thus be prohibitive for large data sets.

Another class of models, the ones that are the focus of this post, are artificial neural networks (ANNs). ANNs are nonlinear models motivated by the physiological architecture of the nervous system. They involve a cascade of simple nonlinear computations that, when aggregated, can implement robust and complex nonlinear functions. In fact, depending on how they are constructed, ANNs can approximate any nonlinear function, making them a quite powerful class of models[^1].

In recent years ANNs that use multiple stages of nonlinear computation (aka “deep learning”)  have been able obtain outstanding performance on an array of complex tasks ranging from visual object recognition to natural language processing. I find ANNs super interesting due to their computational power and their intersection with computational neuroscience.  However, I’ve found that most of the available tutorials on ANNs are either dense with formal details and contain little information about implementation or any examples, while others skip a lot of the mathematical detail and provide implementations that seem to come from thin air.  This post aims at giving a more complete overview of ANNs, including (varying degrees of) the math behind ANNs, how ANNs are implemented in code, and finally some toy examples that point out the strengths and weaknesses of ANNs.

# Single-layer Neural Networks

The simplest ANN (Figure 1) takes a set of observed inputs $$\mathbf{a}=(a_1, a_2, ..., a_N)$$, multiplies each of them by their own associated weight $$\mathbf{w} = (w_1, w_2, ...w_N)$$ , and sums the weighted values to form a pre-activation $$z$$.  Oftentimes there is also a bias $$b$$ that is tied to an input that is always +1 included in the preactivation calculation. The network then transforms the pre-activation using a nonlinear activation function $$g(z)$$ to output a final activation $$a_{\text{out}}$$.

---

<center>
    <br>
    <div id="container">
        <img width="320" src="assets/images/a-gentle-introduction-to-neural-networks/perceptron2.png">
    </div>
</center>


***Figure 1***: Diagram of a single-layered artificial neural network.

---
<b>

There are many options available for the form of the activation function g(z), and the choice generally depends on the task we would like the network to perform. For instance, if the activation function is the identity function:

$$ \begin{array}{rcl}g_{\text{linear}}(z) = z\end{array},$$

which outputs continuous values $$a_{linear}\in (-\infty, \infty)$$, then the network implements a linear model akin to used in standard linear regression. Another choice for the activation function is the logistic sigmoid:

$$ \begin{array}{rcl}g_{\text{logistic}}(z) = \frac{1}{1+e^{-z}}\end{array},$$

which outputs values $$a_{logistic} \in (0,1)$$. When the network outputs use the logistic sigmoid activation function, the network implements linear binary classification. Binary classification can also be implemented using the hyperbolic tangent function, $$\text{tanh}(z)$$, which outputs values $$a_{\text{tanh}}\in (-1, 1)$$ (note that the classes must also be coded as either -1 or 1 when using $$\text{tanh}$$. Single-layered neural networks used for classification are often referred to as “perceptrons,” a name given to them when they were first developed in the late 1950s.

---
<center>
    <br>
    <div id="container">
        <img width="680" src="assets/images/a-gentle-introduction-to-neural-networks/activation_functions.png">
    </div>
</center>


***Figure 2:*** Common activation functions functions used in artificial neural, along with their derivatives


<details >

```python
# % DEFINE A FEW COMMON ACTIVATION FUNCTIONS
# gLinear = inline('z','z');
# gSigmoid = inline('1./(1+exp(-z))','z');
# gTanh = inline('tanh(z)','z');

g_linear = lambda z: z
g_sigmoid = lambda z: 1./(1. + np.exp(-z))
g_tanh = lambda z: np.tanh(z)

 
# % ...DEFINE THEIR DERIVATIVES
# gPrimeLinear = inline('ones(size(z))','z');
# gPrimeSigmoid = inline('1./(1+exp(-z)).*(1-1./(1+exp(-z)))','z');
# gPrimeTanh = inline('1-tanh(z).^2','z');

g_prime_linear = lambda z: np.ones(len(z))
g_prime_sigmoid = lambda z: 1./(1 + np.exp(-z)) * (1 - 1./(1 + np.exp(-z)))
g_prime_tanh = lambda z: 1 - np.tanh(z) ** 2
 
# % VISUALIZE EACH g(z)
# z = linspace(-4,4,100);
# figure
# set(gcf,'Position',[100,100,960,420])
# subplot(121);  hold on;
# h(1) = plot(z,gLinear(z),'r','Linewidth',2);
# h(2) = plot(z,gSigmoid(z),'b','Linewidth',2);
# h(3) = plot(z,gTanh(z),'g','Linewidth',2);
# set(gca,'fontsize',16)
# xlabel('z')
# legend(h,{'g_{linear}(z)','g_{logistic}(z)','g_{tanh}(z)'},'Location','Southeast')
# title('Some Common Activation Functions')
# hold off, axis square, grid
# ylim([-1.1 1.1])
 
# % VISUALIZE EACH g'(z)
# subplot(122); hold on
# h(1) = plot(z,gPrimeLinear(z),'r','Linewidth',2);
# h(2) = plot(z,gPrimeSigmoid(z),'b','Linewidth',2);
# h(3) = plot(z,gPrimeTanh(z),'g','Linewidth',2);
# set(gca,'fontsize',16)
# xlabel('z')
# legend(h,{'g''_{linear}(z)','g''_{logistic}(z)','g''_{tanh}(z)'},'Location','South')
# title('Activation Function Derivatives')
# hold off, axis square, grid
# ylim([-.5 1.1])
```
</details> 

<b>

---
*NOTE This post is a refactor of content with the same title originally posted on [The Clever Machine](https://theclevermachine.wordpress.com/2014/09/11/a-gentle-introduction-to-artificial-neural-networks/) wordpress blog.*

[^1]: This property is not reserved for ANNs; kernel methods are also considered “universal approximators”; however, it turns out that neural networks with multiple layers are more efficient at approximating arbitrary functions than other methods. I refer the interested reader to [an in-depth discussion](http://yann.lecun.com/exdb/publis/pdf/bengio-lecun-07.pdf) on the topic.)