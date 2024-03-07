---
title: 'Misnomer Alert: Dot Product and Inner Product are not the Same'
date: 2023-09-19
permalink: /posts/2023/09/inner-product-vs-dot-product/
excerpt: "Have you encountered cases where inner product and dot product are used synonymously? If you have, be alert that inner product and dot product are not same."
tags:
  - Machine Learning
  - Mathematics
  - Linear Algebra
  - Dot Product
  - Inner Product
---

Let's jump in by calculating the inner and dot product of two example vectors using NumPy:


```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
```


```python
np.inner(x, y)
```




    32




```python
np.dot(x, y)
```




    32



From the both `np.inner` and `np.dot` function, we got the same result. When popular libraries evaluates both function to same value, it is tempting to think of them as same. Not only that, many books and experts, particularly in machine learning, use the term "inner product" and "dot product" interchangeably, assuming they refer to the same mathematical operation. I also used them in similar fashion until I read the book "Mathematics for Machine Learning" by M. P. Deisenroth, A. A. Faisal, and C. S. Ong. In this blog post, I aim to clear up this misconception and shed light on the differences between these fundamental product in linear algebra.

## Dot Product

Why not start with the definition of dot product, so called "inner product", most of us know. The dot product of vectors $$\mathbf{x}, \mathbf{y} \epsilon \mathbb{R}^{n}$$ is defined as:

$$\mathbf{x}\cdotp\mathbf{y} = \mathbf{x}^{\top}\mathbf{y}=\sum_{i=1}^{n}x_iy_i$$

where $$x_i$$ and $$y_i$$ are the $$i$$th elements of vectors $$\mathbf{x}$$ and $$\mathbf{y}$$ respectively. This is the dot product we use to calculate length of a vector and angle or similarity between two vectors. The NumPy functions, `np.dot` and `np.inner` did the same operation in above example. However, inner product is much more than this definition of dot product. Spoiler alert, the dot product is a specific instance of inner product. We'll see how shortly.

## Inner Product

Please bear with me while I give you the formal(mathematical) definition of the inner product using the fundamental concepts of linear algebra. Let's start with the <b>linear mapping</b>:

Consider two vector spaces $$V$$ and $$W$$, a mapping $$Φ:V\rightarrow W$$ is called linear mapping if

$$\forall \mathbf{x}, \mathbf{y} \epsilon V \forall\lambda,\varphi \epsilon \mathbb{R}: \Phi(\lambda\mathbf{x} + \varphi\mathbf{y})=\lambda\Phi(\mathbf{x})+\varphi\Phi(\mathbf{y})$$

In simple and intuitive terms, a tranformation function between two vector spaces that preserves the origin, collinearity and parallelism is called linear mapping.

Let's expand the linear mapping to two arguments function called <b>bilinear mapping</b>:

Consider two vector spaces $$V$$ and $$W$$, a mapping with two arguments $$Ω: V\times V\rightarrow W$$ is called bilinear mapping if $$\forall \mathbf{x}, \mathbf{y}, \mathbf{z} \epsilon V \forall\lambda,\varphi \epsilon \mathbb{R}:$$

$$\Omega(λ\mathbf{x}+\varphi\mathbf{y}, \mathbf{z})=\lambda\Omega(\mathbf{x}, \mathbf{z})+\varphi\Omega(\mathbf{y}, \mathbf{z})$$

$$\Omega(\mathbf{x}, λ\mathbf{y}+\varphi\mathbf{z})=\lambda\Omega(\mathbf{x}, \mathbf{y})+\varphi\Omega(\mathbf{x}, \mathbf{z})$$

With these two definitions in hand, we can define the <b>inner product</b> formally as:

Consider a vector space $$V$$, a bilinear mapping $$\Omega:V\times V\rightarrow \mathbb{R}$$ is called inner product on $$V$$ if it satisfies the following two conditions:
* $$\Omega$$ is symmetric i.e., $$\forall\mathbf{x},\mathbf{y}\epsilon V:\Omega(\mathbf{x},\mathbf{y})=\Omega(\mathbf{y},\mathbf{x})$$
* $$\Omega$$ is positive definite i.e., $$\forall\mathbf{x}\epsilon V-\{\mathbf{0}\}: \Omega(\mathbf{x}, \mathbf{x})>0$$ and $$\Omega(\mathbf{x}, \mathbf{x})=0$$ if and only if $$\mathbf{x}=0$$

From the definition, we can simply say that inner product can be any functions that takes two vectors as arguments, outputs a real number, is symmetric meaning we can interchange the arguments and evaluates to positive real number when both arguments are same.

## Relationship Between Dot Product and Inner Product

To find the relationship between dot product and inner product, let's expand the definition of inner product in terms of basis of the vector space and coordinate representation of vectors.

Consider $$B=(\mathbf{b}_1, \dots, \mathbf{b}_n)$$ be the basis of the $$n$$-dimensional vector space $$V$$. Also, let's consider two vectors $$\mathbf{x}, \mathbf{y}\epsilon V$$ in terms coordinate vectors $$\hat{\mathbf{x}}, \hat{\mathbf{y}}$$ respectively with respect to the basis $$B$$.

$$\mathbf{x}=\sum_{i=1}^{n}\varphi_i\mathbf{b}_i, \space \mathbf{y}=\sum_{j=1}^{n}\lambda_j\mathbf{b}_j$$

$$\hat{\mathbf{x}}=\begin{bmatrix}\varphi_1\\\vdots\\\varphi_n\end{bmatrix}, \space \hat{\mathbf{y}}=\begin{bmatrix}\lambda_1\\\vdots\\\lambda_n\end{bmatrix}$$

where $$\varphi_i$$ and $$\lambda_j$$ is the $$i$$th and $$j$$th elements of the coordinate representation/coordinate vectors $$\hat{\mathbf{x}}$$ and $$\hat{\mathbf{y}}$$ respectively.

Now, we can write the inner product $$\Omega$$, typically represented as $$\left \langle \cdot, \cdot \right \rangle$$, as the following:

$$\left \langle \mathbf{x}, \mathbf{y} \right \rangle=\left \langle \sum_{i=1}^{n}\varphi_i\mathbf{b}_i, \sum_{j=1}^{n}\lambda_j\mathbf{b}_j \right \rangle$$

Using the property of bilinearity defined above:

$$\left \langle \mathbf{x}, \mathbf{y} \right \rangle=\sum_{i=1}^{n}\sum_{j=1}^{n}\varphi_i\left \langle \mathbf{b}_i, \mathbf{b}_j \right \rangle\lambda_j=\hat{\mathbf{x}}^\top\mathbf{A}\hat{\mathbf{y}}$$

where $$A_{ij}=\left \langle \mathbf{b}_i, \mathbf{b}_j \right \rangle$$ i.e., the inner product of the basis $$\mathbf{b}_i$$ and $$\mathbf{b}_j$$.

From here, we can see that we can define inner product using a square matrix $$\mathbf{A}\epsilon\mathbb{R}^{n\times n}$$ that is symmetric and positive definite.

Now, we can finally relate dot product with the inner product. When the symmetric and positive definite matrix that governed the inner product is an identity matrix, the inner product can be called as dot product. Formally, the <b>dot product</b> on $$n$$-dimensional vector space $$V$$ is defined as a bilinear mapping $$\Omega: V\times V\rightarrow \mathbb{R}$$ where:

$$\forall \mathbf{x},\mathbf{y}\epsilon V: \Omega(\mathbf{x},\mathbf{y})=\mathbf{x}^\top\mathbf{I}_n\mathbf{y}=\mathbf{x}^\top\mathbf{y}$$

where $$\mathbf{I}_n$$ is an $$n\times n$$ identity matrix.

## Programmatic Implementation

Let's implement a full-fledged inner product using NumPy:


```python
def inner_product(x, y, A):
    """Returns inner product of x and y governed by matrix A
    Arguments:
    x = 1-D array of shape (n,)
    y = 1-D array of shape (n,)
    A = 2-D array of shape (n, n)
    """
    x_vec, y_vec = x[:, np.newaxis], y[:, np.newaxis]
    xTA = np.matmul(x_vec.T, A)
    in_prod = np.matmul(xTA, y_vec)
    return in_prod
```

Define example vectors and matrices:


```python
x = np.array([1, 2])
y = np.array([3, 4])

# An example symmetric positive definite matrix
A = np.array(
    [[9, 6],
     [6, 5]]
)

# An indentity matrix
I = np.array(
    [[1, 0],
     [0, 1]]
)
```

Calculate inner product governed by A


```python
inner_product(x, y, A)
```




    array([[127]])



Calculate inner product governed by I. We'll get the dot product.


```python
inner_product(x, y, I)
```




    array([[11]])



Let's verify with the numpy implementation.


```python
np.dot(x, y)
```




    11

You've come to the end of the blog post. This post has illuminated the distinction and relationship between the inner product and the dot product in linear algebra. While most of us use them interchangeably, it's crucial to understand that the inner product is a broader concept encompassing a bilinear mapping governed by a symmetric, positive definite matrix. On the other hand, the dot product is a specific instance of the inner product when the governing matrix is the identity matrix.