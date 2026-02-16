+++
date = '2026-02-15'
title = 'BasicsByHand: Logarithms'
tags = ['mathematics', 'logarithms', 'history', 'by-hand', 'euler']
+++

# Premise

While reading a book on the
[history of $e$ and why it appears everywhere](https://press.princeton.edu/books/paperback/9780691168487/e-the-story-of-a-number),
the author focuses a little on how Napier first developed logarithms. While the
book clarifies that Napierian logarithms are different from the definitions we
use today, and demonstrates how they were calculated, it occured to me that I
had no idea how to derive their values by hand.

Searching online had limited results, and it took me quite some time to find
a version that described a simple method that didn't rely on calculus (something
that wasn't developed fully when logarithms were first calculated).

I write this down almost entirely from that source, primarily as a means to make
one more source available on the internet, and for my own curiosity.

# Prerequisites

You need to know very little to read this article.

However, if you truly wish to do it quite literally by hand, you will also need
to know how to calculate square roots by hand. I went down this (small) rabbit
hole myself while doing this research, and have [written down how (again, no
calculus involved)]({{< relref "series/byhand/byhand_001_square_roots.md" >}}).

# Logarithms and key properties

My "first principles" knowledge on Logarithms was a little rusty. I hope this
helps you the same way it did me. However, if you remember your logarithms well,
you should skip this section.

## What is a logarithm?

The logarithm of a number $y$ is defined for a base $b$ as the value $x$ that
satisfies the equation:

$$y=b^x$$

It is more popularly written as:

$$\log_b y=x$$

By definition, therefore,

$$\log_b b^x = x$$

and conversely

$$b^{\log_bx}=x$$

The converse is tricky, but becomes easier if you say $\log_b x=t$, which implies
that $b^t=x$, which is exactly what $b^{\log_b x}$ becomes.

### Restrictions on $b$ and $y$

It is important to note here that $\log_b y$ is defined only for $y>0$ and $b>0$
and $b \ne 1$. These restrictions are necessary for the reasons laid out below:

1. If $b < 0$, then $b^x$ is not always defined (for example, $(-2)^{0.5}$ is not real)
2. If $b = 0$, $0^x$ is 0 for positive $x$ and undefined for $x \leq 0$. It's not invertible.
3. If $b = 1$, $1^x = 1$ for all $x$, which doesn't give a useful $\log$ function.
4. If $y<0$ for valid $b$, then no real $x$ can satisfy $b^x>0$.
5. If $y=0$ for valid $b$, then no real $x$ can satisfy $b^x=0$.

## Properties of logarithms
### Sum of logs to the same base

A very basic property of exponents is that:

$$
\begin{aligned}
b^u\cdots b^v &= \underbrace{b\cdots b\ldots\cdots b}_\text{(u times)}\cdots \underbrace{b\cdots b\ldots\cdots b}_\text{(v times)}\\
&=b^{u+v}
\end{aligned}
$$

To extend this to logarithms, define $u=\log_b x$ and $v=\log_b y$:

$$
\begin{aligned}
b^{u+v} &= b^u\cdots b^v\\
b^{\log_b x+\log_b y}&=b^{\log_b x}\cdots b^{\log_b y}\\
b^{\log_b x+\log_b y}&=xy\\
b^{\log_b x+\log_b y}&=b^{\log_b{(xy)}}\\
\log_bx+\log_by&=\log_b(xy)
\end{aligned}
$$

It is important that all logs are to the same base $b$.

### Logs raised to an arbitrary exponent

This follows directly from sum of logs:

$$
\begin{aligned}
\log_bx^a&=\log_b(\underbrace{x\cdots x\cdots\ldots\cdots x}_{\text{a cdots}})\\
&=\underbrace{\log_bx+\log_b x+\ldots\log_bx}_{\text{a times}}\\
&=a\log_bx
\end{aligned}
$$

Or, when $a$ isn't positive or a whole number, we use the fact that:

$$
(b^x)^a=b^{xa}=b^{ax}
$$

which can be written as:

$$
\begin{aligned}
b^{\log_b(y^a)}&=y^a=(y)^a\\
&=(b^{\log_b y})^a\\
&=b^{a\log_b y}\\
\Rightarrow\log_b(y^a)&=a\log_b y
\end{aligned}
$$

### Difference of logs

Following the first two properties:
$$
\begin{aligned}
\log_bx-\log_by&=\log_bx+(-1)\log_by\\
&=\log_bx+\log_by^{-1}\\
&=\log_bx+\log_b\frac{1}{y}\\
&=\log_b\frac{x}{y}
\end{aligned}
$$

### Changing of base

This is an important one. We need to write:

$$
\log_xy
$$

in the form of just logs to a common base $b$. We can write:

$$
\begin{aligned}
x^\frac{\log_by}{\log_bx}&=(b^{\log_bx})^\frac{\log_by}{\log_bx}\\
&=b^{\frac{\log_bx\cdots\log_by}{\log_bx}}\\
&=b^{\log_by}\\
&=y\\
&=x^{\log_xy}\\
\Rightarrow \dfrac{\log_by}{\log_bx}&=\log_xy
\end{aligned}
$$

# The Euler Method

Let's say that we want to calculate $\log_7 155$. Euler's method involved first
changing the base:

$$
\log_7 155=\dfrac{\log_{10} 155}{\log_{10} 7}
$$

and then using the identity:

$$
\log(\sqrt{x\cdot y})=\dfrac{\log(xy)}{2}=\dfrac{\log x+\log y}{2}
$$

Let's see how. From herewith, if only $\log$ is used, assume that it means
$\log_{10}$. This will simplify a lot of the visual noise.

## Simplifying log calculations

We can first start by writing:

$$
\log155=\log(100\cdot1.55)=2+\log1.55
$$

## Calculating $\log1.55$

### Iteration 1

We know that from the identity above:

$$
\log(\sqrt{1\cdot10}) = \dfrac{\log1 +\log10}{2}=\dfrac{1}{2}=0.5
$$

In other words,

$$
\log{\sqrt{10}} = 0.5
$$

We know ([or can calculate by hand]({{< relref "series/byhand/byhand_001_square_roots.md" >}}))
that $\sqrt{10}\approx3.16228$. You can hopefully now see where this is going.

### Iteration 2

Where does $1.55$ lie? Between $1$ and $\sqrt{10}$, or between $1$ and $3.16228$.
Let's now calculate:

$$
\log{\sqrt{1\cdot3.16228}}=\dfrac{\log1+\log3.16228}{2}=\dfrac{0+0.5}{2}=0.25
$$

Thus, $\sqrt{1\cdot3.16228}$ is $\sqrt{3.16228}=1.77828$.

### Iteration 3

Once again, we make more progress by finding the square root of $1.77828$,
because $1.55$ lies between $1$ and $1.77828$. Therefore,

$$
\log{\sqrt{1\cdot1.77828}}=\dfrac{\log1+\log{1.77828}}{2}=0.125
$$

$\sqrt{1.77828}\approx1.33521$. We're getting rather close.

### Iteration 4

We have a small change here. $1.55$ now lies between $1.33521$ and $1.77828$.
This means that the new value we need to find is:

$$
\begin{aligned}
\log{\sqrt{1.33521\cdot1.77828}}&=\dfrac{\log{1.33521}+\log{1.77828}}{2}\\
&=\dfrac{0.125+0.25}{2}\\
&=0.1875
\end{aligned}
$$

Also, $\sqrt{1.33521\cdot1.77828}=1.54090$

### Further iterations

The next step would be to determine if $1.54090$ is sufficiently close to $1.55$
for us to approximate that $\log 1.54090\approx\log 1.55$, or we need to proceed
further.

Of course, we can continue to refine the values until we're sufficiently satisfied
of our accuracy. What does a calculator tell us about $\log 1.55$?

```python
>>> from math import log
>>> log(1.55, 10)
0.1903316981702915
```

So we were off by around... $0.0028$, which isn't bad at all, considering we did
only around 4 iterations (and many more iterations for calculating the square
root).

## Back to the Euler method

We find that $\log_{10}155=2+0.1875$. We calculate $\log_{10}7$ the same way,
which should come to $0.845$ (according to my calculator), so

$$
\log_7 155\approx\dfrac{2.1875}{0.845}=2.5
$$

and calculators give us:

```python
>>> from math import log
>>> log(155, 7)
2.591
```

Clearly, while this is close to the $2.5$ we got, it's not quite correct - we're
off by $0.091$, an order of magnitude more. If we'd used the true value, $0.1903$,
then we would have got:

$$
\log_7 155\approx\dfrac{2.1903}{0.845}\approx2.592
$$

Which is much closer to the true value. We find that approximating the true value
of the log to a much larger number of decimal points is crucial to get the correct
value.

## Takeaway

A key takeaway for a programmer would be that Euler managed to implement binary
search to efficiently (especially in terms of human effort) find the logarithm
of a number

# References

The primary reference for the material in this article is
[Bureau42](https://bureau42.com/view/7398/teaching-tidbit-calculating-logarithms-by-hand).
While the link works, it looks like the material that this used to point to doesn't
anymore. Some searching led me to a
[backup hosted on GitHub](https://vault.hanover.edu/~vaughnj/Mat%20112%20Calculus%20with%20Review/manual_logarithms.pdf)
and [another Hackernews linked source](http://eulerarchive.maa.org/hedi/HEDI-2005-07.pdf).
