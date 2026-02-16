+++
date = '2026-02-15'
title = 'BasicsByHand: Square Roots'
tags = ['mathematics', 'square-roots', 'by-hand']
+++

# Premise

Long ago in middle school (or even earlier), along with the concept of what a
square root is, I was also taught a method to calculate them by hand. This
method did not involve Taylor series expansions, nor was it the popular
numeric Newton-Raphson method - we didn't have a lick of knowledge of Calculus
to use them anyway.

It worked, and nobody ever quite taught *why* it worked. Given that I'm an adult
now and free to do adult things, I wanted to cross this off my (admittedly
non-existent) list as a part of something else I was trying to understand.

# What method?

This is for my readers who may not be familiar with this method, or just need a
refresher. Suppose you need to calculate the square root of 78129.329 while
you're stuck on an island. Maybe your captor is a math fanatic. This is how you
would go about it.

Start with grouping the digits in pairs - starting from the decimal point,
moving on either side of it:

```
       ┌─────────────────
       │  7,81,29.32,9
```

Now, we begin a method that suspiciously looks a lot like long-form division.
Find the largest number whose square is equal to, or less than the left-most
group.

```
           2
        ┌─────────────────
      2 │  7,81,29.32,9
```

Then, you subtract its square from the digit:

```
           2
        ┌─────────────────
      2 │  7,81,29.32,9
        │ -4
        │─────────────────
        │  3
```

and multiply the number at the top with 2 (how arbitrary! or is it...?) and call
it the new *divisor* you're going to work with. The next task is to find a value
for a **digit** $x$ such that $x \times (40+x)$ is less than or equal to $381$:

```
           2  x
        ┌─────────────────
      2 │  7,81,29.32,9
        │ -4
        │─────────────────
     4x │  3,81
```

This will require some manual multiplication (since we're doing this by hand!).
Looks like $7$ meets this requirement (as $48 \times 8 = 384$). Apply the same
subtraction:

```
           2  7
        ┌─────────────────
      2 │  7,81,29.32,9
        │ -4
        │─────────────────
     47 │  3,81
        │ -3,29
        │─────────────────
        │   52
```

and repeat the process once more. We need to first multiply $27$ by $2$, which
gives us $54$. Similarly, $x \times (540+x)$ should then be less than or equal
to $52,29$:

```
           2  7  x
        ┌─────────────────
      2 │  7,81,29.32,9
        │ -4
        │─────────────────
     47 │  3,81
        │ -3,29
        │─────────────────
    54x │    52,29
```

I'll forgive you if you use the calculator here! I'll skip a step here (we end
up at $9$ as $549\times 9=49,41<52,29$) to show that we need to now find $x$
such that $x \times (5580 + x)$ is less than or equal to $2,88,32$:

```
           2  7  9. x
        ┌─────────────────
      2 │  7,81,29.32,9
        │ -4
        │─────────────────
     47 │  3,81
        │ -3,29
        │─────────────────
    549 │    52,29
        │   -49,41
        │─────────────────
   558x │     2,88,32        
```

We settle on 5. Note here that the next number is $2795\times2$, not
$279.5\times2$, a small subtlety here. The next $x$ is going to be $1$:

```
           2  7  9. 5 x
        ┌─────────────────
      2 │  7,81,29.32,9
        │ -4
        │─────────────────
     47 │  3,81
        │ -3,29
        │─────────────────
    549 │    52,29
        │   -49,41
        │─────────────────
   5585 │     2,88,32     
        │    -2,79,25
        │─────────────────
  5590x │        9,07,90
```

and we get:

```
           2  7  9. 5 1
        ┌─────────────────
      2 │  7,81,29.32,9
        │ -4
        │─────────────────
     47 │  3,81
        │ -3,29
        │─────────────────
    549 │    52,29
        │   -49,41
        │─────────────────
   5585 │     2,88,32     
        │    -2,79,25
        │─────────────────
  55901 │        9,07,90
        │      - 5,59,01
        │─────────────────
        │        3,48,89          
```

We could stop here, or go on further. I elect to stop here, and verify my
result. Let's go to trusty Python:

```python
>>> from math import sqrt
>>> print(sqrt(7_81_29.32_9))
279.51624103082094
```

Excellent, so we were right!

But... why? How did we get it right?

# Building the Intuition


## Getting our foot in the door

Let's start with our previous example.

$$
7,81,29.32,9
$$

Note that commas intentionally retained every two digits instead of the usual
three.

> Why are commands retained every two digits?

> This is because squares of numbers have, up to twice the number of digits of
  the original number. Anecdotally, consider $999^2$. It will be less than
  $1000^2$, which is $1,000,000$ and has 7 digits (and 7 is less than 4, the
  number of digits that $1000$) has.

> On the other hand, smaller numbers like $4$ go to $16$ ($1$ digit to $2$
  digits). $1$ to $3$ remain at $1$ digit (1, 4, 9).

Remember that we usually also write our numbers in base $10$. Since any power
of $10$ is easy to calculate, this makes the **estimate of our first digit**
rather easy. Let's start by finding the closest square of $10$ that is less
than or equal to $7,81,29.329$.

$$
\begin{array}{lrll}
&1,00,00 &\leq 7,81,29.329 &\leq 10,00,00\\
\Rightarrow &10^4 &\leq 7,81,29.32,9 &\leq 10^5\\
\Rightarrow &(10^2)^2 &\leq 7,81,29.32,9 &\leq 10^5\\
\end{array}
$$

Why did we write this as $(10^2)^2$? This tells us that our desired square root
is greater than $100$. What's the closest multiple of hundred that is less than 
or equal to $7,81,29.32,9$? It's 200, because

$$
\begin{array}{rlr}
200^2 &= 4,00,00 &< 7,81,29.32,9\\
300^2 &= 9,00,00 &> 7,81,29.32,9
\end{array}
$$

In other words,

$$
\begin{array}{lrll}
&200^2 &\leq 7,81,29.32,9 &\leq 300^2\\
\end{array}
$$

When we calculated that $200$ was the closest multiple of $100$ that was less than
or equal to $7,81,29.32,9$, we were calculating the first digit of the square root.

## We continue by applying a common formula...

...which is derived below to minimize cognitive load, in case it's been a long
time since you've touched algebra.

$$
\begin{aligned}
(a+x)^2 &=(a+x)(a+x)\\
        &=a(a+x) + x(a+x)\\
        &=a^2 + ax + xa + x^2\\
        &=a^2+2ax+x^2
\end{aligned}
$$

## Connecting the dots

Let's consider the next digit. Because the true root lies between $200$ and $300$,
we can say that the next digit of the root will be the closest multiple of $10$ that
is greater than $200$ and whose square is less than $7,81,29.32,9$.

That means we're trying to find a single **digit** $x$ that meets the criteria:

$$
\begin{array}{rll}
&(200+10x)^2 &\leq 7,81,29.32,9\\
\Rightarrow &200^2 + 2\cdot200\cdot10x + 100x^2 &\leq 7,81,29.32,9\\
\Rightarrow &200^2 + 100\cdot\boldsymbol{(40+x)x} &\leq 7,81,29.32,9\\
\Rightarrow &100\cdot\boldsymbol{(40+x)x} &\leq 3,81,29.32,9\\
\Rightarrow &\boldsymbol{(40+x)x} &\leq \boldsymbol{3,81}.29,32,9
\end{array}
$$

Look closely at the bolded parts of the last equation:

$$
\boldsymbol{(40+x)x} \leq \boldsymbol{3,81}
$$

This is **exactly what we did here**:

```
           2  x
        ┌─────────────────
      2 │  7,81,29.32,9
        │ -4
        │─────────────────
     4x │  3,81
```

The single **digit** $x$ that satisfies this equation is $7$, as we had obtained earlier.

## One more step for clearer understanding

Let's do one more round to inscribe the logic in our heads:

We know that $270$ is the next best approximation of the root of $7,81,29.329$. We also
know that to get the next digit, we need the closest multiple of $1$ that is between
$270$ and $280$ whose square is less than $7,81,29.32,9$.

Once again, in trying to find the single **digit** $x$, the expressions become (note that
in this iteration, $x$ is just $x$, not $10x$, because it's a multiple of $1$, not $10$):

$$
\begin{array}{rll}
&(270+x)^2 &\leq 7,81,29.32,9\\
\Rightarrow &270^2 + 2\cdot270\cdot x + x^2 &\leq 7,81,29.32,9\\
\Rightarrow &270^2 + \boldsymbol{(540+x)x} &\leq 7,81,29.32,9\\
\Rightarrow &\boldsymbol{(540+x)x} &\leq 52,29.32,9\\
\Rightarrow &\boldsymbol{(540+x)x} &\leq \boldsymbol{52,29}.32,9
\end{array}
$$

and the bolded parts of the expression are:

$$
\boldsymbol{(540+x)x} \leq \boldsymbol{52,29}
$$

once again, matching exactly to:

```
           2  7  x
        ┌─────────────────
      2 │  7,81,29.32,9
        │ -4
        │─────────────────
     47 │  3,81
        │ -3,29
        │─────────────────
    54x │    52,29
```

## Estimating to more decimal points

You also now understand why this process can continue on till as many
decimal points are needed.

The rest of the estimation, however, is, um... [*left as an exercise to the
reader*](https://en.wikipedia.org/wiki/Proof_by_intimidation) 😬.

# Conclusion

I hope you now understand that the so called "long-division" method is
just encoding the technique that I laid out above step-by-step, and is
not really doing division alone, it's extending the binomial identity
$(a+b)^2$ and refining it successively every iteration.

You will also notice that as we proceed further in calculating the root,
the smaller the error becomes, allowing us to ignore it when we see fit.

# References

The method laid out in the article is described in [NIST article](https://xlinux.nist.gov/dads/HTML/squareRoot.html).
However, the explanation of the method is my own, as I was not able to grasp how the explanation provided
in that link actually worked "all the way" to the end.
