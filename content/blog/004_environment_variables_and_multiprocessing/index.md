---
title: Environment Variables and Multiprocessing
author: Avinash Mallya
date: 2026-01-14
tags: [python, numpy, multiprocessing, parallel, environment, variables]
---

# Premise

I needed to use a codebase that had a mix of light `numpy` operations, coupled with a few heavy mathematical optimization problems that did not use `numpy`.
The API that I had access to provided to be too slow (even asynchronously), so I thought I'll run this locally on a faster system in parallel to speed
things up. Turns out, that was easier said than done, and I picked up the importance of environment variables along the way.

# In detail

## Demonstration code

Let's say that the "black box" code looked something like this.

```python
import numpy as np
import sys

n = 400
acc = 0.0
for _ in range(30):
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = a @ b
    d = np.linalg.inv(c + np.eye(n))
    acc += np.linalg.svd(d, compute_uv=False).sum()

result = acc + float(sys.argv[1])
```

The actual codebase was a full blown package that I wasn't familiar with, except for the following points:

1. The author(s) had not attempted to parallelize the codebase.
2. It used several packages, but most notably `numpy` and a few niche optimization tools.
3. The most important bit: `numpy` was used sparingly, and the compute intensive part was in the optimization tools.

## The problem

I needed to call this script multiple times, supplying different arguments each time. A single call doesn't take much time on my system:

```bash
> time python test.py 0

real    0m0.861s
user    0m26.207s
sys     0m0.120s
```

I could, of course, call it sequentially, but that wouldn't be very helpful:

```bash
> time for i in {1..10}; do python test.py $i; done

real    0m8.100s
user    4m6.980s
sys     0m1.193s
```

This isn't that bad - it looks like it took slightly less than the expected 8.6s, but it's no speedup.

What about switching to good ol' GNU `parallel`? I have 32 cores, so it should be fast, right?

```bash
> time seq 10 | parallel -j32 'python test.py {}'

real    16m50.053s
user    521m15.773s
sys     0m42.995s
```

Wow, that's actually around 125x **SLOWER**.

> An experienced programmer at this point would likely scream one of two words: CONTENTION! OVERSUBSCRIPTION!

Let's explore what that means.

## But why?

`numpy` is a very optimized package. It tries to use all cores available under the hood so that it can do what you asked
in the fastest possible time. This is a great idea when you have large matrices to operate on.

However, we called `numpy` via Python here 10 separate times. Each of those 10 processes were trying to access 32 threads
on the machine. 320 threads on 32 cores, which means that there were significantly more threads than cores, all actively
vying for those 32 cores!

The clearest signal of this is that a *naive attempt at parallelization via multiprocessing* being **SLOWER** than a
sequential set of calls to the same operation. That's why using a system with more cores and trying to parallelize blindly
doesn't always speed things up, and in the worst case, such as this, it considerably slows things down.

## The solution 

`numpy` uses different libraries for multithreading based on the system. Environment variables can be used to control the
number of threads each process creates, a summary of which is provided below.

| Variable |	Backend |
| :-: | :-: |
| `OMP_NUM_THREADS` |OpenMP (used by many BLAS)
| `OPENBLAS_NUM_THREADS` |OpenBLAS
| `MKL_NUM_THREADS` |Intel MKL
| `BLIS_NUM_THREADS` |BLIS
| `VECLIB_MAXIMUM_THREADS` |Apple Accelerate

To avoid oversubscription, we need to tell `numpy` to use fewer threads than the default (which is all threads), because
(1) we are aware that the process running is not compute intensive, and (2) we will handle parallelism ourselves.

On my system, I was using OpenMP, so all that I needed to do was:

```bash
> time seq 10 | parallel -j32 'OMP_NUM_THREADS=1 python test.py {}'

real    0m0.612s
user    0m4.905s
sys     0m0.236s
```

to make the slowest job among a set of 10 *run faster than single call to the script*!

> The real world impact in the actual codebase was 15x response time, and 30x throughput. What used to take 120 machines earlier took only 4 now!

## Ending notes

Why did this work here? Let's revisit the "most important" bit of information from the demonstration code:

> The most important bit: `numpy` was used sparingly, and the compute intensive part was in the optimization tools.

This technique of using environment variables to restrict the number of threads that a process can spawn, and then
running it in parallel (via `parallel` based multiprocessing) will work in any case where multiple processes are
spawned for each physical core, and you want to maximize throughput. In case the computations used by `numpy` are
more compute intensive, you could experiment with allowing more threads while still paralllelizing:

```bash
seq 10 | parallel -j8 'OMP_NUM_THREADS=4 python test.py {}'
```

This allows use of up to 4 threads, while spawning 8 processes in parallel, and in theory still using a total of 32 cores.
You may need to adjust this to find out what works best for your system.

In addition, to make this work less dependent on the libraries that `numpy` can use, just set more (or all) of the
environment variables:

```bash
seq 10 | \
parallel -j32 "OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python test.py {}"
```

An interesting side note is that the parallel, but single core run took 0.61 seconds, **less** than the time it took to run
a single call to the script on multiple cores (via threading in `numpy` internally). For large matrices, this also indicates
that the overhead associated with spawning 32 threads for the single task might actually be significant - and running on a
single core might just be inherently better!

## Footnotes

I've simplified a few things in this blog post:

1. This demo uses heavy `numpy` ops to clearly show the oversubscription effect. The actual codebase had much lighter
usage, but the principle is the same.
2. I've used *contention* and *oversubscription* interchangeably here. The former is a case of threads competing for
the same shared resource, and the latter is a higher thread count than CPU cores. *Oversubscription* here **led** to *contention*.
3. Modern systems have hundreds, or thousands of active threads for very few system cores. The difference with most of
these threads is that they often are "sleeping", and don't *vie* for attention like the `numpy` ones.
