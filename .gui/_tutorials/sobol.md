# 🧮 Sobol Sequences for Hyperparameter Search

<!-- What are SOBOL sequences? -->

<!-- Category: Models -->

<div id="toc"></div>

## Introduction
When you set out to optimize hyperparameters of a model, you face a search space: often multi-dimensional, large, and hostile.
A naïve grid search wastes time covering redundant regions; pure random search leaves holes and clumps.

Enter the **Sobol sequence** — a quasi-random, low-discrepancy sequence that strives to **cover** the hyperparameter space evenly.

## What is a Sobol sequence?
- A Sobol sequence is a **quasi-random** (also called “low‐discrepancy”) sequence of points in the unit hypercube \([0,1]^s\) (for \(s\) dimensions) that fills the space more evenly than random points.
- “Low discrepancy” means the largest gap between covered and uncovered regions is much smaller than in purely random sampling.
- Introduced by [Ilya M. Sobol](https://en.wikipedia.org/wiki/Ilya_M._Sobol%27) in 1967, these sequences have become fundamental in quasi–Monte Carlo methods.

## Why is it helpful for hyperparameter search?
When choosing a certain number of initial hyperparameter settings, random sampling often clusters or misses regions entirely.
A Sobol sequence achieves **better coverage** with the same number of samples, increasing your chance of finding promising areas early.

This makes it ideal for the *initial exploration* phase, before you refine results using models like [BoTorch or other surrogate models](tutorials?tutorial=models).

## How Sobol sequences work

### Direction numbers and binary construction

- For each dimension \(j\) (from \(1\) to \(s\)), select a primitive polynomial over [\(GF(2)\)](https://en.wikipedia.org/wiki/Finite_field) of degree \(s_j\).
- Define “direction numbers” \(v_{k,j}\) as binary fractions: $v_{k,j} = \frac{m_{k,j}}{2^k}$ where \(m_{k,j}\) are odd integers less than \(2^k\).
- Represent the integer index \(i\) in binary: \(i = ( \dots i_3\,i_2\,i_1 )_2\). The \(j\)-th coordinate of the \(i\)-th point is: \(x_{i,j} = i_1 v_{1,j} \oplus i_2 v_{2,j} \oplus i_3 v_{3,j} \oplus \cdots\),  where \(\oplus\) is the bitwise XOR operation.
- The resulting vector \((x_{i,1}, x_{i,2}, \dots, x_{i,s})\) is the \(i\)-th Sobol point.
- The sequence satisfies properties of a \((t, s)\)-sequence in base 2 — ensuring uniform space coverage.

### Compact summary

$$x_{i,j} = \bigoplus_{k=1}^{\infty} i_k\,v_{k,j}$$

where \(i_k\) is the \(k\)-th bit of \(i\), and the direction numbers \(v_{k,j}\) are precomputed constants.

## Summary

Sobol sequences provide a **deterministic, low-discrepancy** way to explore hyperparameter spaces.
They outperform pure random search in uniformity and efficiency, making them ideal for *initial sampling* before model-based refinement.

<div class="caveat warning">
- Sobol sequences do **not guarantee** global optima — they’re best for exploration, not exploitation.
- High-dimensional spaces still suffer from the curse of dimensionality.
</div>
