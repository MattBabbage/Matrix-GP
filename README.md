## Matrix GP - Genetic Programming for Matrix Multiplication

**The Experiment**
Create a Genetic Programming Algorithm to find methods of matrix multiplication.

**Context**
Fast matrix multiplication is partially an unsolved problem, given our current computers.
Multiplication is the easiest way to multiple matrices, as you might expect, but for our computers its faster to add and subtract.
So it turns out you can subtract multiplcations to make it faster! It began with Strassens algorithm.

**How is success measured?**
The number or multiplications is called ω (Omega), Essentially measuring speed of algorithm, as broadly speaking number of +/- is negligble in comparison.

So for a 2x2 Matrix, the general solution has 2^3 (8) multiplications. Where the exponent of the number or rows in a square matrix is ω.

**Project Goals**
For a 2x2 Matrices:
- Find Basic matrix multiplication algorithm (ω = 3)
- Find Strassens Algorithm (ω = 2.8074) 
