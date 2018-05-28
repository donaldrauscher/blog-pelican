Title: 538 Riddler: Who Gets The $100 Bill?
Date: 2016-09-11
Tags: 538, fivethirtyeight, riddler, probability
Slug: dollar-riddler
Resources: katex

I modelled this week's [Riddler](http://fivethirtyeight.com/features/who-keeps-the-money-you-found-on-the-floor/) as an [absorbing Markov chain](https://en.wikipedia.org/wiki/Absorbing_Markov_chain).  This MC has 5 transient states (representing the dollar bill sitting in front of someone) and 5 absorbing states (which represent someone winning).  The transition probability matrix is the following:
<div class="equation" data-expr="
\begin{matrix}
 & 0 & \frac{1}{3} & 0 & 0 & \frac{1}{3} & \frac{1}{3} & 0 & 0 & 0 & 0 \\
 & \frac{1}{3} & 0 & \frac{1}{3} & 0 & 0 & 0 & \frac{1}{3} & 0 & 0 & 0 \\
 & 0 & \frac{1}{3} & 0 & \frac{1}{3} & 0 & 0 & 0 & \frac{1}{3} & 0 & 0 \\
 & 0 & 0 & \frac{1}{3} & 0 & \frac{1}{3} & 0 & 0 & 0 & \frac{1}{3} & 0 \\
 & \frac{1}{3} & 0 & 0 & \frac{1}{3} & 0 & 0 & 0 & 0 & 0 & \frac{1}{3} \\
 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{matrix}
"></div>

From this transition matrix we can calculate the absorbing probabilities using the following formula:
<div class="equation" data-expr="B = \left( I - Q \right)^{-1} * R"></div>
Q here represents the transient-to-transient transition matrix (top left 5x5 in above matrix), and R represents the transient-to-absorbing transition matrix (top right 5x5 in the above matrix).  Calculating this out, the probability of winning if the game starts at you, next to you, or two spots away from you is <span class="inline-equation" data-expr="\frac{5}{11} = 45.45\%"></span>, <span class="inline-equation" data-expr="\frac{2}{11} = 18.18\%"></span>, and <span class="inline-equation" data-expr="\frac{1}{11} = 9.09\%"></span> respectively.

``` R
# number of players
n <- 5

# build Q
Q <- matrix(0, n, n)
tri <- lower.tri(matrix(nrow=n,ncol=n))
tri.inner <- cbind(rbind(FALSE, lower.tri(matrix(nrow=n-1,ncol=n-1))), FALSE)
Q[tri] <- 1/3; Q[tri.inner] <- 0; Q[n,1] <- 1/3
Q[upper.tri(Q)] <- t(Q)[upper.tri(Q)]

# build R
R <- matrix(0, n, n)
diag(R) <- 1/3

# calculate fundamental matrix
N <- solve(diag(n) - Q)

# calculate absorbing probabilities
B <- N %*% R
print(B)
```
<pre>           [,1]       [,2]       [,3]       [,4]       [,5]
[1,] 0.45454545 0.18181818 0.09090909 0.09090909 0.18181818
[2,] 0.18181818 0.45454545 0.18181818 0.09090909 0.09090909
[3,] 0.09090909 0.18181818 0.45454545 0.18181818 0.09090909
[4,] 0.09090909 0.09090909 0.18181818 0.45454545 0.18181818
[5,] 0.18181818 0.09090909 0.09090909 0.18181818 0.45454545</pre>

The only issue with this approach is that it is tough to derive from it an expression for the general N case, unless you're unusually gifted at finding matrix inverses, which I am not.  A perhaps more intuitive approach is to set up a system of equations.  Because the problem is symmetrical, we need to solve for just 3 variables, the probabilities of winning with the bill 0, 1, and 2 people away.  And we can relate these probabilities to one another easily since each turn is independent.  Equations for the N=5, N=6, and N=7 cases:

<table style = 'width:100%;'>
<tr><td style = 'width:33%; vertical-align: top;'>
<div class="equation" data-expr="\begin{cases}
 & P_{2} = \frac{1}{3} P_{2} + \frac{1}{3} P_{1} \\[1em]
 & P_{1} = \frac{1}{3} P_{2} + \frac{1}{3} P_{0} \\[1em]
 & P_{0} + 2 P_{1} + 2 P_{2} = 1
\end{cases}"></div>
</td><td style = 'width:33%; vertical-align: top;'>
<div class="equation" data-expr="\begin{cases}
 & P_{3} = \frac{2}{3} P_{2} \\[1em]
 & P_{2} = \frac{1}{3} P_{3} + \frac{1}{3} P_{1} \\[1em]
 & P_{1} = \frac{1}{3} P_{2} + \frac{1}{3} P_{0} \\[1em]
 & P_{0} + 2 P_{1} + 2 P_{2} + P_{3} = 1
\end{cases}"></div>
</td><td style = 'width:33%; vertical-align: top;'>
<div class="equation" data-expr="\begin{cases}
 & P_{3} = \frac{1}{3} P_{3} + \frac{1}{3} P_{2} \\[1em]
 & P_{2} = \frac{1}{3} P_{3} + \frac{1}{3} P_{1} \\[1em]
 & P_{1} = \frac{1}{3} P_{2} + \frac{1}{3} P_{0} \\[1em]
 & P_{0} + 2 P_{1} + 2 P_{2} + 2 P_{3} = 1
\end{cases}"></div>
</td></tr>
</table>

Here are the tediously-derived solutions to the N=2 through N=10 cases:
<table class="pretty">
<tr><th>N</th><th><span class="inline-equation" data-expr="P_{0}"></span></th></tr>
<tr><td>2</td><td><span class="inline-equation" data-expr="\frac{3}{5} = 60\%"></span></td></tr>
<tr><td>3</td><td><span class="inline-equation" data-expr="\frac{2}{4} = 50\%"></span></td></tr>
<tr><td>4</td><td><span class="inline-equation" data-expr="\frac{7}{15} = 46.66667\%"></span></td></tr>
<tr><td>5</td><td><span class="inline-equation" data-expr="\frac{5}{11} = 45.45455\%"></span></td></tr>
<tr><td>6</td><td><span class="inline-equation" data-expr="\frac{18}{40} = 45\%"></span></td></tr>
<tr><td>7</td><td><span class="inline-equation" data-expr="\frac{13}{29} = 44.82759\%"></span></td></tr>
<tr><td>8</td><td><span class="inline-equation" data-expr="\frac{47}{105} = 44.7619\%"></span></td></tr>
<tr><td>9</td><td><span class="inline-equation" data-expr="\frac{34}{76} = 44.73684\%"></span></td></tr>
<tr><td>10</td><td><span class="inline-equation" data-expr="\frac{123}{275} = 44.72727\%"></span></td></tr>
</table>

We can use the Fibonnaci numbers to come up with general expressions for the odd and even cases:
<div class="equation" data-expr="P_{0} = \begin{cases}
 \frac{F_{N}}{F_{N-1}+F_{N+1}} & \text{if } N \text{ is odd} \\[1em]
 \frac{F_{N-1}+F_{N+1}}{2F_{N-2}+F_{N+3}} & \text{if } N \text{ is even}
\end{cases}"></div>

Using the simpler odd case and some knowledge about the Fibonnaci numbers, we can see what happens as N goes to infinity:
<div class="equation" data-expr="\lim_{x \to \infty } \frac{F_{x+1}}{F_{x}} = \frac{1+\sqrt{5}}{2}"></div>
<div class="equation" data-expr="\lim_{N \to \infty }\frac{F_{N}}{F_{N-1}+F_{N+1}} = \frac{\frac{F_{N}}{F_{N-1}}}{2+\frac{F_{N}}{F_{N-1}}} = \frac{1}{\sqrt{5}} \approx 44.7\%"></div>
