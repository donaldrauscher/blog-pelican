Title: 538 Riddler: 100-Sided Die
Date: 2017-01-14
Tags: 538, fivethirtyeight, riddler, probability
Slug: big-die
Resources: katex

This week's [Riddler](https://fivethirtyeight.com/features/how-long-will-it-take-to-blow-out-the-birthday-candles/) involves a game played with a 100-sided die (I seriously want one).  I started by thinking about the problem as an [absorbing Markov Chain](https://en.wikipedia.org/wiki/Absorbing_Markov_chain) with 101 states, 1 state representing the end of the game and 100 states for each potential previous roll. The transition matrix is the following:
<div class="equation" data-expr="
P = \begin{bmatrix}
 & \frac{1}{100} & \frac{1}{100} & \frac{1}{100} & \cdots & \frac{1}{100} & 0 & \\[0.8em]
 & 0 & \frac{1}{100} & \frac{1}{100} & \cdots & \frac{1}{100} & \frac{1}{100} & \\[0.8em]
 & 0 & 0 & \frac{1}{100} & \cdots & \frac{1}{100} & \frac{2}{100} & \\[0.8em]
 & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\[0.8em]
 & 0 & 0 & 0 & \cdots & \frac{1}{100} & \frac{99}{100} \\[0.8em]
 & 0 & 0 & 0 & \cdots & 0 & 1
\end{bmatrix}
"></div>

We break this transition matrix into three components: transient-to-transient (Q), transient-to-absorbing (R), and absorbing-to-absorbing (the identity matrix by definition).  Q here is simply the above matrix minus the last row and the last column (since we have just 1 absorbing state).  The expected number of rolls before being absorbed when starting at each transient state is the following vector:
<div class="equation" data-expr="t = \left( I - Q \right)^{-1} \mathbf{1}"></div>

The expected number of rolls for the game is simply the average of the values in this vector plus 1, since we're equally likely to start at any one of these initial rolls.  A little R code gives us the answer:

``` R
Q <- matrix(0, ncol=100, nrow=100)
Q[upper.tri(Q,diag=TRUE)] <- 1/100
N <- solve(diag(100) - Q)
t <- N %*% matrix(1, nrow=100, ncol=1)
mean(t)+1
```
```
[1] 2.731999
```

Though this gets us to the answer, it's tough to extend this approach to the general N case.  Let <span class="inline-equation" data-expr="E_{i}"></span> represent the expected number of rolls until the game ends given that the previous roll was <span class="inline-equation" data-expr="i"></span>. We can develop some recurrence relations starting with <span class="inline-equation" data-expr="E_{100}"></span> and working backwards.  Iterative substitution gives us an expression for <span class="inline-equation" data-expr="E_{i}"></span>:
<div class="equation" data-expr="\begin{aligned}
 E_{100} = & \frac{1}{100} E_{100} + 1 = \frac{100}{99} \\
 E_{99} = & \frac{1}{100} E_{99} + \frac{1}{100} E_{100} + 1 = \frac{1}{100} E_{99} + E_{100} = \left( \frac{100}{99} \right)^{2} \\
 E_{98} = & \frac{1}{100} E_{98} + \frac{1}{100} E_{99} + \frac{1}{100} E_{100} + 1 = \frac{1}{100} E_{98} + E_{99} = \left( \frac{100}{99} \right)^{3} \\
 \vdots \\
 E_{i} = & \left( \frac{100}{99} \right)^{100-i+1}
\end{aligned}"></div>

This is analagous to the vector <span class="inline-equation" data-expr="t"></span> that we computed above.  Thus, the average of <span class="inline-equation" data-expr="E_{1}"></span> through <span class="inline-equation" data-expr="E_{100}"></span> plus 1 gives us the expected number of rolls for the game.  And our answer here is consistent with the absorbing Markov Chain approach above.  We can also extend this logic to derive an expression for the N case.  Interestingly, as N goes to infinity, the expected number of rolls converges on e!
<div class="equation" data-expr="E = 1 + \frac{1}{100} \sum_{i=1}^{100} E_{i} = 1 + \frac{1}{100} \sum_{i=1}^{100} \left( \frac{100}{99} \right)^{i} = \left( \frac{100}{99} \right)^{100} = 2.731999"></div>
<div class="equation" data-expr="E(N) = \left( \frac{N}{N-1} \right)^{N} \rightarrow  \lim_{N \to \infty } E(N) = e = 2.718283"></div>
