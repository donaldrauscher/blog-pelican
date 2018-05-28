Title: 538 Riddler: A Variation of the Drunkard's Walk ... In a Bar
Date: 2016-07-14
Tags: 538, fivethirtyeight, riddler, probability
Slug: bar-riddler
Resources: katex

This week’s [Riddler](http://fivethirtyeight.com/features/how-long-will-you-be-stuck-playing-this-bar-game/) is a variation on the well-known OR problem, [the drunkard’s walk](https://en.wikipedia.org/wiki/Random_walk).  We can model this problem as an absorbing Markov Chain with X+Y+1 states.  The transition probability matrix is the following:
<div class="equation" data-expr="
\begin{matrix}
 & 1 & 0 & 0 & 0 & 0 & \cdots & 0 & \\
 & 0.5 & 0 & 0.5 & 0 & 0 & \cdots & 0 & \\
 & 0 & 0.5 & 0 & 0.5 & 0 & \cdots & 0 & \\
 & 0 & 0 & 0.5 & 0 & 0.5 & \cdots & 0 & \\
 & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
 & 0 & 0 & 0 & 0 & 0 & \cdots & 1
\end{matrix}
"></div>

Once we compute the fundamental matrix (N), calculating the expected number of steps until absorption is fairly [easy](https://en.wikipedia.org/wiki/Absorbing_Markov_chain).  Plugging in a few different values of X and Y, a trend emerges: the expected number of coin flips is simply <span style="font-weight: bold;">XY</span>.  Beautiful.

``` R
X <- 4
Y <- 13

# build Q
n_transient_state <- X + Y - 1
Q <- matrix(0.5, n_transient_state, n_transient_state)
tri <- cbind(rbind(FALSE, lower.tri(matrix(nrow=n_transient_state-1,ncol=n_transient_state-1))), FALSE)
Q[tri] <- 0; Q[t(tri)] <- 0; diag(Q) <- 0

# build R
R <- matrix(data = 0, nrow = n_transient_state, ncol = 2)
R[1,1] <- 0.5
R[n_transient_state,2] <- 0.5

# put together into P
P <- rbind(cbind(Q, R), cbind(matrix(0, 2, n_transient_state), diag(2)))

# fundamental matrix
N <- solve(diag(n_transient_state) - Q)

# expected number of steps to absorbtion
t <- N %*% matrix(1, n_transient_state, 1)

ggplot() +
  geom_line(aes(x=-(X-1):(Y-1), y=t)) +
  geom_point(aes(x=0, y=t[X]), colour="red", size=1) +
  geom_text(aes(x=0, y=t[X], label=t[X]), colour="red", nudge_y=2) +
  xlab("Starting Point") + ylab("Expected Game Length")
```
<img src="/images/bar-riddler.png" style="display:block; margin-left:auto; margin-right:auto;">
