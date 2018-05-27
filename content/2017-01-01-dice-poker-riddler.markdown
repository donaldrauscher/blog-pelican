Title: 538 Riddler: Dice Poker Riddler
Date: 2017-01-01
Tags: 538, fivethirtyeight, riddler, game_theory, linear_programming
Slug: dice-poker-riddler

In [this week's Riddler](http://fivethirtyeight.com/features/can-you-deal-with-these-card-game-puzzles/), we have another game theory problem. We can describe each player's strategy with a 6 number tuple. For player A, <span class="inline-equation" data-expr="a_{i}"></span> represents the probability that player A raises given a roll of i.  For player B, <span class="inline-equation" data-expr="b_{i}"></span> represents the probability that player B calls a raise from player A given a roll of i.  Each player's expected winnings can be expressed as follows:
<div class="equation" data-expr="\pi_{A}(a,b) = \frac{1}{36}\sum_{i=1}^{6} \sum_{j=1}^{6} (1-a_{i})*\epsilon_{ij} + a_{i}*\left( 2*b_{j}\epsilon_{ij} + 1*\left(1-b_{j}\right) \right)"></div>
<div class="equation" data-expr="\pi_{B}(a,b) = -\pi_{A}(a,b)"></div>
<div class="equation" data-expr="\text{where } \epsilon_{ij} = \begin{cases}
 1 & \text{if } i>j \\
 0 & \text{if } i=j \\
 -1 & \text{if } i<j
\end{cases}"></div>

We can start by analyzing the [pure strategies](https://en.wikipedia.org/wiki/Strategy_(game_theory)#Pure_and_mixed_strategies).  Pure strategies explicitly define how a player will play a game (e.g. do X if opponent does Y).  In the above definition, a pure strategy is one where <span class="inline-equation" data-expr="a_{i}"></span> and <span class="inline-equation" data-expr="b_{i}"></span> are binary.  For our game, each player has <span class="inline-equation" data-expr="2^{6}"></span> potential pure strategies.  Using the above formula, we can calculate player A's winnings for a every pair of pure strategies, then search the resulting 64x64 grid (<span class="inline-equation" data-expr="\pi_{A}"></span>) for Nash equilibria.  Because this is a zero-sum game, potential pure Nash equilibria <span class="inline-equation" data-expr="(\tilde{a},\tilde{b})"></span> will be "saddle" point(s) satisfying the following conditions:
<div class="equation" data-expr="\max_{1 \leq i \leq 64}\pi_{A}(a,\tilde{b}) = \pi_{A}(\tilde{a},\tilde{b}) = \min_{1 \leq i \leq 64}\pi_{A}(\tilde{a},b)"></div>

In other words, they will be column maximums (meaning player A will have no reason to deviate) and row minimums (meaning player B will have no reason to deviate).  Interestingly, as is seen in the visual below, there are no pure Nash equilibria!  We can see this by looking at a few common strategies.  Player A's most common row-maximizing strategy is always raising: (1,1,1,1,1).  However, if player B knew player A was using this strategy, they would respond by only calling if they had at least a 2, netting them $0.11 of expected winnings.  And if player A knew player B was using this strategy, they would respond by only raising when they had a 4 or higher, netting them $0.17 of expected winnings and triggering yet another change to player B's strategy.

Though there is not a pure strategy Nash equilibrium, [there must exist](https://en.wikipedia.org/wiki/Nash_equilibrium#Nash.27s_Existence_Theorem) a mixed strategy Nash equilibrium. A mixed strategy is simply a linear combination of pure strategies, the coefficients representing how often that strategy is to be used. [Von Neumann's minimax theorem](https://en.wikipedia.org/wiki/Minimax_theorem) tells us that this equilibrium is at the minimax.  We [can construct a pair of linear programs](https://advancedoptimizationatharvard.wordpress.com/2014/02/20/applying-linear-programming-to-game-theory/) to find the first and second player strategies (<span class="inline-equation" data-expr="u"></span> and <span class="inline-equation" data-expr="v"></span> respectively):
<div class="equation" data-expr="\begin{matrix}
\text{max } \lambda & & = & \text{min } \mu & \\
s.t. & u \in \left[ 0,1 \right] & & s.t. & v \in \left[ 0,1 \right] \\
& 1^{T} u = 1 & & & 1^{T} v = 1 \\
& \pi^{T} u \geq \lambda & & & \pi v \leq \mu
\end{matrix}"></div>

Solving these linear programs, we find that player A's optimal strategy is to always raise when they have a 5 or 6 and raise <span class="inline-equation" data-expr="\frac{2}{3}"></span> of the time when they have a 1 (i.e. bluff).  Very cool!  Player B's optimal strategy is to always call when they have a 5 or 6 and call <span class="inline-equation" data-expr="\frac{1}{2}"></span>, <span class="inline-equation" data-expr="\frac{5}{6}"></span>, and <span class="inline-equation" data-expr="\frac{1}{3}"></span> of the time when they have a 2, 3, and 4 respectively.  At this equilibrium, player A's expected winnings are $0.093.  

Calculations below in R. Also need to credit Laurent Lessard's [blog post](http://www.laurentlessard.com/bookproofs/baby-poker/) on this problem; as always, he did an awesome job laying out the underlying math.   

``` R
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
suppressMessages(library(ggplot2))
suppressMessages(library(lpSolve))

# win / lose / split matrix
E <- matrix(0, nrow=6, ncol=6)
E[upper.tri(E)] <- 1
E[lower.tri(E)] <- -1

# a is probability of raise, b is probability of call if raised; both are vectors of length 6
winnings <- function(a, b){
  a_matrix <- matrix(rep(a, times=6), ncol=6, byrow=TRUE)
  b_matrix <- matrix(rep(b, times=6), ncol=6, byrow=FALSE)
  temp <- (1-a_matrix)*E + a_matrix*(2*b_matrix*E + 1*(1-b_matrix))
  return(mean(temp))
}

# make exhaustive pure strategy state space
pure_strat <- t(expand.grid(lapply(1:6, function(x) 0:1)))
n_pure_strat <- ncol(pure_strat)
pure_strat_cross <- expand.grid(A=1:n_pure_strat, B=1:n_pure_strat)
pure_strat_cross$W <- with(pure_strat_cross, mapply(function(a, b) winnings(pure_strat[,a], pure_strat[,b]), A, B))

# find pure nash (if exists)
pure_strat_cross <- pure_strat_cross %>%
  group_by(A) %>% mutate(min_W_per_A = min(W)) %>% ungroup() %>%
  group_by(B) %>% mutate(max_W_per_B = max(W)) %>% ungroup() %>%
  mutate(is_nash_eq = (W == min_W_per_A & W == max_W_per_B))

sum(pure_strat_cross$is_nash_eq)
```
<pre>[1] 0</pre>
``` R
ggplot() +
  geom_tile(data=pure_strat_cross, aes(x = A, y = B, fill = W)) +
  scale_fill_gradient(low="red", high="green") +
  geom_tile(data=filter(pure_strat_cross, W == max_W_per_B), aes(x = A, y = B), colour="darkgreen", alpha=0, size=0.75) +
  geom_tile(data=filter(pure_strat_cross, W == min_W_per_A), aes(x = A, y = B), colour="darkred", alpha=0, size=0.75) +
  geom_tile(data=filter(pure_strat_cross, is_nash_eq), aes(x = A, y = B), fill="gold")
```
<img src="/images/dice-poker-pure-strategies.png" style="display:block; margin-left:auto; margin-right:auto;">

``` R
# most used strategies and opposite player responses
top_A_strategy <- pure_strat_cross %>% filter(W == max_W_per_B) %>%
  group_by(A) %>% summarise(num_B = n_distinct(B)) %>%
  arrange(-num_B) %>% head(1) %>% .$A

t(pure_strat[,top_A_strategy])
```
<pre>     Var1 Var2 Var3 Var4 Var5 Var6
[1,]    1    1    1    1    1    1</pre>
``` R
t(pure_strat[,pure_strat_cross %>% filter(A == top_A_strategy & W == min_W_per_A) %>% .$B])
```
<pre>     Var1 Var2 Var3 Var4 Var5 Var6
[1,]    0    0    1    1    1    1
[2,]    0    1    1    1    1    1</pre>
``` R  
top_B_strategy <- pure_strat_cross %>% filter(W == min_W_per_A) %>%
  group_by(B) %>% summarise(num_A = n_distinct(A)) %>%
  arrange(-num_A) %>% head(1) %>% .$B

t(pure_strat[,top_B_strategy])
```
<pre>     Var1 Var2 Var3 Var4 Var5 Var6
[1,]    0    1    1    1    1    1</pre>
``` R
t(pure_strat[,pure_strat_cross %>% filter(B == top_B_strategy & W == max_W_per_B) %>% .$A])
```
<pre>     Var1 Var2 Var3 Var4 Var5 Var6
[1,]    0    0    0    0    1    1
[2,]    0    0    0    1    1    1</pre>
``` R
# mixed strategy nash equilibrium
P <- matrix(pure_strat_cross %>% arrange(A, B) %>% .$W, byrow=TRUE, ncol=n_pure_strat)

obj <- c(rep(0, n_pure_strat), 1)
A_base <- rbind(
  matrix(c(rep(1, n_pure_strat), 0), nrow=1), # sum of probabilities equals 1
  cbind(diag(n_pure_strat), 0) # probabilities >= 0
)
A1 <- rbind(A_base, cbind(t(P), -1)) # minimax constraints
A2 <- rbind(A_base, cbind(-P, 1)) # minimax constraints
dir <- c("=", rep(">=", 2*n_pure_strat))
b <- c(1, rep(0, 2*n_pure_strat))

lp_A <- lp(direction="max", objective.in=obj, const.mat=A1, const.dir=dir, const.rhs=b)
lp_B <- lp(direction="min", objective.in=obj, const.mat=A2, const.dir=dir, const.rhs=b)

# solutions and expected winnings
head(lp_A$solution, -1) %*% t(pure_strat)
```
<pre>          Var1 Var2 Var3 Var4 Var5 Var6
[1,] 0.6666667    0    0    0    1    1</pre>
``` R
head(lp_B$solution, -1) %*% t(pure_strat)
```
<pre>
     Var1 Var2      Var3      Var4 Var5 Var6
[1,]    0  0.5 0.8333333 0.3333333    1    1</pre>
``` R
lp_A$objval
```
<pre>[1] 0.09259259</pre>
