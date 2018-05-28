Title: 538 Riddler: Puzzle of Baseball Divisional Champs
Date: 2016-05-26
Tags: 538, fivethirtyeight, riddler, baseball
Slug: baseball-riddler
Resources: katex

For [this week's Riddler](http://fivethirtyeight.com/features/can-you-solve-the-puzzle-of-the-baseball-division-champs/), I estimated that the division leader would have 88.8 wins after 162 games.  I assumed that each team plays the other teams in it's division 19 times for a total of 76 intradivision games and 86 interdivision games, consistent with the [actual scheduling rules](https://en.wikipedia.org/wiki/Major_League_Baseball_schedule).

Interdivision games are pretty easy to deal with because the outcomes of each team's interdivision games are independent of one another. If all 162 games were interdivision (i.e. the teams in the division somehow never play one another), the answer would be pretty straightforward (code in R):

```R
> cdf <- (pbinom(1:162, 162, 0.5))^5
> pmf <- cdf - c(0, head(cdf, -1))
> sum(1:162 * pmf)
```
<pre>[1] 88.39431</pre>

However, teams in the division obviously do play one another, so win-loss records are not independent.  We can think of intradivision games as a series of consecutive round robins.  Each round robin consists of 10 games, and each team plays in 4 of those games (one game against each of their division foes).  Each team plays 76 intradivision games, which is 19 round robins and 190 games. I started by creating an exhaustive state space (and corresponding probabilities) for the win totals of the 1st, 2nd, 3rd, 4th, and 5th place teams after the 190 intradivision games.  Each state is defined as <span class="inline-equation" data-expr="s = \left(t_{1},t_{2},t_{3},t_{4},t_{5}\right)"></span> such that <span class="inline-equation" data-expr="t_{1} \geq t_{2} \geq t_{3} \geq t_{4} \geq t_{5}"></span>.  As you can expect, there are many possible outcomes (157,470 states in my state space)!

Next, for each state, I calculated the probability that the division winning team would have less than X wins after the remaining 86 interdivision games.  Results of interdivision games are independent, making this pretty easy.  Sum-product with the state probabilities and we have the CDF for X!
<div class="equation" data-expr="CDF \left( x \right) = \sum_{s \in \mathbb{S}} P\left( s \right) * P \left( X \leq \left( x - s \right) \right)"></div>
<div class="equation" data-expr="= \sum_{s \in \mathbb{S}} P\left( s \right) * \prod_{i = 1}^{i = 5} P \left( X \leq \left( x - t_{i} \right) \right) \quad where \quad X \sim B \left( 86, 0.5 \right) \quad \forall x \in \left[ 38, 162 \right]"></div>

<img src="/images/win_distribution.jpeg" style='display:block; margin-left: auto; margin-right: auto;'>

This computes to 88.8 wins.  This is slightly higher than the all-interdivision calculation above, which makes intuitive sense.  In the all-interdivision scenario, we could theoretically have a division leader with 0 wins (all 5 teams go winless).  However, when we have intradivision games, the division leader cannot have fewer wins than the number of intradivision games that they play divided by 2; one team's loss is another team's win.

I've posted my code on my GH [here](https://github.com/donaldrauscher/baseball-riddler). Enjoy!
