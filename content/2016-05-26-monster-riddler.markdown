Title: 538 Riddler: Puzzle of the Monsters' Gems
Date: 2016-05-26
Tags: 538, fivethirtyeight, riddler, probability
Slug: monster-riddler
Resources: katex

There are three ways that this game can end: slaying a rare monster (the most likely), slaying an uncommon monster, or slaying a common monster (the least likely).  I began by thinking about the probability of each of these events happening.  The probability of the game ending by slaying a rare monster can be expressed as the following:
<div class="equation" data-expr="\sum_{i=3}^{\infty} \left( 1 - P \left( R \right) \right)^{i-1} * P \left( R \right) * P \left( S_{U}^{i-1} \geq1 \wedge S_{C}^{i-1} \geq 1 | \left\{ M_{1},...,M_{i-1} \right\} \in \left\{ U,C \right\} \right)"></div>
<div class="equation" data-expr="= \sum_{i=3}^{\infty}\left ( \frac{5}{6} \right )^{i-1}*\left ( \frac{1}{6} \right )*\left (1 - \left ( \frac{3}{5} \right )^{i-1} - \left ( \frac{2}{5} \right )^{i-1} \right ) = \frac{7}{12}"></div>

Extending that same logic, we can calculate the probability that we end the game by slaying an uncommon and common monster as well:
<div class="equation" data-expr="P \left( End \ in \ R \right) = \frac{7}{12} \qquad P \left( End \ in \ U \right) = \frac{4}{15} \qquad P \left( End \ in \ C \right) = \frac{3}{20}"></div>

...these sum to 1 and match what we would expect directionally.

Next, I added the expected number of common monsters slayed to each of these scenarios.  If the game ends by slaying a common monster (which only happens 15% of the time), this is simply 1.  In the other two scenarios, the number of common monsters slayed is a binomial random variable, conditioned on not all the previously slayed monsters being of a single type.  Bayes helps us reduce the expression:
<div class="equation" data-expr="P \left( S_{U}^{i-1} \geq 1 \wedge S_{C}^{i-1} \geq 1 \right) * E \left[ S_{C}^{i-1} | S_{U}^{i-1} \geq 1 \wedge S_{C}^{i-1} \geq1 \right]"></div>
<div class="equation" data-expr="= E \left[ S_{C}^{i-1} \right] - P \left( S_{U}^{i-1} = 0 \right) * E \left[ S_{C}^{i-1} | S_{U}^{i-1} = 0 \right] - P \left( S_{C}^{i-1} = 0 \right) * E \left[ S_{C}^{i-1} | S_{C}^{i-1} = 0 \right]"></div>
<div class="equation" data-expr="= (i - 1) * p - (i - 1) * p^{i-1} \quad where \quad p = \frac{P(C)}{P(C) + P(U)}"></div>

Putting this all together:
<div class="equation" data-expr=" = \sum_{i=2}^{\infty} \left( \frac{5}{6} \right)^{i} * \frac{1}{6} * \left[ i * \frac{3}{5} - i * \left( \frac{3}{5} \right)^{i} \right] + \sum_{i=2}^{\infty} \left( \frac{2}{3} \right)^{i} * \frac{1}{3} * \left[ i * \frac{3}{4} - i * \left( \frac{3}{4} \right)^{i} \right] + \frac{3}{20} = 3.65"></div>
