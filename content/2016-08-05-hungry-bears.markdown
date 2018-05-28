Title: 538 Riddler: Hungry Bears
Date: 2016-08-05
Tags: 538, fivethirtyeight, riddler, probability
Slug: hungry-bears
Resources: katex

My first intuition after reading [this problem](http://fivethirtyeight.com/features/should-the-grizzly-bear-eat-the-salmon/) was "why would the bear ever reject the first fish?" If the first fish is big, then it makes sense to eat it; there are no guarantees the next fish will be as big.  If the first fish is small, then eat it because the next fish is likely to be bigger, meaning we can eat it too. Some simple math confirms this. Let's say the first fish is of <span class="inline-equation" data-expr="x"></span> size. The probability that we eat the next fish is <span class="inline-equation" data-expr="\left( 1-x \right)"></span> and the expected size of this fish is <span class="inline-equation" data-expr="\frac{\left( 1+x \right)}{2}"></span>.  The total meal size if we eat the first fish is <span class="inline-equation" data-expr="x + \frac{\left( 1-x \right)\left( 1+x \right)}{2}"></span> is always greater than or equal <span class="inline-equation" data-expr="\frac{1}{2}"></span>, the expected meal size if we wait for the second fish.  So if we're the bear, our optimal strategy is simple: eat whatever we see.

Given our rather simple strategy, let's come up with a generalized expression for the expected number of kilograms that the bear eats in a <span class="inline-equation" data-expr="N"></span> hour tour:
<div class="equation" data-expr="M_{N} = \sum_{i=1}^{N} F_{i} * I_{\left\{ F_{i} &gt; max \left( F_{1}, ... , F_{i-1} \right) \right\}}; F_{i} \sim U(0,1)"></div>
<div class="equation" data-expr="E[M_{N}] = \sum_{i=1}^{N} \int_{0}^{1} \int_{l}^{1} f * f_{L_{i}}(l) \,df \,dl \qquad L_{i} \sim max \left( F_{1}, ... , F_{i} \right); F_{L_{i}}(x) = x^{i} \rightarrow f_{L_{i}}(x) = i\,x^{i-1}"></div>
<div class="equation" data-expr="= \sum_{i=1}^{N} \int_{0}^{1} \left( \frac{1}{2} - \frac{1}{2} l^{2} \right)\left( i-1 \right ) l^{i-2} \,dl = \sum_{i=1}^{N} \frac{1}{i+1}"></div>

This is the [harmonic series](https://en.wikipedia.org/wiki/Harmonic_series_%28mathematics%29) minus 1! In 2 hours, we expect the bear to eat <span class="inline-equation" data-expr="\frac{1}{2} + \frac{1}{3} = \frac{5}{6}"></span> kilograms. In 3 hours, we expect the bear to eat <span class="inline-equation" data-expr="\frac{1}{2} + \frac{1}{3} + \frac{1}{4} = \frac{13}{12}"></span> kilograms.  The harmonic series is divergent, so the amount eaten by the bear does not converge on a value as N goes to infinity, even though the amount consumed by the bear in the Nth hour converges to 0.
