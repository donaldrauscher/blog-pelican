Title: 538 Riddler: Chance of Being THE Deciding Vote
Date: 2016-11-06
Tags: 538, fivethirtyeight, riddler, probability
Slug: election-riddler
Resources: katex

[This week's Riddler](http://fivethirtyeight.com/features/a-puzzle-will-you-yes-you-decide-the-election/) tasked us with calculating the probability of being the deciding vote in a toss-up election.  For simplicity, I'm going to assume that there are an even number of other voters (an odd number of total voters).  We can model the number of votes for "our" candidate as a binominal random variable, making the probability of a split election simply:
<div class="equation" data-expr="\binom{N}{N/2} \left( \frac{1}{2} \right)^{N}"></div>

Since there is a nice asymptotic expression for the [central binomial coefficient](https://en.wikipedia.org/wiki/Central_binomial_coefficient), we can derive a simple expression for the probability of being the deciding vote for large N:
<div class="equation" data-expr="\lim_{N \to \infty } \binom{2N}{N} \approx \frac{4^{N}}{\sqrt{\pi\,N}} \rightarrow \binom{N}{N/2} \left( \frac{1}{2} \right)^{N} \approx \sqrt{\frac{2}{\pi\,N}} \quad \text{for large N}"></div>
<div class="equation" data-expr="\lim_{N \to \infty } \binom{N}{N/2} \left( \frac{1}{2} \right)^{N} = 0"></div>

Intuitively, it makes sense that, as N goes to infinity, the probability that we are the deciding vote converges to zero.  Although technically the most probable individual outcome, a split election becomes a smaller and smaller proportion of the possible outcomes.  This information will not deter me from voting on Tuesday!
