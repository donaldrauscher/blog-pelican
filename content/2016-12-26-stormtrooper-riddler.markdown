Title: 538 Riddler: Rebel vs. Stormtroopers
Date: 2016-12-26
Tags: 538, fivethirtyeight, riddler, probability
Slug: stormtrooper-riddler
Resources: katex

In [this week's Riddler](http://fivethirtyeight.com/features/build-your-own-death-star-and-defeat-the-stormtroopers/), we are rebels trying to defeat a group of 9 advancing stormtroopers.  Fortunately for us, we are more accurate than the notoriously inaccurate stormtroopers, and the stormtroopers are clumped together, making them easy to pick off.

First, the hit / miss probabilities for the stormtroopers / rebel with N stormtroopers remaining:
<table class="pretty">
<tr><th>Probability</th><th>Hit</th><th>Miss</th></tr>
<tr><td>Stormtrooper</td><td><span class="inline-equation" data-expr="1-\left(\frac{999}{1000}\right)^{N}"></span></td><td><span class="inline-equation" data-expr="\left(\frac{999}{1000}\right)^{N}"></span></td></tr>
<tr><td>Rebel</td><td><span class="inline-equation" data-expr="\frac{K\sqrt{N}}{1000}"></span></td><td><span class="inline-equation" data-expr="1-\frac{K\sqrt{N}}{1000}"></span></td></tr>
</table>

The probability that the rebel shoots one of the N remaining stormtroopers before being shot can be expressed as follows:
<div class="equation" data-expr="\begin{aligned} P(\text{Rebels Win}) = & P(S_M \cap R_H) + P(S_M \cap R_M)*P(S_M \cap R_H) + P(S_M \cap R_M)^{2}*P(S_M \cap R_H) + \ldots \\ = & \frac{P(S_M \cap R_H)}{1-P(S_M \cap R_M)}=\frac{P(S_M \cap R_H)}{P(S_H) + P(S_M \cap R_H)} \\ = & \frac{\left(\frac{999}{1000}\right)^{N}\frac{K\sqrt{N}}{1000}}{1-\left(\frac{999}{1000}\right)^{N}+\left(\frac{999}{1000}\right)^{N}\frac{K\sqrt{N}}{1000}} \end{aligned}"></div>

Putting this together for the entire battle, this is the equation we need to solve:
<div class="equation" data-expr="P(\text{Rebels Win}) = \prod_{i=1}^{9} \frac{\left(\frac{999}{1000}\right)^{i}\frac{K\sqrt{i}}{1000}}{1-\left(\frac{999}{1000}\right)^{i}+\left(\frac{999}{1000}\right)^{i}\frac{K\sqrt{i}}{1000}} = \frac{1}{2}"></div>

That denominator is very hairy (and the word "approximately" in the prompt makes me think closed form is a bit of a pipe dream), so I went at this empirically.  Easy enough with Excel Goal Seek or Wolfram Alpha.  The rebel and stormtroopers are evenly matched when <span class="inline-equation" data-expr="K=26.797"></span>.  It gets a lot tougher for the rebel if the stormtroopers aren't clumped together; in that case, <span class="inline-equation" data-expr="K=62.055"></span> for the battle to be evenly matched.
