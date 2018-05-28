Title: 538 Riddler: The <span style='text-decoration:line-through;'>C</span>Gold War
Date: 2016-12-15
Tags: 538, fivethirtyeight, riddler, game_theory
Slug: gold-war-riddler
Resources: katex

[This week's Riddler](http://fivethirtyeight.com/features/how-much-gold-would-push-you-into-a-war/) challenges us with some game theory.  Each player has a hefty $1 trillion in gold and an army whose strength is uniformly distributed between 0 and 1.  Each player knows their own army's strength but not their opponent's army's strength (obviously).  Each player then simultaneously declares "Peace" or "War".  If either player has declared "War", then war ensues, and the player with the most power army walks away with a cool $2 trillion (and the loser walks away with a big goose egg).  If both players declare "Peace", then both parties retreat with their $1 trillion in tact.  

Let's assume our opponent follows a simple strategy: if their strength is greater than Y, then go to war; if their strength is less than Y, then declare peace.  And let's say we want to adopt a similar strategy but with a cutoff of X. What should our cutoff be?

Right away, we can discard any cutoffs X&gt;Y.  If our strength is [Y,X) and our opponent's strength is [0,Y), we should win $2 trillion, but, with this strategy, we settle for $1 trillion instead. We're not capitalizing on our higher strength as often as we should.  We would always prefer a strategy with X=Y over a strategy with X>Y.

What about X&lt;Y?  First, let's draw out the state space:
[<img src="/images/gold-war.jpg" style="display:block; margin-left:auto; margin-right:auto;" width="500px">](/images/gold-war.jpg)

We can define our expected winnings as follows:
<div class="equation" data-expr="E \left[ W_{1}|S_{1}=x \cap  S_{2}=y \right] = x*y*1 + \frac{1}{2}*(1+x)*(1-x)*2 = xy+1-x^{2}"></div>
<div class="equation" data-expr="\frac{\partial}{\partial x} \left( \cdot \right ) = y-2x = 0 \rightarrow x = \frac{y}{2}"></div>
<div class="equation" data-expr="E \left[ W_{1}|S_{1}=\frac{y}{2} \cap  S_{2}=y \right] = 1 + \frac{y^{2}}{4} &gt; 1"></div>

Our optimal strategy is to be 50% more warmongering than our opponent, a strategy which nets us expected winnings greater than $1 trillion!  Since this is effectively a zero-sum game, our gain comes at our opponent's expense.  How will our opponent react?  Eventually, after playing the game a few times, our opponent will realize that we have a lower war threshold, and they will undercut us.  We will react in kind, and so forth and so forth. Eventually, we will both converge on the origin. This is the only Nash Equilibrium for the game; once at this point, it doesn't make sense for either party to deviate.  This game bears resemblance to the popular [Prisoner's dilemma](https://en.wikipedia.org/wiki/Prisoner's_dilemma).  "Peace" is analagous to "cooperation", and "war" is analagous to "defection".  

**Additional question #1: What if a victory at war yields $5 trillion instead of $2 trillion?**  This doesn't change the dynamics of the game.  In the above scenario, the optimal response strategy becomes <span class="inline-equation" data-expr="\frac{y}{5}"></span>.  It doesn't change the Nash Equilibrium of the game.  If anything, it drives us to converge on the N.E. after fewer iterations.

**Additional question #2: what if the game is played sequentially?** Interestingly, this makes the game completely trivial! Let's say we go second.  There are no strategic decisions for us to make!  After a few iterations of the game, we will deduce our opponent's war threshold X.  If our opponent declares war, then we're going to war regardless.  If they declare peace, this signals to us that their army's strength is <X. So we simply go to war and pocket $2 trillion if our army strength is >X, and otherwise declare peace and pocket $1 trillion. What about the first player?  What is the optimal value of X?  We can show that player 1's expected winnings are $1 trillion for all values of X!
<div class="equation" data-expr="E \left[ W_{1}|S_{1}=x \right] = P(A_{1}&lt;x) * P(A_{2} &lt; x) * 1 + P(A_{1} \geq x) * P(A_{1} &gt; A_{2} | A_{1} \geq x) * 2"></div>
<div class="equation" data-expr=" = x^{2} + (1-x) \left( \frac{1}{2}*(1+x)*2 \right) = 1"></div>
