Title: 538 Riddler: How Big A Table Can The Carpenter Build?
Date: 2016-09-25
Tags: 538, fivethirtyeight, riddler, geometry
Slug: table-riddler
Resources: katex

In [this week's Riddler](http://fivethirtyeight.com/features/how-big-a-table-can-the-carpenter-build/), the largest circular table that we can carve out of our 4x8 piece of wood with two congruent semi-circles has a radius of ~2.70 feet. We can fit the largest semi-circles in the wood by orienting them diagonally:

<img src='/images/table-riddler.jpg' style="display:block; margin-left:auto; margin-right:auto;">

From the above graph, we can use a few equations to solve for X:
<div class="equation" data-expr="\left( 4 - x \right)^{2} + \left( X - A \right)^{2} = X^{2} \rightarrow A = X - \sqrt{8X - 16}"></div>
<div class="equation" data-expr="tan(\theta) = \frac{X - A}{4 - X} = \frac{B - X}{X} \rightarrow B = X \left( 1 + \frac{\sqrt{8X - 16}}{4 - X} \right)"></div>
<div class="equation" data-expr="A + B = 8 \rightarrow X = 2.7054"></div>

We've managed to use <span class="inline-equation" data-expr="\frac{\pi\,2.7054^{2}}{32} \approx 72\%"></span> of the available wood for our circular table.
