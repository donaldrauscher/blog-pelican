Title: 538 Riddler: Untangling the Tangled Wires
Date: 2016-12-17
Tags: 538, fivethirtyeight, riddler, logic
Slug: untangle-riddler
Resources: katex

The strategy for [this week's Riddler](http://fivethirtyeight.com/features/everythings-mixed-up-can-you-sort-it-all-out/) is to continuously split the wires into halves until only pairs remain.  Then, we form circuits between the pairs to pinpoint individual wires.  Using this approach, we can determine the optimal number of trips when N is a power of 2.  For <span class="inline-equation" data-expr="N = 2^{2} = 4"></span>, we need 2 trips.  For <span class="inline-equation" data-expr="N = 2^{3} = 8"></span>, we need 3 trips.  For <span class="inline-equation" data-expr="N = 2^{4} = 16"></span>, we need 4 trips.  Etc.

Next, we need to figure out what happens in between these known power-of-2 points.  Things get tricky when N is odd, or we can only form an odd number of pairs.  With one trip, we can effectively split the problem into two smaller sub-problems.  We can split the N=10 problem into a N=6 problem and a N=4 problem, each of which can be solved in 3 trips. So we need to round up in between powers-of-2.  For <span class="inline-equation" data-expr="N \in \left[ 5, 7 \right]"></span>, we need 3 trips.  For <span class="inline-equation" data-expr="N \in \left[ 9, 15 \right]"></span>, we need 4 trips.  A general expression for the number of trips needed for N wires is therefore:
<div class="equation" data-expr="\left \lceil \frac{\ln(N)}{\ln(2)} \right \rceil"></div>

The following visual illustrates how to solve the N=3 through N=10 cases.  Like colors represent wires that are tied together.  Lines represent known delineations; we can know that a wire is part of a group but know exactly which wire down below it belongs to.
<img src="/images/untangled-riddler.jpg" style="display:block; margin-left:auto; margin-right:auto;">

There are a couple tricky cases that are worth noting.  

+ **N=2**: We actually can't figure out the simplest tangle of all!  Well, we can't figure it out when playing by the current rules.  One option would be to connect a new wire to one of the wires at the bottom, run it up to the top, and then use our circuit tester with the two wires; this effectively makes it a N=3 case where one wire is known.
+ **N=6**: By extension, the fact that N=2 is impossible makes the N=6 case tricky.  Though we can still figure it out in 3 trips.  We start by splitting into two halves, one with 4 wires and one with 2 wires.  We then form a circuit between the two halves, which allows us to pinpoint 3 wires.  Finally, we form 3 circuits between the 3 known wires and the 3 unknown wires.
+ **N=10**: This is why we split this into N=6 and N=4 cases rather than N=8 and N=2 cases!
