Title: 538 Riddler: Defending Against an Alien Invasion
Date: 2016-07-29
Tags: 538, fivethirtyeight, riddler, geometry
Slug: alien-invasion
Resources: katex

This [week's Ridder](http://fivethirtyeight.com/features/solve-the-puzzle-stop-the-alien-invasion/) was the first one that I got wrong! I never really felt confident in my answer, thus no post.  In the end, I did not model random points on the surface of the sphere correctly.  A [good link](http://mathworld.wolfram.com/SpherePointPicking.html) supplied by the 538 folks demonstrates how to do this correctly.

In retrospect, there are two random variables: <span class="inline-equation" data-expr="\phi_{1}"></span> and <span class="inline-equation" data-expr="\phi_{2}"></span>.  The former represents the angular distance between the two alien ships.  The latter represents the angular distance between the defender and the midpoint of the two alien ships.  These two random variables have the following probability distributions:
<div class="equation" data-expr="f \left( \phi_{1} \right) = sin \left( 2\phi_{1} \right) \quad \forall \phi_{1} \in \left[ 0, \frac{ \pi }{2} \right]; \quad f \left( \phi_{2} \right) = \frac{1}{2} sin \left( \phi_{2} \right) \quad \forall \phi_{2} \in \left[ 0, \pi \right]"></div>

These are independent, so we can simply combine them into a joint distribution and integrate to give us the probability that the planet is defended:
<div class="equation" data-expr="P \left( \phi_{2} &lt; 20 \phi_{1} \right) = \int_{0}^{\pi} \int_{\frac{\phi_{2}}{20}}^{\frac{\pi}{2}} \frac{1}{2} sin \left( 2\phi_{1} \right) sin \left( \phi_{2} \right) \,d\phi_{1} \,d\phi_{2} = 99.3\%"></div>
