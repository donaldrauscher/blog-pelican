Title: 538 Riddler: Puzzle of the Overflowing Martini
Date: 2016-05-19
Tags: 538, fivethirtyeight, riddler, geometry
Slug: martini-riddler
Resources: katex

Here's my solution to [this week's 538 Riddler](http://fivethirtyeight.com/features/can-you-solve-the-puzzle-of-the-overflowing-martini-glass/):
[<img src="/images/martini-riddler.jpg" width="885x">](/images/martini-riddler.jpg)

If the liquid reaches <span class="inline-equation" data-expr="p"></span> fraction of the way up the glass when upright, then the liquid goes <span class="inline-equation" data-expr="p^2"></span> fraction of the way up the glass on the opposite side just before it begins to pour.  

[Dandelin spheres](https://en.wikipedia.org/wiki/Dandelin_spheres) were key to my proof. They prove that the top of the liquid (a conic section) forms an ellipse.  They also prove that a sphere tangent to the top of the liquid and the sides of the glass intersects that elliptical conic section and one of its foci!  I derived two expressions for the volume of the liquid (one pouring, one upright) and set them equal to one another.  Interestingly, the steepness of the glass (<span class="inline-equation" data-expr="\theta"></span> in my proof), drops out of the expression!  
