Title: 538 Riddler: Puzzle of the Picky Eater
Date: 2016-06-22
Tags: 538, fivethirtyeight, riddler, geometry
Slug: sandwich-riddler
Resources: katex

I found it easiest to think about this problem in terms of polar coordinates.  The furthest point that we can eat along trajectory <span class="inline-equation" data-expr="\theta"></span> is the following:
<div class="equation" data-expr="r \left( \theta \right) = \frac{1}{2 \left( 1 + cos( \theta ) \right)} \quad \forall \theta \in \left[ 0, \frac{\pi}{4} \right]"></div>

Plotting this, it forms this weird, rounded rectangle shape:
<br>
<img src="/images/sandwich-riddler.jpg" width="400px" style = "display: block; margin-left: auto; margin-right: auto;">

I integrated the above equation to get the area.  Unlike a regular integral where each area is a rectangle (Reimann sum), each incremental integration area is a circular sector with <span class="inline-equation" data-expr="dA = \frac{ \pi dr^{2}}{2 \pi} = \frac{dr^{2}}{2}"></span>.  Putting this all together:
<div class="equation" data-expr="A = 8 \int_{0}^{ \frac{\pi}{4} } \frac{1}{8 \left( 1 + cos(\theta) \right)^2} d \theta = \frac{ sin(\theta) \left( cos(\theta) + 2 \right) }{3 \left( cos(\theta) + 1 \right)^2} \Big|_{0}^{ \frac{\pi}{4} } = \frac{1}{3} \left( 4 \sqrt{2} - 5 \right) = 21.9 \%"></div>

Extending this logic, I also derived an expression for the area eaten for a regular polygon with n sides:
<div class="equation" data-expr="A(n) = \frac{ cos^2(\frac{\pi}{n}) + 2 cos(\frac{\pi}{n}) }{3 \left( cos^2(\frac{\pi}{n}) + 2 cos(\frac{\pi}{n}) + 1 \right)}"></div>
<div class="equation" data-expr="\lim_{n \to \infty} A(n) = \frac{ cos^2(0) + 2 cos(0) }{3 \left( cos^2(0) + 2 cos(0) + 1 \right)} = \frac{1}{4} = 25 \%"></div>

As n goes to infinity, the area that the picky eater eats becomes a circle with a radius half that of the sandwich, making the area that is eaten equal to 25%.  This is the most efficient shape for the picky eater.
