Title: 538 Riddler: Puzzle of the Robot Pizza Cutter
Date: 2016-06-26
Tags: 538, fivethirtyeight, riddler, geometry
Slug: pizza-riddler
Resources: katex

The robot's first cut will create 2 pizza slices.  The robot's second cut will create either 3 or 4 pizza slices.  The robot's third cut will create 4, 5, or 6 slices if starting with 3 slices or 5, 6, or 7 slices if starting with 4 slices.  I began by thinking about the probability of these outcomes in a specific state.  So that the geometry mimics the probability distribution, I'm setting the circumference of the pizza equal to 1.

<table>
<tr><th style = 'width:33%;'>Second Cut</th><th style = 'width:33%;'>Third Cut (Starting w/ 3 Slices)</th><th style = 'width:33%;'>Third Cut (Starting w/ 4 Slices)</th></tr>
<tr><td colspan = '3'><img src = '/images/pizza-riddler.jpg' width='100%'></td></tr>
<tr>
	<td style = "vertical-align: top;">
		<div class="inline-equation" data-expr="\scriptsize{  P(S_{2} = 3) = A^2 + B^2 }"></div>
		<div class="inline-equation" data-expr="\scriptsize{ P(S_{2} = 4) = 2AB }"></div>
	</td>
	<td>
		<div class="inline-equation" data-expr="\scriptsize{ P(S_{3} = 4 | S_{2} = 3) = A^2 + B^2 + C^2 + D^2 + 2BD }"></div>
		<div class="inline-equation" data-expr="\scriptsize{ P(S_{3} = 5 | S_{2} = 3) = 2 \left( AB + BC + CD + DA \right) }"></div>
		<div class="inline-equation" data-expr="\scriptsize{ P(S_{3} = 6 | S_{2} = 3) = 2AC }"></div>
	</td>
	<td>
		<div class="inline-equation" data-expr="\scriptsize{ P(S_{3} = 5 | S_{2} = 4) = A^2 + B^2 + C^2 + D^2 }"></div>
		<div class="inline-equation" data-expr="\scriptsize{ P(S_{3} = 6 | S_{2} = 4) = 2 \left( AB + BC + CD + DA \right) }"></div>
		<div class="inline-equation" data-expr="\scriptsize{ P(S_{3} = 7 | S_{2} = 4) = 2 \left( AC + BD \right) }"></div>
	</td>
</tr>
</table>

Next, I integrated over the entire state space to get the probability of each of these outcomes.  The state space for the second cut (after 1 cut has been made) is very easy to define; we only need 1 variable (x) to specify the size of each of the two slices.  The state space for the third cut (after 2 cuts have been made) is more challenging; we need 3 variables to define the space (x, y, and z). Each integral is multipled by the number of [circular permutations](http://mathworld.wolfram.com/CircularPermutation.html); in the case of the third cut, there are 4 points, so (4 - 1)! = 6.

For that second cut:
<div class="inline-equation" data-expr="(A, B) = \left\{ \left( x, 1 - x \right) : 0 \leq x \leq 1 \right\}"></div>
<div class="inline-equation" data-expr="P(S_{2} = 3) = \int_{0}^{1} x^2 + \left( 1 - x \right) ^2 dx = \frac{2}{3}"></div>
<div class="inline-equation" data-expr="P(S_{2} = 4) = \int_{0}^{1} 2x \left( 1 - x \right) dx = \frac{1}{3}"></div>

For that third cut:
<div class="inline-equation" data-expr="(A, B, C, D) = \left\{ \left( z, y - z, x - y, 1 - x \right) : 0 \leq z \leq y \leq x \leq 1 \right\}"></div>
<div class="inline-equation" data-expr="P(S_{3} = 4 | S_{2} = 3) = 3! \int_{0}^{1} \int_{z}^{1} \int_{y}^{1} z^2 + \left( y - z \right) ^2 + \left( x - y \right) ^2 + \left( 1 - x \right) ^2 + 2 \left( y - z \right) \left( 1 - x \right) \,dx\,dy\,dz = \frac{1}{2}"></div>
<div class="inline-equation" data-expr="P(S_{3} = 5 | S_{2} = 3) = 3! \int_{0}^{1} \int_{z}^{1} \int_{y}^{1} 2 \left( z \left( y - z \right) + \left( x - y \right) \left( y - z \right) + \left( 1 - x \right) \left( x - y \right) + z \left( 1 - x \right) \right) \,dx\,dy\,dz = \frac{2}{5}"></div>
<div class="inline-equation" data-expr="P(S_{3} = 6 | S_{2} = 3) = 3! \int_{0}^{1} \int_{z}^{1} \int_{y}^{1} 2 z \left( x - y \right) \,dx\,dy\,dz = \frac{1}{10}"></div>
<div class="inline-equation" data-expr="P(S_{3} = 5 | S_{2} = 4) = 3! \int_{0}^{1} \int_{z}^{1} \int_{y}^{1} z^2 + \left( y - z \right) ^2 + \left( x - y \right) ^2 + \left( 1 - x \right) ^2 \,dx\,dy\,dz = \frac{2}{5}"></div>
<div class="inline-equation" data-expr="P(S_{3} = 6 | S_{2} = 4) = P(S_{3} = 5 | S_{2} = 3)"></div>
<div class="inline-equation" data-expr="P(S_{3} = 7 | S_{2} = 4) = 3! \int_{0}^{1} \int_{z}^{1} \int_{y}^{1} 2 \left( z \left( x - y \right) + \left( y - z \right) \left( 1 - x \right) \right) \,dx\,dy\,dz = \frac{1}{5}"></div>

Putting this all together:
<div class="equation" data-expr="E\left[S_{3}\right] = \frac{2}{3} \left( 4 * \frac{1}{2} + 5 * \frac{2}{5} + 6 * \frac{1}{10} \right) + \frac{1}{3} \left( 5 * \frac{2}{5} + 6 * \frac{2}{5} + 7 * \frac{1}{5} \right) = 5"></div>
