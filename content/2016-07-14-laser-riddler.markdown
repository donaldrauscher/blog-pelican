Title: 538 Riddler: Defending Riddler Headquarters
Date: 2016-07-14
Tags: 538, fivethirtyeight, riddler
Slug: laser-riddler

The challenging part of this problem was creating an exhaustive state-space of bisectors.  Odd-numbered polygons are a real pain.  I started with a known bisector: a line that goes through one point and intersects the mid-point of the opposite side.  If we shift the line slightly on one side, how much will we need to shift it on the opposing side (a rotation) to keep the area on either side equal?  A little trig helps us figure this out.  Interestingly, these bisectors don't always intersect the centroid.  Because of this, an interesting, star-shaped hot spot forms in the middle of the shape.  We can minimize our chances of getting hit if we stand at the mid point of one of the outer walls of the building, which makes intuitive sense.  

<img src="/images/laser-riddler.jpg" style="display:block; margin-left:auto; margin-right:auto; width: 500px;">

``` R
library(ggplot2)

# make the pentagon
x <- c(0, cumsum(rep(c(1, cos(2*pi/5), -cos(pi/5), -cos(pi/5), cos(2*pi/5))/2, each = 2)))
y <- c(0, cumsum(rep(c(0, sin(2*pi/5), sin(pi/5), -sin(pi/5), -sin(2*pi/5))/2, each = 2)))
pentagon <- data.frame(x = x, y = y)

# make some bisectors
n <- 15
delta.edge <- seq(0, 0.5, 0.5/n)
A <- 0.5/tan(pi/10)
delta.point <- A*delta.edge / (delta.edge*cos(3*pi/10) + A*sin(3*pi/10))

bisectors <- do.call(rbind, lapply(1:5, function(i){

  if ((i %% 2) == 1){
    delta.1 <- delta.point
    delta.2 <- delta.edge
  } else {
    delta.1 <- delta.edge
    delta.2 <- delta.point
  }

  bisectors.begin.x <- x[i] + (x[i+1] - x[i])*delta.1/0.5
  bisectors.begin.y <- y[i] + (y[i+1] - y[i])*delta.1/0.5

  bisectors.end.x <- x[i+5] + (x[i+6] - x[i+5])*delta.2/0.5
  bisectors.end.y <- y[i+5] + (y[i+6] - y[i+5])*delta.2/0.5

  bisectors.x <- do.call(c, lapply(1:n, function(j) c(bisectors.begin.x[j], bisectors.end.x[j])))
  bisectors.y <- do.call(c, lapply(1:n, function(j) c(bisectors.begin.y[j], bisectors.end.y[j])))

  return(data.frame(x = bisectors.x, y = bisectors.y, piece = rep((n*(i-1) + 1):(n*i), each = 2)))

}))

# plot it
ggplot() +
  geom_path(aes(x = pentagon$x, y = pentagon$y)) +
  geom_path(aes(x = bisectors$x, y = bisectors$y, group = bisectors$piece), colour = "red")

```
