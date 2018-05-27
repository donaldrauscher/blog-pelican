Title: 538 Riddler: Martin Gardner's 'Hip' Game
Date: 2017-01-09
Tags: 538, fivethirtyeight, riddler, linear_programming
Slug: hip-game-riddler
Resources: katex

I began [this week's Riddler](https://fivethirtyeight.com/features/dont-throw-out-that-calendar/) by deriving an expression for the number of squares on a n-sized board:
<img src="/images/hip-square-cnt.png" style="display:block; margin-left:auto; margin-right:auto;">
<div class="equation" data-expr="\begin{aligned} S(n) = & \sum_{i=1}^{n-1} i^2*(n-i) = n\sum_{i=1}^{n-1} i^2 - \sum_{i=1}^{n-1} i^3 \\
= & n \left( \frac{n(n-1)(2n-1)}{6}\right) - \frac{n^2(n-1)^2}{4} = \frac{n^2(n^2-1)}{12}
\end{aligned}"></div>
This expression is a polynomial with a degree of 4, which confirms that the number of squares grows more quickly than the area of the board, making it increasingly difficult to achieve a draw.

To find tie configurations, I used an integer program.  The objective function, to be minimized, represents the number of squares formed.  There are two constraints for each square, each requiring that all 4 points not be assigned to a single player.  And one constraint requires that the points be evenly divided between the two players.  All variables are binary.
<div class="equation" data-expr="\begin{aligned} \\
\text{min} \quad & \sum_{s \in S} u_{s} + v_{s} \\
s.t. \quad & \sum_{i=1}^{n^2} x_{i} = \left \lceil \frac{n^2}{2} \right \rceil \\
& x_{s1} + x_{s2} + x_{s3} + x_{s4} + u_{s} &gt; 0 & \forall s \in S \\
& x_{s1} + x_{s2} + x_{s3} + x_{s4} - v_{s} &lt; 4 & \\
& x_{i}, u_{s}, v_{s} \in \left\{ 0,1 \right\} \\
\end{aligned}"></div>

A 6x6 board is the largest board for which the optimal solution to this integer program is 0, indicating a tie. For a 7x7 board, the optimal solution still has 3 squares. And ties must be impossible on any larger boards since they will of course contain a 7x7 sub-board. Here's what optimal solutions on the 6x6 and 7x7 boards look like:
<table style="width:100%;"><tr>
<td style="width:50%; "><img src="/images/hip-n6.png" style="display:block; margin-left:auto; margin-right:auto;"></td>
<td style="width:50%;"><img src="/images/hip-n7.png" style="display:block; margin-left:auto; margin-right:auto;"></td>
</tr></table>

``` R
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))
suppressMessages(library(ggplot2))
suppressMessages(library(lpSolve))

# make the squares
n <- 6
squares <- data.frame()
for (i in 1:(n-1)){
  for (j in 2:n){
    p1 <- c(i, j)
    p2 <- expand.grid(x2=(p1[1]+1):n, y2=p1[2]:n)
    p3 <- data.frame(x3=p2$x2+p2$y2-p1[2], y3=p2$y2-p2$x2+p1[1])
    p4 <- data.frame(x4=p1[1]+p2$y2-p1[2], y4=p1[2]-p2$x2+p1[1])
    squares_temp <- cbind(data.frame(x1=p1[1], y1=p1[2]),p2,p3,p4)
    squares_temp <- squares_temp[apply(squares_temp, 1, function(x) all(x>=1 & x<=n)),]
    squares <- rbind(squares, squares_temp)
  }
}

squares2 <- squares %>% mutate(c1=(x1-1)*n+y1, c2=(x2-1)*n+y2, c3=(x3-1)*n+y3, c4=(x4-1)*n+y4)
for (i in 1:n^2){
  squares2[[paste0("p",i)]] <- ifelse(apply(squares2[,c("c1","c2","c3","c4")], 1, function(x) any(x==i)),1,0)
}
squares2 <- as.matrix(squares2[,paste0("p",1:n^2)])

n_squares <- nrow(squares2)
n_squares
```
<pre>105</pre>
``` R
# make linear program
temp <- matrix(0, ncol=n_squares*2, nrow=n_squares*2)
diag(temp) <- rep(c(1,-1), each=n_squares)

A <- cbind(rbind(1, squares2, squares2), rbind(0, temp))
b <- c(ceiling(n^2/2), rep(c(1,3), each=n_squares))
dir <- c("=", rep(c(">=","<="), each=n_squares))

c <- rep(1, ncol(A))
c[1:n^2] <- 0

program <- lp(direction="min", objective.in=c, const.mat=A, const.dir=dir, const.rhs=b, all.bin=TRUE)
program
```
<pre>Success: the objective function is 0</pre>
``` R
# plot solution
solution <- data.frame(point=1:n^2, player=ifelse(head(program$solution,n^2)==1,"P1","P2"))
solution <- mutate(solution, x=ceiling(point/n), y=ifelse(point%%n==0,n,point%%n))

squares_made <- tail(program$solution, -n^2)
squares_made <- which(head(squares_made, n_squares)==1 | tail(squares_made, -n_squares)==1)

squares_made <- squares %>%
  mutate(s=row_number(), x5=x1, y5=y1) %>%
  gather(dimension, value, -s) %>%
  rowwise() %>% do(data.frame(s=.$s, value=.$value, dimension = paste(strsplit(.$dimension, "")[[1]], collapse="_"), stringsAsFactors=FALSE)) %>%
  ungroup() %>% separate(dimension, c("xy","order")) %>%
  group_by(s, order) %>% spread(xy, value) %>%
  filter(s %in% squares_made)

ggplot() +
  geom_tile(data=solution, aes(x=x,y=y,fill=player)) +
  geom_path(data=squares_made, aes(x=x,y=y,group=s))
```
