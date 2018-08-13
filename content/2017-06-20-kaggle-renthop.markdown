Title: Dealing with High Cardinality Categorical Variables & Other Learnings from the Kaggle Renthop Challenge
Date: 2017-06-20
Tags: kaggle, r, predictive_modeling
Slug: kaggle-renthop
Resources: katex

I recently completed the [Kaggle Renthop Competition](kaggle.com/c/two-sigma-connect-rental-listing-inquiries).  I had a lot of fun with it.  One of my biggest takeaways from the competition was developing a transferable approach for dealing with high cardinality categorical variables like ZIP codes, NAICS industry codes, ICD10 diagnosis codes etc.  I developed a simple Bayesian approach to encode these variables as probabilities which can then be included as features in a model.  More on this in a few.  

### My Kaggle Model

A summary of my final solution:

* My best solution was a model stack with 2 XGB models and 2 GLMNET models.
* I rather lazily tuned my XGB models by looking at the parameters in high-performing public notebooks: eta=0.02, min_child_weight=3, max_depth=6, colsample_bytree=0.4, subsample=1, gamma=0, nrounds = 2000.  I also implemented early stopping to prevent overfitting.  For GLMNET, I used Lasso (alpha = 1); a safe choice to limit model complexity.  For both models, I used 5-fold cross validation.
* I blended these models using a simple linear combination, which I tuned on cross-validated, out-of-sample predictions.
* As already noted, I spent _a lot_ of time dealing with the two high cardinality categorical variables: building_id and manager_id.  I included one-hots for high-frequency levels.  I also encoded these variables as probabilities, which I fed into the model.  More on this later.
* Rather than use geographic clustering, I used the latitude/longitude variables to place each listing in a specific neighborhood.  Neighborhood shapes from [Zillow](https://www.zillow.com/howto/api/neighborhood-boundaries.htm).
* An awesome estimate of [price per square foot](https://www.kaggle.com/arnaldcat/a-proxy-for-sqft-and-the-interest-on-1-2-baths) from fellow contributor Darnal
* I extract n-grams from the listing descriptions with high odds ratios, then used PCA to combine into some uncorrelated, keyword-based features.  

My score (mlogloss) was 0.53243, which was ~650 out of ~2500 (~25th percentile).  Not a great showing, but I can live with it.  Or rather I _must_ live with it because the competition is over (thank goodness).  

I think I could have gotten some improvements by (1) doing more feature engineering and/or (2) using a more comprehensive model stack.  My favorite solution writeup was from [9th place James Trotman](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32146).  In it, he details some very clever features that he used (e.g. descriptive stats on corpus of manager descriptions, avg. time between listings for each manager, total photo size and area, number of listing "mistakes") as well as how he constructed his 3-level model stack.  Several other top solutions (e.g. [2nd](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32148), [3rd](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/32123)) cited using a meta-modeling framework called [StackNet](https://github.com/kaz-Anova/StackNet).  

### High Cardinality Categorical Variables

One of the things that I quickly noticed was that we had two challenging high cardinality categorical variables: building_id and manager_id.  Across test (N = 74,659) and train (N = 49,352), these two variables had 11,635 and 4,399 unique values respectively.  With so many levels, one-hotting is not a viable solution.  A more elegant approach is to instead transform the original categorical values into probabilities via a simple Bayesian model:
<div class="equation" data-expr="X_{i} \rightarrow S_{i} \approx P(Y|X = X_{i})"></div>

The question: how do we estimate these probabilities?  A simple approach might be to just take the average interest level (excluding the point itself from the calculation). Famous Kaggler Owen Zhang is a proponent of this approach, which he calls "leave-one-out encoding" [here](https://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions).  My main objection to this approach is that there will obviously be a lot of variance for low-frequency categories.  Intuitively, it makes more sense to do a weighted average of the posterior distribution (the average of the observed points) and the prior distribution (the average for the overall population) and have the posterior/prior weight depend on sample size.  For categories with more data points, we should weight the posterior distribution more.  For categories with fewer data points, we should weight the prior distribution more.       

I instead thought of each category as being a binomial distribution for which we don't know the probability of success (q) for each trial, which is known as a beta-binomial distribution.  The prior distribution for the probability of success and the posterior distribution given s successes and f failures (success = high or medium interest) are [conjugate beta distributions](https://en.wikipedia.org/wiki/Conjugate_prior#Example):
<div class="equation" data-expr="Q \sim Beta(a,b)"></div>
<div class="equation" data-expr="Q|s,f \sim Beta(a+s,b+f)"></div>
<div class="equation" data-expr="E[Q|s,f] = \frac{a+s}{a+b+s+f}"></div>

So simple!  And it's doing what we want it to do intuitively, which is assign more weight to the posterior distribution when there are more data points and more credit to the prior distribution when there are fewer data points.  Though we still need to estimate our prior distribution hyperparameters, alpha and beta.  For this, I just used MLE.  To prevent our beta binomial PMF from returning unmanageably small values, I capped the number of trials at 100.  Here is a nifty little function that I wrote for calculating these:  

``` R
library(dplyr)
library(lazyeval)

# fit beta distribution with mle
dbetabinom <- function(k, n, a, b) {
  n2 <- ifelse(n > 100, 100, n)
  k2 <- round(k * n2 / n)
  beta(k2 + a, n2 - k2 + b) / beta(a, b)
}

betabinom_ll <- function(k, n, par) {
  sum(-log(dbetabinom(k, n, par[1], par[2])))
}

beta_mle <- function(...){
  par <- optim(par = c(1,1), fn=betabinom_ll, method="L-BFGS-B", lower=c(0.5,0.5), upper=c(500,500), ...)$par
  return(data.frame(a = par[1], b = par[2]))
}

# function for probabilizing high cardinality categorical variable
probabilize_high_card_cat <- function(df, y, x, seg = 1, loo = 1){

  # set x, y, and seg
  df$y <- f_eval(f_capture(y), df)
  df$x <- f_eval(f_capture(x), df)
  df$seg <- f_eval(f_capture(seg), df)

  # determine prior for each segment
  dist <- df %>%
    filter(!is.na(y)) %>% # df includes both test and train
    group_by(seg, x) %>% summarise(k = sum(y), n = n()) %>% ungroup() %>%
    group_by(seg) %>% do(beta_mle(k = .$k, n = .$n)) %>% ungroup()

  # calculate posterior probabilities
  df <- df %>%
    left_join(dist, by = c("seg")) %>%
    group_by(x) %>% mutate(
      k = sum(y, na.rm = TRUE) - loo * ifelse(!is.na(y), y, 0),
      n = sum(!is.na(y), na.rm = TRUE) - loo * as.integer(!is.na(y))
    ) %>% ungroup() %>%
    mutate(y2 = (a + k) / (a + b + n))

  return(df$y2)

}

# example
df$building_interest_h <- probabilize_high_card_cat(df, ifelse(interest_level == "high", 1, 0), building_id, 1, 1)

```

Overall, great challenge.  You can find my entire code base on my GH [here](https://github.com/donaldrauscher/kaggle-renthop).  Cheers!
