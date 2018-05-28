Title: Using the US Census API(s)
Date: 2016-11-10
Tags: api, census, data-scraping
Slug: census-api

The other day I was building a model and wanted to layer in some ZIP-level census data. Some quick Googling led me to [this .gov site](http://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml) where you can search a database of stock reports and data cuts.  However, I wasn't able to find what I was looking for.  Not sure if I was searching ineffectively (the interface is a little clunky) or my somewhat-specific request didn't align with any of the stock reports.

In either case, I stumbled upon a [post](https://blog.splitwise.com/2013/09/18/the-2010-us-census-population-by-zip-code-totally-free/) from the data folks at Splitwise (if you're not already a user, I highly recommend) which cued me into the US Census API(s).  An API!  It is not the most elegant or well-documented but is still very useful once you get the hang of it.  Each data source (e.g. 2010 Dicentennial Census) has its own API endpoint, and each endpoint has a list of geographies that you can pull data at and a list of variables that can be pulled.

A few useful notes:
+ You can apply for an API key [here](http://api.census.gov/data/key_signup.html)
+ Here's a [link](https://api.census.gov/data.html) to all the APIs available and their respective documentation
+ In some cases, you can't pull data at the level you want in a single API call.  For instance, for the [2010 Dicentennial Census](https://api.census.gov/data/2010/sf1/geography.html) data, you must specify a state when requesting ZIP-level data.  So you need to download a list of states, then loop through the states and download ZIP-level information for each one  (adding `&in=state:XX` into each call).  Example [here](https://github.com/donaldrauscher/census-api/blob/master/population.py).  For other APIs like the [American Community Survey](https://api.census.gov/data/2015/acs5/geography.html) data, this isn't necessary.
+ You may need to assemble ratio metrics by downloading the numerator and the denominator as separate variables. For instance, to get % of households in each ZIP that are renters using the American Community Survey API, I would download the numerator ([B07013_003E](https://api.census.gov/data/2009/acs5/variables.html#B07013_003E)) and the denominator ([B07013_001E](https://api.census.gov/data/2009/acs5/variables.html#B07013_001E)) and assemble the metric on the back end. Example [here](https://github.com/donaldrauscher/census-api/blob/master/socioeconomic.py).  This is actually kind of nice because it lets me decide what sample size I want to require to populate the metric.  

I posted a few code snippets which pull data by ZIP code on my [GH](https://github.com/donaldrauscher/census-api).  Cheers!
