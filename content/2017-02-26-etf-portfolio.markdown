Title: Building an Optimal Portfolio of ETFs
Date: 2017-02-26
Tags: investing, portfolio_optimization, etfs, luigi
Slug: etf-portfolio
Resources: plotly

Exchange traded funds (ETFs) have taken the market by storm.  Over the last few years, we’ve seen a [huge shift](http://www.icifactbook.org/ch3/16_fb_ch3) in assets towards passive investing, motivated by ETF’s low fee structure and the revelation that most active managers cannot beat their benchmark.  This shouldn't be terribly surprising.  It only takes [simple arithmetic](https://web.stanford.edu/~wfsharpe/art/active/active.htm) to demonstrate that active management is a zero-sum before fees and a losing proposition after fees. Even world reknown active investor Warren Buffet has [suggested](http://www.berkshirehathaway.com/letters/2013ltr.pdf) a simple portfolio of inexpensive index funds for the heirs to his own fortune.

However, just because ETFs are themselves portfolios doesn't mean that we don't need to think about portfolio optimization.  There are well-known factors which earn market-adjusted positive returns (e.g. small > large, value > growth, high momentum > low momentum).  Can we build a portfolio of ETFs that takes advantage of these tilts?

## Thesis

A well-constructed portfolio of ETFs can give you similar return with less risk and less market exposure than a single, market-mirroring ETF.  Benchmarks for my analysis are the SPY and the MDY.

## Methodology

1. **“Investible” ETF Universe** – [ETF.com](http://www.etf.com/) maintains a great database of over ~1800 ETFs.  From this, I restricted my universe to funds that are US-based, are non-sector focused, are highly liquid (334 ETFs) and have at least 5 years of returns (211/334).
2. **Factor Modeling** – I regressed ETF returns (Yahoo Finance) against known factors including market returns, size, value, momentum, profitability, investment, variance, net shares issued, and accruals.  Factor data generously provided by [Kenneth French](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) (the CRSP database license is a little outside my price range). I used L1 regularization to prevent overfitting factor loadings.  From this, I was able to calculate each ETF’s expected return/variance and covariances with other ETFs.
3. **Portfolio Optimization** – I made portfolios with target results of 6%, 8%, 10% and 12%.  I used a [quadratic optimizer](https://en.wikipedia.org/wiki/Quadratic_programming) to minimize variance within constraints.  Minimum asset weights of 2.5%.  Also, to ensure that a variety of factors were driving returns, I required positive tilts on size, value, momentum, profitability, and investment factors.

## Results

Overall, all 4 of my portfolios generated higher Sharpe ratios (1.6-1.8 vs. 1.1-1.3) and lower draw-downs (2%-5% vs. 9%-13%) but lower returns than the SPY and MDY, which both have returned a staggering 13.2% annually over the last 5 years.  This isn't totally surprising.  The model, to the extent that it can, tries to balance systematic market risk and factor risk, resulting in lower betas.  Normally, this would be a good thing, except our 5 year test period sits squarely in the middle of the [second longest bull market ever](http://seekingalpha.com/article/3987722-2nd-longest-bull-market-history).  I think we can expect more modest returns for the benchmarks moving forward; my model projects 8.1% and 8.9% for the SPY and MDY respectively.  Expected Sharpe ratios for my portfolios (1.4-1.5) are nearly 3x higher than the benchmarks (0.5)!  My target return 12% portfolio (TR12) has a beta of just 0.51 and positive loadings on size, profitability, investment, momentum, and accrual factors.  

Portfolio Performance Results:

<table class="pretty">
  <thead>
    <tr><th></th><th colspan="3">Expected</th><th colspan="4">Actual (Last 5 Years)</th></tr>
    <tr><th>Portfolio</th><th>Return</th><th>SD</th><th>Sharpe</th><th>Return</th><th>SD</th><th>Sharpe</th><th>Max Draw Down</th></tr>
  </thead>
  <tbody>
  {% csv_loader('content/data/etf_portfolio/etf_portfolio_summary.csv', header=False, table_tag=False) %}
  </tbody>
</table>

Portfolio Asset Weights:

{% csv_loader('content/data/etf_portfolio/etf_portfolio_weights.csv', classes=['pretty']) %}

All of my code is posted on my [GitHub](https://github.com/donaldrauscher/etf-portfolio).  The universe of ETFs analyzed and their tilts can be downloaded <a href="/data/etf_portfolio/etf_db.csv" target="_blank">here</a> and <a href="/data/etf_portfolio/etf_tilts.csv" target="_blank">here</a>.  Cheers!

<div id="risk_vs_return" style="width: 800; height: 500;"></div>
<script>

  Plotly.d3.csv("/data/etf_portfolio/viz1_1.csv", function(data1){
    Plotly.d3.csv("/data/etf_portfolio/viz1_2.csv", function(data2){

      // pull data from CSVs into arrays
      var x = [], y = [], size = [], opacity = [], label = [], color = [];
      data1.forEach(function(d){
        x.push( parseFloat(d.X) );
        y.push( parseFloat(d.Y) );
        size.push( parseFloat(d.Size) );
        opacity.push( parseFloat(d.Opacity) );
        label.push( d.Label );
        color.push( parseFloat(d.Color) );
      });

      var x2 = [], y2 = [], label2 = [], color2 = [];
      data2.forEach(function(d){
        x2.push( parseFloat(d.X) );
        y2.push( parseFloat(d.Y) );
        label2.push( d.Label );
        color2.push( parseFloat(d.Color) );
      });

      // create lines for different sharpe ratios
      var max_x = Math.max.apply(Math, x), max_y = Math.max.apply(Math, y);
      sharpe1 = [(2*max_x>max_y)?max_y/2:max_x,(2*max_x>max_y)?max_y:2*max_x];
      sharpe2 = [(1*max_x>max_y)?max_y/1:max_x,(1*max_x>max_y)?max_y:1*max_x];
      sharpe3 = [(0.5*max_x>max_y)?max_y/0.5:max_x,(0.5*max_x>max_y)?max_y:0.5*max_x];

      // my colorscale
      mycolors = [[0, 'rgb(255,51,51)'], [0.25, 'rgb(255,51,51)'], [0.5, 'rgb(255,215,0)'], [1, 'rgb(0,153,76)']]

      var trace1 = {
        name: 'ETFs',
        x: x, y: y, text: label, mode: 'markers',
        marker: {
          color: color, cmin: 0, cmax: 2, colorscale: mycolors,
          size: size, opacity: opacity
        }
      };
      var trace2 = {
        name: 'ETF Portfolios',
        x: x2, y: y2, text: label2, mode: 'lines+markers',
        marker: {
          color: color2, cmin: 0, cmax: 2, colorscale: mycolors,
          symbol: 'star', size: 10
        },
        line: {
          color: 'lightgrey', dash: 'dot'
        }
      };
      var trace3 = {name: 'Sharpe=2', x: [0, sharpe1[0]], y: [0, sharpe1[1]], mode: 'lines', line: { color: mycolors[3][1], dash: 'dot' } };
      var trace4 = {name: 'Sharpe=1', x: [0, sharpe2[0]], y: [0, sharpe2[1]], mode: 'lines', line: { color: mycolors[2][1], dash: 'dot' } };
      var trace5 = {name: 'Sharpe=0.5', x: [0, sharpe3[0]], y: [0, sharpe3[1]], mode: 'lines', line: { color: mycolors[1][1], dash: 'dot' } };

      var traces = [trace1, trace2, trace3, trace4, trace5];

      var layout = {
        title: 'Risk vs. Return',
        xaxis: {title: 'Return Standard Deviation (%)'}, yaxis: {title: 'Expected Return (%)'},
        showlegend: true, height: 500, width: 800
      };

      Plotly.newPlot('risk_vs_return', traces, layout);
    });
  });
</script>

<div id="cumulative_returns" style="width: 800; height: 500;"></div>
<script>
  Plotly.d3.csv("/data/etf_portfolio/viz2.csv", function(data){

    // organize by column (d3.csv reads in as array of rows)
    data2 = {}
    columns = Object.keys(data[0]);
    columns.forEach(function(x){ data2[x] = []; });
    data.forEach(function(row){
      for (var x in row){
        data2[x].push(row[x]);
      }
    });

    // make the plot
    var traces = []
    for (var ticker in data2){
      if (ticker == 'Month'){
        continue;
      }
      var trace = {name: ticker, x: data2['Month'], y: data2[ticker], mode: 'lines'};
      traces.push(trace);
    }

    var layout = {
      title: 'Cumulative Returns',
      xaxis: {title: 'Month'}, yaxis: {title: 'Cumulative Return', hoverformat: '.2f'},
      showlegend: true, height: 500, width: 800
    };

    Plotly.newPlot('cumulative_returns', traces, layout);
  });
</script>

<div id="weights" style="width: 800; height: 350;"></div>
<script>
  Plotly.d3.csv("/data/etf_portfolio/viz3.csv", function(data){

    // organize by column (d3.csv reads in as array of rows)
    data2 = {};
    columns = Object.keys(data[0]);
    columns.forEach(function(x){ data2[x] = []; });
    data.forEach(function(row){
      for (var x in row){
        data2[x].push(row[x]);
      }
    });

    // make the plot
    var i=0, traces = [], annotations = [];
    for (var portfolio in data2){
      if (portfolio == 'Ticker'){
        continue;
      }

      i++;
      xdomain = [(i-1)/(columns.length-1)+0.01, i/(columns.length-1)-0.01];
      xdomainmid = (xdomain[0]+xdomain[1])/2;

      // make trace
      text = [];
      var trace = {
        name: portfolio, values: data2[portfolio], labels: data2.Ticker, domain: {x: xdomain},
        hoverinfo: 'label+percent', textposition: 'inside', hole: .4, type: 'pie'
      };
      traces.push(trace);

      // make annotation
      var annotation = {
        font: {size: 16}, showarrow: false, xanchor: 'center', text: portfolio,
        x: xdomainmid, y: 0.5
      };
      annotations.push(annotation);
    }

    var layout = {
      title: 'Portfolio Weights', annotations: annotations,
      showlegend: false, height: 350, width: 800
    };

    Plotly.newPlot('weights', traces, layout);
  });
</script>

<div id="tilts" style="width: 800; height: 500;"></div>
<script>
  Plotly.d3.csv("/data/etf_portfolio/viz4.csv", function(data){

    // make trace for each row
    var traces = []
    data.forEach(function(d){
      ticker = d.Ticker;
      delete d.Ticker;
      variables = Object.keys(d);
      values = Object.values(d);
      var trace = {name: ticker, x: variables, y: values, mode: 'lines+markers'};
      traces.push(trace);
    });

    var layout = {
      title: 'Factor Tilts',
      yaxis: {title: 'Coefficient', hoverformat: '.2f'},
      showlegend: true, height: 500, width: 800
    };

    Plotly.newPlot('tilts', traces, layout);
  });
</script>
