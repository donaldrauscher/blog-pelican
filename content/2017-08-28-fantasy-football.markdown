Title: Add Some Game Theory to Your Fantasy Football Draft
Date: 2017-08-31
Tags: fantasy_football, game_theory, spreadsheet_modeling
Slug: fantasy-football
Resources: plotly

Does your projected starting quarterback have an [18-42](https://www.pro-football-reference.com/players/M/McCoJo01.htm) starting record?  Did your team decide to complement its [crazy wide receiver](https://www.sbnation.com/2017/7/27/16053650/odell-beckham-jr-highest-paid-player-nfl) with...[another crazy wide receiver](http://bleacherreport.com/articles/2685133-brandon-marshall-comments-on-jets-season-and-locker-room-tension)?  Did your team sign Mike Glennon, who has a [5-13](https://www.pro-football-reference.com/players/G/GlenMi00.htm) starting record, to a $43.5M deal because he's...tall, then trade the #3 pick, a third round pick (#67), a fourth round pick (#111) and a 2018 third round pick TO MOVE UP 1 SPOT to draft Mitch Trubisky, who probably would have been available at #3 anyways?  Did your team maintain its SB-winning core, add a top 10 wide receiver, add smoking Jay Cutler to the division for 2 easy wins, and remain the [clear favorite to repeat](http://www.espn.com/chalk/story/_/id/18614149/nfl-latest-odds-all-32-teams-win-super-bowl-lii)?  It doesn't matter.  You'll always have fantasy football.  And when all your starters get injured, you'll have daily fantasy football (eh, [maybe not](https://www.bloomberg.com/news/articles/2015-09-10/you-aren-t-good-enough-to-win-money-playing-daily-fantasy-football)).

In any case, fantasy football is incredibly fun.  The most important part of course being the draft.  For quite a long time, the table stakes for draft strategy has been value-based drafting (VBD), pioneered by [Joe Bryant of Footballguys.com in 2001](http://www.footballguys.com/bryantvbd.htm). The core idea behind VBD is that a player’s value isn’t based on how many _absolute_ points he scores, but rather how many points he scores _relative_ to a "baseline" player at his position.  The most common strategy for establishing the baseline is to compare each player to the last starting player at that position (Value Over Last Starter of VOLS).  Let's say we have a 10 person league with standard rules (1 starting QB, 2 starting RB, 3 starting WR, 1 TE, 1 Flex, 1 DEF).   Quarterbacks are compared to the 10th best quarterback, RBs are compared to the 20th best RB, WRs are compared to the 30th best WR, etc.  

<div id="vbd_plot" style="width: 885; height: 400;"></div>

<script>
  Plotly.d3.csv("/data/fantasy_football/plot1.csv", function(data){

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
    var colors = Plotly.d3.scale.category20();
    var color_map = {QB: colors(0), RB: colors(1), WR: colors(2), TE: colors(3), D: colors(4), K: colors(5)};
    var colors2 = data2['color'].map(function(x){
      return color_map[x];
    });
    var trace = {x: data2['x'], y: data2['y'], text: data2['text'], marker: {color: colors2}, type: 'bar'};
    var data = [trace];

    var layout = {
      title: 'VBD vs. VBD Draft Position',
      xaxis: {title: 'VBD Draft Position'}, yaxis: {title: 'VBD', hoverformat: '.2f'}, height: 400, width: 885
    };

    Plotly.newPlot('vbd_plot', data, layout);
  });
</script>
Source: [NumberFire](http://www.numberfire.com/nfl/fantasy/fantasy-football-cheat-sheet/overall#)

So if VBD is so good, then why is our worst fear forgetting about the draft and letting autodraft (which uses a VBD-ordered player list) pick our team?  A few things:

* __There is not consensus on player value__ - There are lots of places to get player projects.  Certain players are really hard to project, like rookies.  In other cases, people may not feel that projections reflect injury risk.
* __It does not take into account where we are in the draft__ - The descent curve between the best player and the replacement player isn't smooth; it has kinks and plateaus.  Plus, it doesn't often make sense to take whatever person is at the top of the VBD heap.  Do you really want to draft another QB before you have 3 starting WR?
* __People aren't good at committing to a strategy__ - People get attached to specific players.  They jump the gun to draft players they like even if it isn't the logical choice.

Actual draft behavior looks more like this:

<div id="adp_plot" style="width: 885; height: 400;"></div>

<script>
  Plotly.d3.csv("/data/fantasy_football/plot2.csv", function(data){

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
    var colors = Plotly.d3.scale.category20();
    var color_map = {QB: colors(0), RB: colors(1), WR: colors(2), TE: colors(3), D: colors(4), K: colors(5)};
    var colors2 = data2['color'].map(function(x){
      return color_map[x];
    });
    var trace = {x: data2['x'], y: data2['y'], text: data2['text'], marker: {color: colors2}, type: 'bar'};
    var data = [trace];

    var layout = {
      title: 'VBD vs. Average Draft Position',
      xaxis: {title: 'Average Draft Position (ADP)'}, yaxis: {title: 'VBD', hoverformat: '.2f'}, height: 400, width: 885
    };

    Plotly.newPlot('adp_plot', data, layout);
  });
</script>

Source: [NumberFire](http://www.numberfire.com/nfl/fantasy/fantasy-football-cheat-sheet/overall#)

The peaks and troughs in the above graph represent opportunity!  We can derive value by anticipating what our opponents are going to do.  It makes more sense to have a strategy that adjusts dynamically based on where we are in the draft: what players we need, what players our opponents need, and what players are still remaining.  

# Improvement #1: Value Over Next Available (VONA)

Instead of using the last starter as our baseline, we can use the value of the next available player at that position.  If it's early in the draft, maybe I think 5 RBs and 5 WRs will come off the board by the time the draft snakes back to me.  I can use this information to determine which position will suffer the steepest drop off and draft accordingly.

# Improvement #2: Incorporating Average Draft Position (ADP)

Next, we can use ADP data to estimate how people are going to deviate from our projections.  For instance, T.Y. Hilton is #14 on my VBD draft board.  However, on average, he is being drafted #28.  Though I would be content with T.Y. Hilton at #14, maybe I instead draft someone else because I think that T.Y. Hilton will still be available by the time the draft snakes back to me.


I put together a small Google Spreadsheet to help me manage my drafts. You can check it out [here](https://docs.google.com/spreadsheets/d/1HYnnCKMtnFk3GpSvG1agGO_XuSXgpJGBcDVEKKBQx0w/edit?usp=sharing).  Just don't share it with anyone in my league.  Cheers!

<iframe width="885" height="500" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQ_uGw-RogvgcRGYfwu8TouoWkRqf0nH6rOyKM3aEYlYF5eXNlwFLycDIQ-9OcR7-8gwkXJVIRcWMcf/pubhtml?widget=true&amp;headers=false"></iframe>
