Title: 538 Riddler: Allison, Bob, and the Technicolor Dream Map
Date: 2016-11-13
Tags: 538, fivethirtyeight, riddler
Slug: map-game-riddler
Resources: jquery

<svg id="map-game-riddler" style="display:block; margin-left:auto; margin-right:auto;" width="580" height="400" xmlns="http://www.w3.org/2000/svg">
  <g>
    <ellipse stroke="#000000" ry="172" rx="239" cy="202" cx="285" fill="#ffffaa"/>
    <text x="285" y="65" dy="0.3em" text-anchor="middle" font-size="24">9</text>
  </g>
  <g>
    <ellipse stroke="#000000" ry="93" rx="121" cy="171" cx="354" fill="#ffaaff"/>
    <text x="354" y="130" dy="0.3em" text-anchor="middle" font-size="24">8</text>
  </g>
  <g>
    <ellipse stroke="#000000" ry="73" rx="86" cy="181" cx="187" fill="#ffd4aa"/>
    <text x="165" y="145" dy="0.3em" text-anchor="middle" font-size="24">7</text>
  </g>
  <g>
    <ellipse stroke="#000000" ry="39" rx="51" cy="191" cx="366" fill="#56aaff"/>
    <text x="366" y="191" dy="0.3em" text-anchor="middle" font-size="24">6</text>
  </g>
  <g>
    <ellipse stroke="#000000" ry="39" rx="51" cy="191" cx="210" fill="#ffaaaa"/>
    <text x="210" y="191" dy="0.3em" text-anchor="middle" font-size="24">5</text>
  </g>
  <g>
    <ellipse stroke="#000000" ry="39" rx="51" cy="290" cx="285" fill="#56ffaa"/>
    <text x="285" y="290" dy="0.3em" text-anchor="middle" font-size="24">4</text>
  </g>
  <g>
    <ellipse stroke="#000000" ry="39" rx="51" cy="191" cx="285" fill="#56ffaa"/>
    <text x="285" y="191" dy="0.3em" text-anchor="middle" font-size="24">3</text>
  </g>
  <g>
    <ellipse stroke="#000000" ry="39" rx="124" cy="240" cx="374" fill="#ffaaaa"/>
    <text x="374" y="240" dy="0.3em" text-anchor="middle" font-size="24">2</text>
  </g>
  <g>
    <ellipse stroke="#000000" ry="39" rx="124" cy="240" cx="197" fill="#56aaff"/>
    <text x="197" y="240" dy="0.3em" text-anchor="middle" font-size="24">1</text>
  </g>
</svg>

<a class="animate">Animate</a>

This week's Riddler was very interesting!  I'll start with my big ah-ha: any shape touching N shapes will "bury" at least N-2 shape(s).  Take a simple example where we have 3 touching circles, 3 different colors.  We can't draw a 4th circle whose area borders each of these 3 circles without fully concealing one of the 3 circles.  Meaning we won't have 4 exposed circles when we're done and thus won't be able to get to a 5th color (Bob can use the color of the circle that we've concealed).  It therefore follows that Allison will have to deliberately bury some shapes and therefore cannot win the game outright in 6 moves.

My best solution allows Allison to win in 9 moves.  I started by drawing a cluster of 6 circles with 3 colors, each color used twice.  Allison's 7th circle (4th color) buries 1 circle, Allison's 8th circle (5th color) buries 2 circles, and Allison's 9th circle (6th color) is simply a big circle encompassing the entire graph.  So we need to draw 3 extra circles and 9 circles total to force Bob to use 6 different colors.

<script type="text/javascript">
  $(document).ready(function(){  

    svg = $("svg#map-game-riddler");
    elements = svg.find("g");

    function loop(el){
      el.show(0);
      next_el = el.prev();
      if (next_el.length > 0){
        setTimeout(function(){ loop(next_el); }, 750);         
      }
    }

    svg.parent().next().find("a.animate").click(function(){
      elements.hide(0);
      loop(elements.last());      
    });

  });
</script>
