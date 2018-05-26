Title: Post-Specific Resources in Jekyll
Date: 2017-03-11
Tags: jekyll, blog, liquid_templating
Slug: jekyll-resources

I set up this blog on Jekyll last year (largely just to have a repository for my 538 Riddler solutions haha).  I really like Jekyll because it is simple, supports Markdown and Liquid templating, can be [hosted for free](https://help.github.com/articles/using-jekyll-as-a-static-site-generator-with-github-pages/) on Github. However, I did recently notice that my site was getting a little bloated. Pages were taking a long time to load because I was _always_ loading a ton of Javascript resources (jQuery, Katex, d3, and plotly.js).  However, not every post uses all of these, obviously.  I recently implemented some logic that only loads the required resources for each page/post.  Here's a quick overview.

## 1. Add Include For Each Resource

First, we need to make an include for each resource.  For some resources like [plotly](https://github.com/donaldrauscher/blog/blob/gh-pages/_includes/plotly.html), this is just a tag referencing the JS library and/or CSS stylesheets.
For other resources like [Katex](https://github.com/donaldrauscher/blog/blob/gh-pages/_includes/katex.html), it includes some additional scripting to implement the resource as well.

plotly.html
``` html
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
```

## 2. Add Resources Array to Post YAML Front Matter

For each post, I added an array called 'resources' to the YAML front matter.  This array holds a list of all the resources that the post requires.

``` markdown
---
layout: post
title: 'Test'
date:   1970-01-01
tags: [lorem, ipsum]
permalink: /test
resources: [plotly]
---
...
```

## 3. Add Include for Loading Resource If Required

Next, I made another include which adds a resource if it is required. For paginator pages, we need to loop through each post on the page. For specific post pages, we obviously just check that individual page.

{% raw %}
add_resource.html
``` html
{% assign add_resource = false %}
{% if paginator.posts %}
  {% for post in paginator.posts %}
    {% if post.resources contains resource %}
      {% assign add_resource = true %}
      {% break %}
    {% endif %}
  {% endfor %}
{% endif %}
{% if page.resources contains resource %}
  {% assign add_resource = true %}
{% endif %}
{% if add_resource %}
  {% include {{ resource | append:'.html' }} %}
{% endif %}
```
{% endraw %}

## 4. Add Logic to Check Each Resource

Finally, I added a little more Liquid logic to the head tag (which is in a head.html include in my case) to check each resource.  In your \_config.yml file, add an array called 'resources' which holds a list of all available resources.  Names should match the file names of the includes for each resource.

\_config.yml
``` yaml
...
resources: [katex, plotly, jquery, d3]
```

{% raw %}
head.html
``` markdown
<head>
  ...
  {% for resource in site.resources %}
    {% include add_resource.html resource=resource %}
  {% endfor %}
</head>
```
{% endraw %}

And that's it!  You can see this implemented in my Jekyll site [here](https://github.com/donaldrauscher/blog).  Enjoy.
