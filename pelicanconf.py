#!/usr/bin/env python

from __future__ import unicode_literals
from markdown import Markdown
from pelican.readers import ensure_metadata_list

AUTHOR = 'Donald Rauscher'
SITEURL = 'http://donaldrauscher.com'
SITENAME = 'DonaldRauscher.com'
SITESUBTITLE = 'A Blog About D4T4 & M47H'
SITEDESCRIPTION = 'I manage the D4T4 & Analytics group at [AbleTo](https://www.ableto.com/), a healthcare startup based in NYC.  I studed Operations Research & Industrial Engineering at Cornell and received my MBA from the University of Michigan.'

FEED_DOMAIN = SITEURL
FEED_ATOM = 'feeds/atom.xml'
FEED_RSS = 'feeds/rss.xml'

LINKEDIN_USERNAME = 'donaldrauscher'
GITHUB_USERNAME = 'donaldrauscher'

GOOGLE_ANALYTICS = 'UA-89570331-1'

TIMEZONE = 'America/Chicago'
DEFAULT_LANG = 'en'

PATH = 'content'
THEME = 'theme'
STATIC_PATHS = ['images']

DEFAULT_PAGINATION = 3
PAGINATION_WINDOW = 4

PLUGIN_PATHS = ['plugins']
PLUGINS = ['assets']

RELATIVE_URLS = True
DELETE_OUTPUT_DIRECTORY = True

RESOURCES = ['katex', 'plotly', 'jquery']

markdown = Markdown(extensions=['markdown.extensions.extra'])

JINJA_FILTERS = {
    'markdownify': lambda x: markdown.convert(x),
    'date_format': lambda x: x.strftime('%d %B &#8217;%y'),
    'ensure_metadata_list': ensure_metadata_list
}

JINJA_ENVIRONMENT = {
    'extensions': ['jinja2.ext.loopcontrols']
}