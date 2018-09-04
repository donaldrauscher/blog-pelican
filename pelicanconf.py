#!/usr/bin/env python

from __future__ import unicode_literals
from markdown import markdown
from pelican.readers import ensure_metadata_list

import sys
import os
sys.path.append(os.path.dirname(__file__))

from extensions import ScriptExtension

ENV = os.getenv('ENV', 'prod')

AUTHOR = 'Donald Rauscher'
SITEURL = 'http://www.donaldrauscher.com'
SITENAME = 'DonaldRauscher.com'
SITESUBTITLE = 'A Blog About D4T4 & M47H'
SITEDESCRIPTION = 'I am a Senior Data Scientist at [MOBE](https://www.mobeforlife.com/), a healthcare startup based in Minneapolis.  I studied Operations Research & Industrial Engineering at Cornell and received my MBA from the University of Michigan.'

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
STATIC_PATHS = ['images', 'data']

DEFAULT_PAGINATION = 3
PAGINATION_WINDOW = 4

PLUGIN_PATHS = ['plugins']
PLUGINS = ['assets', 'load_csv']

RELATIVE_URLS = True
DELETE_OUTPUT_DIRECTORY = True

RESOURCES = ['katex', 'plotly', 'jquery']

JINJA_FILTERS = {
    'markdownify': lambda x: markdown(x, extensions=['markdown.extensions.extra']),
    'date_format': lambda x: x.strftime('%d %B &#8217;%y'),
    'ensure_metadata_list': ensure_metadata_list
}

JINJA_ENVIRONMENT = {
    'extensions': ['jinja2.ext.loopcontrols']
}

MARKDOWN = {
    'extensions': [ScriptExtension()],
    'extension_configs': {
        'markdown.extensions.codehilite': {'css_class': 'highlight'},
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {}
    },
    'output_format': 'html5',
}