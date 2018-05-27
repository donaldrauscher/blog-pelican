#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Template CSV loader
-------------------
Authored by Lucy Park <me@lucypark.kr>, 2015
Modified by Don Rauscher <donald.rauscher@gmail.com>, 2018
Released under the BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
"""

import re
import io
import csv
from pelican import signals


def csv_loader(filepath, header=True, delim=',', classes=[], table_tag=True):

    csv_string = ''
    with io.open(filepath, mode='r', encoding='utf-8') as f:
        linefeed = csv.reader(f, delimiter=delim)

        for i, row in enumerate(linefeed):
            if header and (i == 0):
                csv_string += '<thead><tr><th>%s</th></tr></thead>' % '</th><th>'.join(row)
            else:
                csv_string += '<tr><td>%s</td></tr>' % '</td><td>'.join(row)

    if table_tag:
        if classes:
            csv_string = '<table class="%s">%s</table>' % (' '.join(classes), csv_string)
        else:
            csv_string = '<table>%s</table>' % (csv_string)

    return csv_string


# A function to read through each page / post as it comes through from Pelican,
# find all instances of {% csv .*? %} code blocks, and convert into HTML
def load_csv(_, content):
    content._content = re.sub('{% (csv_loader\\(.*?\\)) %}', lambda x: eval(x.group(1)), content._content, flags=re.DOTALL)


def register():
    signals.article_generator_write_article.connect(load_csv)