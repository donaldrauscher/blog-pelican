from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from jsmin import jsmin
import re


# minifies inline JS so markdown doesn't break it
class ScriptPreprocessor(Preprocessor):

    @staticmethod
    def minify(x):
        return "<script{}>{}</script>".format(x.group(1), jsmin(x.group(2)))

    def run(self, lines):
        text = "\n".join(lines)
        text2 = re.sub("<script(.*?)>(.*?)<\/script>", self.minify, text, flags=re.DOTALL)
        return text2.split("\n")


class ScriptExtension(Extension):
    def extendMarkdown(self, md, md_globals):
        md.preprocessors.add('script', ScriptPreprocessor(md), "<normalize_whitespace")
