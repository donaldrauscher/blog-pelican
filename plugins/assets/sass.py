import os
from webassets.filter import register_filter
from webassets.filter.sass import Sass as SassBase

# sass filter with options matching v1.3.2
# https://github.com/sass/dart-sass/releases/tag/1.3.2
class Sass(SassBase):

    name = 'sass'
    options = {
        'binary': 'SASS_BIN',
        'debug_info': 'SASS_DEBUG_INFO',
        'as_output': 'SASS_AS_OUTPUT',
        'load_paths': 'SASS_LOAD_PATHS',
        'style': 'SASS_STYLE'
    }
    max_debug_level = None

    def _apply_sass(self, _in, out, cd=None):
        # Switch to source file directory if asked, so that this directory
        # is by default on the load path. We could pass it via -I, but then
        # files in the (undefined) wd could shadow the correct files.
        orig_cwd = os.getcwd()
        child_cwd = orig_cwd
        if cd:
            child_cwd = cd

        args = [self.binary or 'sass',
                '--stdin',
                '--style', self.style or 'expanded']
        if (self.ctx.environment.debug if self.debug_info is None else self.debug_info):
            args.append('--trace')
            args.append('--no-quiet')
        for path in self.load_paths or []:
            if os.path.isabs(path):
                abs_path = path
            else:
                abs_path = self.resolve_path(path)
            args.extend(['-I', abs_path])

        return self.subprocess(args, out, _in, cwd=child_cwd)

register_filter(Sass)
