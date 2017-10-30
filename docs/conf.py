import sys
import pkg_resources

sys.path.insert(0, '_base')
from conf import *

autoapi_modules = {
    'cxflow': {
        # 'override': True,
        # 'output': 'auto',
        'prune': True
    }
}

# General information about the project.
project = 'cxflow'
copyright = '2017, Cognexa Solutions s.r.o.'
author = 'Petr Belohlavek, Adam Blazek, Filip Matzner'

# The short X.Y version.
version = '.'.join(pkg_resources.get_distribution("cxflow").version.split('.')[:2])
# The full version, including alpha/beta/rc tags.
release = pkg_resources.get_distribution("cxflow").version

html_context.update(analytics_id="UA-108931454-1")

html_theme_options.update({
    # Navigation bar title. (Default: ``project`` value)
    'navbar_title': "cxflow",

    # Tab name for entire site. (Default: "Site")
    'navbar_site_name': "Pages",

    # A list of tuples containing pages or urls to link to.
    'navbar_links': [
        ("Getting Started", "getting_started"),
        ("Tutorial", "tutorial"),
        ("Advanced", "advanced/index"),
        ("CLI Reference", "cli"),
        ("API Reference", "cxflow/index"),
    ],

    # HTML navbar class (Default: "navbar") to attach to <div> element.
    # For black navbar, do "navbar navbar-inverse"
    'navbar_class': "navbar navbar-inverse",
})

