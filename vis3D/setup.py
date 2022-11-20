'''
Following info from: https://stackoverflow.com/a/50468400
Install with
pip install -e </path/to/folder/containing/setup.py>
-e stands for editable,  so it will then create a link from the site-packages
directory to the directory in which the code lives, meaning the latest version
will be used without need to reinstall.

TODO: requires PyQt5, tifffile, Pillow
'''
from setuptools import setup

setup(
    name='vis3D',
    version='0.0.1',
    entry_points={
        'console_scripts': [
            'vis3D=vis3D:main'
        ]
    }
)
