
from distutils.core import setup

import os
import sys

if sys.version_info[0] < 3:
    with open('README.rst') as f:
        long_description = f.read()
else:
    with open('README.rst', encoding='utf-8') as f:
        long_description = f.read()

setup(
  name = 'STATFOX',         # How you named your package folder (MyLib)
  packages = ['STATFOX'],   # Chose the same as "name"
  version = '0.24',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Advanced statistical functions',   # Give a short description about your library
  long_description = long_description,
  author = 'Devin D. Whitten, MS',                   # Type in your name
  author_email = 'devin.d.whitten@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/user/reponame',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/DevinWhitten/STATFOX',    # I explain this later on
  keywords = ['Python', 'Statistics', 'Data Science', 'Machine Learning'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pandas',
          'numpy',
          'scipy',
          'statsmodels'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
