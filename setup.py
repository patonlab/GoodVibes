from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'goodvibes',
  packages = ['goodvibes'],
  version = '2.0.3',
  description = 'Calculates quasi-harmonic free energies from Gaussian output files with temperature and haptic corrections',
  long_description=long_description,
  long_description_content_type='text/markdown'
  author = 'Robert Paton',
  author_email = 'robert.paton@colostate.edu',
  url = 'https://github.com/bobbypaton/goodvibes',
  download_url = 'https://github.com/bobbypaton/GoodVibes/archive/2.0.3.zip',
  keywords = ['compchem', 'thermochemistry', 'gaussian', 'vibrational-entropies', 'temperature'],
  classifiers = [],
  install_requires=["numpy", ],
  python_requires='>=2.6',
  include_package_data=True,
)
