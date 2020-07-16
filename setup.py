from setuptools import setup
import io

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'goodvibes',
  packages = ['goodvibes'],
  version = '3.0.0',
  description = 'A python program to compute corrections to thermochemical data from frequency calculations',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Paton Research Group',
  author_email = 'robert.paton@colostate.edu',
  url = 'https://github.com/bobbypaton/goodvibes',
  download_url = 'https://github.com/bobbypaton/GoodVibes/archive/v3.0.0.zip',
  keywords = ['compchem', 'thermochemistry', 'gaussian', 'vibrational-entropies', 'temperature'],
  classifiers = [
    'Intended Audience :: Chemists',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',],
  install_requires=["numpy", ],
  python_requires='>=2.6',
  include_package_data=True,
)
