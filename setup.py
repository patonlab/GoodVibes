from setuptools import setup
setup(
  name = 'goodvibes',
  packages = ['goodvibes'],
  version = '2.0.3',
  description = 'Calculates quasi-harmonic free energies from Gaussian output files with temperature and haptic corrections',
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
