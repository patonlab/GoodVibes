import os
import sys

# If we are running from a wheel, add the wheel to sys.path
# This allows the usage python pip-*.whl/pip install pip-*.whl

if not __package__:
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, path)

from goodvibes import GoodVibes  # noqa

if __name__ == '__main__':
    sys.exit(GoodVibes.main())
