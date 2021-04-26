# ref: https://stackoverflow.com/a/41177081/13095028
# contents of conftest.py
import sys
from os.path import abspath, dirname

package_path = abspath(dirname(dirname(__file__)))
sys.path.insert(0, package_path)
