from loguru import logger
import sys

print(sys.path)
from skyrim import Skyrim

for module_name in sorted(sys.modules):
    print(module_name)

# # Print module names and their paths
# for name, module in sorted(sys.modules.items()):
#     if hasattr(module, "__file__"):
#         print(f"{name}: {module.__file__}")
#     else:
#         print(f"{name}: Built-in or dynamically created module")


logger.success("imports successful")

from skyrim.models import PanguWeather, GraphCast

logger.success("imports successful")


from skyrim.models import BaseModel

logger.success("imports successful")

from skyrim.libs.plotting import plot_wind_components

logger.success("imports successful")
