""" Setup Python path """
import sys
import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(-1, path)

# Add project root (contains `near` module)
this_dir = os.path.dirname(__file__)
add_path(this_dir)

print("Added MultiNeAR project root to Python path.")
