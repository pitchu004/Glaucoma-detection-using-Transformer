import sys
import os

# Force Python to find all modules from this folder
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now run train.py's main function
from train import main
main()