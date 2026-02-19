"""Pytest path setup for sprint02."""
import sys, os
# sprint01 goes in first, sprint02 overrides it (position 0)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sprint01'))
sys.path.insert(0, os.path.dirname(__file__))
