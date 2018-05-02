"""
Created on Wed Mar 2 11:41:00 2018

@author: jercas
"""
import os
import glob

def counter(path):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(path):
    return 0
  count = 0
  for r, dirs, files in os.walk(path):
    for dr in dirs:
      count += len(glob.glob(os.path.join(r, dr + "/*")))
  return count