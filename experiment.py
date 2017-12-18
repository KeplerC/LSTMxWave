import os
import numpy as np
import matplotlib.pyplot as plt
import utils.py
import fastgen.py #TODO: an experiment based on it 
from IPython.display import Audio

encoding = fastgen.encode(audio, 'model.ckpt-200000', sample_length)
