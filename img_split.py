# Randomlysplit the set of images into a train and test set

from pathlib import Path
import random

def img_splt(path, train_size = 49, seed=None):
    folder = Path(path)
    names = sorted([p.name for p in folder.iterdir() if p.is_file()])
    rng = random.Random(seed) 
    random_names_train = rng.sample(names, train_size) # gets 49 random image names
    random_names_test = [n for n in names if n not in random_names_train] # gets the rest of the image names
    return random_names_train, random_names_test

