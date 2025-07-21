import numpy as np
from numpy import default_rng


rng = default_rng(seed=100)

ssp = [1, 0]

asp = [1, 0]

def epoch():
    for _ in range(100):
        a = rng.choice(asp)
        s = rng.choices(ssp)
        if a == s:
            tr += 1
    return tr

rl = np.array([epoch() for _ in range(250)])
print(rl[:10])
print(rl.mean())
