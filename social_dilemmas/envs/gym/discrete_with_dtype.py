from typing import Optional
from gym.spaces import Discrete


class DiscreteWithDType(Discrete):
    def __init__(self, n: int, dtype=int, seed: Optional[int] = None, start: int = 0):
        assert n >= 0
        self.n = int(n)
        self.start = int(start)
        # Skip Discrete __init__ on purpose, to avoid setting the wrong dtype
        super(Discrete, self).__init__((), dtype, seed)
