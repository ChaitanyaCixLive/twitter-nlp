# Shishir Tandale
from collections import MutableMapping

##TODO implement many-to-many mapping and reversibility

class Symmetric_Dictionary(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._dict = dict()
        self.__dict = dict()
        self.update(dict(*args, **kwargs))
        self.i = 0
    def __deepcopy__(self, dict):
        for k,v in dict:
            self[k] = v
    @property
    def keys(self):
        return list(self._dict.keys())
    @staticmethod
    def copy(d):
        sd = Symmetric_Dictionary()
        sd._dict, sd.__dict = d._dict, d.__dict
        return sd
    def __len__(self):
        return len(self.keys)
    def __contains__(self, key):
        return key in self._dict
    def __getitem__(self, key):
        return self._dict[key]
    def __setitem__(self, key, value):
        self._dict[key]=value
        self.__dict[value]=key
    def __delitem__(self, key):
        del self._dict[key]
        self.keys.discard(key)
    def __iter__(self):
        return self
    def __next__(self):
        if not self.i+1 < len(self.keys):
            raise StopIteration
        self.i += 1
        k = self.keys[self.i]
        return k, self[k]
    def __invert__(self):
        _self = Symmetric_Dictionary()
        _self._dict = self.__dict
        _self.__dict = self._dict
        self.i = 0
        return _self
    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self])

if __name__ == "__main__":
    from numpy.random import randint
    from progress import Progress
    count = 10_000
    nums = range(5, count)
    sd = Symmetric_Dictionary()
    with Progress("Filling symdict", count, precision=2) as (u,_,_):
        for i, key in enumerate(nums):
            sd[key] = u(i)
    print("Reversing...")
    sd = ~sd
    print("Printing...")
    s = [f"{k}: {v}" for k,v in sd]
    print(s)
