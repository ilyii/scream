
class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, attr):
        if attr in self:
            del self[attr]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __getitem__(self, key):
        return self.get(key, None)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)