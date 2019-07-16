class ClosedInterval:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __repr__(self):
        return f"[{self.min},{self.max}]"

    def __contains__(self, value):
        if type(value) is type(self):
            return (self.min <= value.min) and (value.max <= self.max)
        else:
            return self.min <= value <= self.max

    def __eq__(self, value):
        if type(value) is type(self):
            return (self.min == value.min) and (self.max == value.max)
        else:
            return self.min <= value <= self.max

