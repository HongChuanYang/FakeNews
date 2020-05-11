import numpy as np


class Iterator(object):
    """
    数据迭代器
    """

    def __init__(self, x, y=None):
        self.x = x
        self.sample_num = len(self.x)
        self.y = y

    def next(self, batch_size, shuffle=True):
        if shuffle:
            np.random.shuffle(self.x)
        l = 0
        while l < self.sample_num:
            r = min(l + batch_size, self.sample_num)
            batch_size = r - l
            x_part = self.x[l:r]
            if self.y is not None:
                y_part = self.y[l:r]
                l += batch_size
                yield x_part, y_part
            else:
                l += batch_size
                yield x_part
