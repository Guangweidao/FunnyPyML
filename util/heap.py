# -*- coding: utf-8 -*-

def default_compare(a, b):
    return a > b


class Heap(object):
    def __init__(self, k=None, compare=default_compare):
        super(Heap, self).__init__()
        self._K = k
        self._compare = compare
        self._size = 0
        self._heap = ['#']

    def insert(self, x):
        if self._K is None:
            self._heap.append(x)
            self._size += 1
            self.up(self._size)
        else:
            if self._size < self._K:
                self._heap.append(x)
                self._size += 1
                self.up(self._size)
            else:
                # if you use max heap, then self._heap maintain the top k minimun value
                # heap type depends on compare type, default compare is max heap
                if not self._compare(x, self._heap[1]):
                    self._heap[1] = x
                    self.down(1)

    def up(self, index):
        if index == 1:
            return
        if self._compare(self._heap[index], self._heap[index / 2]):
            self._heap[index / 2], self._heap[index] = self._heap[index], self._heap[index / 2]
            self.up(index / 2)

    def down(self, index):
        if 2 * index > self._size:
            return
        if 2 * index < self._size:
            if self._compare(self._heap[2 * index], self._heap[2 * index + 1]):
                child_ix = 2 * index
            else:
                child_ix = 2 * index + 1
        else:
            child_ix = 2 * index
        if self._compare(self._heap[child_ix], self._heap[index]):
            self._heap[child_ix], self._heap[index] = self._heap[index], self._heap[child_ix]
            self.down(child_ix)

    def pop(self, index):
        assert 0 < index <= self._size, 'delete index is invalid.'
        self._heap[index], self._heap[self._size] = self._heap[self._size], self._heap[index]
        x = self._heap.pop()
        self._size -= 1
        self.down(index)
        return x

    def clear(self):
        self._heap = ['#']
        self._size = 0

    def __getitem__(self, item):
        assert 0 < item <= self._size, 'index is invalid.'
        return self._heap[item]

    def __setitem__(self, key, value):
        assert 0 < key <= self._size, 'index is invalid.'
        self._heap[key] = value

    def __iter__(self):
        return iter(self._heap[1:])

    def __len__(self):
        return self._size


class HeapSort(object):
    def sort(self, array):
        assert isinstance(array, list), 'input must be a list.'
        heap = Heap()
        map(lambda x: heap.insert(x), array)
        result = [heap.pop(1) for i in xrange(len(heap))]
        return result


if __name__ == '__main__':
    a = [3, 2, 5, 6, 1, 5, 6, 7]
    hs = HeapSort()
    print hs.sort(a)
