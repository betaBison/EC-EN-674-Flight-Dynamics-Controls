"""
Heap helper class for Dijkstra Searching
"""

class PqueueHeap:
    heap = []
    fill = 0
    indeces = {}

    def priority(self, index):
        return self.heap[index][0]

    def value(self, index):
        return self.heap[index][1]

    def __init__(self):
        self.heap = []
        self.indeces = {}

    # O(log(|V|))
    def insert(self, value, priority):
        while len(self.heap) <= self.fill:
            self.heap.append(None)
        self.heap[self.fill] = (priority, value)
        self.indeces[value] = self.fill

        index = self.fill
        self.ascendHeap(index)

        self.fill += 1

    # O(log(|V|))
    def deleteMin(self):
        if self.fill == 0:
            return None

        self.fill -= 1

        result = self.heap[0]
        self.heap[0] = self.heap[self.fill]
        index = 0
        self.descendHeap(index)
        self.heap[self.fill] = None

        del self.indeces[result[1]]

        return result[1]

    # O(log(|V|))
    def decreaseKey(self, value, priority):
        if not value in self.indeces:
            self.insert(value, priority)
            return

        index = self.indeces[value]

        self.heap[index] = (priority, value)
        self.ascendHeap(index)


    # O(log(|V|))
    def ascendHeap(self, index):
        # move child up while smaller than its parent
        while index > 0:
            parentIndex = int((index - 1) / 2)
            if self.priority(parentIndex) > self.priority(index):
                tmp = self.heap[index]
                self.heap[index] = self.heap[parentIndex]
                self.heap[parentIndex] = tmp

                self.indeces[self.value(parentIndex)] = parentIndex
                self.indeces[self.value(index)] = index

                index = parentIndex
            else:
                break

    # O(log(|V|))
    def descendHeap(self, index):
        # move child down while larger than its parent
        while index < self.fill:
            childIndex1 = index * 2 + 1
            childIndex2 = index * 2 + 2
            if childIndex1 >= self.fill:
                break
            minChildIndex = childIndex1
            if childIndex2 < self.fill and self.priority(childIndex2) < self.priority(childIndex1):
                minChildIndex = childIndex2
            if self.priority(minChildIndex) < self.priority(index):
                tmp = self.heap[index]
                self.heap[index] = self.heap[minChildIndex]
                self.heap[minChildIndex] = tmp

                self.indeces[self.value(minChildIndex)] = minChildIndex
                self.indeces[self.value(index)] = index

                index = minChildIndex
            else:
                break
