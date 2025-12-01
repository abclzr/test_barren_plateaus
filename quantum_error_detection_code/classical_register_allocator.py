class ClassicalRegisterAllocator:
    def __init__(self, clibts: int):
        self.clibts = clibts
        self.cnt = 0
    
    def get_clbit_index(self):
        self.cnt += 1
        return self.cnt - 1
    
    def copy(self):
        new_allocator = ClassicalRegisterAllocator(self.clibts)
        new_allocator.cnt = self.cnt
        return new_allocator


