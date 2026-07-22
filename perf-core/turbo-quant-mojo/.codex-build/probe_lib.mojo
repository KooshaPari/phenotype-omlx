from std.memory import UnsafePointer, alloc

@export("probe")
def probe(data: UnsafePointer[Float32, MutUntrackedOrigin], result: UnsafePointer[Float32, MutUntrackedOrigin]) abi("C") -> Bool:
    result.store(data.load(0))
    return True

