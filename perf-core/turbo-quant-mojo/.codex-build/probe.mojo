from std.memory import UnsafePointer, alloc

@export("probe")
def probe(data: UnsafePointer[Float32, MutUntrackedOrigin], result: UnsafePointer[Float32, MutUntrackedOrigin]) abi("C") -> Bool:
    result.store(data.load(0))
    return True

def main():
    var p = alloc[Float32](1)
    p.store(0, Float32(2.5))
    print(p.load(0))
    p.free()
