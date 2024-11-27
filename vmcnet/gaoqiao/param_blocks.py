def block_fn(block):
    if not isinstance(block, dict):
        return False
    return set() < set(block.keys()) <= {"w", "b"}