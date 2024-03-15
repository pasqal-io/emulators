class Register:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def dist2(left: Register, right: Register) -> float:
    return (left.x - right.x) * (left.x - right.x) + (left.y - right.y) * (
        left.y - right.y
    )
