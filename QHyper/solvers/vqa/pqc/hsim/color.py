class Color():
    def __init__(self, j, k, f) -> None:
        self.j = j
        self.k = k
        self.f = f

    def __eq__(self, __value: object) -> bool:
        return self.j == __value.j and self.k == __value.k and self.f == __value.f
    
    def __hash__(self) -> int:
        return hash((self.j, self.k, self.f))
    
    def __str__(self) -> str:
        return f"Color{{j={self.j}, k={self.k}, f={self.f}}}"
    
    def __repr__(self) -> str:
        return str(self)
