class Board:
    EMPTY = 0
    WHITE = 1
    BLACK = 2
    
    def __init__(self):
        self.turn = self.WHITE
        self.outputIndex = {}
        self.board = [self.EMPTY, self.EMPTY, self.EMPTY,
                      self.EMPTY, self.EMPTY, self.EMPTY,
                      self.EMPTY, self.EMPTY, self.EMPTY]
        
        # white forward moves
        self.outputIndex["(6, 3)"] = 0
        self.outputIndex["(7, 4)"] = 1
        self.outputIndex["(8, 5)"] = 2
        self.outputIndex["(3, 0)"] = 3
        self.outputIndex["(4, 1)"] = 4
        self.outputIndex["(5, 2)"] = 5
        # ...

    def getNetworkOutputIndex(self, move):
        return self.outputIndex[str(move)]

    def applyMove(self, move):
        fromSquare = move[0]
        toSquare = move[1]
        self.board[toSquare] = self.board[fromSquare]
        self.board[fromSquare] = self.EMPTY
        if self.turn == self.WHITE:
            self.turn = self.BLACK
        else:
            self.turn = self.WHITE
        self.legal_moves = None
