from enum import Enum

#Enumerador representando as cores possíveis de serem jogadas mais uma cor vazia para preencher o objeto que não tem uma jogada ainda
class Color(Enum):
    RED = 'Red'
    BLUE = 'Blue'
    GRAY = 'Gray'
    EMPTY = 'Empty'

#Enumerador que representa as direções onde o tabuleiro pode ser movido
class Moviment(Enum):
    UP = 'U'
    DOWN = 'D'
    LEFT = 'L'
    RIGHT = 'R'

#Representa uma peça dentro tabuleiro, podendo indicar também um espaço vazio com a cor Empty e o valor 0
class Piece:
    def __init__(self, color = Color.EMPTY, value = 0):
        self.color = color
        self.value = value
    def __str__(self) -> str:
        return f"Cor: {self.color} Valor: {self.value}"

#Classe base para o ambiente, onde é guardado o estado atual do jogo, efetua operações no tabuleiro (colocar peças e somar valores no movimento)
class Enviremont:
    def __init__(self):
        self.board = [
                        [Piece(),Piece(value = 3, color = Color.BLUE),Piece(),Piece()],
                        [Piece(),Piece(value = 3, color = Color.BLUE),Piece(),Piece(value = 3, color = Color.BLUE)],
                        [Piece(),Piece(value = 3, color = Color.BLUE),Piece(),Piece()],
                        [Piece(),Piece(value = 3, color = Color.BLUE),Piece(),Piece()],
                    ]

    #Efetua a lógica de somar ou anular as peças de acordo com a lógica do jogo, cores e valores iguais gera uma nova peça da mesma cor e com o triplo do valor original,
    #duas peças de mesmo valor e cores diferentes se anulam e caso seja de valores diferentes não faz nada.
    def sum_pieces(self, piece1: Piece, piece2: Piece) -> Piece:
        new_value = 0
        if (piece1.value == piece2.value):
            if (piece1.color == piece2.color):
                new_value = piece1.value * 3
                return Piece(color = piece1.color, value = new_value)
            else:
                return Piece()

    def put_piece(self, position_x: int, position_y: int, piece: Piece):
        self.board[position_x, position_y] = piece;

    def calculate_board(self, direction: Moviment):
        iteration = range(3,-1,-1) if direction == Moviment.RIGHT or direction == Moviment.DOWN else range(4) 
        for i in range(4):
            for j in iteration:
                actual_piece = self.board[i][j] if direction == Moviment.LEFT or direction == Moviment.RIGHT else self.board[j][i]
                new_piece = Piece()
                before_piece = Piece()
                if (actual_piece.value == 0):
                    continue
                k = j + 1 if direction == Moviment.LEFT or direction == Moviment.UP else j - 1
                while before_piece.value == 0 and (k >= 0 and k <= 3):
                    before_piece = self.board[i][k] if direction == Moviment.LEFT or direction == Moviment.RIGHT else self.board[k][i]
                    k = k - 1 if direction == Moviment.RIGHT or direction == Moviment.DOWN else k + 1
                if (before_piece.value != 0):
                    new_piece = self.sum_pieces(actual_piece, before_piece)
                    if (new_piece != None):
                        if direction == Moviment.LEFT or direction == Moviment.RIGHT:
                            self.board[i][j] = new_piece
                            self.board[i][k + 1 if direction == Moviment.RIGHT else k - 1] = Piece()
                        else:
                            self.board[j][i] = new_piece
                            self.board[k + 1 if direction == Moviment.DOWN else k - 1][i] = Piece()
            if direction == Moviment.LEFT or direction == Moviment.RIGHT:
                filled_elements = [item for item in self.board[i] if item.value != 0]
                self.board[i] = [Piece()]*(len(self.board[i])-len(filled_elements)) + filled_elements if direction == Moviment.RIGHT else filled_elements + [Piece()]*(len(self.board[i])-len(filled_elements))
            else:
                filled_elements = []
                for l in iteration:
                    if self.board[l][i].value != 0:
                        filled_elements.append(self.board[l][i])
                for l in iteration:
                    if len(filled_elements) != 0:
                        self.board[l][i] = filled_elements.pop(0)
                    else:
                        self.board[l][i] = Piece()

if __name__ == "__main__":
    ambiente = Enviremont()
    for linha in ambiente.board:
        for elemento in linha:
            print(elemento, end=' ')
        print()
    print()
    movement = input('Digite o movimento(U, D, L, R): ')

    ambiente.calculate_board(direction=Moviment(movement))
    for linha in ambiente.board:
        for elemento in linha:
            print(elemento, end=' ')
        print()