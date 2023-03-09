#!/usr/bin/env python3
import sys
from enum import Enum
from copy import deepcopy
import random

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
        return f"{self.color.value}{self.value}"
    
    def __eq__(self, other):
        if isinstance(other, Piece):
            return self.color.value == other.color.value and self.value == other.value
        return False

#Classe base para o ambiente, onde é guardado o estado atual do jogo, efetua operações no tabuleiro (colocar peças, somar valores no movimento ou anular peças.)
class Enviremont:
    def __init__(self):
        self.board = [
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                    ]

    #Checa se a posição é válida para por uma peça
    def __is_empty_space(self, position_x: int, position_y: int) -> bool:
        return self.board[position_x][position_y].value == 0
    
    #Checa se o tabuleiro atual é igual ao tabuleiro totalmente vazio
    def is_board_empty(self) -> bool:
        return self.board == [
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
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

    #Coloca uma peça em uma determinada posição, se for uma posição valida (vazia) retorna true, caso contrário false 
    def put_piece(self, position_x: int, position_y: int, piece: Piece) -> bool:
        if self.__is_empty_space(position_x, position_y):
            self.board[position_x][position_y] = piece
            return True
        else:
            return False

    #Atualiza o estado do tabulerio de acordo com o movimento selecionado, validando todas as regras descritas pelo jogo.
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

log = open("player.log", "w")

def gera_posicao() -> tuple:
    return (random.randint(0,3), random.randint(0,3))

def get_color(round: int) -> Color:
    calc = round % 10
    if calc in [1,6]:
        return Color.BLUE
    elif calc in [3,8]:
        return Color.GRAY
    elif calc in [2,7]:
        return Color.RED

def joga(round: int, ambiente: Enviremont):
    if round % 10 in [1,2,3,6,7,8]:
        posicao = gera_posicao()
        while not ambiente.put_piece(posicao[0],posicao[1], Piece(color=get_color(round), value=1)):
            posicao = gera_posicao()
        print(f'{posicao[0]+1}{posicao[1]+1}')
        sys.stdout.flush()
    else:
        move = random.choice(list(Moviment))
        old_board = deepcopy(ambiente.board)
        ambiente.calculate_board(move)
        while old_board == ambiente.board and not ambiente.is_board_empty():
            move = random.choice(list(Moviment))
            ambiente.calculate_board(move)
        print(f'{move.value}')
        sys.stdout.flush()

def le_jogada(ambiente: Enviremont):
    entrada = sys.stdin.readline().strip()
    if entrada == 'Quit':
        log.close()
        sys.exit(0)
    if entrada in map(lambda item: item.value, list(Moviment)):
        move = Moviment(entrada)
        ambiente.calculate_board(move)
    else:
        ambiente.put_piece(int(entrada[0])-1, int(entrada[1])-1, Piece(color=get_color(round), value=1))

def registra_log(ambiente: Enviremont):
    for linha in ambiente.board:
        for elemento in linha:
            log.write(f'{elemento} ')
        log.write('\n')
    log.write('\n')
if __name__ == "__main__":
    ambiente = Enviremont()
    round = 1
    jogador = sys.stdin.readline().strip()
    if (jogador == "A"):
        while True:
            if round % 2 != 0:
                joga(round, ambiente)
            else:
                le_jogada(ambiente)
            round += 1
            registra_log(ambiente)
    else:
        while True:
            if jogador == 'Quit':
                break
            if round % 2 == 0:
                joga(round, ambiente)
            else:
                le_jogada(ambiente)
            round += 1
            registra_log(ambiente)
    log.close()
    