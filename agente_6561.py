#!/usr/bin/env python3
import sys
from enum import Enum
from copy import deepcopy
from pprint import pprint
from functools import reduce
import random
import time

#Enumerador representando as cores possíveis de serem jogadas mais uma cor vazia para preencher o objeto que não tem uma jogada ainda
class Color(Enum):
    BLUE = 1
    RED = 2
    GRAY = 3
    EMPTY = 4

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
    def __hash__(self) -> int:
        return hash((self.color, self.value))
    
    def __repr__(self):
        return f"Piece(color={self.color}, value={self.value})"

#Classe base para o ambiente, onde é guardado o estado atual do jogo, efetua operações no tabuleiro (colocar peças, somar valores no movimento ou anular peças.)
class Enviremont:
    def __init__(self):
        self.board = [
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                    ]

    #Reinicia o jogo
    def reset_game(self):
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
    
    #Verifica se não existem mais espaços vazios no tabuleiro, caso não exista o jogo deve parar
    def is_board_full(self) -> bool:
        result = True
        for linha in self.board:
            for item in linha:
                if item.value == 0:
                    result = False
                    break
            else:
                continue
            break
        return result
    
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
    #Caso seja fornecido um tabuleiro as regras são aplicadas nele sem impactar no tabuleiro original
    #A função retorna o novo tabuleiro, quantas peças se somaram e quantas se anularam respectivamente
    def calculate_board(self, movement: str, board = None):
        direction = Moviment(movement)
        removals = 0
        additions = 0
        if not board:
            board = self.board
        iteration = range(3,-1,-1) if direction == Moviment.RIGHT or direction == Moviment.DOWN else range(4) 
        for i in range(4):
            for j in iteration:
                actual_piece = board[i][j] if direction == Moviment.LEFT or direction == Moviment.RIGHT else board[j][i]
                new_piece = Piece()
                before_piece = Piece()
                if (actual_piece.value == 0):
                    continue
                k = j + 1 if direction == Moviment.LEFT or direction == Moviment.UP else j - 1
                while before_piece.value == 0 and (k >= 0 and k <= 3):
                    before_piece = board[i][k] if direction == Moviment.LEFT or direction == Moviment.RIGHT else board[k][i]
                    k = k - 1 if direction == Moviment.RIGHT or direction == Moviment.DOWN else k + 1
                if (before_piece.value != 0):
                    new_piece = self.sum_pieces(actual_piece, before_piece)
                    if (new_piece != None):
                        if (new_piece.value == 0):
                            removals += 1
                        else:
                            additions += 1
                        if direction == Moviment.LEFT or direction == Moviment.RIGHT:
                            board[i][j] = new_piece
                            board[i][k + 1 if direction == Moviment.RIGHT else k - 1] = Piece()
                        else:
                            board[j][i] = new_piece
                            board[k + 1 if direction == Moviment.DOWN else k - 1][i] = Piece()
            if direction == Moviment.LEFT or direction == Moviment.RIGHT:
                filled_elements = [item for item in board[i] if item.value != 0]
                board[i] = [Piece()]*(len(board[i])-len(filled_elements)) + filled_elements if direction == Moviment.RIGHT else filled_elements + [Piece()]*(len(board[i])-len(filled_elements))
            else:
                filled_elements = []
                for l in iteration:
                    if board[l][i].value != 0:
                        filled_elements.append(board[l][i])
                for l in iteration:
                    if len(filled_elements) != 0:
                        board[l][i] = filled_elements.pop(0)
                    else:
                        board[l][i] = Piece()
        return board, additions, removals
    def get_score(self, board = None) -> int:
        if not board:
            board = self.board
        score = 0
        for linha in board:
            for coluna in linha:
                score += coluna.value
        return score

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.3, q_values = {}):
        self.q_values = q_values # dicionário para armazenar os valores Q
        self.alpha = alpha # taxa de aprendizagem
        self.gamma = gamma # fator de desconto
        self.epsilon = epsilon # probabilidade de explorar

    def get_legal_actions(self, state) -> list:
        moves = []
        for i in range(4):
            if (state[i][0]) != 'X':
                moves.append(state[i][0])
        return moves
    
    def get_q_value(self, state, action):
        """
        Obtém o valor Q para um determinado estado e ação
        """
        if (tuple(map(tuple, state)), action) not in self.q_values:
            self.q_values[(tuple(map(tuple, state)), action)] = 0.0
        return self.q_values[(tuple(map(tuple, state)), action)]

    def get_value(self, state):
        """
        Obtém o valor V para um determinado estado
        """
        possible_actions = self.get_legal_actions(state)
        if not possible_actions:
            return 0.0
        return max([self.get_q_value(state, action) for action in possible_actions])

    def get_policy(self, state):
        """
        Obtém a política para um determinado estado
        """
        possible_actions = self.get_legal_actions(state)
        if not possible_actions:
            return {}
        best_action = max(possible_actions, key=lambda action: self.get_q_value(state, action))
        policy = {action: (1 - self.epsilon) if action == best_action else self.epsilon/len(possible_actions) for action in possible_actions}
        return policy

    def update(self, state, action, next_state, reward):
        """
        Atualiza os valores Q para um determinado estado e ação
        """
        q_value = self.get_q_value(state, action)
        value = self.get_value(next_state)
        self.q_values[(tuple(map(tuple, state)), action)] = q_value + self.alpha * (reward + self.gamma * value - q_value)

    def get_action(self, state):
        """
        Obtém uma ação de acordo com a política atual
        """
        policy = self.get_policy(state)
        return random.choices(list(policy.keys()), weights=list(policy.values()))[0]

class RandomPlayer:
    def __init__(self) -> None:
        self.log = open("player.log", "w")

    def gera_posicao(self) -> tuple:
        return (random.randint(0,3), random.randint(0,3))

    def joga(self, round: int, ambiente: Enviremont):
        if round % 10 in [1,2,3,6,7,8]:
            posicao = self.gera_posicao()
            while not ambiente.put_piece(posicao[0],posicao[1], Piece(color=self.get_color(round), value=1)):
                posicao = self.gera_posicao()
            print(f'{posicao[0]+1}{posicao[1]+1}')
            sys.stdout.flush()
        else:
            move = random.choice(list(Moviment))
            old_board = deepcopy(ambiente.board)
            ambiente.calculate_board(movement=move.value)
            while old_board == ambiente.board and not ambiente.is_board_empty():
                move = random.choice(list(Moviment))
                ambiente.calculate_board(movement=move.value)
            print(f'{move.value}')
            sys.stdout.flush()

    def le_jogada(self, ambiente: Enviremont):
        entrada = sys.stdin.readline().strip()
        if entrada == 'Quit':
            self.log.close()
            sys.exit(0)
        if entrada in map(lambda item: item.value, list(Moviment)):
            move = Moviment(entrada)
            ambiente.calculate_board(movement=move.value)
        else:
            ambiente.put_piece(int(entrada[0])-1, int(entrada[1])-1, Piece(color=self.get_color(round), value=1))

    def registra_log(self, ambiente: Enviremont):
        for linha in ambiente.board:
            for elemento in linha:
                self.log.write(f'{elemento} ')
            self.log.write('\n')
        self.log.write('\n')

def calculate_ai_state(board, calculate_board, is_board_empty, get_score):
    state = []
    for move in list(Moviment):
        new_board, additions, removals = calculate_board(movement=move.value, board=deepcopy(board))
        movement = move.value if new_board != board or is_board_empty else 'X'
        positive_score = 0
        balance = 0
        if (get_score(new_board) - get_score(board)> 0):
            positive_score = 1
        else:
            positive_score = -1
        if (removals - additions >= 1 ):
            balance = 1
        else:
            balance = -1
        state.append([movement, balance, positive_score])
    return state

def get_best_move(board:list[list[Piece]], piece: Piece):
    max_piece = reduce(lambda x, y: x if x.value > y.value else y, [item for row in board for item in row])
    best_value = -1
    best_x = -1
    best_y = -1
    last_empty_x = -1
    last_empty_y = -1
    for i in range(4):
        for j in range(4):
            if board[i][j] == Piece():
                last_empty_x = i
                last_empty_y = j
                temp_board = deepcopy(board)
                temp_board[i][j] == piece
                value = 0
                for k in range(4):
                    if (max_piece.value == 1 or piece.color == max_piece.color):
                        if board[i][k].value == piece.value and board[i][k].color == piece.color and k != j:
                            if (k+1 == j or k-1 == j):
                                value += 1
                            value += 1
                        elif board[k][j].value == piece.value and board[k][j].color == piece.color and k != i:
                            if (k+1 == i or k-1 == i):
                                value += 1
                            value += 1
                        if board[i][k].value == piece.value and board[i][k].color != piece.color and k != j:
                            value -= 1
                        elif board[k][j].value == piece.value and board[k][j].color != piece.color and k != i:
                            value -= 1
                    else:
                        if board[i][k].value == piece.value and board[i][k].color == piece.color and k != j:
                            value -= 1
                        elif board[k][j].value == piece.value and board[k][j].color == piece.color and k != i:
                            value -= 1
                        if board[i][k].value == piece.value and board[i][k].color != piece.color and board[i][k].color != max_piece.color and k != j:
                            if (k+1 == j or k-1 == j):
                                value += 1
                            value += 1
                        elif board[k][j].value == piece.value and board[k][j].color != piece.color and board[k][j].color != max_piece.color and k != i:
                            if (k+1 == i or k-1 == i):
                                value += 1
                            value += 1
                if value > best_value:
                    best_value = value
                    best_x = i
                    best_y = j
    return best_x if best_x > -1 else last_empty_x, best_y if best_y > -1 else last_empty_y

def get_color(round: int) -> Color:
        calc = round % 10
        if calc in [1,6]:
            return Color.BLUE
        elif calc in [3,8]:
            return Color.GRAY
        elif calc in [2,7]:
            return Color.RED

def le_jogada(ambiente: Enviremont, round: int):
    entrada = sys.stdin.readline().strip()
    if entrada == 'Quit':
        sys.exit(0)
    if entrada in map(lambda item: item.value, list(Moviment)):
        move = Moviment(entrada)
        ambiente.calculate_board(movement=move.value)
    else:
        ambiente.put_piece(int(entrada[0])-1, int(entrada[1])-1, Piece(color=get_color(round), value=1))

def traningAI(enviremont: Enviremont, ai_player: QLearningAgent):
    for i in range(1,20000):
        enviremont.reset_game()
        for round in range(1,1001):
            if (enviremont.is_board_full()):
                break
            if round % 10 in [1,2,3,6,7,8]:
                posicao = get_best_move(enviremont.board, Piece(get_color(round), 1))
                enviremont.put_piece(position_x=posicao[0], position_y=posicao[1], piece=Piece(color=get_color(round), value=1))
            else:
                state = calculate_ai_state(enviremont.board, enviremont.calculate_board, enviremont.is_board_empty(), enviremont.get_score)
                score = enviremont.get_score()
                action = ai_player.get_action(state)
                next_state = calculate_ai_state(enviremont.board, enviremont.calculate_board, enviremont.is_board_empty(), enviremont.get_score)
                enviremont.calculate_board(movement=action)
                new_score = enviremont.get_score()
                reward = new_score - score
                reward -= 5 if reward == 0 else 0
                reward -= 10 if reward<0 and reward < -4 else 0
                reward += 10 if reward>0 and reward > 2 else 0
                ai_player.update(state, action, next_state, reward)
    file = open("q_values.txt", "w")
    pprint(ai_player.q_values, stream=file, indent=4, width=80)
    file.close()
    end_time = time.time()
    print(end_time - start_time)

if __name__ == "__main__":
    traningTime = False
    start_time = time.time()
    ambiente = Enviremont()
    ai_player = QLearningAgent(q_values={((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): -46.783798736352146,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -44.021109811304896,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -28.876829389444403,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): -48.81637542937679,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'D'): 10.911380933206967,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'L'): 9.482453827347399,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'R'): 21.47110281533764,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'U'): 11.831913209406395,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'D'): -13.811943060534986,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): -10.05954506341292,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): -13.2148932069577,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'U'): -10.106894563439234,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): -142.25347410535736,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -146.12985793320843,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): -142.6098651948091,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'D'): -7.5,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): 20.651007291113533,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): 8.173914962568524,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'U'): 7.144007041463341,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 242.2116746332307,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 215.55256949672878,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 381.9924411522662,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 301.80979865137846,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): 7.60460724205909,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 10.863015485657314,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 7.786077811904726,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): 8.707309256881008,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 388.1613102160419,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 403.46785970855876,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 408.2217532439929,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 377.7075271356242,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): -13.620291510000929,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): -13.116672373064764,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): -11.873481255469681,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): -10.161686171269682,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): 10.000303860074094,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 7.888752595162036,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 11.329585098153196,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): 10.86947891696946,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): -30.98529214833107,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): -22.88410297367932,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): -22.05890275197088,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): -33.691476030090534,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 409.7360025988273,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 418.1760180696151,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 482.60065099911026,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 399.55845972960014,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 393.74254354058564,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 507.57984927318194,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 417.8644110863413,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 404.467854919394,
    ((('U', -1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): -141.771090626962,
    ((('U', -1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -142.23513822506862,
    ((('U', -1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): -142.058853578825,
    ((('U', -1, -1), ('D', -1, -1), ('X', -1, -1), ('X', -1, -1)), 'D'): -45.15676852426157,
    ((('U', -1, -1), ('D', -1, -1), ('X', -1, -1), ('X', -1, -1)), 'U'): -33.771818677963196,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 16.897757468634932,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 11.935729390985273,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 10.840355232819924,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 11.859717595842564,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): 1.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 29.289024710730523,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 10.818297888281581,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 15.554362974373005,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 17.9981621875,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): -0.5,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'R'): -0.5,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 1.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 21.30229877741495,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 17.098494256709458,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 18.385865538859584,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 20.05812529608281,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 23.303377828125004,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 89.01871433515626,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): -0.75,
    ((('U', -1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 30.62071705134626,
    ((('U', -1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): 2.079298948638943,
    ((('U', -1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 26.51875670615238,
    ((('U', -1, -1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): -13.060248954488728,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -15.258880262639817,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -18.771320967243604,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): -10.005321701531125,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'D'): -14.361696733949035,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): -16.07590157705367,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): -14.776122939443727,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'U'): -14.96371949390923,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): -54.50414212548654,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -57.22323774332338,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): -38.60228683374343,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'D'): -2.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): 1.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): -0.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 136.1995002902193,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 154.997618279961,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 172.82422448461662,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 140.99661169632572,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): -1.6246496015625,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 4.89576759299282,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.30816622265624993,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): 1.35372169140625,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): -2.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): -8.750312279642326,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): -7.968278380681397,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): -6.886429206779778,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): -6.831252734375,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): -1.485391940800781,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 0.7890105222559304,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 4.59639912337363,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): -0.10818749999999999,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): -23.71906557548906,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): -18.989745874089586,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): -20.565058557684086,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): -21.454620874657657,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 277.49832496772166,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 319.19810775960667,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 284.01135168181446,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 245.79862607011546,
    ((('U', -1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): -52.74028620708032,
    ((('U', -1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -53.16201487278647,
    ((('U', -1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): -47.319377351702414,
    ((('U', -1, -1), ('D', 1, -1), ('X', -1, -1), ('X', -1, -1)), 'D'): -3.9,
    ((('U', -1, -1), ('D', 1, -1), ('X', -1, -1), ('X', -1, -1)), 'U'): -4.012630607616211,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -139.84202975090977,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -140.41713240139407,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): -140.26200193414496,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'L'): 21.674489574678013,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'R'): 22.41057743934769,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'U'): 6.869378058080522,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): -36.42357055968735,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): -50.72008458696592,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'U'): -51.0173078215811,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -149.99999999999966,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): -149.99999999999966,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): 21.917643064856847,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): 17.563391718695673,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'U'): 5.860130775516419,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 259.8093543952669,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 260.30808833979376,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 207.04192162790952,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 10.026141717992012,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 7.067507268925212,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): -5.969760470992471,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 336.6085696569555,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 512.9096666931026,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 49.584330845707036,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): -46.57348120815536,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): -34.01266175510142,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): -47.5053879952941,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 7.234240723726162,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 10.111807409103388,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): -5.812698017718457,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): -20.319229481813274,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): -21.348767991733098,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): -33.7412667540549,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'U'): -7.5,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 394.0139757975786,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 487.71390075117006,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 366.2591987692143,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 480.0169795583794,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 462.21469513657894,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 420.9538338373869,
    ((('U', -1, -1), ('X', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -149.99999999999966,
    ((('U', -1, -1), ('X', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): -149.99999999999966,
    ((('U', -1, -1), ('X', -1, -1), ('X', -1, -1), ('X', -1, -1)), 'U'): -149.99999999999966,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): -0.04999999999999999,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -3.5749999999999997,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): 4.48685415199808,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): 13.188767474237698,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'U'): 4.52438125,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): 3.6373187499999995,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -0.31651901235204205,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): 28.16660190300859,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 3.529216118398438,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 7.395011805507814,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): -0.5,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): -1.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): 1.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): 1.95,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 17.0847512585239,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 15.318826844327214,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 14.433152508752848,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): 20.03719607479274,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 44.06875284296875,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 99.041672984375,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 88.06875284296875,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 3.709875,
    ((('U', -1, 1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): 23.66453308770687,
    ((('U', -1, 1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): 11.78500216538379,
    ((('U', -1, 1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): 29.583696674072996,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 271.41808434083487,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 262.1605761045015,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 256.5034128963366,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 283.69774345772163,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'L'): -0.5,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'U'): 51.943920006741216,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): 102.3614727900854,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): 106.13699679482329,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 107.19635640884827,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): 106.85178438761355,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 142.1115830345258,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 115.16873731700258,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 151.6907039577553,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'D'): 50.648023645703134,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'R'): 8.371028593750001,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'U'): 8.5,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 162.32357024735103,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 162.34836238535428,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 176.50147307604294,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 161.19403838792226,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 210.34498266114053,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 213.76575519077016,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 211.972074441947,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 215.93820518770139,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'D'): 49.43104703125,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'L'): 221.50342480772662,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'R'): 156.03802232268586,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'U'): 33.296797672636714,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'D'): 122.24785721456391,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'L'): 109.99960300198993,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'R'): 54.09551454635745,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'U'): 126.39723826020885,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 258.44607509880365,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 203.845015670343,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 201.09242788650963,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): 210.16130885649716,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 90.28161227398246,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 85.6472854637531,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 83.73185297738314,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 86.9812139625977,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'D'): 10.339608059199218,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'L'): 112.35664506746213,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'R'): 225.45353569936742,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'U'): 120.65578515563348,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 449.77546771948187,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 494.4341807784121,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 463.15499316376633,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 441.432293120139,
    ((('U', -1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 267.711103475797,
    ((('U', -1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): 218.906324838528,
    ((('U', -1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 209.908822982643,
    ((('U', -1, 1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'D'): 59.726938542977756,
    ((('U', -1, 1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'U'): 84.12689022189666,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): 8.10847193092523,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): 7.678754077312652,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): 9.146903432600345,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): 11.108099165623202,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'D'): 6.346696297919367,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): 6.925747210357349,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): 4.029104875151368,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'U'): 9.678277411438694,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): 14.145232210203908,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -3.2255707918576038,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): 19.50312523574535,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'U'): 2.262190625,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 309.2103998868138,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 314.21254978893273,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 306.6543231415694,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 312.35753261559404,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): 1.8273675420220774,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 6.503129896948496,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): 3.369658193657275,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 78.65419271912361,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 0.5,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): -0.3581875,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): 1.9675958528939763,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): 4.38742339236747,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): 7.496559102575448,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): -3.8443718749999998,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): -0.16527812500000005,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 1.334721875,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): 3.3657956871093755,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 9.601190929737498,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 9.488488503072048,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 8.021833323438447,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): 14.613463003637374,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 0.5,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 354.91867046567506,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 353.16468534940304,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 382.2174024605181,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 341.9195008499392,
    ((('U', -1, 1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): 8.242195793972321,
    ((('U', -1, 1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -4.968364137600183,
    ((('U', -1, 1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): 16.67152847542799,
    ((('U', -1, 1), ('D', 1, -1), ('X', -1, -1), ('X', -1, -1)), 'D'): 0.689960625242227,
    ((('U', -1, 1), ('D', 1, -1), ('X', -1, -1), ('X', -1, -1)), 'U'): 5.367087698402466,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 375.4185920692873,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 313.76786064634973,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 400.0219780888558,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 448.31534959670375,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): 17.55,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 239.30411923708806,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 269.41188540579395,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 432.82289830827256,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 125.02540604299531,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 195.5626044128171,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 109.26805395380327,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 258.48258723171296,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 26.1,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'U'): 9.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 546.5495600206436,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 378.5711514546134,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 454.744222732996,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 435.41899972758733,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 52.363875,
    ((('U', -1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 239.58849902611087,
    ((('U', -1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): 46.871541726562505,
    ((('U', -1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 85.20781640822267,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): -10.041628289532095,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -10.187639648371096,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -10.248208587361553,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): -13.289724274182085,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'D'): -0.425,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'L'): -0.3,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'R'): 1.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'U'): -3.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'D'): -14.749146177084636,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): -17.92113966318019,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): -8.551307298166826,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'U'): -9.947950406815504,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): -64.90427077705885,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -71.33792291256373,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): -66.54704016628455,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'D'): -0.5,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'U'): -2.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 151.19078892992337,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 154.32893559884172,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 154.0567476140773,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 147.8824066563738,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): 3.1681767380074146,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 5.8187966478082265,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.3355143091992191,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): -2.5372812500000004,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 65.30625,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 7.5,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): -6.253971891437661,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): -7.938325032442011,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): -6.415140775914578,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): -7.865978091785992,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): 2.3928689186375465,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 1.0385793978960989,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 5.8187966478082265,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): 0.5382846898832934,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): -25.813977861971573,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): -31.231407585297948,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): -23.774421000634707,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): -32.98703093361201,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): -0.5,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 16.5,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 25.425,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): -2.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 376.3364683059899,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 422.81999984254026,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 561.0285589203547,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 400.10770917362464,
    ((('U', 1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): -58.6856044826084,
    ((('U', 1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -67.32842686787001,
    ((('U', 1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): -57.23900047788034,
    ((('U', 1, -1), ('D', -1, -1), ('X', -1, -1), ('X', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('X', -1, -1), ('X', -1, -1)), 'U'): -2.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 10.710034158151991,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 8.231840753261062,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 8.329546030336783,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 7.142255990709589,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'R'): 6.5,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'U'): 1.9250000000000003,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): 9.147424096656916,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): 5.574586627462438,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 2.422953197024506,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): 3.731334522351871,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 10.573287837704264,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): -4.813080521042689,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 7.818583031452041,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'R'): -0.5,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'U'): -1.5,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 257.70277633489275,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 292.7708136733962,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 249.42179332863213,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 245.14608353578825,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 8.492626805862518,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 2.913272576469842,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 1.9888387399448382,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'D'): 3.0166270390625005,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'U'): -1.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'D'): 7.853612360570625,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'L'): 1.3141595749041595,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'R'): -0.275,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'U'): -0.14251783242187477,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 9.515054747505765,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 1.967472681703482,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 6.9763799677484455,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): 3.6410825925204686,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 10.001031431864922,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 7.014240470256048,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 7.030369664144554,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 7.0025567101056705,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'D'): 1.42625,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'U'): -1.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 465.4135910835563,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 641.310984293915,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 480.34108318589665,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 496.33189155267024,
    ((('U', 1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 13.412898049197018,
    ((('U', 1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): -2.0848350972577876,
    ((('U', 1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 9.315136385757611,
    ((('U', 1, -1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'D'): 1.8549375,
    ((('U', 1, -1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'U'): -1.0,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): -32.01881909995854,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -35.32863153517838,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -34.38388077571457,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): -26.469663303513826,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'D'): 21.110236467100073,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'L'): 25.777578503159717,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'R'): 27.645401964441746,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'U'): 19.838656530480282,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'D'): -25.94176198105108,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): -25.26241357070925,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): -30.78489255890048,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'U'): -31.787212567213384,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): -30.978144491851303,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -37.247169915294805,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): -25.1703303777948,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'D'): 24.802067213972443,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): 30.297271065665626,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): 28.185423149425418,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'U'): 14.799649817363031,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 122.89837673665545,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 143.53263604077227,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 135.9244978148578,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 128.4727141783827,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): 6.787014836328685,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 10.005821864089818,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 7.003789225597722,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): 5.951167489420268,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 318.389180431352,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 408.12549099806637,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 331.3828545629116,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 331.34226982733355,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): -21.943575084193466,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): -27.67546068408749,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): -21.61085447531395,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): -22.06027989868772,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): 6.47418001148799,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 7.175773701520148,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 10.096856537966755,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): 7.158790692344433,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): -22.858145198031742,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): -22.059764127371295,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): -21.54715557077052,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): -21.97360982984315,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 474.87995019391605,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 661.4943897705074,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 916.9981045987194,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 426.2241831167564,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 308.2608696492316,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 332.2846557077467,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 323.31224976834676,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 310.62671294092075,
    ((('U', 1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): -24.736588313316446,
    ((('U', 1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -36.67962722336938,
    ((('U', 1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): -26.017041110216297,
    ((('U', 1, -1), ('D', 1, -1), ('X', -1, -1), ('X', -1, -1)), 'D'): -21.30586998781533,
    ((('U', 1, -1), ('D', 1, -1), ('X', -1, -1), ('X', -1, -1)), 'U'): -28.987898811475173,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 531.4760959253881,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 468.9260901098018,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 451.97907608052355,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 530.3738201366772,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 1.375,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): 7.5,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 553.6064599382746,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 143.01311985336622,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 69.7574024339605,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 136.555713336907,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 317.38203421423805,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 404.75983301090383,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 17.55,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'U'): 14.625,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 70.0171875,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): 55.59375,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 479.43772799802105,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 442.27863256551166,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 458.17565396279855,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 464.86242627528407,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 23.625,
    ((('U', 1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 320.5099275992302,
    ((('U', 1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): 193.1757420741419,
    ((('U', 1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 219.79325905735323,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 411.5222470129123,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 406.437709018692,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 414.4044340455134,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 411.9780327240345,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, 1)), 'D'): 106.17120194438678,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, 1)), 'L'): 69.21527512160156,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, 1)), 'R'): 36.03330269101563,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, 1)), 'U'): 89.8728960179624,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): 554.4965357354884,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): 398.9079977444295,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 437.9796698828958,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): 438.13121685736564,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 609.1960835745697,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 501.4381683838417,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 510.867644126595,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'D'): 47.19646546875001,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'L'): 18.033062500000003,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'R'): 19.770562500000004,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'U'): 15.5,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 278.1502159399526,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 271.9492998531387,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 270.20853458742994,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 288.080006399868,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 499.65954111037263,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 448.16682968927,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 422.20564172278813,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 424.69035176266163,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'U'): 7.5,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'D'): 429.22644901134373,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'L'): 324.6564411536774,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'R'): 240.87442438639187,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'U'): 459.08891735380234,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 422.2676627658709,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 327.10696701500194,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 255.0843319867538,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): 317.8120240903417,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 449.2341000284115,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 400.6589642638405,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 453.4728733505103,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 446.2012488847171,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'D'): 27.482257031250004,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'L'): 27.482257031250004,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'R'): 44.405015625000004,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'U'): 7.5,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 172.43575743971118,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 163.0589422963223,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 181.81340059045777,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 175.38850371003832,
    ((('U', 1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 532.483812254098,
    ((('U', 1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): 471.2502380132368,
    ((('U', 1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 533.9100384477042,
    ((('U', 1, 1), ('D', 1, 1), ('X', -1, -1), ('X', -1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', 1, 1), ('X', -1, -1), ('X', -1, -1)), 'U'): 21.39375,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): -99.07787808819639,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -97.57637045979536,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -99.42876852680692,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'D'): 2.649900490531797,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'L'): 16.692983270846735,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'R'): 19.850179069239076,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'D'): -49.37438850070809,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): -45.267819780745555,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): -41.94990763026203,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): -149.99999999999966,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -149.99999999999966,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'D'): 6.710057696530875,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): 25.488073981893145,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): 22.322762062443047,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 162.49833030773215,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 163.62223365349806,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 206.74253773957722,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): -4.331893756866892,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 11.16531966827548,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 8.818009770557051,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 177.5319599353346,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 130.8390019344336,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 290.09175560695024,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): -76.26108546147972,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): -71.91759831597875,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): -64.94704696959167,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): -5.997491109458939,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 7.003321199974696,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 10.002361921491858,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): -34.33162150662562,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): -21.116110351887748,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): -22.242542766080625,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 89.18561729478691,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 231.2608266630865,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 172.32051762527342,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 468.4121775346649,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 460.4192409867542,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 485.2480612251401,
    ((('X', -1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): -149.99999999999966,
    ((('X', -1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -149.99999999999966,
    ((('X', -1, -1), ('D', -1, -1), ('X', -1, -1), ('X', -1, -1)), 'D'): -149.99999999999966,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -23.87959621354252,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -30.920359700127992,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): -3.0166270390625005,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): -6.66762618359375,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -149.99999999999966,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): 1.0,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 113.73840099030258,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 112.14355205815036,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 2.6490810937500004,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.1920864921875003,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): -9.725,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): -0.5,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 3.6085135449566503,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 7.616933782928612,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): -22.117527214225426,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): -22.403986339384346,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 7.5,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 0.0,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 64.04375,
    ((('X', -1, -1), ('X', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -149.99999999999966})

    if (traningTime):
        traningAI(enviremont=ambiente, ai_player=ai_player)
    
    jogador = sys.stdin.readline().strip()
    round = 1
    ambiente.reset_game()
    if (jogador == "A"):
        while True:
            if round % 2 != 0:
                if round % 10 in [1,2,3,6,7,8]:
                    posicao = get_best_move(ambiente.board, Piece(get_color(round), 1))
                    ambiente.put_piece(position_x=posicao[0], position_y=posicao[1], piece=Piece(color=get_color(round), value=1))
                    print(f'{posicao[0]+1}{posicao[1]+1}')
                    sys.stdout.flush()
                else:
                    state = calculate_ai_state(ambiente.board, ambiente.calculate_board, ambiente.is_board_empty(), ambiente.get_score)
                    action = ai_player.get_action(state)
                    ambiente.calculate_board(action)
                    print(action)
                    sys.stdout.flush()
            else:
                le_jogada(ambiente, round)
            round += 1
    else:
        while True:
            if jogador == 'Quit':
                break
            if round % 2 == 0:
                if round % 10 in [1,2,3,6,7,8]:
                    posicao = get_best_move(ambiente.board, Piece(get_color(round), 1))
                    ambiente.put_piece(position_x=posicao[0], position_y=posicao[1], piece=Piece(color=get_color(round), value=1))
                    print(f'{posicao[0]+1}{posicao[1]+1}')
                    sys.stdout.flush()
                else:
                    state = calculate_ai_state(ambiente.board, ambiente.calculate_board, ambiente.is_board_empty(), ambiente.get_score)
                    action = ai_player.get_action(state)
                    ambiente.calculate_board(action)
                    print(action)
                    sys.stdout.flush()
            else:
                le_jogada(ambiente, round)
            round += 1
    