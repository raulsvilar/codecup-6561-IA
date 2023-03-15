#!/usr/bin/env python3
import sys
from enum import Enum
from copy import deepcopy
from pprint import pprint
from functools import reduce
import random
import time

class Color(Enum):
    """ Enumerador representando as cores possíveis de serem jogadas mais uma cor vazia para preencher o objeto que não tem uma jogada ainda. """
    BLUE = 1
    RED = 2
    GRAY = 3
    EMPTY = 4

class Moviment(Enum):
    """ Enumerador que representa as direções onde o tabuleiro pode ser movido. """
    UP = 'U'
    DOWN = 'D'
    LEFT = 'L'
    RIGHT = 'R'

class Piece:
    """ Representa uma peça dentro tabuleiro, podendo indicar também um espaço vazio com a cor Empty e o valor 0. """
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

class Environment:
    """ Classe base para o ambiente, onde é guardado o estado atual do jogo, efetua operações no tabuleiro (colocar peças, somar valores no movimento ou anular peças). """
    def __init__(self):
        self.board = [
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                    ]

    def reset_game(self):
        """
        Reinicia o jogo.
        """
        self.board = [
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                    ]
        
    def __is_empty_space(self, position_x: int, position_y: int) -> bool:
        """
        Checa se a posição é válida para por uma peça.
        """
        return self.board[position_x][position_y].value == 0
    
    def is_board_empty(self) -> bool:
        """
        Checa se o tabuleiro atual é igual ao tabuleiro totalmente vazio.
        """
        return self.board == [
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                        [Piece(),Piece(), Piece(),Piece()],
                    ]
    
    def is_board_full(self) -> bool:
        """
        Verifica se não existem mais espaços vazios no tabuleiro, caso não exista o jogo deve parar.
        """
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
    
    def sum_pieces(self, piece1: Piece, piece2: Piece) -> Piece:
        """
        Efetua a lógica de somar ou anular as peças de acordo com a lógica do jogo, cores e valores iguais gera uma nova
        peça da mesma cor e com o triplo do valor original, duas peças de mesmo valor e cores diferentes se anulam e caso
        seja de valores diferentes não faz nada.
        """
        new_value = 0
        if (piece1.value == piece2.value):
            if (piece1.color == piece2.color):
                new_value = piece1.value * 3
                return Piece(color = piece1.color, value = new_value)
            else:
                return Piece()

    def put_piece(self, position_x: int, position_y: int, piece: Piece) -> bool:
        """
        Coloca uma peça em uma determinada posição, se for uma posição valida (vazia) retorna true, caso contrário false.
        """
        if self.__is_empty_space(position_x, position_y):
            self.board[position_x][position_y] = piece
            return True
        else:
            return False

    def calculate_board(self, movement: str, board = None):
        """
        Atualiza o estado do tabulerio de acordo com o movimento selecionado, validando todas as regras descritas pelo jogo.
        Caso seja fornecido um tabuleiro as regras são aplicadas nele sem impactar no tabuleiro original.
        A função retorna o novo tabuleiro, quantas peças se somaram e quantas se anularam respectivamente.
        """
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
        """
        Calcula a pontuação de cada uma das cores presentes no tabuleiro.
        """
        if not board:
            board = self.board
        score_blue = 0
        score_red = 0
        score_gray = 0
        for row in board:
            for piece in row:
                if (piece.color == Color.BLUE):
                    score_blue += piece.value
                elif (piece.color == Color.RED):
                    score_red += piece.value
                else:
                    score_gray += piece.value
        return (max([score_blue, score_red, score_gray]) * 2) - score_blue - score_red - score_gray

class QLearningAgent:
    """Classe Q-learning"""

    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1, q_values = {}):
        self.q_values = q_values # dicionário para armazenar os valores Q
        self.alpha = alpha # taxa de aprendizagem
        self.gamma = gamma # fator de desconto
        self.epsilon = epsilon # probabilidade de explorar

    def get_legal_actions(self, state) -> list:
        """
        Valida o movimento.
        """
        moves = []
        for i in range(4):
            if (state[i][0]) != 'X':
                moves.append(state[i][0])
        return moves

    def get_q_value(self, state, action):
        """
        Obtem o Q-value para um determinado estado e ação
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
        Atualiza os valores para um determinado estado e ação
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
    """ Classe de um jogador player aleatório. """
    def __init__(self) -> None:
        self.log = open("player.log", "w")

    def generate_position(self) -> tuple:
        """
        Gera uma posição aleatória de jogo.
        """
        return (random.randint(0,3), random.randint(0,3))

    def play(self, round: int, environment: Environment):
        """
        Efetua uma jogada.
        """
        if round % 10 in [1,2,3,6,7,8]:
            posicao = self.generate_position()
            while not environment.put_piece(posicao[0],posicao[1], Piece(color=self.get_color(round), value=1)):
                posicao = self.generate_position()
            print(f'{posicao[0]+1}{posicao[1]+1}')
            sys.stdout.flush()
        else:
            move = random.choice(list(Moviment))
            old_board = deepcopy(environment.board)
            environment.calculate_board(movement=move.value)
            while old_board == environment.board and not environment.is_board_empty():
                move = random.choice(list(Moviment))
                environment.calculate_board(movement=move.value)
            print(f'{move.value}')
            sys.stdout.flush()

    def read_move(self, environment: Environment):
        """
        Lê movimentos imputados.
        """
        entrada = sys.stdin.readline().strip()
        if entrada == 'Quit':
            self.log.close()
            sys.exit(0)
        if entrada in map(lambda item: item.value, list(Moviment)):
            move = Moviment(entrada)
            environment.calculate_board(movement=move.value)
        else:
            environment.put_piece(int(entrada[0])-1, int(entrada[1])-1, Piece(color=self.get_color(round), value=1))

    def register_log(self, environment: Environment):
        """
        Efetua o registro de logs.
        """
        for linha in environment.board:
            for elemento in linha:
                self.log.write(f'{elemento} ')
            self.log.write('\n')
        self.log.write('\n')


def calculate_ai_state(board, calculate_board, is_board_empty, get_score):
    """
    Calcula a recompensa da IA.
    """
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
        state.append([movement, positive_score, balance])
    return state

def get_best_pick(board:list[list[Piece]], piece: Piece):
    """
    Valida se a maior peça existente no tabuleiro é 1 ou colocando uma peça igual a de maior valor
    para colocar cores iguais juntas e distanciar as de cores diferentes.
    """
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
                        elif board[k][j].value == piece.value and board[k][j].color == piece.color and k != i:
                            if (k+1 == i or k-1 == i):
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
    """
    Identifica a cor pelo round.
    """
    calc = round % 10
    if calc in [1,6]:
        return Color.BLUE
    elif calc in [3,8]:
        return Color.GRAY
    elif calc in [2,7]:
        return Color.RED

def read_move(environment: Environment, round: int):
    """
    Lê movimentos imputados.
    """
    entrada = sys.stdin.readline().strip()
    if entrada == 'Quit':
        sys.exit(0)
    if entrada in map(lambda item: item.value, list(Moviment)):
        move = Moviment(entrada)
        environment.calculate_board(movement=move.value)
    else:
        environment.put_piece(int(entrada[0])-1, int(entrada[1])-1, Piece(color=get_color(round), value=1))

def trainingAI(environment: Environment, ai_player: QLearningAgent):
    """
    Efetua o trainamento da IA.
    """
    score_file = open("score_log.log", "w")
    average = 0
    for partidas in range(1,10000):
        environment.reset_game()
        for round in range(1,1001):
            if (environment.is_board_full()):
                break
            if round % 10 in [1,2,3,6,7,8]:
                posicao = get_best_pick(environment.board, Piece(get_color(round), 1))
                environment.put_piece(position_x=posicao[0], position_y=posicao[1], piece=Piece(color=get_color(round), value=1))
            else:
                state = calculate_ai_state(environment.board, environment.calculate_board, environment.is_board_empty(), environment.get_score)
                action = ai_player.get_action(state)
                next_state = calculate_ai_state(environment.board, environment.calculate_board, environment.is_board_empty(), environment.get_score)
                score = environment.get_score()
                environment.calculate_board(movement=action)
                reward = environment.get_score() - score
                reward += 10 if reward > 2 else 0
                reward -= 5 if reward == 0 else 0
                ai_player.update(state, action, next_state, reward)
        score_partida = environment.get_score()
        average += environment.get_score()
        score_file.write(f'{score_partida}\n')
    score_file.write(f'{average/partidas}\n')
    file = open("q_values.txt", "w")
    pprint(ai_player.q_values, stream=file, indent=4, width=80)
    file.close()
    score_file.close()
    end_time = time.time()
    print(end_time - start_time)

if __name__ == "__main__":
    traningTime = False
    start_time = time.time()
    environment = Environment()
    ai_player = QLearningAgent(q_values={   ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): -49.999999999999915,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -49.999999999999915,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -49.999999999999915,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): -49.999999999999915,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'U'): -2.5,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'D'): -2.5,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): 1.6709645627180043,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): 6.934945363406964,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'U'): -2.3838127968749996,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'D'): 3.8991729843750003,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'L'): -0.5,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'R'): 17.554984340232423,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'U'): -2.5,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): -49.999762406338405,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -49.999750539144195,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): -49.99976112181078,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): -49.78369812411438,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): -49.78439717072824,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): -49.795052110512515,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): -49.79097769471483,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): 9.269459747794883,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 19.417415323302414,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 59.66901453078084,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): -2.5,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 13.000000156669689,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 13.000005104293454,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 20.000000000000057,
    ((('U', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 13.00027596675726,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): -2.5,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): 10.963046647808227,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): -0.11237739845307604,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): -1.8385571691503195,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): -2.5,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 5.123250208844701,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): -2.606261406260742,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 143.54658892023434,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 156.33116876474065,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 152.43088288707003,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): 148.63869423287824,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'D'): 186.207106499737,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'L'): 194.11440358964592,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'R'): 189.90901854747878,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'U'): 191.2660268713968,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'D'): 2.978911867286268,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'L'): 16.800979525135144,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'R'): 6.330095146614578,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'U'): 7.04241817106663,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 13.773247723952899,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 20.015879999042145,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 13.639299765557418,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 13.790469913455677,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'D'): 34.17964290846986,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'L'): 163.92376739979488,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'R'): 128.76951060460812,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'U'): 16.86956012068964,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 79.08401294020612,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 104.41825304044076,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 107.39269099619881,
    ((('U', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 68.24884112857028,
    ((('U', -1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): -49.47886931779772,
    ((('U', -1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -49.91573403038309,
    ((('U', -1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): -49.45831902224637,
    ((('U', -1, -1), ('D', -1, -1), ('X', -1, -1), ('X', -1, -1)), 'D'): -22.98199561686815,
    ((('U', -1, -1), ('D', -1, -1), ('X', -1, -1), ('X', -1, -1)), 'U'): -22.98199561686815,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): -13.565569407518709,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): -13.58664768993318,
    ((('U', -1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): -9.973942171975992,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): -2.5,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 118.02154849570313,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 4.155195518202291,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 23.188746093982616,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): -13.78738854472099,
    ((('U', -1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): -13.972324329959967,
    ((('U', -1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): -9.99695879779604,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): 8.064081093750001,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'D'): 12.675,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'U'): -2.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): 8.177190625,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): 0.9297357812499998,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 87.67258998792533,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 34.89492021558603,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 10.321315156250002,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): -2.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 24.1141875,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): -2.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 35.13385974696722,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): -2.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'D'): 6.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'U'): -2.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'U'): -2.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 18.541249999999998,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): -2.5,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 124.85087574587673,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 79.28331628429189,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 39.74153568745444,
    ((('U', -1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 11.637593295616453,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): -0.46402843749999967,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 1.11136754685459,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): 2.8525,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'D'): 8.025261215232423,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 16.678332320247854,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 0.5695491695118604,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 5.753493841903493,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 19.287041354988794,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 6.016398086155334,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 8.04201336276428,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 9.709907739597988,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 1.95,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.95,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'D'): 19.419290727646846,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'L'): 4.624785802760361,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'R'): 15.248623791441865,
    ((('U', -1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'U'): 7.292686421162347,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 2.8525,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): -0.75,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 147.80011523883988,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 141.6038211662712,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 211.83759145301076,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 0.12749999999999995,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'D'): 1.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'D'): 1.95,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'L'): 8.599949163836571,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'R'): -2.5,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'D'): 1.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'U'): -0.5,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 52.59094489126167,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 44.21389560144782,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 59.011901458719706,
    ((('U', -1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 49.88397836882898,
    ((('U', -1, -1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 18.110635117245245,
    ((('U', -1, -1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): -0.01704988359374937,
    ((('U', -1, -1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 4.486916983027404,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -49.27885795407468,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -49.27368030546318,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): -49.458761862123566,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'L'): -10.007892585159873,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'R'): -13.935391747254652,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'U'): -13.921837256678725,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): -0.5,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): 0.5,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'U'): -2.5,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'L'): 9.153350958232235,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'R'): 18.01119486025815,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'U'): 1.4869169830274043,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -49.999999999999915,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): -49.999999999999915,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): -13.897652350042932,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): -9.999351309116845,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'U'): -12.044788651798207,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): -49.999999999999915,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): -49.999999999999915,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): -49.999999999999915,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 6.284638953256898,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 10.019672413288252,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): 5.140666819024141,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 13.000000000000053,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 20.000000000000057,
    ((('U', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 13.000000000000053,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): 13.269937500000001,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): -2.5,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 19.760596926194385,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 7.892748928114308,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): 12.713814532066596,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 176.41597045809183,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 213.20214213507788,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): 175.9864375905749,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'L'): 492.39943507399954,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'R'): 59.57972277851563,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'U'): -2.5,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'L'): 19.97585527359894,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'R'): 14.450289409461632,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'U'): 11.066336145215859,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 20.000979860271414,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 13.381553826563763,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 13.030084243367646,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'L'): 432.86029156194707,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'R'): 281.8622706912281,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'U'): 179.1886263037084,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 21.99590897772442,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 23.74819641061614,
    ((('U', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 23.67737781781404,
    ((('U', -1, -1), ('X', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -49.999999999999915,
    ((('U', -1, -1), ('X', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): -49.999999999999915,
    ((('U', -1, -1), ('X', -1, -1), ('X', -1, -1), ('X', -1, -1)), 'U'): -49.99999999999834,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): -9.991432861770543,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -13.89570041629546,
    ((('U', -1, 1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): -13.315980336460317,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 196.3020519625251,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 369.85426602514735,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): 162.6844197113163,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): -0.5,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 27.61189587671901,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): -10.597263066080131,
    ((('U', -1, 1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -13.93328606684586,
    ((('U', -1, 1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): -13.544721503269969,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): -49.718159324981755,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): -49.71616854083126,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): -49.046123436644926,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): -49.72185149000509,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): -3.75,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): -2.5,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 63.53527017492442,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'D'): 12.083050969496192,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'L'): 18.52226110433436,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'R'): 21.848273361151072,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'U'): 14.693435211723537,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): -49.999999999999915,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): -49.999999999999915,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): -49.999999999999915,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, -1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): -49.99979033419727,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): -49.999792491376475,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): -49.99979113036285,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): -49.999799641181696,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): -2.5,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): -2.5,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 19.245924564387675,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 2.4536489913683672,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'D'): 13.218204969716416,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'L'): 13.223507746686412,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'R'): 20.000529477844914,
    ((('U', -1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'U'): 13.261721103452304,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'D'): 47.432372553461875,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'L'): 121.2129915366803,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'R'): 48.682372553461875,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 11.768997577326132,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 19.417112630984406,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 5.652610011861968,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): 6.801071570801485,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 147.29018787466265,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 248.5936803784694,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 150.8408998185502,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 147.60893110062534,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'D'): 126.15918918826009,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'L'): 147.26641497398458,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'R'): 140.8354453675561,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'U'): 53.512786479920635,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'D'): 17.884272176376868,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'L'): 22.287075611108456,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'R'): 21.25607872126153,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'U'): 10.392218136683635,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'D'): 13.62256047948275,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'L'): 20.16591184292956,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'R'): 13.772729692660516,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'U'): 13.635314212153911,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'D'): 106.46677888383596,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'L'): 187.7248897430615,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'R'): 189.52218660365497,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'U'): 79.47470068600397,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 33.3658228594569,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 37.3097547389477,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 37.41257368536172,
    ((('U', -1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 33.90091205442047,
    ((('U', -1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): -49.999999999999915,
    ((('U', -1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): -49.999999999999915,
    ((('U', -1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): -49.999999999999915,
    ((('U', -1, 1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'D'): -49.95313956246247,
    ((('U', -1, 1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'U'): -49.95177306701675,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): 22.986062579550783,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): 5.20375,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): -0.8249999999999997,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'D'): 0.975,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'L'): 0.9618123437499999,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'R'): 4.08604828125,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'U'): -2.7774015624999997,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): 10.000800270781905,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): 4.1725777246478675,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): 4.003793618509485,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 13.587529679316404,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 16.388374024806353,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 4.953973158051032,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 19.93464065341204,
    ((('U', -1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 11.052618097883396,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): 0.5,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 278.29473340863694,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 235.46914567673576,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 369.21417077176926,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): 171.67344039541933,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'D'): 4.866579167204948,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'U'): -3.3112500000000002,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'D'): 7.2261042687816595,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'R'): 0.8574821675781252,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'U'): 0.318966588537684,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 18.997171118729675,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 19.999999999983025,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 12.99977189151042,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 12.999226855240709,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'D'): 5.367087698402466,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'L'): 2.94039958475593,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 19.74009529910613,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 20.803035650837323,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 20.261420337450033,
    ((('U', -1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 15.184925467005268,
    ((('U', -1, 1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): 10.876807525009632,
    ((('U', -1, 1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): 6.794194013390534,
    ((('U', -1, 1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): 7.2885420524511595,
    ((('U', -1, 1), ('D', 1, -1), ('X', -1, -1), ('X', -1, -1)), 'D'): 1.42625,
    ((('U', -1, 1), ('D', 1, -1), ('X', -1, -1), ('X', -1, -1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 20.000000000000057,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 13.000000007150273,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 13.000000003725344,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 13.00000000017236,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): 8.6239981544708,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): -3.75,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'D'): 18.680586036442556,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'R'): 12.04745974860944,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'U'): -1.7658270156249993,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 19.999999999999957,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 12.99999999999995,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 12.999999999999954,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 22.150918363595903,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 13.126673708231253,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 13.000960047595369,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 15.089478448938243,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 19.99926204913161,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 12.815737432714762,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 18.88956963240663,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 12.504551433003373,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'D'): 19.99899611072057,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'L'): 12.643699383114452,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'R'): 19.799549970652834,
    ((('U', -1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'U'): 12.718978525490002,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'D'): 6.033254078125001,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 19.99999896092015,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 18.942588165526516,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 12.955576910160499,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): 12.912970998218327,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 109.92073883291835,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 521.2720823568898,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 158.28488139833678,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 104.69479309876706,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'D'): 18.01119486025815,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'U'): 4.843766807879815,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'D'): 15.481289180148684,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'U'): -2.5,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'D'): 19.686207886684755,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'L'): 19.08114270332174,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'R'): 9.131509900986874,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'U'): 10.468073675434823,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'D'): 5.298162187500001,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'R'): 121.8717575910569,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'U'): 57.17199052122458,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 33.99256244077684,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 24.839842351106057,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 45.688715284111495,
    ((('U', -1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 30.81107279930864,
    ((('U', -1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 20.000000000000057,
    ((('U', -1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): 13.000000000000053,
    ((('U', -1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 13.000000000000053,
    ((('U', -1, 1), ('D', 1, 1), ('X', -1, -1), ('X', -1, -1)), 'D'): 18.20510336138298,
    ((('U', -1, 1), ('D', 1, 1), ('X', -1, -1), ('X', -1, -1)), 'U'): 7.17096371920601,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): 0.8418125000000001,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): 13.208331093750001,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'D'): 0.42500000000000027,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'U'): 6.5,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): 0.8574821675781252,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): 3.697505902753907,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 72.84752654559415,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 119.98478219644092,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): -2.5,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 19.130409531110352,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 59.7531886038572,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): 9.065795687109375,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 72.86460904054843,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 74.30685928733617,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 83.6640963671623,
    ((('U', 1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): 9.427630607616212,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 1.3065503611858498,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 1.0566490425781248,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 8.609248224378288,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'U'): 0.5,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 5.68228509687103,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 7.013808312929719,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 10.13144933534954,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 24.405804240935755,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 0.5,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'D'): 21.515724555307337,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'L'): 12.746239321755565,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'R'): 27.492447783503327,
    ((('U', 1, -1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'U'): 28.945309341070057,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 252.91289430969113,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 209.9930866476825,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 259.5380077105878,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 200.13050147290033,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'D'): 49.05937720211296,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'L'): 120.7653172213,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'R'): 3.967472681703482,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'U'): 73.97484816743275,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'U'): 3.3657956871093755,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'D'): 5.282516411488495,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'L'): 19.38872708173352,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'R'): 6.672642312233451,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'U'): 9.179168609744957,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'U'): 4.59639912337363,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 76.69617044757535,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 75.84349484674699,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 81.19693642359891,
    ((('U', 1, -1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 76.19306315139113,
    ((('U', 1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 8.977479349911373,
    ((('U', 1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): 9.032121816478972,
    ((('U', 1, -1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 23.928380358525583,
    ((('U', 1, -1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'U'): 6.226463974646924,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): 203.70078365713152,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): 194.05364622521267,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): 138.49037897890094,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): 258.97215039220697,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'D'): 116.62660311625956,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'R'): 8.351384375,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', -1, 1)), 'U'): 77.19988777605172,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'D'): 55.742577778329654,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): 31.677206710225704,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): 37.089706710225705,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, -1)), 'U'): 90.43758023900608,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'D'): 85.2682204351945,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'L'): 108.15613344597378,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'R'): 118.982616271166,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'U'): 153.38171585825478,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): 233.27736940799107,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): 204.43138656299928,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): 213.28176267682238,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'D'): 82.14999434492258,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): 0.42500000000000027,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): 114.05411942769024,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, -1)), 'U'): 158.01515948270486,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 146.5014347716435,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 133.15064927213803,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 133.4717841582865,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 146.49659480733158,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): 237.92130130057114,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 188.00079534947378,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 174.17419625950913,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): 202.91747169550257,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 112.00204358627673,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 109.68342976266212,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 105.67811782390916,
    ((('U', 1, -1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 160.17852253539252,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): 214.99633724329624,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): 148.19721695259113,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): 170.96168007776714,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 166.26101034986993,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 155.273059142195,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): 166.91312046262607,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 175.90799972468199,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 187.7608228385878,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 187.3559748827049,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): 169.61397443573867,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'D'): 184.81310069858156,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'L'): 170.3808971611994,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'R'): 181.04083462380328,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'U'): 180.04036337110142,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'D'): 142.0974587890205,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'L'): 137.70392732237008,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'R'): 138.2361316030931,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'U'): 138.6286741671156,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 73.75651598484546,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 67.8322244399667,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 62.53517131930876,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 66.90381455944039,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'D'): 132.01726155324255,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'L'): 125.5956728243128,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'R'): 143.8202800226378,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'U'): 128.07177067704765,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 74.6438322025378,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 126.23715396130419,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 123.25545431026526,
    ((('U', 1, -1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 136.6846610970582,
    ((('U', 1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): 194.12948683753962,
    ((('U', 1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): 188.4275255077892,
    ((('U', 1, -1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): 325.4132506146433,
    ((('U', 1, -1), ('D', 1, -1), ('X', -1, -1), ('X', -1, -1)), 'D'): 55.07963018356942,
    ((('U', 1, -1), ('D', 1, -1), ('X', -1, -1), ('X', -1, -1)), 'U'): 64.06305261015538,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 132.99263689809445,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 81.3657273234527,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 22.59769970725527,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 466.8933895507588,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 60.029242273437504,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 265.9194864179481,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'L'): 5.843562499999999,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'U'): 18.541249999999998,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 147.00497051529527,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 118.82391283740692,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 140.80362130099752,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 153.0067467639957,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 21.86625,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'U'): 125.7019172904408,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): 24.1141875,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 140.27324573043978,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 151.9643487294585,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 151.17836373113983,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 452.0512846202762,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'D'): 3.9250000000000003,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'U'): 53.19330643242188,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'U'): 32.905665625,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'L'): 3.709875,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'R'): -1.216375,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'U'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'U'): 6.5,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 121.9959921755303,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 123.83318127467419,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 119.33417056007806,
    ((('U', 1, -1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 136.5576109025857,
    ((('U', 1, -1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): 102.37379850001201,
    ((('U', 1, -1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 148.77283788701357,
    ((('U', 1, -1), ('D', 1, 1), ('X', -1, -1), ('X', -1, -1)), 'D'): 0.0,
    ((('U', 1, -1), ('D', 1, 1), ('X', -1, -1), ('X', -1, -1)), 'U'): 43.75534393242188,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): 7.460894866432712,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): 13.434901983406599,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): 40.355392028422294,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'U'): 1.95,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): 1.8841729843750006,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): 23.795105783648143,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 14.951942782935399,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 12.754490901613975,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 9.748757399848227,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 21.711408372279806,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 3.206591804666016,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 6.42628004685459,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): 24.562648563004046,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 14.160219513224547,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): 1.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): 3.709875,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 106.29351736930158,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 202.3738831157056,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 51.07644614816223,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): 76.50378963300106,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'U'): 3.709875,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'U'): 2.8525,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 6.033254078125001,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'U'): 2.8525,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 21.354830091523638,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 19.283365628287953,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 24.170609036147642,
    ((('U', 1, 1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 21.455048421084484,
    ((('U', 1, 1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): 10.096263717633038,
    ((('U', 1, 1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): 18.20510336138298,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 14.24269912619867,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 13.00916078602351,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 15.156753394544442,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 21.228312772502353,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'L'): -0.04999999999999999,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', -1, 1)), 'U'): 2.8525,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): -2.5,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): 8.025261215232423,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('R', 1, 1)), 'U'): 5.298162187500001,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 12.999999999999961,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 12.999999999999961,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 19.999999999999957,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 13.017644626437342,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 12.78309124557106,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 13.000921779902031,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 20.00000032817241,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 12.785794906637701,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 12.987118934682616,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 18.400983465115992,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 19.99973545500226,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'D'): 12.982839699184023,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'L'): 12.05171401267066,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'R'): 18.440277152316128,
    ((('U', 1, 1), ('D', -1, 1), ('L', -1, 1), ('R', 1, 1)), 'U'): 19.99993028707721,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'D'): 3.6768776082696313,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, -1)), 'U'): 16.31948179528848,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 5.666197828569354,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 8.649785802760361,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): 19.249517215777676,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 104.80388283216303,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 107.73999845812432,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 132.4036042202789,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 124.60016646786178,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'D'): 75.07761313115203,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'L'): 139.38385433999588,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'R'): 113.24402866882394,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, -1), ('R', 1, 1)), 'U'): 25.69775879021984,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'D'): 2.3993818125831914,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'L'): 5.83037892856222,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'R'): 2.827755312478516,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, -1)), 'U'): 12.452927949293848,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'D'): 13.855400892381265,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'L'): 18.103920401744034,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'R'): 12.968688484253068,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', -1, 1)), 'U'): 20.14354113869747,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'D'): -0.8305562500000001,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'R'): 191.58709706250232,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, -1)), 'U'): 89.06913117812604,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 41.27879847200661,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 48.476805220096416,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 46.121808445350815,
    ((('U', 1, 1), ('D', -1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 49.00429023744834,
    ((('U', 1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 13.113256461551046,
    ((('U', 1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): 13.019885371000363,
    ((('U', 1, 1), ('D', -1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 26.000617556357685,
    ((('U', 1, 1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', -1, 1), ('X', -1, -1), ('X', -1, -1)), 'U'): 8.6239981544708,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): 15.9399509375,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): 48.88351103705291,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): 22.12651698149671,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('R', -1, -1)), 'U'): 127.32116410544945,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('R', 1, 1)), 'U'): 2.8525,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): 63.558938522559565,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, -1), ('X', -1, -1)), 'U'): 54.60424641104722,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): 161.64416888182302,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): 97.63306888646838,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', -1, 1)), 'U'): 181.92951440049683,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): 24.9715625,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, -1)), 'U'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 121.73159274012689,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 68.73652808003024,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 134.34860244128427,
    ((('U', 1, 1), ('D', 1, -1), ('L', -1, 1), ('R', 1, 1)), 'U'): 133.60209332678002,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, -1)), 'U'): 1.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 0.95,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', -1, 1)), 'U'): 17.633193749999997,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 224.6638983674345,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 227.4987496980831,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 363.5061737484557,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, -1)), 'U'): 226.8393338985672,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'R'): 9.816736093749999,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, -1), ('R', 1, 1)), 'U'): 44.22949092013769,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, -1)), 'U'): 38.29771548965613,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 48.34360888730934,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 47.540544729295284,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', -1, 1)), 'U'): 99.25226109517894,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, -1)), 'U'): 17.76263235450432,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 139.93839757305818,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 134.64432474858407,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 135.49610443434298,
    ((('U', 1, 1), ('D', 1, -1), ('L', 1, 1), ('R', 1, 1)), 'U'): 139.3710440731786,
    ((('U', 1, 1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, -1), ('X', -1, -1), ('R', -1, -1)), 'U'): 376.72434885750357,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'D'): 29.52276722094271,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'L'): 20.322585433764225,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'R'): 21.83754975114318,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, -1)), 'U'): 25.155374806579694,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, 1)), 'D'): 0.0,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, 1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', -1, 1)), 'U'): 16.835139243007816,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'D'): 49.724841619729254,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'L'): 17.937281742199477,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'R'): 22.564685437106675,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, -1)), 'U'): 70.22058005146954,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'D'): 27.458299028148225,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'L'): 25.73931885471991,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'R'): 30.480500542726165,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('R', 1, 1)), 'U'): 33.62729251290662,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'D'): 23.694367111767477,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'L'): 17.097640612519047,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, -1), ('X', -1, -1)), 'U'): 27.200868966521963,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'D'): 20.359897846369908,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'R'): 13.153772562222814,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, -1)), 'U'): 31.483838449334353,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'D'): 21.84627197557951,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'L'): 17.04699512980563,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'R'): 19.200273904407055,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', -1, 1)), 'U'): 21.83474741443846,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'D'): 20.447369012604206,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'L'): 16.732561288796973,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'R'): 19.773100594911366,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, -1)), 'U'): 20.467299503704304,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'D'): 34.89645811594687,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'L'): 31.324161271308682,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'R'): 41.27650729161344,
    ((('U', 1, 1), ('D', 1, 1), ('L', -1, 1), ('R', 1, 1)), 'U'): 47.71700639322796,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'D'): 57.689387839456984,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'L'): 0.0,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'R'): 11.341950475000488,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, -1)), 'U'): 84.70631175421951,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'D'): 22.584989193277288,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'L'): 19.750644463118633,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'R'): 16.812772038652806,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', -1, 1)), 'U'): 22.45934717275575,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'D'): 184.0696593582426,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'L'): 250.28192765395005,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'R'): 329.5691158979488,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, -1)), 'U'): 226.73777852630207,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'D'): 124.90545782343725,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'L'): 140.85830788939407,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'R'): 129.62569160394395,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, -1), ('R', 1, 1)), 'U'): 125.04248645127234,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'D'): 35.02573773241869,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'L'): 34.47462609512041,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'R'): 32.84177020483605,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, -1)), 'U'): 34.381205310808646,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'D'): 269.30079064829476,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'L'): 43.82397599203351,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'R'): 42.437583816239055,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', -1, 1)), 'U'): 43.1178463285618,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'D'): 123.92434122441742,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'L'): 130.34897360809754,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'R'): 121.47681796262607,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, -1)), 'U'): 124.37310478692078,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'D'): 44.926155154970786,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'L'): 45.050704955357006,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'R'): 45.18556178460055,
    ((('U', 1, 1), ('D', 1, 1), ('L', 1, 1), ('R', 1, 1)), 'U'): 40.75063197370062,
    ((('U', 1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'D'): 20.65202520189878,
    ((('U', 1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'R'): 15.368800096534185,
    ((('U', 1, 1), ('D', 1, 1), ('X', -1, -1), ('R', -1, -1)), 'U'): 20.64214577565454,
    ((('U', 1, 1), ('D', 1, 1), ('X', -1, -1), ('X', -1, -1)), 'D'): 24.27631444534864,
    ((('U', 1, 1), ('D', 1, 1), ('X', -1, -1), ('X', -1, -1)), 'U'): 24.852723403537645,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'D'): -49.83399037605486,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -49.34413735226026,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -49.18667915210515,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'D'): -13.185842512528666,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'L'): -9.88813867247641,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', -1, 1)), 'R'): -13.574361080017415,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'D'): -2.5,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'L'): -0.5,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, -1)), 'R'): 8.064081093750001,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'D'): 3.593766807879815,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'L'): -0.5,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('R', 1, 1)), 'R'): 17.558269025263037,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'D'): -49.999999999999915,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -49.999999999999915,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'D'): -14.06962420357147,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'L'): -14.01094226894741,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, -1)), 'R'): -10.350431133316972,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'D'): -49.999999999999915,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): -49.999999999999915,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): -49.999999999999915,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'D'): 5.175782153496915,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): 4.483260074689888,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 10.095067935823701,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'D'): 13.146041488837943,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): 13.020548923715232,
    ((('X', -1, -1), ('D', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 20.0024670478753,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'D'): -2.5,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'L'): 4.012630607616211,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, -1)), 'R'): 0.33472187499999995,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'D'): 9.628460149826068,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 11.386615042842198,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 7.168947029535962,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'D'): 204.9110298305898,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 255.99455626811414,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 221.97222045332438,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'D'): 155.51276637697197,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'L'): 194.14753088404856,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'R'): 173.87929835910097,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'D'): 7.771979740235219,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'L'): 20.06280929691122,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, -1)), 'R'): 15.0057863543515,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'D'): 12.999999999999956,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'L'): 19.999999999999957,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', -1, 1)), 'R'): 12.999999999892825,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'D'): -2.5,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'L'): 179.42088524508722,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'R'): 104.6216687026349,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'D'): 17.751542108021727,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 20.43265292254849,
    ((('X', -1, -1), ('D', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 20.435904556983523,
    ((('X', -1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'D'): -49.999999999999915,
    ((('X', -1, -1), ('D', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -49.999999999999915,
    ((('X', -1, -1), ('D', -1, -1), ('X', -1, -1), ('X', -1, -1)), 'D'): -49.999999999999915,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'L'): -7.5599375,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, -1), ('R', -1, -1)), 'R'): -5.569687500000001,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, -1), ('X', -1, -1)), 'L'): -49.999999999999915,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'L'): -48.412215184521855,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', -1, 1)), 'R'): -48.36616602135619,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'L'): -2.5,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, -1)), 'R'): 4.012630607616211,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'L'): -2.5,
    ((('X', -1, -1), ('X', -1, -1), ('L', -1, 1), ('R', 1, 1)), 'R'): 14.452208537563319,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'L'): 2.262190625,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', -1, 1)), 'R'): 0.0,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'L'): 111.92756403581046,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, -1)), 'R'): 134.2616496258768,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'L'): 34.438054218750004,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, -1), ('R', 1, 1)), 'R'): 3.9250000000000003,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'L'): 1.0,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, -1)), 'R'): 0.0,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'L'): 17.62212085939667,
    ((('X', -1, -1), ('X', -1, -1), ('L', 1, 1), ('R', 1, 1)), 'R'): 16.78732933151268,
    ((('X', -1, -1), ('X', -1, -1), ('X', -1, -1), ('R', -1, -1)), 'R'): -49.999999999999915}
)

    if (traningTime):
        trainingAI(environment=environment, ai_player=ai_player)
    
    jogador = sys.stdin.readline().strip()
    round = 1
    environment.reset_game()
    if (jogador == "A"):
        while True:
            if round % 2 != 0:
                if round % 10 in [1,2,3,6,7,8]:
                    posicao = get_best_pick(environment.board, Piece(get_color(round), 1))
                    environment.put_piece(position_x=posicao[0], position_y=posicao[1], piece=Piece(color=get_color(round), value=1))
                    print(f'{posicao[0]+1}{posicao[1]+1}')
                    sys.stdout.flush()
                else:
                    state = calculate_ai_state(environment.board, environment.calculate_board, environment.is_board_empty(), environment.get_score)
                    action = ai_player.get_action(state)
                    environment.calculate_board(action)
                    print(action)
                    sys.stdout.flush()
            else:
                read_move(environment, round)
            round += 1
    else:
        while True:
            if jogador == 'Quit':
                break
            if round % 2 == 0:
                if round % 10 in [1,2,3,6,7,8]:
                    posicao = get_best_pick(environment.board, Piece(get_color(round), 1))
                    environment.put_piece(position_x=posicao[0], position_y=posicao[1], piece=Piece(color=get_color(round), value=1))
                    print(f'{posicao[0]+1}{posicao[1]+1}')
                    sys.stdout.flush()
                else:
                    state = calculate_ai_state(environment.board, environment.calculate_board, environment.is_board_empty(), environment.get_score)
                    action = ai_player.get_action(state)
                    environment.calculate_board(action)
                    print(action)
                    sys.stdout.flush()
            else:
                read_move(environment, round)
            round += 1
    