import random
import math
import time

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

def get_legal_moves(board: Board) -> list[Move]:
    """Get a list of all the empty tiles on the board."""
    moves: list[Move] = []
    for x in range(board.size):
        for y in range(board.size):
            if board.tiles[x][y].colour is None:
                moves.append(Move(x, y))
    return moves


def clone_board(board: Board) -> Board:
    """Clone the existing board by copying tile colours over."""
    new_board = Board(board_size=board.size)
    for x in range(board.size):
        for y in range(board.size):
            new_board.tiles[x][y].colour = board.tiles[x][y].colour
    return new_board


def apply_move(board: Board, move: Move, colour: Colour) -> None:
    """Set the chosen tile to the given colour."""
    x, y = move.x, move.y 
    board.set_tile_colour(x, y, colour)

class MCTSNode:
    """
    A node in the MCTS tree.

    board          : position BEFORE player_to_move makes a move
    player_to_move : whose turn it is in this position
    parent         : parent node (None for root)
    move           : move that led from parent -> this node (None for root)
    children       : list of child nodes
    untried_moves  : legal moves from this position not expanded yet
    visits         : how many times this node was visited
    value          : cumulative reward from our agent's point of view
    """

    def __init__(
        self,
        board: Board,
        player_to_move: Colour,
        parent: "MCTSNode | None" = None,
        move: Move | None = None,
    ):
        self.board = board
        self.player_to_move = player_to_move
        self.parent = parent
        self.move = move

        self.children: list[MCTSNode] = []
        self.untried_moves: list[Move] = get_legal_moves(board)
        self.visits: int = 0
        self.value: float = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def _has_someone_won(self) -> bool:
        # This board is BEFORE player_to_move moves,
        # so the previous mover could have won.
        return (
            self.board.has_ended(Colour.RED)
            or self.board.has_ended(Colour.BLUE)
        )

    def is_terminal(self) -> bool:
        return self._has_someone_won()


    def ucb1(self, child: "MCTSNode", exploration: float = 1.4) -> float:
        """UCB1 formula for balancing exploration vs exploitation."""
        if child.visits == 0:
            return float("inf")  # always explore unvisited child first
        exploit = child.value / child.visits
        explore = exploration * math.sqrt(math.log(self.visits) / child.visits)
        return exploit + explore

    def select_child(self, exploration: float = 1.4) -> "MCTSNode":
        """Pick child with highest UCB1 score."""
        return max(self.children, key=lambda c: self.ucb1(c, exploration))


    def add_child(self, move: Move) -> "MCTSNode":
        """
        Take one untried move from this node, create the resulting child node,
        and return it.
        """
        new_board = clone_board(self.board)
        apply_move(new_board, move, self.player_to_move)
        next_player = Colour.opposite(self.player_to_move)

        child = MCTSNode(
            board=new_board,
            player_to_move=next_player,
            parent=self,
            move=move,
        )
        self.children.append(child)
        self.untried_moves.remove(move)
        return child


    def update(self, reward: float) -> None:
        """
        Update this node's stats with the result of a simulation.
        +1 win, -1 loss.
        """
        self.visits += 1
        self.value += reward

def play_from_node(node: MCTSNode, my_colour: Colour) -> float:
    """
    From this node's position, play random moves until someone wins.
    Return +1 if my_colour wins or -1 if my_colour loses or 0 for draw (shouldn't happen in Hex).
    """
    board = clone_board(node.board)
    current_player = node.player_to_move

    while True:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            # Shouldn't happen in real Hex, but safe:
            return 0.0

        move = random.choice(legal_moves)
        apply_move(board, move, current_player)

        # Only the player who just moved can have just won
        if board.has_ended(current_player):
            if current_player == my_colour:
                return 1.0
            else:
                return -1.0

        current_player = Colour.opposite(current_player)


def mcts_search(
    root_board: Board,
    my_colour: Colour,
    max_iterations: int = 1000,
    max_time_seconds: float | None = 2,
) -> Move:
    """
    Run MCTS from root_board for my_colour and return the chosen Move.

    max_iterations  : hard cap on how many simulations we run
    max_time_seconds: soft time budget per move (to stay within 5 min total)
    """
    root = MCTSNode(
        board=clone_board(root_board),
        player_to_move=my_colour,
        parent=None,
        move=None,
    )

    start_time = time.perf_counter()
    it = 0

    while True:
        if max_time_seconds is not None and (time.perf_counter() - start_time) > max_time_seconds:
            break
        if it >= max_iterations:
            break
        it += 1

        node = root

        # 1) SELECTION: move down while node is fully expanded and not terminal
        while node.is_fully_expanded() and node.children and not node.is_terminal():
            node = node.select_child()

        # 2) EXPANSION: if non-terminal and has untried moves, expand one
        if not node.is_terminal() and node.untried_moves:
            move = random.choice(node.untried_moves)
            node = node.add_child(move)

        # 3) SIMULATION: random playout from this node
        reward = play_from_node(node, my_colour=my_colour)

        # 4) BACKPROPAGATION: update nodes along path back to root
        while node is not None:
            node.update(reward)
            node = node.parent

    # After search: pick child with the most visits
    if not root.children:
        # No children (e.g. board full / very tiny time budget) – just play random legal move
        legal_moves = get_legal_moves(root_board)
        return random.choice(legal_moves)

    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move


# To run the agent:
# python3 Hex.py -p1 "agents.Group16.HexAgent HexAgent" -p1Name "Group16" -p2 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p2Name "TestAgent"
# python3 Hex.py -p1 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p2Name "TestAgent" -p2 "agents.Group16.HexAgent HexAgent" -p1Name "Group16" 

class HexAgent(AgentBase):
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Swap on turn 2 otherwise use MCTS to select the move
        """

        # Always swap on turn 2
        # TODO: add some smarter swap logic
        if turn == 2:
            return Move(-1, -1)

        chosen_move = mcts_search(
            root_board=board,
            my_colour=self.colour,
            max_iterations=5000,     # max number of random plays
            max_time_seconds=2    # time limit per move
        )
        return chosen_move
