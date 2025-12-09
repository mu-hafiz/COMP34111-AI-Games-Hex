import random
import math
import time
import multiprocessing

from multiprocessing import Pool
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

def get_fair_first_moves(board: Board) -> set[tuple[int, int]]:
    """
    Choose a fair first move to combat the swap rule.
    Returns a list of Move objects.
    """
    size = board.size

    base_candidates = {(1, 2), (1, 7), (2, 5), (8, 5), (9, 2), (9, 7)}

    # Add edge candidates avoiding corners
    candidates = base_candidates.copy()
    for x in range(size):
        if x >= 2:                 # left edge
            candidates.add((x, 0))
        if x <= size - 3:          # right edge
            candidates.add((x, size - 1))
    return candidates

def is_central(move: Move, size: int) -> bool:
    """
    Check if a move is in the central area of the board.
    """
    central_corners = [(2, 0), (8, size-1), (2, 0), (8, size-1)]

    if (2 <= move.x <= size-3):
        for x, y in central_corners:
            if (move.x, move.y) == (x, y):
                return False
        return True
    
    return False

def should_swap(board: Board, opp_move: Move) -> bool:
    """
    Decide whether to swap based on opponent's first move.
    If opponent plays in central area or obtuse corners, swap.
    """
    size = board.size
    obtuse_corners = [(0, size-1), (1, size-2), (1, size-1), (9, 0), (9, 1), (10, 0)]

    if is_central(opp_move, size):
        return True

    for x, y in obtuse_corners:
        if (opp_move.x, opp_move.y) == (x, y):
            return True

    return False
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

        self.amaf_visits: int = 0
        self.amaf_value: float = 0.0
        
        self.move_count_sum: float = 0.0    # sum of rollout lengths (for avg)
        self.move_count_min: float = float('inf')  # NEW: track minimum rollout length

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def _has_someone_won(self) -> bool:
        # This board is BEFORE player_to_move moves,
        # so the previous mover could have won.
        return self.board.has_ended(Colour.RED) or self.board.has_ended(Colour.BLUE)

    def is_terminal(self) -> bool:
        return self._has_someone_won()

    def ucb1(
        self, child: "MCTSNode", exploration: float = 1.4, amaf_persistence: int = 300
    ) -> float:
        """UCB1 formula for balancing exploration vs exploitation."""
        # print("I've been visited {} times".format(self.visits))
        if child.visits == 0:
            return float("inf")  # always explore unvisited child first

        exploit = child.value / child.visits

        if child.amaf_visits > 0:
            amaf_exploit = child.amaf_value / child.amaf_visits
        else:
            amaf_exploit = 0

        alpha = max(0, (amaf_persistence - self.visits) / amaf_persistence)

        combined_exploit = ((1 - alpha) * exploit) + (alpha * amaf_exploit)

        explore = exploration * math.sqrt(math.log(self.visits) / child.visits)
        return combined_exploit + explore

    def select_child(self, col: Colour, exploration: float = 1.4) -> "MCTSNode":
        """Pick child with highest UCB1 score."""
        if col == self.player_to_move:
            return max(self.children, key=lambda c: self.ucb1(c, exploration))
        else:
            return min(self.children, key=lambda c: self.ucb1(c, exploration))

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

    def update(
        self, reward: float, rollout_moves: set[tuple[int, int]], root_colour: Colour
    ) -> None:
        """
        Update this node's stats with the result of a simulation.
        +1 win, -1 loss.
        if (reward == 1 and (self.player_to_move == root_colour)) or (reward == -1 and (self.player_to_move == Colour.opposite(root_colour))):           
            self.value += reward  
        """
        rollout_length = len(rollout_moves)
        
        for child in self.children:
            if child.move and (child.move.x, child.move.y) in rollout_moves:
                child.amaf_visits += 1
                child.amaf_value += reward
                child.move_count_sum += rollout_length
                if reward > 0:  # only track min for winning rollouts
                    child.move_count_min = min(child.move_count_min, rollout_length)

        self.amaf_visits += 1
        self.amaf_value += reward
        self.move_count_sum += rollout_length
        if reward > 0:
            self.move_count_min = min(self.move_count_min, rollout_length)

        self.value += reward
        self.visits += 1


"""
Non serialisable version of MCTS
The issue is that the function takes parameter "node" (obj), which is too heavy
We must create a serialisable function
"""


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


def play_from_node_S(
    board_state: Board, player_to_move: Colour, my_colour: Colour
) -> tuple[float, set[tuple[int, int]]]:
    """
    From this node's position, play random moves until someone wins.
    Return +1 if my_colour wins or -1 if my_colour loses or 0 for draw (shouldn't happen in Hex).
    Taken from Sasha's version and made serialisable by expanding the MCTSNode object into python types
    """
    board = clone_board(board_state)
    current_player = player_to_move
    rollout_moves = set()

    while True:
        legal_moves = get_legal_moves(board)
        if not legal_moves:
            # Shouldn't happen in real Hex, but safe:
            return (0.0, rollout_moves)

        move = random.choice(legal_moves)
        apply_move(board, move, current_player)
        rollout_moves.add((move.x, move.y))

        # Only the player who just moved can have just won
        if board.has_ended(current_player):
            if current_player == my_colour:
                return (1.0, rollout_moves)
            else:
                return (-1.0, rollout_moves)

        current_player = Colour.opposite(current_player)


"""
Functional version of parallelisation such that we can adapt it modularly
If you believe in YAGNI you can just delete this

def parallelised_rollouts(child_board: Board, child_player: Colour, my_col: Colour, workers):

    # Give each worker a lighter copy of the gamestate to run simulations on


    rollouts = [(child_board,child_player,my_col)]*workers

    with Pool(processes = workers) as pool:
        rewards = [pool.apply_async(play_from_node_S, copy) for copy in rollouts]
        rewards = [r.get() for r in rewards]

    # unpickle rewards
    return rewards  

"""


def mcts_search(
    root_board: Board,
    my_colour: Colour,
    max_iterations: int = 1000,
    max_time_seconds: float | None = 2,
    root_allowed_moves: list[Move] | None = None,
    report_top_k: int | None = None,             # how many top entries to report (None=off)
    exploration: float = 1.4,                    # exploration constant 
) -> Move:
    """
    Run MCTS from root_board for my_colour and return the chosen Move.

    max_iterations  : hard cap on how many simulations we run
    max_time_seconds: soft time budget per move (to stay within 5 min total)
    report_top_k    : if set, print top-k children by visits and by UCB1 at end of search
    report_verbose  : whether to print textual output (useful to toggle)
    exploration     : exploration constant used in UCB1 selection and reporting
    """
    root = MCTSNode(
        board=clone_board(root_board),
        player_to_move=my_colour,
        parent=None,
        move=None,
    )

     # If a restricted candidate list is provided, limit/expand the root to only those moves.
    if root_allowed_moves is not None:
        # keep only allowed moves in root.untried_moves
        root.untried_moves = [m for m in root.untried_moves if (m.x, m.y) in root_allowed_moves]
        # pre-expand each allowed move as a child so the search distributes sims among them
        for move in list(root.untried_moves):
            root.add_child(move)

    start_time = time.perf_counter()
    it = 0

    workers = (
        multiprocessing.cpu_count()
    )  # adjust this for ur pc (run nproc in terminal or tinker urself)

    instant_victory = None

    with Pool(processes=workers) as pool:
        while True:
            if (
                max_time_seconds is not None
                and (time.perf_counter() - start_time) > max_time_seconds
            ):
                break
            if it >= max_iterations:
                break

            node = root

            # bad version


            # 1) SELECTION: move down while node is fully expanded and not terminal
            while node.is_fully_expanded() and node.children and not node.is_terminal():
                node = node.select_child(my_colour, exploration=0.1)

            # 2) EXPANSION: if non-terminal and has untried moves, expand one
            if not node.is_terminal() and node.untried_moves:
                move = random.choice(node.untried_moves)
                node = node.add_child(move)

                child = node  # Checkpoint for rewarding rollouts
            # 3) SIMULATION: random playout from this node


            # how do we return the move we just played as a 
            if node.is_terminal(): 
                instant_victory = move


            

            """
            not serialisable
            reward = play_from_node(node, my_colour=my_colour)
            """

            """
            serialisable
            """
            # reward = play_from_node_S(node.board,node.player_to_move, my_colour) (this is the serial integration of parallelisation i.e. workers = 1)
            rollouts = [(child.board, child.player_to_move, my_colour)] * workers
            rollout_results = [
                pool.apply_async(play_from_node_S, args=args) for args in rollouts
            ]
            rollout_results = [r.get() for r in rollout_results]

            it += len(rollout_results)
            # 4) BACKPROPAGATION: update nodes along path back to root
            for reward, rollout_moves in rollout_results:
                node = child
                while node is not None:
                    node.update(reward, rollout_moves, my_colour)
                    node = node.parent
            if instant_victory: break

    # After search: pick child with the most visits
    if not root.children:
        # No children (e.g. board full / very tiny time budget) – just play random legal move
        legal_moves = get_legal_moves(root_board)
        return random.choice(legal_moves)

    # Optionally prepare and print top-k rankings
    if report_top_k is not None and report_top_k > 0 and root.children:
        # compute stats for each child
        def child_stats(child: MCTSNode):
            visits = child.visits
            amafvisits = child.amaf_visits
            winrate = (child.value / visits) if visits > 0 else 0.0
            ucb = root.ucb1(child, exploration)
            mv = child.move
            avg_move_count = (child.move_count_sum / amafvisits) if amafvisits > 0 else 0.0
            min_move_count = (child.move_count_min+1) if child.move_count_min != float('inf') else 0
            return {"move": (mv.x, mv.y), "visits": visits, "winrate": winrate, "ucb1": ucb, "avg_move_count": avg_move_count, "min_move_count": min_move_count}

        children = list(root.children)
        by_valuevisits = sorted(children, key=lambda c: c.value/c.visits, reverse=True)[:report_top_k]

        print("MCTS rankings (Top {}) after {} iterations".format(report_top_k, it))
        print("Top by winrate:")
        for c in by_valuevisits:
            s = child_stats(c)
            print(f"  move={s['move']} visits={s['visits']} winrate={s['winrate']:.3f} ucb1={s['ucb1']:.3f} min={s['min_move_count']:.0f} avg={s['avg_move_count']:.1f}")

    best_child = max(
        root.children, 
        key=lambda c: (
            c.value / c.visits,  # primary: winrate (higher is better)
            -c.move_count_min if c.move_count_min != float('inf') else 0  # secondary: shortest winning rollout
        )
    )
    if instant_victory: return move
    return best_child.move


# To run the agent:
# python3 Hex.py -p1 "agents.Group16.HexAgent HexAgent" -p1Name "Group16" -p2 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p2Name "TestAgent"
# python3 Hex.py -p1 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p1Name "TestAgent" -p2Name "Group16" -p2 "agents.Group16.HexAgent HexAgent"

# to be clear, these two commands change the names of which agent is the MCTS agent and which one is the random agent
# in particular, MCTS is called "G16" for 1st cmd
# and MCTS is called "TestAgent" for 2nd cmd
# im not changing it incase this is on purpose

# To play it against itself
# python3 Hex.py -p1 "agents.Group16.HexAgent HexAgent" -p1Name "G16Player1" -p2 "TestAgent" -p2 "agents.Group16.HexAgent HexAgent" -p2Name "G16Player2"

# To run the analysis over 100 games (this took 2-3 hours for me):
# python3 Hex.py -p1 "agents.Group16.HexAgent HexAgent" -p1Name "Group16" -p2 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p2Name "TestAgent" -a -g 50


class HexAgent(AgentBase):
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Decide on a move to make given the current turn, board state, and opponent's last move.
        Turn 1: Make a fair move to combat the swap rule.
        Turn 2: Decide whether to swap based on opponent's first move.
        Subsequent turns: Use MCTS to select the best move.
        """

        if turn == 2:
            if should_swap(board, opp_move):
                return Move(-1, -1)

        chosen_move = mcts_search(
            root_board=board,
            my_colour=self.colour,
            max_iterations=5000,          # max number of random plays
            max_time_seconds=2,           # time limit per move
            report_top_k=5,               # show top-5 for normal turns
            root_allowed_moves=get_fair_first_moves(board) if turn == 1 else None
        )
        return chosen_move
