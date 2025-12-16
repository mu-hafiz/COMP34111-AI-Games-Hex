import random
import math
import time
import multiprocessing

from multiprocessing import Pool
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from itertools import combinations


def setup_walls(colour):
    wall_list = []
    if colour == colour.RED:
        for i in range(11):
            wall_list.append(Move(-1,i))
            wall_list.append(Move(11,i))
    if colour == colour.BLUE:
        for i in range(11):
            wall_list.append(Move(i,-1))
            wall_list.append(Move(i,11))
    return wall_list

def get_colour_moves(board: Board, colour: Colour) -> list[Move]:
    """Get a list of all the empty tiles on the board."""
    moves: list[Move] = []
    for x in range(board.size):
        for y in range(board.size):
            if board.tiles[x][y].colour is colour:
                moves.append(Move(x, y))
    return moves


def identify_decision(information_set):
    """
    This is the part that decides the decision we should make given the change in boardstate
    If list of pairs is empty: we dont care
    If len 1: This case
    I.e., the cases 

    Returns as the key of execution_flow_dict
    """
    # We want to unpack the information set
    # So that we know what each index corresponds to
    # The permutation of available information will decide
    # What flag we want to activate
    # We should probably print what flag we decided during each move, so we know what decision we made during debugging

    """
    It is an inevitability that the expected information_set will change over time as our bot changes
    As a result, we should have a written out version as such
    {
    information_set[piece_of_information] = (What is the information,What decisions does it affect)
    }
    """

    """
    information_set[1] : (bridges_lost, undecided)
        explanation - 
    """

    # Priority ranking of our decisions, lower index = higher priority
    priority_list = ["Defend", "Win", "Play Best Fair Move","Swap","Central Move","Potential Connections"]




    # We just add things to this list as we go
    list_of_decisions = []
    if information_set["Turn"] == 1:
        list_of_decisions.append("Play Best Fair Move")
    if information_set["Turn"] == 2:
        # Figure out if we want to swap or not
        Swap_Decision = should_swap(information_set["Board"],information_set["Opp Move"])
        if Swap_Decision:
            list_of_decisions.append("Swap")
        else:
            list_of_decisions.append("Central Move")
    if information_set["Turn"] == 3:
        # Check whether our opponent swapped us or not
        if information_set["Opp Move"] == Move(-1,-1):
            # We got swapped
            list_of_decisions.append("Central Move")
    if len(information_set["Lost Bridges"]) == 0:
        # If the enemy never threatened anything, play
        list_of_decisions.append("Potential Connections")
    if len(information_set["Lost Bridges"]) != 0 and information_set["Opp Move"] != Move(-1,-1):
        # The enemy has threatened a strong connection of ours
        list_of_decisions.append("Defend")
    if check_reach(information_set["Colour"], information_set["Board"], information_set["Bridges"]):
        list_of_decisions.append("Win")
    
    # print(list_of_decisions)

    """
    Determine priority
    """

    list_of_decisions = sorted(list_of_decisions,key = lambda c: priority_list.index(c))
    # just return the first thing we think of doing for now
    print(list_of_decisions)
    return list_of_decisions

def execution_flow(old_current_bridges,turn,colour,board,opp_move):
    """

    Gets our information
    Spits out the set of moves we should consider

    Step by step
    execution_flow_dict{
        str : corresponding function for decision 
        }    
    """

    # Generate our current information
    new_current_bridges = generate_current_bridges(colour,board)
    # print(new_current_bridges)
    # Using our new current bridges and our old current bridges
    # Check what's different
    bridges_lost = compare_previous_board_state(old_current_bridges,new_current_bridges)
    # Returns a List(tuple) containing the difference of old bridges compared to new bridges

    # print(bridges_lost)
    
    # This is the function that takes all the information we have
    # And all the factors we need to consider
    information_set = {"Lost Bridges": bridges_lost, "Turn": turn, "Colour":colour, "Board": board, "Opp Move": opp_move, "Bridges": new_current_bridges}
    set_of_constraints = identify_decision(information_set)

    execution_flow_dict = {
        "Play Best Fair Move" : get_fair_first_moves,
        "Swap" : swap,
        "Central Move" : central_move,
        "Potential Connections" : generate_potential_connections,
        "Defend" : defend,
        "Win": win
    }
    function_parameters ={
        # We dont know how to not give parameters this way, dont question it
        "Play Best Fair Move" : [],
        "Swap": [],
        "Central Move" : [board],
        "Potential Connections": (colour,board),
        "Defend" : (bridges_lost,opp_move),
        "Win": [colour,board,new_current_bridges]
    }

    # When we consider all of the constraints that exist inside of decisions
    # We can do one of two things logically
    # Return all of them
    # Or return the best one for MCTS
    # This is just a matter of constraint relaxation

    collection_of_movesets = []
    
    for constraint in set_of_constraints:
        move_set = execution_flow_dict[constraint](*function_parameters[constraint])
        collection_of_movesets.append(set(move_set))


    return constraint_moveset(collection_of_movesets)

def check_reach(colour, board, bridges):
    """
    Given a board, can the agent win the game if it were to play all its strong connections?
    Returns True if we can
    Return False otherwise


    Eventually it would be nice to DFS this (via all of our strong connections) to find the shortest path
    """
    board_copy = clone_board(board)
    # Add strongly connected bridges to the board as if they've already been played

    # Unpack bridges from list of pairs (tuples) to 1d array of all bridges
    bridges = [x for sublist in bridges for x in sublist]
    bridges = list(set(bridges))

    for move in bridges:
        board_copy.tiles[move.x][move.y].colour = colour

    return board_copy.has_ended(colour)

def win(colour,board,new_current_bridges):
        """
        Reduce the board
        Note: might not always give the same path 
        """
        # Check if we win
        smallest_so_far = new_current_bridges.copy()

        #Unpack bridges    
        for bridge_pair in new_current_bridges:
            # This is a pair of bridges
            test_bridges = smallest_so_far.copy()
            test_bridges.remove(bridge_pair)
            if check_reach(colour,board,test_bridges):
                smallest_so_far = test_bridges.copy()

        smallest_so_far = list(set([x for sublist in smallest_so_far for x in sublist]))
        return smallest_so_far    
        

def defend(bridges_lost,opp_move):
    # Go through all pairs of bridges inside bridges_lost

    opposite_list = list(set([x for sublist in bridges_lost for x in sublist]))
    opposite_list.remove(opp_move)
    return opposite_list

def constraint_moveset(movesets):
    current = movesets[0]
    for move_set in movesets:
        if len(current & move_set) == 0:
            return list(current)
        current = current & move_set
    return current

def swap():
    return [Move(-1,-1)]

def central_move(board):
    """
    Check if a move is in the central area of the board.
    """
    central_corners = [(2, 0), (8, 10), (2, 0), (8, 10)]

    legals = get_legal_moves(board)

    central_moves = []
    for x in range(2,9):
        for y in range(11):
            if (x, y) not in central_corners:
                central_moves.append(Move(x,y))
    obtuse_corners = [Move(0, 10), Move(1, 9), Move(1, 10), Move(9, 0), Move(9, 1), Move(10, 0)]
    central_moves += obtuse_corners

    # Just make sure that the central move that we are trying to play is actually legal
    central_moves = set(central_moves) & set(legals)
    
    return list(central_moves)


def get_fair_first_moves():
    """
    Choose a fair first move to combat the swap rule.
    Returns a list of Move objects.
    """
    size = 11

    base_candidates = {Move(1, 2), Move(1, 7), Move(2, 5), Move(8, 5), Move(9, 2), Move(9, 7)}

    # Add edge candidates avoiding corners
    candidates = base_candidates.copy()
    for x in range(size):
        if x >= 2:                 # left edge
            candidates.add(Move(x, 0))
        if x <= size - 3:          # right edge
            candidates.add(Move(x, size - 1))
    
    return candidates

def generate_potential_connections(colour,board):
    """
    Go through all points that are our colours
    Run cardinal_dirs of that point
    Return new potential_connections
    Dictionary Potential connections{
        Tile: List[Tile]
        Potential connection : List of pairs bridges that are dependent on that potential connection
    } 
    """

    wall_list = setup_walls(colour)

    
    # We want a list of our tiles
    our_tiles = get_colour_moves(board,colour) + wall_list
    
    potential_connections = []

    # For each tile
    for tile in our_tiles:
        potential_connections += cardinal_dirs(tile,wall_list,None,board)
    return list(set(potential_connections))


def generate_current_bridges(colour, board,chosen_move = None):
    """
    Go through all points that are our colours
    Run cardinal_dirs of that point
    Return new current_bridges
    For triplet(a,b,c)
    If a,b is open, check if c is ours (wall or our actual move)
    If so, a,b are bridges
    Return new current_bridges

    This must be done at the start AND at the end of our turn
    """


    wall_list = setup_walls(colour)

    if chosen_move:
        board.tiles[chosen_move.x][chosen_move.y].colour = colour
    
    # We want a list of our tiles
    our_tiles = get_colour_moves(board,colour)
    # If we're retrospectively checking our current bridges we've generated
    # AFTER we've played our move
    # We have to add the move we've played to the list of our current tiles

    current_bridges = []

    # For each tile
    for tile in our_tiles:
        current_bridges += cardinal_dirs(tile,wall_list,colour,board)
    # print(wall_list)
    return list(set(current_bridges))


def compare_previous_board_state(old_current_bridges,new_current_bridges):
    """
    Check if old current_bridges is different to new current_bridges
    Should return a list of the pairs that are different
    i.e. has the enemy taken a bridge from us
    """
    # print("Old",old_current_bridges)
    # print("New",new_current_bridges)

    return list(set(old_current_bridges).difference(set(new_current_bridges)))
    


def get_legal_moves(board) -> list[Move]:
    """Get a list of all the empty tiles on the board."""
    moves: list[Move] = []
    for x in range(11):
        for y in range(11):
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
    obtuse_corners = [(0, 10), (1, 9), (1, size-1), (9, 0), (9, 1), (10, 0)]

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








def get_colour_moves(board: Board, colour: Colour) -> list[Move]:
    """Get a list of all the empty tiles on the board."""
    moves: list[Move] = []
    for x in range(11):
        for y in range(11):
            if board.tiles[x][y].colour is colour:
                moves.append(Move(x, y))
    return moves


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
    if root_allowed_moves:
        # keep only allowed moves in root.untried_moves
        root.untried_moves = root_allowed_moves
        # pre-expand each allowed move as a child so the search distributes sims among them


    start_time = time.perf_counter()
    it = 0

    workers = (
        multiprocessing.cpu_count()
    )  


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

            # 1) SELECTION: move down while node is fully expanded and not terminal
            while node.is_fully_expanded() and node.children and not node.is_terminal():
                node = node.select_child(my_colour, exploration=0.1)

            # 2) EXPANSION: if non-terminal and has untried moves, expand one
            if not node.is_terminal() and node.untried_moves:
                move = random.choice(node.untried_moves)
                node = node.add_child(move)

                child = node  # Checkpoint for rewarding rollouts
            # 3) SIMULATION: random playout from this node

            # If the node we picked wins in one move, play it
            if node.is_terminal(): 
                instant_victory = move
                return move

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
        children, 
        key=lambda c: (
            c.value / c.visits,  # primary: winrate (higher is better)
            -c.move_count_min if c.move_count_min != float('inf') else 0  # secondary: shortest winning rollout
        )
    )
    if instant_victory: return move
    return best_child.move


def cardinal_dirs(current_tile:Move, walls, flag, board):
    """
    Calculate all possible strong connections from a given tile
    We want to calculate POTENTIAL connections and EXISTING connections now

    Use flag to determine what we're trying to find
    If flag == None: We're looking for potential
    If flag == colour: We're looking for existing connections
    """
    x = current_tile.x
    y = current_tile.y
    
    # A dictionary of triplets (a,b,c) representing strong connections and the bridges between them
    # a : Bridge
    # b : Bridge
    # c : Strong connection
    dirs = {
        "N" : [Move(x-1,y),Move(x-1,y+1),Move(x-2,y+1)],
        "NE" : [Move(x-1,y+1),Move(x,y+1),Move(x-1,y+2)],
        "NW" : [Move(x-1,y),Move(x,y-1),Move(x-1,y-1)],
        "SE" : [Move(x,y+1),Move(x+1,y),Move(x+1,y+1)],
        "SW" : [Move(x,y-1),Move(x+1,y-1),Move(x+1,y-2)],
        "S" : [Move(x+1,y-1),Move(x+1,y),Move(x+2,y-1)],
        }


    legals = get_legal_moves(board)
    if flag:
        our_moves = get_colour_moves(board,flag)
    result = []

    if flag == None:
        for triplet in dirs.values():
            if triplet[0] in legals and triplet[1] in legals and triplet[2] in legals:
                result.append(triplet[-1])
    else:
        for triplet in dirs.values():
            if triplet[0] in legals and triplet[1] in legals:
                    if (triplet[2] in our_moves) or (triplet[2] in walls):
                        sorted_pair = tuple(sorted((triplet[0],triplet[1]), 
                        key=lambda move: (
                            move.x,  
                            move.y
                        )))
                        result.append(sorted_pair)
    return result


class RefactoredHexAgent(AgentBase):
    _board_size: int = 11

    """
    HexAgent class
    """


    def __init__(self, colour: Colour):
        self.walls = []

        # Old: Move : List[Move]
        # New: List(Tuple(Move,Move))
        self.current_bridges = []
        
        super().__init__(colour)


    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Decide on a move to make given the current turn, board state, and opponent's last move.
        Turn 1: Make a fair move to combat the swap rule.
        Turn 2: Decide whether to swap based on opponent's first move.
        Subsequent turns: Use MCTS to select the best move.
        """
        move_set = list(execution_flow(self.current_bridges,turn,self.colour,board,opp_move))


        if len(move_set) == 1:
            # If we only have one option
            chosen_move = move_set[0]
            # Don't bother checking others (case where hand is forced)
        else:
            # Otherwise, try to figure out which move is our best move
            chosen_move = mcts_search(
                root_board=board,
                my_colour=self.colour,
                max_iterations=5000,          # max number of random plays
                max_time_seconds=2,           # time limit per move
                report_top_k=1,               # show top-5 for normal turns
                root_allowed_moves=move_set
            )

        self.current_bridges = generate_current_bridges(self.colour,board,chosen_move)
        return chosen_move

# RUN CMDS

# Agent VS Naive:
# Red
# python3 Hex.py -p1 "agents.Group16.RefactoredHexAgent RefactoredHexAgent" -p1Name "Group16" -p2 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p2Name "TestAgent"
# Blue
# python3 Hex.py -p1 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p1Name "TestAgent" -p2 "agents.Group16.RefactoredHexAgent RefactoredHexAgent" -p2Name "G16"

# Agent VS Agent
# python3 Hex.py -p1 "agents.Group16.RefactoredHexAgent RefactoredHexAgent" -p1Name "G16Player1" -p2 "TestAgent" -p2 "agents.Group16.RefactoredHexAgent RefactoredHexAgent" -p2Name "G16Player2"

# To run the analysis over 100 games (this took 2-3 hours for me):
# python3 Hex.py -p1 "agents.Group16.RefactoredHexAgent RefactoredHexAgent" -p1Name "Group16" -p2 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p2Name "TestAgent" -a -g 50

# Playing main vs new agent
# python3 Hex.py -p1 "agents.Group16.RefactoredHexAgent RefactoredHexAgent" -p1Name "(New) Rewritten HexAgent" -p2 "agents.Group16.HexAgent HexAgent" -p2Name "(Old) MCTS Only" -a -g 50

# To play the agent against a human:
# python3 Hex.py -p1 "agents.Group16.RefactoredHexAgent RefactoredHexAgent" -p1Name "Group16" -p2 "agents.Human.HumanPlayer HumanPlayer" -p2Name "Human"
# python3 Hex.py -p2 "agents.Group16.RefactoredHexAgent RefactoredHexAgent" -p2Name "Group16" -p1 "agents.Human.HumanPlayer HumanPlayer" -p1Name "Human"


"""
Hex agent but it actually works this time
Heres Sasha's plan:

These are the different cases/stages of the game:
1. Its the start of the game and we are red so we use a fair first moveset and do mcts on that.
2. Its the second turn of the game and we are blue so we need to check if we want to swap reds move by checking if it falls inside the 'winning' region.
    a. If it does we swap and we are now red.
    b. If it doesnt we dont swap and we remain blue and we need to play a good starting move. I currently dont know what the moveset should be but making a 
       connection with a wall seems good for now.
3. Its the third turn of the game and we were red but then we got swapped so we are blue and have no tiles on the board. Same idea as case 2b we need to make a good move.
4. Its our turn and the enemy has played a tile that threatens one of our  existing strong connections. We have to defend ourselves and take the other side of the connection.
    a. It is a simple formation and the enemy's move only disrupts one of our connections so we automatically select to play the other side of the connection.
    b. It is a complex formation and the enemy's move disrupts multiple connections. We run mcts on a movese of defensive plays to find the best one.
5. Its our turn and the enemy has played a tile that is not threatening any of our existing strong connections. We can do mcts on a moveset of potential connections
6. We are winning and should fill in our connections.


can we combine movesets if multiple cases occur???? run mcts on the lot. itll decide what was best to do in that situation 


Note: 
Some cases can occur at the same time
its third turn, enemy didnt swap, enemy couldve fucked up a connection we had to the wall = defense time!!!
we are winning and enemy fucked up connection = defense time!!!
if enemy didnt fuck up connection and we are winning then just do the winning thing
if enemy 

"""

"""
Keep track of the previous current bridges to compare against the new one.
Start of the go generate a new bridges and pot cons 
current bridges should be stored as a list of tuples where each tuple is the two bridges??? maybe this makes things cleaner.
(pair1, pair2)
Pot cons can be dict like it was. the value at least matches the pairs in bcurrent bridges now
"""


"""
What we do each turn:
1. get our current bridges and pot cons
2. go through the cases


AWESOME NEW THINGS WE CAN DO BY MAKING MULTIPLE DECISIONS THAT CONSTRAIN MOVESET AND THEN FINDING THE MOVES THAT SATISFY THE MOST CONSTRAINTS 

1. Identifying weak connections we have and filling them in.
        Pattern: Two of our tiles on the same row or column with one blank tile between them
        Method: 
        Find all the empty tiles adjacent to our tiles.
        If they adjacent to exactly 2 on the same row or column then we have a weak connection and should fill it in.
        Note: This would work really nicely with potential connections, but it trumps potential connections if theres none in common.
        Two cases:
            1. If its both an adjacent tile and a potential connection then we should prioritise it.
            2. If its adjacent to two of our tiles AND they on same row and column then prioritise it.
        Either of those combos help us to fill in weak connections i think they are equal priority (in my basic test case flowers)

2. Identifying that the enemy has a weak connection and blocking it.
        Two patterns:
            1. Two of their tiles on the same row or column with one blank tile between
            2. Two of their tiles on the same row or column with two blank tiles between them
        Method:
        Find all the empty tiles adjacent to their tiles.
        If they adjacent to exactly 2 on the same row or column then we have a weak connection and should block it.
        Some check to see if their tile is two away from another one of their tiles in the same row or column
        Note: This would work really nicely with potential connections, but it trumps potential connections if theres none in common.
        We want to restrict our moveset to the empty tiles that satisfy these patterns.
        Pattern 2 is more important than pattern 1.


   






"""
