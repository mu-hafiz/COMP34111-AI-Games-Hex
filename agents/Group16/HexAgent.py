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

    base_candidates = {Move(1, 2), Move(1, 7), Move(2, 5), Move(8, 5), Move(9, 2), Move(9, 7)}

    # Add edge candidates avoiding corners
    candidates = base_candidates.copy()
    for x in range(size):
        if x >= 2:                 # left edge
            candidates.add(Move(x, 0))
        if x <= size - 3:          # right edge
            candidates.add(Move(x, size - 1))
    
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






def get_colour_moves(board: Board, colour: Colour) -> list[Move]:
    """Get a list of all the empty tiles on the board."""
    moves: list[Move] = []
    for x in range(board.size):
        for y in range(board.size):
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
    if root_allowed_moves is not None:
        # keep only allowed moves in root.untried_moves
        root.untried_moves = root_allowed_moves
        # pre-expand each allowed move as a child so the search distributes sims among them
        for move in list(root.untried_moves):
            root.add_child(move)

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
    report_top_k = None
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


def cardinal_dirs(board, current_tile:Move, walls):
    """
    Calculate all possible strong connections from a given tile
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
    result = []

    # For each triplet associated with a strong connection
    # If one tile of that triplet is illegal, disregard the entire triplet
    # Otherwise, add it to the list of possible strong connections
    for triplet in dirs.values():
        if triplet[0] in legals:
            if triplet[1] in legals:
                # Thought being held
                # Check triplet 1 and 2 for being legal
                # We need to check if triplet[2] is a wall
                # Make a list of what we believe or wall to be, if it's LEGAL or a wall, let him through
                # else pass 
                if (triplet[2] in legals) or (triplet[2] in walls):
                    result.append(triplet)


    return result


class HexAgent(AgentBase):
    _board_size: int = 11

    """
    HexAgent class
    ---
    Attributes
    ---
    Dictionary adjacency_matrix
        Given a tile (Move), returns adjacent tiles in a list (List[Move])
    defaultdict(False) current_bridges: Key is a tile, Returned value is a tile, Default value is False (None doesn't work)
        Given a tile (Move), if that tile is a bridge, return the other bridge it's corresponding bridge (Move), otherwise return False
    Dictionary potential_connections: 
        Given a tile (Move) which is a strong connection, return the bridges it is dependent on 
    ---
    """


    def __init__(self, colour: Colour):
        self.walls = []
        self.current_bridges = {}
        self.potential_connections = {}
        self.setup_walls(colour)
        super().__init__(colour)

    def setup_wall_connections(self,colour):
        if colour == colour.RED:
            for i in range(10):
                self.potential_connections[Move(1,i)] = [Move(0,i),Move(0,i+1)]
                self.potential_connections[Move(9,i+1)] = [Move(10,i-1),Move(10,i)]
        if colour == colour.BLUE:
            for i in range(10):
                self.potential_connections[Move(i,1)] = [Move(i,1),Move(i+1,0)]
                self.potential_connections[Move(i+1,9)] = [Move(i-1,10),Move(i,10)]

    def setup_walls(self,colour):
        wall_list = []
        if colour == colour.RED:
            for i in range(11):
                wall_list.append(Move(-1,i))
                wall_list.append(Move(11,i))
        if colour == colour.BLUE:
            for i in range(11):
                wall_list.append(Move(i,-1))
                wall_list.append(Move(i,11))
        self.walls = wall_list


        self.setup_wall_connections(colour)

    def check_reach(self, board, bridges):
        """
        Given a board, can the agent win the game if it were to play all its strong connections?
        Returns True if we can
        Return False otherwise


        Eventually it would be nice to DFS this (via all of our strong connections) to find the shortest path
        """
        board_copy = clone_board(board)
        # Add strongly connected bridges to the board as if they've already been played
        bridges=bridges.keys()
        if not bridges: return False

        for move in bridges:
            board_copy.tiles[move.x][move.y].colour = self.colour


        # print("okay, this is what I think abt our board state",board_copy.has_ended(self.colour))
        return board_copy.has_ended(self.colour)
        # Return false (we dont win rn)

    def establish_connection(self, board, chosen_move):


        """
        After this point, we will have chosen a move to play.
        Update the following:
            self.potential_connections
                - Our move must have already existed in potential connections (i feel like this is wrong but we'll cross that bridge when we get to it)
                - It's no longer potential, as we are now playing this move
                - So remove it from our dictionary of potential connections
            current_bridges
                - Adding a new strong connection inherently must come with bridges
                - Add those to our current bridges
        """

        # Add the new potential triplets to the potential connections
        potential_triplets = cardinal_dirs(board,chosen_move,self.walls)
        # print("Walls are",self.walls)
        for triplet in potential_triplets:
            # Create format for potential connections dictionary
            if triplet[-1] not in self.walls:
                self.potential_connections[triplet[-1]] = (triplet[0],triplet[1])
            else:
                if triplet[0] in self.current_bridges.keys():
                    self.current_bridges[triplet[0]].append(triplet[1])
                else:
                    self.current_bridges[triplet[0]] = [triplet[1]]
                if triplet[1] in self.current_bridges.keys():
                    self.current_bridges[triplet[1]].append(triplet[0])
                else:
                    self.current_bridges[triplet[1]] = [triplet[0]]
                    

        # print("I reached the part of the code where we need to add to bridges")
        # Add the linked pair of bridges to current_bridges
        bridges = self.potential_connections.get(chosen_move,[])
        # print(bridges)
        if len(bridges) > 0:
            # print("Checking trash",self.current_bridges.get(bridges[0],[]).append("blah"))
            if bridges[0] in self.current_bridges.keys():
                self.current_bridges[bridges[0]].append(bridges[1])
            else:
                self.current_bridges[bridges[0]] = [bridges[1]]

            if bridges[1] in self.current_bridges.keys():
                self.current_bridges[bridges[1]].append(bridges[0])
            else:
                self.current_bridges[bridges[1]] = [bridges[0]]



        # Remove the connection we just made from potential_connections
            del self.potential_connections[chosen_move]

        if chosen_move in self.current_bridges.keys():
            for bridge in self.current_bridges[chosen_move]:
                del self.current_bridges[bridge]
            del self.current_bridges[chosen_move]


    def reduce_board(self, board):
        """
        
        Reduce the board
        """
        # Check if we win
        smallest_so_far = self.current_bridges.copy()

        for bridge in self.current_bridges.keys():
            # This is exactly one bridge, remove it from the board
            #print("This is the smallest set we need",smallest_so_far)
            test_bridges = smallest_so_far.copy()
            del test_bridges[bridge]
            if self.check_reach(board,test_bridges):
                smallest_so_far = test_bridges.copy()



        # What we're doing is looking at the set of current bridges that we've determined, is the smallest set of bridges we need
        # So that we can win the game
        # However, in this process we sometimes lose the other side of strong connections of the bridges 
        # As a result, we do the following
        # 1. Take an empty dict, temp
        # 2. Go through each key in smallest_so_far
        # 3. If it exists in temp, then append the original values of that key (in smallest_so_far) to temp
        # 4. Otherwise, plainly set it to the original values
        # 5. Now, for each key that value that we have
        # 6. Doubly link it back onto it's key
        # 7. In the case where the value already exists as a key, append it instead, otherwise just set it

        # This can be slightly optimised by rather than taking temp as an empty list
        # We can initialise it as smallest_so_far already
        # print("This is before we've doubly linked",smallest_so_far)


        temp_inverted_list = smallest_so_far.copy()
        for key in smallest_so_far.keys():
            # Grab the values, could be a list
            values = smallest_so_far[key]
            for move in values:
                # Does this move exist inside temp_inverted_list?
                if move in temp_inverted_list.keys():
                     # Only add the move if it doesn't already exist
                    if key not in temp_inverted_list[move]:
                        temp_inverted_list[move].append(key)
                else:
                    temp_inverted_list[move] = [key]
        





        # print("And this is after",temp_inverted_list)
        self.current_bridges = temp_inverted_list.copy()



            # We win
            # Keep cutting bridges out the path until we have the only essential path we need to reach the end
            # Return the list of strong connections we need to win
            
            
            #


    def reinstate_links(self):

        temp_inverted_list = self.current_bridges.copy()
        for key in self.current_bridges.keys():
            # Grab the values, could be a list
            values = self.current_bridges[key]
            for move in values:
                # Does this move exist inside temp_inverted_list?
                if move in temp_inverted_list.keys():
                    # Only add the move if it doesn't already exist
                    if key not in temp_inverted_list[move]:
                        temp_inverted_list[move].append(key)
                else:
                    temp_inverted_list[move] = [key]
        

        self.current_bridges = temp_inverted_list

    def remove_taken_potential_connection(self,opp_move):
        temp_inverted_list = {}
        for pot_con in list(self.potential_connections.keys()):
            for bridge in self.potential_connections[pot_con]:
                if bridge in temp_inverted_list.keys():
                    temp_inverted_list[bridge].append(pot_con)
                else:
                    temp_inverted_list[bridge] = [pot_con]

        # print("this is truly disgusting")
        # print(temp_inverted_list)
        if temp_inverted_list.get(opp_move,[]):
            for connection in temp_inverted_list[opp_move]:
                # print("HE TOOK OUR BRIDGE")
                # print("We're going to prune",self.potential_connections[connection])
                del self.potential_connections[connection]


    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Decide on a move to make given the current turn, board state, and opponent's last move.
        Turn 1: Make a fair move to combat the swap rule.
        Turn 2: Decide whether to swap based on opponent's first move.
        Subsequent turns: Use MCTS to select the best move.
        """
        
        self.reinstate_links()
        """
        print("I think I'm ",self.colour)
        print("I think these are my current bridges",self.current_bridges)
        print("I think my walls",self.walls)
        """
        move_set = []

        if turn == 2:
            if should_swap(board, opp_move):
                self.potential_connections = {}
                self.current_bridges = {}
                self.setup_walls(Colour.opposite(self.colour))
                self.establish_connection(board,opp_move)
                return Move(-1, -1)

        if opp_move == Move(-1,-1):
            self.potential_connections = {}
            self.current_bridges = {}
            self.setup_walls(self.colour)
        # print("I think these are my potential connections",self.potential_connections)

        """
        This section is for determining what the goal of the Agent should be
        And so therefore what root_allowed_moves (designed to move_set) should be
        
        I think it would be good if we were to create a (written) priority list so we know what our move_set should be
        """

        # Check if the opponent has taken a bridge from us 
        # If so, take the capture the other untaken bridge we still have

        if self.current_bridges.get(opp_move,[]):
            # This can either be either 1, or 2  OR 3 bridges
            connected_bridges = self.current_bridges[opp_move]
            
            if len(connected_bridges) > 1:
                # Implement logic
                # Limit MCTS to the two moves we have to choose between
                # Whichever one we pick, remove 
                #   - the opposite side from the list of bridges
                #   - the bridge that we actually did take
                move_set = connected_bridges
                chosen_move = mcts_search(
                    root_board=board,
                    my_colour=self.colour,
                    max_iterations=5000,          # max number of random plays
                    max_time_seconds=0.5,           # time limit per move
                    report_top_k=5,               # show top-5 for normal turns
                    root_allowed_moves=move_set
                )

                # We now know our chosen move,
                # Remove our chosen move from moveset
                # And also remove their adjacent bridges
                # Explicitly do MCTS here in this case

                # i think the error is in here
                # we picked a move out of defense, prune the rest of the 3
                for unchosen_move in move_set:
                    for adjacent in self.current_bridges[unchosen_move]:
                        del self.current_bridges[adjacent]
                self.reinstate_links()
                self.remove_taken_potential_connection(opp_move)
                self.establish_connection(board,chosen_move)

                return chosen_move

            else:
                
                move_we_take = self.current_bridges[opp_move]
                for adjacent in self.current_bridges[move_we_take[0]]:
                    if adjacent in self.current_bridges.keys():
                        del self.current_bridges[adjacent]
                self.reinstate_links()
                del self.current_bridges[move_we_take[0]]
                # print("I exited through return statement 1")
                self.remove_taken_potential_connection(opp_move)
                self.establish_connection(board,move_we_take[0])

                return move_we_take[0]


        # Determine if we've already won
        if self.check_reach(board,self.current_bridges):
            # We've determined that we can win, and we need to reduce current_bridges
            # We need to check all values
            # If there is a value (for a key), that is not itself a key in current_bridges
            # Remove that value from the list of values for the key
            self.reduce_board(board)
            # To implement
            move_set = list(self.current_bridges.keys())

        if self.potential_connections.get(opp_move,[]):
            # The enemy took a pot con
            del self.potential_connections[opp_move]



        self.remove_taken_potential_connection(opp_move)


        # If it's the first turn
        if turn == 1:
            # Pick a "fair" move
            move_set = get_fair_first_moves(board)
        else:
            # If we have decided on moves already
            # Choose them
            if not(move_set):
                # Otherwise, pick a virtual connection
                move_set = list(self.potential_connections.keys()) if self.potential_connections.keys() else None


        # root_allowed_moves/move_set
        # The set of moves that we will allow MCTS to access for the purpose of simulating victories.
        # root_allowed_moves is dependent on the analysis of the boardstate
        # E.g.
        #   If we have what we need to win, root_allowed_moves = our strong connections
        #   If we still don't have what we need but we're winning, establish new connections?


        # print(move_set)
        chosen_move = mcts_search(
            root_board=board,
            my_colour=self.colour,
            max_iterations=5000,          # max number of random plays
            max_time_seconds=0.8,           # time limit per move !!!WARNING!!! THIS BREAKS WHEN YOU MAKE IT BELOW A CERTAIN THRESHOLD (~0.5)
            report_top_k=5,               # show top-5 for normal turns
            root_allowed_moves=move_set
        )





        self.establish_connection(board,chosen_move)


        return chosen_move


# Agent VS Naive:
# Red
# python3 Hex.py -p1 "agents.Group16.HexAgent HexAgent" -p1Name "Group16" -p2 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p2Name "TestAgent"
# Blue
# python3 Hex.py -p1 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p1Name "TestAgent" -p2Name "Group16" -p2 "agents.Group16.HexAgent HexAgent"

# Agent VS Agent
# python3 Hex.py -p1 "agents.Group16.HexAgent HexAgent" -p1Name "G16Player1" -p2 "TestAgent" -p2 "agents.Group16.HexAgent HexAgent" -p2Name "G16Player2"

# To run the analysis over 100 games (this took 2-3 hours for me):
# python3 Hex.py -p1 "agents.TestAgents.RandomValidAgent RandomValidAgent" -p1Name "TestAgent" -p2 "agents.Group16.HexAgent HexAgent" -p2Name "Group16" -a -g 50



"""
Todo list
1. If blue chooses not to swap, it's always going to end up playing a move that is right next to a wall, which is not a good starting move
     It needs to be encouraged to play something more centrally or else it will be at a disadvantage
     So we need to broaden the list of moves that it looks at after its 1st moves
     I.e. add - to its potential connections - a list of good moves to its list of potential connections/move_set 
     So that it considers it for MCTS
2. Fix weird error
3. Improve analysis to check for errors
"""