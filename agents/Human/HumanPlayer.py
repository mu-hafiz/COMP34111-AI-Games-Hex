from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class HumanPlayer(AgentBase):
    """This class describes a human Hex player. It will allow a human to input
    their move via the console.
    """

    _choices: list[Move]
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent move
        """

        return self.get_human_move()
        
    def get_human_move(self) -> Move:
        """Prompt the human player to input their move via the console.

        Returns:
            Move: The human player's move
        """ 
        while True:
            try:
                user_input = input("Enter your move (format: x, y): ")
                x_str, y_str = user_input.strip().split(",")
                x, y = int(x_str), int(y_str)
                return Move(x, y)
            except ValueError:
                print("Invalid input format. Please enter two integers separated by a space.")
