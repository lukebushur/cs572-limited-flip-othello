from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import Cell, DiskColor, BoardState, State, Action


BOARD_SIZE = 8
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # up
    (-1, 1),  # up-right
    (0, 1),   # right
    (1, 1),   # down-right
    (1, 0),   # down
    (1, -1),  # down-left
    (0, -1),  # left
    (-1, -1), # up-left
)

@dataclass
class GameResult:
    """
    Represents the final outcome of a game.

    Attributes
    ----------
    black_disks : int
        Number of black disks on the board.
    white_disks : int
        Number of white disks on the board.
    winner : Optional[DiskColor]
        Winning player, or None if the game is a draw.
    """
    black_disks: int
    white_disks: int
    winner: Optional[DiskColor]  # None for draw


class Board:
    """
    Limited-Flip Othello environment.

    Rules:
    - Standard Othello legality rules.
    - When a move captures in a direction, only up to `flip_limit`
      opponent disks closest to the placed disk are flipped in that direction.

    The board is represented as an 8x8 NumPy array.
    """
    def __init__(self, initial: Optional[BoardState] = None, flip_limit: int = 2) -> None:
        """
        Initialize the board.

        Parameters
        ----------
        initial : Optional[BoardState]
            Custom starting board. If None, uses standard Othello setup.
        flip_limit : int
            Maximum number of disks flipped per direction.
        """
        if flip_limit <= 0:
            raise ValueError("flip_limit must be >= 1")

        self.flip_limit = flip_limit

        if initial is not None:
            board = initial.astype(np.int8, copy=True)
            self.state: State = (board, DiskColor.BLACK)
        else:
            board = np.full((BOARD_SIZE, BOARD_SIZE), Cell.EMPTY, dtype=np.int8)
            board[3, 3] = Cell.WHITE
            board[3, 4] = Cell.BLACK
            board[4, 3] = Cell.BLACK
            board[4, 4] = Cell.WHITE

            self.state: State = (board, DiskColor.BLACK)

    def copy(self) -> Board:
        """
        Create a deep copy of the board.

        Returns
        -------
        Board
            Independent copy of the current board.
        """
        board, to_move = self.state
        new_board = Board(initial=board.copy(), flip_limit=self.flip_limit)
        new_board.state = (board.copy(), to_move)
        return new_board

    @staticmethod
    def initial_board() -> BoardState:
        """
        Create the standard Othello starting position.

        Returns
        -------
        BoardState
            8x8 NumPy array with initial disk placement.
        """
        board = np.full((BOARD_SIZE, BOARD_SIZE), Cell.EMPTY, dtype=np.int8)
        board[3, 3] = Cell.WHITE
        board[3, 4] = Cell.BLACK
        board[4, 3] = Cell.BLACK
        board[4, 4] = Cell.WHITE
        return board

    def get_state(self) -> State:
        """
        Return a copy of the current game state.

        Returns
        -------
        State
            (board copy, player to move)
        """
        board, to_move = self.state
        return board.copy(), to_move

    @staticmethod
    def in_bounds(row: int, col: int) -> bool:
        """Check if a position is within board boundaries."""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def _get_flips_for_direction(
        self,
        board: BoardState,
        row: int,
        col: int,
        player: DiskColor,
        dr: int,
        dc: int,
    ) -> list[tuple[int, int]]:
        """
        Compute flips in a single direction.

        Implements the Limited-Flip rule: only the closest k disks
        (k = flip_limit) are flipped.

        Returns
        -------
        list of (row, col)
            Positions to flip in this direction.
        """
        flips: list[tuple[int, int]] = []
        opponent = player.opponent()

        r, c = row + dr, col + dc

        # Collect opponent disks
        while self.in_bounds(r, c) and board[r, c] == opponent:
            flips.append((r, c))
            r += dr
            c += dc

        # Must end on a player's own disk to be valid
        if not flips or not self.in_bounds(r, c) or board[r, c] != player:
            return []

        # Limited-Flip Othello flips only the closest k.
        return flips[: self.flip_limit]

    def get_flips_for_move(self, state: State, action: Action) -> list[tuple[int, int]]:
        """
        Compute all flips for a given move.

        Returns empty list if move is invalid.
        """
        board, to_move = state
        row, col, color = action

        if color != to_move or not self.in_bounds(row, col) or board[row, col] != Cell.EMPTY:
            return []

        all_flips: list[tuple[int, int]] = []
        for dr, dc in DIRECTIONS:
            all_flips.extend(self._get_flips_for_direction(board, row, col, color, dr, dc))

        return all_flips

    def is_legal_action(self, state: State, action: Action) -> bool:
        """Return True if the move is legal."""
        return len(self.get_flips_for_move(state, action)) > 0

    def get_actions(self, state: Optional[State] = None) -> list[Action]:
        """
        Generate all legal moves for the current player.
        """
        if state is None:
            state = self.state

        board, to_move = state
        actions: list[Action] = []

        empties = np.argwhere(board == Cell.EMPTY)
        for row, col in empties:
            action: Action = (int(row), int(col), to_move)
            if self.is_legal_action(state, action):
                actions.append(action)

        return actions

    def apply_action(self, state: State, action: Action) -> State:
        """
        Apply a move and return the resulting state.
        """
        board, to_move = state
        row, col, color = action

        if color != to_move:
            raise ValueError("Action color does not match player to move.")

        flips = self.get_flips_for_move(state, action)
        if not flips:
            raise ValueError(f"Illegal move: {action}")

        new_board = board.copy()
        new_board[row, col] = color

        # Flip captured disks
        for r, c in flips:
            new_board[r, c] = color

        return new_board, to_move.opponent()

    def pass_turn(self, state: State) -> State:
        """Pass the turn to the opponent."""
        board, to_move = state
        return board.copy(), to_move.opponent()

    def is_terminal(self, state: Optional[State] = None) -> bool:
        """
        Check if the game is over (no legal moves for both players).
        """
        if state is None:
            state = self.state

        if self.get_actions(state):
            return False

        opponent_state = self.pass_turn(state)
        return len(self.get_actions(opponent_state)) == 0

    def disk_counts(self, state: Optional[State] = None) -> tuple[int, int]:
        """Return the number of black and white disks."""
        if state is None:
            state = self.state

        board, _ = state
        black = int(np.sum(board == Cell.BLACK))
        white = int(np.sum(board == Cell.WHITE))
        return black, white

    def get_result(self, state: Optional[State] = None) -> GameResult:
        """Compute the final game result."""
        if state is None:
            state = self.state

        black, white = self.disk_counts(state)
        if black > white:
            winner = DiskColor.BLACK
        elif white > black:
            winner = DiskColor.WHITE
        else:
            winner = None

        return GameResult(
            black_disks=black,
            white_disks=white,
            winner=winner,
        )

    def utility(self, state: State, maximizing_player: DiskColor) -> int:
        """
        Terminal utility: disk difference.
        """
        black, white = self.disk_counts(state)
        diff = black - white
        return diff if maximizing_player == DiskColor.BLACK else -diff

    def evaluate(self, state: State, maximizing_player: DiskColor) -> float:
        """
        Heuristic evaluation for non-terminal states.

        Combines:
        - disk difference
        - mobility (number of legal moves)
        - corner occupancy (highly valuable positions)
        """
        if self.is_terminal(state):
            return float(self.utility(state, maximizing_player) * 10_000)

        board, to_move = state
        opponent = maximizing_player.opponent()

        black_count, white_count = self.disk_counts(state)
        disk_diff = black_count - white_count
        if maximizing_player == DiskColor.WHITE:
            disk_diff = -disk_diff

        my_actions = len(self.get_actions((board, maximizing_player)))
        opp_actions = len(self.get_actions((board, opponent)))
        mobility = my_actions - opp_actions

        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        my_corners = 0
        opp_corners = 0
        for r, c in corners:
            if board[r, c] == maximizing_player:
                my_corners += 1
            elif board[r, c] == opponent:
                opp_corners += 1
        corner_score = my_corners - opp_corners

        # Weights can be tuned experimentally later.
        return 1.0 * disk_diff + 4.0 * mobility + 25.0 * corner_score

    def make_move(self, action: Action) -> None:
        """
        Apply a legal move to the board's current internal state.

        Parameters
        ----------
        action : Action
            The move to apply, represented as (row, col, player_color).
        """
        self.state = self.apply_action(self.state, action)

    def step_pass_if_needed(self) -> bool:
        """
        If current player has no legal moves, pass the turn.
        Returns True if a pass happened.
        """
        if self.get_actions(self.state):
            return False
        if not self.is_terminal(self.state):
            self.state = self.pass_turn(self.state)
            return True
        return False

    def _get_cell_display(self, cell_value: int) -> str:
        """
        Convert a board cell value into a printable character.

        Parameters
        ----------
        cell_value : int
            Integer representation of a board cell.

        Returns
        -------
        str
            Single-character display symbol:
            - '.' for empty
            - 'B' for black
            - 'W' for white
            - '?' for unexpected values
        """
        if cell_value == Cell.EMPTY:
            return "."
        if cell_value == Cell.BLACK:
            return "B"
        if cell_value == Cell.WHITE:
            return "W"
        return "?"

    def display_state(self, state: Optional[State] = None) -> None:
        """
        Print a human-readable representation of the board state.

        Parameters
        ----------
        state : Optional[State]
            State to display. If None, displays the board's current internal state.
        """
        if state is None:
            state = self.state

        board, to_move = state
        print("    0 1 2 3 4 5 6 7")
        print("  +-----------------+")
        for i in range(BOARD_SIZE):
            print(f"{i} |", end=" ")
            for j in range(BOARD_SIZE):
                print(self._get_cell_display(int(board[i, j])), end=" ")
            print("|")
        print("  +-----------------+")
        print(f"To move: {'BLACK' if to_move == DiskColor.BLACK else 'WHITE'}")

    def display_actions(self, state: Optional[State] = None) -> None:
        """
        Print all legal actions available in a given state.

        Parameters
        ----------
        state : Optional[State]
            State from which to generate legal actions. If None, uses the
            board's current internal state.
        """
        if state is None:
            state = self.state
        actions = self.get_actions(state)
        print("Legal actions:")
        for row, col, color in actions:
            print(f"  ({row}, {col}, {'BLACK' if color == DiskColor.BLACK else 'WHITE'})")