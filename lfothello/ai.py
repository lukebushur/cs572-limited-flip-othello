from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional

from .game import Board
from .types import Action, BoardState, DiskColor, State


@dataclass
class SearchStats:
    """
    Stores statistics for a single search.

    Attributes
    ----------
    nodes_expanded : int
        Total number of nodes explored.
    cutoffs : int
        Number of alpha-beta pruning cutoffs.
    max_depth_reached : int
        Deepest level reached during search.
    elapsed_seconds : float
        Total time spent searching.
    tt_hits : int
        Number of transposition table hits.
    """
    nodes_expanded: int = 0
    cutoffs: int = 0
    max_depth_reached: int = 0
    elapsed_seconds: float = 0.0
    tt_hits: int = 0


@dataclass
class TTEntry:
    """
    Entry stored in the transposition table.

    Attributes
    ----------
    depth : int
        Depth at which this value was computed.
    value : float
        Evaluation value for the state.
    best_action : Optional[Action]
        Best action found from this state.
    """
    depth: int
    value: float
    best_action: Optional[Action]


class Player:
    """
    Abstract base class for all players.
    """

    def __init__(self, color: DiskColor) -> None:
        """
        Initialize the player with a specific disk color.
        """
        self.color = color

    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        """
        Compute the next move for this player.

        Must be implemented by subclasses.
        """
        raise NotImplementedError


class RandomPlayer(Player):
    """
    Baseline agent that selects a random legal move.
    """

    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        """
        Select a random legal move to play.
        """
        actions = board.get_actions(state)
        if not actions:
            return None
        return random.choice(actions)


class MinimaxAlphaBetaPlayer(Player):
    """
    Minimax agent with alpha-beta pruning and optional enhancements.

    Features
    --------
    - Minimax search with alpha-beta pruning
    - Optional transposition table (TT)
    - Optional iterative deepening (ID)
    - Heuristic move ordering

    Parameters
    ----------
    color : DiskColor
        Player color.
    max_depth : int
        Maximum search depth.
    use_transposition_table : bool
        Whether to enable TT caching.
    use_iterative_deepening : bool
        Whether to use iterative deepening.
    time_limit_seconds : float
        Time limit per move when using iterative deepening.
    """

    def __init__(
        self,
        color: DiskColor,
        max_depth: int = 4,
        use_transposition_table: bool = True,
        use_iterative_deepening: bool = True,
        time_limit_seconds: float = 2.0,
    ) -> None:
        """
        Initialize a MinimaxAlphaBetaPlayer.

        Parameters
        ----------
        color : DiskColor
            The player's disk color (BLACK or WHITE).
        max_depth : int, default=4
            Maximum search depth for minimax. When iterative deepening is enabled,
            this is the maximum depth the search will attempt to reach.
        use_transposition_table : bool, default=True
            Whether to enable a transposition table (TT) for caching previously
            evaluated states to avoid redundant computation.
        use_iterative_deepening : bool, default=True
            Whether to use iterative deepening search. If enabled, the agent
            repeatedly searches at increasing depths until the time limit is reached.
        time_limit_seconds : float, default=2.0
            Time limit per move (in seconds) when iterative deepening is enabled.
            Ignored if iterative deepening is disabled.
        """
        super().__init__(color)
        self.max_depth = max_depth
        self.use_transposition_table = use_transposition_table
        self.use_iterative_deepening = use_iterative_deepening
        self.time_limit_seconds = time_limit_seconds

        self.stats = SearchStats()

        # Exact-value transposition table
        self._tt: dict[tuple[bytes, int, int], TTEntry] = {}
        # Move-order table (depth-independent)
        self._move_order_table: dict[tuple[bytes, int], Action] = {}

        self._search_start_time = 0.0

    def reset_stats(self) -> None:
        """Reset search statistics."""
        self.stats = SearchStats()

    def clear_tt(self) -> None:
        """Clear all cached TT and move-order entries."""
        self._tt.clear()
        self._move_order_table.clear()

    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        """
        Compute the best move from the given state.

        Uses either:
        - Iterative deepening (if enabled), or
        - Fixed-depth alpha-beta search.
        """
        self.reset_stats()
        self._search_start_time = time.perf_counter()

        actions = board.get_actions(state)
        if not actions:
            self.stats.elapsed_seconds = time.perf_counter() - self._search_start_time
            return None

        if self.use_iterative_deepening:
            best_action = self._iterative_deepening_search(board, state)
        else:
            _, best_action = self._alphabeta(
                board=board,
                state=state,
                depth=self.max_depth,
                alpha=-math.inf,
                beta=math.inf,
                maximizing=(state[1] == self.color),
            )

        self.stats.elapsed_seconds = time.perf_counter() - self._search_start_time
        return best_action

    def _time_exceeded(self) -> bool:
        """
        Check if the search time limit has been exceeded.
        """
        if not self.use_iterative_deepening:
            return False
        return (time.perf_counter() - self._search_start_time) >= self.time_limit_seconds

    def _board_key(self, board_state: BoardState, to_move: DiskColor, depth: int) -> tuple[bytes, int, int]:
        """
        Generate a unique key for TT lookup.

        Includes depth to ensure correctness of cached values.
        """
        return (board_state.tobytes(), int(to_move), depth)

    def _move_order_key(
        self,
        board_state: BoardState,
        to_move: DiskColor,
    ) -> tuple[bytes, int]:
        """
        Generate a depth-independent key for move ordering.
        """
        return (board_state.tobytes(), int(to_move))

    def _ordered_actions(
        self,
        board: Board,
        state: State,
        actions: list[Action],
        tt_best_action: Optional[Action] = None,
    ) -> list[Action]:
        """
        Order actions to improve alpha-beta pruning efficiency.

        Priority:
        1. TT best action (if available)
        2. Corner moves
        3. Moves flipping more disks
        """
        corners = {(0, 0), (0, 7), (7, 0), (7, 7)}

        def score(action: Action) -> tuple[int, int]:
            r, c, _ = action
            corner_bonus = 100 if (r, c) in corners else 0
            flip_bonus = len(board.get_flips_for_move(state, action))
            return (corner_bonus, flip_bonus)

        ordered = sorted(actions, key=score, reverse=True)

        if tt_best_action is not None and tt_best_action in ordered:
            ordered.remove(tt_best_action)
            ordered.insert(0, tt_best_action)

        return ordered

    def _iterative_deepening_search(self, board: Board, state: State) -> Optional[Action]:
        """
        Perform iterative deepening search.

        Repeatedly increases search depth until:
        - maximum depth is reached, or
        - time limit is exceeded.
        """
        best_action: Optional[Action] = None
        depth = 1

        while depth <= self.max_depth:
            if self._time_exceeded():
                break

            try:
                value, action = self._alphabeta(
                    board=board,
                    state=state,
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing=(state[1] == self.color),
                )
                if action is not None:
                    best_action = action
                self.stats.max_depth_reached = max(self.stats.max_depth_reached, depth)
            except TimeoutError:
                break

            depth += 1

        # Fallback if no action is found
        if best_action is None:
            actions = board.get_actions(state)
            return actions[0] if actions else None

        return best_action

    def _alphabeta(
        self,
        board: Board,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> tuple[float, Optional[Action]]:
        """
        Perform recursive minimax search with alpha-beta pruning.

        This method evaluates the game tree rooted at `state` up to the given
        search depth and returns both:
        - the minimax value of the state, and
        - the best action found from that state.

        The search supports two optional enhancements:
        1. Transposition tables, which cache previously evaluated states.
        2. TT-based move ordering, which prioritizes actions that were previously
        found to be strong from the same position.

        Parameters
        ----------
        board : Board
            Game environment used for move generation, state transitions,
            terminal checks, and evaluation.
        state : State
            Current search state, represented as (board_state, player_to_move).
        depth : int
            Remaining search depth. A depth of 0 causes the algorithm to stop
            expanding the tree and use the heuristic evaluation function instead.
        alpha : float
            Best value currently guaranteed to the maximizing player along the
            current search path.
        beta : float
            Best value currently guaranteed to the minimizing player along the
            current search path.
        maximizing : bool
            True if the current node is a maximizing node for this agent,
            False if it is a minimizing node.

        Returns
        -------
        tuple[float, Optional[Action]]
            A pair `(value, best_action)` where:
            - `value` is the minimax value of the state
            - `best_action` is the best action found from this state, or None
            if the state is terminal or depth limit has been reached

        Raises
        ------
        TimeoutError
            Raised if iterative deepening is enabled and the search exceeds the
            configured time limit.
        """
        # Stop search immediately if iterative deepening has exceeded its time limit.
        if self._time_exceeded():
            raise TimeoutError

        # Count this node as visited for performance analysis.
        self.stats.nodes_expanded += 1

        # Track how deeply the search has gone relative to the configured maximum.
        self.stats.max_depth_reached = max(
            self.stats.max_depth_reached,
            self.max_depth - depth if self.max_depth >= depth else 0,
        )

        board_state, to_move = state

        # Keys used for exact-value caching and move ordering.
        key = None
        move_key = None

        # Try exact transposition-table lookup first.
        # If found, reuse the cached value instead of re-searching the subtree.
        if self.use_transposition_table:
            key = self._board_key(board_state, to_move, depth)
            move_key = self._move_order_key(board_state, to_move)

            entry = self._tt.get(key)
            if entry is not None:
                self.stats.tt_hits += 1
                return entry.value, entry.best_action

        # Base case: stop expanding at depth limit or terminal state.
        # Use the heuristic evaluation function.
        if depth == 0 or board.is_terminal(state):
            value = board.evaluate(state, self.color)
            if self.use_transposition_table and key is not None:
                self._tt[key] = TTEntry(depth=depth, value=value, best_action=None)
            return value, None

        # Generate all legal actions for the current player.
        actions = board.get_actions(state)

        # If no legal move exists, pass the turn and continue searching.
        # This is part of standard Othello rules and also applies here.
        if not actions:
            next_state = board.pass_turn(state)
            value, _ = self._alphabeta(
                board=board,
                state=next_state,
                depth=depth,
                alpha=alpha,
                beta=beta,
                maximizing=not maximizing,
            )
            if self.use_transposition_table and key is not None:
                self._tt[key] = TTEntry(depth=depth, value=value, best_action=None)
            return value, None

        # If available, retrieve a previously strong move for this position
        # and use it to improve move ordering before search.
        tt_best_action = None
        if self.use_transposition_table and move_key is not None:
            tt_best_action = self._move_order_table.get(move_key)

        actions = self._ordered_actions(board, state, actions, tt_best_action=tt_best_action)

        best_action: Optional[Action] = None

        if maximizing:
            # Maximizing player tries to make the value as large as possible.
            value = -math.inf
            for action in actions:
                child_state = board.apply_action(state, action)
                child_value, _ = self._alphabeta(
                    board=board,
                    state=child_state,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=False,
                )
                if child_value > value:
                    value = child_value
                    best_action = action

                # Improve alpha bound based on the best value seen so far.
                alpha = max(alpha, value)

                # Prune remaining children if the minimizing player would never allow this branch.
                if beta <= alpha:
                    self.stats.cutoffs += 1
                    break
        else:
            # Minimizing player tries to make the value as small as possible.
            value = math.inf
            for action in actions:
                child_state = board.apply_action(state, action)
                child_value, _ = self._alphabeta(
                    board=board,
                    state=child_state,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=True,
                )
                if child_value < value:
                    value = child_value
                    best_action = action

                # Improve beta bound based on the best value seen so far.
                beta = min(beta, value)

                # Prune remaining children if the maximizing player would never allow this branch.
                if beta <= alpha:
                    self.stats.cutoffs += 1
                    break

        # Store exact value and best move for future reuse.
        if self.use_transposition_table:
            if key is not None:
                self._tt[key] = TTEntry(depth=depth, value=value, best_action=best_action)
            if move_key is not None and best_action is not None:
                self._move_order_table[move_key] = best_action

        return value, best_action