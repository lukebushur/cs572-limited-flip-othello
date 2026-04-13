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
    nodes_expanded: int = 0
    cutoffs: int = 0
    max_depth_reached: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class TTEntry:
    depth: int
    value: float
    best_action: Optional[Action]


class Player:
    def __init__(self, color: DiskColor) -> None:
        self.color = color

    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        raise NotImplementedError


class RandomPlayer(Player):
    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        actions = board.get_actions(state)
        if not actions:
            return None
        return random.choice(actions)


class CLIPlayer(Player):
    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        actions = board.get_actions(state)
        if not actions:
            return None

        while True:
            raw = input("Enter move as: row col\n> ").strip()
            try:
                row_str, col_str = raw.split()
                row, col = int(row_str), int(col_str)
                candidate = (row, col, self.color)
                if candidate in actions:
                    return candidate
                print("Illegal move. Legal moves are:")
                for act in actions:
                    print(f"  {act}")
            except ValueError:
                print("Invalid input. Example: 2 3")


class MinimaxAlphaBetaPlayer(Player):
    def __init__(self, color: DiskColor, max_depth: int = 4) -> None:
        super().__init__(color)
        self.max_depth = max_depth
        self.stats = SearchStats()
        self._tt: dict[tuple[bytes, int, int], TTEntry] = {}

    def reset_stats(self) -> None:
        self.stats = SearchStats()

    def clear_tt(self) -> None:
        self._tt.clear()

    def _board_key(
        self,
        board_state: BoardState,
        to_move: DiskColor,
        depth: int,
    ) -> tuple[bytes, int, int]:
        return (board_state.tobytes(), int(to_move), depth)

    def get_next_move(self, board: Board, state: State) -> Optional[Action]:
        self.reset_stats()
        start = time.perf_counter()

        actions = board.get_actions(state)
        if not actions:
            self.stats.elapsed_seconds = time.perf_counter() - start
            return None

        _, best_action = self._alphabeta(
            board=board,
            state=state,
            depth=self.max_depth,
            alpha=-math.inf,
            beta=math.inf,
            maximizing=(state[1] == self.color),
            current_depth=0,
        )

        self.stats.elapsed_seconds = time.perf_counter() - start
        return best_action

    def _alphabeta(
        self,
        board: Board,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        current_depth: int,
    ) -> tuple[float, Optional[Action]]:
        self.stats.nodes_expanded += 1
        self.stats.max_depth_reached = max(self.stats.max_depth_reached, current_depth)

        board_state, to_move = state
        key = self._board_key(board_state, to_move, depth)

        cached = self._tt.get(key)
        if cached is not None:
            self.stats.tt_hits += 1
            return cached.value, cached.best_action

        if depth == 0 or board.is_terminal(state):
            value = board.evaluate(state, self.color)
            self._tt[key] = TTEntry(depth=depth, value=value, best_action=None)
            return value, None

        actions = board.get_actions(state)

        # Pass turn if no legal moves but state is not terminal
        if not actions:
            next_state = board.pass_turn(state)
            value, _ = self._alphabeta(
                board=board,
                state=next_state,
                depth=depth,
                alpha=alpha,
                beta=beta,
                maximizing=not maximizing,
                current_depth=current_depth + 1,
            )
            self._tt[key] = TTEntry(depth=depth, value=value, best_action=None)
            return value, None

        best_action: Optional[Action] = None

        if maximizing:
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
                    current_depth=current_depth + 1,
                )

                if child_value > value:
                    value = child_value
                    best_action = action

                alpha = max(alpha, value)
                if beta <= alpha:
                    self.stats.cutoffs += 1
                    break

            self._tt[key] = TTEntry(depth=depth, value=value, best_action=best_action)
            return value, best_action

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
                current_depth=current_depth + 1,
            )

            if child_value < value:
                value = child_value
                best_action = action

            beta = min(beta, value)
            if beta <= alpha:
                self.stats.cutoffs += 1
                break

        self._tt[key] = TTEntry(depth=depth, value=value, best_action=best_action)
        return value, best_action