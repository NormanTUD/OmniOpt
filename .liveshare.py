"""File-based live-share control markers for OmniOpt2.

A normal run enables live-sharing with ``--live_share`` at start-up.  This
module adds a file based control channel that also works for runs that are
already in progress:

    touch liveshare          # enable live-sharing
    touch stop_liveshare     # disable live-sharing again

The marker files are looked up in several scopes.  ``stop_liveshare`` always
wins over ``liveshare``.  The supported scopes are, from most to least
specific::

    <run-folder>/liveshare               # only this run
    <runs>/<experiment>/liveshare        # all runs of this experiment
    <current working directory>/liveshare  # all runs
    <omniopt directory>/liveshare          # all runs

Removing a marker never changes the current state by itself.  This mirrors
the intended behaviour: once sharing was enabled, deleting ``liveshare`` does
not stop it; an explicit ``stop_liveshare`` marker is required.  The same is
true for the opposite direction.
"""

from __future__ import annotations

import os
import re
from enum import Enum
from typing import Any, Callable, List, NamedTuple, Optional, Set

START_MARKER = "liveshare"
STOP_MARKER = "stop_liveshare"

_URL_RE = re.compile(r"https?://[^\s]+")
_URL_TRAILING_CHARS = ".,;:)]}'\">"


class MarkerAction(Enum):
    """Action requested by the marker files."""

    START = "start"
    STOP = "stop"


class MarkerLocation(NamedTuple):
    """A folder in which marker files are looked up."""

    scope: str
    folder: str

    def path(self, marker: str) -> str:
        return os.path.join(self.folder, marker)


class MarkerDecision(NamedTuple):
    """The action selected by the marker files and where it was found."""

    action: MarkerAction
    scope: str
    path: str


def marker_locations(
    run_folder: Optional[str],
    cwd: Optional[str] = None,
    script_dir: Optional[str] = None,
) -> List[MarkerLocation]:
    """Return marker folders ordered from most to least specific.

    ``run_folder`` is normally ``<runs>/<experiment>/<run-nr>``.  Its parent
    is used as the experiment scope.  The current working directory and the
    OmniOpt installation directory are both global scopes; the current
    working directory is checked first.

    Duplicate folders (for example when OmniOpt is invoked from its own
    directory) are only reported once.
    """
    locations: List[MarkerLocation] = []
    seen: Set[str] = set()

    def add(scope: str, folder: Optional[str]) -> None:
        if not folder:
            return
        absolute = os.path.abspath(folder)
        key = os.path.normcase(absolute)
        if key in seen:
            return
        seen.add(key)
        locations.append(MarkerLocation(scope=scope, folder=absolute))

    if run_folder:
        add("run", run_folder)
        add("experiment", os.path.dirname(os.path.abspath(run_folder)))

    add("global", cwd if cwd is not None else os.getcwd())
    add("global", script_dir)

    return locations


def decide(
    run_folder: Optional[str],
    cwd: Optional[str] = None,
    script_dir: Optional[str] = None,
    exists: Callable[[str], bool] = os.path.exists,
) -> Optional[MarkerDecision]:
    """Return the action requested by the marker files, if any.

    All stop markers are checked before any start marker, so an explicit
    ``stop_liveshare`` always wins.  Among multiple stop (or start) markers
    the most specific scope is reported.
    """
    locations = marker_locations(run_folder, cwd=cwd, script_dir=script_dir)

    for location in locations:
        stop_path = location.path(STOP_MARKER)
        if exists(stop_path):
            return MarkerDecision(
                action=MarkerAction.STOP,
                scope=location.scope,
                path=stop_path,
            )

    for location in locations:
        start_path = location.path(START_MARKER)
        if exists(start_path):
            return MarkerDecision(
                action=MarkerAction.START,
                scope=location.scope,
                path=start_path,
            )

    return None


def extract_share_url(text: str) -> str:
    """Extract the first ``http(s)`` URL from a share command's output."""
    match = _URL_RE.search(text or "")
    if match is None:
        return ""
    return match.group(0).rstrip(_URL_TRAILING_CHARS)


def build_activation_lines(
    decision: MarkerDecision,
    url: str = "",
    error: str = "",
) -> List[str]:
    """Build the user-visible lines announcing an activation."""
    lines = [
        f"LiveShare activated ({decision.scope} marker: {decision.path})",
    ]
    if url:
        lines.append(f"LiveShare URL: {url}")
    elif error:
        lines.append(f"LiveShare URL: not available yet ({error.strip()})")
    else:
        lines.append("LiveShare URL: not available yet")
    return lines


def build_stop_lines(decision: MarkerDecision) -> List[str]:
    """Build the user-visible lines announcing that sharing stopped."""
    return [
        f"LiveShare stopped ({decision.scope} marker: {decision.path})",
    ]


def write_above_progress_bar(
    progress_bar: Optional[Any],
    text: str,
    fallback: Callable[[str], None] = print,
) -> bool:
    """Write ``text`` above a tqdm progress bar without corrupting it.

    The tqdm write method clears the current bar line, prints the message and
    then redraws the bar below it.  If no usable bar is available (disabled
    tqdm, the bar was not started yet, or the write fails) the text is passed
    to ``fallback`` instead.

    Returns ``True`` when the text was written through tqdm.
    """
    if progress_bar is None or getattr(progress_bar, "disable", False):
        fallback(text)
        return False

    writer = getattr(progress_bar, "write", None)
    if writer is None:
        fallback(text)
        return False

    try:
        writer(text, file=getattr(progress_bar, "fp", None))
        return True
    except Exception:
        fallback(text)
        return False


class State:
    """Track the current live-share state and apply marker decisions.

    A decision only produces a transition when it actually changes the
    state.  This is what guarantees that an activation is announced exactly
    once: a marker that stays present is a no-op on every later poll.
    """

    def __init__(self, active: bool = False):
        self.active = bool(active)

    def apply(self, decision: Optional[MarkerDecision]) -> Optional[MarkerAction]:
        """Update the state and return the transition that happened."""
        if decision is None:
            return None

        if decision.action is MarkerAction.START:
            if self.active:
                return None
            self.active = True
            return MarkerAction.START

        if not self.active:
            return None
        self.active = False
        return MarkerAction.STOP


class Controller:
    """Execute the side effects of a marker state transition.

    ``share`` is expected to return the ``(stdout, stderr)`` pair of the
    share command.  ``announce`` receives the already formatted lines.  The
    optional ``on_start`` / ``on_stop`` callbacks run before the share /
    announcement and can be used to persist the state or start the periodic
    updater.
    """

    def __init__(
        self,
        state: State,
        share: Callable[[], Any],
        announce: Callable[[List[str]], None],
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
    ):
        self.state = state
        self.share = share
        self.announce = announce
        self.on_start = on_start
        self.on_stop = on_stop

    def apply(self, decision: Optional[MarkerDecision]) -> Optional[MarkerAction]:
        """Apply ``decision`` and perform at most one transition."""
        action = self.state.apply(decision)

        if action is MarkerAction.START:
            assert decision is not None
            self._call(self.on_start)
            stdout, stderr, error = self._share()
            url = extract_share_url(stdout) or extract_share_url(stderr)
            if not url and not error:
                error = (stderr or stdout).strip()
            self.announce(build_activation_lines(decision, url=url, error=error))
        elif action is MarkerAction.STOP:
            assert decision is not None
            self._call(self.on_stop)
            self.announce(build_stop_lines(decision))

        return action

    def _call(self, callback: Optional[Callable[[], None]]) -> None:
        if callback is None:
            return
        try:
            callback()
        except Exception:
            # State changes must never depend on optional side effects.
            pass

    def _share(self) -> tuple[str, str, str]:
        try:
            stdout, stderr = self.share()
            return str(stdout), str(stderr), ""
        except Exception as exc:  # noqa: BLE001
            return "", "", str(exc)
