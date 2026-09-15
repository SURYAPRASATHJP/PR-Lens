"""Unified diff to hunks.

Three shapes of diff arrive from GitHub and all three reduce to the same thing:
the per-file `patch` on the files endpoint, the single `diff_hunk` carried by a review
comment, and a whole pull request fetched with the diff media type. A hunk is the unit a
line comment attaches to, so this is the smallest piece of the reviewer that has to be
exactly right.
"""

import re
from dataclasses import dataclass

# @@ -12,7 +12,9 @@ def parse(self):
# The counts are optional and mean 1 when absent, which is the case a naive regex gets
# wrong on single-line hunks.
_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_lines>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_lines>\d+))? @@(?: (?P<section>.*))?$"
)
_FILE_HEADER = re.compile(r"^\+\+\+ (?:b/)?(?P<path>.+)$")


@dataclass(frozen=True, slots=True)
class Hunk:
    path: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    section: str | None
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        header = f"@@ -{self.old_start},{self.old_lines} +{self.new_start},{self.new_lines} @@"
        if self.section:
            header = f"{header} {self.section}"
        return "\n".join([header, *self.lines])

    def added(self) -> list[tuple[int, str]]:
        """Added lines with their line number in the new file.

        This is what a line comment needs: GitHub wants a line number on the right-hand
        side, and counting it from the hunk header is the only way to get it, because the
        diff itself does not carry per-line numbers.
        """
        return [(n, t) for n, t, marker in self._walk() if marker == "+"]

    def removed(self) -> list[str]:
        return [line[1:] for line in self.lines if line.startswith("-")]

    def new_file_lines(self) -> list[tuple[int, str]]:
        """Every line that exists in the new file, added and unchanged alike."""
        return [(n, t) for n, t, marker in self._walk() if marker in "+ "]

    def _walk(self) -> list[tuple[int, str, str]]:
        walked = []
        new_line = self.new_start
        for line in self.lines:
            marker = line[:1] or " "
            if marker == "-":
                continue
            if marker == "\\":  # "\ No newline at end of file"
                continue
            walked.append((new_line, line[1:], marker))
            new_line += 1
        return walked


def parse_patch(patch: str, path: str) -> list[Hunk]:
    """Parse one file's patch, the shape GitHub puts in a file's `patch` field."""
    return _parse(patch, default_path=path)


def parse_diff(diff: str) -> list[Hunk]:
    """Parse a whole-pull-request unified diff, where the path comes from +++ lines."""
    return _parse(diff, default_path=None)


def parse_diff_hunk(diff_hunk: str, path: str) -> Hunk | None:
    """Parse the single hunk a review comment carries as context.

    GitHub truncates diff_hunk to the last few lines of context, so the header's line
    counts routinely disagree with the lines that follow. Both are kept as sent: the
    header is what GitHub's own line arithmetic uses, so correcting it would put us out
    of step with the API rather than into step with the file.
    """
    hunks = _parse(diff_hunk, default_path=path)
    return hunks[0] if hunks else None


def _parse(text: str, *, default_path: str | None) -> list[Hunk]:
    hunks: list[Hunk] = []
    path = default_path
    header: re.Match[str] | None = None
    body: list[str] = []
    old_left = new_left = 0

    def flush() -> None:
        if header is None:
            return
        if path is None:
            # A hunk with no file is not attributable to a line, so it is not usable.
            return
        hunks.append(
            Hunk(
                path=path,
                old_start=int(header["old_start"]),
                old_lines=int(header["old_lines"] or 1),
                new_start=int(header["new_start"]),
                new_lines=int(header["new_lines"] or 1),
                section=(header["section"] or None),
                lines=tuple(body),
            )
        )

    for line in text.splitlines():
        if header is not None and (old_left > 0 or new_left > 0):
            # Inside a hunk the header's counts decide what is content, never the text. A
            # removed line reading "-- note" arrives as "--- note" and an added "++ x" as
            # "+++ x", and taking either for the next file's header cuts the hunk short
            # and shifts every line number after it.
            body.append(line)
            marker = line[:1]
            if marker == "-":
                old_left -= 1
            elif marker == "+":
                new_left -= 1
            elif marker != "\\":
                old_left -= 1
                new_left -= 1
            continue

        if header is not None and line.startswith("\\"):
            # "\ No newline at end of file" trails the hunk's last counted line.
            body.append(line)
            continue

        match = _HEADER.match(line)
        if match:
            flush()
            header, body = match, []
            old_left = int(match["old_lines"] or 1)
            new_left = int(match["new_lines"] or 1)
            continue

        # Past the counted lines, so this is preamble: "diff --git", "index", "---", "+++".
        flush()
        header, body = None, []
        file_header = _FILE_HEADER.match(line)
        if file_header:
            path = file_header["path"] if file_header["path"] != "/dev/null" else None

    flush()
    return hunks
