from pr_lens.ingest.chunking import MAX_CHARS, chunk_file, chunk_markdown, chunk_python, chunk_rst

MODULE = '''"""Module docstring."""

import os

CONSTANT = 1


# Explains why parse exists at all.
@decorator
def parse(value):
    """Parse it."""
    return value + 1


class Thing:
    def method(self):
        return 2


TRAILING = os
'''


def test_each_top_level_definition_becomes_its_own_chunk() -> None:
    symbols = [c.symbol for c in chunk_python(MODULE)]
    assert "parse" in symbols
    assert "Thing" in symbols


def test_a_decorator_and_the_comment_above_it_stay_with_their_function() -> None:
    parse = next(c for c in chunk_python(MODULE) if c.symbol == "parse")
    assert "# Explains why parse exists at all." in parse.text
    assert "@decorator" in parse.text
    assert parse.text.startswith("# Explains why")


def test_imports_do_not_get_swallowed_into_the_first_function() -> None:
    parse = next(c for c in chunk_python(MODULE) if c.symbol == "parse")
    assert "import os" not in parse.text
    preamble = next(c for c in chunk_python(MODULE) if c.symbol is None)
    assert "import os" in preamble.text


def test_line_numbers_point_at_the_real_lines() -> None:
    lines = MODULE.splitlines()
    for chunk in chunk_python(MODULE):
        assert chunk.text.splitlines()[0] == lines[chunk.start_line - 1]
        assert chunk.text.splitlines()[-1] == lines[chunk.end_line - 1]


def test_a_class_too_big_to_keep_whole_splits_into_methods() -> None:
    body = "".join(f"    def m{i}(self):\n        return {'x' * 400!r}\n" for i in range(20))
    symbols = [c.symbol for c in chunk_python(f"class C:\n{body}")]
    assert "C.m5" in symbols
    assert all(len(c.text) <= MAX_CHARS for c in chunk_python(f"class C:\n{body}"))


def test_a_syntax_error_falls_back_to_line_windows_rather_than_losing_the_file() -> None:
    broken = "def (:\n    pass\n" + "x = 1\n" * 50
    chunks = chunk_file("broken.py", broken)
    assert chunks
    assert "".join(c.text for c in chunks).count("x = 1") == 50


def test_markdown_splits_at_headings_and_names_the_breadcrumb() -> None:
    text = (
        "# Title\n\n" + "Intro paragraph long enough to stand on its own here.\n\n"
        "## Install\n\n" + "Run the installer, and then run it again for good measure.\n\n"
        "## Usage\n\n" + "Call the function with an argument that means something.\n"
    )
    chunks = chunk_markdown(text)
    assert [c.symbol for c in chunks] == ["Title", "Title > Install", "Title > Usage"]


def test_a_hash_inside_a_fenced_block_is_not_a_heading() -> None:
    text = (
        "# Title\n\nSome prose that is long enough not to be merged into the next one.\n\n"
        "```sh\n# not a heading at all\necho hi\n```\n\nMore prose after the fence here.\n"
    )
    assert len(chunk_markdown(text)) == 1


def test_rst_headings_are_found_by_their_underline() -> None:
    text = (
        "Title\n=====\n\nBody text that is comfortably long enough to survive merging.\n\n"
        "Section\n-------\n\nMore body text, also long enough to stand on its own here.\n"
    )
    assert [c.symbol for c in chunk_rst(text)] == ["Title", "Section"]


def test_a_heading_with_nothing_under_it_merges_forwards() -> None:
    text = "# Empty\n\n## Real\n\n" + "Body text that carries the whole of this section.\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert "# Empty" in chunks[0].text
    assert "Body text" in chunks[0].text


def test_an_unknown_suffix_falls_back_to_line_windows() -> None:
    chunks = chunk_file("data.cfg", "a = 1\n" * 5)
    assert len(chunks) == 1
    assert chunks[0].symbol is None


def test_no_chunk_exceeds_the_size_limit() -> None:
    text = "\n".join("line " + "y" * 200 for _ in range(200))
    assert all(len(c.text) <= MAX_CHARS for c in chunk_file("big.txt", text))


def test_an_empty_file_produces_nothing() -> None:
    assert chunk_file("empty.py", "") == []
    assert chunk_file("blank.md", "\n\n\n") == []
