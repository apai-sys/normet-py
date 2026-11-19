"""Load dialogs name the file that could not be read."""

from __future__ import annotations

from normet.gui._errors import read_error


def test_a_reader_error_names_the_file_first():
    said = read_error("/tmp/obs.csv", ValueError("No columns to parse from file"))
    assert said == "Could not read obs.csv.\nNo columns to parse from file"


def test_a_sentence_that_starts_with_the_path_does_not_say_it_twice():
    said = read_error("/tmp/obs.csv", ValueError("/tmp/obs.csv: the date column is empty"))
    assert said == "Could not read obs.csv.\nthe date column is empty"
