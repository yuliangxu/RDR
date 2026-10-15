"""Subject allocation and exact source taxonomy contracts for the AGP study."""

import csv

import numpy as np
import pandas as pd
import pytest

from experiments.AGP import selection_data as data


def test_subject_targets_keep_entire_subjects_and_use_every_row():
    subjects = np.repeat([f"subject-{i:02d}" for i in range(20)], [1, 3, 2, 4] * 5)
    original = pd.DataFrame(dict(subject_id=subjects, source_row=np.arange(len(subjects))))
    pieces = data._subject_parts(original, [.7, .1, .1, .1], 19)
    repeated = data._subject_parts(original, [.7, .1, .1, .1], 19)
    assert sorted(np.concatenate([part.source_row for part in pieces])) == list(range(len(original)))
    previous = set()
    for part, second in zip(pieces, repeated):
        assert not previous.intersection(part.subject_id)
        previous.update(part.subject_id)
        pd.testing.assert_frame_equal(part, second)
        assert part.source_row.is_monotonic_increasing
        for subject, group in part.groupby("subject_id"):
            assert len(group) == original.subject_id.eq(subject).sum()


def test_subject_prefix_tie_chooses_shorter_half():
    frame = pd.DataFrame(dict(subject_id=["a", "b", "c"], source_row=range(3)))
    parts = data._subject_parts(frame, [.5, .5], 42)
    assert [len(part) for part in parts] == [1, 2]


def test_row_targets_do_not_lose_rows_at_floating_point_boundaries():
    frame = pd.DataFrame(dict(source_row=np.arange(100)))
    parts = data._row_parts(frame, [.7, .1, .1, .1], 42)
    assert [len(part) for part in parts] == [70, 10, 10, 10]
    assert sorted(np.concatenate([part.source_row for part in parts])) == list(range(100))
    assert all(part.source_row.is_monotonic_increasing for part in parts)


def test_taxa_rows_follow_abundance_order_and_blank_ranks_are_explicit(tmp_path):
    path = tmp_path / "abundance.csv"
    taxa = ["k__Bacteria; p__Firmicutes; c__; o__; f__; g__A; s__one",
            "k__Bacteria; p__[Thermi]; c__Deinococci; o__; f__; g__; s__"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "sample-1"])
        writer.writerows([[identifier, value] for value, identifier in enumerate(taxa)])
    result = data._taxa_table(path, 2)
    assert result.taxon_id.tolist() == taxa
    assert result.taxon_index.tolist() == [0, 1]
    assert result.phylum.tolist() == ["Firmicutes", "[Thermi]"]
    assert result["class"].tolist() == ["Unclassified", "Deinococci"]


@pytest.mark.parametrize("kind", ["role", "side"])
def test_partition_rejects_subject_overlap_across_either_axis(kind):
    frame = pd.DataFrame(dict(
        distribution=["P", "Q", "P", "Q"], role=["train", "train", "test", "test"],
        sample_id=["a", "b", "c", "d"], source_file=["raw.csv"] * 4,
        source_row=range(4), original_distribution=["P"] * 4,
        subject_id=["subject-a", "subject-b", "subject-c", "subject-d"],
    ))
    data._assert_partition(frame, ["train", "test"], "null")
    frame.loc[2 if kind == "role" else 1, "subject_id"] = "subject-a"
    with pytest.raises(AssertionError, match="subject crosses"):
        data._assert_partition(frame, ["train", "test"], "null")
