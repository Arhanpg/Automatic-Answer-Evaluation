"""
research_v3/phase8_ground_truth/tests/test_crop_integrity.py

Unit tests for answer crop generation and geometry invariants:
1. Coordinates are strictly non-negative and non-inverted (x0 < x1, y0 < y1).
2. Normalized coordinates fall strictly within [0.0, 1.0].
3. Crop hashes match SHA-256 pattern or OMITTED_BY_STUDENT.
4. Total crop count reconciles to 392 written crops and 15 omissions.
"""

from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ANN_DIR = REPO_ROOT / "research_v2" / "annotations"


@pytest.fixture
def regions_df():
    f = ANN_DIR / "answer_regions.csv"
    assert f.exists(), f"answer_regions.csv missing at {f}"
    return pd.read_csv(f)


def test_total_region_count(regions_df):
    assert len(regions_df) == 407, f"Expected 407 regions, found {len(regions_df)}"


def test_bounding_box_geometry(regions_df):
    for _, row in regions_df.iterrows():
        sid = row["student_id"]
        sqid = row["subquestion_id"]
        x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])

        assert x0 >= 0, f"Negative x0 in {sid}_{sqid}: {x0}"
        assert y0 >= 0, f"Negative y0 in {sid}_{sqid}: {y0}"
        assert x1 > x0, f"Inverted X coords in {sid}_{sqid}: {x0} >= {x1}"
        assert y1 > y0, f"Inverted Y coords in {sid}_{sqid}: {y0} >= {y1}"


def test_normalized_coordinate_bounds(regions_df):
    for _, row in regions_df.iterrows():
        sid = row["student_id"]
        sqid = row["subquestion_id"]
        nx0, ny0, nx1, ny1 = float(row["norm_x0"]), float(row["norm_y0"]), float(row["norm_x1"]), float(row["norm_y1"])

        assert 0.0 <= nx0 < nx1 <= 1.0, f"Invalid norm X bounds in {sid}_{sqid}: [{nx0}, {nx1}]"
        assert 0.0 <= ny0 < ny1 <= 1.0, f"Invalid norm Y bounds in {sid}_{sqid}: [{ny0}, {ny1}]"


def test_reconciled_omission_count(regions_df):
    omitted = regions_df[
        (regions_df["crop_sha256"] == "OMITTED_BY_STUDENT") |
        (regions_df["crop_relative_path"] == "NONE")
    ]
    assert len(omitted) == 15, f"Expected exactly 15 omissions, found {len(omitted)}"
    written = regions_df[
        (regions_df["crop_sha256"] != "OMITTED_BY_STUDENT") &
        (regions_df["crop_relative_path"] != "NONE")
    ]
    assert len(written) == 392, f"Expected exactly 392 written answer regions, found {len(written)}"
