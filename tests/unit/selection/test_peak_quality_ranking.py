from sifter.selection.quality import CandidateQuality, rank_quality_first


def test_quality_first_ranking_prioritizes_local_fit_over_bic() -> None:
    rows = (
        CandidateQuality(
            candidate_index=0,
            severe_note_count=0,
            worst_local_area_error=0.35,
            median_height_error=0.10,
            normalized_rmse=0.020,
            peak_count=3,
            bic=10.0,
        ),
        CandidateQuality(
            candidate_index=1,
            severe_note_count=0,
            worst_local_area_error=0.08,
            median_height_error=0.06,
            normalized_rmse=0.030,
            peak_count=3,
            bic=30.0,
        ),
        CandidateQuality(
            candidate_index=2,
            severe_note_count=1,
            worst_local_area_error=0.01,
            median_height_error=0.01,
            normalized_rmse=0.010,
            peak_count=2,
            bic=1.0,
        ),
    )

    assert [row.candidate_index for row in rank_quality_first(rows)] == [1, 0, 2]
