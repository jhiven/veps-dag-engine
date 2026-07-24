"""Unit tests for DAG graph topology construction."""

from __future__ import annotations

from usecases.video_analytics.config import RTDETRConfig
from usecases.video_analytics.graph import build_video_analytics_specification


def test_graph_topology_and_stable_node_ids() -> None:
    """Verify stable node IDs, pin names, and edges in video analytics specification."""
    init_cfg = RTDETRConfig(model_id="PekingU/rtdetr_r18vd")
    cand_cfg = RTDETRConfig(model_id="PekingU/rtdetr_r50vd")

    spec_init = build_video_analytics_specification(detector_config=init_cfg)
    spec_cand = build_video_analytics_specification(detector_config=cand_cfg)

    # Node IDs must be identical
    init_node_ids = tuple(n.node_id for n in spec_init.nodes)
    cand_node_ids = tuple(n.node_id for n in spec_cand.nodes)
    assert init_node_ids == ("detector", "person_filter", "tracker", "sink")
    assert cand_node_ids == ("detector", "person_filter", "tracker", "sink")

    # Edges must be identical
    assert spec_init.edges == spec_cand.edges
    assert len(spec_init.edges) == 3

    # Only detector configuration differs
    det_init = next(n for n in spec_init.nodes if n.node_id == "detector")
    det_cand = next(n for n in spec_cand.nodes if n.node_id == "detector")

    assert det_init.configuration["model_id"] == "PekingU/rtdetr_r18vd"
    assert det_cand.configuration["model_id"] == "PekingU/rtdetr_r50vd"

    # Tracker nodes must be identical
    trk_init = next(n for n in spec_init.nodes if n.node_id == "tracker")
    trk_cand = next(n for n in spec_cand.nodes if n.node_id == "tracker")
    assert trk_init == trk_cand
