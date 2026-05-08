"""Tests for the OSI SensorView trace writer / reader round-trip."""

import tempfile
from pathlib import Path

import osi3.osi_groundtruth_pb2 as osi_gt
import pytest

from osc_simulator.output.osi_writer import SensorViewTraceReader, SensorViewTraceWriter
from osc_simulator.parser.openscenario import ScenarioParser
from osc_simulator.simulation.engine import SimulationEngine

EXAMPLE = Path(__file__).parent.parent / "examples" / "simple_scenario.xosc"


def _build_minimal_gt(seconds: int = 0, nanos: int = 0) -> osi_gt.GroundTruth:
    gt = osi_gt.GroundTruth()
    gt.timestamp.seconds = seconds
    gt.timestamp.nanos = nanos
    return gt


def test_writer_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {0: Path(tmpdir) / "ch0.osi", 1: Path(tmpdir) / "ch1.osi"}
        with SensorViewTraceWriter(paths) as writer:
            writer.write_frame(0.0, _build_minimal_gt())
        for p in paths.values():
            assert p.exists()
            assert p.stat().st_size > 0


def test_round_trip_frame_count():
    scenario = ScenarioParser().parse(EXAMPLE)
    engine = SimulationEngine(scenario, step_size=0.1)
    frames = list(engine.run())

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.osi"
        with SensorViewTraceWriter({0: path}) as writer:
            for ts, gt in frames:
                writer.write_frame(ts, gt)

        read_frames = list(SensorViewTraceReader(path))

    assert len(read_frames) == len(frames)


def test_round_trip_timestamps():
    scenario = ScenarioParser().parse(EXAMPLE)
    engine = SimulationEngine(scenario, step_size=0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.osi"
        source_timestamps = []
        with SensorViewTraceWriter({0: path}) as writer:
            for ts, gt in engine.run():
                writer.write_frame(ts, gt)
                source_timestamps.append(ts)

        for sv, expected_ts in zip(SensorViewTraceReader(path), source_timestamps, strict=False):
            reconstructed = sv.timestamp.seconds + sv.timestamp.nanos * 1e-9
            assert reconstructed == pytest.approx(expected_ts, abs=1e-6)


def test_sensor_view_has_channel_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ch42.osi"
        with SensorViewTraceWriter({42: path}) as writer:
            writer.write_frame(0.0, _build_minimal_gt())

        frames = list(SensorViewTraceReader(path))
        assert frames[0].sensor_id.value == 42


def test_multi_channel_independent_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {i: Path(tmpdir) / f"ch{i}.osi" for i in range(3)}
        with SensorViewTraceWriter(paths) as writer:
            writer.write_frame(1.0, _build_minimal_gt(1, 0))

        for ch_id, path in paths.items():
            frames = list(SensorViewTraceReader(path))
            assert len(frames) == 1
            assert frames[0].sensor_id.value == ch_id
