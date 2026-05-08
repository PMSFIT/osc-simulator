"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from osi3.osi_version_pb2 import DESCRIPTOR as _OSI_FILE_DESCRIPTOR
from osi3.osi_version_pb2 import current_interface_version as _osi_version_ext

from osc_simulator import __version__
from osc_simulator.output.osi_writer import SensorViewTraceWriter
from osc_simulator.parser.openscenario import ScenarioParser
from osc_simulator.simulation.engine import SimulationEngine


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="osc-simulator",
        description="Execute an ASAM OpenSCENARIO file and emit ASAM OSI SensorView trace files.",
    )
    p.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    p.add_argument("scenario", type=Path, help="Path to the .xosc input file")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Directory for output .osi trace files (default: current directory)",
    )
    p.add_argument(
        "--step-size",
        type=float,
        default=0.05,
        metavar="SEC",
        help="Simulation time step in seconds (default: 0.05)",
    )
    p.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[0],
        metavar="ID",
        help="Sensor channel IDs to record (default: 0)",
    )
    _osi_version = _OSI_FILE_DESCRIPTOR.GetOptions().Extensions[_osi_version_ext]
    _osi_version_str = (
        f"{_osi_version.version_major}.{_osi_version.version_minor}.{_osi_version.version_patch}"
    )

    p.add_argument(
        "--reported-osi-version",
        type=str,
        default=_osi_version_str,
        metavar="VERSION",
        choices=[
            "3.0.0",
            "3.0.1",
            "3.1.0",
            "3.1.1",
            "3.1.2",
            "3.2.0",
            "3.3.0",
            "3.3.1",
            "3.4.0",
            "3.5.0",
            "3.6.0",
            "3.7.0",
            _osi_version_str,
        ],
        help="Reported OSI version in the output trace files",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    scenario_path: Path = args.scenario
    if not scenario_path.is_file():
        print(f"error: file not found: {scenario_path}", file=sys.stderr)
        return 1

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {scenario_path} …")
    parser = ScenarioParser()
    scenario = parser.parse(scenario_path)

    channel_paths = {
        ch_id: output_dir / f"{scenario_path.stem}_channel{ch_id}.osi" for ch_id in args.channels
    }
    print("Output channels:")
    for ch_id, path in channel_paths.items():
        print(f"  channel {ch_id} → {path}")

    with SensorViewTraceWriter(channel_paths, args.reported_osi_version) as writer:
        engine = SimulationEngine(scenario, step_size=args.step_size)
        frame_count = 0
        for timestamp, ground_truth in engine.run():
            writer.write_frame(timestamp, ground_truth)
            frame_count += 1

    print(f"Done — {frame_count} frames written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
