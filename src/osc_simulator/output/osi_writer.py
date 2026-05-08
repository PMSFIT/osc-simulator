"""Write ASAM OSI SensorView multi-channel trace files via osi_utilities."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

from osi3.osi_sensorview_pb2 import SensorView
from osi3.osi_version_pb2 import DESCRIPTOR as _OSI_FILE_DESCRIPTOR
from osi3.osi_version_pb2 import current_interface_version as _osi_version_ext
from osi_utilities import MessageType, SingleTraceReader, SingleTraceWriter

_OSI_VERSION = _OSI_FILE_DESCRIPTOR.GetOptions().Extensions[_osi_version_ext]
_OSI_VERSION_STR = (
    f"{_OSI_VERSION.version_major}.{_OSI_VERSION.version_minor}.{_OSI_VERSION.version_patch}"
)


class SensorViewTraceWriter:
    """Context manager that writes one ``.osi`` trace file per channel.

    Parameters
    ----------
    channel_paths:
        Mapping of ``channel_id → output file path``.
    reported_osi_version:
        OSI version to report in the output trace files.
    """

    def __init__(
        self, channel_paths: dict[int, Path], reported_osi_version: str = _OSI_VERSION_STR
    ) -> None:
        self._channel_paths = channel_paths
        self._reported_osi_version = list(map(int, reported_osi_version.split(".")))  # type: ignore[assignment]
        assert len(self._reported_osi_version) == 3, "Invalid OSI version format"
        self._writers: dict[int, SingleTraceWriter] = {}

    def __enter__(self) -> SensorViewTraceWriter:
        for ch_id, path in self._channel_paths.items():
            writer = SingleTraceWriter()
            writer.open(path)
            self._writers[ch_id] = writer
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def write_frame(self, timestamp_seconds: float, ground_truth: Any) -> None:
        """Serialise *ground_truth* into a ``SensorView`` and write to all channels."""
        for ch_id, writer in self._writers.items():
            sv = self._build_sensor_view(ch_id, timestamp_seconds, ground_truth)
            writer.write_message(sv)

    def _build_sensor_view(
        self, channel_id: int, timestamp_seconds: float, ground_truth: Any
    ) -> SensorView:
        sv = SensorView()
        sv.sensor_id.value = channel_id
        sv.timestamp.CopyFrom(ground_truth.timestamp)
        sv.global_ground_truth.CopyFrom(ground_truth)
        sv.global_ground_truth.version.version_major = self._reported_osi_version[0]
        sv.global_ground_truth.version.version_minor = self._reported_osi_version[1]
        sv.global_ground_truth.version.version_patch = self._reported_osi_version[2]
        sv.version.CopyFrom(sv.global_ground_truth.version)
        sv.mounting_position.position.x = 0.0
        sv.mounting_position.position.y = 0.0
        sv.mounting_position.position.z = 0.0
        sv.mounting_position.orientation.roll = 0.0
        sv.mounting_position.orientation.pitch = 0.0
        sv.mounting_position.orientation.yaw = 0.0
        if len(ground_truth.moving_object) > 0:
            sv.host_vehicle_id.CopyFrom(ground_truth.moving_object[0].id)
        return sv


class SensorViewTraceReader:
    """Iterate over ``SensorView`` frames in a single-channel ``.osi`` file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def __iter__(self) -> Iterator[SensorView]:
        reader = SingleTraceReader()
        reader.set_message_type(MessageType.SENSOR_VIEW)
        if not reader.open(self._path):
            return
        try:
            for result in reader:
                if result.message is not None:
                    yield result.message  # type: ignore[misc]
        finally:
            reader.close()
