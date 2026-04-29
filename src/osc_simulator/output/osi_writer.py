"""Write ASAM OSI SensorView multi-channel trace files via osi_utilities."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Iterator

from osi3.osi_sensorview_pb2 import SensorView
from osi3.osi_version_pb2 import DESCRIPTOR as _OSI_FILE_DESCRIPTOR, current_interface_version as _OSI_VERSION_EXT
from osi_utilities import SingleTraceWriter, SingleTraceReader, MessageType

_OSI_VERSION = _OSI_FILE_DESCRIPTOR.GetOptions().Extensions[_OSI_VERSION_EXT]


class SensorViewTraceWriter:
    """Context manager that writes one ``.osi`` trace file per channel.

    Parameters
    ----------
    channel_paths:
        Mapping of ``channel_id → output file path``.
    """

    def __init__(self, channel_paths: dict[int, Path]) -> None:
        self._channel_paths = channel_paths
        self._writers: dict[int, SingleTraceWriter] = {}

    def __enter__(self) -> "SensorViewTraceWriter":
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
        if len(ground_truth.moving_object) > 0:
            sv.host_vehicle_id.CopyFrom(ground_truth.moving_object[0].id)
        sv.version.CopyFrom(_OSI_VERSION)
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
