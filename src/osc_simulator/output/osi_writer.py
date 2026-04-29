"""Write ASAM OSI SensorView multi-channel trace files.

File format
-----------
Each ``.osi`` file is a binary stream of length-prefixed serialised protobuf
messages (the OSI binary trace format):

    [uint32 LE size][serialised SensorView bytes] ...

One file is written per requested sensor channel.  Every frame contains a
full ``SensorView`` with:
  * ``SensorView.sensor_id``   – the channel identifier
  * ``SensorView.global_ground_truth`` – simulation ground truth at that step
  * ``SensorView.host_vehicle_id``     – set to the first entity if available

The same ground truth is broadcast to all channels; real multi-channel
scenarios would differ by the sensor's mount position and field-of-view
filtering, which can be added on top of this base implementation.
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO

import osi3.osi_sensorview_pb2 as osi_sv


_HEADER = struct.Struct("<I")  # 4-byte little-endian uint32 message size


class SensorViewTraceWriter:
    """Context manager that writes one ``.osi`` trace file per channel.

    Parameters
    ----------
    channel_paths:
        Mapping of ``channel_id → output file path``.
    """

    def __init__(self, channel_paths: dict[int, Path]) -> None:
        self._channel_paths = channel_paths
        self._files: dict[int, BinaryIO] = {}

    def __enter__(self) -> "SensorViewTraceWriter":
        self._files = {
            ch_id: path.open("wb")
            for ch_id, path in self._channel_paths.items()
        }
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()

    def write_frame(self, timestamp_seconds: float, ground_truth: Any) -> None:
        """Serialise *ground_truth* into a ``SensorView`` and write to all channels."""
        for ch_id, fh in self._files.items():
            sv = self._build_sensor_view(ch_id, timestamp_seconds, ground_truth)
            data = sv.SerializeToString()
            fh.write(_HEADER.pack(len(data)))
            fh.write(data)

    # ------------------------------------------------------------------

    def _build_sensor_view(
        self, channel_id: int, timestamp_seconds: float, ground_truth: Any
    ) -> Any:
        sv = osi_sv.SensorView()

        # Sensor identity
        sv.sensor_id.value = channel_id

        # Timestamp mirrors ground truth
        sv.timestamp.CopyFrom(ground_truth.timestamp)

        # Embed full ground truth
        sv.global_ground_truth.CopyFrom(ground_truth)

        # Host vehicle: first moving object if present
        if len(ground_truth.moving_object) > 0:
            sv.host_vehicle_id.CopyFrom(ground_truth.moving_object[0].id)

        # Version
        sv.version.version_major = 3
        sv.version.version_minor = 7
        sv.version.version_patch = 0

        return sv


# ---------------------------------------------------------------------------
# Convenience reader for testing / inspection
# ---------------------------------------------------------------------------

class SensorViewTraceReader:
    """Iterate over ``SensorView`` frames in a single-channel ``.osi`` file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def __iter__(self):  # type: ignore[override]
        with self._path.open("rb") as fh:
            while True:
                header = fh.read(4)
                if not header:
                    break
                (size,) = _HEADER.unpack(header)
                data = fh.read(size)
                if len(data) < size:
                    break
                sv = osi_sv.SensorView()
                sv.ParseFromString(data)
                yield sv
