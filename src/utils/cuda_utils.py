from __future__ import annotations

import torch


class CudaStreamManager:
    """Manage two CUDA streams for concurrent inference on separate feeds.

    On CUDA, run Feed A operations inside ``with torch.cuda.stream(stream_A):``
    and Feed B operations inside ``with torch.cuda.stream(stream_B):``. Call
    :meth:`synchronize` before compositing to ensure both feeds are complete.
    On CPU, both streams are ``None`` and synchronization is a no-op.
    """

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self.stream_A = None
        self.stream_B = None

        if self.device.type != "cuda":
            return

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {self.device}")

        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
            self.device = torch.device("cuda", device_index)
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device_index} is invalid; "
                f"available device count is {torch.cuda.device_count()}."
            )

        self.stream_A = torch.cuda.Stream(device=self.device)
        self.stream_B = torch.cuda.Stream(device=self.device)

    def get_streams(self) -> tuple[torch.cuda.Stream | None, torch.cuda.Stream | None]:
        """Return the streams for Feed A and Feed B, or ``(None, None)`` on CPU."""
        return self.stream_A, self.stream_B

    def synchronize(self) -> None:
        """Wait for queued work on both streams before downstream compositing."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)