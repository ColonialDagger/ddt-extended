import math
import struct

import numpy as np
import wgpu
import wgpu.utils


class GPUPipeline:
    """
    Minimal GPU pipeline for Destiny 2 boss health estimation.

    - Single-pass shader: classify + total counts
    - Ping-pong stats buffers for one-frame latency
    - Reads back 8 bytes per read (two u32 counters)
    - Optional: read back only every N frames (readback_interval)
    """

    def __init__(
        self,
        width: int,
        height: int,
        lut_path: str,
        shader_path: str,
        readback_interval: int = 1,
    ):
        """
        Args:
            width: cropped bar width in pixels.
            height: cropped bar height in pixels.
            lut_path: path to LUT binary (uint32[256*256]).
            shader_path: path to WGSL shader with entry_point "classify_main".
            readback_interval: how often to read GPU results (1 = every frame,
                               2 = every other frame, etc.).
        """
        self.width = width
        self.height = height
        self.readback_interval = max(1, int(readback_interval))

        # -----------------------------
        # Device + Queue
        # -----------------------------
        self.device = wgpu.utils.get_default_device()
        self.queue = self.device.queue

        # -----------------------------
        # Load LUT (uint32[256*256])
        # -----------------------------
        with open(lut_path, "rb") as f:
            lut_bytes = f.read()

        self.lut_buffer = self.device.create_buffer_with_data(
            data=lut_bytes,
            usage=wgpu.BufferUsage.STORAGE,
        )

        # -----------------------------
        # Load WGSL shader
        # -----------------------------
        with open(shader_path, "r", encoding="utf-8") as f:
            shader_code = f.read()

        shader_module = self.device.create_shader_module(code=shader_code)

        # -----------------------------
        # Create buffers
        # -----------------------------
        pixel_buf_size = width * height * 4  # one u32 per pixel
        mask_buf_size = width * height * 4   # one u32 per pixel

        self.pixel_buffer = self.device.create_buffer(
            size=pixel_buf_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        self.mask_buffer = self.device.create_buffer(
            size=mask_buf_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        # Two tiny stats buffers: [total_healthy, total_mask] as u32
        self.stats_size_bytes = 2 * 4  # 2 u32
        self.stats_buffers = [
            self.device.create_buffer(
                size=self.stats_size_bytes,
                usage=(
                    wgpu.BufferUsage.STORAGE
                    | wgpu.BufferUsage.COPY_SRC
                    | wgpu.BufferUsage.COPY_DST
                ),
            )
            for _ in range(2)
        ]

        # Params buffer (width, height) as uniform (padded to 16 bytes)
        self.params_buffer = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # -----------------------------
        # Bind group layout / pipeline
        # -----------------------------
        bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
            ]
        )

        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        self.compute_pipeline = self.device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "classify_main"},
        )

        self.bind_groups = [
            self.device.create_bind_group(
                layout=bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": self.pixel_buffer}},
                    {"binding": 1, "resource": {"buffer": self.mask_buffer}},
                    {"binding": 2, "resource": {"buffer": self.lut_buffer}},
                    {"binding": 3, "resource": {"buffer": stats_buf}},
                    {"binding": 4, "resource": {"buffer": self.params_buffer}},
                ],
            )
            for stats_buf in self.stats_buffers
        ]

        # -----------------------------
        # State
        # -----------------------------
        self.current_idx = 0        # which stats buffer this frame uses
        self.warmup_done = False    # whether we have a previous frame's result
        self.last_health: float = 0.0
        self.frame_index: int = 0   # total frames processed

        # Static params (width, height) + padding
        params = struct.pack("IIII", self.width, self.height, 0, 0)
        self.queue.write_buffer(self.params_buffer, 0, params)

    def run(
        self,
        cropped_img: np.ndarray,
        neg_mask: np.ndarray,
    ) -> float:
        """
        Run one GPU classification pass.

        Returns:
            health in [0.0, 1.0].

        Behavior:
            - On the very first call, returns 0.0 (no previous frame yet).
            - Otherwise returns the last known GPU health.
            - If readback_interval > 1, health updates every N frames.
        """
        h, w, c = cropped_img.shape
        assert c == 3
        assert w == self.width and h == self.height

        # -----------------------------
        # Upload pixel data (BGR -> packed BGRA u32)
        # -----------------------------
        b = cropped_img[..., 0].astype(np.uint32)
        g = cropped_img[..., 1].astype(np.uint32)
        r = cropped_img[..., 2].astype(np.uint32)
        a = np.full_like(r, 255, dtype=np.uint32)

        packed = (a << 24) | (r << 16) | (g << 8) | b
        self.queue.write_buffer(self.pixel_buffer, 0, packed.tobytes())

        # -----------------------------
        # Upload neg_mask as u32
        # -----------------------------
        if neg_mask.ndim == 3:
            mask_u32 = neg_mask[..., 0].astype(np.uint32)
        else:
            mask_u32 = neg_mask.astype(np.uint32)

        self.queue.write_buffer(self.mask_buffer, 0, mask_u32.tobytes())

        # -----------------------------
        # Ping-pong stats buffers
        # -----------------------------
        idx = self.current_idx
        prev_idx = 1 - idx

        # Clear current stats buffer
        zero_bytes = bytes(self.stats_size_bytes)
        self.queue.write_buffer(self.stats_buffers[idx], 0, zero_bytes)

        # -----------------------------
        # Dispatch classify_main for current frame
        # -----------------------------
        x_groups = math.ceil(self.width / 16)
        y_groups = math.ceil(self.height / 16)

        encoder = self.device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(self.compute_pipeline)
        pass_enc.set_bind_group(0, self.bind_groups[idx])
        pass_enc.dispatch_workgroups(x_groups, y_groups, 1)
        pass_enc.end()

        self.queue.submit([encoder.finish()])

        # -----------------------------
        # Read back previous frame's result occasionally (sync)
        # -----------------------------
        do_readback = (
            self.warmup_done
            and (self.frame_index % self.readback_interval == 0)
        )

        if do_readback:
            stats_bytes = self.queue.read_buffer(
                self.stats_buffers[prev_idx], 0, self.stats_size_bytes
            )
            stats_u32 = np.frombuffer(stats_bytes, dtype=np.uint32)
            total_healthy = int(stats_u32[0])
            total_mask = int(stats_u32[1])

            if total_mask > 0:
                health_val = total_healthy / total_mask
                health_val = max(0.0, min(1.0, float(health_val)))
            else:
                health_val = 0.0

            self.last_health = health_val
        else:
            if not self.warmup_done:
                # First frame: we just ran classification but have
                # no previous stats yet. Mark warmup done so the next
                # frame can read back.
                self.warmup_done = True

        # Advance ping-pong index and frame counter
        self.current_idx = prev_idx
        self.frame_index += 1

        return self.last_health
