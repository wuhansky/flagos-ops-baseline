# OpenCV CUDA baselines

`add.py` collects **cv::cuda::add**, not CPU OpenCV or a Torch substitute.
Initial scope: contiguous H×W single-channel matrices, float32 and uint8,
same output dtype, no mask, distinct preallocated output. uint8 addition
saturates to 255. Sizes are synthetic, selected for coverage; they are not
claimed to come from a production workload. Keys are `[H, W]` within this
fixed contract. Mask/channel/output-dtype variants require distinct keys.

Build OpenCV and opencv_contrib at the same version with CUDA enabled and
modules core/cudev/cudaarithm. A non-CUDA OpenCV build will fail preflight;
there is no CPU fallback. Then build the small bridge in this directory:

```sh
cmake -S ops/opencv -B /tmp/opencv-bridge \
  -DOpenCV_DIR=/path/to/opencv-install/lib/cmake/opencv4
cmake --build /tmp/opencv-bridge
export OPENCV_BASELINE_LIBRARY=/tmp/opencv-bridge/libopencv_baseline.so
python -m pytest tests/test_opencv_add.py -q
python tools/collect_baseline_nvidia.py --repo opencv --ops add \
  --device 0 --output op_opencv_perf_baseline.json --report-dir /tmp/opencv-ncu
```

Use `--no-ncu` only for latency-only validation. Actual GPU identity is in
`_environment`; `--reference-chip H800` does not make a run on H20 an H800
baseline. No static production H800 snapshot is included until collected there.

The bridge wraps existing Torch device pointers as GpuMat headers and uses
Torch's current CUDA stream. Allocation, input creation, copies and output
validation happen outside timing; the timed callable enqueues native add into
the collector's CUDA-event stream. NCU uses the same adapter in its subprocess.
Per-record config includes OpenCV version, build-info digest and bridge digest.
Input tensors are read-only; output is fully overwritten on every invocation.

This profile was validated against OpenCV 4.12.0; the request linked the moving
5.x source. Source API: opencv_contrib/modules/cudaarithm/src/element_operations.cpp.
