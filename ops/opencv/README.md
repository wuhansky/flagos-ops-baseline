# OpenCV CUDA baselines

`add.py` collects **cv::cuda::add**, not CPU OpenCV or a Torch substitute.
Initial scope: contiguous H×W single-channel matrices, float32 and uint8,
same output dtype, no mask, distinct preallocated output. uint8 addition
saturates to 255. Sizes are synthetic, selected for coverage; they are not
claimed to come from a production workload. Keys are `[H, W]` within this
fixed contract. Mask/channel/output-dtype variants require distinct keys.

Build OpenCV and opencv_contrib at the same version with CUDA enabled and
modules core/cudev/cudaarithm. For example, with both official 4.12.0 release
sources extracted (adjust CUDA_ARCH_BIN for the actual device):

```sh
cmake -S /path/to/opencv-4.12.0 -B /tmp/opencv-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/opencv-install \
  -DOPENCV_EXTRA_MODULES_PATH=/path/to/opencv_contrib-4.12.0/modules \
  -DBUILD_LIST=core,cudev,cudaarithm -DWITH_CUDA=ON \
  -DCUDA_ARCH_BIN=9.0 -DCUDA_ARCH_PTX= \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_python3=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF
cmake --build /tmp/opencv-build -j4
cmake --install /tmp/opencv-build
```

OpenCV 4.12.0 tag: `cbee6841638edb6fbc8110df7cd52bb8e3d66211`;
opencv_contrib tag: `7deb35fde4d38d73b6173c7ab2aeddc8df5a89e3`.
A non-CUDA OpenCV build will fail preflight;
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
