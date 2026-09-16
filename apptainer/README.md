# Speck Apptainer environment

This image provides the Python and native libraries needed to use SynSense
Speck devices with Sinabs and Samna. It also includes JupyterLab for tutorials.
PyTorch includes CUDA 12.8 for training on NVIDIA GPUs, including Blackwell.
CPU execution remains available; a GPU is not required to deploy or run a
network on Speck.

## Build

Run this from the repository root:

```bash
apptainer build --fakeroot apptainer/speck.sif apptainer/speck.def
```

If unprivileged builds are disabled on the machine, use the build method
provided by its administrator, or build with `sudo apptainer build` when that
is permitted.

Verify the installed Python stack without a device:

```bash
apptainer test apptainer/speck.sif
```

The build and basic test do not require a GPU. CUDA runtime libraries are
installed with PyTorch; the NVIDIA driver stays on the host. Compared with
the old CPU-only image, allow several additional GB for downloads, build
space and the final image. An existing CPU-only SIF must be rebuilt.

On the GPU host, require CUDA and test a small Sinabs forward/backward pass:

```bash
apptainer test --nv --env SPECK_REQUIRE_CUDA=1 apptainer/speck.sif
```

This must print `GPU computation: OK`, not just a successful import. Without
`SPECK_REQUIRE_CUDA=1`, GPU computation is skipped when no GPU is available.
The CUDA wheel selection follows the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v280)
and [Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/).

## Configure USB access on the host (once)

Container udev rules cannot change permissions on host USB devices. Install the
provided rules on the host, reload udev, and then reconnect Speck:

```bash
sudo install -m 0644 apptainer/99-synsense.rules /etc/udev/rules.d/99-synsense.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

The supplied rules match Samna's official rules and grant read/write access to
all local users. On a multi-user machine, ask the administrator to replace
`MODE="0666"` with a suitable group-based policy.

Use a USB 3 cable and port. A Speck2f Dev Kit normally appears as `337d:5bca`:

```bash
lsusb
```

## Run

Open an interactive shell with the USB bus and NVIDIA GPU visible:

```bash
apptainer shell --nv apptainer/speck.sif
```

Omit `--nv` when GPU access is not needed. See the
[Apptainer GPU documentation](https://apptainer.org/docs/user/latest/gpu.html)
for how the host driver and GPU are exposed to the container.

Check that Samna can discover the board:

```bash
apptainer exec apptainer/speck.sif \
  python -c 'import samna; print(samna.device.get_unopened_devices())'
```

Start JupyterLab in the current repository:

```bash
apptainer exec --nv apptainer/speck.sif \
  jupyter lab --ip=127.0.0.1 --no-browser
```

For Samna's graphical visualizer, pass the X11 socket explicitly if it is not
already available through the host's Apptainer configuration:

```bash
apptainer exec \
  --bind /tmp/.X11-unix:/tmp/.X11-unix \
  --env DISPLAY="$DISPLAY" \
  apptainer/speck.sif python your_visualizer_script.py
```

Do not add `--contain` when using Speck; that option replaces the normal `/dev`
view with a minimal one. This setup relies on Apptainer's default host `/dev`
mount. An explicit `/dev/bus/usb` user bind can acquire the `nodev` option and
prevent Samna from opening the device even though `lsusb` can still list it.

## Included versions

- Ubuntu 24.04 / Python 3.12
- PyTorch 2.8.0+cu128 (CUDA 12.8, also supports CPU execution)
- Sinabs 3.1.3
- Samna 0.48.6 (native SynSense wheel)
- NumPy 1.26.4 and OpenCV headless 4.11.0.86 (compatible with Tonic)
- Tonic 1.6.0 and tqdm 4.x for N-MNIST training
- JupyterLab 4.x and ipywidgets 8.x
