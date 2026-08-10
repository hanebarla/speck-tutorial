# Speck Apptainer environment

This image provides the Python and native libraries needed to use SynSense
Speck devices with Sinabs and Samna. It also includes JupyterLab for tutorials.
The default PyTorch build is CPU-only because a GPU is not required to deploy
or run a network on Speck.

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

Open an interactive shell with the USB bus visible:

```bash
apptainer shell --bind /dev/bus/usb:/dev/bus/usb apptainer/speck.sif
```

Check that Samna can discover the board:

```bash
apptainer exec --bind /dev/bus/usb:/dev/bus/usb apptainer/speck.sif \
  python -c 'import samna; print(samna.device.get_unopened_devices())'
```

Start JupyterLab in the current repository:

```bash
apptainer exec --bind /dev/bus/usb:/dev/bus/usb apptainer/speck.sif \
  jupyter lab --ip=127.0.0.1 --no-browser
```

For Samna's graphical visualizer, pass the X11 socket explicitly if it is not
already available through the host's Apptainer configuration:

```bash
apptainer exec \
  --bind /dev/bus/usb:/dev/bus/usb \
  --bind /tmp/.X11-unix:/tmp/.X11-unix \
  --env DISPLAY="$DISPLAY" \
  apptainer/speck.sif python your_visualizer_script.py
```

Do not add `--contain` unless the USB device and X11 socket are also bound into
the container; that option replaces the normal `/dev` view with a minimal one.

## Included versions

- Ubuntu 24.04 / Python 3.12
- PyTorch 2.8.0 (CPU)
- Sinabs 3.1.3
- Samna 0.48.6 (native SynSense wheel)
- JupyterLab 4.x and ipywidgets 8.x
