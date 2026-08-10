# speck-tutorial

## DVS recording

`02_record_dvs.py` records events until `q` is pressed, then converts them to a
30 FPS MP4. Run it from an interactive Apptainer shell so that a single key can
be read without pressing Enter:

Rebuild the SIF from the current `apptainer/speck.def` before running this
script; video output requires the OpenCV package included in that definition.

```bash
python script/02_record_dvs.py
```

The default output is `output/dvs_recording.mp4`. Useful options include:

```bash
python script/02_record_dvs.py \
  --output output/example.mp4 \
  --fps 30 \
  --gain 32
```

For a fixed-duration, non-interactive recording, use `--duration`:

```bash
python script/02_record_dvs.py --duration 5
```
