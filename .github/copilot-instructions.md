# Copilot Instructions for RGB Sync

## Build, test, and lint commands
- This repository is a single Python script (`sync_kde_rgb.py`) with no project-defined build, lint, or test tooling.
- Run the CLI help to validate argument parsing and defaults:
  - `python sync_kde_rgb.py --help`
- Run the script in wallpaper mode (single-run path):
  - `python sync_kde_rgb.py --mode wallpaper`
- Run the script in screen mode (continuous loop path):
  - `python sync_kde_rgb.py --mode screen --fps 1`
- There is no existing single-test command because no test suite is currently present.

## High-level architecture
- Entry point is `main()`, which parses CLI flags and chooses one of two execution paths:
  - **Screen mode**: `run_screen_sync(args)` captures live frames from KDE/Wayland through `PortalScreenCapture`, computes dominant color per frame, smooths/throttles updates, then pushes color to supported devices.
  - **Wallpaper mode**: `get_kde_wallpaper()` resolves current wallpaper path from KDE config/cache, then `get_dominant_color()` computes a single color and applies it to devices once.
- Color processing flow:
  - For live frames, `dominant_color_from_image()` downsamples and quantizes (`Pillow`) to select a dominant RGB value with brightness filtering fallback.
  - `smooth_color()` and `color_distance()` control update stability (`--alpha`, `--min-delta`, `--instant`, `--force-update`).
- Device output flow sends the same RGB result to two targets:
  - `set_razer_color()` via OpenRazer (`openrazer.client.DeviceManager`)
  - `set_keyboard_color()` via raw HID command packets (Rainy75/VIA-QMK path)

## Key conventions in this codebase
- CLI defaults are intentionally opinionated for this user setup:
  - `--mode screen`
  - `--monitor-name "Samsung Odyssey"`
  - `--scale 160x90`, `--fps 1`, `--colors 5`
- `--instant` is a shortcut mode that mutates runtime args (`alpha=1.0`, `min_delta=0`) instead of branching through separate logic.
- Screen capture monitor selection priority is:
  1. `--monitor-index` (if valid)
  2. `--monitor-name` substring match (unless `"auto"`)
  3. Largest stream by area
- Optional runtime env flags affecting capture internals:
  - `SYNC_KDE_RGB_PIPEWIRE_TARGET=1` switches `pipewiresrc` from `path=` to `target-object=`
  - `SYNC_KDE_RGB_DEBUG_PIPELINE=1` prints full GStreamer pipeline string
- Error handling convention is print-to-stderr + continue where possible for device-level failures, but exit non-zero on unrecoverable capture/setup failures in `main()`.
