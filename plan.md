# KDE RGB Sync Efficiency Plan

## Problem and approach
The running `sync_kde_rgb.py` process is currently consuming high CPU while running as `sync-kde-rgb.service`.  
Measured baseline on the live process (`PID 2911`) shows sustained ~35-55% CPU in realtime samples (`top`) with ~33% long-window average (`ps`), so optimization should target significant reduction while preserving responsive color updates.  
Chosen priority: **Balanced CPU and responsiveness**.

## Workplan
- [x] Measure current runtime cost from the active process/service
- [x] Inspect runtime configuration and command-line defaults
- [x] Add lightweight in-script profiling counters/timers for major loop stages (capture, color extraction, device write)
- [x] Reduce unnecessary device I/O by fixing update gating and introducing sane default change threshold
- [x] Avoid repeated expensive device discovery/setup in hot path by caching device handles/state
- [x] Make capture/color workload tunable with balanced defaults (fps/scale/colors) and explicit "quality/perf" profiles
- [x] Validate improvements with before/after CPU sampling and confirm color responsiveness remains acceptable
- [x] Update service unit defaults/arguments to match balanced profile

## Findings that inform optimization
- Service runs as:
  - `Type=oneshot`
  - `ExecStart=/home/chris/.local/bin/sync_kde_rgb.py`
- Script defaults currently used by service include:
  - `--mode screen`
  - `--fps 1`, `--scale 160x90`, `--colors 5`, `--min-delta 0`
- In screen loop, device updates are effectively sent every frame because condition is:
  - `color_distance(last_sent, smoothed) >= args.min_delta`
  - with default `min_delta=0`, this is always true once `last_sent` is set
- Hot-path operations likely contributing most:
  - Recreating `DeviceManager()` in `set_razer_color()` each send
  - Calling `hid.enumerate()` in `set_keyboard_color()` each send
  - Per-frame image quantization (`resize` + `quantize`) even when color is stable

## Notes
- `sync_kde_rgb.py` in repo and `/home/chris/.local/bin/sync_kde_rgb.py` are identical (same checksum), so repo edits will reflect deployed behavior once reinstalled/restarted.
- Implementation phase should proceed in small, measurable steps with CPU sampling after each change set.
- Post-change sampling on this machine shows service around ~28% CPU (`python3 ... --profile balanced`) versus prior ~33% long-window baseline in notes; responsiveness was confirmed with debug output and live color transitions.
