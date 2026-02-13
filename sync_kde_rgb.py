#!/usr/bin/env python3
import argparse
import asyncio
import colorsys
import configparser
import glob
import os
import sys
import time
import uuid

import hid
from PIL import Image
from openrazer.client import DeviceManager

# --- Configuration & Constants ---
KDE_CONFIG_PATH = os.path.expanduser(
    "~/.config/plasma-org.kde.plasma.desktop-appletsrc"
)
POTD_CACHE_DIR = os.path.expanduser("~/.cache/plasma_engine_potd/")

# VIA/QMK Constants for Rainy75
ID_LIGHTING_SET_VALUE = 0x07
CHANNEL_RGB_MATRIX = 0x03
PROP_BRIGHTNESS = 0x01
PROP_EFFECT = 0x02
PROP_COLOR = 0x04
EFFECT_SOLID = 6  # "LIGHT_MODE" from Rainy75 JSON

DEFAULT_FPS = 1
DEFAULT_SCALE = "160x90"
DEFAULT_COLORS = 5
DEFAULT_ALPHA = 1.0
DEFAULT_MIN_DELTA = 0
DEFAULT_MIN_BRIGHTNESS = 30
DEFAULT_DEBUG_INTERVAL = 60


def get_kde_wallpaper():
    """
    Parses the KDE Plasma config to find the current wallpaper image.
    Prioritizes the first valid image found on any screen.
    """
    if not os.path.exists(KDE_CONFIG_PATH):
        print(f"KDE Config not found at {KDE_CONFIG_PATH}", file=sys.stderr)
        return None

    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(KDE_CONFIG_PATH)
    except Exception as e:
        print(f"Error parsing KDE Config: {e}", file=sys.stderr)
        return None

    # Iterate sections to find Containments (Desktops)
    for section in config.sections():
        # Look for sections that define the wallpaper plugin
        # "Containments][273" -> count("]") == 1
        if section.startswith("Containments][") and section.count("]") == 1:
            plugin = config.get(section, "wallpaperplugin", fallback=None)

            if plugin == "org.kde.image":
                # Static Image Mode
                # Look for [Containments][ID][Wallpaper][org.kde.image][General]
                # configparser section key is "Containments][273][Wallpaper][org.kde.image][General"
                # (No trailing ']')
                image_section = f"{section}][Wallpaper][org.kde.image][General"
                if image_section in config:
                    image_path = config.get(image_section, "Image", fallback=None)
                    if image_path:
                        # Clean up file:// prefix
                        if image_path.startswith("file://"):
                            image_path = image_path[7:]
                        return image_path

            elif plugin == "org.kde.potd":
                # Picture of the Day Mode
                # Look for [Containments][ID][Wallpaper][org.kde.potd][General]
                potd_section = f"{section}][Wallpaper][org.kde.potd][General"
                if potd_section in config:
                    provider = config.get(potd_section, "Provider", fallback="bing")

                    if provider == "bing":
                        glob_path = os.path.join(POTD_CACHE_DIR, "bing*")
                        candidates = glob.glob(glob_path)
                        # Filter out .json
                        candidates = [c for c in candidates if not c.endswith(".json")]
                        if candidates:
                            # Return the largest file (most likely the high res image)
                            return max(candidates, key=os.path.getsize)

                    elif provider == "apod":  # NASA
                        return os.path.join(POTD_CACHE_DIR, "apod")

                    elif provider == "wcpotd":  # Wikimedia
                        return os.path.join(POTD_CACHE_DIR, "wcpotd")

                    # Generic fallback
                    glob_path = os.path.join(POTD_CACHE_DIR, f"{provider}*")
                    candidates = glob.glob(glob_path)
                    candidates = [c for c in candidates if not c.endswith(".json")]
                    if candidates:
                        return candidates[0]

    print("No valid wallpaper configuration found in KDE config.", file=sys.stderr)
    return None

    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(KDE_CONFIG_PATH)
    except Exception as e:
        print(f"Error parsing KDE Config: {e}", file=sys.stderr)
        return None

    # Iterate sections to find Containments (Desktops)
    for section in config.sections():
        # Look for sections that define the wallpaper plugin
        # "Containments][273" -> count("]") == 1
        if section.startswith("Containments][") and section.count("]") == 1:
            plugin = config.get(section, "wallpaperplugin", fallback=None)

            if plugin == "org.kde.image":
                # Static Image Mode
                # Look for [Containments][ID][Wallpaper][org.kde.image][General]
                # Because configparser strips outer brackets, section is "Containments][273"
                # We need to construct "Containments][273][Wallpaper]..."
                # So we append "][" + remainder.
                image_section = f"{section}][Wallpaper][org.kde.image][General]"
                if image_section in config:
                    image_path = config.get(image_section, "Image", fallback=None)
                    if image_path:
                        # Clean up file:// prefix
                        if image_path.startswith("file://"):
                            image_path = image_path[7:]
                        return image_path

            elif plugin == "org.kde.potd":
                # Picture of the Day Mode
                # Look for [Containments][ID][Wallpaper][org.kde.potd][General]
                potd_section = f"{section}][Wallpaper][org.kde.potd][General]"
                if potd_section in config:
                    provider = config.get(potd_section, "Provider", fallback="bing")

                    if provider == "bing":
                        glob_path = os.path.join(POTD_CACHE_DIR, "bing*")
                        candidates = glob.glob(glob_path)
                        # Filter out .json
                        candidates = [c for c in candidates if not c.endswith(".json")]
                        if candidates:
                            # Return the largest file (most likely the high res image)
                            return max(candidates, key=os.path.getsize)

                    elif provider == "apod":  # NASA
                        return os.path.join(POTD_CACHE_DIR, "apod")

                    elif provider == "wcpotd":  # Wikimedia
                        return os.path.join(POTD_CACHE_DIR, "wcpotd")

                    # Generic fallback
                    glob_path = os.path.join(POTD_CACHE_DIR, f"{provider}*")
                    candidates = glob.glob(glob_path)
                    candidates = [c for c in candidates if not c.endswith(".json")]
                    if candidates:
                        return candidates[0]

    print("No valid wallpaper configuration found in KDE config.", file=sys.stderr)
    return None


def get_dominant_color(image_path):
    if not image_path or not os.path.exists(image_path):
        print(f"Image path invalid: {image_path}", file=sys.stderr)
        return None

    try:
        img = Image.open(image_path)
        color = (
            img.resize((1, 1), resample=Image.Resampling.LANCZOS)
            .convert("RGB")
            .getpixel((0, 0))
        )
        return color
    except Exception as e:
        print(f"Error processing image {image_path}: {e}", file=sys.stderr)
        return None


def dominant_color_from_image(
    image,
    colors=DEFAULT_COLORS,
    min_brightness=DEFAULT_MIN_BRIGHTNESS,
    sample_size=(64, 36),
):
    try:
        img = image.convert("RGB")
        img = img.resize(sample_size, resample=Image.Resampling.LANCZOS)
        quant = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        palette = quant.getpalette()
        if not palette:
            return None

        color_counts = quant.getcolors() or []
        if not color_counts:
            return None

        def index_to_rgb(index):
            base = index * 3
            return (palette[base], palette[base + 1], palette[base + 2])

        filtered = []
        for count, idx in color_counts:
            rgb = index_to_rgb(idx)
            if sum(rgb) >= min_brightness:
                filtered.append((count, rgb))

        if filtered:
            filtered.sort(key=lambda item: item[0], reverse=True)
            return filtered[0][1]

        brightest = None
        brightest_value = -1
        for count, idx in color_counts:
            rgb = index_to_rgb(idx)
            value = sum(rgb)
            if value > brightest_value:
                brightest_value = value
                brightest = rgb

        return brightest
    except Exception as e:
        print(f"Error computing dominant color: {e}", file=sys.stderr)
        return None


def set_razer_color(rgb):
    try:
        device_manager = DeviceManager()
        if len(device_manager.devices) == 0:
            time.sleep(1)

        devices = device_manager.devices
        print(f"Found {len(devices)} Razer devices.")

        for device in devices:
            try:
                if "Naga X" in device.name and device.fx.advanced:
                    rows = device.fx.advanced.rows
                    cols = device.fx.advanced.cols
                    for r in range(rows):
                        for c in range(cols):
                            device.fx.advanced.matrix[r, c] = rgb
                    device.fx.advanced.draw()
                    print(f" - Set {device.name} matrix color to {rgb}")
                else:
                    if hasattr(device.fx, "static"):
                        device.fx.static(*rgb)
                    print(f" - Set {device.name} static color to {rgb}")
            except Exception as e:
                print(f" - Failed to set {device.name}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Razer init failed: {e}")


def set_keyboard_color(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    qmk_hue = int(h * 255)
    qmk_sat = int(s * 255)

    target_vid = 0x320F
    usage_page = 0xFF60
    usage = 0x61

    found_any = False

    try:
        for d_info in hid.enumerate():
            if (
                d_info["vendor_id"] == target_vid
                and d_info["usage_page"] == usage_page
                and d_info["usage"] == usage
            ):
                try:
                    h_dev = hid.device()
                    h_dev.open_path(d_info["path"])

                    cmd_effect = [
                        ID_LIGHTING_SET_VALUE,
                        CHANNEL_RGB_MATRIX,
                        PROP_EFFECT,
                        EFFECT_SOLID,
                    ]
                    cmd_effect += [0] * (32 - len(cmd_effect))
                    h_dev.write([0] + cmd_effect)
                    time.sleep(0.05)

                    # Ensure Max Brightness
                    cmd_bright = [
                        ID_LIGHTING_SET_VALUE,
                        CHANNEL_RGB_MATRIX,
                        PROP_BRIGHTNESS,
                        255,
                    ]
                    cmd_bright += [0] * (32 - len(cmd_bright))
                    h_dev.write([0] + cmd_bright)
                    time.sleep(0.05)

                    cmd_color = [
                        ID_LIGHTING_SET_VALUE,
                        CHANNEL_RGB_MATRIX,
                        PROP_COLOR,
                        qmk_hue,
                        qmk_sat,
                    ]
                    cmd_color += [0] * (32 - len(cmd_color))
                    h_dev.write([0] + cmd_color)

                    print(
                        f" - Set Rainy75 ({d_info.get('product_string', 'Unknown')}) to Hue={qmk_hue}, Sat={qmk_sat}"
                    )
                    h_dev.close()
                    found_any = True
                except Exception as ex:
                    print(f" - Error writing to keyboard: {ex}", file=sys.stderr)

        if not found_any:
            print("No Rainy 75 keyboard found.")

    except Exception as e:
        print(f"Error enumerating HID devices: {e}", file=sys.stderr)


def smooth_color(previous, current, alpha):
    if previous is None:
        return current
    return tuple(int(previous[i] * (1 - alpha) + current[i] * alpha) for i in range(3))


def color_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


class PortalScreenCapture:
    def __init__(self, width, height, fps, preferred_name=None, preferred_index=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.preferred_name = preferred_name
        self.preferred_index = preferred_index
        self.pipeline = None
        self.sink = None
        self.session_handle = None
        self.fd = None
        self.stream_info = None
        self.streams = None
        self.portal_proxy = None
        self.last_bus_error = None

    async def start(self):
        try:
            import gi  # type: ignore[import-not-found]

            gi.require_version("Gio", "2.0")
            gi.require_version("Gst", "1.0")
            from gi.repository import Gio, GLib, Gst  # type: ignore[import-not-found]
        except Exception as e:
            raise RuntimeError(
                "Missing dependencies for portal capture. "
                "Need python-gobject, gstreamer, pipewire, "
                "xdg-desktop-portal. Error: "
                f"{e}"
            ) from e

        Gst.init(None)

        portal = Gio.DBusProxy.new_for_bus_sync(
            Gio.BusType.SESSION,
            Gio.DBusProxyFlags.NONE,
            None,
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
            "org.freedesktop.portal.ScreenCast",
            None,
        )
        self.portal_proxy = portal

        def wait_for_response(request_path):
            request = Gio.DBusProxy.new_for_bus_sync(
                Gio.BusType.SESSION,
                Gio.DBusProxyFlags.NONE,
                None,
                "org.freedesktop.portal.Desktop",
                request_path,
                "org.freedesktop.portal.Request",
                None,
            )

            loop = GLib.MainLoop()
            result = {}

            def on_signal(_proxy, _sender, signal_name, params):
                if signal_name != "Response":
                    return
                code, details = params.unpack()
                result["code"] = code
                result["details"] = details
                loop.quit()

            request.connect("g-signal", on_signal)
            loop.run()

            if result.get("code") != 0:
                raise RuntimeError(f"Portal request failed: {result.get('code')}")
            return result.get("details", {})

        token = f"sync_kde_rgb_{uuid.uuid4().hex}"
        create_params = GLib.Variant(
            "(a{sv})",
            (
                {
                    "handle_token": GLib.Variant("s", token),
                    "session_handle_token": GLib.Variant("s", f"{token}_session"),
                },
            ),
        )
        request_path = portal.call_sync(
            "CreateSession", create_params, Gio.DBusCallFlags.NONE, -1, None
        ).unpack()[0]
        session_results = wait_for_response(request_path)
        self.session_handle = session_results.get("session_handle")
        if not self.session_handle:
            raise RuntimeError("Portal did not return a session handle.")

        select_params = GLib.Variant(
            "(oa{sv})",
            (
                self.session_handle,
                {
                    "types": GLib.Variant("u", 1),
                    "multiple": GLib.Variant("b", True),
                    "cursor_mode": GLib.Variant("u", 2),
                },
            ),
        )
        request_path = portal.call_sync(
            "SelectSources", select_params, Gio.DBusCallFlags.NONE, -1, None
        ).unpack()[0]
        wait_for_response(request_path)

        start_params = GLib.Variant(
            "(osa{sv})",
            (
                self.session_handle,
                "",
                {"handle_token": GLib.Variant("s", f"{token}_start")},
            ),
        )
        request_path = portal.call_sync(
            "Start", start_params, Gio.DBusCallFlags.NONE, -1, None
        ).unpack()[0]
        start_results = wait_for_response(request_path)
        streams = start_results.get("streams") or []
        if not streams:
            raise RuntimeError("Portal did not provide any streams.")
        self.streams = streams

        selected_stream = None

        if self.preferred_index is not None:
            if 0 <= self.preferred_index < len(streams):
                selected_stream = streams[self.preferred_index]

        if selected_stream is None and self.preferred_name:
            preferred = self.preferred_name.strip().lower()
            if preferred != "auto":
                for stream in streams:
                    props = stream[1] if len(stream) > 1 else {}
                    if not isinstance(props, dict):
                        continue
                    display_name = (
                        props.get("display_name")
                        or props.get("name")
                        or props.get("monitor_name")
                    )
                    if (
                        isinstance(display_name, str)
                        and preferred in display_name.lower()
                    ):
                        selected_stream = stream
                        break

        if selected_stream is None:
            selected_stream = streams[0]
            best_area = -1
            for stream in streams:
                props = stream[1] if len(stream) > 1 else {}
                size = props.get("size") if isinstance(props, dict) else None
                if size and len(size) == 2:
                    area = size[0] * size[1]
                    if area > best_area:
                        best_area = area
                        selected_stream = stream

        node_id = selected_stream[0]
        self.stream_info = selected_stream[1] if len(selected_stream) > 1 else {}
        remote_params = GLib.Variant("(oa{sv})", (self.session_handle, {}))
        fd_result, fd_list = portal.call_with_unix_fd_list_sync(
            "OpenPipeWireRemote",
            remote_params,
            Gio.DBusCallFlags.NONE,
            -1,
            None,
        )
        fd_index = fd_result.unpack()[0]
        self.fd = fd_list.get(fd_index)

        use_target = os.environ.get("SYNC_KDE_RGB_PIPEWIRE_TARGET") == "1"
        source = "target-object" if use_target else "path"
        pipeline_desc = (
            "pipewiresrc fd={fd} {source}={path} do-timestamp=true ! "
            "queue leaky=downstream max-size-buffers=1 ! "
            "videoconvert ! videoscale ! "
            "video/x-raw,format=RGB,width={width},height={height} "
            "! appsink name=sink max-buffers=1 drop=true sync=false"
        ).format(
            fd=self.fd,
            path=node_id,
            width=self.width,
            height=self.height,
            source=source,
        )

        if os.environ.get("SYNC_KDE_RGB_DEBUG_PIPELINE") == "1":
            print(f"Pipeline: {pipeline_desc}")

        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.sink = self.pipeline.get_by_name("sink")
        state_change = self.pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer pipeline failed to start.")

        change, _state, _pending = self.pipeline.get_state(2 * Gst.SECOND)
        if change == Gst.StateChangeReturn.FAILURE:
            bus = self.pipeline.get_bus()
            msg = bus.pop_filtered(Gst.MessageType.ERROR)
            if msg:
                err, debug = msg.parse_error()
                raise RuntimeError(f"GStreamer error: {err} {debug}")
            raise RuntimeError("GStreamer pipeline failed to reach PLAYING.")

    def read_frame(self):
        if not self.sink:
            return None

        import gi  # type: ignore[import-not-found]

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore[import-not-found]

        timeout = int(max(0.5, 1.0 / max(self.fps, 1)) * 1_000_000_000)
        sample = self.sink.emit("try-pull-sample", timeout)
        if sample is None:
            bus = self.pipeline.get_bus() if self.pipeline else None
            if bus:
                msg = bus.pop_filtered(Gst.MessageType.ERROR)
                if msg:
                    err, debug = msg.parse_error()
                    self.last_bus_error = f"{err} {debug}"
            return None

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None

        try:
            data = map_info.data
            return Image.frombytes("RGB", (self.width, self.height), data)
        finally:
            buffer.unmap(map_info)

    def stop(self):
        if self.pipeline:
            import gi  # type: ignore[import-not-found]

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst  # type: ignore[import-not-found]

            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None


async def run_screen_sync(args):
    width, height = args.scale
    capture = PortalScreenCapture(
        width,
        height,
        args.fps,
        preferred_name=args.monitor_name,
        preferred_index=args.monitor_index,
    )
    await capture.start()

    interval = 1.0 / max(args.fps, 1)
    previous = None
    last_sent = None
    frame_count = 0
    loop_count = 0

    print("--- KDE Screen RGB Sync ---")
    print(f"Capture {width}x{height} at {args.fps} fps")
    if capture.streams and args.list_streams:
        print("Available streams:")
        for idx, stream in enumerate(capture.streams):
            props = stream[1] if len(stream) > 1 else {}
            name = None
            size = None
            if isinstance(props, dict):
                name = (
                    props.get("display_name")
                    or props.get("name")
                    or props.get("monitor_name")
                )
                size = props.get("size")
            label = f"{idx}: {name or 'unknown'}"
            if size and len(size) == 2:
                label += f" ({size[0]}x{size[1]})"
            print(label)
        return

    if capture.stream_info and isinstance(capture.stream_info, dict):
        display_name = (
            capture.stream_info.get("display_name")
            or capture.stream_info.get("name")
            or capture.stream_info.get("monitor_name")
        )
        size = capture.stream_info.get("size")
        if isinstance(display_name, str):
            print(f"Selected stream: {display_name}")
        if size and len(size) == 2:
            print(f"Selected stream size: {size[0]}x{size[1]}")

    try:
        while True:
            start = time.monotonic()
            loop_count += 1
            frame = capture.read_frame()
            if frame is not None:
                frame_count += 1
                color = dominant_color_from_image(
                    frame,
                    colors=args.colors,
                    min_brightness=args.min_brightness,
                )
                if color:
                    smoothed = smooth_color(previous, color, args.alpha)
                    previous = smoothed

                    should_send = args.force_update
                    if (
                        last_sent is None
                        or color_distance(last_sent, smoothed) >= args.min_delta
                    ):
                        should_send = True

                    if should_send:
                        set_razer_color(smoothed)
                        set_keyboard_color(smoothed)
                        last_sent = smoothed
                    if args.debug and loop_count % args.debug_interval == 0:
                        print(f"Color RGB{smoothed} (raw {color})")
                elif args.debug and loop_count % args.debug_interval == 0:
                    print("No dominant color computed.")
            elif args.debug and loop_count % args.debug_interval == 0:
                if capture.last_bus_error:
                    print(
                        f"No frame received from capture. Bus error: {capture.last_bus_error}"
                    )
                else:
                    print("No frame received from capture.")

            elapsed = time.monotonic() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    finally:
        capture.stop()


def parse_scale(value):
    if "x" not in value:
        raise argparse.ArgumentTypeError("Scale must be WIDTHxHEIGHT")
    width_str, height_str = value.lower().split("x", 1)
    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Scale must be WIDTHxHEIGHT") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Scale must be positive")

    return width, height


def main():
    parser = argparse.ArgumentParser(description="Sync RGB to KDE screen or wallpaper.")
    parser.add_argument(
        "--mode",
        choices=["screen", "wallpaper"],
        default="screen",
        help="Sync to live screen or static wallpaper",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--scale", type=parse_scale, default=parse_scale(DEFAULT_SCALE))
    parser.add_argument("--colors", type=int, default=DEFAULT_COLORS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--min-delta", type=int, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--min-brightness", type=int, default=DEFAULT_MIN_BRIGHTNESS)
    parser.add_argument("--debug", action="store_true", help="Print debug output")
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=DEFAULT_DEBUG_INTERVAL,
        help="Print debug every N frames",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Send color every frame",
    )
    parser.add_argument(
        "--instant",
        action="store_true",
        help="Disable smoothing and thresholds",
    )
    parser.add_argument(
        "--monitor-name",
        default="Samsung Odyssey",
        help='Match display name ("auto" for largest stream)',
    )
    parser.add_argument(
        "--monitor-index",
        type=int,
        default=None,
        help="Select stream by index (overrides name)",
    )
    parser.add_argument(
        "--list-streams",
        action="store_true",
        help="List available portal streams and exit",
    )
    args = parser.parse_args()

    if args.instant:
        args.alpha = 1.0
        args.min_delta = 0

    if args.mode == "screen":
        try:
            asyncio.run(run_screen_sync(args))
        except KeyboardInterrupt:
            print("Stopped.")
        except Exception as e:
            print(f"Screen capture failed: {e}", file=sys.stderr)
            sys.exit(1)
        return

    print("--- KDE Wallpaper RGB Sync ---")
    image_path = get_kde_wallpaper()

    if image_path:
        print(f"Detected Wallpaper: {image_path}")
        color = get_dominant_color(image_path)

        if color:
            print(f"Calculated Average Color: RGB{color}")
            set_razer_color(color)
            set_keyboard_color(color)
        else:
            print("Could not determine color.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Could not detect current wallpaper from KDE config.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
