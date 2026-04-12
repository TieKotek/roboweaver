import argparse
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_robots import (
    SceneBuilder,
    SimulationTimeVideoScheduler,
    build_arg_parser,
    configure_export_camera,
    create_video_writer,
    describe_video_dimension_adjustment,
    normalize_video_dimensions,
    should_launch_viewer,
    should_sleep_for_realtime_pacing,
    validate_video_export_args,
)


class VideoExportArgValidationTests(unittest.TestCase):
    def test_allows_video_export_without_named_camera(self):
        args = argparse.Namespace(
            save_video="logs/test.mp4",
            video_fps=30.0,
            width=1280,
            height=720,
            camera_zoom_out=1.5,
        )

        validate_video_export_args(args)

    def test_rejects_non_positive_video_fps(self):
        args = argparse.Namespace(
            save_video="logs/test.mp4",
            video_fps=0.0,
            width=1280,
            height=720,
            camera_zoom_out=1.5,
        )

        with self.assertRaisesRegex(ValueError, "video-fps"):
            validate_video_export_args(args)

    def test_rejects_non_positive_dimensions(self):
        args = argparse.Namespace(
            save_video="logs/test.mp4",
            video_fps=30.0,
            width=0,
            height=720,
            camera_zoom_out=1.5,
        )

        with self.assertRaisesRegex(ValueError, "width"):
            validate_video_export_args(args)

    def test_rejects_non_positive_camera_zoom_out(self):
        args = argparse.Namespace(
            save_video="logs/test.mp4",
            video_fps=30.0,
            width=1280,
            height=720,
            camera_zoom_out=0.0,
        )

        with self.assertRaisesRegex(ValueError, "camera-zoom-out"):
            validate_video_export_args(args)


class SimulationTimeSchedulerTests(unittest.TestCase):
    def test_renders_time_zero_frame_before_stepping(self):
        scheduler = SimulationTimeVideoScheduler(video_fps=2.0)

        self.assertTrue(scheduler.should_render_initial_frame())
        self.assertEqual(scheduler.next_frame_time, 0.0)

    def test_emits_frames_when_sim_time_crosses_frame_boundaries(self):
        scheduler = SimulationTimeVideoScheduler(video_fps=2.0)
        scheduler.mark_frame_rendered()

        emitted = []
        for sim_time in [0.1, 0.49, 0.5, 0.99, 1.0]:
            if scheduler.should_render_at(sim_time):
                emitted.append(round(scheduler.next_frame_time, 2))
                scheduler.mark_frame_rendered()

        self.assertEqual(emitted, [0.5, 1.0])


class ArgParserTests(unittest.TestCase):
    def test_parser_accepts_default_camera_export_options(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "examples/skydio_drone_demo.json",
                "--save-video",
                "logs/out.mp4",
                "--video-fps",
                "24",
                "--width",
                "640",
                "--height",
                "360",
                "--camera-zoom-out",
                "1.75",
            ]
        )

        self.assertEqual(args.save_video, "logs/out.mp4")
        self.assertEqual(args.video_fps, 24.0)
        self.assertEqual(args.width, 640)
        self.assertEqual(args.height, 360)
        self.assertEqual(args.camera_zoom_out, 1.75)


class VideoWriterFactoryTests(unittest.TestCase):
    def test_create_video_writer_raises_clear_error_when_dependency_missing(self):
        with self.assertRaisesRegex(RuntimeError, "imageio"):
            create_video_writer("logs/out.mp4", 24.0, force_missing_dependency=True)


class VideoDimensionNormalizationTests(unittest.TestCase):
    def test_keeps_dimensions_already_aligned_to_macro_block_size(self):
        self.assertEqual(normalize_video_dimensions(640, 368), (640, 368))

    def test_rounds_non_aligned_dimensions_up_to_next_multiple_of_16(self):
        self.assertEqual(normalize_video_dimensions(640, 360), (640, 368))


class VideoDimensionAdjustmentMessageTests(unittest.TestCase):
    def test_reports_adjusted_dimensions_when_size_changes(self):
        self.assertEqual(
            describe_video_dimension_adjustment(640, 360, 640, 368),
            "Adjusting video size from 640x360 to 640x368 for H.264 compatibility.",
        )

    def test_returns_none_when_no_adjustment_is_needed(self):
        self.assertIsNone(describe_video_dimension_adjustment(640, 368, 640, 368))


class ExportCameraConfigTests(unittest.TestCase):
    def test_build_export_camera_uses_fixed_world_view(self):
        camera = SimpleNamespace(distance=4.0, azimuth=120.0, elevation=-20.0)
        camera.lookat = [1.0, 2.0, 3.0]

        configured = configure_export_camera(camera, 1.5)

        self.assertIs(configured, camera)
        self.assertEqual(configured.lookat, [0.0, 0.0, 0.0])
        self.assertEqual(configured.azimuth, 90.0)
        self.assertEqual(configured.elevation, -45.0)
        self.assertEqual(configured.distance, 6.0)


class ViewerLaunchPolicyTests(unittest.TestCase):
    def test_save_video_disables_interactive_viewer(self):
        args = argparse.Namespace(headless=False, save_video="logs/out.mp4")
        self.assertFalse(should_launch_viewer(args))

    def test_normal_mode_launches_viewer(self):
        args = argparse.Namespace(headless=False, save_video=None)
        self.assertTrue(should_launch_viewer(args))


class RealtimePacingPolicyTests(unittest.TestCase):
    def test_export_mode_still_uses_realtime_pacing(self):
        self.assertTrue(should_sleep_for_realtime_pacing(export_enabled=True))

    def test_non_export_mode_uses_realtime_pacing(self):
        self.assertTrue(should_sleep_for_realtime_pacing(export_enabled=False))


class SceneBuilderOffscreenFramebufferTests(unittest.TestCase):
    def test_apply_export_offscreen_framebuffer_size_sets_global_dimensions(self):
        builder = SceneBuilder()
        scene_root = ET.fromstring("<mujoco><visual><global azimuth='120' elevation='-20'/></visual></mujoco>")

        builder._apply_export_offscreen_framebuffer_size(scene_root, 1920, 1088)

        global_elem = scene_root.find("visual/global")
        self.assertIsNotNone(global_elem)
        self.assertEqual(global_elem.get("offwidth"), "1920")
        self.assertEqual(global_elem.get("offheight"), "1088")


if __name__ == "__main__":
    unittest.main()
