import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_robots import (
    SimulationTimeVideoScheduler,
    build_arg_parser,
    create_video_writer,
    get_named_camera_id,
    should_launch_viewer,
    validate_video_export_args,
)


class VideoExportArgValidationTests(unittest.TestCase):
    def test_requires_camera_when_save_video_enabled(self):
        args = argparse.Namespace(
            save_video="logs/test.mp4",
            camera=None,
            video_fps=30.0,
            width=1280,
            height=720,
        )

        with self.assertRaisesRegex(ValueError, "camera"):
            validate_video_export_args(args)

    def test_rejects_non_positive_video_fps(self):
        args = argparse.Namespace(
            save_video="logs/test.mp4",
            camera="track",
            video_fps=0.0,
            width=1280,
            height=720,
        )

        with self.assertRaisesRegex(ValueError, "video-fps"):
            validate_video_export_args(args)

    def test_rejects_non_positive_dimensions(self):
        args = argparse.Namespace(
            save_video="logs/test.mp4",
            camera="track",
            video_fps=30.0,
            width=0,
            height=720,
        )

        with self.assertRaisesRegex(ValueError, "width"):
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


class NamedCameraLookupTests(unittest.TestCase):
    def test_returns_camera_id_for_existing_name(self):
        model = SimpleNamespace(
            ncam=2,
            cam=lambda index: SimpleNamespace(name=["front", "track"][index]),
        )

        self.assertEqual(get_named_camera_id(model, "track"), 1)

    def test_accepts_unique_suffix_match_for_prefixed_camera_names(self):
        model = SimpleNamespace(
            ncam=2,
            cam=lambda index: SimpleNamespace(name=["drone1_track", "arm_cam"][index]),
        )

        self.assertEqual(get_named_camera_id(model, "track"), 0)

    def test_lists_available_camera_names_for_missing_camera(self):
        model = SimpleNamespace(
            ncam=1,
            cam=lambda index: SimpleNamespace(name="track"),
        )

        with self.assertRaisesRegex(ValueError, "track"):
            get_named_camera_id(model, "missing")


class ArgParserTests(unittest.TestCase):
    def test_parser_accepts_video_export_options(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "examples/skydio_drone_demo.json",
                "--save-video",
                "logs/out.mp4",
                "--camera",
                "track",
                "--video-fps",
                "24",
                "--width",
                "640",
                "--height",
                "360",
            ]
        )

        self.assertEqual(args.save_video, "logs/out.mp4")
        self.assertEqual(args.camera, "track")
        self.assertEqual(args.video_fps, 24.0)
        self.assertEqual(args.width, 640)
        self.assertEqual(args.height, 360)


class VideoWriterFactoryTests(unittest.TestCase):
    def test_create_video_writer_raises_clear_error_when_dependency_missing(self):
        with self.assertRaisesRegex(RuntimeError, "imageio"):
            create_video_writer("logs/out.mp4", 24.0, force_missing_dependency=True)


class ViewerLaunchPolicyTests(unittest.TestCase):
    def test_save_video_disables_interactive_viewer(self):
        args = argparse.Namespace(headless=False, save_video="logs/out.mp4")
        self.assertFalse(should_launch_viewer(args))

    def test_normal_mode_launches_viewer(self):
        args = argparse.Namespace(headless=False, save_video=None)
        self.assertTrue(should_launch_viewer(args))


if __name__ == "__main__":
    unittest.main()
