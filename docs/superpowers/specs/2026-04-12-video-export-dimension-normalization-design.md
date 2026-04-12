# Video Export Dimension Normalization Design

## Goal

Remove the MP4 encoder warning caused by non-macroblock-aligned export dimensions, while preserving the current offscreen video export workflow.

This change also cleans up the temporary git worktrees created during implementation now that the feature has already been merged into `main`.

## Scope

Included:

- automatic normalization of export width and height to the next multiple of `16`
- a short runtime message when the requested size is adjusted
- tests for the normalization helper
- cleanup of the temporary worktrees and feature branches used during implementation

Excluded:

- changing the public CLI option names
- preserving arbitrary source dimensions by forcing `macro_block_size=1`
- changing the encoded video codec
- changes to unrelated simulation behavior

## Existing Context

The current export path writes MP4 through `imageio` / `ffmpeg`.

When the user requests dimensions such as `640x360`, the encoder warns that the image is not divisible by `macro_block_size=16` and silently resizes it to `640x368`.

That behavior works, but it is noisy and leaves the actual encoded size implicit rather than explicit in the runtime behavior.

There are also two implementation worktrees still present:

- `.worktrees/feature-offscreen-video-export`
- `.worktrees/integrate-offscreen-video-export`

## Recommended Approach

Normalize dimensions in application code before creating the renderer and writer.

Rules:

- if `width` and `height` are already multiples of `16`, keep them unchanged
- otherwise round each non-conforming dimension upward to the next multiple of `16`
- print a concise message such as:
  - `Adjusting video size from 640x360 to 640x368 for H.264 compatibility.`

This keeps the behavior explicit and avoids the downstream encoder warning.

## Helper Design

Add a small helper in `run_robots.py`, for example:

- `normalize_video_dimensions(width: int, height: int, block_size: int = 16) -> tuple[int, int]`

Behavior:

- return the original dimensions when no adjustment is required
- return the adjusted dimensions when rounding is necessary
- keep the logic deterministic and easy to unit test

Use the normalized dimensions for:

- `mujoco.Renderer(...)`
- `imageio.get_writer(...)`

## Logging Behavior

The runtime should only print an adjustment message when the normalized dimensions differ from the user-requested dimensions.

No warning or error should be emitted for this case, because the adjustment is expected behavior.

## Verification Strategy

Implementation is complete only after these checks pass:

1. add a focused unit test that shows `640x360` becomes `640x368`
2. add a focused unit test that dimensions already aligned to `16` are unchanged
3. run the export-specific test file
4. run a real export command with `640x360`
5. confirm the previous encoder warning no longer appears

## Cleanup Plan

After the code change is merged and verified on `main`, remove the temporary branches and worktrees used during implementation:

- delete branch `feature/offscreen-video-export`
- delete branch `integrate/offscreen-video-export`
- remove worktree `.worktrees/feature-offscreen-video-export`
- remove worktree `.worktrees/integrate-offscreen-video-export`

Cleanup must not touch the main working tree at `D:\Study\robots\roboweaver`.
