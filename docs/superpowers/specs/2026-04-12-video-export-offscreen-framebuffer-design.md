# Video Export Offscreen Framebuffer Design

## Goal

Make high-resolution video export actually work by automatically resizing MuJoCo's offscreen framebuffer to match the final export resolution whenever `--save-video` is enabled.

## Scope

Included:

- detect the final export width and height after dimension normalization
- set `<visual><global offwidth="..." offheight="..."/>` in the generated temporary scene
- apply this only for video export runs
- keep ordinary non-export runs unchanged

Excluded:

- manual XML edits by users
- changes to video encoding format
- changes to camera behavior
- changes to simulation pacing

## Existing Context

The current export path can normalize requested video size, but the generated MuJoCo scene still inherits the base scene's small default offscreen framebuffer.

When the requested export resolution exceeds that framebuffer, MuJoCo rejects renderer creation with errors such as:

- `Image width 1920 > framebuffer width 640`

That means high-resolution export currently cannot work even though the video pipeline otherwise supports it.

## Recommended Approach

When `--save-video` is active:

1. compute the final export dimensions after the existing 16-pixel alignment logic
2. pass those dimensions into `SceneBuilder`
3. ensure the generated scene contains:
   - `<visual>`
   - `<global>`
4. set:
   - `offwidth=<final_width>`
   - `offheight=<final_height>`

This ensures MuJoCo allocates an offscreen framebuffer large enough for the requested export size.

## Builder Integration

Recommended shape:

- extend `SceneBuilder.build(...)` with optional export framebuffer dimensions
- when dimensions are present:
  - create/find `visual`
  - create/find `global`
  - set `offwidth` and `offheight`
- when dimensions are absent:
  - preserve existing behavior

## Verification Strategy

Implementation is complete only after these checks pass:

1. add a focused test for the scene-builder framebuffer override helper
2. run the export-specific test file
3. run a real `1920x1080` export command
4. confirm renderer creation no longer fails because of framebuffer size

## Implementation Notes

- the framebuffer dimensions should use the normalized export size, not the raw requested size
- this fix should be local to scene generation and export setup
- do not change the no-export path
