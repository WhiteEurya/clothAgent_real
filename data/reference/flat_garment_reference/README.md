# Flat garment reference

This is the canonical Camera A reference for the fully spread shirt.

- `camera_A_flat_reference.png`: raw RGB reference image.
- `camera_A_flat_reference_anchors.png`: centerline-first semantic anchor overlay.
- `reference_anchors.json`: anchor pixels and provenance.

The reference is a semantic/layout prior, not a pixel-to-pixel template. During
a folded observation, the pipeline re-detects the current collar/hem centerline
and current visible anchors. It may use the reference to identify anchor type
and expected garment-side semantics, but it must not copy reference pixels into
the folded image. Occluded or geometrically inconsistent anchors are marked
unavailable rather than forced into a grasp target.
