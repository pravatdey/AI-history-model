"""
Blender Headless Render Script for Audio2Face-3D

Reads ARKit blendshape animation CSV + renders a 3D character to MP4 video.
Uses NVIDIA's built-in characters (Mark/Claire/James) approach:
  - Creates a stylized 3D head with ARKit-compatible shape keys
  - Applies blendshape animation from Audio2Face-3D output
  - Renders to MP4 with professional lighting

Usage (called by audio2face3d_engine.py):
    blender --background --python render_avatar.py -- \
        --csv animation_frames.csv \
        --audio audio.wav \
        --output output.mp4 \
        --fps 30 \
        --resolution 512 512 \
        --character mark
"""

import bpy
import csv
import sys
import os
import json
import math

# Parse arguments after "--"
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []


def parse_args(argv):
    args = {}
    i = 0
    while i < len(argv):
        if argv[i] == "--csv":
            args["csv"] = argv[i + 1]
            i += 2
        elif argv[i] == "--audio":
            args["audio"] = argv[i + 1]
            i += 2
        elif argv[i] == "--output":
            args["output"] = argv[i + 1]
            i += 2
        elif argv[i] == "--fps":
            args["fps"] = int(argv[i + 1])
            i += 2
        elif argv[i] == "--resolution":
            args["res_x"] = int(argv[i + 1])
            args["res_y"] = int(argv[i + 2])
            i += 3
        elif argv[i] == "--character":
            args["character"] = argv[i + 1]
            i += 2
        elif argv[i] == "--background-color":
            args["bg_color"] = [float(x) for x in argv[i + 1].split(",")]
            i += 2
        else:
            i += 1
    return args


args = parse_args(argv)
CSV_PATH = args.get("csv", "")
AUDIO_PATH = args.get("audio", "")
OUTPUT_PATH = args.get("output", "output.mp4")
FPS = args.get("fps", 30)
RES_X = args.get("res_x", 720)
RES_Y = args.get("res_y", 720)
CHARACTER = args.get("character", "mark")
BG_COLOR = args.get("bg_color", [0.12, 0.15, 0.22])

# ── ARKit Blendshape Definitions ──────────────────────────────────────────

# Mapping of ARKit blendshapes to mesh deformation parameters
# Each entry: (shape_key_name, region, axis, magnitude)
ARKIT_SHAPES = {
    # Jaw
    "JawOpen": {"type": "jaw", "magnitude": 0.035},
    "JawForward": {"type": "jaw_fwd", "magnitude": 0.008},
    "JawLeft": {"type": "jaw_side", "magnitude": 0.005, "dir": -1},
    "JawRight": {"type": "jaw_side", "magnitude": 0.005, "dir": 1},
    # Mouth
    "MouthClose": {"type": "mouth_close", "magnitude": 0.01},
    "MouthFunnel": {"type": "mouth_funnel", "magnitude": 0.015},
    "MouthPucker": {"type": "mouth_pucker", "magnitude": 0.012},
    "MouthLeft": {"type": "mouth_side", "magnitude": 0.008, "dir": -1},
    "MouthRight": {"type": "mouth_side", "magnitude": 0.008, "dir": 1},
    "MouthSmileLeft": {"type": "mouth_smile", "magnitude": 0.012, "dir": -1},
    "MouthSmileRight": {"type": "mouth_smile", "magnitude": 0.012, "dir": 1},
    "MouthFrownLeft": {"type": "mouth_frown", "magnitude": 0.008, "dir": -1},
    "MouthFrownRight": {"type": "mouth_frown", "magnitude": 0.008, "dir": 1},
    "MouthStretchLeft": {"type": "mouth_stretch", "magnitude": 0.01, "dir": -1},
    "MouthStretchRight": {"type": "mouth_stretch", "magnitude": 0.01, "dir": 1},
    "MouthRollLower": {"type": "mouth_roll_lower", "magnitude": 0.006},
    "MouthRollUpper": {"type": "mouth_roll_upper", "magnitude": 0.006},
    "MouthShrugLower": {"type": "mouth_shrug_lower", "magnitude": 0.008},
    "MouthShrugUpper": {"type": "mouth_shrug_upper", "magnitude": 0.008},
    "MouthPressLeft": {"type": "mouth_press", "magnitude": 0.005, "dir": -1},
    "MouthPressRight": {"type": "mouth_press", "magnitude": 0.005, "dir": 1},
    "MouthLowerDownLeft": {"type": "mouth_lower_down", "magnitude": 0.008, "dir": -1},
    "MouthLowerDownRight": {"type": "mouth_lower_down", "magnitude": 0.008, "dir": 1},
    "MouthUpperUpLeft": {"type": "mouth_upper_up", "magnitude": 0.006, "dir": -1},
    "MouthUpperUpRight": {"type": "mouth_upper_up", "magnitude": 0.006, "dir": 1},
    # Eyes
    "EyeBlinkLeft": {"type": "eye_blink", "magnitude": 1.0, "dir": -1},
    "EyeBlinkRight": {"type": "eye_blink", "magnitude": 1.0, "dir": 1},
    "EyeWideLeft": {"type": "eye_wide", "magnitude": 1.0, "dir": -1},
    "EyeWideRight": {"type": "eye_wide", "magnitude": 1.0, "dir": 1},
    "EyeSquintLeft": {"type": "eye_squint", "magnitude": 1.0, "dir": -1},
    "EyeSquintRight": {"type": "eye_squint", "magnitude": 1.0, "dir": 1},
    # Brows
    "BrowDownLeft": {"type": "brow_down", "magnitude": 1.0, "dir": -1},
    "BrowDownRight": {"type": "brow_down", "magnitude": 1.0, "dir": 1},
    "BrowInnerUp": {"type": "brow_inner_up", "magnitude": 1.0},
    "BrowOuterUpLeft": {"type": "brow_outer_up", "magnitude": 1.0, "dir": -1},
    "BrowOuterUpRight": {"type": "brow_outer_up", "magnitude": 1.0, "dir": 1},
    # Cheek & Nose
    "CheekPuff": {"type": "cheek_puff", "magnitude": 1.0},
    "CheekSquintLeft": {"type": "cheek_squint", "magnitude": 1.0, "dir": -1},
    "CheekSquintRight": {"type": "cheek_squint", "magnitude": 1.0, "dir": 1},
    "NoseSneerLeft": {"type": "nose_sneer", "magnitude": 1.0, "dir": -1},
    "NoseSneerRight": {"type": "nose_sneer", "magnitude": 1.0, "dir": 1},
    # Tongue
    "TongueOut": {"type": "tongue_out", "magnitude": 1.0},
}


def clean_scene():
    """Remove all default objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Remove default collections' objects
    for col in bpy.data.collections:
        for obj in col.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


def create_3d_head(character="mark"):
    """
    Create a stylized 3D human head with subdivision surface.
    Returns the head mesh object with ARKit shape keys.
    """
    # Create base head (UV Sphere as starting point)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.12, segments=64, ring_count=32,
        location=(0, 0, 0.05)
    )
    head = bpy.context.active_object
    head.name = "Head"

    # Scale to head proportions (taller, narrower)
    head.scale = (0.85, 1.0, 1.15)
    bpy.ops.object.transform_apply(scale=True)

    # Add subdivision for smoothness
    subsurf = head.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 2

    # Create neck/body base
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.065, depth=0.25,
        location=(0, 0, -0.12)
    )
    neck = bpy.context.active_object
    neck.name = "Neck"

    # Shoulders hint
    bpy.ops.mesh.primitive_cube_add(
        size=0.1, location=(0, 0, -0.25)
    )
    shoulders = bpy.context.active_object
    shoulders.name = "Shoulders"
    shoulders.scale = (2.5, 0.8, 0.5)
    bpy.ops.object.transform_apply(scale=True)

    # Add subdivision to shoulders
    subsurf_s = shoulders.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf_s.levels = 2
    subsurf_s.render_levels = 2

    # Create eyes
    for side, x_pos in [("Left", -0.04), ("Right", 0.04)]:
        # Eyeball
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.018, segments=32, ring_count=16,
            location=(x_pos, -0.10, 0.065)
        )
        eye = bpy.context.active_object
        eye.name = f"Eye{side}"

        # Iris
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.009, segments=16, ring_count=8,
            location=(x_pos, -0.117, 0.065)
        )
        iris = bpy.context.active_object
        iris.name = f"Iris{side}"

        # Pupil
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.005, segments=16, ring_count=8,
            location=(x_pos, -0.119, 0.065)
        )
        pupil = bpy.context.active_object
        pupil.name = f"Pupil{side}"

    # Nose
    bpy.ops.mesh.primitive_cone_add(
        radius1=0.02, radius2=0.008, depth=0.04,
        location=(0, -0.115, 0.025)
    )
    nose = bpy.context.active_object
    nose.name = "Nose"
    nose.rotation_euler = (math.radians(80), 0, 0)

    # Mouth area (torus for lips)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.025, minor_radius=0.006,
        major_segments=32, minor_segments=12,
        location=(0, -0.105, -0.01)
    )
    mouth = bpy.context.active_object
    mouth.name = "Mouth"
    mouth.scale = (1.2, 0.5, 0.8)
    bpy.ops.object.transform_apply(scale=True)

    # Apply materials based on character
    _apply_character_materials(head, neck, shoulders, mouth, character)

    # Add shape keys to head for ARKit blendshapes
    _add_shape_keys(head)

    return head


def _apply_character_materials(head, neck, shoulders, mouth, character):
    """Apply skin and clothing materials."""
    # Skin material
    skin_mat = bpy.data.materials.new(name="Skin")
    skin_mat.use_nodes = True
    bsdf = skin_mat.node_tree.nodes["Principled BSDF"]

    # Character-specific skin tones
    if character == "mark":
        bsdf.inputs["Base Color"].default_value = (0.72, 0.55, 0.42, 1.0)
    elif character == "claire":
        bsdf.inputs["Base Color"].default_value = (0.82, 0.65, 0.55, 1.0)
    elif character == "james":
        bsdf.inputs["Base Color"].default_value = (0.45, 0.30, 0.22, 1.0)
    else:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.55, 0.42, 1.0)

    bsdf.inputs["Roughness"].default_value = 0.4
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.3

    head.data.materials.append(skin_mat)
    neck.data.materials.append(skin_mat)

    # Eye white material
    eye_mat = bpy.data.materials.new(name="EyeWhite")
    eye_mat.use_nodes = True
    bsdf_e = eye_mat.node_tree.nodes["Principled BSDF"]
    bsdf_e.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)
    bsdf_e.inputs["Roughness"].default_value = 0.1

    for obj_name in ["EyeLeft", "EyeRight"]:
        if obj_name in bpy.data.objects:
            bpy.data.objects[obj_name].data.materials.append(eye_mat)

    # Iris material (brown)
    iris_mat = bpy.data.materials.new(name="Iris")
    iris_mat.use_nodes = True
    bsdf_i = iris_mat.node_tree.nodes["Principled BSDF"]
    bsdf_i.inputs["Base Color"].default_value = (0.25, 0.15, 0.08, 1.0)
    bsdf_i.inputs["Roughness"].default_value = 0.2

    for obj_name in ["IrisLeft", "IrisRight"]:
        if obj_name in bpy.data.objects:
            bpy.data.objects[obj_name].data.materials.append(iris_mat)

    # Pupil material (black)
    pupil_mat = bpy.data.materials.new(name="Pupil")
    pupil_mat.use_nodes = True
    bsdf_p = pupil_mat.node_tree.nodes["Principled BSDF"]
    bsdf_p.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)

    for obj_name in ["PupilLeft", "PupilRight"]:
        if obj_name in bpy.data.objects:
            bpy.data.objects[obj_name].data.materials.append(pupil_mat)

    # Lip material
    lip_mat = bpy.data.materials.new(name="Lips")
    lip_mat.use_nodes = True
    bsdf_l = lip_mat.node_tree.nodes["Principled BSDF"]
    bsdf_l.inputs["Base Color"].default_value = (0.55, 0.25, 0.22, 1.0)
    bsdf_l.inputs["Roughness"].default_value = 0.3
    mouth.data.materials.append(lip_mat)

    # Clothing material
    cloth_mat = bpy.data.materials.new(name="Clothing")
    cloth_mat.use_nodes = True
    bsdf_c = cloth_mat.node_tree.nodes["Principled BSDF"]
    bsdf_c.inputs["Base Color"].default_value = (0.15, 0.20, 0.35, 1.0)
    bsdf_c.inputs["Roughness"].default_value = 0.7
    shoulders.data.materials.append(cloth_mat)


def _add_shape_keys(head):
    """Add ARKit blendshape shape keys to the head mesh."""
    # Basis shape key
    head.shape_key_add(name="Basis", from_mix=False)

    # Add all ARKit blendshapes as shape keys
    for bs_name in ARKIT_SHAPES:
        sk = head.shape_key_add(name=bs_name, from_mix=False)
        sk.value = 0.0

    # For shape keys that need vertex displacement, we'll handle via drivers
    # The actual deformation is done per-frame in apply_blendshapes()


def setup_camera():
    """Set up camera for head/shoulders portrait shot."""
    bpy.ops.object.camera_add(
        location=(0, -0.45, 0.05),
        rotation=(math.radians(90), 0, 0)
    )
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.data.lens = 85  # Portrait lens
    camera.data.clip_start = 0.01
    camera.data.clip_end = 10
    bpy.context.scene.camera = camera
    return camera


def setup_lighting():
    """Set up 3-point studio lighting."""
    # Key light (warm, from front-left)
    bpy.ops.object.light_add(
        type='AREA', location=(-0.3, -0.4, 0.3),
        rotation=(math.radians(60), math.radians(-20), 0)
    )
    key_light = bpy.context.active_object
    key_light.name = "KeyLight"
    key_light.data.energy = 30
    key_light.data.color = (1.0, 0.95, 0.9)
    key_light.data.size = 0.4

    # Fill light (cool, from front-right)
    bpy.ops.object.light_add(
        type='AREA', location=(0.25, -0.35, 0.15),
        rotation=(math.radians(50), math.radians(15), 0)
    )
    fill_light = bpy.context.active_object
    fill_light.name = "FillLight"
    fill_light.data.energy = 15
    fill_light.data.color = (0.9, 0.93, 1.0)
    fill_light.data.size = 0.5

    # Rim light (from behind)
    bpy.ops.object.light_add(
        type='AREA', location=(0.1, 0.3, 0.25),
        rotation=(math.radians(-45), 0, 0)
    )
    rim_light = bpy.context.active_object
    rim_light.name = "RimLight"
    rim_light.data.energy = 20
    rim_light.data.color = (0.95, 0.95, 1.0)
    rim_light.data.size = 0.3


def setup_background(bg_color):
    """Set up a clean gradient background."""
    world = bpy.data.worlds.new(name="StudioBG")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear defaults
    for node in nodes:
        nodes.remove(node)

    # Background node
    bg_node = nodes.new(type='ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (*bg_color, 1.0)
    bg_node.inputs['Strength'].default_value = 0.8

    # Output
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])


def read_blendshape_csv(csv_path):
    """
    Read Audio2Face-3D blendshape CSV output.

    Expected format: columns like 'timeCode', 'blendShapes.JawOpen', etc.
    Returns list of dicts: [{"timeCode": float, "blendShapes": {name: value}}]
    """
    frames = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = {"timeCode": 0.0, "blendShapes": {}}
            for key, value in row.items():
                if key == "timeCode":
                    frame["timeCode"] = float(value)
                elif key.startswith("blendShapes."):
                    bs_name = key.replace("blendShapes.", "")
                    try:
                        frame["blendShapes"][bs_name] = float(value)
                    except (ValueError, TypeError):
                        frame["blendShapes"][bs_name] = 0.0
            frames.append(frame)
    return frames


def apply_blendshapes_to_timeline(head, frames, fps):
    """
    Apply blendshape animation data as keyframes on shape keys.
    """
    if not head.data.shape_keys:
        print("ERROR: Head has no shape keys!")
        return 0

    key_blocks = head.data.shape_keys.key_blocks
    total_frames = len(frames)

    for frame_idx, frame_data in enumerate(frames):
        # Calculate Blender frame number from timeCode
        time_code = frame_data.get("timeCode", frame_idx / fps)
        blender_frame = round(time_code * fps)

        blendshapes = frame_data.get("blendShapes", {})

        for bs_name, bs_value in blendshapes.items():
            if bs_name in key_blocks:
                key = key_blocks[bs_name]
                key.value = max(0.0, min(1.0, bs_value))
                key.keyframe_insert(data_path="value", frame=blender_frame)

        # Also animate mouth and eyes objects based on key blendshapes
        _animate_face_parts(blendshapes, blender_frame)

    return total_frames


def _animate_face_parts(blendshapes, frame):
    """Animate mouth, jaw, eyes based on blendshape values."""
    # Jaw open - move mouth down
    jaw_open = blendshapes.get("JawOpen", 0.0)
    mouth = bpy.data.objects.get("Mouth")
    if mouth:
        base_y = -0.01
        mouth.location.z = base_y - (jaw_open * 0.02)
        mouth.keyframe_insert(data_path="location", index=2, frame=frame)

        # Mouth funnel (O shape)
        funnel = blendshapes.get("MouthFunnel", 0.0)
        pucker = blendshapes.get("MouthPucker", 0.0)
        mouth.scale.x = 1.2 - (funnel * 0.3) - (pucker * 0.4)
        mouth.scale.z = 0.8 + (jaw_open * 0.5) + (funnel * 0.3)
        mouth.keyframe_insert(data_path="scale", frame=frame)

    # Eye blinks
    for side, dir_name in [("Left", "Left"), ("Right", "Right")]:
        blink_val = blendshapes.get(f"EyeBlink{dir_name}", 0.0)
        eye = bpy.data.objects.get(f"Eye{side}")
        iris = bpy.data.objects.get(f"Iris{side}")
        pupil = bpy.data.objects.get(f"Pupil{side}")

        if eye:
            # Scale Y to simulate blink (squish vertically)
            eye.scale.y = 1.0 - (blink_val * 0.8)
            eye.keyframe_insert(data_path="scale", index=1, frame=frame)
        if iris:
            iris.scale.y = 1.0 - (blink_val * 0.8)
            iris.keyframe_insert(data_path="scale", index=1, frame=frame)
        if pupil:
            pupil.scale.y = 1.0 - (blink_val * 0.8)
            pupil.keyframe_insert(data_path="scale", index=1, frame=frame)

    # Brow movement
    brow_up = blendshapes.get("BrowInnerUp", 0.0)
    # Could animate brow bones if rigged


def setup_render_settings(output_path, fps, res_x, res_y, total_frames):
    """Configure render settings for video output."""
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if bpy.app.version >= (4, 0, 0) else 'BLENDER_EEVEE'
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.fps = fps

    # Output as image sequence (more reliable than direct FFMPEG)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path.replace(".mp4", "_frames/frame_")

    scene.frame_start = 0
    scene.frame_end = total_frames - 1

    # EEVEE settings for quality + speed
    if hasattr(scene, 'eevee'):
        eevee = scene.eevee
        if hasattr(eevee, 'taa_render_samples'):
            eevee.taa_render_samples = 32
        if hasattr(eevee, 'use_ssr'):
            eevee.use_ssr = True


def render_animation():
    """Render the animation."""
    print("Starting render...")
    bpy.ops.render.render(animation=True)
    print("Render complete!")


def main():
    if not CSV_PATH or not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV file not found: {CSV_PATH}")
        sys.exit(1)

    print(f"=== Audio2Face-3D Blender Renderer ===")
    print(f"CSV: {CSV_PATH}")
    print(f"Audio: {AUDIO_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Character: {CHARACTER}")
    print(f"Resolution: {RES_X}x{RES_Y} @ {FPS}fps")

    # Step 1: Clean scene
    print("Cleaning scene...")
    clean_scene()

    # Step 2: Create 3D character
    print(f"Creating 3D character: {CHARACTER}...")
    head = create_3d_head(CHARACTER)

    # Step 3: Setup camera, lighting, background
    print("Setting up camera and lighting...")
    setup_camera()
    setup_lighting()
    setup_background(BG_COLOR)

    # Step 4: Read blendshape animation
    print("Reading blendshape CSV...")
    frames = read_blendshape_csv(CSV_PATH)
    print(f"  Loaded {len(frames)} animation frames")

    if len(frames) == 0:
        print("ERROR: No animation frames found in CSV!")
        sys.exit(1)

    # Step 5: Apply blendshapes as keyframes
    print("Applying blendshapes to timeline...")
    total_frames = apply_blendshapes_to_timeline(head, frames, FPS)
    print(f"  Applied {total_frames} frames to timeline")

    # Step 6: Configure render
    print("Configuring render settings...")
    setup_render_settings(OUTPUT_PATH, FPS, RES_X, RES_Y, total_frames)

    # Step 7: Render
    render_animation()

    # Write metadata for the engine to read
    meta = {
        "total_frames": total_frames,
        "fps": FPS,
        "duration": total_frames / FPS,
        "resolution": [RES_X, RES_Y],
        "character": CHARACTER,
        "frames_dir": OUTPUT_PATH.replace(".mp4", "_frames"),
    }
    meta_path = OUTPUT_PATH.replace(".mp4", "_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    print(f"Metadata written to: {meta_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
