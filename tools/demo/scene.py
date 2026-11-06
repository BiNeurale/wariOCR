"""Blender scene: the demo sheet photographed like a phone shot (angle, curl, shadow).

    blender --background --python tools/demo/scene.py -- PAGE.png OUT.png VARIANT [PERCENT]

VARIANT 1: wooden table, warm side light, soft shadow of the phone over a corner.
VARIANT 2: grey desk, cooler light, stronger tilt and a lifted page edge.
Everything is procedural: no third-party textures.
"""
from __future__ import annotations

import math
import os
import sys

import bpy

PAGE_W, PAGE_H = 0.210, 0.297  # metres


def args() -> tuple[str, str, int, int]:
    a = sys.argv[sys.argv.index("--") + 1:]
    return os.path.abspath(a[0]), os.path.abspath(a[1]), int(a[2]), int(a[3]) if len(a) > 3 else 100


def reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def node_mat(name: str):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree.nodes, m.node_tree.links


def table(variant: int) -> None:
    bpy.ops.mesh.primitive_plane_add(size=1.6, location=(0, 0, 0))
    obj = bpy.context.object
    m, n, lk = node_mat("table")
    bsdf = n["Principled BSDF"]
    coord = n.new("ShaderNodeTexCoord")
    ramp = n.new("ShaderNodeValToRGB")
    if variant == 1:
        noise = n.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 18.0
        noise.inputs["Detail"].default_value = 6.0
        wave = n.new("ShaderNodeTexWave")
        wave.inputs["Scale"].default_value = 4.0
        wave.inputs["Distortion"].default_value = 3.0
        wave.inputs["Detail"].default_value = 4.0
        lk.new(coord.outputs["Object"], noise.inputs["Vector"])
        lk.new(coord.outputs["Object"], wave.inputs["Vector"])
        mix = n.new("ShaderNodeMath")
        mix.operation = "MULTIPLY_ADD"
        mix.inputs[1].default_value = 0.6
        lk.new(noise.outputs["Fac"], mix.inputs[0])
        lk.new(wave.outputs["Fac"], mix.inputs[2])
        lk.new(mix.outputs[0], ramp.inputs["Fac"])
        ramp.color_ramp.elements[0].color = (0.11, 0.05, 0.02, 1)
        ramp.color_ramp.elements[1].color = (0.28, 0.14, 0.06, 1)
        bsdf.inputs["Roughness"].default_value = 0.45
    else:
        noise = n.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 160.0
        noise.inputs["Detail"].default_value = 8.0
        lk.new(coord.outputs["Object"], noise.inputs["Vector"])
        lk.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        ramp.color_ramp.elements[0].color = (0.09, 0.10, 0.11, 1)
        ramp.color_ramp.elements[1].color = (0.20, 0.21, 0.23, 1)
        bsdf.inputs["Roughness"].default_value = 0.8
    lk.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(m)


def sheet(page_png: str, variant: int):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=90, y_subdivisions=120, size=1)
    obj = bpy.context.object
    for v in obj.data.vertices:
        u, w = v.co.x + 0.5, v.co.y + 0.5  # 0..1 across width, height
        x, y = (u - 0.5) * PAGE_W, (w - 0.5) * PAGE_H
        if variant == 1:
            # a page that was folded in three: soft ridges plus a lifted top-right corner
            z = 0.0012 * (1 - math.cos(2 * math.pi * w * 1.5)) / 2
            z += 0.010 * max(0.0, u + w - 1.55) ** 2 / 0.45 ** 2
        else:
            # bound-book style curl along the right edge
            z = 0.014 * max(0.0, u - 0.62) ** 2 / 0.38 ** 2 + 0.0015 * math.sin(math.pi * w)
        v.co = (x, y, z + 0.0004)
    bpy.ops.object.shade_smooth()
    m, n, lk = node_mat("paper")
    bsdf = n["Principled BSDF"]
    tex = n.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(page_png)
    tex.interpolation = "Cubic"
    uv = n.new("ShaderNodeTexCoord")
    lk.new(uv.outputs["UV"], tex.inputs["Vector"])
    lk.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Subsurface Weight"].default_value = 0.08
    obj.data.materials.append(m)
    obj.rotation_euler[2] = math.radians(-8 if variant == 1 else 14)
    return obj


def lights(variant: int) -> None:
    world = bpy.data.worlds.new("w")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    tint = (0.9, 0.85, 0.78, 1) if variant == 1 else (0.75, 0.82, 0.9, 1)
    bg.inputs["Color"].default_value = tint
    bg.inputs["Strength"].default_value = 0.25
    if variant == 1:
        loc, energy, color, size = (-0.55, 0.35, 0.75), 30, (1.0, 0.82, 0.62), 0.15
    else:
        loc, energy, color, size = (0.5, -0.1, 0.9), 40, (0.92, 0.96, 1.0), 0.3
    bpy.ops.object.light_add(type="AREA", location=loc)
    lamp = bpy.context.object
    lamp.data.energy = energy
    lamp.data.color = color
    lamp.data.size = size
    direction = -lamp.location
    lamp.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    # the photographer: phone and hand between the lamp and the sheet
    occ = (-0.13, 0.04, 0.23) if variant == 1 else (0.17, 0.01, 0.36)
    bpy.ops.mesh.primitive_cube_add(size=1, location=occ)
    o = bpy.context.object
    o.scale = (0.09, 0.17, 0.01)
    o.rotation_euler = (0.2, -0.4, 0.5)
    o.visible_camera = False


def camera(variant: int) -> None:
    if variant == 1:
        loc, look, roll = (0.03, -0.23, 0.40), (0.0, 0.01, 0.0), 3.0
    else:
        loc, look, roll = (-0.14, -0.16, 0.30), (0.0, 0.0, 0.0), -5.0
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    from mathutils import Vector
    d = Vector(look) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.rotate_axis("Z", math.radians(roll))
    cam.data.sensor_fit = "VERTICAL"
    cam.data.sensor_height = 36
    cam.data.lens = 34
    bpy.context.scene.camera = cam


def render(out: str, percent: int) -> None:
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
    s.cycles.samples = 24
    s.cycles.use_denoising = True
    s.render.resolution_x, s.render.resolution_y = 3024, 4032
    s.render.resolution_percentage = percent
    s.view_settings.view_transform = "AgX"
    s.view_settings.look = "AgX - Base Contrast"
    s.render.image_settings.file_format = "PNG"
    s.render.filepath = out
    bpy.ops.render.render(write_still=True)


def main() -> None:
    page, out, variant, percent = args()
    reset()
    table(variant)
    sheet(page, variant)
    lights(variant)
    camera(variant)
    render(out, percent)


main()
