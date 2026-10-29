"""V1.5a renderer.

The drawing primitives and colors remain exactly those of V1.3; the dynamic
environment exposes its current zone positions through ``env.zones``.
"""

from mini_vla.data.grasp_place_render_v13 import render_v13


def render_dynamic_v15a(env, image_size=64):
    return render_v13(env, image_size=image_size)
