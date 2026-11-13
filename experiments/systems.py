"""The simulated worlds the model learns to predict.

Each world is written as MuJoCo XML (a text description of bodies and joints)
and wrapped in `SimEnv`: set gravity and friction, reset to a random pose,
step forward, read the state.

A state is [qpos, qvel]: one position and one speed per joint. For a hinge the
position is its angle in radians; for a slide joint it is a distance in metres.
So D (numbers per state) = 2 x number of joints.

Every world is `count` copies that never touch each other, placed in a row, so
adding copies makes D bigger without making any single copy harder.

    pendulum       Hanging pendulums. Each is a chain of `links` hinged rods that
                   swings below its mount. Nothing ever touches anything.
    ground         The original setup from the first version of this project:
                   rods hinged at the floor, starting upright (angle 0 = up), so
                   they fall over and hit the floor. The floor is solid.
    spring_chain   Masses hanging on springs, moving up and down. Smooth and linear; no floor.
    cartpole       A cart sliding on a rail with a freely swinging pole (no motor).
    balls          Balls bouncing in a box (move in x and z). Sudden bounces.

All worlds advance 0.01 s per step.
"""
from __future__ import annotations

import numpy as np
import mujoco

SYSTEMS = ("pendulum", "ground", "spring_chain", "cartpole", "balls")

LINK_LENGTH = 0.3
ROD_RADIUS = 0.03
COLORS = (
    "0.95 0.25 0.22 1", "1 0.58 0.12 1", "0.95 0.82 0.18 1", "0.18 0.75 0.35 1",
    "0.08 0.72 0.9 1", "0.25 0.42 0.95 1", "0.62 0.3 0.88 1", "0.95 0.3 0.65 1",
)


def _row(count: int, spacing: float) -> list[float]:
    """x positions of `count` units placed in one row, centred on 0."""
    return [(i - (count - 1) / 2) * spacing for i in range(count)]


def _world(name: str, timestep: float, bodies: str, floor_size: float, solid_floor: bool = False,
           floor: bool = True) -> str:
    """`floor=False` leaves the floor out entirely (spring chains hang below it at strong gravity)."""
    contact = 'contype="1" conaffinity="1"' if solid_floor else 'contype="0" conaffinity="0"'
    # The solid floor is drawn see-through, so a predicted rod that sinks into it stays visible.
    floor_rgba = "1 1 1 0.4" if solid_floor else "1 1 1 1"
    return f"""
    <mujoco model="{name}">
      <option timestep="{timestep}" gravity="0 0 -9.81"/>
      <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.8 0.8 0.8" specular="0.15 0.15 0.15"/></visual>
      <asset>
        <texture name="sky" type="skybox" builtin="gradient" rgb1="0.8 0.88 0.96" rgb2="0.25 0.38 0.52" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.88 0.9" rgb2="0.78 0.8 0.83" width="300" height="300"/>
        <material name="floor" texture="grid" texrepeat="8 8" rgba="{floor_rgba}"/>
      </asset>
      <worldbody>
        <light pos="0 -1 4" dir="0 1 -1"/>
        {f'<geom name="floor" type="plane" size="{floor_size} {floor_size} 0.1" material="floor" {contact}/>' if floor else ""}
        {bodies}
      </worldbody>
    </mujoco>
    """


def _hanging_grid(count: int, links: int, stacked: bool, link_length: float = LINK_LENGTH) -> tuple[int, int, float]:
    """(columns, rows, row spacing) for hanging pendulums.

    Simulation always uses one row. For videos (`stacked`), more than 8 units are stacked in
    rows so they stay readable. Nothing collides in the hanging world, so the layout does not
    change the motion; it only shifts floating-point rounding, which is why it is kept out of
    the simulation (cached clips stay bit-identical).
    """
    chain = link_length * links
    row_spacing = 2 * chain + 0.15
    if count <= 8 or not stacked:
        return count, 1, row_spacing
    width = 2 * chain + 0.05
    shapes = [(c, -(-count // c)) for c in range(1, count + 1)]
    cols, rows = min(shapes, key=lambda cr: abs(np.log(cr[0] * width / (cr[1] * row_spacing) / 1.6)))
    return cols, rows, row_spacing


def _pendulum_xml(count: int, links: int, ground: bool, stacked: bool = False,
                  lengths=None, masses=None) -> str:
    """Hanging (rods point down from a raised mount) or ground (rods point up from the floor).

    `lengths` (metres per rod) and `masses` (factor of the normal mass) give every unit its own
    rods; all links of one chain share them. Left out, every rod is LINK_LENGTH long with the
    normal mass, and the XML is exactly what it was before these options existed.
    """
    link_len = [LINK_LENGTH] * count if lengths is None else [float(v) for v in lengths]
    chain = max(link_len) * links
    sign = 1 if ground else -1                       # +1: links point up, -1: links point down
    # Ground hinges sit 5 cm above the floor. (In the very first version they sat exactly on
    # it, so the rounded rod end was always 3 cm inside the floor and in permanent contact.)
    height = 0.05 if ground else chain + 0.2
    # Rods only collide with the floor (contype 2 / conaffinity 1), never with each other.
    rod_contact = 'contype="2" conaffinity="1" solref="0.005 1"' if ground else 'contype="0" conaffinity="0"'
    parts = []
    if ground:
        places = [(x, height) for x in _row(count, spacing=2 * chain + 0.05)]
    else:
        cols, rows, row_spacing = _hanging_grid(count, links, stacked, max(link_len))
        xs = _row(cols, spacing=2 * chain + 0.05)
        places = [(xs[i % cols], height + (rows - 1 - i // cols) * row_spacing) for i in range(count)]
    for i, (x, z) in enumerate(places):
        xml = (f'<body name="mount_{i}" pos="{x} 0 {z}">'
               f'<geom type="cylinder" size="0.05 0.03" euler="90 0 0" rgba="0.25 0.27 0.3 1" contype="0" conaffinity="0"/>')
        length = LINK_LENGTH if lengths is None else link_len[i]
        density = "" if masses is None else f' density="{1000.0 * float(masses[i]):.6g}"'   # MuJoCo default: 1000
        for k in range(links):
            pos = "0 0 0" if k == 0 else f"0 0 {sign * length}"
            xml += (f'<body name="p{i}_l{k}" pos="{pos}">'
                    f'<joint name="p{i}_j{k}" type="hinge" axis="0 1 0" armature="0.01"/>'
                    f'<geom name="rod_{i}_{k}" type="capsule" fromto="0 0 0 0 0 {sign * length}" '
                    f'size="{ROD_RADIUS}"{density} rgba="{COLORS[i % len(COLORS)]}" {rod_contact}/>')
        xml += "</body>" * (links + 1)
        parts.append(xml)
    name = f"{count}_{'ground' if ground else 'hanging'}_pendulums_{links}_links"
    # 0.002 s physics steps (5 per 0.01 s state step) keep long chains and floor hits stable.
    return _world(name, 0.002, "".join(parts), max(2.0, count * (chain + 0.3)), solid_floor=ground)


def _spring_chain_xml(count: int, links: int) -> str:
    rest = 0.25
    parts = []
    for i, x in enumerate(_row(count, spacing=0.5)):
        xml = (f'<body name="anchor_{i}" pos="{x} 0 {rest * links + 0.4}">'
               f'<geom type="box" size="0.06 0.06 0.01" rgba="0.25 0.27 0.3 1" contype="0" conaffinity="0"/>')
        for k in range(links):
            xml += (f'<body name="c{i}_m{k}" pos="0 0 {-rest}">'
                    f'<joint name="c{i}_j{k}" type="slide" axis="0 0 1" stiffness="40"/>'
                    f'<geom type="sphere" size="0.05" mass="0.2" rgba="{COLORS[i % len(COLORS)]}" contype="0" conaffinity="0"/>')
        xml += "</body>" * (links + 1)
        parts.append(xml)
    return _world(f"{count}_spring_chains", 0.01, "".join(parts), 2.0, floor=False)


def _cartpole_xml(count: int) -> str:
    parts = []
    for i, x in enumerate(_row(count, spacing=1.2)):
        parts.append(
            f'<body name="rail_{i}" pos="{x} 0 0.6">'
            f'<geom type="capsule" fromto="-0.5 0 0 0.5 0 0" size="0.01" rgba="0.4 0.4 0.4 1" contype="0" conaffinity="0"/>'
            f'<body name="cart_{i}"><joint name="cart_{i}_x" type="slide" axis="1 0 0"/>'
            f'<geom type="box" size="0.08 0.05 0.04" mass="1.0" rgba="0.25 0.27 0.3 1" contype="0" conaffinity="0"/>'
            f'<body name="pole_{i}"><joint name="pole_{i}_hinge" type="hinge" axis="0 1 0"/>'
            f'<geom type="capsule" fromto="0 0 0 0 0 {-LINK_LENGTH}" size="0.025" mass="0.3" '
            f'rgba="{COLORS[i % len(COLORS)]}" contype="0" conaffinity="0"/>'
            f'</body></body></body>')
    return _world(f"{count}_cartpoles", 0.01, "".join(parts), 3.0)


def _balls_xml(count: int) -> str:
    # Each ball has its own lane (different y) and only hits the floor and the two walls.
    # solref < 0 means (-stiffness, -damping): fairly elastic bounces.
    lane, half_width = 0.25, 1.0
    depth = lane * count / 2 + 0.2
    parts = [
        f'<geom type="box" pos="0 0 -0.05" size="{half_width} {depth} 0.05" rgba="0.7 0.72 0.75 1" contype="1" conaffinity="1"/>'
        f'<geom type="box" pos="{-half_width - 0.05} 0 1" size="0.05 {depth} 1" rgba="0.7 0.72 0.75 0.3" contype="1" conaffinity="1"/>'
        f'<geom type="box" pos="{half_width + 0.05} 0 1" size="0.05 {depth} 1" rgba="0.7 0.72 0.75 0.3" contype="1" conaffinity="1"/>'
    ]
    for i in range(count):
        parts.append(
            f'<body name="ball_{i}" pos="0 {(i - (count - 1) / 2) * lane} 0">'
            f'<joint name="ball_{i}_x" type="slide" axis="1 0 0"/><joint name="ball_{i}_z" type="slide" axis="0 0 1"/>'
            f'<geom type="sphere" size="0.08" mass="0.1" rgba="{COLORS[i % len(COLORS)]}" contype="2" conaffinity="1" '
            f'solref="-60000 -6" solimp="0.95 0.99 0.001" friction="0 0 0" condim="1"/></body>')
    return _world(f"{count}_balls", 0.002, "".join(parts), 3.0)


class SimEnv:
    """One compiled MuJoCo world from SYSTEMS, with a common interface."""

    def __init__(self, system: str, count: int, links: int = 1, img_size: tuple[int, int] | None = None,
                 lengths=None, masses=None):
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system '{system}'. Choose from {SYSTEMS}.")
        if count <= 0 or links <= 0:
            raise ValueError("count and links must be positive.")
        if (lengths is not None or masses is not None) and system not in ("pendulum", "ground"):
            raise ValueError("lengths / masses only exist for pendulum worlds.")
        self.system, self.count, self.links = system, count, links
        self.link_length = LINK_LENGTH if lengths is None else float(np.max(lengths))
        xml = {
            "pendulum": lambda: _pendulum_xml(count, links, ground=False, stacked=img_size is not None,
                                              lengths=lengths, masses=masses),
            "ground": lambda: _pendulum_xml(count, links, ground=True, lengths=lengths, masses=masses),
            "spring_chain": lambda: _spring_chain_xml(count, links),
            "cartpole": lambda: _cartpole_xml(count),
            "balls": lambda: _balls_xml(count),
        }[system]()
        self.model = mujoco.MjModel.from_xml_string(xml)
        # Physics steps per state step, so that every world advances 0.01 s per step.
        self.substeps = round(0.01 / self.model.opt.timestep)
        if img_size:
            self.model.vis.global_.offwidth, self.model.vis.global_.offheight = img_size
        self.data = mujoco.MjData(self.model)
        self.nq, self.nv = self.model.nq, self.model.nv
        self.state_dim = self.nq + self.nv
        self.img_size = img_size
        self.renderer = None
        mujoco.mj_forward(self.model, self.data)

    def set_physics(self, gravity: float = 9.81, damping: float = 0.0) -> None:
        """Gravity (m/s^2, pulling down) and joint damping (friction that brakes moving joints)."""
        if not np.isfinite(gravity) or gravity <= 0:
            raise ValueError("gravity must be finite and positive.")
        self.model.opt.gravity[:] = [0.0, 0.0, -float(gravity)]
        self.model.dof_damping[:] = float(damping)

    def get_state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float64)

    def set_state(self, state: np.ndarray) -> None:
        self.data.qpos[:] = state[: self.nq]
        self.data.qvel[:] = state[self.nq:]
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> np.ndarray:
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def unstable(self) -> bool:
        """True if MuJoCo hit NaN/huge accelerations since the last reset (it then silently resets)."""
        return self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        """Random starting state, chosen per world."""
        mujoco.mj_resetData(self.model, self.data)
        qpos, qvel = np.zeros(self.nq), np.zeros(self.nv)
        if self.system == "ground":            # as in the original setup: upright, random push
            qvel = rng.uniform(-2.0, 2.0, size=self.nv)
        elif self.system == "pendulum":
            qpos = rng.uniform(-np.pi / 2, np.pi / 2, size=self.nq)
            qvel = rng.uniform(-2.0, 2.0, size=self.nv)
        elif self.system == "spring_chain":
            qpos = rng.uniform(-0.08, 0.08, size=self.nq)
            qvel = rng.uniform(-0.5, 0.5, size=self.nv)
        elif self.system == "cartpole":        # joints alternate: cart x, pole angle
            qpos[0::2] = rng.uniform(-0.2, 0.2, size=self.count)
            qpos[1::2] = rng.uniform(-np.pi, np.pi, size=self.count)
            qvel[0::2] = rng.uniform(-0.5, 0.5, size=self.count)
            qvel[1::2] = rng.uniform(-2.0, 2.0, size=self.count)
        else:                                  # balls: joints alternate x, z
            qpos[0::2] = rng.uniform(-0.8, 0.8, size=self.count)
            qpos[1::2] = rng.uniform(0.3, 1.5, size=self.count)
            qvel = rng.uniform(-2.0, 2.0, size=self.nv)
        self.set_state(np.concatenate([qpos, qvel]))
        return self.get_state()

    # ------------------------------------------------------------------ floor check (ground world)
    def lowest_point(self, states: np.ndarray) -> np.ndarray:
        """Height of the lowest point of any rod, for each state (in metres; < 0 = inside the floor)."""
        rods = [g for g in range(self.model.ngeom) if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        half = self.model.geom_size[rods, 1]
        radius = self.model.geom_size[rods, 0]
        out = np.empty(len(states))
        for i, state in enumerate(states):
            self.data.qpos[:] = state[: self.nq]
            mujoco.mj_kinematics(self.model, self.data)
            centre_z = self.data.geom_xpos[rods, 2]
            axis_z = np.abs(self.data.geom_xmat[rods, 8])     # how vertical each rod is
            out[i] = np.min(centre_z - axis_z * half - radius)
        return out

    # ------------------------------------------------------------------ rendering
    def _view_box(self) -> tuple[float, float, float, float]:
        """(half width, z low, z high, elevation) of the region the camera should show."""
        chain = self.link_length * self.links
        row = lambda spacing, size: (self.count - 1) * spacing / 2 + size
        if self.system == "pendulum":     # mount at chain + 0.2, chains can swing up to 2 * chain + 0.2
            cols, rows, row_spacing = _hanging_grid(self.count, self.links, self.img_size is not None, self.link_length)
            half = (cols - 1) * (2 * chain + 0.05) / 2 + chain
            return half, 0.0, 2 * chain + 0.25 + (rows - 1) * row_spacing, -6
        if self.system == "ground":       # rods lie on the floor or stand up to `chain`
            return row(2 * chain + 0.05, chain), -0.25, chain + 0.1, -12
        if self.system == "spring_chain":
            return row(0.5, 0.25), 0.0, 0.25 * self.links + 0.5, -6
        if self.system == "cartpole":
            return row(1.2, 0.6), 0.2, 1.0, -6
        return 1.7, 0.0, 2.2, -12         # balls: box of half width 1 seen from the front, plus margin

    def render(self, state: np.ndarray) -> np.ndarray:
        if not self.img_size:
            raise ValueError("Create the environment with img_size=(width, height) to render.")
        if self.renderer is None:
            width, height = self.img_size
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            half_width, z_low, z_high, elevation = self._view_box()
            # Distance at which the box fits a 45-degree vertical field of view (with 10% margin).
            fit = 1.1 / (2 * np.tan(np.radians(22.5)))
            distance = fit * max(2 * half_width * height / width, z_high - z_low)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self.camera)
            self.camera.lookat[:] = [0.0, 0.0, (z_low + z_high) / 2]
            self.camera.distance, self.camera.azimuth, self.camera.elevation = distance, 90, elevation
        self.set_state(state)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()
