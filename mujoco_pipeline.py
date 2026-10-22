import mujoco
import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_n_link_arm_xml(num_links: int) -> str:
    """Generates an N-link robot arm MJCF XML to dynamically scale state dimension D. Each link has a hinge joint and a capsule geometry.
      The base is fixed to the world."""
    body_xml = ""
    indent = "    "
    
    # build the kinematic chain of links
    for i in range(num_links):
        body_xml += f'{indent * (i+1)}<body name="link_{i}" pos="0 0 0.3">\n'
        body_xml += f'{indent * (i+1)}  <joint name="joint_{i}" type="hinge" axis="0 1 0"/>\n'
        body_xml += f'{indent * (i+1)}  <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" rgba="0.{i%9+1} 0.5 0.8 1"/>\n'
    
    # close the body tags in reverse order
    for i in range(num_links - 1, -1, -1):
        body_xml += f'{indent * (i+1)}</body>\n'

    xml = f"""
    <mujoco model="{num_links}_link_arm">
      <option timestep="0.01" gravity="0 0 -9.81"/>
      <worldbody>
        <light pos="0 -1 2" dir="0 1 -1"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="base" pos="0 0 0">
          <geom type="cylinder" size="0.1 0.05" rgba="0.2 0.2 0.2 1"/>
            {body_xml}
        </body>
      </worldbody>
    </mujoco>
    """
    print(f"Generated MJCF XML for {num_links}-link arm with state dimension D={num_links*2}.")
    return xml

class MuJoCoEnvironment:
    """A simple wrapper around MuJoCo to create an N-link robot arm environment with state dimension D=2*num_links (positions + velocities)"""
    def __init__(self, num_links: int = 3, img_width: int = 240, img_height: int = 240):
        xml_string = generate_n_link_arm_xml(num_links)
        self.model = mujoco.MjModel.from_xml_string(xml_string) # static properties of the simulation - body, joint, geom, gravity, actuator, etc.
        self.data = mujoco.MjData(self.model) # dynamic state of the simulation - positions, velocities, forces, etc.
        self.renderer = mujoco.Renderer(self.model, height=img_height, width=img_width) # creates an offscreen OpenGL context for rendering frames

        # number of generalized coordinates (qpos) and velocities (qvel) = number of joints = num_links
        self.nq = self.model.nq
        self.nv = self.model.nv

        # State dimension D = nq + nv = 2*num_links
        self.state_dim = self.nq + self.nv

    def get_state(self) -> np.ndarray:
        """Returns the current state vector s = [qpos, qvel] as a D-dimensional numpy array."""
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()]) # [qpos_1, qpos_2, ..., qvel_1, qvel_2, ...]

    def set_state(self, state: np.ndarray) -> None:
        """Sets qpos and qvel explicitly and updates kinematics."""
        self.data.qpos[:] = state[:self.nq]
        self.data.qvel[:] = state[self.nq:]
        # update Cartesian positions, transformations, and sensors without advancing time
        mujoco.mj_forward(self.model, self.data)

    def render_rgb(self, state: np.ndarray = None) -> np.ndarray:
        """Given a state vector s, updates simulation and returns RGB array (H, W, 3)."""
        if state is not None:
            self.set_state(state)
        else:
            mujoco.mj_forward(self.model, self.data)
            
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def step_simulation(self) -> np.ndarray:
        """Steps physics forward by 1 time step."""
        mujoco.mj_step(self.model, self.data)
        return self.get_state()

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"PyTorch active device: {device}")

    # example usage: create a 5-link arm
    # state dimension D = 10: 5 positions + 5 velocities
    env = MuJoCoEnvironment(num_links=5, img_width=320, img_height=320)
    print(f"State dimension D: {env.state_dim} (nq={env.nq}, nv={env.nv})")

    # apply initial velocity kick to start swinging motion
    env.data.qvel[:] = np.array([2.0, -1.5, 1.0, -0.5, 1.2])

    # run physics to step 45 and record state and RGB
    target_step= 45
    for step in range(target_step):
        env.step_simulation()

    # get the state vector at step 45
    state_45= env.get_state()
    rgb_45_original= env.render_rgb()

    print(f"=== Step {target_step} Captured ===")
    print(f"qpos (Joint Angles in radians) : {state_45[:env.nq]}")
    print(f"qvel (Angular Velocities rad/s): {state_45[env.nq:]}")
    print(f"Full State Vector s_45 shape   : {state_45.shape}\n")

    # teleport sim to a completely different random state
    scrambled_state = np.random.uniform(-2.0, 2.0, size=env.state_dim)
    env.set_state(scrambled_state)
    rgb_scrambled = env.render_rgb()


    # reconstruct step 45 directly from the state45 vector
    rgb_45_reconstructed = env.render_rgb(state_45)

    # verify byte-for-byte pixel identity
    max_pixel_diff = np.max(np.abs(rgb_45_original.astype(int) - rgb_45_reconstructed.astype(int)))
    print(f"Max pixel difference between original Step 45 and restored Step 45: {max_pixel_diff}")
    assert max_pixel_diff == 0, "Error: Restored RGB does not match original state!"

    # manually modify the joint angle in the state vector
    custom_state = state_45.copy()
    custom_state[0] += np.pi / 2  # rotate Joint 0 by +90 degrees (+1.57 rad)
    rgb_custom = env.render_rgb(custom_state)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(rgb_45_original)
    axes[0].set_title(f"1. Original Step {target_step}")
    axes[0].axis("off")

    axes[1].imshow(rgb_scrambled)
    axes[1].set_title("2. Scrambled State")
    axes[1].axis("off")

    axes[2].imshow(rgb_45_reconstructed)
    axes[2].set_title(f"3. Restored from s_{target_step}")
    axes[2].axis("off")

    axes[3].imshow(rgb_custom)
    axes[3].set_title("4. Modified s (Joint 0 +90°)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig("state_rendering_demo.png", dpi=300)
    print("Saved comparison image to 'state_rendering_demo.png'.")
