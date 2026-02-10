
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict


class Go2wSideflip(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 56
        self.cfg["env"]["numActions"] = 16

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.leg_dof_indices = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],device=self.device,dtype=torch.long)
        self.wheel_dof_indices = torch.tensor([3, 7, 11, 15],device=self.device,dtype=torch.long)
        
        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/go2w/urdf/go2w.urdf"
        

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        go2w_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(go2w_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(go2w_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(go2w_asset)
        self.dof_names = self.gym.get_asset_dof_names(go2w_asset)
        extremity_name = "calf" if asset_options.collapse_fixed_joints else "foot"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "thigh" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(go2w_asset)
        self.torque_limits = torch.tensor(
                                dof_props['effort'],
                                device=self.device
                            ).unsqueeze(0)  # shape [1, num_dof]
        
        for i in range(len(self.dof_names)):
            name = self.dof_names[i]

            if "wheel" in name:
                dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
                dof_props["stiffness"][i] = 0.0
                dof_props["damping"][i] = 0.0
            else:
                dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
                dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd



        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.go2w_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            go2w_handle = self.gym.create_actor(env_ptr, go2w_asset, start_pose, "go2w", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, go2w_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, go2w_handle)
            self.envs.append(env_ptr)
            self.go2w_handles.append(go2w_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.go2w_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.go2w_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.go2w_handles[0], "base")

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device)

        # -------------------------
        # Split actions
        # -------------------------
        leg_actions = self.actions[:, self.leg_dof_indices] 
        wheel_actions = self.actions[:, self.wheel_dof_indices] 


        # -------------------------
        # LEG POSITION CONTROL
        # -------------------------
        # Target = nominal + delta
        leg_targets = (
            self.default_dof_pos[:, self.leg_dof_indices]
            + leg_actions * self.action_scale
        )

        # Full DOF position target tensor
        dof_pos_targets = torch.zeros(
            (self.num_envs, 16),
            device=self.device
        )
        dof_pos_targets[:, self.leg_dof_indices] = leg_targets

        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(dof_pos_targets)
        )

        # -------------------------
        # WHEEL TORQUE CONTROL
        # -------------------------
        wheel_torques = wheel_actions * self.torque_limits[:, self.wheel_dof_indices]
        # Safety clamp
        wheel_torques = torch.clamp(
            wheel_torques,
            -self.torque_limits[:, self.wheel_dof_indices],
            self.torque_limits[:, self.wheel_dof_indices]
        )
        # Full DOF torque tensor
        dof_torques = torch.zeros(
            (self.num_envs, 16),
            device=self.device
        )
        dof_torques[:, self.wheel_dof_indices] = wheel_torques

        self.gym.set_dof_actuation_force_tensor(
            self.sim,
            gymtorch.unwrap_tensor(dof_torques)
        )

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.gt_rew_buf, self.reset_buf[:], self.consecutive_successes[:] = compute_success(
            self.root_states,
            self.commands,
            self.torques,
            self.contact_forces,
            self.feet_indices,
            self.consecutive_successes,
            self.progress_buf,
            self.rew_scales,
            self.base_index,
            self.max_episode_length,
        )
        self.extras['gt_reward'] = self.gt_rew_buf.mean()
        self.extras['consecutive_successes'] = self.consecutive_successes.mean() 

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_go2w_observations(  # tensors
                                                        self.root_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale,
                                                        self.leg_dof_indices,
                                                        self.wheel_dof_indices
        )

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

@torch.jit.script
def compute_go2w_observations(root_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale,
                                leg_dof_indices,
                                wheel_dof_indices
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, Tensor, Tensor) -> Tensor
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    leg_dof_pos_scaled = (dof_pos[:, leg_dof_indices] - default_dof_pos[:, leg_dof_indices]) * dof_pos_scale
    leg_dof_vel_scaled = dof_vel[:, leg_dof_indices] * dof_vel_scale
    
    wheel_dof_vel_scaled = dof_vel[:, wheel_dof_indices] * dof_vel_scale
    
    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)
    
    leg_actions = actions[:, leg_dof_indices]
    
    wheel_actions = actions[:, wheel_dof_indices]

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     leg_dof_pos_scaled,
                     leg_dof_vel_scaled,
                     wheel_dof_vel_scaled,
                     leg_actions,
                     wheel_actions
                     ), dim=-1)

    return obs

def quat_to_rpy(q):
    """
    Convert quaternion to roll, pitch, yaw.
    
    Args:
        q: Tensor of shape (N, 4) in (x, y, z, w) format
    
    Returns:
        roll, pitch, yaw: each of shape (N,)
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Roll (x-axis rotation)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr, cosr)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)  # numerical safety
    pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny, cosy)

    return roll, pitch, yaw


@torch.jit.script
#### Side Flip Task Success Function ####
def compute_success(
    root_states,
    commands,
    torques,
    contact_forces,
    feet_indices,
    consecutive_successes,
    episode_lengths,
    rew_scales,
    base_index,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int) -> Tuple[Tensor, Tensor, Tensor]
    
    batch_size = root_states.shape[0]
    device = root_states.device
    
    # ----------------------------
    # Base state extraction
    # ----------------------------
    base_pos = root_states[:, :3]
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])
    
    base_height = base_pos[:, 2]
    
    # Angular velocities in body frame
    roll_rate = base_ang_vel[:, 0]  # Side flip = roll rotation
    pitch_rate = base_ang_vel[:, 1]
    yaw_rate = base_ang_vel[:, 2]
    
    # Get roll, pitch, yaw angles
    roll, pitch, yaw = quat_to_rpy(base_quat)
    
    # Command extraction
    flip_cmd = commands[:, 0] > 0.5  # Side flip when commanded
    flip_direction = torch.sign(commands[:, 1])  # +1 = right, -1 = left, 0 = either
    
    # ----------------------------
    # Contact detection
    # ----------------------------
    contact_thresh = 1.0
    foot_forces = torch.norm(contact_forces[:, feet_indices, :], dim=2)
    foot_contact = (foot_forces > contact_thresh).float()
    any_foot_contact = torch.any(foot_contact > 0.5, dim=1)
    
    all_feet_contact = torch.all(foot_contact > 0.5, dim=1)
    airborne = ~any_foot_contact
    
    # Detect asymmetric contact for launch
    left_feet_contact = foot_contact[:, 0] + foot_contact[:, 2]  # FL + RL
    right_feet_contact = foot_contact[:, 1] + foot_contact[:, 3]  # FR + RR
    
    # ----------------------------
    # Side flip phases
    # ----------------------------
    
    # Phase 1: PRE-LAUNCH (setup on one side)
    # - Feet on ground
    # - Weight shifted to one side (asymmetric contact)
    # - Preparing for lateral launch
    weight_shifted = torch.abs(left_feet_contact - right_feet_contact) > 1.0
    stable_stance = (torch.abs(pitch_rate) < 0.5) & (torch.abs(yaw_rate) < 0.5)
    pre_launch = all_feet_contact & weight_shifted & stable_stance
    
    # Phase 2: LAUNCH (explosive lateral jump with roll initiation)
    # - Leaving ground or just airborne
    # - Upward velocity
    # - Starting to roll in commanded direction
    upward_vel = root_states[:, 9]  # World frame Z velocity
    
    # Check if rolling in correct direction (if specified)
    correct_roll_direction = torch.ones_like(roll_rate, dtype=torch.bool)
    if flip_direction.abs().max() > 0.1:  # Direction specified
        correct_roll_direction = (torch.sign(roll_rate) == flip_direction) | (flip_direction == 0)
    
    launching = (
        (upward_vel > 0.5) &
        airborne &
        (base_height > 0.28) &
        (torch.abs(roll_rate) > 1.0) &  # Starting roll rotation
        correct_roll_direction
    )
    
    # Phase 3: ROTATION (in-air spinning laterally)
    # - Fully airborne
    # - High roll rate (rotating sideways)
    # - Sufficient height
    # - Accumulated rotation
    high_airborne = airborne & (base_height > 0.4)
    fast_roll_rotation = torch.abs(roll_rate) > 3.0  # Fast lateral rotation
    
    # Track rotation progress (roll should go through ±π for 360° flip)
    rotation_progress = torch.abs(roll) / (2.0 * torch.pi)  # 0 to 1+ for full rotation
    
    good_rotation = high_airborne & fast_roll_rotation & (rotation_progress > 0.2)
    
    # Phase 4: COMPLETION (full rotation achieved)
    # - Completed ~360° roll rotation
    # - Slowing rotation for landing
    full_rotation = torch.abs(roll) > 5.5  # ~315° (allowing margin before 360°)
    rotation_slowing = torch.abs(roll_rate) < 5.0  # Starting to slow down
    completion_phase = airborne & full_rotation & rotation_slowing
    
    # Phase 5: LANDING (controlled touchdown)
    # - Feet touching ground
    # - Body near upright (completed rotation)
    # - Low velocities
    landing = any_foot_contact & (base_height < 0.5)
    
    # Check if landed with reasonable orientation (within 45° of upright)
    landed_upright = landing & (torch.abs(roll) < 0.8) & (torch.abs(pitch) < 0.8)
    
    # Stable after landing
    low_ang_vel = (torch.abs(roll_rate) < 1.0) & (torch.abs(pitch_rate) < 1.0)
    low_lin_vel = torch.norm(base_lin_vel, dim=1) < 0.5
    stable_landing = landed_upright & low_ang_vel & low_lin_vel
    
    # ----------------------------
    # Side flip quality metrics
    # ----------------------------
    
    # 1. Launch quality (explosive upward and lateral motion)
    launch_quality = torch.exp(-(upward_vel.clamp(max=3.0) - 2.0) ** 2 / 0.5)
    launch_quality = launch_quality * launching.float()
    
    # 2. Rotation speed (fast enough to complete flip)
    rotation_speed_score = (torch.abs(roll_rate) / 5.0).clamp(0, 1)
    rotation_speed_score = rotation_speed_score * good_rotation.float()
    
    # 3. Height maintenance (stay high enough during flip)
    height_score = (base_height.clamp(max=0.8) / 0.8)
    height_score = height_score * airborne.float()
    
    # 4. Rotation alignment (minimize pitch/yaw during roll)
    # Side flip should primarily rotate around roll axis
    alignment_score = torch.exp(-(torch.abs(pitch_rate) + torch.abs(yaw_rate)) / 2.0)
    alignment_score = alignment_score * airborne.float()
    
    # 5. Landing precision (land upright)
    landing_precision = torch.exp(-torch.abs(roll) / 0.5) * landed_upright.float()
    
    # 6. In-place constraint (minimize forward/backward drift)
    # Side flip should not travel forward
    forward_vel = torch.abs(base_lin_vel[:, 0])
    in_place_score = torch.exp(-forward_vel / 0.5)
    
    # 7. Lateral control (some lateral motion acceptable during side flip)
    # But should not drift too much
    lateral_vel = torch.abs(base_lin_vel[:, 1])
    lateral_control = torch.exp(-lateral_vel / 1.0)  # More lenient than forward
    
    # 8. Direction consistency (if direction specified, maintain it)
    direction_score = torch.ones(batch_size, dtype=torch.float, device=device)
    if flip_direction.abs().max() > 0.1:
        # Penalize if rolling opposite to commanded direction
        direction_match = (torch.sign(roll_rate) == flip_direction) | (roll_rate.abs() < 0.5)
        direction_score = direction_match.float() * airborne.float() + (~airborne).float()
    
    # ----------------------------
    # Side flip quality composite score
    # ----------------------------
    flip_quality = (
        pre_launch.float() * 0.08 +           # 8%: Proper setup
        launch_quality * 0.18 +                # 18%: Explosive launch
        rotation_speed_score * 0.25 +          # 25%: Fast rotation
        height_score * 0.15 +                  # 15%: Maintain height
        alignment_score * 0.12 +               # 12%: Clean rotation axis
        landing_precision * 0.12 +             # 12%: Upright landing
        in_place_score * 0.05 +                # 5%: Minimal forward drift
        lateral_control * 0.03 +               # 3%: Controlled lateral motion
        direction_score * 0.02                 # 2%: Correct direction
    )
    
    # ----------------------------
    # Rewards (command-conditioned)
    # ----------------------------
    
    # When flip commanded
    rew_flip = flip_quality * rew_scales.get("flip", 3.0) * flip_cmd.float()
    
    # Specific phase rewards for shaping
    rew_launch = launching.float() * rew_scales.get("launch", 0.5) * flip_cmd.float()
    rew_rotation = good_rotation.float() * rew_scales.get("rotation", 1.0) * flip_cmd.float()
    rew_completion = completion_phase.float() * rew_scales.get("completion", 2.0) * flip_cmd.float()
    rew_landing = stable_landing.float() * rew_scales.get("landing", 1.5) * flip_cmd.float()
    
    # Reward for weight shifting during pre-launch (helps with learning asymmetric launch)
    rew_weight_shift = weight_shifted.float() * all_feet_contact.float() * rew_scales.get("weight_shift", 0.2) * flip_cmd.float()
    
    # When idle (no flip commanded) - reward staying still
    idle_reward = (
        all_feet_contact.float() * 
        low_ang_vel.float() * 
        low_lin_vel.float() * 
        rew_scales.get("idle", 0.5) * 
        (~flip_cmd).float()
    )
    
    # Energy penalties
    rew_torque = -torch.sum(torch.abs(torques), dim=1) * rew_scales.get("torque", 0.0001)
    
    # Penalty for excessive forward/backward drift
    drift_penalty = -forward_vel * rew_scales.get("drift_penalty", 0.15)
    
    # Penalty for bad landing (high impact)
    impact_forces = torch.sum(foot_forces, dim=1)
    impact_penalty = -(impact_forces.clamp(max=500.0) / 500.0) * landing.float() * rew_scales.get("impact_penalty", 0.2)
    
    # Penalty for wrong rotation direction (if specified)
    wrong_direction_penalty = torch.zeros(batch_size, dtype=torch.float, device=device)
    if flip_direction.abs().max() > 0.1:
        wrong_direction = (torch.sign(roll_rate) != flip_direction) & (roll_rate.abs() > 1.0) & airborne
        wrong_direction_penalty = -wrong_direction.float() * rew_scales.get("wrong_direction", 0.5)
    
    total_reward = (
        rew_flip +
        rew_launch +
        rew_rotation +
        rew_completion +
        rew_landing +
        rew_weight_shift +
        idle_reward +
        rew_torque +
        drift_penalty +
        impact_penalty +
        wrong_direction_penalty
    )
    
    total_reward = torch.clamp(total_reward, min=-10.0, max=10.0)
    
    # ----------------------------
    # Reset conditions
    # ----------------------------
    
    # Body contact (belly/side flop)
    base_contact = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.0
    
    # Excessive tilt during non-airborne phases
    up_vec_local = torch.zeros(batch_size, 3, dtype=torch.float, device=device)
    up_vec_local[:, 2] = 1.0
    up_vec = quat_rotate(base_quat, up_vec_local)
    excessive_tilt = (up_vec[:, 2] < 0.2) & (~airborne)  # >78° while on ground
    
    # Failed landing (landed but badly oriented)
    bad_landing = landing & (torch.abs(roll) > 1.5) & (episode_lengths > 50)
    
    # Timeout
    time_out = episode_lengths >= max_episode_length - 1
    
    reset = base_contact | excessive_tilt | bad_landing | time_out
    
    # ----------------------------
    # Success metric
    # ----------------------------
    
    # Complete side flip success criteria:
    # 1. Launched successfully with roll rotation
    # 2. Completed full rotation (|roll| > 5.5 radians)
    # 3. Landed with feet on ground
    # 4. Final orientation upright (|roll| < 0.8, |pitch| < 0.8)
    # 5. Stable after landing
    
    flip_completed = (
        stable_landing &
        (episode_lengths > 30)  # Minimum time for flip
    )
    
    # For command-conditioned success
    flip_success = flip_completed & flip_cmd
    idle_success = (all_feet_contact & low_ang_vel & low_lin_vel) & (~flip_cmd)
    
    is_success = flip_success | idle_success
    
    # Track consecutive successful flips
    consecutive_successes = torch.where(
        is_success,
        consecutive_successes + 1,
        torch.zeros_like(consecutive_successes)
    )
    
    consecutive_successes = (consecutive_successes.float()).mean()

    return total_reward.detach(), reset, consecutive_successes
