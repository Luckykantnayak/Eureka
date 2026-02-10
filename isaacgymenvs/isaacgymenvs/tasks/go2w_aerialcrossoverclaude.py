
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict


class Go2wAerialcrossoverClaude(VecTask):

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
        asset_options.collapse_fixed_joints = False
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
        self.rew_buf[:], self.rew_dict = compute_reward(self.root_states, self.commands, self.dof_pos, self.default_dof_pos, self.dof_vel, self.actions, self.contact_forces, self.leg_dof_indices, self.wheel_dof_indices, self.gravity_vec)
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
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
#### Aerial crossover Task Success Function ####
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

    # --- Base kinematics ---
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])

    # --- Forward velocity tracking (vx > 0 only) ---
    vx_cmd = torch.clamp(commands[:, 0], min=0.0)
    vx_error = torch.square(vx_cmd - base_lin_vel[:, 0])
    rew_vx = torch.exp(-vx_error / 0.25) * rew_scales["lin_vel_xy"]

    # --- Contact logic ---
    # feet_indices assumed ordered: [FL, FR, RL, RR]
    foot_forces = torch.norm(contact_forces[:, feet_indices, :], dim=2)
    foot_contact = foot_forces > 1.0  # contact threshold

    # Diagonal pairs
    diag_1 = foot_contact[:, 0] & foot_contact[:, 3]  # FL + RR
    diag_2 = foot_contact[:, 1] & foot_contact[:, 2]  # FR + RL

    # Alternating diagonal lift (XOR)
    diagonal_wave = (diag_1 ^ diag_2).float()
    rew_diagonal_wave = diagonal_wave * rew_scales.get("diagonal_wave", 1.0)

    # --- Penalize full aerial or full stance ---
    all_contact = torch.all(foot_contact, dim=1)
    no_contact = torch.all(~foot_contact, dim=1)
    rew_contact_balance = (~all_contact & ~no_contact).float() * rew_scales.get("contact_balance", 1.0)

    # --- Torque regularization ---
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    # --- Total reward ---
    total_reward = (
        rew_vx
        + rew_diagonal_wave
        + rew_contact_balance
        + rew_torque
    )
    total_reward = torch.clip(total_reward, 0.0, None)

    # --- Reset conditions ---
    base_contact = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.0
    time_out = episode_lengths >= max_episode_length - 1
    reset = base_contact | time_out

    # --- Success metric (for logging / curriculum) ---
    consecutive_successes = (
        rew_vx.mean()
        + rew_diagonal_wave.mean()
    )

    return total_reward.detach(), reset, consecutive_successes


from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    commands: torch.Tensor,
    dof_pos: torch.Tensor,
    default_dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    actions: torch.Tensor,
    contact_forces: torch.Tensor,
    leg_dof_indices: torch.Tensor,
    wheel_dof_indices: torch.Tensor,
    gravity_vec: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Extract commanded velocities
    cmd_vx = commands[:, 0]
    cmd_vy = commands[:, 1]
    cmd_yaw = commands[:, 2]
    
    # Extract base states
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])
    base_height = root_states[:, 2]
    
    # Projected gravity for orientation tracking
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    
    # Get leg and wheel velocities
    leg_dof_vel = dof_vel[:, leg_dof_indices]
    wheel_dof_vel = dof_vel[:, wheel_dof_indices]
    leg_dof_pos = dof_pos[:, leg_dof_indices]
    
    # Determine mode based on commanded forward velocity
    forward_mode = cmd_vx > 0.1
    stationary_mode = ~forward_mode
    
    # ===== Forward Mode Rewards (vx > 0.1) =====
    
    # 1. Forward velocity tracking - use quadratic error with high temperature
    forward_vel_temp: float = 6.0
    forward_vel_error = torch.square(base_lin_vel[:, 0] - cmd_vx)
    forward_vel_reward = torch.exp(-forward_vel_error / forward_vel_temp)
    
    # 2. Lateral velocity tracking
    lateral_vel_temp: float = 0.5
    lateral_vel_error = torch.abs(base_lin_vel[:, 1] - cmd_vy)
    lateral_vel_reward = torch.exp(-lateral_vel_error / lateral_vel_temp)
    
    # 3. Yaw velocity tracking
    yaw_vel_temp: float = 0.5
    yaw_vel_error = torch.abs(base_ang_vel[:, 2] - cmd_yaw)
    yaw_vel_reward = torch.exp(-yaw_vel_error / yaw_vel_temp)
    
    # 4. Wheel velocity - adaptive to commanded velocity
    wheel_vel_temp: float = 3.0
    target_wheel_vel = torch.clamp(cmd_vx * 2.5, min=2.0, max=10.0)
    wheel_vel_magnitude = torch.abs(wheel_dof_vel).mean(dim=-1)
    wheel_spinning_error = torch.abs(wheel_vel_magnitude - target_wheel_vel)
    wheel_spinning_reward = torch.exp(-wheel_spinning_error / wheel_vel_temp)
    
    # 5. REVISED: Leg motion reward - simple total velocity magnitude
    leg_motion_temp: float = 2.5
    leg_vel_magnitude = torch.sum(torch.abs(leg_dof_vel), dim=-1)
    # Target around 15-20 rad/s total across all leg joints
    target_leg_vel: float = 18.0
    leg_motion_error = torch.abs(leg_vel_magnitude - target_leg_vel)
    leg_motion_reward = torch.exp(-leg_motion_error / leg_motion_temp)
    
    # 6. Diagonal leg coordination
    diagonal_coord_temp: float = 2.0
    if leg_dof_indices.shape[0] >= 12:
        fl_vel = leg_dof_vel[:, 0]
        fr_vel = leg_dof_vel[:, 3]
        rl_vel = leg_dof_vel[:, 6]
        rr_vel = leg_dof_vel[:, 9]
        
        diag1_sync = torch.abs(fl_vel - rr_vel)
        diag2_sync = torch.abs(fr_vel - rl_vel)
        diagonal_coord_error = (diag1_sync + diag2_sync) / 2.0
        diagonal_coord_reward = torch.exp(-diagonal_coord_error / diagonal_coord_temp)
    else:
        diagonal_coord_reward = torch.zeros_like(cmd_vx)
    
    # 7. Upright orientation - RELAXED temperature for more tolerance
    upright_temp: float = 0.8
    upright_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)
    upright_reward = torch.exp(-upright_error / upright_temp)
    
    # 8. Base height maintenance for forward motion
    height_temp: float = 0.25
    target_height: float = 0.28
    height_error = torch.abs(base_height - target_height)
    height_reward = torch.exp(-height_error / height_temp)
    
    # 9. Action rate penalty
    action_rate_temp: float = 50.0
    action_rate = torch.sum(torch.square(actions), dim=-1)
    action_smoothness = torch.exp(-action_rate / action_rate_temp)
    
    # 10. Wheel ground contact
    wheel_contact_temp: float = 30.0
    wheel_contact_forces = contact_forces[:, wheel_dof_indices, 2]  # z-component
    wheel_contact_magnitude = torch.sum(torch.abs(wheel_contact_forces), dim=-1)
    wheel_contact_reward = 1.0 - torch.exp(-wheel_contact_magnitude / wheel_contact_temp)
    
    # 11. Leg clearance reward - REDUCED temperature for better gradient
    leg_clearance_temp: float = 0.08
    leg_pos_deviation = torch.sum(torch.square(leg_dof_pos - default_dof_pos[:, leg_dof_indices]), dim=-1)
    leg_clearance_reward = 1.0 - torch.exp(-leg_pos_deviation / leg_clearance_temp)
    
    # ===== Stationary Mode Rewards (vx <= 0.1) =====
    
    # 1. Zero linear velocity
    stationary_lin_vel_temp: float = 0.3
    stationary_lin_vel_penalty = torch.sum(torch.square(base_lin_vel), dim=-1)
    stationary_lin_vel_reward = torch.exp(-stationary_lin_vel_penalty / stationary_lin_vel_temp)
    
    # 2. Zero angular velocity - slightly increased temperature
    stationary_ang_vel_temp: float = 2.0
    stationary_ang_vel_penalty = torch.sum(torch.square(base_ang_vel), dim=-1)
    stationary_ang_vel_reward = torch.exp(-stationary_ang_vel_penalty / stationary_ang_vel_temp)
    
    # 3. Seated posture - try higher temperature for easier learning
    seated_posture_temp: float = 5.0
    seated_pos_error = torch.sum(torch.square(leg_dof_pos - default_dof_pos[:, leg_dof_indices]), dim=-1)
    seated_posture_reward = torch.exp(-seated_pos_error / seated_posture_temp)
    
    # 4. Wheel stationary - SIGNIFICANTLY increased temperature
    wheel_stationary_temp: float = 120.0
    wheel_vel_penalty = torch.sum(torch.square(wheel_dof_vel), dim=-1)
    wheel_stationary_reward = torch.exp(-wheel_vel_penalty / wheel_stationary_temp)
    
    # 5. Leg stability - increased temperature
    leg_stability_temp: float = 35.0
    leg_vel_penalty = torch.sum(torch.square(leg_dof_vel), dim=-1)
    leg_stability_reward = torch.exp(-leg_vel_penalty / leg_stability_temp)
    
    # 6. Low base height for seated posture
    seated_height_temp: float = 0.18
    seated_target_height: float = 0.18
    seated_height_error = torch.abs(base_height - seated_target_height)
    seated_height_reward = torch.exp(-seated_height_error / seated_height_temp)
    
    # 7. Upright even in stationary mode
    stationary_upright_reward = upright_reward
    
    # ===== Combine Rewards Based on Mode =====
    
    # Forward mode: MAXIMUM priority on forward velocity
    forward_mode_reward = (
        15.0 * forward_vel_reward +           # INCREASED from 12.0 - absolute priority
        2.0 * lateral_vel_reward +
        2.0 * yaw_vel_reward +
        2.5 * wheel_spinning_reward +
        2.5 * leg_motion_reward +             # Increased slightly
        1.5 * diagonal_coord_reward +
        4.0 * upright_reward +                # Increased from 3.0 - critical for stability
        1.5 * height_reward +
        4.0 * wheel_contact_reward +
        1.5 * leg_clearance_reward +          # Increased from 1.0
        0.5 * action_smoothness
    )
    
    # Stationary mode: balanced priorities
    stationary_mode_reward = (
        6.0 * stationary_lin_vel_reward +
        5.0 * stationary_ang_vel_reward +
        5.0 * seated_posture_reward +         # Increased from 4.0
        7.0 * wheel_stationary_reward +       # Increased from 6.0
        4.0 * leg_stability_reward +          # Increased from 3.0
        3.0 * seated_height_reward +
        2.0 * stationary_upright_reward
    )
    
    # Select reward based on mode
    total_reward = torch.where(forward_mode, forward_mode_reward, stationary_mode_reward)
    
    # Create reward dictionary
    reward_components = {
        "forward_vel_reward": forward_vel_reward,
        "lateral_vel_reward": lateral_vel_reward,
        "yaw_vel_reward": yaw_vel_reward,
        "wheel_spinning_reward": wheel_spinning_reward,
        "leg_motion_reward": leg_motion_reward,
        "diagonal_coord_reward": diagonal_coord_reward,
        "upright_reward": upright_reward,
        "height_reward": height_reward,
        "action_smoothness": action_smoothness,
        "wheel_contact_reward": wheel_contact_reward,
        "leg_clearance_reward": leg_clearance_reward,
        "stationary_lin_vel_reward": stationary_lin_vel_reward,
        "stationary_ang_vel_reward": stationary_ang_vel_reward,
        "seated_posture_reward": seated_posture_reward,
        "wheel_stationary_reward": wheel_stationary_reward,
        "leg_stability_reward": leg_stability_reward,
        "seated_height_reward": seated_height_reward,
        "total_reward": total_reward
    }
    
    return total_reward, reward_components
