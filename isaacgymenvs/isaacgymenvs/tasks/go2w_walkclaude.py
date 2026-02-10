
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict


class Go2wWalkClaude(VecTask):

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
        self.rew_buf[:], self.rew_dict = compute_reward(self.root_states, self.commands, self.dof_pos, self.default_dof_pos, self.dof_vel, self.actions, self.contact_forces, self.leg_dof_indices, self.wheel_dof_indices, self.gravity_vec)
        self.extras['gpt_reward'] = self.rew_buf.mean()
        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
        self.gt_rew_buf, self.reset_buf[:], self.consecutive_successes[:] = compute_success(
            self.root_states,
            self.commands,
            self.torques,
            self.contact_forces,
            self.feet_indices,
            self.dof_vel[:, self.wheel_dof_indices],
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
#### Walking Task Success Function ####
def compute_success(
    root_states,
    commands,
    torques,
    contact_forces,
    feet_indices,
    wheel_dof_vel,
    consecutive_successes,
    episode_lengths,
    rew_scales,
    base_index,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int) -> Tuple[Tensor, Tensor, Tensor]

    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    vx_cmd = commands[:, 0]

    # -------------------------
    # Contact logic
    # -------------------------
    feet_contact = torch.norm(
        contact_forces[:, feet_indices, :], dim=2
    )
    foot_in_contact = feet_contact > 1.0

    num_feet_contact = torch.sum(foot_in_contact, dim=1)

    # At least one foot in air → stepping
    stepping = num_feet_contact <= (len(feet_indices) - 1)

    # -------------------------
    # Forward walking
    # -------------------------
    forward_speed = base_lin_vel[:, 0]
    moving_forward = forward_speed > 0.3

    # -------------------------
    # Wheel suppression
    # -------------------------
    wheel_speed = torch.mean(
        torch.abs(wheel_dof_vel), dim=1
    )
    wheels_quiet = wheel_speed < 1.0

    # -------------------------
    # Base stability
    # -------------------------
    upright = torch.norm(torch.abs(base_ang_vel[:, :2]), p=2, dim=1) < 1.0

    # -------------------------
    # Command-conditioned success
    # -------------------------
    walk_cmd = vx_cmd > 0.0
    idle_cmd = vx_cmd <= 0.0

    legged_walk_success = (
        moving_forward &
        stepping &
        wheels_quiet &
        upright
    )

    idle_success = (
        (torch.norm(base_lin_vel[:, :2], p=2, dim=1) < 0.2) &
        (torch.norm(base_ang_vel, p=2, dim=1) < 0.5) &
        (num_feet_contact >= len(feet_indices) - 1)
    )

    success = torch.where(
        walk_cmd,
        legged_walk_success,
        idle_success
    )

    # -------------------------
    # Reset logic
    # -------------------------
    base_contact = torch.norm(contact_forces[:, base_index, :], p=2, dim=1) > 1.0

    time_out = episode_lengths >= max_episode_length - 1
    reset = base_contact | time_out

    consecutive_successes = success.float()
    consecutive_successes = consecutive_successes.mean()

    # -------------------------
    # Rewards
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]
    total_reward = rew_torque
    total_reward = torch.clip(total_reward, 0., None)


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
    
    # Extract state information
    base_quat = root_states[:, 3:7]
    base_lin_vel = root_states[:, 7:10]
    base_ang_vel = root_states[:, 10:13]
    
    # Transform velocities to body frame
    base_lin_vel_body = quat_rotate_inverse(base_quat, base_lin_vel)
    
    # Extract commanded velocities
    cmd_vx = commands[:, 0]
    
    # Compute whether robot should be walking
    is_walking = torch.abs(cmd_vx) > 0.1
    
    # === Reward 1: Forward velocity tracking ===
    velocity_tracking_error = torch.square(base_lin_vel_body[:, 0] - cmd_vx)
    velocity_temp = 0.25
    velocity_reward = torch.exp(-velocity_tracking_error / velocity_temp)
    
    # === Reward 2: Minimize lateral velocity ===
    lateral_vel_error = torch.square(base_lin_vel_body[:, 1])
    lateral_temp = 0.15
    lateral_reward = torch.exp(-lateral_vel_error / lateral_temp)
    
    # === Reward 3: Minimize vertical velocity ===
    vertical_vel_error = torch.square(base_lin_vel_body[:, 2])
    vertical_temp = 0.15
    vertical_reward = torch.exp(-vertical_vel_error / vertical_temp)
    
    # === Reward 4: STRICT wheel velocity penalty ===
    wheel_dof_vel = dof_vel[:, wheel_dof_indices]
    wheel_vel_squared = torch.sum(torch.square(wheel_dof_vel), dim=-1)
    wheel_vel_temp = 0.5  # Much stricter (was 2.0)
    wheel_passive_reward = torch.exp(-wheel_vel_squared / wheel_vel_temp)
    
    # === Reward 5: STRICT wheel torque penalty ===
    wheel_actions = actions[:, wheel_dof_indices]
    wheel_torque_squared = torch.sum(torch.square(wheel_actions), dim=-1)
    wheel_torque_temp = 0.3  # Much stricter (was 1.0)
    wheel_torque_reward = torch.exp(-wheel_torque_squared / wheel_torque_temp)
    
    # === Reward 6: Encourage leg joint velocities when walking ===
    leg_dof_vel = dof_vel[:, leg_dof_indices]
    leg_vel_magnitude = torch.sum(torch.abs(leg_dof_vel), dim=-1)
    leg_motion_temp = 3.0
    # Reward higher leg velocities when walking
    leg_motion_reward = torch.where(
        is_walking,
        torch.tanh(leg_vel_magnitude / leg_motion_temp),  # Encourage motion
        torch.exp(-leg_vel_magnitude / leg_motion_temp)   # Penalize when idle
    )
    
    # === Reward 7: Foot clearance - reward lifting feet ===
    num_feet = 4
    foot_contact_forces = contact_forces[:, :num_feet, :]
    foot_contact_magnitude = torch.norm(foot_contact_forces, dim=-1)
    is_contact = foot_contact_magnitude > 1.0
    num_feet_in_air = torch.sum(~is_contact, dim=-1).float()
    
    # Reward having 1-2 feet in air when walking (alternating gait)
    foot_clearance_temp = 1.0
    ideal_feet_in_air = 1.5
    feet_in_air_error = torch.square(num_feet_in_air - ideal_feet_in_air)
    foot_clearance_reward = torch.where(
        is_walking,
        torch.exp(-feet_in_air_error / foot_clearance_temp),
        torch.exp(-num_feet_in_air / foot_clearance_temp)  # All feet down when idle
    )
    
    # === Reward 8: Upright orientation ===
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
    upright_error = torch.square(projected_gravity[:, 2] + 1.0)
    upright_temp = 0.3
    upright_reward = torch.exp(-upright_error / upright_temp)
    
    # === Reward 9: Minimize base angular velocity ===
    ang_vel_squared = torch.sum(torch.square(base_ang_vel), dim=-1)
    ang_vel_temp = 0.25  # Stricter (was 0.5)
    ang_vel_reward = torch.exp(-ang_vel_squared / ang_vel_temp)
    
    # === Reward 10: Leg action magnitude when walking ===
    leg_actions = actions[:, leg_dof_indices]
    leg_action_magnitude = torch.sum(torch.abs(leg_actions), dim=-1)
    leg_action_temp = 8.0
    # Encourage leg actions when walking, minimize when idle
    leg_action_reward = torch.where(
        is_walking,
        torch.tanh(leg_action_magnitude / leg_action_temp),  # Encourage
        torch.exp(-leg_action_magnitude / leg_action_temp)   # Minimize
    )
    
    # === Reward 11: Gait frequency - encourage rhythmic motion ===
    # Sum of absolute velocities across leg joints
    leg_vel_variance = torch.var(leg_dof_vel, dim=-1)
    gait_temp = 2.0
    gait_rhythm_reward = torch.where(
        is_walking,
        torch.tanh(leg_vel_variance / gait_temp),
        torch.zeros_like(leg_vel_variance)
    )
    
    # Combine rewards with adjusted weights
    reward = (
        3.0 * velocity_reward +                    # Increased priority
        0.5 * lateral_reward +
        0.5 * vertical_reward +
        3.0 * wheel_passive_reward +               # Much higher weight
        3.0 * wheel_torque_reward +                # Much higher weight
        2.0 * leg_motion_reward +                  # Increased priority
        1.5 * foot_clearance_reward +              # New - encourage stepping
        1.0 * upright_reward +
        1.5 * ang_vel_reward +                     # Increased
        1.5 * leg_action_reward +                  # New - encourage leg use
        0.5 * gait_rhythm_reward                   # New - rhythmic gait
    )
    
    reward_components = {
        "velocity_reward": velocity_reward,
        "lateral_reward": lateral_reward,
        "vertical_reward": vertical_reward,
        "wheel_passive_reward": wheel_passive_reward,
        "wheel_torque_reward": wheel_torque_reward,
        "leg_motion_reward": leg_motion_reward,
        "foot_clearance_reward": foot_clearance_reward,
        "upright_reward": upright_reward,
        "ang_vel_reward": ang_vel_reward,
        "leg_action_reward": leg_action_reward,
        "gait_rhythm_reward": gait_rhythm_reward
    }
    
    return reward, reward_components
