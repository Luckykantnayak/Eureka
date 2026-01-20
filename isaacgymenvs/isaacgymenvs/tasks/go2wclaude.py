
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict


class Go2wClaude(VecTask):

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
            self.knee_indices,
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



@torch.jit.script
def compute_success(
    root_states,
    commands,
    torques,
    contact_forces,
    knee_indices,
    consecutive_successes,
    episode_lengths,
    rew_scales,
    base_index,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int) -> Tuple[Tensor, Tensor, Tensor]

    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque
    total_reward = torch.clip(total_reward, 0., None)
    reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
    reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out
    
    consecutive_successes = -(lin_vel_error + ang_vel_error).mean()
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
    
    # Extract base orientation and velocities
    base_quat = root_states[:, 3:7]
    base_lin_vel = root_states[:, 7:10]
    base_ang_vel = root_states[:, 10:13]
    
    # Transform velocities to base frame
    base_lin_vel_base = quat_rotate_inverse(base_quat, base_lin_vel)
    base_ang_vel_base = quat_rotate_inverse(base_quat, base_ang_vel)
    
    # Extract target velocities from commands: [vx, vy, vyaw]
    target_vx = commands[:, 0]
    target_vy = commands[:, 1]
    target_vyaw = commands[:, 2]
    
    # Extract actual velocities in base frame
    actual_vx = base_lin_vel_base[:, 0]
    actual_vy = base_lin_vel_base[:, 1]
    actual_vyaw = base_ang_vel_base[:, 2]
    
    # Velocity tracking rewards - primary objective
    vx_error = torch.square(actual_vx - target_vx)
    vy_error = torch.square(actual_vy - target_vy)
    vyaw_error = torch.square(actual_vyaw - target_vyaw)
    
    vx_temp: float = 0.5
    vy_temp: float = 0.5
    vyaw_temp: float = 0.5
    
    vx_reward = torch.exp(-vx_error / vx_temp)
    vy_reward = torch.exp(-vy_error / vy_temp)
    vyaw_reward = torch.exp(-vyaw_error / vyaw_temp)
    
    # Penalize excessive leg joint velocities (encourage static legs)
    leg_dof_vel_magnitude = torch.sum(torch.square(dof_vel[:, leg_dof_indices]), dim=-1)
    leg_vel_temp: float = 50.0
    leg_vel_penalty = torch.exp(-leg_dof_vel_magnitude / leg_vel_temp)
    
    # Penalize large deviations from default leg pose (stability)
    leg_dof_pos_error = torch.sum(torch.square(dof_pos[:, leg_dof_indices] - default_dof_pos[:, leg_dof_indices]), dim=-1)
    leg_pos_temp: float = 1.0
    leg_pos_penalty = torch.exp(-leg_dof_pos_error / leg_pos_temp)
    
    # Penalize rapid changes in wheel actions (encourage smooth control)
    wheel_action_magnitude = torch.square(actions[:, wheel_dof_indices]).sum(dim=-1)
    wheel_action_temp: float = 4.0
    wheel_action_smoothness = torch.exp(-wheel_action_magnitude / wheel_action_temp)
    
    # Encourage upright orientation
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
    upright_error = torch.square(projected_gravity[:, 2] + 1.0)
    upright_temp: float = 0.3
    upright_reward = torch.exp(-upright_error / upright_temp)
    
    # Penalize leg action magnitude (keep legs mostly static)
    leg_action_magnitude = torch.sum(torch.square(actions[:, leg_dof_indices]), dim=-1)
    leg_action_temp: float = 10.0  # Increased from 5.0 to reduce saturation
    leg_action_smoothness = torch.exp(-leg_action_magnitude / leg_action_temp)
    
    # Penalize vertical velocity (should move horizontally, not bounce)
    vz = base_lin_vel_base[:, 2]
    vz_error = torch.square(vz)
    vz_temp: float = 0.5
    vz_penalty = torch.exp(-vz_error / vz_temp)
    
    # Energy efficiency: penalize excessive torques
    torque_magnitude = torch.sum(torch.square(actions), dim=-1)
    torque_temp: float = 10.0
    energy_penalty = torch.exp(-torque_magnitude / torque_temp)
    
    # Encourage wheel usage - wheels should be spinning when there's a velocity command
    command_magnitude = torch.sqrt(torch.square(target_vx) + torch.square(target_vy))
    wheel_vel_magnitude = torch.abs(dof_vel[:, wheel_dof_indices]).mean(dim=-1)
    # Only reward wheel spinning when there's a command
    wheel_usage = torch.where(
        command_magnitude > 0.1,
        torch.clamp(wheel_vel_magnitude / 10.0, 0.0, 1.0),
        torch.ones_like(wheel_vel_magnitude)
    )
    
    # Combine rewards with adjusted weights to make total more positive
    # Increased weights on primary tracking objectives
    total_reward = (
        5.0 * vx_reward +                  # Primary: track x velocity (increased from 4.0)
        5.0 * vy_reward +                  # Primary: track y velocity (increased from 4.0)
        4.0 * vyaw_reward +                # Primary: track yaw velocity (increased from 3.0)
        0.5 * leg_vel_penalty +            # Keep legs still (increased from 0.3)
        0.3 * leg_pos_penalty +            # Maintain stable leg pose (increased from 0.2)
        0.5 * wheel_action_smoothness +    # Smooth wheel control (increased from 0.4)
        2.0 * upright_reward +             # Stay upright (increased from 1.5)
        0.3 * leg_action_smoothness +      # Minimize leg actuation (increased from 0.2)
        0.8 * vz_penalty +                 # Minimize vertical motion (increased from 0.5)
        0.4 * energy_penalty +             # Energy efficiency (increased from 0.3)
        0.5 * wheel_usage                  # Encourage wheel usage (new)
    )
    
    reward_components = {
        "vx_reward": vx_reward,
        "vy_reward": vy_reward,
        "vyaw_reward": vyaw_reward,
        "leg_vel_penalty": leg_vel_penalty,
        "leg_pos_penalty": leg_pos_penalty,
        "wheel_action_smoothness": wheel_action_smoothness,
        "upright_reward": upright_reward,
        "leg_action_smoothness": leg_action_smoothness,
        "vz_penalty": vz_penalty,
        "energy_penalty": energy_penalty,
        "wheel_usage": wheel_usage
    }
    
    return total_reward, reward_components
