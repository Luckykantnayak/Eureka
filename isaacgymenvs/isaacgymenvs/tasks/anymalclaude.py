
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict


class AnymalClaude(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 48
        self.cfg["env"]["numActions"] = 12

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

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

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

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

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

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
        asset_file = "urdf/anymal_c/urdf/anymal.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
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

        anymal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(anymal_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(anymal_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(anymal_asset)
        self.dof_names = self.gym.get_asset_dof_names(anymal_asset)
        extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "FOOT"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "THIGH" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(anymal_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.anymal_handles = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            anymal_handle = self.gym.create_actor(env_ptr, anymal_asset, start_pose, "anymal", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, anymal_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, anymal_handle)
            self.envs.append(env_ptr)
            self.anymal_handles.append(anymal_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "base")

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.rew_dict = compute_reward(self.root_states, self.commands, self.dof_pos, self.dof_vel, self.actions, self.gravity_vec, self.default_dof_pos, self.contact_forces)
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

        self.obs_buf[:] = compute_anymal_observations(  # tensors
                                                        self.root_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions,
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale
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
def compute_anymal_observations(root_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions
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
def compute_reward(root_states: Tensor,
                   commands: Tensor,
                   dof_pos: Tensor,
                   dof_vel: Tensor,
                   actions: Tensor,
                   gravity_vec: Tensor,
                   default_dof_pos: Tensor,
                   contact_forces: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
    
    device = root_states.device
    num_envs = root_states.shape[0]
    
    # Extract relevant states
    base_quat = root_states[:, 3:7]
    base_lin_vel = root_states[:, 7:10]
    base_ang_vel = root_states[:, 10:13]
    base_pos = root_states[:, 0:3]
    
    # Quaternion components
    qx, qy, qz, qw = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
    
    # Rotate linear velocity to base frame
    lin_vel_base_x = (1 - 2*qy*qy - 2*qz*qz) * base_lin_vel[:, 0] + 2*(qx*qy + qw*qz) * base_lin_vel[:, 1] + 2*(qx*qz - qw*qy) * base_lin_vel[:, 2]
    lin_vel_base_y = 2*(qx*qy - qw*qz) * base_lin_vel[:, 0] + (1 - 2*qx*qx - 2*qz*qz) * base_lin_vel[:, 1] + 2*(qy*qz + qw*qx) * base_lin_vel[:, 2]
    ang_vel_base_z = 2*(qx*qz + qw*qy) * base_ang_vel[:, 0] + 2*(qy*qz - qw*qx) * base_ang_vel[:, 1] + (1 - 2*qx*qx - 2*qy*qy) * base_ang_vel[:, 2]
    
    # 1. Velocity tracking reward (working, keep and improve)
    vel_error_x = torch.square(commands[:, 0] - lin_vel_base_x)
    vel_error_y = torch.square(commands[:, 1] - lin_vel_base_y)
    vel_error_yaw = torch.square(commands[:, 2] - ang_vel_base_z)
    
    vel_tracking_temp = 1.0
    vel_tracking_reward = torch.exp(-vel_tracking_temp * (vel_error_x + vel_error_y + 0.5 * vel_error_yaw))
    
    # 2. Front leg contact penalty - PENALIZE any contact on front feet
    # Contact forces shape: (num_envs, num_bodies, 3)
    # Assuming feet indices: LF=1, RF=2, LH=3, RH=4 (adjust based on actual URDF)
    front_lf_contact = torch.norm(contact_forces[:, 1, :], dim=-1)
    front_rf_contact = torch.norm(contact_forces[:, 2, :], dim=-1)
    hind_lh_contact = torch.norm(contact_forces[:, 3, :], dim=-1)
    hind_rh_contact = torch.norm(contact_forces[:, 4, :], dim=-1)
    
    # Strongly penalize front leg contact
    front_contact_penalty_temp = 0.01
    front_contact_total = front_lf_contact + front_rf_contact
    no_front_contact_reward = torch.exp(-front_contact_penalty_temp * front_contact_total)
    
    # Reward back leg contact (they should be on ground)
    back_contact_min = torch.minimum(hind_lh_contact, hind_rh_contact)
    back_contact_threshold = 10.0
    back_contact_reward = torch.clamp(back_contact_min / back_contact_threshold, 0.0, 1.0)
    
    # 3. Front leg joint position reward - lift legs by bending joints
    # DOF order: LF_HAA(0), LF_HFE(1), LF_KFE(2), RF_HAA(3), RF_HFE(4), RF_KFE(5), 
    #            LH_HAA(6), LH_HFE(7), LH_KFE(8), RH_HAA(9), RH_HFE(10), RH_KFE(11)
    
    # For front legs, we want HFE to be more negative (retract leg back) and KFE more positive (bend knee)
    # This pulls the front legs up and back
    front_lf_hfe = dof_pos[:, 1]
    front_rf_hfe = dof_pos[:, 4]
    front_lf_kfe = dof_pos[:, 2]
    front_rf_kfe = dof_pos[:, 5]
    
    # Target: HFE should go negative (pull back), KFE should go positive (bend)
    target_front_hfe = -0.5  # Pull leg backward
    target_front_kfe = 1.2   # Bend knee significantly
    
    hfe_error = torch.square(front_lf_hfe - target_front_hfe) + torch.square(front_rf_hfe - target_front_hfe)
    kfe_error = torch.square(front_lf_kfe - target_front_kfe) + torch.square(front_rf_kfe - target_front_kfe)
    
    front_joint_temp = 1.0
    front_leg_pose_reward = torch.exp(-front_joint_temp * (hfe_error + 0.5 * kfe_error))
    
    # 4. Back legs stance - keep close to default for stability
    back_lh_hfe = dof_pos[:, 7]
    back_rh_hfe = dof_pos[:, 10]
    back_lh_kfe = dof_pos[:, 8]
    back_rh_kfe = dof_pos[:, 11]
    
    back_hfe_error = torch.square(back_lh_hfe - default_dof_pos[:, 7]) + torch.square(back_rh_hfe - default_dof_pos[:, 10])
    back_kfe_error = torch.square(back_lh_kfe - default_dof_pos[:, 8]) + torch.square(back_rh_kfe - default_dof_pos[:, 11])
    
    back_stance_temp = 1.0
    back_leg_stance_reward = torch.exp(-back_stance_temp * (back_hfe_error + back_kfe_error))
    
    # 5. Pitch reward - encourage backward lean (nose up)
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    
    # Target slight backward pitch (negative = nose up)
    target_pitch = -0.15
    pitch_temp = 4.0
    pitch_reward = torch.exp(-pitch_temp * torch.square(pitch - target_pitch))
    
    # Also reward any backward pitch beyond a threshold
    backward_pitch_bonus = torch.clamp(-pitch - 0.05, 0.0, 0.5) * 2.0  # Bonus for pitching back
    
    # 6. Roll penalty - keep level sideways (working well)
    sin_roll = 2.0 * (qw * qx + qy * qz)
    cos_roll = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sin_roll, cos_roll)
    
    roll_temp = 5.0
    roll_reward = torch.exp(-roll_temp * torch.square(roll))
    
    # 7. Height reward - encourage being upright (already working)
    height_temp = 5.0
    height_target = 0.48
    height_error = torch.square(base_pos[:, 2] - height_target)
    height_reward = torch.exp(-height_temp * height_error)
    
    # 8. Stability - penalize angular velocity (working well)
    stability_temp = 0.3
    ang_vel_magnitude = torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)
    stability_reward = torch.exp(-stability_temp * ang_vel_magnitude)
    
    # 9. Action smoothness reward
    action_temp = 0.02
    action_penalty = torch.sum(torch.square(actions), dim=-1)
    action_reward = torch.exp(-action_temp * action_penalty)
    
    # 10. Combined bipedal reward - high reward only if front legs not touching AND back legs touching
    bipedal_score = no_front_contact_reward * back_contact_reward
    
    # Total reward with emphasis on achieving bipedal stance
    total_reward = (0.25 * vel_tracking_reward + 
                    0.20 * no_front_contact_reward +
                    0.10 * back_contact_reward +
                    0.15 * front_leg_pose_reward +
                    0.05 * back_leg_stance_reward +
                    0.05 * pitch_reward + 
                    0.02 * backward_pitch_bonus +
                    0.03 * roll_reward +
                    0.03 * height_reward +
                    0.05 * stability_reward +
                    0.02 * action_reward +
                    0.05 * bipedal_score)
    
    reward_components = {
        "vel_tracking_reward": vel_tracking_reward,
        "no_front_contact_reward": no_front_contact_reward,
        "back_contact_reward": back_contact_reward,
        "front_leg_pose_reward": front_leg_pose_reward,
        "back_leg_stance_reward": back_leg_stance_reward,
        "pitch_reward": pitch_reward,
        "backward_pitch_bonus": backward_pitch_bonus,
        "roll_reward": roll_reward,
        "height_reward": height_reward,
        "stability_reward": stability_reward,
        "action_reward": action_reward,
        "bipedal_score": bipedal_score
    }
    
    return total_reward, reward_components
