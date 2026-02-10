
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict


class Go2BoundClaude(VecTask):

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
        asset_file = "urdf/go2/urdf/go2.urdf"

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

        go2_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(go2_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(go2_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(go2_asset)
        self.dof_names = self.gym.get_asset_dof_names(go2_asset)
        extremity_name = "calf" if asset_options.collapse_fixed_joints else "FOOT"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "thigh" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(go2_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.go2_handles = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            go2_handle = self.gym.create_actor(env_ptr, go2_asset, start_pose, "go2", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, go2_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, go2_handle)
            self.envs.append(env_ptr)
            self.go2_handles.append(go2_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.go2_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.go2_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.go2_handles[0], "base")

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
        self.rew_buf[:], self.rew_dict = compute_reward(self.root_states, self.dof_pos, self.dof_vel, self.contact_forces, self.commands, self.actions, self.default_dof_pos, self.gravity_vec)
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

        self.obs_buf[:] = compute_go2_observations(  # tensors
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
def compute_go2_observations(root_states,
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
    # Base velocities
    # ----------------------------
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])
    base_height = root_states[:, 2]

    # Extract commanded velocities
    cmd_vx = commands[:, 0]  # Forward velocity command
    cmd_vy = commands[:, 1]  # Lateral velocity command
    cmd_vyaw = commands[:, 2]  # Yaw rate command

    # Velocity tracking errors
    vx_error = torch.square(cmd_vx - base_lin_vel[:, 0])
    vy_error = torch.square(cmd_vy - base_lin_vel[:, 1])
    vyaw_error = torch.square(cmd_vyaw - base_ang_vel[:, 2])

    # Stronger velocity tracking rewards (exponential)
    rew_lin_vel_x = torch.exp(-vx_error / 0.25) * rew_scales.get("lin_vel_x", 2.0)
    rew_lin_vel_y = torch.exp(-vy_error / 0.25) * rew_scales.get("lin_vel_y", 0.5)
    rew_ang_vel_z = torch.exp(-vyaw_error / 0.25) * rew_scales.get("ang_vel_z", 0.5)

    # ----------------------------
    # Contact detection
    # foot order: [FL, FR, RL, RR]
    # ----------------------------
    contact_thresh = 1.0
    foot_forces = torch.norm(contact_forces[:, feet_indices, :], dim=2)
    foot_contact = (foot_forces > contact_thresh).float()

    front_pair = foot_contact[:, 0] * foot_contact[:, 1]  # Both front feet
    rear_pair = foot_contact[:, 2] * foot_contact[:, 3]   # Both rear feet
    
    # ----------------------------
    # Bounding gait metrics
    # ----------------------------
    
    # 1. Paired contact (front OR rear, not both)
    paired_contact = (front_pair + rear_pair).clamp(0, 1)
    both_pairs_contact = front_pair * rear_pair  # Bad: all feet down
    
    # 2. Phase alternation (front and rear should alternate)
    phase_alternation = paired_contact * (1.0 - both_pairs_contact)
    
    # 3. Flight phase detection (no feet in contact)
    no_contact = (1.0 - foot_contact.sum(dim=1).clamp(0, 1))
    
    # 4. Symmetry within pairs (left-right force balance)
    front_symmetry = 1.0 - torch.abs(foot_forces[:, 0] - foot_forces[:, 1]) / (
        foot_forces[:, 0] + foot_forces[:, 1] + 1e-6
    )
    rear_symmetry = 1.0 - torch.abs(foot_forces[:, 2] - foot_forces[:, 3]) / (
        foot_forces[:, 2] + foot_forces[:, 3] + 1e-6
    )
    symmetry = (front_symmetry + rear_symmetry) * 0.5 * paired_contact
    
    # 5. IMPROVED: Forward propulsion (strongly reward forward velocity during contact)
    # This is the key fix - reward actual forward velocity, not just motion detection
    forward_velocity = base_lin_vel[:, 0]
    
    # When commanded to move forward, reward achieving that velocity
    commanded_to_move = cmd_vx.abs() > 0.1
    
    # Reward forward velocity achievement during paired contacts
    velocity_achievement = torch.exp(-torch.square(forward_velocity - cmd_vx) / 0.5) * paired_contact
    
    # Also reward forward velocity during flight (maintains momentum)
    flight_momentum = (forward_velocity / (cmd_vx.abs() + 1e-6)).clamp(0, 2.0) * no_contact
    
    # 6. Height maintenance during bounding
    target_height = 0.35  # Target CoM height for GO2
    height_error = torch.square(base_height - target_height)
    height_reward = torch.exp(-height_error / 0.05)
    
    
    # 8. Orientation stability (minimal roll and yaw)
    up_vec_local = torch.zeros(batch_size, 3, dtype=torch.float, device=device)
    up_vec_local[:, 2] = 1.0
    up_vec = quat_rotate(base_quat, up_vec_local)
    
    # Forward vector should point forward (minimal yaw deviation)
    forward_vec_local = torch.zeros(batch_size, 3, dtype=torch.float, device=device)
    forward_vec_local[:, 0] = 1.0
    forward_vec = quat_rotate(base_quat, forward_vec_local)
    
    # Reward upright orientation
    upright_reward = up_vec[:, 2]  # Should be close to 1.0
    forward_alignment = forward_vec[:, 0]  # Should be close to 1.0
    
    orientation_reward = (upright_reward * 0.6 + forward_alignment * 0.4)
    
    # 9. Lateral drift penalty (should move straight)
    lateral_velocity = torch.abs(base_lin_vel[:, 1])
    lateral_penalty = -lateral_velocity * rew_scales.get("lateral_drift", 0.3)
    
    # ----------------------------
    # IMPROVED: Bounding quality score
    # ----------------------------
    # Reweighted to emphasize velocity tracking and forward propulsion
    bounding_quality = (
        phase_alternation * 0.25 +          # 25%: Correct pair contacts
        no_contact * 0.15 +                  # 15%: Flight phases present
        symmetry * 0.15 +                    # 15%: Left-right symmetry
        velocity_achievement * 0.30 +        # 30%: Achieving commanded velocity (NEW - highest weight!)
        flight_momentum * 0.10 +             # 10%: Maintaining momentum in flight
        orientation_reward * 0.05            # 5%: Staying upright and forward-facing
    )
    
    # ----------------------------
    # Penalties
    # ----------------------------
    
    # Penalize all-feet-down stance (prevents trotting)
    stance_penalty = both_pairs_contact * rew_scales.get("stance_penalty", -1.0)
    
    # Energy penalties
    rew_torque = -torch.sum(torch.abs(torques), dim=1) * rew_scales.get("torque", 0.0001)
    rew_torque_rate = -torch.sum(torch.square(torques), dim=1) * rew_scales.get("torque_square", 0.00005)
    
    # Penalize wrong gait patterns
    single_foot = (foot_contact.sum(dim=1) == 1.0).float()  # Hopping on one foot
    three_feet = (foot_contact.sum(dim=1) == 3.0).float()   # Awkward tripod
    wrong_pattern_penalty = -(single_foot + three_feet) * rew_scales.get("wrong_pattern", 0.5)
    
    # ----------------------------
    # IMPROVED: Reward structure
    # ----------------------------
    
    # When commanded to move (|cmd_vx| > 0.1)
    moving_rewards = (
        rew_lin_vel_x +              # Track forward velocity (high weight)
        rew_lin_vel_y +              # Track lateral velocity
        rew_ang_vel_z +              # Track yaw rate
        bounding_quality * rew_scales.get("bounding", 3.0) +  # Bounding quality (increased weight)
        height_reward * rew_scales.get("height", 0.5) +       # Maintain height
        lateral_penalty +            # Minimize lateral drift
        stance_penalty +             # Avoid stance phase
        wrong_pattern_penalty +      # Avoid wrong patterns
        rew_torque +                 # Energy efficiency
        rew_torque_rate              # Smooth torques
    ) * commanded_to_move.float()
    
    # When commanded to stay still (|cmd_vx| <= 0.1)
    still_reward = (
        (torch.abs(base_lin_vel[:, 0]) < 0.1).float() *
        (torch.abs(base_lin_vel[:, 1]) < 0.1).float() *
        (torch.abs(base_ang_vel[:, 2]) < 0.1).float() *
        foot_contact.sum(dim=1).clamp(min=2.0) / 4.0 *  # At least 2 feet on ground
        rew_scales.get("idle", 1.0)
    ) * (~commanded_to_move).float()
    
    total_reward = moving_rewards + still_reward
    total_reward = torch.clamp(total_reward, min=-10.0, max=10.0)
    
    # ----------------------------
    # Reset conditions
    # ----------------------------
    base_contact = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.0
    
    # Robot tilted too much
    tilted = up_vec[:, 2] < 0.3  # More lenient than before (cos(72°))
    
    # Height failure
    height_fail = (base_height < 0.20) | (base_height > 0.60)
    
    # Time out
    time_out = episode_lengths >= max_episode_length - 1
    
    reset = base_contact | tilted | height_fail | time_out

    # ----------------------------
    # IMPROVED: Success metric
    # ----------------------------
    
    # Success requires:
    # 1. Good bounding quality (> 0.6)
    # 2. Actually achieving commanded velocity (within 20%)
    # 3. No wrong patterns (stance phase or single-foot)
    
    velocity_achieved = torch.abs(forward_velocity - cmd_vx) < (0.2 * cmd_vx.abs() + 0.1)
    
    is_success = (
        (bounding_quality > 0.6) &
        velocity_achieved &
        (both_pairs_contact < 0.5) &
        (single_foot < 0.5) &
        commanded_to_move
    ) | (
        # Or successfully idle when commanded
        (torch.abs(forward_velocity) < 0.15) &
        (~commanded_to_move)
    )
    
    consecutive_successes = torch.where(
        is_success,
        consecutive_successes + 1,
        torch.zeros_like(consecutive_successes)
    )
    
    consecutive_successes = (consecutive_successes.float()).mean()

    return total_reward.detach(), reset, consecutive_successes

from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(
    root_states: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    contact_forces: torch.Tensor,
    commands: torch.Tensor,
    actions: torch.Tensor,
    default_dof_pos: torch.Tensor,
    gravity_vec: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Improved reward function for Go2 bounding gait.
    """
    
    # Extract robot state
    base_pos = root_states[:, :3]
    base_quat = root_states[:, 3:7]
    base_lin_vel = root_states[:, 7:10]
    base_ang_vel = root_states[:, 10:13]
    
    # Transform velocities to body frame
    base_lin_vel_body = quat_rotate_inverse(base_quat, base_lin_vel)
    base_ang_vel_body = quat_rotate_inverse(base_quat, base_ang_vel)
    
    # Projected gravity for orientation
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
    
    # Device reference
    device = root_states.device
    
    # Contact detection
    contact_force_norm = torch.norm(contact_forces, dim=2)  # (N, 4)
    contact_threshold = 1.0
    contacts = contact_force_norm > contact_threshold  # (N, 4) [FL, FR, RL, RR]
    
    # Front and rear pair contacts
    front_contacts = contacts[:, :2]  # FL, FR
    rear_contacts = contacts[:, 2:]   # RL, RR
    
    front_pair_contact = torch.sum(front_contacts.float(), dim=1)  # (N,)
    rear_pair_contact = torch.sum(rear_contacts.float(), dim=1)    # (N,)
    total_contacts = front_pair_contact + rear_pair_contact
    
    # Command analysis
    cmd_x = commands[:, 0]
    cmd_magnitude = torch.norm(commands[:, :2], dim=1)
    is_moving = (cmd_magnitude >= 0.1).float()
    should_stand = (cmd_magnitude < 0.1).float()
    
    # Temperature parameters
    temp_vel_tracking = 0.2
    temp_upright = 0.3
    temp_ang_vel = 0.5
    temp_height = 0.5
    temp_vertical_vel = 1.0
    temp_contact_timing = 1.0
    temp_stand_vel = 0.3
    
    # 1. Forward velocity tracking (increased importance)
    vel_x = base_lin_vel_body[:, 0]
    vel_y = base_lin_vel_body[:, 1]
    vel_tracking_error = torch.square(vel_x - cmd_x) + torch.square(vel_y - commands[:, 1])
    vel_tracking_reward = torch.exp(-vel_tracking_error / temp_vel_tracking)
    
    # 2. Upright orientation reward (critical - keep body upright)
    # Gravity should point down in body frame
    upright_error = torch.square(projected_gravity[:, 0]) + torch.square(projected_gravity[:, 1])
    upright_reward = torch.exp(-upright_error / temp_upright)
    
    # 3. Angular velocity penalty (minimize roll/yaw rotation)
    ang_vel_xy = torch.square(base_ang_vel_body[:, 0]) + torch.square(base_ang_vel_body[:, 2])
    ang_vel_reward = torch.exp(-ang_vel_xy / temp_ang_vel)
    
    # 4. Body height maintenance
    base_height = root_states[:, 2]
    target_height = 0.35
    height_error = torch.square(base_height - target_height)
    height_reward = torch.exp(-height_error / temp_height)
    
    # 5. Vertical velocity reward (encourage explosive push-offs)
    vertical_vel = base_lin_vel[:, 2]
    # Reward positive vertical velocity during contact (push-off)
    is_in_contact = (total_contacts > 0).float()
    vertical_push_reward = is_in_contact * torch.clamp(vertical_vel, 0.0, 2.0)
    vertical_vel_reward = torch.exp(vertical_push_reward / temp_vertical_vel) - 1.0
    
    # 6. Flight phase reward (encourage airborne time)
    is_airborne = (total_contacts == 0).float()
    flight_reward = is_airborne
    
    # 7. Pair synchronization (more discriminative)
    # Both feet in pair should have similar contact state
    front_both_contact = (front_pair_contact == 2).float()
    front_no_contact = (front_pair_contact == 0).float()
    front_sync_score = front_both_contact + front_no_contact
    
    rear_both_contact = (rear_pair_contact == 2).float()
    rear_no_contact = (rear_pair_contact == 0).float()
    rear_sync_score = rear_both_contact + rear_no_contact
    
    pair_sync_reward = 0.5 * (front_sync_score + rear_sync_score)
    
    # 8. Alternating gait (penalize all four feet or mixed contacts)
    # Good states: 0 feet (flight), 2 front, 2 rear
    good_contact_pattern = ((total_contacts == 0) | (total_contacts == 2)).float()
    # Also check that when 2 contacts, they're from same pair
    front_only = (front_pair_contact == 2) & (rear_pair_contact == 0)
    rear_only = (rear_pair_contact == 2) & (front_pair_contact == 0)
    valid_pair_contact = (front_only | rear_only).float()
    
    alternating_reward = good_contact_pattern * (1.0 + valid_pair_contact)
    
    # 9. Force symmetry within pairs (only when in contact)
    front_in_contact = (front_pair_contact > 0).float()
    rear_in_contact = (rear_pair_contact > 0).float()
    
    front_force_diff = torch.abs(contact_force_norm[:, 0] - contact_force_norm[:, 1])
    rear_force_diff = torch.abs(contact_force_norm[:, 2] - contact_force_norm[:, 3])
    
    front_symmetry = front_in_contact * torch.exp(-front_force_diff / 50.0)
    rear_symmetry = rear_in_contact * torch.exp(-rear_force_diff / 50.0)
    force_symmetry_reward = 0.5 * (front_symmetry + rear_symmetry)
    
    # 10. Standing still reward (when commanded velocity is near zero)
    standing_vel_error = torch.norm(base_lin_vel_body, dim=1) + torch.norm(base_ang_vel_body, dim=1)
    standing_reward = should_stand * torch.exp(-standing_vel_error / temp_stand_vel) * (total_contacts == 4).float()
    
    # 11. Contact timing reward (encourage regular alternation)
    # Penalize having same pair in contact for too long - implicitly via alternating reward
    
    # 12. Energy efficiency (limit joint velocities during stance)
    joint_vel_norm = torch.mean(torch.square(dof_vel), dim=1)
    energy_reward = is_in_contact * torch.exp(-joint_vel_norm / 100.0)
    
    # 13. Action smoothness
    action_norm = torch.mean(torch.square(actions), dim=1)
    action_reward = torch.exp(-action_norm / 2.0)
    
    # Combine rewards with adjusted weights
    bounding_reward = (
        5.0 * vel_tracking_reward +          # Increased: critical for task
        4.0 * upright_reward +                # Increased: robot was falling
        3.0 * ang_vel_reward +                # Increased: stability crucial
        2.0 * height_reward +                 # Keep stable
        2.0 * vertical_vel_reward +           # New emphasis on explosiveness
        2.5 * flight_reward +                 # Increased: need more airtime
        2.0 * pair_sync_reward +              # Keep important
        3.0 * alternating_reward +            # Increased: core gait pattern
        1.0 * force_symmetry_reward +         # Keep for balance
        0.5 * energy_reward +                 # Minor consideration
        0.5 * action_reward                   # Minor consideration
    )
    
    # Total reward
    total_reward = is_moving * bounding_reward + standing_reward * 10.0
    
    # Reward components dictionary
    reward_components = {
        "vel_tracking": vel_tracking_reward,
        "upright": upright_reward,
        "ang_vel": ang_vel_reward,
        "height": height_reward,
        "vertical_vel": vertical_vel_reward,
        "flight": flight_reward,
        "pair_sync": pair_sync_reward,
        "alternating": alternating_reward,
        "force_symmetry": force_symmetry_reward,
        "standing": standing_reward,
        "energy": energy_reward,
        "action": action_reward
    }
    
    return total_reward, reward_components
