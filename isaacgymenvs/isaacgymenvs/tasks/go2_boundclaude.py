
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
        self.rew_buf[:], self.rew_dict = compute_reward(self.root_states, self.commands, self.dof_pos, self.dof_vel, self.contact_forces, self.actions, self.default_dof_pos, self.dt)
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
    commands: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    contact_forces: torch.Tensor,
    actions: torch.Tensor,
    default_dof_pos: torch.Tensor,
    dt: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reward function for Go2 bounding gait with velocity tracking.
    
    Args:
        root_states: (num_envs, 13) root position, orientation, linear and angular velocities
        commands: (num_envs, 3) commanded velocities [vx, vy, vyaw]
        dof_pos: (num_envs, 12) joint positions
        dof_vel: (num_envs, 12) joint velocities
        contact_forces: (num_envs, 4, 3) contact forces for 4 feet [FL, FR, RL, RR]
        actions: (num_envs, 12) actions
        default_dof_pos: (num_envs, 12) default joint positions
        dt: timestep
    """
    num_envs = root_states.shape[0]
    device = root_states.device
    
    # Extract state information
    base_pos = root_states[:, :3]
    base_quat = root_states[:, 3:7]
    base_lin_vel = root_states[:, 7:10]
    base_ang_vel = root_states[:, 10:13]
    
    # Convert to body frame
    base_lin_vel_body = quat_rotate_inverse(base_quat, base_lin_vel)
    base_ang_vel_body = quat_rotate_inverse(base_quat, base_ang_vel)
    
    # Compute contact state (threshold for contact detection)
    contact_threshold: float = 1.0
    contact_forces_norm = torch.norm(contact_forces, dim=-1)  # (num_envs, 4)
    feet_in_contact = contact_forces_norm > contact_threshold  # (num_envs, 4)
    
    # Feet indices: FL=0, FR=1, RL=2, RR=3
    fl_contact = feet_in_contact[:, 0]
    fr_contact = feet_in_contact[:, 1]
    rl_contact = feet_in_contact[:, 2]
    rr_contact = feet_in_contact[:, 3]
    
    # Bounding gait pattern rewards
    # Front pair synchronized (both or neither)
    front_sync = (fl_contact == fr_contact).float()
    
    # Rear pair synchronized (both or neither)
    rear_sync = (rl_contact == rr_contact).float()
    
    # Alternation: front and rear should not both be in contact (flight phase exists)
    front_pair_contact = fl_contact & fr_contact
    rear_pair_contact = rl_contact & rr_contact
    no_simultaneous_contact = (~(front_pair_contact & rear_pair_contact)).float()
    
    # Flight phase reward: all feet off ground
    all_feet_off = (~(fl_contact | fr_contact | rl_contact | rr_contact)).float()
    
    # Temperature parameters
    temp_sync: float = 0.5
    temp_alternation: float = 0.5
    temp_flight: float = 0.3
    
    # Gait pattern reward
    gait_reward = torch.exp(temp_sync * front_sync) + torch.exp(temp_sync * rear_sync)
    gait_reward += torch.exp(temp_alternation * no_simultaneous_contact)
    gait_reward += torch.exp(temp_flight * all_feet_off)
    
    # Velocity tracking reward
    cmd_vx = commands[:, 0]
    cmd_vy = commands[:, 1]
    cmd_vyaw = commands[:, 2]
    
    vx_error = torch.abs(base_lin_vel_body[:, 0] - cmd_vx)
    vy_error = torch.abs(base_lin_vel_body[:, 1] - cmd_vy)
    vyaw_error = torch.abs(base_ang_vel_body[:, 2] - cmd_vyaw)
    
    temp_vel: float = 0.5
    vel_tracking_reward = torch.exp(-temp_vel * vx_error) + torch.exp(-temp_vel * vy_error) + torch.exp(-temp_vel * vyaw_error)
    
    # Orientation reward (minimize roll and yaw, allow pitch variation)
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32).repeat(num_envs, 1)
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
    
    # Roll and yaw stability (upright orientation)
    roll_pitch_error = torch.abs(projected_gravity[:, 0]) + torch.abs(projected_gravity[:, 1])
    temp_orientation: float = 1.0
    orientation_reward = torch.exp(-temp_orientation * roll_pitch_error)
    
    # Left-right force symmetry within pairs
    fl_force = contact_forces_norm[:, 0]
    fr_force = contact_forces_norm[:, 1]
    rl_force = contact_forces_norm[:, 2]
    rr_force = contact_forces_norm[:, 3]
    
    front_symmetry_error = torch.abs(fl_force - fr_force) / (fl_force + fr_force + 1e-6)
    rear_symmetry_error = torch.abs(rl_force - rr_force) / (rl_force + rr_force + 1e-6)
    
    temp_symmetry: float = 2.0
    symmetry_reward = torch.exp(-temp_symmetry * front_symmetry_error) + torch.exp(-temp_symmetry * rear_symmetry_error)
    
    # Joint velocity penalty (minimize slipping during contact)
    dof_vel_penalty = torch.sum(torch.square(dof_vel), dim=-1)
    temp_dof_vel: float = 0.01
    dof_vel_reward = torch.exp(-temp_dof_vel * dof_vel_penalty)
    
    # Body height consistency
    body_height = base_pos[:, 2]
    target_height: float = 0.3
    height_error = torch.abs(body_height - target_height)
    temp_height: float = 5.0
    height_reward = torch.exp(-temp_height * height_error)
    
    # Action smoothness penalty
    action_diff = torch.sum(torch.square(actions), dim=-1)
    temp_action: float = 0.01
    action_reward = torch.exp(-temp_action * action_diff)
    
    # Standing reward when commanded velocity is near zero
    cmd_norm = torch.norm(commands, dim=-1)
    is_standing_cmd = cmd_norm < 0.1
    
    # Standing posture: all feet in contact, low velocities
    all_feet_contact = (fl_contact & fr_contact & rl_contact & rr_contact).float()
    low_vel = torch.norm(base_lin_vel, dim=-1) < 0.1
    low_ang_vel = torch.norm(base_ang_vel, dim=-1) < 0.1
    
    standing_reward = is_standing_cmd.float() * all_feet_contact * low_vel.float() * low_ang_vel.float()
    temp_standing: float = 1.0
    standing_reward = torch.exp(temp_standing * standing_reward)
    
    # Explosive push-off reward (vertical velocity during takeoff)
    vertical_vel = base_lin_vel[:, 2]
    temp_vertical: float = 1.0
    vertical_reward = torch.exp(temp_vertical * torch.clamp(vertical_vel, min=0.0, max=1.0))
    
    # Total reward with weights
    w_gait: float = 2.0
    w_vel: float = 3.0
    w_orient: float = 1.0
    w_symmetry: float = 1.0
    w_dof_vel: float = 0.5
    w_height: float = 1.0
    w_action: float = 0.1
    w_standing: float = 1.5
    w_vertical: float = 0.5
    
    total_reward = (
        w_gait * gait_reward +
        w_vel * vel_tracking_reward +
        w_orient * orientation_reward +
        w_symmetry * symmetry_reward +
        w_dof_vel * dof_vel_reward +
        w_height * height_reward +
        w_action * action_reward +
        w_standing * standing_reward +
        w_vertical * vertical_reward
    )
    
    reward_components = {
        "gait_reward": gait_reward,
        "vel_tracking_reward": vel_tracking_reward,
        "orientation_reward": orientation_reward,
        "symmetry_reward": symmetry_reward,
        "dof_vel_reward": dof_vel_reward,
        "height_reward": height_reward,
        "action_reward": action_reward,
        "standing_reward": standing_reward,
        "vertical_reward": vertical_reward
    }
    
    return total_reward, reward_components
