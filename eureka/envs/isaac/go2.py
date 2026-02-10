
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict


class Go2(VecTask):

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
    max_episode_length,
    dt=0.02  # timestep
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, float) -> Tuple[Tensor, Tensor, Tensor]

    batch_size = root_states.shape[0]
    device = root_states.device

    # ----------------------------
    # Base velocities
    # ----------------------------
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    lin_vel_error = torch.sum((commands[:, :2] - base_lin_vel[:, :2]) ** 2, dim=1)
    ang_vel_error = (commands[:, 2] - base_ang_vel[:, 2]) ** 2

    rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * rew_scales.get("lin_vel_xy", 1.0)
    rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * rew_scales.get("ang_vel_z", 0.5)

    # ----------------------------
    # Contact detection
    # foot order: [FL, FR, RL, RR]
    # Trot diagonals: (FL + RR) and (FR + RL)
    # ----------------------------
    contact_thresh = 1.0
    foot_forces = torch.norm(contact_forces[:, feet_indices, :], dim=2)
    foot_contact = (foot_forces > contact_thresh).float()

    # Diagonal pairs for trotting
    diagonal1 = foot_contact[:, 0] * foot_contact[:, 3]  # FL + RR
    diagonal2 = foot_contact[:, 1] * foot_contact[:, 2]  # FR + RL
    
    # ----------------------------
    # Trot gait metrics
    # ----------------------------
    
    # 1. Diagonal contact (one diagonal pair down, not both)
    diagonal_contact = (diagonal1 + diagonal2).clamp(0, 1)
    both_diagonals = diagonal1 * diagonal2  # Bad: all feet down
    
    # 2. Diagonal alternation (key characteristic of trot)
    diagonal_alternation = diagonal_contact * (1.0 - both_diagonals)
    
    # 3. Duty cycle (percentage of time feet are in contact)
    # Trot typically has 50-60% duty cycle (longer than bounding)
    total_contact = foot_contact.sum(dim=1)
    duty_cycle = total_contact / 4.0  # Normalized 0-1
    # Ideal trot duty cycle around 0.5-0.6 (2 feet down at a time)
    duty_cycle_score = torch.exp(-((duty_cycle - 0.5) ** 2) / 0.1)
    
    # 4. Prevent wrong pairs (front pair or rear pair together)
    front_pair = foot_contact[:, 0] * foot_contact[:, 1]
    rear_pair = foot_contact[:, 2] * foot_contact[:, 3]
    wrong_pairs = front_pair + rear_pair
    
    # 5. Diagonal force symmetry (forces should be balanced in each diagonal)
    diagonal1_symmetry = 1.0 - torch.abs(foot_forces[:, 0] - foot_forces[:, 3]) / (
        foot_forces[:, 0] + foot_forces[:, 3] + 1e-6
    )
    diagonal2_symmetry = 1.0 - torch.abs(foot_forces[:, 1] - foot_forces[:, 2]) / (
        foot_forces[:, 1] + foot_forces[:, 2] + 1e-6
    )
    diagonal_symmetry = (diagonal1_symmetry + diagonal2_symmetry) * 0.5 * diagonal_contact
    
    # 6. Stability (body should remain level during trot)
    # Check roll and pitch angles
    up_vec_local = torch.zeros(batch_size, 3, dtype=torch.float, device=device)
    up_vec_local[:, 2] = 1.0
    up_vec = quat_rotate(base_quat, up_vec_local)
    
    # Body levelness (up vector should point mostly upward)
    body_level = up_vec[:, 2]  # Closer to 1.0 = more upright
    stability_score = (body_level > 0.85).float()  # Within ~30 degrees
    
    # 7. Forward motion consistency
    forward_velocity = base_lin_vel[:, 0]
    moving_forward = (forward_velocity > 0.1).float()
    velocity_consistency = moving_forward * diagonal_contact
    
    # 8. Minimal lateral drift (trot should be straight)
    lateral_drift = torch.abs(base_lin_vel[:, 1])
    drift_penalty = torch.exp(-lateral_drift / 0.5)
    
    # 9. Height consistency (CoM should stay relatively constant)
    base_height = root_states[:, 2]
    # Penalize if too low (crouching) or bouncing too much
    height_score = torch.exp(-((base_height - 0.35) ** 2) / 0.05)  # Target ~0.35m
    
    
    
    # ----------------------------
    # Trot quality score
    # ----------------------------
    trot_quality = (
        diagonal_alternation * 0.30 +    # 30%: Correct diagonal contacts
        duty_cycle_score * 0.15 +         # 15%: Appropriate duty cycle
        diagonal_symmetry * 0.20 +        # 20%: Diagonal force balance
        stability_score * 0.15 +          # 15%: Body stability
        velocity_consistency * 0.10 +     # 10%: Consistent forward motion
        drift_penalty * 0.10              # 10%: Minimal lateral drift
    )
    
    # Penalties for incorrect gait patterns
    wrong_pair_penalty = wrong_pairs * rew_scales.get("wrong_pair_penalty", -0.3)
    both_diag_penalty = both_diagonals * rew_scales.get("both_diag_penalty", -0.4)
    
    # ----------------------------
    # Rewards
    # ----------------------------
    rew_trot = trot_quality * rew_scales.get("trot", 2.0)
    rew_torque = -torch.sum(torch.abs(torques), dim=1) * rew_scales.get("torque", 0.0001)
    rew_height = height_score * rew_scales.get("height", 0.3)
    
    # Energy efficiency (smooth, consistent motion)
    torque_smoothness = -torch.sum(torques ** 2, dim=1) * rew_scales.get("smoothness", 0.0001)
    
    total_reward = (
        rew_lin_vel_xy + 
        rew_ang_vel_z + 
        rew_trot + 
        rew_torque + 
        rew_height +
        torque_smoothness +
        wrong_pair_penalty +
        both_diag_penalty
    )
    
    # ----------------------------
    # Reset conditions
    # ----------------------------
    base_contact = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.0
    
    # Robot tilted too much (more lenient than bounding)
    tilted = up_vec[:, 2] < 0.3  # ~72 degrees (trot is more stable)
    
    # Height failure (too low or flipped)
    height_fail = (base_height < 0.15) | (base_height > 0.6)
    
    reset = base_contact | tilted | height_fail
    time_out = episode_lengths >= max_episode_length - 1
    reset = reset | time_out

    # ----------------------------
    # Success metric (temporal consistency)
    # ----------------------------
    # Success = consistent trotting over time
    is_success = (trot_quality > 0.65) & (wrong_pairs < 0.5)
    consecutive_successes = torch.where(
        is_success,
        consecutive_successes + 1,
        torch.zeros_like(consecutive_successes)
    )
    
    # Normalized success (for logging)
    # Trot needs longer consistency than bounding (20+ steps)
    success_rate = (consecutive_successes > 20).float().mean()

    return total_reward.detach(), reset, success_rate