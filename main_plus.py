from __future__ import annotations

import os
import numpy as np
import genesis as gs
import argparse
import json

def quat_geodesic_angle(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    d = abs(np.dot(q1, q2))
    d = np.clip(d, 0.0, 1.0)
    d = 2.0 * np.arccos(d)
    return np.min([d, 2 * np.pi - d])

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quat_to_rotmat(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion in [w, x, y, z] format.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.

    Notes
    -----
    Assumes right-handed coordinate system and active rotation.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("Quaternion must be shape (4,), given {}".format(q.shape))

    # Normalize to guard against drift
    n = np.dot(q, q)
    if n < np.finfo(float).eps:
        raise ValueError("Zero-norm quaternion is not valid")
    q = q / np.sqrt(n)

    w, x, y, z = q

    # Precompute products
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z

    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [    2*(xy + wz),  1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

    return R

def rotation_to_quaternion(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 - R[0,0] + R[1,1] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 - R[0,0] - R[1,1] + R[2,2]) * 2
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return q

def recover_pos_quat(P, Q):
    P = np.asarray(P)
    Q = np.asarray(Q)
    assert P.shape == Q.shape
    assert P.shape[1] == 3

    N = P.shape[0]

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = P_centered.T @ Q_centered

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_Q - R @ centroid_P

    return t, rotation_to_quaternion(R)

def quat_to_axis_angle(quat):
    if quat[0] < 0:
        quat = -quat
    quat /= np.linalg.norm(quat)
    axis = quat[1:]
    if np.linalg.norm(axis) < 1e-8:
        return np.array([0.0, 0.0, 0.0])
    else:
        angle = 2.0 * np.arccos(quat[0])
        axis = axis / np.linalg.norm(axis)
        return axis * angle

def cube_sdf(lower_bound, upper_bound, x):
    sdf = 0.0
    normal = [0.0, 0.0, 0.0]
    diff = [0.0, 0.0, 0.0]
    dim_sdf = [0.0, 0.0, 0.0]
    for i in range(3):
        dim_sdf[i] = max(lower_bound[i] - x[i], x[i] - upper_bound[i])
        if x[i] < lower_bound[i]:
            diff[i] = x[i] - lower_bound[i]
        elif x[i] > upper_bound[i]:
            diff[i] = x[i] - upper_bound[i]
    if dim_sdf[0] < 0 and dim_sdf[1] < 0 and dim_sdf[2] < 0:
        sdf = max(dim_sdf[0], dim_sdf[1], dim_sdf[2])
        if sdf == dim_sdf[0]:
            if x[0] - lower_bound[0] < upper_bound[0] - x[0]:
                normal = [-1.0, 0.0, 0.0]
            else:
                normal = [1.0, 0.0, 0.0]
        elif sdf == dim_sdf[1]:
            if x[1] - lower_bound[1] < upper_bound[1] - x[1]:
                normal = [0.0, -1.0, 0.0]
            else:
                normal = [0.0, 1.0, 0.0]
        else:
            if x[2] - lower_bound[2] < upper_bound[2] - x[2]:
                normal = [0.0, 0.0, -1.0]
            else:
                normal = [0.0, 0.0, 1.0]
    else:
        sdf = np.linalg.norm(diff)
        if sdf > 1e-8:
            normal = diff / sdf
        else:
            normal = [0.0, 0.0, 0.0]
    return sdf, normal

def t_shape_sdf(x):
    lower_bound = np.array([-0.075, 0.005, -0.025])
    upper_bound = np.array([0.075, 0.055, 0.025])
    sdf1, normal1 = cube_sdf(lower_bound, upper_bound, x)

    lower_bound = np.array([-0.25, -0.095, -0.025])
    upper_bound = np.array([0.25, 0.055, 0.025])
    sdf2, normal2 = cube_sdf(lower_bound, upper_bound, x)

    if sdf1 < sdf2:
        return sdf1, normal1
    else:
        return sdf2, normal2

class ControlPolicy:
    def __init__(self):
        
        self.key_point_poses = [
            np.array([0.000, 0.055, 0.000]), # 0
            np.array([0.025, 0.055, 0.000]), # 1
            np.array([0.050, 0.055, 0.000]), # 2
            np.array([0.075, 0.055, 0.000]), # 3
            np.array([0.075, 0.055, 0.000]), # 4
            np.array([0.075, 0.055, 0.000]), # 5
            np.array([0.075, 0.030, 0.000]), # 6
            np.array([0.075, 0.005, 0.000]), # 7
            np.array([0.075, 0.005, 0.000]), # 8
            np.array([0.075, 0.005, 0.000]), # 9
            np.array([0.050, 0.005, 0.000]), # 10
            np.array([0.025, -0.020, 0.000]), # 11
            np.array([0.025, -0.045, 0.000]), # 12
            np.array([0.025, -0.070, 0.000]), # 13
            np.array([0.025, -0.095, 0.000]), # 14
            np.array([0.025, -0.095, 0.000]), # 15
            np.array([0.025, -0.095, 0.000]), # 16
            np.array([0.000, -0.095, 0.000]), # 17
            np.array([-0.025, -0.095, 0.000]), # 18
            np.array([-0.025, -0.095, 0.000]), # 19
            np.array([-0.025, -0.095, 0.000]), # 20
            np.array([-0.025, -0.070, 0.000]), # 21
            np.array([-0.025, -0.045, 0.000]), # 22
            np.array([-0.025, -0.020, 0.000]), # 23
            np.array([-0.050, 0.005, 0.000]), # 24
            np.array([-0.075, 0.005, 0.000]), # 25
            np.array([-0.075, 0.005, 0.000]), # 26
            np.array([-0.075, 0.005, 0.000]), # 27
            np.array([-0.075, 0.030, 0.000]), # 28
            np.array([-0.075, 0.055, 0.000]), # 29
            np.array([-0.075, 0.055, 0.000]), # 30
            np.array([-0.075, 0.055, 0.000]), # 31
            np.array([-0.050, 0.055, 0.000]), # 32
            np.array([-0.025, 0.055, 0.000]), # 33
        ]

        self.key_point_normals = [
            np.array([0.0, 1.0, 0.0]), # 0
            np.array([0.0, 1.0, 0.0]), # 1
            np.array([0.0, 1.0, 0.0]), # 2
            np.array([0.0, 1.0, 0.0]), # 3
            np.array([1.0, 1.0, 0.0]), # 4
            np.array([1.0, 0.0, 0.0]), # 5
            np.array([1.0, 0.0, 0.0]), # 6
            np.array([1.0, 0.0, 0.0]), # 7
            np.array([1.0, -1.0, 0.0]), # 8
            np.array([0.0, -1.0, 0.0]), # 9
            np.array([0.0, -1.0, 0.0]), # 10
            np.array([1.0, 0.0, 0.0]), # 11
            np.array([1.0, 0.0, 0.0]), # 12
            np.array([1.0, 0.0, 0.0]), # 13
            np.array([1.0, 0.0, 0.0]), # 14
            np.array([1.0, -1.0, 0.0]), # 15
            np.array([0.0, -1.0, 0.0]), # 16
            np.array([0.0, -1.0, 0.0]), # 17
            np.array([0.0, -1.0, 0.0]), # 18
            np.array([-1.0, -1.0, 0.0]), # 19
            np.array([-1.0, 0.0, 0.0]), # 20
            np.array([-1.0, 0.0, 0.0]), # 21
            np.array([-1.0, 0.0, 0.0]), # 22
            np.array([-1.0, 0.0, 0.0]), # 23
            np.array([0.0, -1.0, 0.0]), # 24
            np.array([0.0, -1.0, 0.0]), # 25
            np.array([-1.0, -1.0, 0.0]), # 26
            np.array([-1.0, 0.0, 0.0]), # 27
            np.array([-1.0, 0.0, 0.0]), # 28
            np.array([-1.0, 0.0, 0.0]), # 29
            np.array([-1.0, 1.0, 0.0]), # 30
            np.array([0.0, 1.0, 0.0]), # 31
            np.array([0.0, 1.0, 0.0]), # 32
            np.array([0.0, 1.0, 0.0]), # 33
        ]
        
        for key_point_normals in self.key_point_normals:
            key_point_normals /= np.linalg.norm(key_point_normals)

        self.last_index = -1

    def control(self, env: TaskEnv):
        pos, quat = env.get_pos_quat()
        print(f"pos: {pos}, quat: {quat}")
        R = quat_to_rotmat(quat)
        head_pos = env.get_head_pos()
        rel_pos = head_pos - pos
        rel_pos_in_obj = R.T @ rel_pos
        move_direction = -rel_pos_in_obj.copy()

        tpos_diff = (env.targ_pos - pos) / 0.1
        tpos_diff_in_obj = R.T @ tpos_diff
        tquat_diff = quat_multiply(quat_conjugate(env.targ_quat), quat)
        tangle_diff = quat_to_axis_angle(tquat_diff)[2]


        # find the closest key point
        best_score = 1e8
        best_index = -1
        keep_index = False
        for i, key_point in enumerate(self.key_point_poses):
            # cross 
            r = key_point
            f = -self.key_point_normals[i]
            tau = np.cross(r / 0.1, f)
            omega = tau / 0.3
            E = np.dot(tpos_diff_in_obj, tpos_diff_in_obj)
            E_dot = -2.0 * np.dot(tpos_diff_in_obj, f)

            E += tangle_diff ** 2
            E_dot += 2.0 * tangle_diff * omega[2]

            if i == self.last_index and E_dot < 0:
                keep_index = True
                break

            if E_dot < best_score:
                best_score = E_dot
                best_index = i

        if keep_index:
            best_index = self.last_index

        self.last_index = best_index

        kp_pos = self.key_point_poses[best_index].copy()
        kp_pos[2] = 0.025
        kp_normal = self.key_point_normals[best_index]
        start_point = kp_pos + kp_normal * 0.02

        if env.debug_mode:
            env.mark_point.set_pos(R @ start_point + pos)

        targ_z = 0.025
        speed = 0.02
        move_dir_xy = start_point[:2] - rel_pos_in_obj[:2]
        move_dir_z = targ_z - rel_pos_in_obj[2]
        if np.linalg.norm(rel_pos_in_obj[:2] - start_point[:2]) > 0.005:
            targ_z = 0.035
            speed = 0.05
            if rel_pos_in_obj[2] < 0.032:
                # set move_dir_xy to zero
                move_dir_xy[:] = 0.0
        else:
            targ_z = 0.025

        move_dir_z = targ_z - rel_pos_in_obj[2]
        move_direction[:2] = move_dir_xy
        move_direction[2] = move_dir_z

        kp_diff = rel_pos_in_obj - kp_pos
        ortho_kp_diff = kp_diff - np.dot(kp_diff, kp_normal) * kp_normal
        if np.dot(kp_diff, kp_normal) > 0 and np.linalg.norm(ortho_kp_diff) < 0.005:
            move_direction = -kp_normal - ortho_kp_diff * 100.0
            speed = 0.02


        # normalize
        move_direction /= (np.linalg.norm(move_direction) + 1e-8)
        env.set_head_pos(head_pos + R @ move_direction * speed)

        

class TaskEnv:
    def __init__(self, debug_mode=True, material="rigid"):
        self.data_dir = f"data/{material}"
        os.makedirs(self.data_dir, exist_ok=True)
        self.debug_mode = debug_mode
        self.material = material

        self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                    dt=5e-3,
                    substeps=50,
                ),
                vis_options=gs.options.VisOptions(
                    visualize_mpm_boundary=self.debug_mode
                ),
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(3, -1, 1.5),
                    camera_lookat=(0.0, 0.0, 0.0),
                    camera_fov=30,
                    max_FPS=60,
                ),
                mpm_options=gs.options.MPMOptions(
                    lower_bound=(0.0, -0.5, 0.0),
                    upper_bound=(1.0, 0.5, 1.0),
                ),
                fem_options=gs.options.FEMOptions(
                    use_implicit_solver=False,
                    damping=3.0
                ),
                pbd_options=gs.options.PBDOptions(
                    particle_size=0.003,
                ),
                show_viewer=True
            )

        # self.franka = self.scene.add_entity(
        #         gs.morphs.URDF(file="embodiments/franka-panda/panda.urdf", fixed=True, merge_fixed_links=False),
        #         material=gs.materials.Rigid(coup_friction=1.0),
        #     )
        # self.hand_link_name = "panda_hand"
        self.franka = self.scene.add_entity(
                gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, 0.1)),
                material=gs.materials.Rigid(coup_friction=0.0, friction=0.01, coup_restitution=0.1, coup_softness=0.02),
            )
        self.hand_link_name = "hand"

        self.plane = self.scene.add_entity(
                gs.morphs.Plane(),
            )
        
        self.targ_pos = np.array([0.5, -0.05, 0.125])
        self.targ_quat = np.array([1.0, 0.0, 0.0, 1.0])
        self.targ_quat /= np.linalg.norm(self.targ_quat)


        # sample from poffset from [-0.2, 0.2]^2 and quat_offset from [-1, 1]^2
        self.t_init_pos = np.array([0.55, -0.05, 0.126])
        self.t_init_quat = np.array([1.0, 0.0, 0.0, 0.0])

        table_material = gs.materials.Rigid(friction=1.0, coup_friction=0.5)
        self.next_round_wait_step = 50
        if self.material == "rigid":
            material = None
        elif self.material == "mpm":
            material = gs.materials.MPM.Elastic(rho=200)
        elif self.material == "fem":
            material = gs.materials.FEM.Elastic(
                E=1.0e5,  # stiffness
                nu=0.45,  # compressibility (0 to 0.5)
                rho=200.0,  # density
                friction_mu=3.0,
                hydroelastic_modulus=1e8,
                model="stable_neohookean")

        self.t_shape = self.scene.add_entity(
            gs.morphs.Mesh(force_retet=True, file="T-shape-modified.obj", pos=self.t_init_pos, quat=self.t_init_quat, scale=1.0), surface=gs.surfaces.Default(color=(0.6, 0.7, 0.8)),
            material=material
        )
        self.marker = self.scene.add_entity(gs.morphs.Mesh(file = "T-shape-modified.obj", pos=self.targ_pos - np.asarray([0.0, 0.0, 0.0495]), quat=self.targ_quat, scale=1.0, collision=False, fixed=True), surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)))
        self.table = self.scene.add_entity(gs.morphs.Box(size=(2.0, 2.0, 0.1), pos=(0.0, 0.0, 0.05), fixed=True), material=table_material, surface=gs.surfaces.Default(color=(0.8, 0.7, 0.6)))


        if self.debug_mode:
            self.mark_point = self.scene.add_entity(gs.morphs.Sphere(radius=0.02, collision=False, fixed=True), surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0), opacity=0.5))
            self.targ_point = self.scene.add_entity(gs.morphs.Sphere(radius=0.02, collision=False, fixed=True), surface=gs.surfaces.Default(color=(1.0, 1.0, 0.0), opacity=0.5))
            self.current_point = self.scene.add_entity(gs.morphs.Sphere(radius=0.02, collision=False, fixed=True), surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0), opacity=0.5))

        self.hand_cam = self.scene.add_camera(GUI=False, fov=70, res=(320, 320))
        self.scene_cam = self.scene.add_camera(GUI=False, fov=40, res=(320, 320), pos=(2, 0, 1.5), lookat=(0.0, 0.0, 0.0))
        self.cams = [self.hand_cam, self.scene_cam]
        T = np.eye(4)
        T[:3, :3] = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        T[:3, 3] = np.array([0.1, 0.0, 0.1])
        self.hand_cam.attach(self.franka.get_link("hand"), T)

        self.scene.build()

        print(f"{self.t_shape.get_mass_mat()}")

        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)

        # Optional: set control gains
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        # franka.control_dofs_position(np.array([0, 0]), fingers_dof) # you can also use position control
        self.control_policy = ControlPolicy()

    def set_pos_quat(self, pos, quat):
        self.scene.reset()
        
        self.t_pos = pos
        self.t_quat = quat

        if self.material == "rigid":
            self.t_shape.set_pos(pos)
            self.t_shape.set_quat(quat)
        elif self.material == "mpm":
            relative_particles = self.t_shape.get_particles() - self.t_init_pos
            self.P = relative_particles[0]
            R = quat_to_rotmat(quat)
            relative_particles = relative_particles.swapaxes(1, 2)
            relative_particles = R @ relative_particles
            relative_particles = relative_particles.swapaxes(1, 2)
            new_particles = relative_particles + pos
            self.t_shape.set_pos(0, new_particles)
        elif self.material == "fem":
            relative_particles = self.t_shape.get_state().pos.cpu().numpy() - self.t_init_pos
            # print(f"positions: ", relative_particles)
            self.P = relative_particles[0]
            R = quat_to_rotmat(quat)
            relative_particles = relative_particles.swapaxes(1, 2)
            relative_particles = R @ relative_particles
            relative_particles = relative_particles.swapaxes(1, 2)
            new_particles = relative_particles + pos
            new_particles = np.ascontiguousarray(new_particles)
            self.t_shape.set_position(new_particles)


    def get_pos_quat(self):
        if self.material == "rigid":
            return self.t_shape.get_pos().cpu().numpy(), self.t_shape.get_quat().cpu().numpy()
        elif self.material == "mpm":
            Q = self.t_shape.get_particles()[0]
            t, q = recover_pos_quat(self.P, Q)
            return t, q
        elif self.material == "fem":
            Q = self.t_shape.get_state().pos.cpu().numpy()[0]
            t, q = recover_pos_quat(self.P, Q)
            return t, q
        return self.t_pos, self.t_quat


    def reset_by_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
        poffset = np.random.uniform(-0.1, 0.1, size=(2,))
        quat_offset = np.random.uniform(-1, 1, size=(2,))
        quat_offset /= np.linalg.norm(quat_offset)

        pos=np.array([0.55, -0.05, 0.125]) + np.array([poffset[0] * 0.5, poffset[1], 0.0])
        quat=np.array([quat_offset[0], 0.0, 0.0, quat_offset[1]])

        # pos = np.array([0.0, 0.0, 0.125])
        # quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.set_pos_quat(pos, quat)

        self.end_targ_pos = np.array([0.45, 0.0, 0.3])

        self.end_effector = self.franka.get_link(self.hand_link_name)
        # move to pre-grasp pose
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=self.end_targ_pos,
            quat=np.array([0, 1, 0, 0]),
        )
        qpos[-2:] = 0.0
        self.franka.set_dofs_position(qpos[:-2], self.motors_dof)
        self.franka.set_dofs_position(qpos[-2:], self.fingers_dof)

        # grasp with 1N force
        self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
        self.franka.control_dofs_force(np.array([-1, -1]), self.fingers_dof)
        
        self.end_targs = []
        self.qposes = []
        self.qvels = []

    def combined_dist(dpos, dquat):
        return dpos * dpos * 100 + dquat * dquat

    def targ_dist(self):
        pos=self.targ_pos
        quat=self.targ_quat

        tpos, tquat = self.get_pos_quat()
        dpos = np.linalg.norm(pos - tpos)
        dquat = quat_geodesic_angle(quat, tquat)
        return dpos, dquat
    
    def reward(self):
        dpos, dquat = self.targ_dist()
        return TaskEnv.combined_dist(dpos, dquat)
    
    def get_head_pos(self):
        return self.end_effector.get_pos().cpu().numpy() + [-0.001, -0.001, -0.103]
    
    def set_head_pos(self, new_head_pos):
        self.end_targ_pos = new_head_pos + [0.003, 0.004, 0.112]

    def step(self):
        # get R from t_shape.get_quat with numpy functions
        tpos, tquat = self.get_pos_quat()
        R = quat_to_rotmat(tquat)
        targ_R = quat_to_rotmat(self.targ_quat)

        self.scene.step()
        if self.debug_mode:
            self.targ_point.set_pos(self.end_targ_pos + [-0.003, -0.004, -0.112])

            # Get position of end effector
            self.current_point.set_pos(self.get_head_pos())
        r = self.targ_dist()
        print("targ dist:", r)
        for cam in self.cams:
            cam.render()
        self.end_targs.append(self.end_targ_pos.copy())
        self.qposes.append(self.franka.get_dofs_position().cpu().numpy())
        self.qvels.append(self.franka.get_dofs_velocity().cpu().numpy())
        return r

    def run(self, seed=0):
        self.reset_by_seed(seed)
        for cam in self.cams:
            cam.start_recording()
        for i in range(10):
            self.step()

        last_dpos, last_dquat = self.targ_dist()

        # lift
        step = 0
        while last_dpos > 0.005 or last_dquat > 0.05:
            step += 1
            self.control_policy.control(self)
            
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=self.end_targ_pos,
                quat=np.array([0, 1, 0, 0]),
            )
            self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
            self.step()
            
            new_dpos, new_dquat = self.targ_dist()
            new_combined_score = TaskEnv.combined_dist(new_dpos, new_dquat)

            last_dpos, last_dquat = new_dpos, new_dquat

        self.hand_cam.stop_recording(f"{self.data_dir}/hand_record{self.seed}.mp4")
        self.scene_cam.stop_recording(f"{self.data_dir}/scene_record{self.seed}.mp4")

        # make all arrays python list
        self.qposes = [qpos.tolist() for qpos in self.qposes]
        self.qvels = [qvel.tolist() for qvel in self.qvels]
        self.end_targs = [end_targ.tolist() for end_targ in self.end_targs]

        data_dict = {
            "qpos": self.qposes,
            "qvel": self.qvels,
            "end_targ": self.end_targs
        }

        # save to episode.json
        with open(f"{self.data_dir}/episode{self.seed}.json", "w") as f:
            json.dump(data_dict, f)

def main():
    # make dir data/ is not exists
    gs.init(backend=gs.gpu)
    env = TaskEnv(debug_mode=False, material="rigid")
    for seed in range(100):
        env.run(seed=seed)

if __name__ == "__main__":
    main()