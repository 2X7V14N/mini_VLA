import numpy as np
import mujoco


def ik(model, data, target, site_id, n_iter=300, step=0.5, damping=1e-3, verbose=False):
    """迭代求解：让 site 到达 target（世界坐标）。

    参数:
        target:  目标点 [x, y, z]
        site_id: 末端的 site id
        n_iter:  最大迭代次数
        step:    每次迭代走多大一步（太大会震荡，太小会慢）
        damping: 阻尼系数（在奇异位形附近保持稳定）
        verbose: 是否打印每次迭代的误差

    返回:
        (关节角, 最终误差)
    """
    jacp = np.zeros((3, model.nv))      # 位置雅可比
    jacr = np.zeros((3, model.nv))      # 旋转雅可比（这里不用）

    q = np.array(data.qpos[:3], dtype=float)   # 从当前姿态出发

    for i in range(n_iter):
        # ① 把当前关节角写进去，更新运动学
        data.qpos[:3] = q
        mujoco.mj_forward(model, data)

        # ② 现在末端在哪？离目标差多少？
        now = np.asarray(data.site(site_id).xpos)
        err = target - now
        if verbose and i % 20 == 0:
            print(f"      iter {i:3d}  误差 {np.linalg.norm(err):.6f}")
        if np.linalg.norm(err) < 1e-6:
            break

        # ③ 取雅可比："关节转一点点，末端动多少"
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp[:, :3]                 # 只取机械臂那 3 个关节对应的列

        # ④ 反解：想让末端走 err，关节该转多少
        #    阻尼最小二乘  dq = Jᵀ (J Jᵀ + λI)⁻¹ err
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)

        # ⑤ 走一步
        q = q + step * dq

    # 最后写回去
    data.qpos[:3] = q
    mujoco.mj_forward(model, data)
    return q, float(np.linalg.norm(target - np.asarray(data.site(site_id).xpos)))


def main():
    model = mujoco.MjModel.from_xml_path("src/mini_vla/simulation/scene.xml")
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # 目标：红方块正上方 0.10 米
    target = np.array([0.20, 0.07, 0.04 + 0.035 + 0.10])
    print(f"目标点: {target.round(4).tolist()}\n")

    q, err = ik(model, data, target, site_id, verbose=True)

    print(f"\n结果:")
    print(f"  关节角   {q.round(4).tolist()}")
    print(f"  末端位置 {np.asarray(data.site(site_id).xpos).round(4).tolist()}")
    print(f"  残余误差 {err:.8f}")
    print(f"  {'✅ 到达' if err < 1e-4 else '❌ 没到达'}")


if __name__ == "__main__":
    main()
