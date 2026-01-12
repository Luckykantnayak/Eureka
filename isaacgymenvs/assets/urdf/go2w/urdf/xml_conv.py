import mujoco

# Load URDF directly
model = mujoco.MjModel.from_xml_path("go2w.urdf")

# Save as MuJoCo XML
mujoco.mj_saveLastXML("go2w.xml", model)
