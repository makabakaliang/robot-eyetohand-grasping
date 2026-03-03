import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

profile = pipeline.start(config)
dev = profile.get_device()
print("started:", dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))

for i in range(30):
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    print(i, "color=", bool(color), "depth=", bool(depth))

pipeline.stop()
print("done")
