import sys
sys.path.append('/home/rahul/Desktop/Robin/FoundationPoseROS2/FoundationPose')
sys.path.append('/home/rahul/Desktop/Robin/FoundationPoseROS2/FoundationPose/nvdiffrast')




import rclpy
from rclpy.node import Node
from estimater import (
    FoundationPose,
    ScorePredictor,
    PoseRefinePredictor,
    draw_posed_3d_box,
    draw_xyz_axis,
)
import cv2
import numpy as np
import trimesh
import nvdiffrast.torch as dr
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import argparse
import os
from scipy.spatial.transform import Rotation as R
from ultralytics import SAM
import tkinter as tk
from tkinter import Listbox, END, Button, Frame
import glob

from tf2_ros import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Pose


# FoundationPose to track whether register() has been called
original_init = FoundationPose.__init__
original_register = FoundationPose.register

def modified_init(self, model_pts, model_normals, symmetry_tfs=None,
                  mesh=None, scorer=None, refiner=None, glctx=None,
                  debug=0, debug_dir='./FoundationPose'):
    original_init(self, model_pts, model_normals, symmetry_tfs,
                  mesh, scorer, refiner, glctx, debug, debug_dir)
    self.is_registered_once = False

def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    if pose is not None:
        self.is_registered_once = True
    return pose

FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

class FileSelectorGUI:
    def __init__(self, master, file_paths):
        self.master = master
        self.master.title("Library: Sequence Selector")
        self.file_paths = file_paths
        self.reordered_paths = None

        self.master.minsize(500, 350)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=0)
        self.master.grid_columnconfigure(0, weight=1)

        self.font_large = ("Arial", 20)

        self.listbox = Listbox(
            master,
            selectmode="extended",
            width=70,
            height=15,
            font=self.font_large
        )
        self.listbox.grid(
            row=0, column=0,
            padx=10, pady=10,
            sticky="nsew"
        )

        for file_path in self.file_paths:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.listbox.insert(END, file_name)

        button_frame = Frame(master)
        button_frame.grid(row=1, column=0, pady=5)

        self.up_button = Button(
            button_frame, text="Move Up",
            command=self.move_up, font=self.font_large
        )
        self.up_button.pack(side="left", padx=5, pady=5)

        self.down_button = Button(
            button_frame, text="Move Down",
            command=self.move_down, font=self.font_large
        )
        self.down_button.pack(side="left", padx=5, pady=5)

        self.done_button = Button(
            button_frame, text="Done",
            command=self.done, font=self.font_large
        )
        self.done_button.pack(side="left", padx=5, pady=5)

    def move_up(self):
        selected = list(self.listbox.curselection())
        for idx in selected:
            if idx > 0:
                name = self.listbox.get(idx)
                self.listbox.delete(idx)
                self.listbox.insert(idx - 1, name)
                self.listbox.selection_set(idx - 1)

    def move_down(self):
        selected = list(self.listbox.curselection())
        for idx in reversed(selected):
            if idx < self.listbox.size() - 1:
                name = self.listbox.get(idx)
                self.listbox.delete(idx)
                self.listbox.insert(idx + 1, name)
                self.listbox.selection_set(idx + 1)

    def done(self):
        names = self.listbox.get(0, END)
        mapping = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in self.file_paths
        }
        self.reordered_paths = [mapping[n] for n in names]
        self.master.quit()
        self.master.destroy()

    def get_reordered_paths(self):
        return self.reordered_paths


def rearrange_files(file_paths):
    root = tk.Tk()
    gui = FileSelectorGUI(root, file_paths)
    root.mainloop()
    return gui.get_reordered_paths()

parser = argparse.ArgumentParser()
parser.add_argument('--est_refine_iter', type=int, default=4)
args = parser.parse_args()

class PoseEstimationNode(Node):
    def __init__(self, new_file_paths):
        super().__init__('pose_estimation_node')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ROS topics
        self.color_image_topic = '/zed/zed_node/rgb/image_rect_color'
        self.depth_image_topic = '/zed/zed_node/depth/depth_registered'
        self.camera_info_topic = '/zed/zed_node/rgb/camera_info'
        self.camera_frame_id = 'zed_left_camera_optical_frame'

        self.image_sub = self.create_subscription(
            Image, self.color_image_topic,
            self.image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_image_topic,
            self.depth_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic,
            self.camera_info_callback, 10
        )

        self.object_pose_camera_publisher = self.create_publisher(
            PoseStamped,
            '/robot_pose_estimation/object_pose_camera_frame',
            10
        )
        self.object_pose_base_publisher = self.create_publisher(
            PoseStamped,
            '/robot_pose_estimation/object_pose_base_frame',
            10
        )

        self.object_pose_for_path_planning_pub = self.create_publisher(
            Pose, '/object_pose_for_path_planning', 10
        )
       

        self.get_logger().info(
            f"Subscribing to color image: {self.color_image_topic}"
        )
        self.get_logger().info(
            f"Subscribing to depth image: {self.depth_image_topic}"
        )
        self.get_logger().info(
            f"Subscribing to camera info: {self.camera_info_topic}"
        )
        self.get_logger().info(
            f"Publishing object poses in camera frame: {self.camera_frame_id}"
        )

        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        self.cam_K = None
        # self.cam_K = np.array([
        #                             [265.3854, 0.0, 320.0],
        #                             [0.0, 265.3854, 180.0],
        #                             [0.0, 0.0, 1.0]
        #                         ])

        # Mesh data
        self.mesh_files = new_file_paths
        self._reload_mesh_data()

        # Control flags
        self.first_round = True
        self._hover_idx = None

        # Estimation utilities
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.seg_model = SAM("sam2.1_b.pt")

        # State
        self.state = "WAITING_FOR_CAMERA_INFO"
        self.pending_frame = False
        self.current_mesh_index = 0
        self.selected_mesh_info = []
        self.waiting_for_next_command = False
        self.visualization_image = None

        # Timers
        self.create_timer(0.05, self.keypress_handler)

    def _reload_mesh_data(self):
        self.initial_meshes = [trimesh.load(p) for p in self.mesh_files]
        self.initial_bounds = [trimesh.bounds.oriented_bounds(m)
                               for m in self.initial_meshes]
        self.initial_bboxes = [
            np.stack([-ext/2, ext/2], axis=0).reshape(2,3)
            for (_, ext) in self.initial_bounds
        ]

    def camera_info_callback(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.k).reshape((3,3))
            if self.cam_K[0,0] == 0 or self.cam_K[1,1] == 0:
                self.get_logger().warn("Camera intrinsic has zero focal lengths.")
                return
            self.get_logger().info(f"K matrix: {self.cam_K}")
            if self.state == "WAITING_FOR_CAMERA_INFO":
                self.state = "READY_FOR_SELECTION"
                self.get_logger().info("Ready for object selection.")

    def image_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.pending_frame = True

    def depth_callback(self, msg):
        d = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        d[np.isnan(d)] = 0
        d[(d<0.1)|(d==np.inf)] = 0
        self.depth_image = d
        self.pending_frame = True

    def process_images_main_loop(self):
        if self.state == "READY_FOR_SELECTION" and self.pending_frame:
            if self.color_image is None or self.depth_image is None or self.cam_K is None:
                return
            if self.cam_K[0,0] == 0 or self.cam_K[1,1] == 0:
                self.get_logger().warn("Skipping frame: invalid K matrix.")
                return

            self.pending_frame = False
            H, W = self.color_image.shape[:2]
            color = cv2.resize(self.color_image, (W,H), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(self.depth_image, (W,H), interpolation=cv2.INTER_NEAREST)

            self.get_logger().info("Starting new object selection round...")
            self.perform_object_selection(color, depth, H, W)
            self.state = "WAITING_FOR_USER_INPUT"

    def perform_object_selection(self, color, depth, H, W):
        # only reorder after first round
        if not self.first_round and len(self.mesh_files) > 1:
            reordered = rearrange_files(self.mesh_files)
            if reordered:
                self.mesh_files = reordered
                self._reload_mesh_data()
        self.first_round = False

        self.current_mesh_index = 0
        masks_accepted = False
        while not masks_accepted:
            res = self.seg_model.predict(color)[0]
            if not res:
                self.get_logger().warn("No masks from SAM, retrying...")
                continue

            candidates = []
            for r in res:
                for idx, c in enumerate(r):
                    mask = np.zeros((H,W), np.uint8)
                    contour = c.masks.xy.pop().astype(np.int32).reshape(-1,1,2)
                    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
                    candidates.append({
                        'mask': mask,
                        'contour': contour,
                        'box': c.boxes.xyxy.tolist().pop(),
                        'id': f"sam_obj_{idx}"
                    })
            if not candidates:
                self.get_logger().warn("SAM found no objects, retrying...")
                continue

            temp_selected = []
            def click_event(event, x, y, flags, params):
                if event == cv2.EVENT_MOUSEMOVE:
                    hits = [(i,obj) for i,obj in enumerate(candidates)
                            if cv2.pointPolygonTest(obj['contour'], (x,y), False) >= 0]
                    self._hover_idx = hits[0][0] if hits else None
                    refresh()
                elif event == cv2.EVENT_LBUTTONDOWN:
                    clicked = [o for o in candidates if o['mask'][y,x] == 255]
                    if not clicked: return
                    # Choose the smallest mask to avoid selecting a large background object like steel_plate
                    chosen = min(clicked, key=lambda o: np.sum(o['mask']))

                    # clicked = [o for o in candidates if o['mask'][y,x]==255]
                    # if not clicked: return
                    # chosen = max(clicked,
                    #              key=lambda o: cv2.pointPolygonTest(o['contour'], (x,y), True))
                    if self.current_mesh_index < len(self.initial_meshes):
                        mesh = self.initial_meshes[self.current_mesh_index]
                        bounds = self.initial_bounds[self.current_mesh_index]
                        bbox   = self.initial_bboxes[self.current_mesh_index]
                        name   = os.path.splitext(os.path.basename(self.mesh_files[self.current_mesh_index]))[0]
                        pe = FoundationPose(
                            model_pts=mesh.vertices,
                            model_normals=mesh.vertex_normals,
                            mesh=mesh,
                            scorer=self.scorer,
                            refiner=self.refiner,
                            glctx=self.glctx
                        )
                        temp_selected.append({
                            'mask': chosen['mask'],
                            'contour': chosen['contour'],
                            'pose_est': pe,
                            'to_origin': bounds[0],
                            'bbox': bbox,
                            'object_name': name
                        })
                        self.current_mesh_index += 1
                        self.get_logger().info(f"Assigned {name} ({len(temp_selected)})")
                        refresh()

            def refresh():
                #disp = color.copy()
                disp = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)  # match OpenCV expectations
                for o in candidates:
                    cv2.drawContours(disp, [o['contour']], -1, (0,255,0), 2)
                if self._hover_idx is not None:
                    cnt = candidates[self._hover_idx]['contour']
                    cv2.drawContours(disp, [cnt], -1, (0,255,255), 3)
                for sel in temp_selected:
                    cv2.drawContours(disp, [sel['contour']], -1, (255,0,0), 2)

                next_name = "N/A"
                if self.current_mesh_index < len(self.mesh_files):
                    next_name = os.path.splitext(os.path.basename(self.mesh_files[self.current_mesh_index]))[0]

                dlg = (
                    f"Next: {next_name}\n"
                    "- Click mask to assign\n"
                    "- c/Enter/Space to confirm\n"
                    "- r to redo\n"
                    "- q to quit"
                )
                y0, dy = 30, 20
                for i,ln in enumerate(dlg.split('\n')):
                    cv2.putText(disp, ln, (10, y0+i*dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255,255,255),1,cv2.LINE_AA)

                cv2.imshow('Click on objects to track', disp)
                cv2.setMouseCallback('Click on objects to track', click_event)

            refresh()

            while True:
                key = cv2.waitKey(0)
                if key == ord('r'):
                    self.get_logger().info("Redoing selection round.")
                    temp_selected.clear()
                    self.current_mesh_index = 0
                    break
                if key in (ord('q'), 27):
                    self.get_logger().info("Quitting.")
                    rclpy.shutdown()
                    sys.exit(0)
                if key in (ord('c'), 13, 32) and temp_selected:
                    self.selected_mesh_info = temp_selected.copy()
                    masks_accepted = True
                    break

        cv2.destroyWindow('Click on objects to track')
        self.get_logger().info("Mask selection confirmed.")

        vis_img = color.copy()
        published = False
        for od in self.selected_mesh_info:
            pe = od['pose_est']
            if not pe.is_registered_once:
                self.get_logger().info(f"Registering {od['object_name']}")
                pose = pe.register(
                    K=self.cam_K, rgb=color,
                    depth=depth, ob_mask=od['mask'],
                    iteration=args.est_refine_iter
                )
                if pose is not None:
                    obj_in_cam = pose @ np.linalg.inv(od['to_origin'])
                    self.publish_object_pose_in_camera_frame(
                        obj_in_cam,
                        self.camera_frame_id,
                        f"{od['object_name']}_frame"
                    )
                    self.publish_object_pose_in_base_frame(
                        obj_in_cam,
                        f"{od['object_name']}_frame"
                    )
                    vis_img = self.visualize_pose(vis_img, obj_in_cam, od['bbox'])
                    published = True
        if published:
            self.visualization_image = vis_img
            self.waiting_for_next_command = True
            cv2.namedWindow('Pose Estimation Result', cv2.WINDOW_NORMAL)
            #cv2.imshow('Pose Estimation Result', vis_img[..., ::-1])
            cv2.imshow('Pose Estimation Result', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)  # force refresh
            self.get_logger().info("Press 'n' for next, 'q' to quit.")
        else:
            self.get_logger().warn("No poses published.")
            self.state = "READY_FOR_SELECTION"

    def visualize_pose(self, image, center_pose, bbox):
        vis = draw_posed_3d_box(self.cam_K, img=image,
                                ob_in_cam=center_pose,
                                bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose,
                            scale=0.1, K=self.cam_K,
                            thickness=3, transparency=0,
                            is_input_rgb=True)
        return vis

    def publish_object_pose_in_camera_frame(
        self, object_pose_matrix, camera_frame_id, object_frame_id):
        
        now = self.get_clock().now().to_msg()

        # PoseStamped
        msg = PoseStamped()
        msg.header.stamp = now
        msg.header.frame_id = camera_frame_id

        pos = object_pose_matrix[:3, 3]
        quat = R.from_matrix(object_pose_matrix[:3, :3]).as_quat()

        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        #self.object_pose_camera_publisher.publish(msg)
        self.object_pose_camera_publisher.publish(msg)


        self.get_logger().info(f"Published pose in camera frame: {object_frame_id}")

        # Static TF
        static_tf = TransformStamped()
        static_tf.header.stamp = now
        static_tf.header.frame_id = camera_frame_id
        static_tf.child_frame_id = object_frame_id

        static_tf.transform.translation.x = pos[0]
        static_tf.transform.translation.y = pos[1]
        static_tf.transform.translation.z = pos[2]

        static_tf.transform.rotation.x = quat[0]
        static_tf.transform.rotation.y = quat[1]
        static_tf.transform.rotation.z = quat[2]
        static_tf.transform.rotation.w = quat[3]

        self.static_tf_broadcaster.sendTransform(static_tf)
        self.get_logger().info(f"Published static TF: {camera_frame_id} → {object_frame_id}")


    def publish_object_pose_in_base_frame(
        self, object_pose_matrix, object_frame_id):

        T_base_to_cam = np.array([
            [ 0.98903289, -0.14725002,  0.01146187, -0.0321638 ],
            [-0.14722002, -0.98909784, -0.00342339,  0.67219865],
            [ 0.01184101,  0.00169843, -0.99992845,  0.66978157],
            [0.0, 0.0, 0.0, 1.0]
        ])

        T_obj = T_base_to_cam @ object_pose_matrix

        translation = T_obj[:3, 3]

        # Force Z-up
        z_axis = np.array([0, 0, -1])
        original_x = T_obj[:3, 0]
        x_axis = original_x - np.dot(original_x, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Rebuild transform
        T_obj_in_base = np.eye(4)
        T_obj_in_base[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        T_obj_in_base[:3, 3] = translation



        now = self.get_clock().now().to_msg()

        msg = PoseStamped()
        msg.header.stamp = now
        msg.header.frame_id = 'base_link'

        pos = T_obj_in_base[:3, 3]
        quat = R.from_matrix(T_obj_in_base[:3, :3]).as_quat()
        print(f"[DEBUG] Object position in base: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        #self.object_pose_base_publisher.publish(msg)
        self.object_pose_base_publisher.publish(msg)

        self.get_logger().info(f"Published pose in base frame: {object_frame_id}")

        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = 'base_link'
        tf_msg.child_frame_id = f"{object_frame_id}_base"

        tf_msg.transform.translation.x = pos[0]
        tf_msg.transform.translation.y = pos[1]
        tf_msg.transform.translation.z = pos[2]

        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]

        self.static_tf_broadcaster.sendTransform(tf_msg)
        self.get_logger().info(f"Published static TF: base_link → {object_frame_id}_base")

        # Create Pose message
        pose_msg = Pose()
        pose_msg.position.x = pos[0]
        pose_msg.position.y = pos[1]
        pose_msg.position.z = pos[2]
        pose_msg.orientation.x = quat[0]
        pose_msg.orientation.y = quat[1]
        pose_msg.orientation.z = quat[2]
        pose_msg.orientation.w = quat[3]

        self.object_pose_for_path_planning_pub.publish(pose_msg)
        self.get_logger().info("Published pose for path planning topic.")

        



    def keypress_handler(self):
        if self.waiting_for_next_command:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('n'):
                self.get_logger().info("Next round.")
                cv2.destroyWindow('Pose Estimation Result')
                self.waiting_for_next_command = False
                self.state = "READY_FOR_SELECTION"
                self.pending_frame = False  # wait for fresh frame
            elif key == ord('q'):
                self.get_logger().info("Quitting.")
                cv2.destroyAllWindows()
                rclpy.shutdown()
                sys.exit(0)


def main(args=None):
    source_directory = "demo_data"
    file_paths = glob.glob(
        os.path.join(source_directory, '**', '*.obj'), recursive=True
    ) + glob.glob(
        os.path.join(source_directory, '**', '*.stl'), recursive=True
    ) + glob.glob(
        os.path.join(source_directory, '**', '*.STL'), recursive=True
    )

    if not file_paths:
        print(f"No .obj/.stl in '{source_directory}'.")
        return

    new_file_paths = rearrange_files(file_paths)
    if not new_file_paths:
        print("No files selected. Exiting.")
        return

    rclpy.init(args=args)
    node = PoseEstimationNode(new_file_paths)
    node.create_timer(0.1, node.process_images_main_loop)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
