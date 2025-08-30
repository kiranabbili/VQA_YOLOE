#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import clip
import torch
import supervision as sv
from ultralytics import YOLOE
from PIL import Image
import numpy as np
import cv2

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge


class YOLOENode(Node):
    def __init__(self):
        super().__init__('yoloe_node')

        # Params
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("classes", ["truck", "person","dog", "bicycle", "tree"])
        self.declare_parameter("weights", "yoloe-v8l-seg.pt")

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.names = self.get_parameter("classes").get_parameter_value().string_array_value
        weights_path = self.get_parameter("weights").get_parameter_value().string_value

        # Load YOLOE model
        self.model = YOLOE(weights_path).cpu()
        self.model.set_classes(self.names, self.model.get_text_pe(self.names))
        self.get_logger().info(f"Loaded YOLOE model with classes: {self.names}")

        # CV bridge
        self.bridge = CvBridge()

        # ROS2 I/O
        self.subscription = self.create_subscription(
            ROSImage,
            image_topic,
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(ROSImage, "/yoloe/annotated_image", 10)

        # Annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def image_callback(self, msg: ROSImage):
        # Convert ROS image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run YOLOE
        results = self.model.predict(pil_img, conf=0.1, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        # Annotate
        annotated_img = pil_img.copy()
        annotated_img = self.box_annotator.annotate(scene=annotated_img, detections=detections)
        annotated_img = self.label_annotator.annotate(scene=annotated_img, detections=detections)

        # Convert back to ROS Image
        annotated_cv = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
        ros_img = self.bridge.cv2_to_imgmsg(annotated_cv, encoding="bgr8")

        # Publish
        self.publisher.publish(ros_img)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOENode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
