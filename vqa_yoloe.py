import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLOE
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image as PILImage
import torch

class VQAYoloNode(Node):
    def __init__(self):
        super().__init__('vqa_yolo_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Publishers
        self.detection_pub = self.create_publisher(String, '/detections', 10)
        self.answer_pub = self.create_publisher(String, '/vqa/answer', 10)

        # Subscriber for user questions
        self.question_sub = self.create_subscription(
            String, '/vqa/question', self.question_callback, 10)

        # Default question if none received yet
        self.current_question = "What is in the image?"

        # Load YOLOE
        self.model = YOLOE("yoloe-v8l-seg.pt")  # CPU is fine if no GPU
        self.names = ["tree", "dog", "car"]
        self.model.set_classes(self.names, self.model.get_text_pe(self.names))

        # Load BLIP VQA
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    def question_callback(self, msg: String):
        """Callback to update the current VQA question."""
        self.current_question = msg.data
        self.get_logger().info(f"Updated VQA question: {self.current_question}")

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_img = cv2.resize(cv_img, (320, 240))
        pil_img = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        # YOLOE detection
        results = self.model.predict(pil_img, conf=0.2, verbose=False)
        detections = results[0].boxes.cls.cpu().numpy()

        # Publish detections
        det_str = f"Detected: {[self.names[int(c)] for c in detections]}"
        self.detection_pub.publish(String(data=det_str))
        self.get_logger().info(det_str)

        # Run VQA with the latest user-provided question
        inputs = self.processor(pil_img, self.current_question, return_tensors="pt")
        out = self.vqa_model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)

        # Publish answer
        self.answer_pub.publish(String(data=answer))
        self.get_logger().info(f"Q: {self.current_question} | A: {answer}")


def main(args=None):
    rclpy.init(args=args)
    node = VQAYoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
