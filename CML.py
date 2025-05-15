#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from mr_voice.msg import Voice
import numpy as np
from RobotChassis import RobotChassis
from geometry_msgs.msg import Twist
import cv2
from ultralytics import YOLO
from pcms.openvino_models import HumanPoseEstimation
import time


# Global Variables
_image = None
_depth = None
voice_text = ""
voice_direction = 0
last_voice_time = 0
asked_bag = False
turn_start_time = None
hund = ""


def callback_depth(msg):
   """Callback function for depth image."""
   global _depth
   try:
       tmp = CvBridge().imgmsg_to_cv2(msg, "passthrough")
       _depth = np.array(tmp, dtype=np.float32)
   except Exception as e:
       rospy.logerr(f"Failed to process depth image: {e}")


def callback_image(msg):
   """Callback function for RGB image."""
   global _image
   _image = CvBridge().imgmsg_to_cv2(msg, 'bgr8')


def callback_voice(msg):
   """Callback function for voice commands."""
   global voice_text, voice_direction, last_voice_time
   voice_text = msg.text
   voice_direction = msg.direction
   last_voice_time = rospy.get_time()


def get_real_xyz(dp, x, y):
   """Calculate real-world coordinates from depth."""
   a = 49.5 * np.pi / 180
   b = 60.0 * np.pi / 180
   d = dp[y][x]
   h, w = dp.shape[:2]
   x = int(x) - int(w // 2)
   y = int(y) - int(h // 2)
   real_y = round(y * 2 * d * np.tan(a / 2) / h)
   real_x = round(x * 2 * d * np.tan(b / 2) / w)
   return real_x, real_y, d


def is_pointing_to_box(finger, box):
   """Check if a finger is pointing to a box."""
   box_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
   distance = np.linalg.norm(np.array(finger) - np.array(box_center))
   return distance


if __name__ == "__main__":
   rospy.init_node("ros_tutorial")
   rospy.loginfo("Node started!")


   # Subscribers
   rospy.Subscriber('/camera/rgb/image_raw', Image, callback_image)
   rospy.Subscriber('/camera/depth/image_raw', Image, callback_depth)
   rospy.Subscriber('/voice/text', Voice, callback_voice)


   # Publishers
   pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
   pub_say = rospy.Publisher('/speaker/say', String, queue_size=10, latch=True)


   # Command and models
   cmd = Twist()
   pose_model = HumanPoseEstimation()
   model = YOLO("/home/pcms/Downloads/green_chair.pt")
   chassis = RobotChassis()


   # State and Logic Variables
   rate = rospy.Rate(10)
   rospy.sleep(1)
   target_found_once = False
   wait_start_time = rospy.get_time()
   step = 0
   name = None
   drink = None
   turn_angle = 0
   turning = False
   turn_start_time = None


   # Predefined lists
   nameList = ["adam", "oxo", "axel", "chris", "hunter", "jack", "max", "paris", "robin", "olivia", "william"]
   drinkList = ["coke", "coffee", "cocoa", "lemonade", "coconut milk", "orange juice", "black tea", "wine", "green tea", "soda", "water"]


   while not rospy.is_shutdown():
       if _image is None or _depth is None:
           rospy.logwarn("Waiting for image and depth data...")
           rate.sleep()
           continue


       frame = _image.copy()
       depth = _depth.copy()


       if depth is None or depth.size == 0:
           rospy.logwarn("Depth data is missing or invalid!")
           rate.sleep()
           continue


       poses = pose_model.forward(frame)


       if poses is not None:
           tx, ty, td = -1, -1, -1
           for pose in poses:
               x8, y8, c8 = map(int, pose[12])
               x11, y11, c11 = map(int, pose[11])
               x, y = (x8 + x11) // 2, (y8 + y11) // 2
               if td == -1 or (depth[y][x] > 0 and depth[y][x] < td):
                   tx, ty, td = x, y, depth[y][x]


           if td != -1:
               cv2.circle(frame, (tx, ty), 12, (0, 255, 0), -1)


       # Object detection
       boxes = model(frame)[0].boxes
       for conf, xyxy in zip(boxes.conf, boxes.xyxy):
           if conf < 0.5:
               continue
           x1, y1, x2, y2 = map(int, xyxy)
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


       # Step logic
       h, w = depth.shape[:2]
       middle_w, middle_h = w // 2, h // 2
       deepdistance = depth[middle_h, middle_w]
       if np.isnan(deepdistance):
           rospy.logwarn("Invalid depth reading!")
           continue


       if step == 0 and 100 < deepdistance < 1200:
           rospy.loginfo("Detected a person within range.")
           pub_say.publish("WELCOME! What is your name?")
           step = 1


       elif step == 1:
           if voice_text:
               for name_candidate in nameList:
                   if name_candidate in voice_text.lower():
                       name = name_candidate
                       pub_say.publish(f"Is your name {name}?")
                       step = 2
                       break
               voice_text = ""


       elif step == 2:
           if "yes" in voice_text.lower():
               pub_say.publish(f"Hi, {name}, what is your favorite drink?")
               voice_text = ""
               step = 3
           elif "no" in voice_text.lower():
               pub_say.publish("Okay, then what is your name?")
               voice_text = ""
               step = 1


       elif step == 3:
           if voice_text:
               for drink_candidate in drinkList:
                   if drink_candidate in voice_text.lower():
                       drink = drink_candidate
                       pub_say.publish(f"Is your favorite drink {drink}?")
                       step = 4
                       break
               voice_text = ""


       elif step == 4:
           if "yes" in voice_text.lower():
               pub_say.publish(f"Okay, {name}, please follow me.")
               voice_text = ""
               step = 8


       elif step == 8:
           rospy.loginfo(f"Moving with {name} to a location.")
           chassis.move_to(3.08, -0.929, 0.00501)
          
           # Start following the person while moving to destination
           following_person = True
           last_follow_time = rospy.get_time()
           step = 8.5  # Intermediate step for following
          
       elif step == 8.5:
           # Follow the person using YOLO detection
           if poses is not None and len(poses) > 0:
               # Find the closest person
               min_distance = float('inf')
               target_pose = None
               for pose in poses:
                   x8, y8, c8 = map(int, pose[12])  # Right wrist
                   x11, y11, c11 = map(int, pose[11])  # Left wrist
                   x, y = (x8 + x11) // 2, (y8 + y11) // 2
                  
                   if 0 <= x < w and 0 <= y < h:
                       distance = depth[y][x]
                       if distance > 0 and distance < min_distance:
                           min_distance = distance
                           target_pose = pose
              
               if target_pose is not None:
                   x8, y8, c8 = map(int, target_pose[12])
                   x11, y11, c11 = map(int, target_pose[11])
                   x, y = (x8 + x11) // 2, (y8 + y11) // 2
                  
                   # Calculate position relative to center
                   error_x = x - w // 2
                   error_z = min_distance - 1000  # Target distance in mm
                  
                   # Simple proportional control
                   cmd.angular.z = -0.002 * error_x
                   cmd.linear.x = 0.0005 * error_z
                  
                   # Limit speeds
                   cmd.angular.z = max(-0.5, min(0.5, cmd.angular.z))
                   cmd.linear.x = max(-0.2, min(0.2, cmd.linear.x))
                  
                   last_follow_time = rospy.get_time()
                  
           # Check if we've arrived at destination or lost the person
           if chassis.status_code == 3:  # Arrived at destination
               step = 9
               # Stop the robot
               cmd.linear.x = 0
               cmd.angular.z = 0
               pub_cmd.publish(cmd)
           elif rospy.get_time() - last_follow_time > 3.0:  # Lost person for 3 seconds
               pub_say.publish(f"{name}, I lost you. Please come closer.")
               step = 8  # Go back to step 8 to restart movement


       elif step == 9:
           rospy.loginfo("Starting 180-degree turn and person positioning")
          
           # First, stop any movement
           cmd.linear.x = 0
           cmd.angular.z = 0
           pub_cmd.publish(cmd)
          
           # Check if we see the person
           person_centered = False
           person_distance_ok = False
           person_detected = False
          
           if poses is not None and len(poses) > 0:
               # Find the closest person
               min_distance = float('inf')
               target_pose = None
               for pose in poses:
                   x8, y8, c8 = map(int, pose[12])
                   x11, y11, c11 = map(int, pose[11])
                   x, y = (x8 + x11) // 2, (y8 + y11) // 2
                  
                   if 0 <= x < w and 0 <= y < h:
                       distance = depth[y][x]
                       if distance > 0 and distance < min_distance:
                           min_distance = distance
                           target_pose = pose
              
               if target_pose is not None:
                   person_detected = True
                   x8, y8, c8 = map(int, target_pose[12])
                   x11, y11, c11 = map(int, target_pose[11])
                   x, y = (x8 + x11) // 2, (y8 + y11) // 2
                  
                   # Check if person is centered
                   error_x = x - w // 2
                   if abs(error_x) < 50:  # Within 50 pixels of center
                       person_centered = True
                  
                   # Check if person is at good distance (800-1200mm)
                   if 800 < min_distance < 1200:
                       person_distance_ok = True
          
           # If we don't see the person, perform 180-degree turn
           if not person_detected:
               if not turning:
                   turning = True
                   turn_start_time = rospy.get_time()
                   cmd.angular.z = 0.5  # Turn at 0.5 rad/s (about 28.6 degrees/sec)
                   pub_cmd.publish(cmd)
                   rospy.loginfo("Starting 180-degree turn to find person")
               else:
                   # Check if turn is complete (180 degrees at 0.5 rad/s = ~3.14/0.5 = 6.28 sec)
                   if rospy.get_time() - turn_start_time > 3.14/0.5:
                       turning = False
                       cmd.angular.z = 0
                       pub_cmd.publish(cmd)
                       rospy.loginfo("180-degree turn complete")
           else:
               # If we see the person but not centered or at wrong distance, adjust
               if not person_centered or not person_distance_ok:
                   if target_pose is not None:
                       x8, y8, c8 = map(int, target_pose[12])
                       x11, y11, c11 = map(int, target_pose[11])
                       x, y = (x8 + x11) // 2, (y8 + y11) // 2
                      
                       # Center the person
                       error_x = x - w // 2
                       cmd.angular.z = -0.002 * error_x
                      
                       # Adjust distance
                       if min_distance < 800:
                           cmd.linear.x = -0.1  # Move backward
                       elif min_distance > 1200:
                           cmd.linear.x = 0.1   # Move forward
                       else:
                           cmd.linear.x = 0
                      
                       pub_cmd.publish(cmd)
               else:
                   # Person is centered and at good distance
                   cmd.linear.x = 0
                   cmd.angular.z = 0
                   pub_cmd.publish(cmd)
                   rospy.loginfo("Person properly positioned")
                   pub_say.publish(f"This is {name}, and their favorite drink is {drink}.")
                   step = 10


       elif step == 10:
           # Final state, do nothing
           pass


       # Publish commands and display
       pub_cmd.publish(cmd)
       cv2.imshow("FSM Tracking", frame)
       if cv2.waitKey(1) in [27, ord('q')]:
           break


       rate.sleep()


   rospy.loginfo("Node shutting down.")
   cv2.destroyAllWindows()
