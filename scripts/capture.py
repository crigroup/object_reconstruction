#!/usr/bin/env python
import os
import sys
import copy
import math
import rospy
import criros
import argparse
import textwrap
import numpy as np
import dynamic_reconfigure.client
# OpenCV
import cv2
# PCL
import pcl
from ros_numpy.point_cloud2 import get_xyz_points, pointcloud2_to_array
# Python ensenso snatcher
from ensenso.snatcher import Snatcher
# Messages
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import (
  Image, 
  PointCloud2
)


class CaptureServer(Snatcher):
  def __init__(self, options):
    super(CaptureServer, self).__init__()
    # Config stuff
    np.set_printoptions(precision=6, suppress=True)
    self.nviews = options.nviews
    # Setup publishers and subscribers
    self.dbg_pub = rospy.Publisher('debug/image_color', Image, queue_size=1)
    rospy.Subscriber('pattern/pose', PoseStamped, self.cb_pattern_pose)
    rospy.on_shutdown(self.on_shutdown)
  
  def cb_pattern_pose(self, msg):
    self.pattern_pose = criros.conversions.from_pose(msg.pose)
    self.headers['pattern_pose'] = copy.deepcopy(msg.header)
  
  def good_msgs_headers(self):
    required_keys = ['raw_left', 'raw_right', 'rect_left', 'rect_right', 'pattern_pose']
    if not criros.utils.has_keys(self.headers, required_keys):
      return False
    stamps = []
    frames = []
    for header in self.headers.values():
      stamps.append( header.stamp.to_sec() )
      frames.append( header.frame_id )
    all_frames_equal = frames.count(frames[0]) == len(frames)
    all_stamps_equal = np.all([ np.isclose(stamps[0],x,atol=1e-3) for x in stamps ])
    return (all_frames_equal and all_stamps_equal)
  
  def execute(self):
    # Start streaming
    self.enable_lights(projector=True, frontlight=False)
    self.enable_streaming(cloud=False, images=True)
    rospy.sleep(2.0)
    taken = 0
    while taken < self.nviews and not rospy.is_shutdown():
      self.reset_snapshots()
      self.take_snapshot(exposure_time=0.001, success_fn=self.has_images)
      if not self.good_msgs_headers():
        rospy.logdebug('The headers of the images and pattern must be equal')
        continue
      #~ # Debugging image
      #~ dbg_image = cv2.cvtColor(self.rect_left, cv2.COLOR_GRAY2RGB)
      #~ self.dbg_pub.publish( self.bridge.cv2_to_imgmsg(dbg_image, 'rgb8') )
  
  def on_shutdown(self):
    # Stop streaming
    self.enable_lights(projector=False, frontlight=False)
    self.enable_streaming(cloud=False, images=False)


def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=textwrap.dedent(
"""Captures data appropriate for training object recognition pipelines.
Assumes that there is a known fiducial in the scene, and captures views of the 
object sparsely, depending on the angle_thresh setting."""),
                                  fromfile_prefix_chars='@')
  parser.add_argument('-o', '--output', metavar='BAG_FILE', dest='bag', type=str,
                      default='',
                      help='A bagfile to write to.')
  parser.add_argument('-a', '--angle_thresh', metavar='RADIANS', dest='angle_thresh', type=float,
                      default= 10 * math.pi / 180,
                      help='The delta angular threshold in pose.'
                           'Frames will not be recorded unless they are not closer to any other pose by this amount. default=%(default).2f')
  parser.add_argument('-n', '--nviews', metavar='NVIEWS', dest='nviews', type=int,
                      default=36,
                      help='Number of desired views. default=%(default)d')
  parser.add_argument('-p', '--preview', dest='preview', action='store_true',
                      default=False, help='Preview the pose estimator.')
  parser.add_argument('--debug', action='store_true',     
                      help='If set, will show additional debugging information')
  args = parser.parse_args(rospy.myargv()[1:])
  if not args.preview and len(args.bag) < 1:
    parser.print_help()
    print '\nYou must supply a bag name, or run in --preview mode'
    sys.exit(1)
  return args

if "__main__" == __name__:
  options = parse_args()
  log_level= rospy.DEBUG if options.debug else rospy.INFO
  node_name = os.path.splitext(os.path.basename(__file__))[0]
  rospy.init_node(node_name, log_level=log_level)
  server = CaptureServer(options)
  server.execute()
