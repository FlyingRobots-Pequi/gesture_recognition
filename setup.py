from setuptools import setup
import os
import glob

package_name = 'gesture_recognition'

setup(
 name=package_name,
 version='0.0.0',
 packages=[package_name],
 data_files=[
     ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
     ('share/' + package_name, ['package.xml']),
     (os.path.join('share', package_name), [os.path.join(package_name, 'conv1d.pth')]),
   ],
 install_requires=['setuptools'],
 zip_safe=True,
 maintainer='luisa',
 maintainer_email='luisafrancielle@todo.todo',
 description='The gesture_detected package for ROS 2',
 license='Apache-2.0',
 tests_require=['pytest'],
 entry_points={
     'console_scripts': [
             'gesture_detection = gesture_recognition.gesture_detection:main',
             'gesture_webcam = gesture_recognition.gesture_detection_webcam:main'
     ],
   },
)