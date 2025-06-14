from setuptools import setup

package_name = 'gesture_detected'

setup(
 name=package_name,
 version='0.0.0',
 packages=[package_name],
 data_files=[
     ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
     ('share/' + package_name, ['package.xml']),
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
             'gesture_detected = gesture_detected.gesture_detected:main'
     ],
   },
)