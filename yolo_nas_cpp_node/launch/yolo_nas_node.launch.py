# Copyright 2023 Ar-Ray-code
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    container = ComposableNodeContainer(
        name='yolo_nas_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='v4l2_camera',
                plugin='v4l2_camera::V4L2Camera',
                name='v4l2_camera',
                parameters=[{
                    'video_device': LaunchConfiguration('video_device'),
                    'image_size': [640, 480]
                }]),
            ComposableNode(
                package='yolo_nas_cpp_node',
                plugin='yolo_nas_cpp_node::YoloNasNode',
                name='yolo_nas_cpp_node',
                parameters=[{
                    'imshow_isshow': True
                }]
                ),
        ],
        output='screen',
    )

    ld = LaunchConfiguration
    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video0'
        ),
        container
    ])