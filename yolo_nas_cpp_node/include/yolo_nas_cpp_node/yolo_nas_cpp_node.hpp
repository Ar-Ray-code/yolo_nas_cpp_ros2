// Copyright 2023 Ar-Ray-code
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cmath>
#include <chrono>

#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>


#include "bboxes_ex_msgs/msg/bounding_box.hpp"
#include "bboxes_ex_msgs/msg/bounding_boxes.hpp"

#include "yolo_nas_cpp/yolo_nas.hpp"
#include "yolo_nas_cpp/utils.hpp"
#include "yolo_nas_param/yolo_nas_param.hpp"

namespace yolo_nas_cpp_node{

    class YoloNasNode : public rclcpp::Node
    {
    public:
        YoloNasNode(const rclcpp::NodeOptions&);

    protected:
        std::shared_ptr<yolo_nas_parameters::ParamListener> param_listener_;
        yolo_nas_parameters::Params params_;
    private:
        void onInit();
        rclcpp::TimerBase::SharedPtr init_timer_;

        std::unique_ptr<yolo_nas_cpp::YoloNas> yolo_nas_;
        std::vector<std::string> class_names_;

        image_transport::Subscriber sub_image_;
        void colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr&);

        rclcpp::Publisher<bboxes_ex_msgs::msg::BoundingBoxes>::SharedPtr pub_bboxes_;
        image_transport::Publisher pub_image_;

        bboxes_ex_msgs::msg::BoundingBoxes objects_to_bboxes(cv::Mat, std::vector<yolo_nas_cpp::Object>, std_msgs::msg::Header);
    };
}