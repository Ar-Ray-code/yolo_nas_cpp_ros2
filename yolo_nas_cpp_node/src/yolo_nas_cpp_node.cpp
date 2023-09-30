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

#include "yolo_nas_cpp_node/yolo_nas_cpp_node.hpp"

namespace yolo_nas_cpp_node
{
    YoloNasNode::YoloNasNode(const rclcpp::NodeOptions &options)
        : Node("yolo_nas_cpp_node", options)
    {
        using namespace std::chrono_literals; // NOLINT
        this->init_timer_ = this->create_wall_timer(
            0s, std::bind(&YoloNasNode::onInit, this));
    }

    void YoloNasNode::onInit()
    {
        this->init_timer_->cancel();
        this->param_listener_ = std::make_shared<yolo_nas_parameters::ParamListener>(
            this->get_node_parameters_interface());

        this->params_ = this->param_listener_->get_params();

        if (this->params_.imshow_isshow)
        {
            cv::namedWindow("yolo_nas", cv::WINDOW_AUTOSIZE);
        }

        if (this->params_.class_labels_path != "")
        {
            RCLCPP_INFO(this->get_logger(), "read class labels from '%s'", this->params_.class_labels_path.c_str());
            this->class_names_ = yolo_nas_cpp::read_class_labels_file(this->params_.class_labels_path);
        }
        else
        {
            this->class_names_ = yolo_nas_cpp::COCO_LABELS;
        }

        nlohmann::json prepSteps_null;
        std::vector<int> imgsz = {(int)this->params_.model_width, (int)this->params_.model_height};
        this->yolo_nas_ = std::make_unique<yolo_nas_cpp::YoloNas>(
            this->params_.model_path,
            false,
            prepSteps_null,
            imgsz,
            this->params_.nms,
            this->params_.conf,
            this->class_names_);

        RCLCPP_INFO(this->get_logger(), "model loaded");

        this->sub_image_ = image_transport::create_subscription(
            this, this->params_.src_image_topic_name,
            std::bind(&YoloNasNode::colorImageCallback, this, std::placeholders::_1),
            "raw");
        this->pub_bboxes_ = this->create_publisher<bboxes_ex_msgs::msg::BoundingBoxes>(
            this->params_.publish_boundingbox_topic_name,
            10);
        this->pub_image_ = image_transport::create_publisher(this, this->params_.publish_image_topic_name);
    }

    void YoloNasNode::colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &ptr)
    {
        auto img = cv_bridge::toCvCopy(ptr, "bgr8");
        cv::Mat frame = img->image;

        auto now = std::chrono::system_clock::now();
        auto objects = this->yolo_nas_->predict(frame);

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - now);
        RCLCPP_INFO(this->get_logger(), "Inference: %f FPS", 1000.0f / elapsed.count());

        yolo_nas_cpp::draw_objects(frame, objects, this->class_names_);
        if (this->params_.imshow_isshow)
        {
            cv::imshow("yolo_nas", frame);
            auto key = cv::waitKey(1);
            if (key == 27)
            {
                rclcpp::shutdown();
            }
        }

        auto boxes = objects_to_bboxes(frame, objects, img->header);
        this->pub_bboxes_->publish(boxes);

        sensor_msgs::msg::Image::SharedPtr pub_img;
        pub_img = cv_bridge::CvImage(img->header, "bgr8", frame).toImageMsg();
        this->pub_image_.publish(pub_img);
    }


    bboxes_ex_msgs::msg::BoundingBoxes YoloNasNode::objects_to_bboxes(cv::Mat frame, std::vector<yolo_nas_cpp::Object> objects, std_msgs::msg::Header header)
    {
        bboxes_ex_msgs::msg::BoundingBoxes boxes;
        boxes.header = header;
        for (auto obj : objects)
        {
            bboxes_ex_msgs::msg::BoundingBox box;
            box.probability = obj.prob;
            box.class_id = yolo_nas_cpp::COCO_LABELS[obj.label];
            box.xmin = obj.rect.x;
            box.ymin = obj.rect.y;
            box.xmax = (obj.rect.x + obj.rect.width);
            box.ymax = (obj.rect.y + obj.rect.height);
            box.img_width = frame.cols;
            box.img_height = frame.rows;
            boxes.bounding_boxes.emplace_back(box);
        }
        return boxes;
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_nas_cpp_node::YoloNasNode)