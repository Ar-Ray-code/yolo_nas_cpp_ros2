#include "yolo_nas_cpp/yolo_nas.hpp"

namespace yolo_nas_cpp
{

YoloNas::YoloNas(std::string netPath, bool cuda, json &prepSteps, std::vector<int> imgsz, float score, float iou, std::vector<std::string> &labels)
{
    net = cv::dnn::readNetFromONNX(netPath);
    if (cuda && cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        std::cout << LogInfo("Backend", "Attempting to use CUDA") << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    netInputShape[3] = imgsz[0];
    netInputShape[2] = imgsz[1];

    scoreThresh = score;
    iouThresh = iou;
    classLabels = labels;

    preprocess = PreProcessing(prepSteps, imgsz);
    postprocess = PostProcessing(prepSteps, score, iou);

    warmup(3);
}

void YoloNas::warmup(int round)
{
    cv::Mat mat(4, netInputShape, CV_32F);
    std::vector<std::vector<cv::Mat>> out;
    for (int i = 0; i < round; i++)
    {
        randu(mat, cv::Scalar(0), cv::Scalar(1));
        net.setInput(mat);
        net.forward(out, net.getUnconnectedOutLayersNames());
    }

    mat.release();
    out[0][0].release();
    out[1][0].release();
}

std::vector<Object> YoloNas::predict(cv::Mat &img)
{
    cv::Mat imgInput;
    std::vector<Object> out_obj;
    json metadata = preprocess.run(img, imgInput);

    std::vector<std::vector<cv::Mat>> out;
    net.setInput(imgInput);
    net.forward(out, net.getUnconnectedOutLayersNames());
    imgInput.release();

    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    std::vector<int> labels, selectedIDX;

    postprocess.run(out, boxes, labels, scores, selectedIDX, metadata);

    for (auto &x : selectedIDX)
    {
        int classID = labels[x];
        float score = scores[x];
        cv::Rect_<float> rect = boxes[x];
        Object obj;
        obj.label = classID;
        obj.prob = score;
        obj.rect = rect;
        out_obj.push_back(obj);
    }
    return out_obj;
}

} // namespace yolo_nas_cpp
