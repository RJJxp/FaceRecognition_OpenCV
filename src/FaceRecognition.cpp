#include "FaceRecognition.h"

// cpp
#include <cmath>

// yaml
#include "yaml-cpp/yaml.h"

FaceRecognition::FaceRecognition() {

}

FaceRecognition::~FaceRecognition() {

}

int FaceRecognition::loadModel(std::string path) {
    // read path in config
    YAML::Node config = YAML::LoadFile(path);
    std::string face_model_path = config["FaceModelPath"].as<std::string>();
    std::string eyes_model_path = config["EyesModelPath"].as<std::string>();
    std::string nose_model_path = config["NoseModelPath"].as<std::string>();
    std::string mouth_model_path = config["MouthModelPaht"].as<std::string>();

    std::cout << face_model_path << std::endl;
    std::cout << eyes_model_path << std::endl;
    std::cout << nose_model_path << std::endl;
    std::cout << mouth_model_path << std::endl;

    if (!_face_cascade.load(face_model_path)) {
        std::cout << "loading face model failed." << std::endl;
        return 0;
    }

    if (!_eyes_cascade.load(eyes_model_path)) {
        std::cout << "loading eyes model failed." << std::endl;
        return 0;
    }

    if (!_nose_cascade.load(nose_model_path)) {
        std::cout << "loading nose model failed." << std::endl;
        return 0;
    }

    if (!_mouth_cascade.load(mouth_model_path)) {
        std::cout << "loading mouth model failed." << std::endl;
        return 0;
    }

    std::cout << "load all model sucessfully." << std::endl;
    return 1;
}

int FaceRecognition::detectAll(std::string path) {
    cv::Mat frame = cv::imread(path);
    if (frame.empty()) {
        std::cout << "load image failed." << std::endl;
        return -1;
    }

    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;

    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    // --Detect faces
    std::vector<int> reject_level;
    std::vector<double> level_width;
    _face_cascade.detectMultiScale(frame_gray, faces, reject_level, level_width, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
    std::cout << "face size is " << faces.size() << std::endl;
    // std::cout << "reject level size is " << reject_level.size() << std::endl;
    // std::cout << "reject level is " << reject_level[0] << std::endl;
    // std::cout << "reject level width is " << level_width[0] << std::endl;
    
    for( size_t i = 0; i < 1; i++ )
    {
        _roi_img = frame(faces[i]);
        cv::Mat faceROI = frame_gray(faces[i]);
        std::vector<cv::Rect> eyes;

        // -- In each face, detect eyes
        std::vector<int> eyes_reject_level;
        std::vector<double> eyes_level_weight;
        _eyes_cascade.detectMultiScale( faceROI, eyes, eyes_reject_level, eyes_level_weight, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
        std::cout << "eyes: " << eyes.size() << std::endl;

        if (getPreciseEyes(eyes) == 1) {    // return 1 which indicates there are at most 2 eyes detected.
            if (eyes.size() == 1) {
                _eye_01_score.first = eyes_reject_level[0];
                _eye_01_score.second = eyes_level_weight[0];
                _eye_02_score = _eye_01_score;
            } else if (eyes.size() == 2) {
                _eye_01_score.first = eyes_reject_level[0];
                _eye_01_score.second = eyes_level_weight[0];
                _eye_02_score.first = eyes_reject_level[1];
                _eye_02_score.second = eyes_level_weight[1];
            } else {
                // ....
            }
            
        } else {    // reutrn -1, no eyes detected or more than 2 eyes detected.
            // TODO:
            std::cout << "getPresiceEyes failed." << std::endl;
            return -1;
        }

        // detect nose;
        std::vector<int> nose_reject_level;
        std::vector<double> nose_level_weight;
        std::vector<cv::Rect> nose;
        _nose_cascade.detectMultiScale(faceROI, nose, nose_reject_level, nose_level_weight, 1.1, 2 , 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
        std::cout << "nose: " << nose.size() << std::endl;
        
        int good_nose_index = getPreciseNose(nose);
        if(good_nose_index != -1) {
            _nose_score.first = nose_reject_level[good_nose_index];
            _nose_score.second = nose_level_weight[good_nose_index];
        } else {
            std::cout << "getPreciseNose failed." << std::endl;
            return -1;
        }

        // for (size_t j = 0; j < nose.size(); ++j) {
        //     cv::Point center(nose[j].x + nose[j].width*0.5, nose[j].y + nose[j].height*0.5);
        //     int radius = cvRound((nose[j].width + nose[j].height)*0.25);
        //     cv::circle(faceROI_2show, center, radius, cv::Scalar(255, 255, 0), 4, 8, 0);
        //     std::cout << j << "\t nose level is " << nose_reject_level[j] << " weight is " << nose_level_weight[j] << std::endl;
        // }

        // detect mouth
        std::vector<cv::Rect> mouth;
        std::vector<int> mouth_reject_level;
        std::vector<double> mouth_level_weight;
        _mouth_cascade.detectMultiScale(faceROI, mouth, mouth_reject_level, mouth_level_weight, 1.1, 2 , 0 |CV_HAAR_SCALE_IMAGE, cv::Size(30, 30), cv::Size(), true);
        std::cout << "mouth: " << mouth.size() << std::endl;
        int good_mouth_index = getPreciseMouth(mouth);
        if (good_mouth_index != -1) {   // detect sucessfully.
            _mouth_score.first = mouth_reject_level[good_mouth_index];
            _mouth_score.second = mouth_level_weight[good_mouth_index];
        } else {    // detect failde.
            std::cout << "getPreciseMouth failed" << std::endl;
            return -1;
        }

        // for (size_t j = 0; j < mouth.size(); ++j) {
        //     cv::Point center(mouth[j].x + mouth[j].width*0.5, mouth[j].y + mouth[j].height*0.5);
        //     int radius = cvRound( (mouth[j].width + mouth[j].height)*0.25);
        //     cv::circle(faceROI_2show, center, radius, cv::Scalar(0, 0, 255), 4, 8, 0);
        //     std::cout << j << "\t nose level is " << mouth_reject_level[j] << " weight is " << mouth_reject_level[j] << std::endl;
        // }
        // show
        // cv::imshow("face\t" + std::to_string(i), faceROI_2show);
        // cv::waitKey(0);
    }
    std::cout << "detectAll() successfully." << std::endl;
    return 1;
}

int FaceRecognition::getPreciseEyes(const std::vector<cv::Rect>& eyes_rects) {
    int eyes_size = eyes_rects.size();
    switch (eyes_size) {
    case 1: {
        cv::Point eye_01_center(eyes_rects[0].x + eyes_rects[0].width * 0.5, 
                                eyes_rects[0].y + eyes_rects[0].height * 0.5);
        if (eye_01_center.y < _roi_img.rows * 0.5) {    // make sure they are on the top of human face
            _eye_01 = eyes_rects[0];
            _eye_02 = eyes_rects[0];
            return 1;
        } else {
            std::cout << "getPreciseEyes failed." << std::endl;
            return -1;
        }
        break;
    }
    case 2: {
        cv::Point eye_01_center(eyes_rects[0].x + eyes_rects[0].width * 0.5, 
                                eyes_rects[0].y + eyes_rects[1].height * 0.5);
        cv::Point eye_02_center(eyes_rects[1].x + eyes_rects[1].width * 0.5, 
                                eyes_rects[1].y + eyes_rects[1].height * 0.5);
        int diff_y = fabs(eye_01_center.y - eye_02_center.y);    // check whether the 2 eyes are level
        int center_y = (eye_01_center.y + eye_02_center.y) / 2; // make sure they are on the top of human face
        if (diff_y < _roi_img.rows * 0.1 && center_y < _roi_img.rows * 0.5) {
            _eye_01 = eyes_rects[0];
            _eye_02 = eyes_rects[1];
            return 1;
        } else {
            std::cout << "getPreciseEyes failed." << std::endl;
            return -1;
        }
        break;
    }
    default: {
        // TODO: if eyes detection is more than 3, fuck'em
        return -1;
        break;
    }   // swithc---default
    }   // switch
}

// neareset to the mid 
int FaceRecognition::getPreciseNose(const std::vector<cv::Rect>& nose_rects) {
    int roi_width = _roi_img.cols;
    std::vector<double> dist_2_mid;
    for (int i = 0; i < nose_rects.size(); ++i) {
        double this_nose_x = nose_rects[i].x + nose_rects[i].width * 0.5;
        dist_2_mid.push_back(fabs(this_nose_x - _roi_img.cols));
    }
    // find the min index
    auto iter = std::min_element(dist_2_mid.begin(), dist_2_mid.end());
    int min_index = std::distance(dist_2_mid.begin(), iter);
    // whether the nose is under eyes
    double eyes_height = (_eye_01.y + _eye_01.height * 0.5 + _eye_02.y + _eye_02.height * 0.5) * 0.5;
    double good_nose_height = nose_rects[min_index].y + nose_rects[min_index].height * 0.5;
    if (good_nose_height > eyes_height) {
        _nose = nose_rects[min_index];
        return min_index;
    } else {
        return -1;
    }
}

int FaceRecognition::getPreciseMouth(const std::vector<cv::Rect>& mouth_rects) {
    int roi_width = _roi_img.cols;
    std::vector<double> dist_2_mid;
    for (int i = 0; i < mouth_rects.size(); ++i) {
        double this_mouth_x = mouth_rects[i].x + mouth_rects[i].width * 0.5;
        dist_2_mid.push_back(fabs(this_mouth_x - _roi_img.cols));
    }
    // find the min index
    auto iter = std::min_element(dist_2_mid.begin(), dist_2_mid.end());
    int min_index = std::distance(dist_2_mid.begin(), iter);
    double nose_height = _nose.y + _nose.height * 0.5;
    double good_mouth_height = mouth_rects[min_index].x + mouth_rects[min_index].height * 0.5;
    if (good_mouth_height > nose_height) {
        return min_index;
    } else {
        return -1;
    }
}

void FaceRecognition::showResult() {
    // print score
    std::cout << "***********************" << std::endl;
    std::cout << "eye_01: level is " << _eye_01_score.first 
              << "\t weight is " << _eye_01_score.second << std::endl;
    std::cout << "eye_02: level is " << _eye_02_score.first 
              << "\t weight is " << _eye_02_score.second << std::endl;
    std::cout << "nose: level is " << _nose_score.first 
              << "\t weight is " << _nose_score.second << std::endl;
    std::cout << "mouth: level is " << _mouth_score.first
              << "\t weight is " << _mouth_score.second << std::endl;
    
    // drawing part
    cv::Point eye_01_center(_eye_01.x + _eye_01.width*0.5, _eye_01.y + _eye_01.height*0.5);
    cv::Point eye_02_center(_eye_02.x + _eye_02.width*0.5, _eye_02.y + _eye_02.height*0.5);
    int eyes_radius = cvRound((_eye_01.width + _eye_01.height + _eye_02.width + _eye_02.height)*0.25*0.5);
    cv::circle(_roi_img, eye_01_center, eyes_radius, cv::Scalar(255, 0, 0), 4, 8, 0);
    cv::circle(_roi_img, eye_02_center, eyes_radius, cv::Scalar(255, 0, 0), 4, 8, 0);

    cv::Point nose_center(_nose.x + _nose.width * 0.5, _nose.y + _nose.height * 0.5);
    int nose_radius = cvRound((_nose.width + _nose.height) * 0.25);
    cv::circle(_roi_img, nose_center, nose_radius, cv::Scalar(255, 255, 0), 4, 8, 0);

    cv::Point mouth_center(_mouth.x + _mouth.width * 0.5, _mouth.y + _mouth.height * 0.5);
    int mouth_radius = cvRound((_mouth.width + _mouth.height) * 0.5); 
    cv::circle(_roi_img, mouth_center, mouth_radius, cv::Scalar(255, 255, 0), 4, 8, 0);

    cv::imshow("result", _roi_img);
    cv::waitKey(0);
}