#pragma once

#include <jni.h>
#include <Utils.hpp>
#include <ThreadPool.hpp>
#include <torch/script.h>



// Singleton Instance 
// -> Java에 CardFinder 인스턴스를 저장 못 하기 때문에 
//    매 프레임마다 인스턴스 초기화를 방지하기 위해 싱글톤을 사용
class CardFinder
{
public:

    static CardFinder& GetInstance(int& h, int& w, JNIEnv& env, jobject& obj);

    auto Start(cv::Mat& src) -> std::string;


    virtual ~CardFinder() = default;

private:

    CardFinder(int& h, int& w, JNIEnv& env, jobject& obj);

    // 전체 관심영역을 획득하는 함수
    auto SetCapturedArea(int w, int h) -> cv::Rect;
    // 부분관심영역(가로, 세로)을 획득하는 함수
    auto SetPartsOfCapturedArea() -> std::vector<cv::Rect>;
    // 최소자승밝기보정에 사용되는 행렬을 만드는 함수
    auto SetBrightCorrectionModels() -> std::vector<cv::Mat>;
    // 최소자승밝기보정의 결과 영상을 담을 vector<Mat> 타입의 데이터를 만드는 함수
    auto SetBrightCorrectionFields() -> std::vector<cv::Mat>;
    // 푸리에 변환에 사용되는 가우시안 필터를 만드는 함수
    auto SetGaussianFilters(cv::Size size, double D0) -> const std::vector<cv::Mat>;
    // 최소자승밝기보정을 수행하는 함수
    auto BrightCorrect(cv::Mat& src) -> cv::Mat&;
    // 푸리에 HomomorhpicFitering을 수행하는 함수
    virtual auto HomomorphicCorrect(cv::Mat& src) -> cv::Mat;
    // 숫자영역의 이미지를 전처리하는 함수
    auto AreaPreProcess(cv::Mat& src, cv::Mat& dst) -> void;
    // 체크카드의 가로 세로 직선을 얻는 함수
    auto GetIntercept(cv::Mat& src) -> std::vector<std::tuple<cv::Point2f, cv::Point2f>>;
    // 체크카드의 가로 세로 직선들에서 교점을 구하고 세 점을 이용하여 각도를 얻는 함수
    auto GetAngleFromLine(
            std::vector<std::tuple<cv::Point2f, cv::Point2f>>& intercept_col,
            std::vector<std::tuple<cv::Point2f, cv::Point2f>>& intercept_row) -> float;
    // 숫자로 추정되는 영역을 확장시켜 그리는 함수
    auto AreaDraw(cv::Mat& src)->cv::Mat;
    // 이진영상에서 객체들의 잡음을 없애는 함수
    auto AreaMasking(cv::Mat& src) -> void;
    // 이진영상에서 객체들을 분류하는 함수
    virtual auto AreaSegmant(cv::Mat& src) -> std::vector<cv::Rect>;
    // 숫자영역들만 추려내는 함수
    auto DataClassification(std::vector<cv::Rect>& rois) -> std::vector<cv::Rect>;
    // 숫자영역을 Pytorch Script를 활용하여 인식하는 함수
    auto DataDiscriminationCNN(cv::Mat& src, std::vector<cv::Rect>& areas) -> std::string;

    // 스레드 갯수
    int m_thread_num;
    // 전체 관심영역
    cv::Rect m_captured_area;
    // 가로세로 관심영역
    std::vector<cv::Rect> m_parts_of_captured_area;
    // 최소자승의 전체영역, 가로, 세로에 해당하는 A 행렬
    std::vector<cv::Mat> m_A;
    // 최소자승 밝기 보정의 결과가 담기는 전체, 가로, 세로, 영역의 Mat 타입 이미지
    std::vector<cv::Mat> m_br_correction_fields;
    // 스레드풀
    ThreadPool::ThreadPool m_pool;
    // pytorch 스크립트 모듈
    torch::jit::script::Module m_module;
    // android studio java 함수를 호출하는 jclass
    jclass m_java_class;
    // 전체 관심영역으로 포착된 이미지
    cv::Mat m_captured_img;


protected:
    CardFinder(int& h, int& w);
    cv::Mat m_kernel;
    std::vector<cv::Mat> m_gaussian_filters;

};


