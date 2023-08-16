#pragma once

#include <jni.h>
#include <Utils.hpp>
#include <Client.hpp>
#include <thread>
#include <mutex>
#include <future>

// Singleton Instance 
// Java�� CardFinder �ν��Ͻ��� ���� �� �ϱ� ������
// �� �����Ӹ��� �ν��Ͻ� �ʱ�ȭ�� �����ϱ� ���� �̱����� ���
class CardFinder
{
public:
    using Lines = std::vector<cv::Vec4f>;
    using Rects = std::vector<cv::Rect>;
    
    // ���, ����, �ϴ�, ���� ������ �����ϴ� enum class
    enum class AreaLocation
    {
        TOP,
        LEFT,
        BOTTOM,
        RIGHT
    };

    // Singleton Instance ���� �Լ�
    static CardFinder& GetInstance(JNIEnv& env, jobject& obj, int& w, int& h);
    // ���� �м� �����Լ�
    auto Start(unsigned char* data, jint& col, jint& row) -> void;
    // ��ü ���ɿ��� ��ȯ �Լ�
    // üũī�带 �����Ͽ� ���� �� ���� ������ ��ȯ�ϴ� �Լ�
    auto GetCoordinates() -> std::vector<float>;
    // �� ���� ���� �����Ͱ� ����� ��ü�� �ʱ�ȭ�ϴ� �Լ�
    auto ResetCoordinates() -> void;
    // m_client�� ����� base64 ���� ���۸� �ʱ�ȭ�ϴ� �Լ�
    auto ResetClientBuffer() -> void;
    auto ResetStopImagePreprocessingBool() -> void;
    auto IsEmptyBuffer() -> bool;

    // �м���� ��ȯ�Լ�
    auto GetResult() -> std::string;

    auto PullJobs() -> void;
    // �Ҹ���
    ~CardFinder();


private:
    // CardFinder Instance ������
    CardFinder(JNIEnv& env, jobject& obj, int& w, int& h);
    // android studio java PreviewView�� roi size�� ȹ���ϴ� �Լ�
    auto GetImageViewSize(JNIEnv& env, jobject& obj, const char* class_dir) -> cv::Size;
    // ��ü ���ɿ����� ȹ���ϴ� �Լ�
    auto SetCapturedArea(int w, int h) -> cv::Rect;
    // ���������͸� ����ϴ� �Լ�
    auto SetScaleFactor(int w, int h) -> cv::Point2f;
    // �κа��ɿ���(����, ����)�� ȹ���ϴ� �Լ�
    auto SetPartsOfCapturedArea() -> std::vector<cv::Rect>;
    // �ּ��ڽ¹�⺸���� ���Ǵ� ����� ����� �Լ�
    auto SetBrightCorrectionModel() -> cv::Mat;
    // �ּ��ڽ¹�⺸���� ��� ������ ���� vector<Mat> Ÿ���� �����͸� ����� �Լ�
    auto SetBrightCorrectionFields() -> cv::Mat;
    // Ǫ���� ��ȯ�� ���Ǵ� ����þ� ���͸� ����� �Լ�
    auto SetGaussianFilters(cv::Size size, double D0) -> std::vector<cv::Mat>;
    // �� ���� ������ Scale factor�� ���ϴ� �Լ�
    auto SetCoordinates(cv::Point2f& pt1, cv::Point2f& pt2, cv::Point2f& pt3, cv::Point2f& pt4) -> void;
    // Contrast Limiting Adaptive Histogram Equalization ��ü ���� �Լ�
    auto SetCLAHE(double limit_var, cv::Size tile_size) -> cv::Ptr<cv::CLAHE>;
    // üũī���� ���� ���� ������ ��� �Լ�
    auto FindLines(cv::Mat& src, AreaLocation arealoc) -> Lines;
    // üũī���� �𼭸� ������ ã�� �Լ�
    auto FindCorner(Lines& lines1, Lines& lines2) -> cv::Point2f;
    // �ּ��ڽ¹�⺸���� �����ϴ� �Լ�
    auto BrightCorrect(cv::Mat& src) -> cv::Mat&;
    // Ǫ���� HomomorhpicFitering�� �����ϴ� �Լ�
    auto HomomorphicCorrect(cv::Mat& src, cv::Mat& filter) -> cv::Mat;

    // android studio java analize �Լ����� �޾ƿ��� ImageProxy �ν��Ͻ��� ���� ���� ������
    cv::Size m_image_view_size;
    // android studio java PreviewView�� roi size
    cv::Size m_image_proxy_size;
    // ��ü ���ɿ���
    cv::Rect m_captured_area;
    // ImageAnalysis�� roi���� ȹ���� �������� ��ġ�� �������ִ� ����������
    cv::Point2f m_scale_factor;
    // ��ü ���ɿ������� ���� �� �ϴ� ������ ��������
    cv::Point m_start_pt_of_right_area, m_start_pt_of_bottom_area;
    // ���, ����, �ϴ�, ���� ���ɿ���
    std::vector<cv::Rect> m_parts_of_captured_area;
    // �ּ��ڽ��� ��ü����, ����, ���ο� �ش��ϴ� A ���
    cv::Mat m_A;
    // �ּ��ڽ� ��� ������ ����� ���� ��ü, ����, ����, ������ Mat Ÿ�� �̹���
    cv::Mat m_br_correction_field;
    // ���������� ħ�� �� Ȯ�� ���� ���Ǵ� Ŀ��
    cv::Mat m_kernel;
    // ������׷� ����ȭ�� ���Ǵ� CLAHE ��ü
    std::shared_ptr<cv::CLAHE> m_clahe;
    // ����þ� ������ �� ������ ����
    std::vector<cv::Mat> m_gaussian_filters;
    // ��ü ���ɿ������� ������ �̹���
    std::vector<jfloat> m_res_coordinate;
    // Ŭ���̾�Ʈ
    Client m_client;
    // json ������ response ��Ŷ�� body �޼����� ����Ǵ� string ��ü
    std::string m_result;
    // �̹��� ��ó�� �ߴܿ� atomic<bool>
    std::atomic_bool m_stop_image_preprocessing;
    // pulling ������ ������ while ������ atomic<bool>
    std::atomic_bool m_pull_thr_on;
    // �̹��� ��ó�� �۾� ť
    std::queue<std::future<void>> m_image_preprocessing_jobs;
    // task queue pulling ������
    std::vector<std::thread> m_pull_thr;
    // ������ ���� ��ü
    std::condition_variable m_conv;
    // ������ �Ӱ迵�� ó���� mutex
    std::mutex m_mutex;
};

