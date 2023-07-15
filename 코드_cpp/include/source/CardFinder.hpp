#pragma once

#include <jni.h>
#include <Utils.hpp>
#include <ThreadPool.hpp>
#include <torch/script.h>


// Singleton Instance 
// Java�� CardFinder �ν��Ͻ��� ���� �� �ϱ� ������
// �� �����Ӹ��� �ν��Ͻ� �ʱ�ȭ�� �����ϱ� ���� �̱����� ���
class CardFinder
{
public:
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
    auto Start(cv::Mat& src) -> std::string;
    // ��ü ���ɿ��� ��ȯ �Լ�
    auto GetCapturedArea() -> cv::Rect&;
    // üũī�带 �����Ͽ� ���� �� ���� ������ ��ȯ�ϴ� �Լ�
    auto GetCoordinates() -> std::vector<float>;
    // �Ҹ���
    virtual ~CardFinder() = default;

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
    auto SetBrightCorrectionModels() -> cv::Mat;
    // �ּ��ڽ¹�⺸���� ��� ������ ���� vector<Mat> Ÿ���� �����͸� ����� �Լ�
    auto SetBrightCorrectionFields() -> cv::Mat;
    // Ǫ���� ��ȯ�� ���Ǵ� ����þ� ���͸� ����� �Լ�
    auto SetGaussianFilters(cv::Size size, double D0) -> const std::vector<cv::Mat>;
    // �� ���� ������ Scale factor�� ���ϴ� �Լ�
    auto SetCoordinates(cv::Point2f& pt1, cv::Point2f& pt2, cv::Point2f& pt3, cv::Point2f& pt4) -> void;
    // Contrast Limiting Adaptive Histogram Equalization ��ü ���� �Լ�
    auto SetCLAHE(double limit_var, cv::Size tile_size) -> cv::Ptr<cv::CLAHE>;
    // pytorch ���� �ҷ����� �Լ�
    auto SetTorchModel(JNIEnv& env, jobject& obj, const char* class_dir) -> std::vector<torch::jit::script::Module>;
    // üũī���� ���� ���� ������ ��� �Լ�
    auto GetLines(cv::Mat& src, AreaLocation arealoc) -> std::vector<cv::Vec4f>;
    // üũī���� ���� ���� �����鿡�� ������ ���ϰ� �� ���� �̿��Ͽ� ������ ��� �Լ�
    auto GetCrossPointFromTwoLines(std::vector<cv::Vec4f>& line1, std::vector<cv::Vec4f>& line2)->cv::Point2f;
    // �� ���� �̷�� ����(üũī���� ����)�� ���ϴ� �Լ�
    auto GetAngleFromTwoPoints(cv::Point2f pt1, cv::Point2f pt2, AreaLocation arealoc) -> float;
    // �ּ��ڽ¹�⺸���� �����ϴ� �Լ�
    auto BrightCorrect(cv::Mat& src) -> cv::Mat&;
    // Ǫ���� HomomorhpicFitering�� �����ϴ� �Լ�
    virtual auto HomomorphicCorrect(cv::Mat& src) -> cv::Mat;
    // ���ڿ����� �̹����� ��ó���ϴ� �Լ�
    auto AreaPreProcess(cv::Mat& src, cv::Mat& dst) -> void;
    // �������󿡼� ��ü���� ������ ���ִ� �Լ�
    auto AreaMasking(cv::Mat& src) -> void;
    // �������󿡼� ��ü���� �з��ϴ� �Լ�
    virtual auto AreaSegmant(cv::Mat& src, int offset_width, int offset_height) -> std::vector<cv::Rect>;
    // ���ڿ����鸸 �߷����� �Լ�
    virtual auto DataClassification(std::vector<cv::Rect>& rois) -> std::vector<cv::Rect>;
    // ���ڿ����� Pytorch Script�� Ȱ���Ͽ� �ν��ϴ� �Լ�
    auto DataDiscrimination(cv::Mat& src, std::vector<cv::Rect>& areas, torch::jit::script::Module& module, std::map<int, char>& labels) -> std::string;

    // ������ ����
    int m_thread_num;
    // android studio java PreviewView�� roi size
    cv::Size m_image_view_size;
    // ��ü ���ɿ���
    cv::Rect m_captured_area;
    // ImageAnalysis�� roi���� ȹ���� �������� ��ġ�� �������ִ� ����������
    cv::Point2f m_scale_factor;
    // ��ü ���ɿ������� ���, ����, �ϴ�, ���� ���ɿ����� ������ �� �ʿ��� �� ���� ����
    std::vector<cv::Point2f> m_cross_pt4;
    // ���, ����, �ϴ�, ���� ���ɿ���
    std::vector<cv::Rect> m_parts_of_captured_area;
    // �ּ��ڽ��� ��ü����, ����, ���ο� �ش��ϴ� A ���
    cv::Mat m_A;
    // �ּ��ڽ� ��� ������ ����� ���� ��ü, ����, ����, ������ Mat Ÿ�� �̹���
    cv::Mat m_br_correction_field;
    // ������Ǯ
    ThreadPool::ThreadPool m_pool;
    // pytorch ��ũ��Ʈ ���
    std::vector<torch::jit::script::Module> m_modules;
    // ���������� ħ�� �� Ȯ�� ���� ���Ǵ� Ŀ��
    cv::Mat m_kernel;
    // ������׷� ����ȭ�� ���Ǵ� CLAHE ��ü
    cv::Ptr<cv::CLAHE> m_clahe;
    // ��ü ���ɿ������� ������ �̹���
    std::vector<jfloat> m_res_coordinate;
    // ���̺�
    std::map<int, char> m_labels_number, m_labels_alphabet;


protected:
    // SubProcess instance ���� �� ���Ǵ� ������
    CardFinder(int& w, int& h);
    // ����þ� ������ �� ������ ����
    std::vector<cv::Mat> m_gaussian_filters;

};


