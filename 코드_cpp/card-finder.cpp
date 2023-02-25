#include <CardFinder.hpp>
#include "SubProcess.hpp"

CardFinder::CardFinder(int& h, int& w)
    :
        m_captured_area(SetCapturedArea(w, h)),
        m_pool(0),
        m_kernel(cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3))),
        m_gaussian_filters(SetGaussianFilters(m_captured_area.size(), 30))
{

}

CardFinder::CardFinder(int& h, int& w, JNIEnv& env, jobject& obj)
        :
        m_thread_num(GetThreadInfo() / 2),                                          // 스레드 개수
        m_captured_area(SetCapturedArea(w, h)),                                     // 전체 영역
        m_parts_of_captured_area(SetPartsOfCapturedArea()),                         // 부분영역(가로, 세로)
        m_A(SetBrightCorrectionModels()),                                           // 최소자승 A행렬들(전체영역, 가로영역, 세로영역)
        m_br_correction_fields(SetBrightCorrectionFields()),                        // 최소자승 보정으로 인한 결과 영상을 저장하는 영상(전체영역, 가로영역, 세로영역)
        m_pool(m_thread_num),                                                       // 스레드풀
        m_kernel(cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3))),       // 팽창 및 침식 연산용 커널
        m_gaussian_filters(SetGaussianFilters(m_captured_area.size(), 30))          // 가우시안 low, high 필터
{
    m_java_class = env.FindClass("com/sjlee/cardfinder/ViewActivity");

    jmethodID method_id =                                                           // ViewActivity class에서 GetFileDir 함수의 ID를 얻는다
        env.GetMethodID(m_java_class, "GetFileDir", "()Ljava/lang/String;");

    jstring path_jstring = 
        static_cast<jstring>(env.CallObjectMethod(obj, method_id));                 // GetFileDir 함수 호출하여 jstring 타입의 결과를 얻는다   

    const char* path_char = env.GetStringUTFChars(path_jstring, NULL);             

    std::string filename_number_script = path_char;
    std::string filename_number_area_script = path_char;

    filename_number_script += "/number_script.pt";                                  // number_script.pt 파일 경로

    m_module = torch::jit::load(filename_number_script);                            // pytorch 모델 호출
    m_module.eval();                                                                // 모델 평가모드 전환

    env.ReleaseStringUTFChars(path_jstring, path_char);
}

CardFinder& CardFinder::GetInstance(int& h, int& w, JNIEnv& env, jobject& obj)
{
    static CardFinder singleton(h, w, env, obj);

    return singleton;
}


auto CardFinder::Start(cv::Mat& src) -> std::string
{
    std::string numbers;                                                            // 체크카드 인식 결과를 받는 변수

    auto intercept_proc1 = m_pool.PushJob([&src, this](){                           // 가로영역에서 직선을 얻는 스레드풀 프로세스

        cv::Mat col_area_img = src(m_parts_of_captured_area[0]);
        cv::Mat temp;
        cv::Canny(BrightCorrect(col_area_img), temp, 128, 255);
        cv::dilate(temp, temp, m_kernel, cv::Point(-1, -1), 1);
        return GetIntercept(temp);;
    });


    auto intercept_proc2 = m_pool.PushJob([&src, this](){                           // 세로영역에서 직선을 얻는 스레드풀 프로세스

        cv::Mat row_area_img = src(m_parts_of_captured_area[1]);
        cv::Mat temp;
        cv::Canny(BrightCorrect(row_area_img), temp, 128, 255);
        cv::dilate(temp, temp, m_kernel, cv::Point(-1, -1), 1);
        return GetIntercept(temp);
    });

    auto inter1 = intercept_proc1.get();
    if(inter1.empty()) return numbers;

    auto inter2 = intercept_proc2.get();
    if(inter2.empty()) return numbers;

    auto angle = GetAngleFromLine(inter1, inter2);                                  // 세 점이 이루는 각도를 얻는다

    if (!std::isnan(angle) && angle != 0.0f)
    {
        SubProcess &instance = SubProcess::GetInstance(src.rows, src.cols);         

        m_captured_img = src(m_captured_area);

        m_captured_img =                                                            // HomomorphicFiltering : low pass, high pass 연산
                (0.3f * HomomorphicCorrect(m_captured_img)) +
                (1.5f * instance.HomomorphicCorrect(m_captured_img));


        auto f_processed = m_pool.PushJob([this]() {                                // HomomorphicFiltering : exp 연산 및 최소자승 연산
            cv::normalize(
                m_captured_img, m_captured_img, 0, 1, cv::NORM_MINMAX);
            cv::exp(m_captured_img, m_captured_img);
            cv::normalize(
                m_captured_img, m_captured_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);

            return BrightCorrect(m_captured_img);
        });

        cv::Point center(m_captured_img.cols / 2, m_captured_img.rows / 2);         // 관심영역의 중심점을 획득

        cv::Mat thresh, m = cv::getRotationMatrix2D(center, angle, 1);              // angle 만큼 회전하는 회전 영상획득

        cv::warpAffine(
            f_processed.get(), m_captured_img, m, m_captured_img.size());           // 회전 연산 실행

        m_captured_img = Rotate90(m_captured_img);                                  // 90도 회전 실행

        cv::threshold(                                                              // 영상 이진화 
            m_captured_img, thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        AreaMasking(thresh);                                                        // 잡음 제거

        cv::Mat mask = AreaDraw(thresh);                                            // 큰 영역 그리기

        std::vector<cv::Rect> rois_single, rois_chunk, result_areas;

        rois_chunk = AreaSegmant(mask);                                             // 숫자가 모여있는 영역 획득

        rois_single = instance.AreaSegmant(thresh);                                 // 숫자로 추정되는 여러 영역들을 획득

        rois_chunk = DataClassification(rois_chunk);                                // 숫자가 모여있는 영역 4개 획득

        if(rois_chunk.size() == 4)                                                  // 숫자가 모여있는 영역 4곳에서 숫자 16개를 얻는 연산
        {
            for (int i = 0; i < rois_chunk.size(); ++i)
            {
                for (int j = 0; j < rois_single.size(); ++j)
                {
                    cv::Point chunk_tl = rois_chunk[i].tl();                        // i번째 4자리 영역의 좌측상단
                    cv::Point single_tl = rois_single[j].tl();                      // j번째 숫자로 추정되는 영역 좌측상단
                    cv::Point chunk_br = rois_chunk[i].br();                        // i번째 4자리 영역의 우측하단
                    cv::Point single_br = rois_single[j].br();                      // j번째 숫자로 추정되는 영역 우측하단

                    if ((chunk_tl.y - 5 <= single_tl.y && chunk_tl.y + 5 >= single_tl.y) && 
                        (chunk_tl.x < single_tl.x && chunk_br.x > single_br.x))     // 숫자 낱개영역의 좌측상단 좌표가 4자리 영역의 좌측상단 좌표의 범위에 들어가는 경우
                    {
                        result_areas.emplace_back(rois_single[j]);
                    }
                }
            }

            if (result_areas.size() == 16)                                          // 숫자 16개 영역을 획득 시
            {
                std::sort(result_areas.begin(), result_areas.end(),
                          [&result_areas](cv::Rect& l, cv::Rect& r) {
                              return l.x < r.x;
                          });


                numbers = DataDiscriminationCNN(thresh, result_areas);              // 숫자인식 실행

            }

        }

    }

    return numbers;
}

auto CardFinder::SetCapturedArea(int w, int h) -> cv::Rect
{
    int roi_width = w * 0.25f;                                                      // 전체관심영역의 가로 길이
    int roi_height = roi_width * 1.618f;                                            // 전체관심영역의 세로 길이

    int left = (w - roi_width) / 2;                                                 // 관심영역이 위치하는 x축
    int top = (h - roi_height) / 2;                                                 // 관심영역이 위치하는 y축

    cv::Rect roi(left, top, roi_width, roi_height);

    return roi;
}

auto CardFinder::SetPartsOfCapturedArea() -> std::vector<cv::Rect>
{
    cv::Point pt1(
        m_captured_area.x, m_captured_area.y+(m_captured_area.height * 0.10f));
    cv::Point pt2(
        m_captured_area.br().x, m_captured_area.y+(m_captured_area.height * 0.10f));
    cv::Point pt3(
        m_captured_area.x+(m_captured_area.width * 0.15f), m_captured_area.y);
    cv::Point pt4(
        m_captured_area.x+(m_captured_area.width * 0.15f), m_captured_area.br().y);

    auto xChild = (                                                                 // x축 분자
        ((pt1.x * pt2.y) - (pt1.y * pt2.x)) * (pt3.x - pt4.x)) - 
        ((pt1.x - pt2.x) * ((pt3.x * pt4.y) - (pt3.y * pt4.x)));
    auto yChild = (                                                                 // y축 분자
        ((pt1.x * pt2.y) - (pt1.y * pt2.x)) * (pt3.y - pt4.y)) - 
        ((pt1.y - pt2.y) * ((pt3.x * pt4.y) - (pt3.y * pt4.x)));

    auto mother = 
        ((pt1.x - pt2.x) * (pt3.y - pt4.y)) - ((pt1.y - pt2.y) * (pt3.x - pt4.x));  // 분모

    cv::Point cross_point1(xChild / mother, yChild / mother);                       // 교점획득

    cv::Rect col_area(                                                              // 가로 영역
            cv::Point(m_captured_area.x, m_captured_area.y),
            cv::Point(m_captured_area.br().x, cross_point1.y));

    cv::Rect row_area(                                                              // 세로 영역
            cv::Point(m_captured_area.x, m_captured_area.y),
            cv::Point(cross_point1.x, m_captured_area.br().y));


    return std::vector<cv::Rect>() = {
            col_area,
            row_area
    };

}

auto CardFinder::SetBrightCorrectionModels() -> std::vector<cv::Mat>
{
    std::vector<cv::Mat> A(3);                                                      // A행렬들(전체영역, 가로영역, 세로영역)
    int n = 0, total = (m_captured_area.width) * (m_captured_area.height);
    A[0] = cv::Mat::zeros(total, 6, CV_32FC1);                                      // (가로*세로)행, 6열 행렬 생성

    for (int y = 0; y < m_captured_area.height; y++)                                // A 행렬에 원소값 삽입
    {
        for (int x = 0; x < m_captured_area.width; x++)
        {
            A[0].ptr<float>(n)[0] = x*x;
            A[0].ptr<float>(n)[1] = y*y;
            A[0].ptr<float>(n)[2] = x*y;
            A[0].ptr<float>(n)[3] = x;
            A[0].ptr<float>(n)[4] = y;
            A[0].ptr<float>(n)[5] = 1;

            ++n;
        }
    }


    total = m_parts_of_captured_area[0].width * m_parts_of_captured_area[0].height; 

    A[1] = cv::Mat::zeros(total, 6, CV_32FC1); 
    n = 0;

    for (int y = 0; y < m_parts_of_captured_area[0].height; y++)
    {
        for (int x = 0; x < m_parts_of_captured_area[0].width; x++)
        {
            A[1].ptr<float>(n)[0] = x*x;
            A[1].ptr<float>(n)[1] = y*y;
            A[1].ptr<float>(n)[2] = x*y;
            A[1].ptr<float>(n)[3] = x;
            A[1].ptr<float>(n)[4] = y;
            A[1].ptr<float>(n)[5] = 1;

            ++n;
        }
    }

    total = m_parts_of_captured_area[1].width * m_parts_of_captured_area[1].height; 

    A[2] = cv::Mat::zeros(total, 6, CV_32FC1);
    n = 0;

    for (int y = 0; y < m_parts_of_captured_area[1].height; y++)
    {
        for (int x = 0; x < m_parts_of_captured_area[1].width; x++)
        {
            A[2].ptr<float>(n)[0] = x*x;
            A[2].ptr<float>(n)[1] = y*y;
            A[2].ptr<float>(n)[2] = x*y;
            A[2].ptr<float>(n)[3] = x;
            A[2].ptr<float>(n)[4] = y;
            A[2].ptr<float>(n)[5] = 1;

            ++n;
        }
    }

    return A;
}

auto CardFinder::SetBrightCorrectionFields() -> std::vector<cv::Mat>
{
    return {
            cv::Mat::zeros(                                                         // 전체영역의 최소자승 연산 결과가 들어가는 영상
                m_captured_area.height, m_captured_area.width, CV_8UC1),                         
            cv::Mat::zeros(                                                         // 가로영역의 최소자승 연산 결과가 들어가는 영상
                m_parts_of_captured_area[0].height, m_parts_of_captured_area[0].width, CV_8UC1),
            cv::Mat::zeros(                                                         // 세로영역의 최소자승 연산 결과가 들어가는 영상
                m_parts_of_captured_area[1].height, m_parts_of_captured_area[1].width, CV_8UC1)
    };
}

auto CardFinder::SetGaussianFilters(cv::Size size, double D0) -> const std::vector<cv::Mat>
{
    std::vector<cv::Mat> filter = {
            cv::Mat::zeros(size.height, size.width, CV_32FC1),
            cv::Mat::zeros(size.height, size.width, CV_32FC1)};

    int u, v;
    double D;
    double H;                                                                       // High값 원소
    double L;                                                                       // Low값 원소
    double centerU = size.width / 2;                                                // mask의 가로 중심
    double centerV = size.height / 2;                                               // mask의 세로 중심

    for(v = 0; v < size.height; v++)
    {
        for(u = 0; u < size.width; u++)
        {
            D = sqrt(pow(u - centerU, 2) + pow(v - centerV, 2));                    // 가우시안 필터의 지름
            L = exp((-1 * pow(D, 2)) / (2.0*pow(D0, 2)));                       
            H = 1 - L;

            filter[0].ptr<float>(v)[u] = L;
            filter[1].ptr<float>(v)[u] = H;
        }
    }
    Shift(filter[0]);
    Shift(filter[1]);

    return filter;
}


auto CardFinder::BrightCorrect(cv::Mat& src) -> cv::Mat&
{
    cv::Mat X, A, Y, temp;                                                           // X행렬, A행렬, Y행렬, 임시행렬
    int total = src.total();
    int counter = 0, result_number = 0;

    float ratio_length =                                                            
        static_cast<float>(src.rows) / static_cast<float>(src.cols);
    bool ratio_bool = ratio_length > 1.61f && ratio_length < 1.619f;

    src.convertTo(Y, CV_32FC1);                                                     // float 타입으로 변환
    Y = Y.reshape(1, total);                                                        // 1행 total열 행렬으로 변환

    if(ratio_bool)                                                                  // 전체영역
    {
        result_number = 0;
        temp = m_br_correction_fields[0];
        X = (m_A[0].t()* m_A[0]).inv()* m_A[0].t()*Y;
    }
    else if(!ratio_bool && src.cols > src.rows)                                     // 가로영역
    {
        result_number = 1;
        temp = m_br_correction_fields[1];
        X = (m_A[1].t()* m_A[1]).inv()* m_A[1].t()*Y;                               
    }
    else if(!ratio_bool && src.cols < src.rows)                                     // 세로영역
    {
        result_number = 2;
        temp = m_br_correction_fields[2];
        X = (m_A[2].t()* m_A[2]).inv()* m_A[2].t()*Y;
    }

    float& a = *X.ptr<float>(0);                
    float& b = *X.ptr<float>(1);
    float& c = *X.ptr<float>(2);
    float& d = *X.ptr<float>(3);
    float& e = *X.ptr<float>(4);
    float& f = *X.ptr<float>(5);

    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            float y = a*(x*x) + b*(y*y) + c*x*y + d*x + e*y + f;                    // 밝기 보정값 획득

            temp.ptr<uchar>(y)[x] =                                                 
                static_cast<uchar>(
                    static_cast<float>(src.ptr<uchar>(y)[x]) - y + 128);            // 최소자승 밝기 보정연산
        }
    }

    switch(result_number)
    {
        case 0:
            return m_br_correction_fields[0];                       
        case 1:
            return m_br_correction_fields[1];
        case 2:
            return m_br_correction_fields[2];
    }
}

auto CardFinder::HomomorphicCorrect(cv::Mat& src) -> cv::Mat                
{
    cv::Mat log, complex;                       

    cv::Mat planes[] = { cv::Mat(), cv::Mat() };                                

    src.convertTo(log, CV_32FC1);                                                   // float 타입 변환
    cv::log((log / 255) + cv::Scalar::all(1), log);                                 // log를 취한 영상 획득

    cv::dft(log, complex, cv::DFT_COMPLEX_OUTPUT);                                  // 푸리에 연산 수행(2채널 실수+허수 영상 획득)

    cv::split(complex, planes);                                                     // 영상 분할

    cv::multiply(planes[0], m_gaussian_filters[0], planes[0]);                      // 실수부분과 가우시안 필터 곱연산
    cv::multiply(planes[1], m_gaussian_filters[0], planes[1]);                      // 허수부분과 가우시안 필터 곱연산

    cv::merge(planes, 2, complex);                                                  // 실수 및 허수 영상 병합

    cv::idft(complex, complex, cv::DFT_REAL_OUTPUT);                                // 역푸리에 연산 후 실수영상 획득

    return complex;
}


auto CardFinder::AreaPreProcess(cv::Mat& src, cv::Mat& dst) -> void
{
    cv::resize(src, dst, src.size() * 8, cv::INTER_CUBIC);                          // 영상 크기 확대

    erode(dst, dst, m_kernel, cv::Point(-1, -1), 1);                                // 침식연산

    morphologyEx(dst, dst, cv::MORPH_OPEN, m_kernel, cv::Point(-1, -1), 1);         // 열림연산(침식후 팽창)

    cv::resize(dst, dst, src.size() / 2, cv::INTER_LANCZOS4);                       // 영상 크기 축소
}



auto CardFinder::GetIntercept(cv::Mat& src) -> std::vector<std::tuple<cv::Point2f, cv::Point2f>>
{
    std::vector<cv::Vec4f> linesP;                                                  // 직선 성분이 담기는 변수
    std::vector<std::tuple<cv::Point2f, cv::Point2f>> lines;                        // 직선의 두 점이 담기는 변수

    int& col = src.cols;
    int& row = src.rows;

    bool col_is_big = col > row;
    bool row_is_big = col < row;

    if(col_is_big)
        cv::HoughLinesP(src, linesP, 1,  CV_PI / 180, 128, col / 2, 20);            // 허프 직선 연산 -> 가로선 획득
    else if(row_is_big)
        cv::HoughLinesP(src, linesP, 1,  CV_PI / 180, 128, row / 2, 20);            // 허프 직선 연산 -> 세로선 획득

    int line_size = linesP.size();

    if(line_size <= 20)
    {
        for(int j = 0; j < line_size; ++j)
        {
            cv::Point2f pt1(linesP[j][0], linesP[j][1]);
            cv::Point2f pt2(linesP[j][2], linesP[j][3]);

            auto dist = GetDist(pt1, pt2);                                          // 두 점 사이의 거리 획득

            if (col_is_big && (dist > col / 2 && dist < col))
            {
                float m1 = (pt2.y - pt1.y) / (pt2.x - pt1.x);
                float n1 = ((-1.0f * m1) * pt1.x) + pt1.y;

                float x = pt2.x - 500;
                float y = (m1 * x) + n1;                                            // y = ax + b

                lines.emplace_back(std::make_tuple(std::move(pt2), cv::Point2f(x, y)));
            }
            else if (row_is_big && (dist > row / 2 && dist < row))
            {
                float m2 = (pt2.y - pt1.y) / (pt2.x - pt1.x);
                float n2 = ((-1.0f * m2) * pt1.x) + pt1.y;

                float y = pt2.y - 500;
                float x = y/m2 + (-1.0f*n2)/m2;                                     // y = ax + b
                lines.emplace_back(std::make_tuple(std::move(pt2), cv::Point2f(x, y)));
            }
        }
    }

    return lines;
}

auto CardFinder::GetAngleFromLine(
        std::vector<std::tuple<cv::Point2f, cv::Point2f>>& intercept_col,
        std::vector<std::tuple<cv::Point2f, cv::Point2f>>& intercept_row) -> float
{

    float result_ang = 0.0f;


    if(intercept_col.size() != 0 && intercept_row.size() != 0)
    {
        int size_col_mn = intercept_col.size();
        int size_row_mn = intercept_row.size();

        for(int i = 0; i < size_col_mn; ++i)
        {
            cv::Point2f &pt1 = std::get<0>(intercept_col[i]);                       // 가로 선에 위치한 끝점(좌측)
            cv::Point2f &pt2 = std::get<1>(intercept_col[i]);                       // 가로 선에 위치한 끝점(우측)

            for(int j = 0; j < size_row_mn; ++j)
            {
                cv::Point2f &pt3 = std::get<0>(intercept_row[j]);                   // 세로 선에 위치한 끝점(상단)
                cv::Point2f &pt4 = std::get<1>(intercept_row[j]);                   // 세로 선에 위치한 끝점(하단)

                auto mother =                                                       // 교점획득연산(분모)
                    ((pt1.x-pt2.x)*(pt3.y-pt4.y)) - ((pt1.y-pt2.y)*(pt3.x-pt4.x));
                auto xChild =                                                       // 교점획득연산(분자x)
                    (((pt1.x*pt2.y)-(pt1.y*pt2.x))*(pt3.x-pt4.x)) - ((pt1.x-pt2.x)*((pt3.x*pt4.y)-(pt3.y*pt4.x)));
                auto yChild =                                                       // 교점획득연산(분자y)
                    (((pt1.x*pt2.y)-(pt1.y*pt2.x))*(pt3.y-pt4.y)) - ((pt1.y-pt2.y)*((pt3.x*pt4.y)-(pt3.y*pt4.x)));

                cv::Point2f cross_pt_on_line(xChild / mother, yChild / mother);     // 교점

                float cos_ang = GetAngleFromDotProduct(                             // 두 벡터가 이루는 각도 획득
                        cross_pt_on_line - pt2,
                        cross_pt_on_line - pt4);

                if (cos_ang > 89.900f && cos_ang < 90.100)                          
                {
                    if (cross_pt_on_line.y > pt2.y)                                 // 교점의 y축이 우측점의 y축보다 높이가 낮은 경우
                    {
                        result_ang =                                                // 교점과 우측점 하단점이 이루는 삼각형의 기울기를 획득
                            GetAngleFromDotProduct(pt2 - cv::Point2f(cross_pt_on_line.x, pt2.y),pt2 - cross_pt_on_line);
                    }
                    else if (cross_pt_on_line.y < pt2.y)                            // 교점의 y축이 우측점의 y축보다 높이가 높은 경우
                    {
                        result_ang =                                                // 교점과 우측점 하단점이 이루는 삼각형의 기울기를 획득
                            -1.0f * GetAngleFromDotProduct(cross_pt_on_line - cv::Point2f(pt2.x, cross_pt_on_line.y),cross_pt_on_line - pt2);
                    }

                    goto exit;
                }
            }
        }
    }
    exit:

    return result_ang;
}


auto CardFinder::AreaDraw(cv::Mat& src) -> cv::Mat
{
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);

    try
    {
        cv::Mat label, stats, centroid;

        int num_labels =                                                            // 이진영상에서 레이블링 연산 실행
            cv::connectedComponentsWithStats(src, label, stats, centroid, 8, CV_32SC1);

        int* mStats_ptr = nullptr;
        for (int i = 0; i < num_labels; ++i)
        {
            mStats_ptr = stats.ptr<int>(i);

            int& left = mStats_ptr[cv::CC_STAT_LEFT];                               // 레이블링된 객체의 x축 좌표
            int& top = mStats_ptr[cv::CC_STAT_TOP];                                 // 레이블링된 객체의 y축 좌표
            int& width = mStats_ptr[cv::CC_STAT_WIDTH];                             // 레이블링된 객체의 가로길이
            int& height = mStats_ptr[cv::CC_STAT_HEIGHT];                           // 레이블링된 객체의 세로길이

            cv::Point pt1(left, top);                                               // 레이블링된 객체의 시작지점(좌측상단)
            cv::Point pt2(left + width, top + height);                              // 레이블링된 객체의 끝지점(우측하단)

            if ((height > 4 && height < 40) && (height > width))
            {
                pt1 = cv::Point (left - 2, top - 2);                                // 좌측상단에서 길이 확대
                pt2 = cv::Point(left + width + 2, top + height + 2);                // 우측하단에서 길이 확대
                cv::Rect rect(pt1, pt2);
                cv::rectangle(dst, rect, cv::Scalar(255), -1);
            }
        }
    }
    catch(...)
    {

    }

    return dst;
}


auto CardFinder::AreaMasking(cv::Mat& src) -> void
{
    try
    {
        std::vector<std::vector<cv::Point>> contours;                               // 외곽선이 이루는 점들을 저장하는 변수
        cv::findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);  // 외곽선획득 연산 실행

        int i, size = contours.size();
        for (i = 0; i < size; ++i)
        {
            std::vector<cv::Point2f> approx;                                        // 외곽선을 근사하는 점들을 저장하는 변수

            approxPolyDP(contours[i], approx, 0.01, true);                          // 근사화된 외곽선 획득

            cv::Rect r = cv::boundingRect(approx);                                  // 근사화된 외곽선을 둘러싸는 바운딩 박스 획득

            if (r.area() <= 5)  
                cv::rectangle(src, r, cv::Scalar(0), -1);                           // 바운딩박스(잡음) 제거
        }
    }
    catch(...)
    {

    }

}


auto CardFinder::AreaSegmant(cv::Mat& src) -> std::vector<cv::Rect>
{
    std::vector<cv::Rect> loc;

    cv::Mat labels, stats, centroid;

    int numLabels = cv::connectedComponentsWithStats(src, labels, stats, centroid, 8, CV_32SC1);

    try
    {
        int* mStats_ptr = nullptr;
        for (int j = 0; j < numLabels; ++j)
        {
            mStats_ptr = stats.ptr<int>(j);

            int& left = mStats_ptr[cv::CC_STAT_LEFT];                               
            int& top = mStats_ptr[cv::CC_STAT_TOP];
            int& width = mStats_ptr[cv::CC_STAT_WIDTH];
            int& height = mStats_ptr[cv::CC_STAT_HEIGHT];

            cv::Rect rect(cv::Point(left, top), cv::Point(left + width, top + height));

            if (rect.width > rect.height && (rect.width > rect.height * 2 && rect.width < rect.height * 3))
                loc.emplace_back(rect);

        }
    }
    catch(...)
    {

    }


    return loc;
}



auto CardFinder::DataClassification(std::vector<cv::Rect>& rois) -> std::vector<cv::Rect>
{
    std::vector<cv::Rect> areas;                                                    
    std::vector<std::vector<cv::Rect>> candidate_areas;                             // 숫자 16자리 영역이 위치해 있는 후보영역

    int i, j, size = rois.size();


    std::sort(rois.begin(), rois.end(),                                             // y축 기준으로 오름차순 정렬
        [&rois](cv::Rect& l, cv::Rect& r) {
            return l.y < r.y;
        });

    for (i = 0; i < size; ++i)                                                      
    {
        for (j = 0; j < size; ++j)
        {
            if (i == j) continue;

            int gap_of_height = cv::abs(rois[i].tl().y - rois[j].tl().y);           // rois[i]와 rois[j]의 y축 좌표 차이를 획득

            if (gap_of_height < 4)                                                  // 좌표차이가 4미만일 경우
            {
                areas.emplace_back(rois[j]);                                        // rois[j]를 areas에 삽입
            }
        }
        if (areas.size() == 3)                                                      // areas 크기가 3일 경우
        {
            areas.emplace_back(rois[i]);                                            // rois[i]를 areas에 삽입
            candidate_areas.emplace_back(areas);                                    // 후보영역에 삽입
        }

        areas.clear();
    }

    for (auto& candi : candidate_areas)                                             // 후보영역은 크기가 4인 vector<Rect> 타입의 자료구조들이 들어있다
    {
        size = candi.size();

        std::sort(candi.begin(), candi.end(),                                       // 각 영역들을 x축으 기준으로 오름차순으로 정렬한다
            [&candi](cv::Rect& l, cv::Rect& r) {
                return l.x < r.x;
            });

        for (i = 0; i < size - 1; ++i)
        {
            int dist =                                                              // candi[i]와 candi[i-1]의 중앙지점간의 거리를 얻는다
                GetDist(cv::Point(candi[i].br().x, candi[i].tl().y), candi[i + 1].tl());

            if (dist <= 5)                                                          // 두 점 사이의 거리가 5이하인 경우 areas에 candi[i + 1]을 삽입
                areas.emplace_back(candi[i + 1]);
            if (areas.size() == 3)
            {
                areas.emplace(areas.begin(), candi[0]);                             // areas의 첫번째 지점에 candi[0] 삽입(숫자 16자리가 모여있는 네 개 영역 획득 완료)
                return areas;
            }

        }
    }

    return areas;
}


auto CardFinder::DataDiscriminationCNN(cv::Mat& src, std::vector<cv::Rect>& areas) -> std::string
{
    if (src.empty() || src.type() != CV_8UC1) return std::string();

    char digits_char[17]{ "0" };                                                    // 숫자인식의 결과가 저장되는 변수
    std::string digits_str;

    if (areas.size() == 16)
    {
        SubProcess& instance = SubProcess::GetInstance(src.rows, src.cols);

        for (int i = 0; i < 16; ++i)
        {
            cv::Mat roi_img = src(areas[i]), num_img;                               // 숫자영상

            AreaPreProcess(roi_img, num_img);                                       // 숫자영상 전처리

            num_img = PlaceMiddle(roi_img);                                         // 숫자객체를 영상의 중앙으로 옮기는 연산

            cv::resize(num_img, num_img, cv::Size(28, 28), cv::INTER_LANCZOS4);     // 숫자영상을 28 X 28 크기로 변형

            cv::normalize(num_img, num_img, 0, 1, cv::NORM_MINMAX, CV_32FC1);       // 숫자영상의 원소값을 0과 1사이의 float 타입으로 변환

            std::vector<torch::IValue> input;                                       // float 타입 숫자영상이 텐서타입으로 변환된 값을 저장하는 변수

            input.push_back(
                torch::from_blob(num_img.data, { 1, 1, 28, 28 }, torch::kFloat32).to(torch::kCPU));

            c10::IValue forward = m_module.forward(input);                          // cnn 추론 연산 수행

            torch::Tensor output = forward.toTensor();                              

            auto pred = output.argmax(1, true);

            digits_char[i] = (char)(pred.cpu()[0][0].template item<int>() + 48);    // 숫자인식 결과 획득
        }
    }
    else return std::string();


    digits_str = digits_char;


    for (int i = 0, j = 0; i < 16; i++)                                             // 숫자 네 자리 마다 - 기호 삽입
    {
        if (i > 0 && i % 4 == 0)
        {
            digits_str.insert(i + j, "-");
            ++j;
        }
    }

    return digits_str;
}

