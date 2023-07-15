#include <CardFinder.hpp>

extern "C"
JNIEXPORT jstring JNICALL
Java_com_sjlee_cardfinder_ViewActivity_AnalizeImage(JNIEnv *env, jobject thiz, jbyteArray array, jint width, jint height) {

    jbyte* yData  = env->GetByteArrayElements(array, nullptr);                      // yData는 GetByteArrayElements함수로 받아온 jbyte 타입 객체를 가리킨다

    unsigned char *yDataPtr = reinterpret_cast<unsigned char *>(yData);             // yData를 unsigned char * 타입으로 타입 캐스팅

    cv::Mat img(height, width, CV_8UC1, yDataPtr, width);                           // width 만큼의 step을 주며 grayscale의 영상 생성

    CardFinder& instance = CardFinder::GetInstance(*env, thiz, width, height);      // 싱글턴 CardFinder 객체 생성

    env->ReleaseByteArrayElements(array, yData, JNI_ABORT);                         // 자원 할당 해제

    cv::Mat roi_img = img(instance.GetCapturedArea());                              // img에서 관심영역을 가리키는 roi_img 생성

    return env->NewStringUTF(instance.Start(roi_img).c_str());                      // 관심영역 이미지를 분석 시작
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_sjlee_cardfinder_ViewActivity_GetCoordinates(JNIEnv *env, jobject thiz, jint width, jint height)
{
    CardFinder& instance = CardFinder::GetInstance(*env, thiz, width, height);      

    std::vector<float> arr = instance.GetCoordinates();                            // 체크카드 네 개 모서리의 교점을 획득

    jfloatArray javaArray = env->NewFloatArray(0);                                 // 비어있는 jfloatArray 생성

    if(!arr.empty())
    {
        int size = arr.size();
        javaArray = env->NewFloatArray(size);                                      // javaArray에 size크기 만큼 할당
        env->SetFloatArrayRegion(javaArray, 0, size, arr.data());                  // javaArray에 arr에 있는 원소들을 삽입
    }
    return javaArray;
}