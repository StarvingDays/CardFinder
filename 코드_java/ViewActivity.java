package com.sjlee.cardfinder;

import androidx.annotation.NonNull;
import android.annotation.SuppressLint;
import androidx.annotation.RequiresApi;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.Rect;

import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.util.Size;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import com.google.common.util.concurrent.ListenableFuture;
import com.googlecode.tesseract.android.TessBaseAPI;

import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ViewActivity extends AppCompatActivity implements ImageAnalysis.Analyzer {

    private ListenableFuture<ProcessCameraProvider> m_camera_provider_future;
    private Point m_window_size;                                                                                         // 스마트폰 화면의 가로 세로 길이를 담는 객체
    private ImageView m_roi_image_view, m_line_image_view;                                                               // 체크카드가 놓일 관심영역
    private ProcessCameraProvider m_camera_provider;                                                                     // 카메라 프로바이더 객체
    private PreviewView m_preview_view;                                                                                  // 미리보기뷰 객체
    private Preview m_preview;                                                                                           // 미리보기 객체
    private ImageAnalysis m_image_analizer;                                                                              // 이미지 아날라이저 객체
    private ViewGroup.LayoutParams m_preview_layout;                                                                     // 뷰의 레이아웃을 변경하는 객체
    private RelativeLayout.LayoutParams m_roi_layout;                                                                    // 이미지 분석 객체 가로 세로 길이
    private int m_imageAnalysis_width, m_imageAnalysis_height;
    private int m_image_view_left, m_image_view_top, m_image_view_w, m_image_view_h;                                     // 미리보기의 관심영역 x축, y축, 가로길이, 세로길이
    private Button m_button_back;                                                                                        // 메인화면으로 이동하는 버튼
    private Canvas m_roi_canvas, m_line_canvas;                                                                          // 미리보기 관심영역에 배경을 그리는 캔버스
    private Bitmap m_roi_bitmap, m_line_bitmap;                                                                          // 미리보기 관심영역에 배경을 그리는 비트맵
    private Paint m_paint;                                                                                               // 관심영역의 모서리 지점의 좌표를 저장하는 객체
    private boolean m_executor_on;                                                                                       // Java 스레드풀 생성 여부와 관련한 boolean 값

    private ViewTreeObserver m_viewTreeObserver;
    private ExecutorService m_thread_pool;                                                                               // Java 스레드풀 객체

    private byte[] m_checkcard_image_buffer;

    private native void InitializeCardFinder(int row, int col);                                                          // CardFinder 객체 초기화 함수
    private native void CallCardFindProcess(byte[] array);                                                               // ImageAnalysis에서 획득한 영상을 분석하는 JNI nateve 함수
    private native float[] GetCoordinates();                                                                             // 교점을 초기화하는 JNI native 함수
    private native byte[] GetImageBuffer();                                                                              // CardFinder 객체로 분석을 완료한 체크카드 이미지를 1차원 배열로 반환하는 함수

    Executor getExecutor() {                                                                                             // 특정 테스크를 비동기적으로 실행시키기 위한
        return ContextCompat.getMainExecutor(this);
    }


    @RequiresApi(api = Build.VERSION_CODES.R)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_view);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        m_preview_view = (PreviewView) findViewById(R.id.previewView);                                                    // 카메라 미리보기 뷰
        m_roi_image_view = (ImageView) findViewById(R.id.roi_view);
        m_line_image_view = (ImageView) findViewById(R.id.line_view);

        SetResolution();

        m_button_back = (Button) findViewById(R.id.button);

        m_button_back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                m_executor_on = false;
                Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                intent.putExtra("Cancel", "");
                setResult(RESULT_CANCELED, intent);
                finish();
            }
        });

        m_camera_provider_future = ProcessCameraProvider.getInstance(this);

        m_camera_provider_future.addListener(() -> {                                                                      // https://developer.android.com/training/camerax/architecture?hl=ko#java

            try {
                m_camera_provider = m_camera_provider_future.get();

                m_camera_provider.unbindAll();

                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                m_preview = new Preview.Builder().build();


                m_preview.setSurfaceProvider(m_preview_view.getSurfaceProvider());

                m_image_analizer = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(m_imageAnalysis_width, m_imageAnalysis_height))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();



                m_image_analizer.setAnalyzer(getExecutor(), this);

                Camera camera =
                        m_camera_provider.bindToLifecycle((LifecycleOwner)
                                this, cameraSelector, m_preview, m_image_analizer);


                camera.getCameraControl().cancelFocusAndMetering();

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, getExecutor());

        m_viewTreeObserver = m_preview_view.getViewTreeObserver();
        m_viewTreeObserver.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {

                m_image_view_h = (int) (m_preview_view.getHeight() * 0.25f);                                                     // 미리보기 화면에서 관심영역의 세로길이
                m_image_view_w = (int) (m_image_view_h * 1.618f);                                                                // 미리보기 화면에서 관심영역의 가로길이
                m_image_view_left = (m_preview_view.getLeft() + (m_preview_view.getWidth() / 2)) - (m_image_view_w / 2);         // 미리보기 화면에서 관심영역이 위치할 x축
                m_image_view_top = (m_preview_view.getTop() + (m_preview_view.getHeight() / 2)) - (m_image_view_h / 2);          // 미리보기 화면에서 관심영역이 위치할 y축


                m_roi_layout = new RelativeLayout.LayoutParams(m_image_view_w, m_image_view_h);                                  // 레이아웃 파라미터 설정
                m_roi_layout.setMargins(m_image_view_left, m_image_view_top, 0, 0);

                m_roi_image_view.setLayoutParams(m_roi_layout);                                                                  // ImageView의 위치와 크기를 변경
                m_line_image_view.setLayoutParams(m_roi_layout);

                m_roi_bitmap = Bitmap.createBitmap(m_roi_layout.width, m_roi_layout.height, Bitmap.Config.ARGB_8888);            // ImageView의 가로세로 크기의 비트맵 생성
                m_line_bitmap = Bitmap.createBitmap(m_roi_layout.width, m_roi_layout.height, Bitmap.Config.ARGB_8888);

                m_roi_canvas = new Canvas(m_roi_bitmap);                                                                         // 캔버스 생성
                m_line_canvas = new Canvas(m_line_bitmap);

                m_paint = new Paint();                                                                                           // 브러쉬 생성
                m_paint.setColor(Color.GREEN);                                                                                   // 브러쉬 색상
                m_paint.setAntiAlias(true);                                                                                      // 안티애일리어싱 설정
                m_paint.setStyle(Paint.Style.STROKE);                                                                            // 페인팅 스타일 설정
                m_paint.setStrokeWidth(10);                                                                                      // 브러쉬 굵기 설정
                m_paint.setARGB(255, 0, 255, 0);                                                                                 // 브러쉬 투명도 설정

                m_roi_canvas.drawRect(new Rect(0, 0, m_roi_layout.width, m_roi_layout.height), m_paint);                         // ImageView의 가장자리에 초록색 직선을 그리기
                m_roi_image_view.setImageBitmap(m_roi_bitmap);


                InitializeCardFinder(m_imageAnalysis_width, m_imageAnalysis_height);


                m_preview_view.getViewTreeObserver().removeOnGlobalLayoutListener(this);                                         // 관찰이 더 이상 필요하지 않으면 리스너 제거

                m_executor_on = true;

                m_thread_pool = Executors.newFixedThreadPool(1);

                m_thread_pool.submit(new Runnable() {
                    @Override
                    public void run() {
                        while (m_executor_on)
                        {
                            try
                            {
                                Thread.sleep(10);

                                float[] coord = GetCoordinates();

                                m_checkcard_image_buffer = GetImageBuffer();

                                if (coord != null && m_checkcard_image_buffer != null)
                                {
                                    DrawCoordinates(                                                                                 // 교점과 교점들을 이은 직선을 그리기
                                            coord[0], coord[1], coord[2], coord[3],
                                            coord[4], coord[5], coord[6], coord[7]);

                                    runOnUiThread(new Runnable() {                                                                   // UI 스레드에서 미리보기 뷰 프리징 실행
                                        @Override
                                        public void run() {
                                            m_camera_provider.unbind(m_preview);
                                        }
                                    });
                                    TessBaseAPI tess = ((MainActivity)MainActivity.m_main_context).m_tess;

                                    tess.setImage(                                                                                   // Teseract 객체에 체크카드 이미지 버퍼를 삽입
                                            BitmapFactory.decodeByteArray( m_checkcard_image_buffer, 0, m_checkcard_image_buffer.length )
                                    );

                                    Intent intent = new Intent(getApplicationContext(), MainActivity.class);                         // MainActivity intend 생성

                                    intent.putExtra("Result", tess.getUTF8Text());                                                   // Tesseract OCR 분석 결과값을 Intent 객체에 삽입

                                    setResult(RESULT_OK, intent);                                                                    // MainActivity로 정보를 전달

                                    m_executor_on = false;
                                }

                            } catch (InterruptedException e) {

                            }
                        }

                        finish();
                    }
                });
            }
        });



    }

    @Override
    protected void onStart()
    {
        super.onStart();
    }

    @Override
    protected void onPause()
    {
        super.onPause();

    }

    @Override
    protected  void onDestroy()
    {
        super.onDestroy();

        m_image_analizer.clearAnalyzer();

        m_camera_provider.unbindAll();

        m_thread_pool.shutdown();

        m_camera_provider_future.cancel(true);
    }



    private void SetResolution()                                                                                          //  16:9와 4:3 화면비율에 맞춰 PreviewView와 ImageAnalysis의 가로 세로 길이를 변경하는 함수
    {
        m_window_size = new Point();

        getWindowManager().getDefaultDisplay().getRealSize(m_window_size);

        m_preview_layout = (RelativeLayout.LayoutParams) m_preview_view.getLayoutParams();

        if ((float) (m_window_size.y) / (float) (m_window_size.x) == (4.0f / 3.0f)) {
            m_preview_layout.width = m_window_size.x;
            m_preview_layout.height = m_window_size.y;

            m_imageAnalysis_width = 1080;
            m_imageAnalysis_height = 1440;
        } else {
            m_preview_layout.width = m_window_size.x;
            m_preview_layout.height = (int) (m_window_size.x * (16.0f / 9.0f));

            m_imageAnalysis_width = 1080;
            m_imageAnalysis_height = 1920;
        }
        m_preview_view.setLayoutParams(m_preview_layout);
    }
    @Override
    public void onBackPressed()                                                                                          // 스마트폰의 뒤로가기 버튼을 누를 시 동작하는 함수
    {
        m_executor_on = false;

        Intent intent = new Intent(getApplicationContext(), MainActivity.class);
        intent.putExtra("Cancel", "");
        setResult(RESULT_CANCELED, intent);
        finish();
    }

    @SuppressLint("UnsafeOptInUsageError")
    @Override
    public void analyze(@NonNull ImageProxy image)                                                                       // 머신러닝 등의 분석을 위해 CPU에서 액세스할 수 있는 버퍼를 획득하는 함수
    {
        CallCardFindProcess(ImageToGrayscaleByte(image.getImage()));

        image.close();
    }

    public byte[] ImageToGrayscaleByte(Image image) {                                                                    // Image에서 GrayScale buffer를 얻는 함수

        Image.Plane[] planes = image.getPlanes();                                                                        // 채널 plane획득

        ByteBuffer yBuffer = planes[0].getBuffer();                                                                      // yChannel buffer 획득

        int ySize = yBuffer.remaining();                                                                                 // yBuffer 크기

        byte[] grayscale = new byte[ySize];                                                                              // ySize만큼 할당

        yBuffer.get(grayscale, 0, ySize);                                                                                // grayscale buffer에 yBuffer에 들어있는 원소들을 모두 삽입

        return grayscale;                                                                                                // grayscale buffer 반환
    }


    private void DrawCoordinates(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4)
    {
        runOnUiThread(new Runnable() {                                                                                   // UI 스레드에서 동작

            @Override
            public void run() {
                m_paint.setStrokeWidth(5);                                                                               // 브러쉬 크기 변경
                m_paint.setARGB(255, 255, 0, 0);                                                                         // 브러쉬 색상 변경
                m_line_canvas.drawLine(x1, y1, x2, y2, m_paint);                                                         // 캔버스에 교점들을 이거 선을 그리기
                m_line_canvas.drawLine(x2, y2, x4, y4, m_paint);
                m_line_canvas.drawLine(x4, y4, x3, y3, m_paint);
                m_line_canvas.drawLine(x3, y3, x1, y1, m_paint);

                m_paint.setStrokeWidth(10);
                m_paint.setARGB(255, 0, 0, 255);
                m_line_canvas.drawCircle(x1, y1,10, m_paint);                                                            // 캔버스에 교점을 그리기
                m_line_canvas.drawCircle(x2, y2,10, m_paint);
                m_line_canvas.drawCircle(x3, y3,10, m_paint);
                m_line_canvas.drawCircle(x4, y4,10, m_paint);
                m_line_image_view.setImageBitmap(m_line_bitmap);
            }
        });
    }


    public int[] GetImageViewSize(){ return new int[]{m_image_view_w, m_image_view_h};}                                  // ImageView의 가로 세로 길이를 반환하는 함수


}

