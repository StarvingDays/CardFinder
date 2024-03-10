package com.sjlee.cardfinder;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;


import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Color;

import android.annotation.TargetApi;
import android.content.pm.PackageManager;
import android.os.Build;

import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import com.googlecode.tesseract.android.TessBaseAPI;

import android.graphics.BitmapFactory;

import static android.Manifest.permission.CAMERA;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CODE = 1234;
    private Button m_view_button, m_clear_button, m_exit_button;                                    // 메튜 버튼, 지우기 버튼, 종료버튼
    private TextView m_number_text_view, m_name_text_view, m_valid_text_view;                       // 체크카드 숫자결과 텍스트뷰, 체크카드 이름 텍스트 뷰
    private String m_manifest_write = Manifest.permission.WRITE_EXTERNAL_STORAGE;                   // 외부 저장소 쓰기 사용 권한
    private String m_manifest_read = Manifest.permission.READ_EXTERNAL_STORAGE;                     // 외부 저장소 읽기 사용 권한
    private String m_manifest_camera = CAMERA;                                                      // 카메라 사용 권한
    static private boolean m_have_permission;

    private TessBaseAPI m_tess;
    private String m_data_path;
    private String m_lan;
    private ActivityResultLauncher<Intent> m_launcher =
            registerForActivityResult(
                    new ActivityResultContracts.StartActivityForResult(), result ->
                    {
                        int code = result.getResultCode();

                        if(code == RESULT_OK)                                                       // Activity Code가 RESULT_OK 이면 Main Activity 화면에 숫자와 문자를 표시한다
                        {
                            try {

                                byte[] byte_arr = result.getData().getByteArrayExtra("Result");     // ViewActivity에서 받아온 체크카드 이미지 버퍼

                                m_tess.setImage(                                                    // tesseract 객체에 bitmap으로 변환한 byte배열을 삽입
                                        BitmapFactory.decodeByteArray( byte_arr, 0, byte_arr.length ));

                                String tess_result = m_tess.getUTF8Text();                          // tesseract ocr 실행

                                String numbers = null, name = null, valid_date = null;

                                Pattern pattern;                                                    // 문자열의 검중을 수행하는 Pattern 객체
                                Matcher matcher;                                                    // 문자열 내에 일치하는 문자열을 확인하기 위해 정규식을 이용하여 찾고 존재 여부를 반환해 주는 Matcher 객체
                                
                                pattern = Pattern.compile("(\\d{4}\\s\\d{4}\\s\\d{4}\\s\\d{4})");   // 0000 0000 0000 0000
                                matcher = pattern.matcher(tess_result);

                                if(matcher.matches())
                                {
                                    numbers = matcher.group();
                                }
                                
                                pattern = Pattern.compile("([A-Z]{6}\\s|[A-Z]{9}\\s|[A-Z]{12}\\s)"); // 이름 알파벳의 개수가 6, 9, 12개인 패턴
                                matcher = pattern.matcher(tess_result);
                          
                                if(matcher.matches())
                                {
                                    name = matcher.group();
                                }
                                
                                pattern = Pattern.compile("(\\d{2}\\/\\d{2})");                     // 길이가 2개인 숫자 / 길이가 2개인 숫자
                                matcher = pattern.matcher(tess_result);

                                if(matcher.matches())
                                {
                                    valid_date = matcher.group();
                                }
                                
                                m_number_text_view.setText(numbers);
                                m_number_text_view.setTextColor(Color.parseColor("#00ff00"));
                                m_name_text_view.setText(name);
                                m_name_text_view.setTextColor(Color.parseColor("#00ff00"));
                                m_valid_text_view.setText(valid_date);
                                m_valid_text_view.setTextColor(Color.parseColor("#00ff00"));
                            } catch (Exception e) {
                                m_number_text_view.setText("Analizing Failed!");
                                m_number_text_view.setTextColor(Color.parseColor("#ff0000"));
                            }


                        }

                        if(code == RESULT_CANCELED){
                            m_number_text_view.setText(result.getData().getStringExtra("Cancel"));
                            m_number_text_view.setTextColor(Color.parseColor("#ff0000"));
                        }

                    });


    static {
        System.loadLibrary("CardFinder");
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        m_number_text_view = (TextView) findViewById(R.id.numaber_text);                            // 특정 안드로이드의 뷰를 view id를 통해 받아온다
        m_name_text_view = (TextView) findViewById(R.id.name_text);
        m_valid_text_view = (TextView) findViewById(R.id.valid_text);

        m_view_button = (Button) findViewById(R.id.view_button);                                    // view activity로 전환하는 버튼
        m_clear_button = (Button) findViewById(R.id.clear_button);                                  // 분석결과 문자를 지우는 버튼
        m_exit_button = (Button) findViewById(R.id.exit_button);

        m_data_path = getCacheDir() + "/tesseract";
        m_lan = "eng";

        m_view_button.setOnClickListener(new View.OnClickListener()                                 // view 버튼에 클릭 함수 등록 : view Activity로 진입한다
        {
            @Override
            public void onClick(View v)
            {
                if(m_have_permission)                                                               // view Activity로 전환되기 전에 모든 퍼미션을 허가 받고 모델파일을 생성한 뒤에 view로 진입한다
                {                                                                                   // 모델을 파일경로에 생성하기 전에 view로 진입하면 모델을 불러오는 코드가 예외가 발생하느 문제를 차단한다
                    if(checkFile(new File(m_data_path + "/tessdata/")))
                    {
                        m_tess = new TessBaseAPI();

                        boolean successity = m_tess.init(m_data_path, m_lan);

                        Intent intent = new Intent(getApplicationContext(), ViewActivity.class);

                        m_launcher.launch(intent);
                    }
                }
            }
        });

        m_clear_button.setOnClickListener(new View.OnClickListener()                                // clear 버튼에 클릭 함수 등록 : 결과 텍스쳐를 지운다
        {
            @Override
            public void onClick(View v) {
                m_number_text_view.setText("");
                m_name_text_view.setText("");
                m_valid_text_view.setText("");
            }
        });

        m_exit_button.setOnClickListener(new View.OnClickListener()                                 // exit 버튼에 클릭 함수 등록   : 앱을 종료한다
        {
            @Override
            public void onClick(View v) {
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setMessage("You Wanna Exit?");
                builder.setTitle("Terminate")
                        .setCancelable(false)
                        .setPositiveButton("Yes", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                finish();
                            }
                        })
                        .setNegativeButton("No", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                dialog.cancel();
                            }
                        });
                AlertDialog alert = builder.create();
                alert.setTitle("Terminate");
                alert.show();
            }
        });
    }


    @Override
    protected void onStart()
    {
        super.onStart();
        CheckPermission();
    }

    @Override
    public void onPause() {
        super.onPause();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        CheckPermission();
    }

    public void onDestroy() {
        super.onDestroy();
    }

    private void checkPermission()
    {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
        {
            int read_per = checkSelfPermission(m_manifest_write);
            int write_per = checkSelfPermission(m_manifest_read);
            int camera_per = checkSelfPermission(m_manifest_camera);

            if (read_per + write_per + camera_per != PackageManager.PERMISSION_GRANTED)             // 모든 퍼미션을 허가 받지 않은 경우,
                requestPermissions(new String[]{                                                    // requestPermission 실행
                                m_manifest_write, m_manifest_read, m_manifest_camera},
                        PERMISSION_REQUEST_CODE);
            else{
                m_have_permission = true;
            }
        }
    }

    @Override
    @TargetApi(Build.VERSION_CODES.M)
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) // 퍼미션 요청의 결과를 확인하는 함수
    {
        if (requestCode == PERMISSION_REQUEST_CODE)
            if((grantResults.length > 0) && (grantResults[0] + grantResults[1] + grantResults[2] == PackageManager.PERMISSION_GRANTED))
            {
                m_have_permission = true;
            }
            else
            {
                m_have_permission = false;
            }

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }



    boolean checkFile(File dir)
    {

        if(!dir.exists() && dir.mkdirs()) {                                                           //디렉토리가 없으면 디렉토리를 만들고 그후에 파일을 카피
            copyFiles();
        }

        if(dir.exists()) {                                                                            //디렉토리가 있지만 파일이 없으면 파일카피 진행
            String datafilepath = m_data_path + "/tessdata/" + m_lan + ".traineddata";
            File datafile = new File(datafilepath);
            if(!datafile.exists()) {
                copyFiles();
            }
        }
        return true;
    }
    void copyFile()
    {
        AssetManager assetMgr = this.getAssets();                                                     // 에셋폴더에 접근

        InputStream is = null;
        OutputStream os = null;

        try {
            is = assetMgr.open(m_lan +  ".traineddata");                                              // 에셋 매니저 객체로 에셋 폴더 내부에 있는 tesseract 학습 데이터 불러오기

            String destFile = m_data_path + "/tessdata/" + m_lan + ".traineddata";                    // 에셋 폴더에 있는 학습자료를 저장할 스마트폰 내부 디렉토리 경로

            os = new FileOutputStream(destFile);

            byte[] buffer = new byte[1024];
            int read;
            while ((read = is.read(buffer)) != -1) {                                                  // 버퍼의 끝지점까지 읽기 반복
                os.write(buffer, 0, read);
            }
            is.close();
            os.flush();
            os.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
