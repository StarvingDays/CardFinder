package com.sjlee.cardfinder;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.os.Bundle;
import android.annotation.TargetApi;
import android.content.pm.PackageManager;
import android.os.Build;

import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;


import static android.Manifest.permission.CAMERA;


public class MainActivity extends AppCompatActivity {
    private static final int PERMISSION_REQUEST_CODE = 1234;
    private Button m_view_button, m_clear_button, m_exit_button;                                    // 메튜 버튼, 지우기 버튼, 종료버튼
    private TextView m_main_text_view, m_sub_text_view;                                             // 체크카드 숫자결과 텍스트뷰, 체크카드 이름 텍스트 뷰
    private String m_self_dir;                                                                      // 파이토치 모델을 저장할 경로
    private String m_manifest_write = Manifest.permission.WRITE_EXTERNAL_STORAGE;                   // 외부 저장소 쓰기 사용 권한 
    private String m_manifest_read = Manifest.permission.READ_EXTERNAL_STORAGE;                     // 외부 저장소 읽기 사용 권한
    private String m_manifest_camera = CAMERA;                                                      // 카메라 사용 권한
    static private boolean m_have_permission;                                                       // 권한 여부 확인 boolean값

    static {
        System.loadLibrary("native-lib");                                                           // Java Native Interface와 연동되는 C++ 라이브러리를 호출하는 함수
    }


    // 엑티비티에서 데이터를 받아오기 위해 사용하는 함수 객체이다 
    // 가령 엑티비티 A와 B가 있으면 A에서 B엑티비티를 생성 하고 
    // B엑티비티에서 A엑티비티로 데이터를 받아오고 싶을 때 사용한다
    //
    private ActivityResultLauncher<Intent> m_launcher =
            registerForActivityResult(
                    new ActivityResultContracts.StartActivityForResult(), result ->
                    {
                        int code = result.getResultCode();

                        if(code == RESULT_OK)                                                       // Activity Code가 RESULT_OK 이면 Main Activity 화면에 숫자와 문자를 표시한다
                        {
                            String data = result.getData().getStringExtra("Result");

                            String[] parts = data.split("/");                                       // 0000-0000-0000-0000/000000000 -> '/' 분리

                            m_main_text_view.setText(parts[0]);
                            m_main_text_view.setTextColor(Color.parseColor("#00ff00"));
                            m_sub_text_view.setText(parts[1]);
                            m_sub_text_view.setTextColor(Color.parseColor("#00ff00"));

                        }

                        if(code == RESULT_CANCELED)
                            m_main_text_view.setText(result.getData().getStringExtra("Cancel"));
                    });


    void copyFiles(String filename)                                                                 // Asset 폴더에 저장된 파일을 버퍼로 읽어와서 특정경로에 저장하는 함수
    {
        AssetManager assetMgr = this.getAssets();

        InputStream is = null;
        OutputStream os = null;
        String destFile = m_self_dir + "/" + filename;

        try {
            is = assetMgr.open(filename);
            os = new FileOutputStream(destFile);

            byte[] buffer = new byte[1024];
            int read;
            // 더 이상 읽을 게 없으면 read 함수는 -1을 반환한다
            // 읽을 게 있으면 버퍼에 담고 읽은 길이를 반환한다
            // while 문을 통해 파일의 끝지점까지 버퍼로 읽고 아웃풋스트림에 작성한다
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }

            is.close();
            os.flush();
            os.close();



        } catch (IOException e) {
            e.printStackTrace();

        } 
    }



    @Override
    protected void onCreate(Bundle savedInstanceState)                                              // 안드로이드 생명주기에서 가장먼저 처음으로 한 번 실행되는 함수
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        m_self_dir = getFilesDir().getPath();                                                       // 어플리케이션의 파일디렉토리의 경로를 받아온다

        m_main_text_view = (TextView)findViewById(R.id.main_text);                                  // 특정 안드로이드의 뷰를 view id를 통해 받아온다
        m_sub_text_view = (TextView)findViewById(R.id.sub_text);

        m_view_button = (Button)findViewById(R.id.view_button);                                     // view activity로 전환하는 버튼
        m_clear_button = (Button)findViewById(R.id.clear_button);                                   // 분석결과 문자를 지우는 버튼
        m_exit_button = (Button)findViewById(R.id.exit_button);                                     // 앱 종료 버튼

        m_view_button.setOnClickListener(new View.OnClickListener()                                 // view 버튼에 클릭 함수 등록 : view Activity로 진입한다
        {
            @Override
            public void onClick(View v)
            {
                if(m_have_permission)                                                               // view Activity로 전환되기 전에 모든 퍼미션을 허가 받고 모델파일을 생성한 뒤에 view로 진입한다
                {                                                                                   // 모델을 파일경로에 생성하기 전에 view로 진입하면 모델을 불러오는 코드가 예외가 발생하느 문제를 차단한다
                    Intent intent = new Intent(getApplicationContext(), ViewActivity.class);
                    m_launcher.launch(intent);
                }
            }
        });

        m_clear_button.setOnClickListener(new View.OnClickListener()                                // clear 버튼에 클릭 함수 등록 : 결과 텍스쳐를 지운다
        {
            @Override
            public void onClick(View v)
            {
                m_main_text_view.setText("");
                m_sub_text_view.setText("");
            }
        });

        m_exit_button.setOnClickListener(new View.OnClickListener()                                 // exit 버튼에 클릭 함수 등록   : 앱을 종료한다
        {
            @Override
            public void onClick(View v)
            {
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


    private void CheckPermission()
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
            if((grantResults.length > 0) 
            && (grantResults[0] + grantResults[1] + grantResults[2] == 
            PackageManager.PERMISSION_GRANTED))                                                       // read write camera 퍼미션을 모두 허가 받는 경우 CopyFiles 함수 실행
            {
                copyFiles("number_model.pt");                                                         // Asset 폴더에 있는 pytorch 모델 파일을 어플리케이션 파일 경로에 복사 붙여넣기한다
                copyFiles("alphabet_model.pt");
                m_have_permission = true;
            }else{
                m_have_permission = false;
            }

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }


}
