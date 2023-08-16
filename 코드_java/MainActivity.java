package com.sjlee.cardfinder;

import androidx.appcompat.app.AppCompatActivity;

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

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.w3c.dom.Text;

import static android.Manifest.permission.CAMERA;
public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CODE = 1234;
    private Button m_view_button, m_clear_button, m_exit_button;                                    // 메튜 버튼, 지우기 버튼, 종료버튼
    private TextView m_number_text_view, m_name_text_view, m_valid_text_view;                       // 체크카드 숫자결과 텍스트뷰, 체크카드 이름 텍스트 뷰
    private String m_manifest_write = Manifest.permission.WRITE_EXTERNAL_STORAGE;                   // 외부 저장소 쓰기 사용 권한
    private String m_manifest_read = Manifest.permission.READ_EXTERNAL_STORAGE;                     // 외부 저장소 읽기 사용 권한
    private String m_manifest_camera = CAMERA;                                                      // 카메라 사용 권한
    static private boolean m_have_permission;

    private ActivityResultLauncher<Intent> m_launcher =
            registerForActivityResult(
                    new ActivityResultContracts.StartActivityForResult(), result ->
                    {
                        int code = result.getResultCode();

                        if(code == RESULT_OK)                                                       // Activity Code가 RESULT_OK 이면 Main Activity 화면에 숫자와 문자를 표시한다
                        {
                            String data = result.getData().getStringExtra("Result");

                            JSONObject jsonObject = null;
                            try {
                                jsonObject = new JSONObject(data).getJSONObject("Result");
                                m_number_text_view.setText(jsonObject.getString("CardNumber"));
                                m_number_text_view.setTextColor(Color.parseColor("#00ff00"));
                                m_name_text_view.setText(jsonObject.getString("Name").toUpperCase());
                                m_name_text_view.setTextColor(Color.parseColor("#00ff00"));
                                m_valid_text_view.setText(jsonObject.getString("Valid").toUpperCase());
                                m_valid_text_view.setTextColor(Color.parseColor("#00ff00"));
                            } catch (JSONException e) {
                                m_number_text_view.setText(data);
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

        m_view_button.setOnClickListener(new View.OnClickListener()                                 // view 버튼에 클릭 함수 등록 : view Activity로 진입한다
        {
            @Override
            public void onClick(View v) {
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
                m_have_permission = true;
            }else{
                m_have_permission = false;
            }

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }


}
