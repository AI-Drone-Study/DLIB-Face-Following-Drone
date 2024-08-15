### cvzone 라이브러리를 이용하여 구현

- cvzone은 mediapipe를 이용하여 사람의 얼굴인식(개별적으로 사람 얼굴 구분 못함) 하는 부분을 dlib face landmark(68개)를 이용하여 개별 사람의 얼굴을 인식하도록 수정함

  참고 : https://github.com/cvzone/cvzone
  
- cvzone의 PID 제어 라이브러리를 이용하여 탐색된 사람의 얼굴을 드론이 따라 다님(face follow 드론 제어)

  참고 : https://www.youtube.com/watch?v=7HnLTrPMjyk
- 데모 동영상 : https://youtube.com/shorts/TopaNzO1dAM?si=HepFjE8-rsVdh5qQ

- requirements <br>
  cvzone 1.6.1 <br>
  dlib 19.24.5 <br>
  face-recognition 1.2.0 <br>
  mediapipe 0.10.14 <br>
  djitellopy 2.4.0 <br>
  opencv-python 4.10.0.84 <br>
  numpy 2.0.1 <br>

- Windows 10
  
