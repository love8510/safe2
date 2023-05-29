import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
import simpleaudio as sa
import threading

weights_path = "best_final.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = attempt_load(weights_path, device=device)

img_size = 416
conf_thres = 0.25
iou_thres = 0.45

cap = cv2.VideoCapture(0)

play_obj = None
play_thread = None
lock = threading.Lock()


def play_sound():
    global play_obj
    wave_obj = sa.WaveObject.from_wave_file("audio2.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1).copy() # -1은 BGR to RGB
    # transpose는 (channel,height,width)의 pytorch형식으로 차원을 재정렬
    img = torch.from_numpy(img).to(device) # pytorch tensor로 convert
    img = img.float() / 255.0 # pixel을 0~1 사이로 정규화(normalize)
    if img.ndimension() == 3: # 입력 이미지가 3차원일 때
        # 기본적으로 [batch_size, channels, height, width]의 입력을 예상하나 3차원이면
        img = img.unsqueeze(0) # batch_size를 추가한다.
        # 딥러닝 모델에서 추론할 때 이미지를 병렬로 처리하여 추론 시간을 단축시키는 역할

    head_detected = False

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=None)
    # for *xyxy, conf, cls in reversed(pred[0]):
    #     label = f"{model.names[int(cls)]} {conf:.2f}"
    #     x1, y1, x2, y2 = map(int, xyxy)
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #     cv2.putText(
    #         frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
    #     )
    for *xyxy, conf, cls in reversed(pred[0]):
        label = f"{model.names[int(cls)]} {conf:.2f}"
        x1, y1, x2, y2 = map(int, xyxy)
        # bbox 좌표값을 416x416 이미지 기준으로 변환
        x1 = int(x1 * frame.shape[1] / img_size)
        y1 = int(y1 * frame.shape[0] / img_size)
        x2 = int(x2 * frame.shape[1] / img_size)
        y2 = int(y2 * frame.shape[0] / img_size)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )
        if model.names[int(cls)] == "head": # head가 detection되면
            head_detected = True

    if head_detected:
        with lock:
            if play_obj is None or not play_obj.is_playing(): #재생중이 아니라면
                play_thread = threading.Thread(target=play_sound) # 스레드를 생성하여 재생하도록 함
                # 스레드 적용 안 하면 재생되는 시간 동안은 캠이 정지됨
                play_thread.start()

    cv2.imshow("YOLOv5", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
