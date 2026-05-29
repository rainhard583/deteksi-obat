import cv2
import requests

API_KEY = "uc5sGP3TXol5G8dbrmo3"

url = f"https://detect.roboflow.com/pills-deteks/3?api_key={API_KEY}"

kamera = cv2.VideoCapture(0)

while True:

    ret, frame = kamera.read()

    cv2.imwrite("frame.jpg", frame)

    with open("frame.jpg", "rb") as image_file:

        response = requests.post(
            url,
            files={
                "file": image_file
            }
        )

    result = response.json()

    obat_bagus = 0
    obat_rusak = 0

    for pred in result["predictions"]:

        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        kelas = pred["class"]

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)

        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        if kelas == "obat_bagus":
            warna = (0, 255, 0)
            obat_bagus += 1

        else:
            warna = (0, 0, 255)
            obat_rusak += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)

        cv2.putText(
            frame,
            kelas,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            warna,
            2
        )

    cv2.putText(
        frame,
        f"Bagus: {obat_bagus}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Rusak: {obat_rusak}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.imshow("Deteksi Obat", frame)

    if cv2.waitKey(1) == 27:
        break

kamera.release()
cv2.destroyAllWindows()