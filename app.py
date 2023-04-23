import cv2
import face_recognition as fr
import random
import numpy as np
import os
from datetime import datetime

path = "imagesDb"
images = []
clases = []
lista = os.listdir(path)

comp1 = 100

for list in lista:
    imgdb = cv2.imread(f"{path}/{list}")
    images.append(imgdb)
    clases.append(os.path.splitext(list)[0])

print(clases)


def encodeFace(images):
    listCodeImg = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cod = fr.face_encodings(img)[0]
        listCodeImg.append(cod)

    return listCodeImg


def register(name):
    with open("register.csv", "r+") as h:
        data = h.readlines()

        listName = []
        savedTime = []

        for line in data:
            insert = line.split(",")
            listName.append(insert[0])
            savedTime.append(insert)

        if name not in listName:
            info = datetime.now()
            date = info.strftime("%Y:%M:%D")
            hour = info.strftime("%H:%M:%S")

            h.writelines(f"\n{name},{date},{hour},Entrada")

        if name in listName:
            for dateDb in range(len(savedTime)-1, 0, -1):
                if savedTime[dateDb][0] == name:
                    info = datetime.now()
                    date = info.strftime("%M:%D:%Y")
                    hour = info.strftime("%H:%M:%S")

                    hourdb = int(savedTime[dateDb][2].split(":")[1])
                    hourlocal = int(hour.split(":")[1])

                    difHour = abs(hourdb - hourlocal)

                    if difHour >= 1:
                        if savedTime[dateDb][3] == "Entrada" and savedTime[dateDb][0] == name:
                            h.writelines(f"\n{name},{date},{hour},Salida")

                            print("////////////////////////////")
                            print(savedTime[dateDb][3])
                            print(f"{difHour} minutos de diferencia")
                            print(name)
                            print(info)

                        if savedTime[dateDb][3] == "Salida" and savedTime[dateDb][0] == name:
                            h.writelines(
                                f"\n{name},{date},{hour},Entrada")

                            print("////////////////////////////")
                            print(savedTime[dateDb][3])
                            print(f"{difHour} minutos de diferencia")
                            print(name)
                            print(info)

                    break


encodeFaces = encodeFace(images)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame2 = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    faces = fr.face_locations(rgb)
    facescod = fr.face_encodings(rgb, faces)

    for facecod, faceloc in zip(facescod, faces):
        comparison = fr.compare_faces(encodeFaces, facecod)

        simi = fr.face_distance(encodeFaces, facecod)
        min = np.argmin(simi)

        if comparison[min]:
            name = clases[min].upper()
            # print(name)

            yi, xf, yf, xi = faceloc
            yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4

            rating = comparison.index(True)

            if comp1 != rating:
                r = random.randrange(0, 225, 50)
                g = random.randrange(0, 225, 50)
                b = random.randrange(0, 225, 50)

                comp1 = rating

            if comp1 == rating:
                cv2.rectangle(frame, (xi, yi), (xf, yf), (r, g, b), 3)
                cv2.rectangle(frame, (xi, yf-35), (xf, yf),
                              (r, g, b), cv2.FILLED)
                cv2.putText(frame, name, (xi+6, yf-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                register(name)

    cv2.imshow("Reconocimiento Facial", frame)
    ecs = cv2.waitKey(5)
    if ecs == 27:
        break

cv2.destroyAllWindows()
cap.relaase()
