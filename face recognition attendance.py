import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

jobs_image = face_recognition.load_image_file("photos/jobs.jpg")
jobs_encoding = face_recognition.face_encoding(jobs_image)[0]

bill_image = face_recognition.load_image_file("photos/bill.jpg")
bill_encoding = face_recognition.face_encoding(bill_image)[0]

barrack_image = face_recognition.load_image_file("photos/barrack.jpg")
barrack_encoding = face_recognition.face_encoding(barrack_image)[0]

megan_image = face_recognition.load_image_file("photos/megan.jpg")
megan_encoding = face_recognition.face_encoding(megan_image)[0]

musk_image = face_recognition.load_image_file("photos/musk.jpg")
musk_encoding = face_recognition.face_encoding(musk_image)[0]

known_face_encoding = [
    jobs_encoding,
    bill_encoding,
    barrack_encoding,
    megan_encoding,
    musk_encoding,
]

known_faces_names = [
    "jobs",
    "bill",
    "barrack",
    "megan",
    "musk",
]

students = known_faces_names.copy()

face_location = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_location = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_location)
        face_names = []
        for face_encodings in face_encodings:
            matches = face_recognition.face_distance(known_face_encoding, face_encodings)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encodings)
            best_match_index = np.argwin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
    cv2.imshow("attendance system", "frame")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

