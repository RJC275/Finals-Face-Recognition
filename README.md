# face_recognition
![image](https://static1.colliderimages.com/wordpress/wp-content/uploads/2022/12/best-movie-posters-of-2022.jpg?q=50&fit=contain&w=767&h=&dpr=1.5)
<div align="justify">
✅Movies, also known as films, are a popular form of visual entertainment that use a combination of moving images and sound to tell stories. They come in various genres, such as drama, comedy, and science fiction. The process of making a movie involves stages like scriptwriting, filming, editing, and distribution. Movies have a significant impact on culture and art, providing a medium for storytelling and artistic expression. They can be experienced in theaters, on television, or through streaming platforms. Throughout history, movies have evolved technologically and artistically, becoming a crucial part of global entertainment.



✅Face recognition is a method of identifying or verifying the identity of an individual using their face. Face recognition systems can be used to identify people in photos, video, or in real-time. Law enforcement may also use mobile devices to identify people during police stops. But face recognition data can be prone to error, which can implicate people for crimes they haven’t committed. Facial recognition software is particularly bad at recognizing African Americans and other ethnic minorities, women, and young people, often misidentifying or failing to identify them, disparately impacting certain groups.Additionally, face recognition has been used to target people engaging in protected speech. In the near future, face recognition technology will likely become more ubiquitous. It may be used to track individuals’ movements out in the world like automated license plate readers track vehicles by plate numbers. Real-time face recognition is already being used in other countries and even at sporting events in the United States. 
</div>

<div align="justify">
✅!git clone https://github.com/RJC275/face_recognition.git:
This command clones (downloads) a Git repository from the URL provided (https://github.com/RJC275/face_recognition.git). It creates a local copy of the repository in your current directory.

✅!pip install face_recognition:
This command uses pip (Python's package installer) to install the face_recognition library. This library provides functionalities for face recognition in Python.
</div>


```
!git clone https://github.com/RJC275/face_recognition.git
!pip install face_recognition
%cd face_recognition
```

✅The code below is creating encoding profiles for facial recognition using the face_recognition library in Python. It loads images of five different individuals, encodes their facial features, and creates a database of known faces for identification purposes.

✅Image Loading: The code loads images (like "Choi hyun wook.jpeg", "Park eun bin.jpg", etc.) of notable figures using the face_recognition.load_image_file function.

✅Feature Encoding: For each image, it computes facial encodings—numerical representations of unique facial features—using face_recognition.face_encodings. These encodings are numerical arrays that represent distinct facial characteristics.

✅Database Creation: The encoded facial features are stored in lists (known_face_encodings) alongside corresponding names (known_face_names). These lists serve as a reference database for recognizing known individuals.

✅Identification Labels: The names provided ("Choi Eun Wook," "Park eun bin," etc.) are associated with the encoded facial features, acting as labels for identification.

✅Recognition System: This sets the foundation for a facial recognition system, where these known faces can be compared against other faces to identify and label individuals in images or video frames based on their facial features.

```
import face_recognition
import numpy as np
from google.colab.patches import cv2_imshow
import cv2

# Creating the encoding profiles
face_1 = face_recognition.load_image_file("Choi hyun wook.jpeg")
face_1_encoding = face_recognition.face_encodings(face_1)[0]

face_2 = face_recognition.load_image_file("Park eun bin.jpg")
face_2_encoding = face_recognition.face_encodings(face_2)[0]

face_3 = face_recognition.load_image_file("kim soo hyun.jpeg")
face_3_encoding = face_recognition.face_encodings(face_3)[0]

face_4 = face_recognition.load_image_file("kim su gyeom.png")
face_4_encoding = face_recognition.face_encodings(face_4)[0]

face_5 = face_recognition.load_image_file("park hae soo.jpg")
face_5_encoding = face_recognition.face_encodings(face_5)[0]

known_face_encodings = [
                        face_1_encoding,
                        face_2_encoding,
                        face_3_encoding,
                         face_4_encoding,
                         face_5_encoding
]

known_face_names = [
                    "Choi hyun wook",
                    "Park eun bin",
                    "kim soo hyun",
                    "kim su gyeom",
                    "park hae soo"

]
```


<div align="justify">
The code below analyzes an "unknown_em.jpeg" image using facial recognition techniques with the `face_recognition` library and OpenCV in Python.

✅ **Image Loading:** It loads the "kim su gyeom.png" image and prepares it for analysis using `face_recognition.load_image_file` and `cv2.imread`.

✅**Face Detection:** The code identifies face locations within the unknown image using `face_recognition.face_locations`.

✅**Encoding Faces:** Facial encodings for these detected face locations are computed using `face_recognition.face_encodings`.

✅**Comparison and Identification:** It compares the unknown face encodings against the known faces' encodings using `face_recognition.compare_faces`. If a match is found, it assigns a name to the recognized face based on the closest match in the known face database and if not is display unknown. 

✅**Visual Representation:** For each recognized face, the code draws a rectangle around the face and labels it with the recognized individual's name using OpenCV's `cv2.rectangle` and `cv2.putText` functions, providing a visual representation of the identified faces in the image.
</div>

✅**Choi Hyun Wook:**

```
file_name = "unknown_chw.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/8bb1c222-458e-4d0f-a52a-19bf0b7b1668)
)

✅**Park Eun Bin:**

```
file_name = "unknown_peb.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/0530ad39-549f-42ad-8f39-aeba4455cdca)


✅**Kim Soo Hyun:**

```
file_name = "unknown_ksh.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/3d18cf7a-5be9-44ba-936f-f027a8396a9f)



✅**Kim Su Gyeom:**
```
file_name = "unknown_ksg.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/558dfb66-6295-4df3-ac6d-3ca2e41af9a2)


✅**Park Hae Soo:**
```
file_name = "unknown_phs.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/4dd708c5-26a1-4291-b444-bb162132e742)


### ✅**Ten Unknown Celebrities:**
```
file_name = "k2.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/3d611373-6568-4b21-a7d1-f930a0256aab)


```
file_name = "d1.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/70308feb-db37-45ec-811d-1334028e2c0c)

```
file_name = "d2.png"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/64aca782-e118-4e71-a3db-d774f8e6c0d9)

```
file_name = "d3.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/d898aa5c-a43e-4b39-be7f-c99d12bb2067)

```
file_name = "d4.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/4ce879bd-ee56-43a3-87c6-ca6e3adbdbd7)

```
file_name = "d5.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/1d6561c4-f7c5-43cc-8c75-3edf4d451e6b)

```
file_name = "d6.webp"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/3e66160b-693b-4f78-87af-deef41d6ca59)

```
file_name = "d7.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/1ce546e5-c31c-4272-bf3e-9b2a04b572ab)

```
file_name = "d8.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/c2bd00e1-9e77-4ce4-b118-2fa4915bb772)

```
file_name = "d9.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,220,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top -20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![image](https://github.com/RJC275/face_recognition/assets/144230425/f85bf177-46ef-4682-a495-9b82d93e3600)

