import cv2

# Завдання 1: Прочитати та відобразити картинку/відео з файлу.
image_path = 'photo.jpg'
video_path = 'video.mp4'

image = cv2.imread(image_path)
video = cv2.VideoCapture(video_path)

# Завдання 2: Зміна кольору фото/відео (конвертувати з RGB в Ч/Б)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Завдання 3: Трешхолд (Threshold)
_, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Завдання 4: Пошук контурів на фото/відео
contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)

# Завдання 5: Детекція лиць (Каскади Хаара), об'єктів
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Завдання 6: Блюр (можете блюрити лиця після їх находжень)
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# Завдання 7: Трекінг об'єктів на відео (можна використовувати як Каскади Хаара так само і різні моделі, ми працюємо з YOLO)


# Завдання 8: Вирізання фону на фото/відео


# Завдання 9: Морфологічні операції
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

# Завдання 10: Обрізати певну область на фото (ROI)
roi = image[100:300, 200:400]

# Завдання 11: Додавати на фрейм текст, лінії, квадрати чи коло
cv2.putText(image, 'OpenCV', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.line(image, (0, 0), (200, 200), (0, 255, 0), 2)
cv2.rectangle(image, (100, 100), (300, 300), (0, 0, 255), 2)
cv2.circle(image, (400, 400), 50, (255, 255, 0), 2)

# Відображення результатів
cv2.imshow('Original Image', image)
cv2.imshow('Gray Image', gray_image)
cv2.imshow('Threshold Image', threshold_image)
cv2.imshow('Contour Image', contour_image)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Morphological Image', morph_image)
cv2.imshow('ROI', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
