import cv2
import serial
import tensorflow as tf
import numpy as np
import time

# Inisialisasi koneksi serial dengan Arduino
arduino = serial.Serial('COM5', 9600)  # Sesuaikan dengan port serial Arduino

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Load model dari Teachable Machine (misalnya model TensorFlow atau Keras)
model = tf.keras.models.load_model("C:\\Users\\user\\OneDrive\\Dokumen\\Kuliah\\Semester 4\\Kecerdasan Buatan & Machine Learning (Artificial Intelligence & Machine Learning)\\Teach Able Machine\\converted_keras\\keras_model.h5")



# Fungsi untuk mengirim perintah ke Arduino
def send_command(command):
    arduino.write(command.encode())
    time.sleep(0.1)  # Tunggu sebentar untuk memastikan data terkirim

# Label kelas (misalnya, 'kaleng' dan 'tissue')
class_labels = ['kaleng', 'tissue']

# Loop untuk membaca frame dari kamera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame ke ukuran yang sesuai dengan model Teachable Machine
    resized_frame = cv2.resize(frame, (224, 224))

    # Normalisasi frame
    normalized_frame = resized_frame / 255.0

    # Lakukan inferensi menggunakan model
    predictions = model.predict(np.expand_dims(normalized_frame, axis=0))

    # Ambil label kelas dengan probabilitas tertinggi
    predicted_label = class_labels[np.argmax(predictions)]

    # Jika wajah terdeteksi, kirim perintah ke Arduino untuk menggerakkan servo
    if predicted_label == 'kaleng':
        send_command('a')  # Perintah untuk bergerak ke posisi tengah
    else:
        send_command('b')  # Perintah untuk bergerak ke kanan (misalnya)

    # Tampilkan frame dengan label prediksi
    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Interface', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop kamera dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
