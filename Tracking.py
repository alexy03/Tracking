import cv2
import mediapipe as mp
import numpy as np
import math

# Inisialisasi MediaPipe Hands (maksimal 2 tangan)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Mapping landmark tiap jari (tiga titik per jari)
# Finger names: thumb, index, middle, ring, pinky
fingers = {
    "thumb":  (2, 3, 4),
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20)
}

ANGLE_THRESHOLD = 160  # Sudut minimal agar tiga titik dianggap "lurus"

def calculate_angle(a, b, c):
    """Menghitung sudut di titik b dari segitiga a-b-c (masing-masing (x,y))."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def get_valid_fingers_per_hand(frame, results):
    """
    Memproses tiap tangan yang terdeteksi dan mengembalikan daftar set jari valid per tangan.
    - Menggambar overlay garis tracking (hand connections) dengan warna hijau, tebal 4 px dan transparansi 40%.
    - Menggambar lingkaran biru di ujung jari yang valid.
    """
    h, w, _ = frame.shape
    valid_fingers_by_hand = []
    
    # Buat overlay untuk menggambar garis tracking
    overlay = frame.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            current_set = set()
            # Gambar garis tracking tangan (warna hijau, tebal 4 px)
            mp_draw.draw_landmarks(
                overlay,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(thickness=4, color=(0,255,0)),
                mp_draw.DrawingSpec(thickness=4, color=(0,255,0))
            )
            for finger_name, (idx_a, idx_b, idx_c) in fingers.items():
                a = hand_landmarks.landmark[idx_a]
                b = hand_landmarks.landmark[idx_b]
                c = hand_landmarks.landmark[idx_c]
                a_coord = (int(a.x * w), int(a.y * h))
                b_coord = (int(b.x * w), int(b.y * h))
                c_coord = (int(c.x * w), int(c.y * h))
                
                angle = calculate_angle(a_coord, b_coord, c_coord)
                if angle > ANGLE_THRESHOLD:
                    current_set.add(finger_name)
                    # Tandai ujung jari dengan lingkaran biru
                    cv2.circle(frame, c_coord, 5, (255, 0, 0), -1)
            valid_fingers_by_hand.append(current_set)
    
    # Gabungkan overlay dengan transparansi 40%
    frame[:] = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    return valid_fingers_by_hand

# Inisialisasi cv2.freetype untuk font Montserrat Italic (jika tersedia)
use_freetype = False
try:
    ft = cv2.freetype.createFreeType2()
    ft.loadFontData(fontFileName="Montserrat-Italic.ttf", id=0)
    use_freetype = True
except Exception as e:
    print("cv2.freetype tidak tersedia atau file font tidak ditemukan. Fallback ke cv2.putText.")

cap = cv2.VideoCapture(0)

# Variabel state untuk mode dan sequence toggle
# mode_text: False = Mode 1 (angka), True = Mode 2 (teks)
mode_text = False
sequence_buffer = []
last_count = None  # Untuk menghindari penambahan berulang

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Dapatkan daftar set jari valid per tangan
    valid_fingers_list = get_valid_fingers_per_hand(frame, results)
    # Total jari valid dari semua tangan
    finger_count = sum(len(s) for s in valid_fingers_list)
    
    # Update sequence_buffer jika jumlah jari berubah (untuk toggle mode)
    if last_count is None or finger_count != last_count:
        sequence_buffer.append(finger_count)
        last_count = finger_count
        if len(sequence_buffer) > 4:
            sequence_buffer.pop(0)
        # Toggle mode jika sequence [5, 3, 2, 3] terdeteksi
        if sequence_buffer == [5, 3, 2, 3]:
            mode_text = not mode_text
            sequence_buffer = []
    
    # Pilih output berdasarkan mode
    if not mode_text:
        # Mode 1: menampilkan total angka jari (0–10)
        message = str(finger_count)
    else:
        # Mode 2: menampilkan pesan sesuai gesture
        # Prioritas pengecekan (dilakukan per tangan)
        message = ""
        # Cek setiap tangan secara terpisah; jika lebih dari satu gesture terdeteksi, prioritas sesuai urutan berikut:
        # 1. "hai": jika satu tangan memiliki semua jari (5) diangkat
        # 2. ":)" : jika satu tangan memiliki hanya {thumb, pinky} yang diangkat
        # 3. "saya": jika satu tangan memiliki tepat {thumb, index, middle} yang diangkat
        # 4. "perkenalkan": jika satu tangan memiliki tepat {ring, pinky} yang diangkat
        # 5. "alex": jika satu tangan memiliki tepat {pinky, index} yang diangkat
        for hand in valid_fingers_list:
            if hand == {"thumb", "index", "middle", "ring", "pinky"}:
                message = "hai"
                break
        if not message:
            for hand in valid_fingers_list:
                if hand == {"thumb", "pinky"}:
                    message = ":)"
                    break
        if not message:
            for hand in valid_fingers_list:
                if hand == {"thumb", "index", "middle"}:
                    message = "saya"
                    break
        if not message:
            for hand in valid_fingers_list:
                if hand == {"ring", "pinky"}:
                    message = "perkenalkan"
                    break
        if not message:
            for hand in valid_fingers_list:
                if hand == {"pinky", "index"}:
                    message = "alex"
                    break
    
    # Tampilkan pesan di bagian bawah tengah layar
    if message:
        (text_width, text_height), baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        scale_factor = 30 / text_height  # Agar tinggi huruf sekitar 30 px
        (msg_width, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale_factor, 2)
        msg_x = (w - msg_width) // 2
        msg_y = h - 10  # 10 px dari bawah
        if use_freetype:
            ft.putText(frame, message, (msg_x, msg_y), 30, (0,140,255), thickness=2,
                       line_type=cv2.LINE_AA, bottomLeftOrigin=False)
        else:
            cv2.putText(frame, message, (msg_x, msg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale_factor, (0,140,255), 2)
    
    cv2.imshow("Hand Gesture Calculator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
