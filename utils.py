import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

# Dicionário de tradução de emoções do DeepFace
emotion_translation = {
    "happy": "Feliz",
    "sad": "Triste",
    "angry": "Bravo",
    "surprise": "Surpreso",
    "fear": "Medo",
    "disgust": "Nojo",
    "neutral": "Neutro"
}

class FaceRecognition:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.4)

    def detect_faces(self, frame):
        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                        int(bboxC.width * iw), int(bboxC.height * ih))
                faces.append(bbox)
        return faces

    def analyze_expression(self, frame, face_bbox):
        x, y, w, h = face_bbox
        face_image = frame[y:y + h, x:x + w]
        try:
            result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            english_emotion = result[0]['dominant_emotion']
            # Traduz a emoção para português usando o dicionário
            translated_emotion = emotion_translation.get(english_emotion, "E. Desconhecida")
            return translated_emotion
        except Exception as e:
            print(f"Erro na análise de expressão: {e}")
            return "E. Desconhecida"


class ActivityRecognition:
    def __init__(self, detection_confidence=0.3, tracking_confidence=0.3, model_complexity=1):
        # Inicializa o módulo de pose com hiperparâmetros customizáveis
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=model_complexity
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def detect_activity(self, frame):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            self.draw_skeleton(frame, results.pose_landmarks)  # Desenha o esqueleto
            return self.categorize_activity(results.pose_landmarks)
        return "Inativo"

    def draw_skeleton(self, frame, landmarks):
        """Desenha o esqueleto do corpo no frame."""
        self.mp_drawing.draw_landmarks(
            frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    def categorize_activity(self, landmarks):
        """Identifica se a pessoa está escrevendo, com a mão levantada, deitada ou inativa."""
        # Obtém as landmarks relevantes
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = landmarks.landmark[
            self.mp_pose.PoseLandmark.NOSE]  # Usa o nariz como referência para a posição da cabeça

        # Checa se as landmarks estão visíveis
        if (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
                left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5 and
                nose.visibility > 0.5):

            # Altura dos ombros
            shoulder_height = (left_shoulder.y + right_shoulder.y) / 2

            # Verifica se alguma das mãos está acima da linha dos ombros
            if left_wrist.y < shoulder_height:
                return "Mao E. Levantada"
            if right_wrist.y < shoulder_height:
                return "Mao D. Levantada"

            # Lógica para determinar se a pessoa está deitada
            if abs(nose.y - shoulder_height) < 0.05:  # Define um limite para considerar os ombros e cabeça na mesma altura
                return "Deitado"

        return "Mov. Anomalo"


def summarize_activities(activities):
    """Gera um resumo das atividades e emoções detectadas em tempo real."""
    summary = {}
    last_activity = None  # Armazena a última atividade detectada

    for activity in activities:
        # Verifica se a atividade mudou em relação à última atividade
        if activity != last_activity:
            if activity not in summary:
                summary[activity] = 0
            summary[activity] += 1  # Incrementa a contagem da nova atividade
            last_activity = activity  # Atualiza a última atividade detectada

    return summary
