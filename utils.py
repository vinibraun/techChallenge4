import cv2
import mediapipe as mp
from deepface import DeepFace

# Dicionário para traduzir emoções detectadas pelo DeepFace do inglês para o português.
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
        # Inicializa o modelo de detecção de rosto do MediaPipe com uma confiança mínima de 0.4.
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.4)

    def detect_faces(self, frame):
        """
        Detecta rostos em um frame.
        - Converte o frame para RGB (necessário para o MediaPipe).
        - Retorna uma lista de bounding boxes dos rostos detectados.
        """
        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = []
        if results.detections:
            for detection in results.detections:
                # Extrai as coordenadas da bounding box relativa ao frame.
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                        int(bboxC.width * iw), int(bboxC.height * ih))
                faces.append(bbox)  # Adiciona as coordenadas do rosto detectado à lista.
        return faces

    def analyze_expression(self, frame, face_bbox):
        """
        Analisa a expressão facial de um rosto detectado.
        - Recorta o rosto detectado no frame.
        - Usa o DeepFace para identificar a emoção dominante.
        - Traduz a emoção para o português usando o dicionário 'emotion_translation'.
        """
        x, y, w, h = face_bbox
        face_image = frame[y:y + h, x:x + w]  # Recorta a imagem do rosto.
        try:
            # Analisa a emoção no rosto usando o DeepFace.
            result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            english_emotion = result[0]['dominant_emotion']  # Obtém a emoção dominante.
            # Traduz a emoção para o português.
            translated_emotion = emotion_translation.get(english_emotion, "E. Desconhecida")
            return translated_emotion
        except Exception as e:
            # Trata erros e retorna uma emoção desconhecida caso algo falhe.
            print(f"Erro na análise de expressão: {e}")
            return "E. Desconhecida"


class ActivityRecognition:
    def __init__(self, detection_confidence=0.3, tracking_confidence=0.3, model_complexity=1):
        """
        Inicializa o módulo MediaPipe Pose para detectar landmarks corporais.
        - detection_confidence: Confiança mínima para detecção inicial.
        - tracking_confidence: Confiança mínima para rastreamento contínuo.
        - model_complexity: Define a complexidade do modelo (impacta desempenho e precisão).
        """
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=model_complexity
        )
        # Utilitários para desenhar landmarks e conexões no frame.
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def detect_activity(self, frame):
        """
        Detecta a atividade do corpo humano no frame.
        - Processa o frame para obter landmarks corporais.
        - Desenha o esqueleto no frame.
        - Classifica a atividade detectada com base na posição das landmarks.
        """
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Converte para RGB antes de processar.
        if results.pose_landmarks:
            self.draw_skeleton(frame, results.pose_landmarks)  # Desenha o esqueleto no frame.
            return self.categorize_activity(results.pose_landmarks)  # Classifica a atividade.
        return "Inativo"  # Retorna "Inativo" se nenhum movimento for detectado.

    def draw_skeleton(self, frame, landmarks):
        """
        Desenha o esqueleto do corpo no frame com conexões e pontos de landmarks.
        - Utiliza as configurações do MediaPipe DrawingUtils.
        """
        self.mp_drawing.draw_landmarks(
            frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    def categorize_activity(self, landmarks):
        """
        Classifica a atividade corporal com base em landmarks selecionadas.
        - Detecta as atividades:
            - "Mão Esquerda Levantada"
            - "Mão Direita Levantada"
            - "Deitado"
            - "Sentado"
            - "Movimento Anômalo" (padrão).
        """
        # Obtém landmarks relevantes
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Checa visibilidade geral antes de analisar
        if all(lm.visibility > 0.5 for lm in [left_shoulder, right_shoulder, left_wrist, right_wrist, nose]):
            # Calcula a altura média dos ombros
            shoulder_height = (left_shoulder.y + right_shoulder.y) / 2

            # Verifica se alguma das mãos está acima da linha dos ombros
            if left_wrist.y < shoulder_height:
                return "Mao E. Levantada"
            if right_wrist.y < shoulder_height:
                return "Mao D. Levantada"

            # Determina se a pessoa está deitada com base na proximidade do nariz e dos ombros
            if abs(nose.y - shoulder_height) < 0.05:
                return "Deitado"

        # Checa visibilidade dos membros inferiores antes de avaliar "Sentado"
        if all(lm.visibility > 0.5 for lm in [left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            # Calcula a altura média dos quadris e ombros
            hip_height = (left_hip.y + right_hip.y) / 2
            shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
            # Avalia a posição dos membros inferiores para identificar "Sentado"
            if (hip_height > shoulder_height and  # Quadris abaixo dos ombros
                    abs(left_knee.y - left_ankle.y) > 0.1 and abs(
                        right_knee.y - right_ankle.y) > 0.1):  # Joelho em ângulo
                return "Sentado"

        # Retorna "Movimento Anômalo" se nenhuma atividade específica for detectada
        return "Mov. Anomalo"
