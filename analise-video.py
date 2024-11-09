import cv2
from utils import FaceRecognition, ActivityRecognition, summarize_activities


def main():
    # Inicializa o reconhecimento facial e de atividades
    face_recognition = FaceRecognition()
    activity_recognition = ActivityRecognition()

    # Variáveis para armazenar contagens e última atividade detectada
    activity_summary = {}
    last_expression = None
    last_activity = None
    frame_count = 0  # Contador de frames

    # Captura o vídeo
    cap = cv2.VideoCapture('video/videoFIAP.mp4')

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        frame_count += 1

        # Detecta rostos
        faces = face_recognition.detect_faces(frame)

        # Exibe o contador de frames no canto inferior esquerdo
        cv2.putText(frame, f"Total de frames: {frame_count}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Para cada rosto detectado, analisa a expressão
        for face_bbox in faces:
            expression = face_recognition.analyze_expression(frame, face_bbox)

            # Incrementa a contagem apenas se a expressão atual for diferente da última detectada
            if expression != last_expression:
                if expression not in activity_summary:
                    activity_summary[expression] = 0
                activity_summary[expression] += 1
                last_expression = expression  # Atualiza a última expressão

            # Desenha a caixa delimitadora e a expressão no frame
            cv2.rectangle(frame, (face_bbox[0], face_bbox[1]),
                          (face_bbox[0] + face_bbox[2], face_bbox[1] + face_bbox[3]),
                          (0, 255, 0), 2)
            cv2.putText(frame, expression, (face_bbox[0], face_bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Detecta atividades motoras
        activity = activity_recognition.detect_activity(frame)

        # Incrementa a contagem apenas se a atividade atual for diferente da última detectada
        if activity != last_activity:
            if activity not in activity_summary:
                activity_summary[activity] = 0
            activity_summary[activity] += 1
            last_activity = activity  # Atualiza a última atividade

        # Exibe o resumo de atividades no canto superior esquerdo do frame
        y_offset = 30
        for activity_name, count in activity_summary.items():
            text = f"{activity_name}: {count}"

            # Se a atividade é a atual, ela vai aparecer em vermelho escuro
            if activity_name == last_activity:  # Aqui você pode usar a variável `last_activity`
                # Cor vermelha escura (BGR: (0, 0, 139))
                color = (0, 0, 139)
            else:
                # Cor padrão (azul)
                color = (255, 0, 0)

            # Coloca o texto na tela
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20  # Move a posição vertical para a próxima linha

        # Exibe o frame com detecções e resumo
        cv2.imshow('Face and Activity Recognition', frame)

        # Para finalizar, pressione 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Exibe o resumo final no console após o término do vídeo
    print(f"Resumo Final das Atividades (Total de frames: {frame_count}):", activity_summary)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
