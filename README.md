# Projeto de Reconhecimento Facial e de Atividades em Vídeo

Este projeto realiza **reconhecimento facial** e **detecção de atividades** a partir de um vídeo, utilizando as bibliotecas **OpenCV**, **MediaPipe** e **DeepFace**. Ele detecta rostos, identifica expressões faciais e reconhece atividades motoras. No final da execução, um resumo das atividades e a expressão facial mais frequente são salvos em um arquivo `JSON`.

## Objetivo principal
Este algoritmo objetivou implementar reconhecimento facial e análise de atividades em vídeo, priorizando precisão e velocidade de execução para processamento em tempo real.

## Funcionalidades

- **Detecção de rostos**: Detecta rostos no vídeo e desenha uma caixa delimitadora ao redor de cada rosto identificado.
- **Análise de expressões faciais**: Analisa expressões como "feliz", "triste", etc., e contabiliza a frequência de cada uma ao longo do vídeo.
- **Reconhecimento de atividades motoras**: Detecta atividades físicas, como "sentado", "deitado", "levantando a mão".
- **Resumo de atividades**: Armazena um resumo com a contagem de cada expressão e atividade detectada.
- **Geração de relatório**: Salva um relatório em `activity_summary.json`, detalhando:
  - Total de frames processados
  - Frequência de cada expressão e atividade detectada

## Estrutura do Projeto

- **main.py**: Arquivo principal que processa o vídeo e exibe o reconhecimento facial e de atividades em tempo real.
- **utils.py**: Contém as classes `FaceRecognition`, `ActivityRecognition`, e a função final `summarize_activities`, responsáveis pela detecção, análise e montagem do resumo.
- **video/videoFIAP.mp4**: Vídeo de exemplo para execução do projeto (substituível por qualquer outro vídeo).
- **activity_summary.json**: Arquivo gerado ao final da execução, contendo o resumo das atividades e expressões detectadas.

## Tecnologias Utilizadas

- **Python**: Linguagem principal do projeto.
- **OpenCV**: Para manipulação de vídeo e processamento de imagens.
- **MediaPipe**: Para detecção rostos, poses e reconhecimento de atividades corporais.
- **DeepFace**: Para análise de expressões faciais.

## Dificuldades Encontradas

Durante o desenvolvimento deste projeto, algumas dificuldades surgiram:

- **Desempenho e Otimização**: Processar vídeos em alta resolução e com muitos frames (como o exemplo de 35GB e 3000+ frames) causou problemas de desempenho, resultando em lentidão e travamentos. A necessidade de equilibrar precisão na detecção com velocidade de processamento foi um desafio.

- **Precisão na Detecção de Expressões e Atividades**: As detecções de expressões faciais e atividades motoras nem sempre foram precisas, especialmente em casos onde o rosto ou corpo não estavam totalmente visíveis ou iluminados. O ajuste de parâmetros de detecção e o uso de modelos pré-treinados foram necessários para melhorar a confiabilidade.

- **Integração das Bibliotecas**: Combinar MediaPipe e DeepFace apresentou dificuldades de compatibilidade e integração, exigindo ajustes no código para garantir que as funções de ambas as bibliotecas trabalhassem de forma coordenada.

- **Processamento em Tempo Real**: Exibir o reconhecimento em tempo real e manter a taxa de atualização dos frames exigiu otimizações, como o redimensionamento do frame e a redução da complexidade de alguns cálculos.
