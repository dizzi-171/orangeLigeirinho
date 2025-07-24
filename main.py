from ultralytics import YOLO
import cv2
import time
import torch
import platform
import comunicacaoPython as comPython
# Detecta sistema operacional (Windows, Linux, etc)
sistema = platform.system()
import serial
import numpy as np
import os
import json
import glob
import os
import subprocess
import math
from threadVideo import VideoStream
from flask import Flask, Response, render_template, jsonify
from threading import Thread

PROCURAR_VITIMA = (b'0')
IDENTIFICAR_TRIANGULO_VERMELHO_HORIZONTAL = (b'1')
IDENTIFICAR_TRIANGULO_VERDE_HORIZONTAL = (b'2')
SEGUIR_LINHA = (b'3')

VERMELHO = 0
VERDE = 1

# Define o número de threads para PyTorch (CPU)
torch.set_num_threads(2)
device = 'cpu'
print(f"Usando dispositivo: {device}")

# Carrega o modelo YOLOv8 (arquivo ONNX)
model = YOLO("./best.onnx")

def encontrar_video_por_porta_usb(porta_usb_alvo):
    video_dir = '/sys/class/video4linux'
    listdirorig = (os.listdir(video_dir))
    print(listdirorig.sort())
    for nome in listdirorig:
        print(nome)
        if not nome.startswith('video'):
            continue

        # Caminho do link simbólico real até o dispositivo
        video_device_path = f'{video_dir}/{nome}/device'

        try:
            # Executa: readlink -f /sys/class/video4linux/videoX/device
            caminho_completo = subprocess.check_output(['readlink', '-f', video_device_path], text=True).strip()

            # Verifica se a porta física está presente no caminho
            if porta_usb_alvo in caminho_completo:
                return nome

        except subprocess.CalledProcessError:
            continue

    return None

# video_name = encontrar_video_por_porta_usb("510")  # ex: 'video#'
# # video_index1 = int(video_name1.replace('video', ''))  # vira #
# video_index = (f"/dev/{video_name}") # ex: '/dev/video#'
# cap = cv2.VideoCapture(video_index ) # Isso sim funciona
video_name = encontrar_video_por_porta_usb("520")  # ex: 'video#'
# video_index1 = int(video_name1.replace('video', ''))  # vira #
video_index = (f"/dev/{video_name}") # ex: '/dev/video#'
cap = VideoStream(video_index) # Isso sim funciona
# cap = VideoStream(video_index)  # Inicia
# ret, frame = cap.read()  # Sempre retorna o frame mais atual
# cam = CameraProcess(video_index, width=320, height=240)
# cam.start()


# video_name2 = encontrar_video_por_porta_usb("520")  # ex: 'video3'
# # video_index2 = int(video_name2.replace('video', ''))  # vira 3
# video_index2 = (f"/dev/{video_name2}")  # ex: '/dev/video#'
# print(video_index2)
# cap2 = cv2.VideoCapture(video_index2)  # Isso sim funciona

# if video_index2.isOpened() == False: 
#     print("Erro ao abrir a segunda câmera. Verifique se o caminho está correto ou se a câmera está conectada.")



# Inicializa a captura da webcam (device 0 windows device 1 orange)
# if sistema == "Windows": cap = cv2.VideoCapture(0)
# else: 
#     cap = cv2.VideoCapture(int(camPath1))
#     # if not cap.isOpened() == None: print("Erro ao abrir a primeira câmera. Verifique se o caminho está correto ou se a câmera está conectada.")
#     cap2 = cv2.VideoCapture(int(camPath2))
#     # if not cap2.isOpened(): print("Erro ao abrir a segunda câmera. Verifique se o caminho está correto ou se a câmera está conectada.")

# Define resolução da captura para 320x240
width = 160
height = 120

# --- Criação da pasta de saída para salvar os frames processados ---
base_dir = "outputs"
os.makedirs(base_dir, exist_ok=True)

# Conta as pastas já existentes para criar uma nova numerada
existing_dirs = sorted(glob.glob(os.path.join(base_dir, "output_*")))
next_index = len(existing_dirs) + 1
output_dir = os.path.join(base_dir, f"output_{next_index:02d}")
os.makedirs(output_dir, exist_ok=True)

# Inicializa variáveis para contar frames e guardar resultados
frame_count = 1
resultados = {}
actual_frame = None  # Frame mais recente para transmissão via streaming

#Inicializa variável para atualizar o resultado da IA em print no terminal e json no html
temp = {
    "classe": None,
    "classe_id": None,
    "diametro": 0,
    "centro": None,
    "conf": None
}

finalResult = {
    "classe": None,
    "classe_id": None,
    "diametro": 0,
    "centro": None,
    "conf": None,
    "mensagemVerde": None
}

# Inicializa o Flask para criar servidor web que exibirá o stream
app = Flask(__name__)
running = True  # Controle para loop principal rodar/parar

# Rota para parar o streaming via requisição POST
@app.route('/stop', methods=['POST'])
def stop():
    global running  # Permite alterar variável global
    running = False
    return jsonify({"message": "Parando o streaming..."})

# Rota para desligar orange via requisição POST
@app.route('/powerOff', methods=['POST'])
def powerOff(): # Permite alterar variável global
    import subprocess
    subprocess.run(["shutdown"])
    return jsonify({"message": "Desligando dispositivo..."})

# Rota principal que retorna a página HTML com o stream
@app.route('/')
def index():
    return render_template('index.html')

# Rota que gera o stream MJPEG dos frames processados
@app.route('/stream')
def stream():
    def generate():
        global actual_frame
        latest_frame = None
        while True:
            if actual_frame is not None:
                if id(actual_frame) != id(latest_frame):
                    latest_frame = actual_frame.copy()

                    # Codifica frame em JPEG para enviar como stream

                    ret, buffer = cv2.imencode('.jpg', actual_frame)
                    frame_bytes = buffer.tobytes()
                    # Envia o frame no formato multipart/x-mixed-replace
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                # print("Frame repetido")
            time.sleep(0.06)  # Delay para controlar taxa de atualização do stream
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_frame')
def latest_frame_route():
    global actual_frame
    if actual_frame is not None:
        # Codifica o frame atual em JPEG
        ret, buffer = cv2.imencode('.jpg', actual_frame)
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    # Se não houver frame disponível, retorna um 204 (No Content)
    return ('', 204)

@app.route('/temp_data')
def get_temp_data():
    global finalResult
    # Envia o dicionário 'temp' atual como JSON
    return jsonify(finalResult)

# Indicado para apresentar um erro em uma situação específica, para mostrar um erro que pode ocorrer dependendo do uso da IA.
class erro(Exception):
    pass


# # Conecta com a porta serial para comunicacao
# def conectar_serial(porta_serial,baud_rate=115200):
#     global ser
#     try:
#         # Inicializa a comunicação serial
#         ser = serial.Serial(porta_serial, baud_rate, timeout=1)
#         print(f"Comunicação estabelecida com sucesso na porta {porta_serial}.")
#         return True

#     except serial.SerialException as e:
#         print(f"Erro ao tentar se comunicar com a porta {porta_serial}: {e}")
#         return False
    
# # Aguarda mensagem do brick
# def aguardarMensagem(tempo):
#     print("Aguardando Mensagem")

#     milliseconds_inicial= int(time.time()*1000)
#     while True: 
#         # print("Lendo mensagem")
#         mensagem = ser.read_all()
#         time.sleep(0.001)
#         # print(mensagem)
#         if mensagem is not (b''): return mensagem
#         # return '0'
#         if tempo and (int(time.time()*1000)-milliseconds_inicial)>2500: 
#             print("Muito tempo sem resposta")
#             return '0'
#         # print(int(time.time()*1000)-milliseconds_inicial)

# # Envia uma mensagem para o brick
# def enviarMensagem(mensagem):
#     ser.write(mensagem.encode())
#     print(f"Mensagem enviada: {mensagem}")

# define quantos graus tem que girar para chegar no centro do objeto
def quantidadeDeGraus(x):
    meioDaImagem=160

    # ele pega a posicao do entro do pixel, subtrai com o centro da imagem e divide por 9 (9 pixeis equivalem a 1 grau)
    graus=int((x-meioDaImagem)/4.44)
    return graus

# calcula a distancia do robo ate a vitima
def distanciaVitima(x):
    # calculo feito para vitimas de 5cm usando funcao exponecial
    distancia=(int(10*(400/x)))
    print("DISTANCIA",distancia)

    return distancia


def aplicar_mascara_hsv(img,cor):
    # primeiro valor - H (0-180) - matix
    # segundo valor - S (0-255) - saturacao
    # terceiro valor - V (0-255) - brilho

    # trasforma a imagem para HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if cor == VERMELHO:
        
        # limite inferior do VERMELHO
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])    
        # limite superior do VERMELHO
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])
        
        # criando mascara baixa e alta
        lower_mask = cv2.inRange(hsv, lower1, upper1)
        upper_mask = cv2.inRange(hsv, lower2, upper2)
        
        # juncao das mascara
        # tudo que é branco é vermelho e o restante (preto) NAO vermelho
        mask = lower_mask + upper_mask


    else: #verde
        lower_green = np.array([45, 35, 30])
        upper_green = np.array([75, 255, 255])

        # criando mascara
        mask = cv2.inRange(hsv, lower_green, upper_green)

    return mask

def identificar_triangulo_horizontal(cor):
    global cap, frame_count, actual_frame

    print("IDENTIFICAR TRIANGULO",cor)
    # quando pegavamos uma unica imagem, por vezes ele nao atualizava (ficava com a imagem anterior)
    # entao fizemos um range de 20 para garantir que houve alteração de imagem
    # i = 0
    # while i < 5:
    #     cap.grab()
    #     i+=1
    # ret, frame = cap.retrieve()
    ret, frame = cap.read()
    if not ret: print("Nao conseugi capturar imagem") 

    frame = cv2.resize(frame, (320, 240))

    # Pega as informações base da imagem original
    img_name = f"F{frame_count:04d}.jpg"
    cv2.imwrite(os.path.join(output_dir, img_name), frame)
    # latest_frame = frame
    largura = frame.shape[1]
    altura = frame.shape[0]
    centroImagem=(altura/2)
    tamanhoDoCorte=1
    frame_count += 1
    # time.sleep(0.5)

    # Aplica a mascara preta e branca na imagem
    img_name = f"M{frame_count:04d}.jpg"
    mask=aplicar_mascara_hsv(frame,cor)
    cv2.imwrite(os.path.join(output_dir, img_name), mask)
    actual_frame = mask
    frame_count += 1
    # time.sleep(0.5)


    # corta a imagem, deixando apenas o meio
    img_name = f"C{frame_count:04d}.jpg"
    mask_corte=mask[int(centroImagem-tamanhoDoCorte):int(centroImagem+tamanhoDoCorte),00:int(largura)]
    cv2.imwrite(os.path.join(output_dir, img_name), mask)
    # latest_frame = mask_corte
    frame_count += 1
    # time.sleep(0.5)


    contadorPixel=0
    pixelInicial= None
    pixelFinal=None
    quantidadeMinimaPixel=20


    # percorre por todo eixo x da imagem
    for x in mask_corte[0]:

        if pixelInicial is not None: # caso nao tenhamos definido o pixel inicial
            if contadorPixel-pixelInicial<quantidadeMinimaPixel: # enquanto nao tiver a quantidade minima de pixel sequencial, verifica a cor
                if x!=255: # caso nao seja vermelho, defina como none
                    pixelInicial= None

            else:
                if pixelFinal is not None: # caso nao tenhamos definido o pixel final
                    if contadorPixel-pixelFinal<quantidadeMinimaPixel: # enquanto nao tiver a quantidade minima de pixel sequencial, verifica a cor
                        if x==255: # caso seja vermelho, defina como none (ainda nao chegou no final)
                            pixelFinal= None
                    else:
                        break

                elif x!=255: # caso nao seja vermelho, defina como pixel final
                    pixelFinal=contadorPixel
        
        else:
            if x==255: # caso seja vermelho, defina como pixel inical
                pixelInicial=contadorPixel
            
        contadorPixel+=1



    if pixelInicial is not None: 
        # a imagem pode ter pego metade do triangulo, se isso acontecer e se eu tiver o pixel inicial definido, defino o pixel final como o ultimo pixel da imagem
        pixelFinal=contadorPixel 
        # calcula o pixel do meio do triangulo
        pixelCentral=int((pixelFinal+pixelInicial)/2)

    else:
        pixelCentral=None

    print("Pixel Incial",pixelInicial)
    print("Pixel Final",pixelFinal)
    print("Pixel Central",pixelCentral)
    return(pixelCentral)

# Função para iniciar servidor Flask em thread separada
def start_streaming_server():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run(host='0.0.0.0', port=8080, threaded=True)


# Código utilizado para realizar o processamento da imagem, reconhecimento de vítima e a criação da "moldura" em volta da vítima.
def processar_frame(stream, model, sistema):
    global actual_frame, resultados, temp, output_dir, frame_count,finalResult
    
    # Inicia um cronometro de tempo.
    start_time = time.time()

    # Se o sistema for o windows, limita o processamento a 4fps, se for linux, será o orange que está processando a imagem, e ele já é limitado.
    if sistema == "Windows": time.sleep(0.333)

    # temp = {"classe": None, "diametro": 0, "centro": None}
    temp["centro"] = None
    temp["diametro"] = 0
    temp["classe"] = None

    # Realiza a leitura da imagem capturada pela câmera, e confere via ret, se a câmera foi lida corretamente, se não, indica falha na captura.
    
    # ret, frame = cap.read()
    # frameOK = None
    # while True:
    #     frameOk = frame
    #     ret, frame = cap.read()
    #     if not ret:
    #         break  # Falha na captura
    # # print(f"Resolução real: {frame.shape[1]}x{frame.shape[0]}")
    
    # i = 0
    # while i < 5:
    #     cap.grab()
    #     i+=1
    # ret, frame = cap.retrieve()
    ret, frame = cap.read()
    if not ret: print("Nao conseugi capturar imagem")

    frame = cv2.resize(frame, (320, 240))

    # Indica os resultados que viram do processamento de imagem.
    results = model(frame, imgsz=320, conf=0.7, device=device)
    annotated_frame = results[0].plot()

    # Indica e mostra o fps na imagem da câmera, mostrando como a câmera mantém sua atualização.
    fps = 1 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    dx = 0
    # Utiliza um laço for, calcula/r o centro da imagem em que a vítima será encontrada.
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dx = int(abs(x1 - x2))
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Salva informações da vítima mais próxima do centro e da câmera.
            if dx > temp["diametro"] and abs(cy - 120) <= 20:
                temp["classe"] = model.names[cls]
                temp["classe_id"] = cls
                temp["diametro"] = dx
                temp["centro"] = [cx, cy]
                temp["conf"] = round(conf, 2)

    # Realiza o frame count, indicando a próxima imagem a ser salva, com seu nome em sequência.
    img_name = f"B{frame_count:04d}.jpg"
    
    # Se a imagem for encontrada, mostra a circunferência em volta da vítima, com base no centro e no raio.
    if temp["centro"]:
        print("\nTemp do frame:", temp)
        finalResult = temp.copy()
        centro = tuple(temp["centro"])
        raio = temp["diametro"] // 2
        marked_frame = cv2.circle(annotated_frame, centro, raio, (0, 255, 0), 2)
        resultados[img_name] = (temp)
    # Se não houver centro, não houve vitima.
    else:
        print("Nada encontrado")
        finalResult = {
        "classe": None,
        "classe_id": None,
        "diametro": 0,
        "centro": None,
        "conf": None
        }

        marked_frame = annotated_frame
        resultados[img_name] = "sem_deteccao"

    # Se o sistema for windows, mostra a imagem na webcam.
    if sistema == "Windows":
        cv2.imshow("YOLOv8 - Webcam 320x320", marked_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise erro

    # Define a imagem redimensionada. 
    # resized_frame = cv2.resize(marked_frame, (160, 120))
    cv2.imwrite(os.path.join(output_dir, img_name), marked_frame)
    # Define a última como a última imagem não redimensionada.
    actual_frame = marked_frame.copy()

    # Printa que a imagem foi salva, e mostra o nome da imagem já formatado na forma correta.
    print(f"Frame salvo: {img_name}\n")

    # Adiciona mais um ao frame counta para continuar printando e adicionando imagens com nomes sequenciais.
    frame_count += 1

def expand_box(box, scale=1.6):
    center = np.mean(box, axis=0)
    expanded = (box - center) * scale + center
    return np.intp(expanded)

def order_box_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect.astype(int)

def draw_quadrant_lines_and_labels(frame, box):
    expanded_box = expand_box(box, scale=1.6)
    rect = order_box_points(expanded_box)

    top_mid = ((rect[0] + rect[1]) // 2)
    bottom_mid = ((rect[2] + rect[3]) // 2)
    left_mid = ((rect[0] + rect[3]) // 2)
    right_mid = ((rect[1] + rect[2]) // 2)

    cv2.line(frame, tuple(left_mid), tuple(right_mid), (255, 255, 255), 1)
    cv2.line(frame, tuple(top_mid), tuple(bottom_mid), (255, 255, 255), 1)
    cv2.polylines(frame, [expanded_box], isClosed=True, color=(255, 255, 255), thickness=1)

    offset = 10
    labels = {
        "(-x, +y)": rect[0] - [offset, offset],
        "(+x, +y)": rect[1] + [offset, -offset],
        "(+x, -y)": rect[2] + [offset, offset],
        "(-x, -y)": rect[3] - [offset, -offset],
    }

    for text, pos in labels.items():
        x, y = pos.astype(int)
        # cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    return rect

def detectarVerde():
    global finalResult, width, height

    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Falha ao capturar frame")
        return None, "Erro captura", None

    try:
        frame = cv2.resize(frame, (width, height))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    except Exception as e:
        print("[ERRO] Preprocessamento falhou:", e)
        return None, "Erro preprocessamento", None

    # --- Máscara do verde ---
    lower_green = np.array([45, 55, 53])  # Ajustado
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # --- Contornos do verde ---
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_threshold = 90

    chosens_quadrants = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        ordered_box = order_box_points(box)
        expanded_box = expand_box(ordered_box, scale=1.8)

        mask_expanded = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_expanded, [expanded_box], 255)

        mask_inner = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_inner, [ordered_box], 255)

        # Máscara de anel
        mask_ring = cv2.bitwise_and(mask_expanded, cv2.bitwise_not(mask_inner))
        if mask_ring is None or mask_ring.shape != frame.shape[:2]:
            print("[ERRO] Máscara de anel inválida")
            continue

        # --- Vetorização: separar quadrantes ---
        Y, X = np.ogrid[:frame.shape[0], :frame.shape[1]]
        cx = int(np.mean([p[0] for p in expanded_box]))
        cy = int(np.mean([p[1] for p in expanded_box]))


        mask_q1 = (X < cx) & (Y < cy) & (mask_ring > 0)
        mask_q2 = (X >= cx) & (Y < cy) & (mask_ring > 0)
        mask_q3 = (X >= cx) & (Y >= cy) & (mask_ring > 0)
        mask_q4 = (X < cx) & (Y >= cy) & (mask_ring > 0)

        quadrants = {
            "(-x, +y)": mask_q1,
            "(+x, +y)": mask_q2,
            "(+x, -y)": mask_q3,
            "(-x, -y)": mask_q4,
        }

        # --- Vetorização: calcular % de pixels escuros ---
        dark_mask = gray < dark_threshold
        dark_counts = {
            q: np.count_nonzero(dark_mask[m]) / max(1, np.count_nonzero(m))
            for q, m in quadrants.items()
        }

        chosen_quadrant = max(dark_counts, key=dark_counts.get)
        if not chosen_quadrant.endswith("-y)") and cy > height // 4:
            chosens_quadrants.append([chosen_quadrant,cx,cy])

        # Desenhar contornos e linhas
        cv2.drawContours(frame, [ordered_box], 0, (0, 0, 255), 2)
        draw_quadrant_lines_and_labels(frame, box)


    # --- Mensagem final ---
    if len(chosens_quadrants) == 0:
        msg = "Nenhum quadrante identificado"
        quantidadeVerde = 0
    else:
        # Encontra a maior coordenada y entre os quadrantes escolhidos, ou seja, a mais baixa na tela
        maior_y = max(chosens_quadrants, key=lambda c: c[2])
        print(chosens_quadrants,maior_y)
        # Filtra a lista, removendo pontos próximos demais do maior_y
        chosens_greens = []

        for ponto in chosens_quadrants:
            # Calcula a distância euclidiana até maior_y
            distancia = maior_y[2] - ponto[2]
            print(distancia)
            distanciaMaxima = height//4  # Distância máxima para considerar o ponto
            if distancia <= distanciaMaxima:
                chosens_greens.append(ponto)

        if len(chosens_greens) == 1:
            if chosens_greens[0][0].startswith("(+x"):
                msg = "Verde Esquerdo"
                direcao = -1
            else:
                msg = "Verde Direito"
                direcao = 1
            quantidadeVerde = 1 * direcao
        else:
            msg = "Dois verdes"
            quantidadeVerde = 2

    finalResult["classe"] = "quadrante"
    finalResult["mensagemVerde"] = msg

    # # --- Resultado chapado (verde, preto e branco) ---
    green_layer = np.full_like(frame, (0, 255, 0))
    black_layer = np.full_like(frame, (0, 0, 0))
    white_layer = np.full_like(frame, (255, 255, 255))

    black_mask = (gray < dark_threshold).astype(np.uint8) * 255
    result = np.where(black_mask[..., None] == 255, black_layer, white_layer)
    result = np.where(mask_green[..., None] == 255, green_layer, result)

    return frame, msg, quantidadeVerde
    return frame, result, msg, quantidadeVerde

  


# frame = None
# while True:

    # if not ret or frame is None:
    #     print("Frame inválido, tentando novamente...")
    #     time.sleep(0.05)
    #     continue
    # break  # Sai do loop quando o frame for válido
# print(frame)
# ret, frame = cap.read()
# frame = cv2.resize(frame, (320, 240))
# Indica os resultados que viram do processamento de imagem.
# results = model(frame, imgsz=320, conf=0.7, device=device)

print("Conectando serial")
# Se estiver em main, incia o stream.
# serial_com = comPython.ComunicacaoSerialJSON(porta='/dev/ttyS5')

brick = comPython.ComunicacaoSerialJSON(porta='/dev/ttyS5')
# if __name__ == '__main__' and serial_com.conectar_serial(porta_serial = '/dev/ttyS5'):
# if __name__ == '__main__':


# Inicia servidor Flask em thread daemon (roda em background)
streaming_thread = Thread(target=start_streaming_server)
streaming_thread.daemon = True
streaming_thread.start()
time.sleep(2)

# Define running como true, enquanto não pedir para finalizar o código, running continua true.
running = True

# Tenta processar o frame enquanto running estiver true.
try:

# video_name = encontrar_video_por_porta_usb("510")  # ex: 'video#'
# # video_index1 = int(video_name1.replace('video', ''))  # vira #
# video_index = (f"/dev/{video_name}") # ex: '/dev/video#'
# cap = cv2.VideoCapture(video_index ) # Isso sim funciona
    while running:
        mensagemFinal = None
        # time.sleep(1)
        
        # else: mensagemRecebida = aguardarMensagem(True)
        mensagemRecebida = brick["comando"]

        # mensagemRecebida = PROCURAR_VITIMA

        print("Mensagem recebida: ",mensagemRecebida)                
        if mensagemRecebida == PROCURAR_VITIMA: 
            processar_frame(cap, model, sistema) 
            if   finalResult["classe_id"] is None: mensagemFinal = 'N .'
            else:
                graus = quantidadeDeGraus(finalResult["centro"][0])
                distancia = distanciaVitima(finalResult["diametro"]/2)
                classeID = finalResult["classe_id"]
                mensagemFinal = str("%s %s %s ."%(graus,distancia, classeID))

        # verifica se esta olhando para um triangulo vermelho, se sim, avanca ate tal
        elif mensagemRecebida==IDENTIFICAR_TRIANGULO_VERMELHO_HORIZONTAL:
            centro=identificar_triangulo_horizontal(VERMELHO)
            if centro is None: 
                print("SEM CENTRO")
                mensagemFinal='N .'
            else: mensagemFinal=str("%s ."%(quantidadeDeGraus(centro)))

        #  verifica se esta olhando para um triangulo vermelho, se sim, avanca ate tal
        elif mensagemRecebida==IDENTIFICAR_TRIANGULO_VERDE_HORIZONTAL:
            centro=identificar_triangulo_horizontal(VERDE)
            if centro is None: 
                print("SEM CENTRO")
                mensagemFinal='N .'
            else: mensagemFinal=str("%s ."%(quantidadeDeGraus(centro)))

        else: 
            print("Ordem não compatível com as existentes - Reiniciando mensagem")
            continue

        # print("MENSAGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEM",mensagemFinal)
        if mensagemFinal: enviarMensagem(mensagemFinal)
        # time.sleep(0.5)
        print("\n\n")

# Se não conseguir, define que houve um erro durante a execução.
except Exception as e:
    print(f"[ERRO durante a execução]: {e}")

# Após realizar tudo, printa que está finalizando e "libera" a câmera.
finally:
    print("Finalizando...")

    # Libera câmera
    cap.release()

    # Fecha janela do OpenCV, se estiver no Windows
    if sistema == "Windows":
        cv2.destroyAllWindows()

    # Salva arquivo JSON com todos os resultados das detecções para análise futura
    try:
        with open(os.path.join(output_dir, "resultados.json"), "w") as f:
            json.dump(resultados, f, indent=4)
        print(f"\n✅ Todos os frames e resultados foram salvos em: {output_dir}")
    except Exception as e:
        print(f"[ERRO ao salvar JSON]: {e}")
else: print("NAO CONECTOU NA SERIAL - REINICIANDO CODIGO")