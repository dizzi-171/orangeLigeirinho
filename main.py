from ultralytics import YOLO
import cv2
import time
import torch
import platform

# Detecta sistema operacional (Windows, Linux, etc)
sistema = platform.system()
import serial

import os
import json
import glob
from flask import Flask, Response, render_template, jsonify
from threading import Thread

PROCURAR_VITIMA = (b'0')

# Define o número de threads para PyTorch (CPU)
torch.set_num_threads(4)
device = 'cpu'
print(f"Usando dispositivo: {device}")

# Carrega o modelo YOLOv8 (arquivo ONNX)
model = YOLO("./best.onnx")

# Inicializa a captura da webcam (device 0 windows device 1 orange)
if sistema == "Windows": cap = cv2.VideoCapture(0)
else: cap = cv2.VideoCapture(1)

# Define resolução da captura para 320x240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
latest_frame = None  # Frame mais recente para transmissão via streaming

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
    "conf": None
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

# Rota principal que retorna a página HTML com o stream
@app.route('/')
def index():
    return render_template('index.html')

# Rota que gera o stream MJPEG dos frames processados
@app.route('/stream')
def stream():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                # Codifica frame em JPEG para enviar como stream
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
                # Envia o frame no formato multipart/x-mixed-replace
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)  # Delay para controlar taxa de atualização do stream
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/temp_data')
def get_temp_data():
    global finalResult
    # Envia o dicionário 'temp' atual como JSON
    return jsonify(finalResult)

# Indicado para apresentar um erro em uma situação específica, para mostrar um erro que pode ocorrer dependendo do uso da IA.
class erro(Exception):
    pass


# Conecta com a porta serial para comunicacao
def conectar_serial(porta_serial,baud_rate=115200):
    global ser
    try:
        # Inicializa a comunicação serial
        ser = serial.Serial(porta_serial, baud_rate, timeout=1)
        print(f"Comunicação estabelecida com sucesso na porta {porta_serial}.")
        return True

    except serial.SerialException as e:
        print(f"Erro ao tentar se comunicar com a porta {porta_serial}: {e}")
        return False
    
# Aguarda mensagem do brick
def aguardarMensagem(tempo):
    print("Aguardando Mensagem")

    milliseconds_inicial= int(time.time()*1000)
    while True: 
        # print("Lendo mensagem")
        mensagem = ser.read_all()
        time.sleep(0.001)
        print(mensagem)
        if mensagem is not (b''): return mensagem
        # return '0'
        if tempo and (int(time.time()*1000)-milliseconds_inicial)>2500: 
            print("Muito tempo sem resposta")
            return '0'
        # print(int(time.time()*1000)-milliseconds_inicial)

# Envia uma mensagem para o brick
def enviarMensagem(mensagem):
    ser.write(mensagem.encode())
    print(f"Mensagem enviada: {mensagem}")

# define quantos graus tem que girar para chegar no centro do objeto
def quantidadeDeGraus(x):
    meioDaImagem=320

    # ele pega a posicao do entro do pixel, subtrai com o centro da imagem e divide por 9 (9 pixeis equivalem a 1 grau)
    graus=int((x-meioDaImagem)/8.88)
    return graus

# calcula a distancia do robo ate a vitima
def distanciaVitima(x):
    # calculo feito para vitimas de 5cm usando funcao exponecial
    distancia=(int(10*(873*x/5-0.941)))
    print("DISTANCIA",distancia)

    return distancia

# Função para iniciar servidor Flask em thread separada
def start_streaming_server():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run(host='0.0.0.0', port=8080, threaded=True)


# Código utilizado para realizar o processamento da imagem, reconhecimento de vítima e a criação da "moldura" em volta da vítima.
def processar_frame(cap, model, sistema):
    global latest_frame, resultados, temp, output_dir, frame_count,finalResult
    
    # Inicia um cronometro de tempo.
    start_time = time.time()

    # Se o sistema for o windows, limita o processamento a 4fps, se for linux, será o orange que está processando a imagem, e ele já é limitado.
    if sistema == "Windows": time.sleep(0.333)

    # temp = {"classe": None, "diametro": 0, "centro": None}
    temp["centro"] = None
    temp["diametro"] = 0
    temp["classe"] = None

    # Realiza a leitura da imagem capturada pela câmera, e confere via ret, se a câmera foi lida corretamente, se não, indica falha na captura.
    ret, frame = cap.read()
    if not ret:
        return False  # Falha na captura
    # print(f"Resolução real: {frame.shape[1]}x{frame.shape[0]}")
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
    img_name = f"{frame_count:04d}.jpg"
    
    # Se a imagem for encontrada, mostra a circunferência em volta da vítima, com base no centro e no raio.
    if temp["centro"]:
        print("\nTemp do frame:", temp)
        finalResult = temp.copy()
        centro = tuple(temp["centro"])
        raio = temp["diametro"] // 2
        marked_frame = cv2.circle(annotated_frame, centro, raio, (0, 255, 0), 2)
        resultados[img_name] = temp
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
    resized_frame = cv2.resize(marked_frame, (160, 120))
    cv2.imwrite(os.path.join(output_dir, img_name), resized_frame)
    # Define a última como a última imagem não redimensionada.
    latest_frame = marked_frame.copy()

    # Printa que a imagem foi salva, e mostra o nome da imagem já formatado na forma correta.
    print(f"Frame salvo: {img_name}\n")

    # Adiciona mais um ao frame counta para continuar printando e adicionando imagens com nomes sequenciais.
    frame_count += 1

ret, frame = cap.read()
frame = cv2.resize(frame, (320, 240))
# Indica os resultados que viram do processamento de imagem.
results = model(frame, imgsz=320, conf=0.7, device=device)

print("Conectando serial")
# Se estiver em main, incia o stream.
if __name__ == '__main__' and conectar_serial(porta_serial = '/dev/ttyS5'):
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

        # while True:
        #     ser.write(1)
        #     print(ser.read_all())
        #     time.sleep(1)
        while running:
            mensagemFinal = None

            

            if sistema == "Windows": mensagem = PROCURAR_VITIMA
            # else: mensagem = aguardarMensagem(True)
            # print("Mensagem recebida: ",mensagem)

            mensagem = PROCURAR_VITIMA

            if mensagem == PROCURAR_VITIMA: 
                processar_frame(cap, model, sistema) 
                if   finalResult["classe_id"] is None: mensagemFinal = 'N'
                else:
                    graus = quantidadeDeGraus(finalResult["centro"][0])
                    distancia = distanciaVitima(finalResult["diametro"])
                    mensagemFinal = str("%s %s ."%(graus,distancia))

            else: 
                print("Ordem não compatível com as existentes - Reiniciando mensagem")
                continue
            
            # if mensagemFinal: enviarMensagem(mensagemFinal)
            if mensagemFinal: print("MENSAGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEM",mensagemFinal)
            # time.sleep(0.5)

    # Se não conseguir, define que houve um erro durante a execução.
    except Exception as e:
        print(f"[ERRO durante a execução]: {e}")

    # Após realizar tudo, printa que estpa finalizando e "libera" a câmera.
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