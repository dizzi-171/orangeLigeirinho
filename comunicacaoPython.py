import serial
import threading
import json
import time

''' Essa classe implementa a comunicação serial em python usando JSON.
    Ela permite enviar e receber atributos de forma assíncrona, com suporte a timeout
    para atributos que não são recebidos em um determinado período.
    A classe utiliza threads para receber dados da porta serial, enviar dados e gerenciar timeouts.
    Ela simplifica o processo de comunicação serial, permitindo que os atributos sejam definidos e obtidos sem preocupação com o sincronismo da comunicação.
'''
class ComunicacaoSerialJSON:
    def __init__(self, porta, baudrate=115200, timeout=1, tempoAtributo=10000):
        self.serial = serial.Serial(porta, baudrate, timeout=timeout)
        self._atributos = {}
        self._temposAtributos = {}
        self._atributosEnviar = {}
        self._lock = threading.Lock()
        self._lockEnviar = threading.Lock()
        self._tempoAtributo = tempoAtributo
        self._executando = True

        self._threadReceber = threading.Thread(target=self._loopReceber, daemon=True)
        self._threadReceber.start()

        self._threadTimeout = threading.Thread(target=self._loopTimeout, daemon=True)
        self._threadTimeout.start()

        self._threadEnviar = threading.Thread(target=self._loopEnviar, daemon=True)
        self._threadEnviar.start()

    def _loopReceber(self):
        while self._executando:
            try:
                linha = self.serial.readline()
                if linha:
                    dados = json.loads(linha.decode())
                    with self._lock:
                        for chave, valor in dados.items():
                            self._atributos[chave] = valor
                            self._temposAtributos[chave] = time.time()
            except Exception:
                pass

    def _loopEnviar(self):
        while self._executando:
            with self._lockEnviar:
                if self._atributosEnviar:
                    try:
                        mensagem = json.dumps(self._atributosEnviar) + '\n'
                        self.serial.write(mensagem.encode())
                        self._atributosEnviar.clear()
                    except Exception:
                        pass
            time.sleep(0.05)

    def _loopTimeout(self):
        while self._executando:
            agora = time.time()
            with self._lock:
                for chave in list(self._atributos.keys()):
                    if agora - self._temposAtributos.get(chave, 0) > self._tempoAtributo:
                        self._atributos[chave] = None
            time.sleep(0.5)

    def definirAtributo(self, chave, valor):
        with self._lockEnviar:
            self._atributosEnviar[chave] = valor

    #assim que leio o atributo, ele é removido do dicionário de atributos, para evitar que seja lido novamente
    def obterAtributo(self, chave):
        with self._lock:
            x = self._atributos.get(chave, None)
            if x is not None:
                self._atributos[chave] = None
            return x

    #aguarda até que o atributo esteja disponível ou o timeout seja atingido
    #retorna o valor do atributo se estiver disponível, ou None se o timeout for atingido
    def aguardarAtributo(self, chave, timeout=2): #tempo em segundos
        inicio = time.time()
        while time.time() - inicio < timeout:
            valor = self.obterAtributo(chave)
            if valor is not None:
                return valor
            time.sleep(0.05)
        return None

        #retorna True se o atributo existe e não é None
    def existeAtributo(self, chave):
        return chave in self._atributos and self._atributos[chave] is not None

    def fechar(self):
        self._executando = False
        self.serial.close()

    def __getitem__(self, chave):
        # print("Obtendo")
        return self.obterAtributo(chave)

    def __setitem__(self, chave, valor):
        self.definirAtributo(chave, valor)