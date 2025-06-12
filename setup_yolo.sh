#!/bin/bash
echo "Criando ambiente virtual 'yolo-env'..."
python3 -m venv yolo-env
echo "Ambiente criado."
echo "Ativando ambiente..."
source yolo-env/bin/activate
echo "Instalando dependências do requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Instalação concluída!"
echo "Para ativar o ambiente manualmente depois, use: source yolo-env/bin/activate"
