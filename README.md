# VISÃO COMPUTACIONAL ORANGEPIZERO2W+EV3
# COLOCAR PRINTS DE EXEMPLO EM TUDO

Tutorial de como instalar armbian e preparar o Orange Pi Zero 2w para projeto de robótica competitiva (Robocup Rescue Line - Equipe Ligeirinho) com Visão Computacional.

Versão do armbian utilizada é baseada em debian 12 (https://www.armbian.com/orange-pi-zero-2w/). Atualmente o projeto utiliza a versão Armbian 24.2.6 Bookworm.

Agora devemos instalar no cartão SD, dar boot, configurar inicialmente a instação.

Devemos executar o script setup_yolo.sh para que as depencências sejam instaladas e o venv (ambiente virtual python) seja criado automaticamente.
Caso queira fazer as instalações por conta própria, as dependências estão lsitadas no arquivo requirements.txt.

- Para utilizar comunicação Serial, você deverá alterar algumas configuraçõoes das portas do Orange.
  - rodar o comando armbian-config (sudo armbian-config), entrar em System > dtc.
  - Nesse arquivo encontre a uart ph que será utilizada, no nosso caso a uart5, copie o valor do phandle dessa porta, mais à frente o utilizaremos.
  - Agora encontre a serial correspondente à essa uart, a ordem das seriais é 0,1,2,3..
  - Nela, faça as seguintes alterações:
    - troque status = 'disabled') para status = 'okay';
    - adicione abaixo dessa linha, as seguintes linhas:
      - pinctrl-names = 'default';
      - pinctrl-0 = <0x55> (no lugar de 0x55, coloque o valor do phandle da uart que irá utilizar);

- Agora, se quiser subir código pelo VSCode somente apertando F5, devemos permitir que nosso computador se conecte sem senha no orange. (opcional).

  - Gera uma chave publica no seu computador, se já fez isso antes, nao é necessário refazer (ssh-keygen -t rsa -b 4096).
  - ssh-copy-id USUARIO@IP (substituia o usuario e o IP pelo correto) (no linux).
  - No windows não tem como ser feito com comando, abra o arquivo id_rsa.pub, na pasta .ssh dentro da pasta seu usuario, copie a linha.
  - Logue na placa por ssh e rode os comandos.
    - mkdir -p ~/.ssh.
    - nano ~/.ssh/authorized_keys.
    - Agora, copie o valor do arquivo id_rsa.pub e cole dentro desse arquivo, salve e saia.
    - Agora, qualquer login por ssh não pedirá mais senha.

- De permissão para o shutdown e outros comandos necessários sejam executados sem precisar de senha. (opcional).

  - Digite sudo visudo
  - Dentro do arquivo, no final dele, coloque as seguintes linhas:
    - orangepi ALL=(ALL) NOPASSWD: /usr/bin/psd-overlay-helper.
    - orangepi ALL=(ALL) NOPASSWD: /sbin/shutdown.
    - orangepi ALL=(ALL) NOPASSWD: /sbin/ifconfig (troque orangepi pelo usuario escolhido).
  - Reinicie a placa e veja se não será mais pedido senha ao utilizar comandos.
 
- Para o código iniciar no momento que ligar, e quando parar de executar, ser excutado novamente para permanecer em funcionamento, iremos criar um serviço. (opcional)
  - Dentro da placa, crie um arquivo start.sh para ser executável, com o conteúdo:
    - O código do que colocar no start.sh está no arquivo start.sh na pasta do projeto, lembre-se de subsituir o que for necessário.
  - Crie um serviço para iniciar o script toda vez que a placa ligar (sudo nano /etc/systemd/system/<nome que quiser>.service).
    - O código do que colocar no serviço está no arquivo exemplo.service na pasta do projeto, lembre-se de substituir o que for necessário.
  - Rode agora sudo systemctl daemon-reexec.
  - Depois, sudo systemctl enable menuPrincipal.service.

- Inserir no script de login do usuario a opção já iniciar o terminal com o ambiente python do usuario

  - abra o arquivo .nashrc (nano ~/.bashrc)
  - no final do arquivo adicione (source ~/yolov8-env/bin/activate) (altere o caminho do venv se necessário)
  - Agora sempre que logar com o usuario no ssh, o ambiente python já será selecionado. (Um comando a menos para esquecer :D )


