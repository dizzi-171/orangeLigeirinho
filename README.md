# VISÃO COMPUTACIONAL ORANGEPIZERO2W+EV3

Tutorial de como instalar armbian e configurar visão computacioanl com IA, no Orange Pi Zero 2w para projeto de robótica competitiva (Robocup Rescue Line - Equipe Ligeirinho).

Versão do armbian utilizada é baseada em debian 12 (https://www.armbian.com/orange-pi-zero-2w/). Atualmente o projeto utiliza a versão Armbian 24.2.6 Bookworm.

Agora devemos instalar no cartão SD, dar boot, configurar inicialmente a instação e depois ativar a porta serial utilizada par acomunicação com o Brick EV3 usando o comando armbian-config (sudo armbian-config).
Entre em System > dtc.
Nesse arquivo encontre a uart ph que será utilizada, no nosso caso a uart5, copie o valor do phandle dessa porta, mais à frente o utilizaremos

Agora encontre a serial correspondente à essa uart, a ordem das seriais é 0,1,2,3.. 
Nela, faça as seguintes alterações:

- troque status = 'disabled') para status = 'okay';

- adicione abaixo dessa linha, as seguintes linhas:
pinctrl-names = 'default';
pinctrl-0 = <0x55> (no lugar de 0x55, coloque o valor do phandle da uart que irá utilizar)

-Agora, se quiser subir código pelo VSCode somente apertando F5, devemos permitir que nosso computador se conecte sem senha no orange:

  - Gera uma chave publica no seu computador, se já fez isso antes, nao é necessário refazer (ssh-keygen -t rsa -b 4096)
  - ssh-copy-id USUARIO@IP (substituia o usuario e o IP pelo correto) (no linux)
  - No windows não tem como ser feito com comando, abra o arquivo id_rsa.pub, na pasta .ssh dentro da pasta seu usuario, copie a linha
  - Logue na placa por ssh e rode os comandos
    - mkdir -p ~/.ssh
    - nano ~/.ssh/authorized_keys
    - Agora, copie o valor do arquivo id_rsa.pub e cole dentro desse arquivo, salve e saia
    - Agora, qualquer login por ssh não pedirá mais senha.



