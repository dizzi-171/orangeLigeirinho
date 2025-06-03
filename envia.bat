@echo off
scp main.py orangepi@192.168.1.126:~
scp ./templates/index.html orangepi@192.168.1.126:~/templates
ssh orangepi@192.168.1.126 "killall python"
@REM ssh banana@192.168.2.151 "bash ~/testeBanana/executa.sh" || (
@REM     echo "Interrupção detectada! Enviando sinal para o script remoto..."
@REM     ssh banana@192.168.2.151 "bash ~/testeBanana/stop.sh"
@REM )