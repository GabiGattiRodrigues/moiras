@echo off
setlocal enableextensions
chcp 65001 >nul 2>&1
title Moiras
cd /d "%~dp0"
echo.
echo  ==========================================================
echo    Moiras - quatro modelos disputam cada case
echo  ==========================================================
echo.

rem --- 1. Onde esta o Python? (mesmo jeito do Dashboard Inteligente)
set "PY="
where py >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto python_ok
where python >nul 2>&1
if not errorlevel 1 set "PY=python"
:python_ok

if not defined PY (
  echo  [ X ]  Nao encontrei o Python nesta maquina.
  echo         Instale em https://www.python.org/downloads/
  echo         e marque "Add Python to PATH" antes de clicar em Install.
  pause
  exit /b 1
)
echo  [ 1/3 ]  Python encontrado.

rem --- 2. Ambiente isolado
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo  [ 2/3 ]  Criando o ambiente isolado ^(so na primeira vez^)...
  %PY% -m venv .venv
  if not exist "%VPY%" (
    echo  [ X ]  Nao consegui criar o ambiente .venv
    pause
    exit /b 1
  )
) else (
  echo  [ 2/3 ]  Ambiente isolado ja existe.
)

rem --- 3. Dependencias (so reinstala se o requirements.txt mudar)
set "MARCA=.venv\instalado.txt"
for %%F in (requirements.txt) do set "REQ=%%~tF %%~zF"
set "JAINSTALADO="
if exist "%MARCA%" set /p JAINSTALADO=<"%MARCA%"
if "%JAINSTALADO%"=="%REQ%" goto deps_prontas
echo  [ 3/3 ]  Instalando as bibliotecas. Leva alguns minutos na primeira vez.
echo.
"%VPY%" -m pip install --upgrade pip --disable-pip-version-check --retries 5 --timeout 60
"%VPY%" -m pip install -r requirements.txt --disable-pip-version-check --retries 5 --timeout 60
if errorlevel 1 (
  echo.
  echo  [ X ]  A instalacao falhou - quase sempre e conexao. Rode de novo.
  pause
  exit /b 1
)
>"%MARCA%" echo %REQ%
goto abrir

:deps_prontas
echo  [ 3/3 ]  Bibliotecas ja instaladas.

:abrir
rem --- Evita o Streamlit parar pedindo e-mail na primeira execucao
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit" >nul 2>&1
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  >"%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >>"%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)
echo.
echo  Abrindo no navegador. Se nao abrir sozinho, entre em http://localhost:8501
echo  Para fechar: Ctrl+C aqui, ou feche esta janela.
echo.
"%VPY%" -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo  [ X ]  O app encerrou com erro. A mensagem esta acima.
  pause
)
endlocal
