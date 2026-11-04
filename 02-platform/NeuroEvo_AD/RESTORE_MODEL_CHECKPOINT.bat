@echo off
REM 还原 DMSA-Net 生产权重。
REM
REM Gitee 单文件上限 50MB，而 final_model_epoch126.pth 有 217.5MB，
REM 因此把它切成 5 块存放（part01 ... part05）。
REM 本脚本按顺序拼回原文件，并校验 SHA-256。
setlocal
cd /d "%~dp0"

set "PARTS=model_artifacts"
set "TARGET=%PARTS%\final_model_epoch126.pth"
set "EXPECT_SHA=ba96575888e35764e3ff196674e52757d9a5f9c90742a38aa8f1d1f9a8457658"
set "EXPECT_SIZE=228051470"

if exist "%TARGET%" (
  echo [NeuroEvo] %TARGET% 已存在，跳过。
  goto :end
)

echo [NeuroEvo] 合并分块 ...
copy /b "%PARTS%\final_model_epoch126.pth.part01" + "%PARTS%\final_model_epoch126.pth.part02" + "%PARTS%\final_model_epoch126.pth.part03" + "%PARTS%\final_model_epoch126.pth.part04" + "%PARTS%\final_model_epoch126.pth.part05" "%TARGET%" >nul
if errorlevel 1 (
  echo [NeuroEvo] 合并失败。 >&2
  exit /b 1
)

for %%A in ("%TARGET%") do set "GOT_SIZE=%%~zA"
if not "%GOT_SIZE%"=="%EXPECT_SIZE%" (
  echo [NeuroEvo] 大小不符: 得到 %GOT_SIZE%, 期望 %EXPECT_SIZE% >&2
  exit /b 1
)

for /f "skip=1 tokens=* delims=" %%H in ('certutil -hashfile "%TARGET%" SHA256') do (
  if not defined GOT_SHA set "GOT_SHA=%%H"
)
set "GOT_SHA=%GOT_SHA: =%"
if /i not "%GOT_SHA%"=="%EXPECT_SHA%" (
  echo [NeuroEvo] SHA-256 不符: 得到 %GOT_SHA% >&2
  echo [NeuroEvo] 期望 %EXPECT_SHA% >&2
  exit /b 1
)

echo [NeuroEvo] 完成: %TARGET% (%GOT_SIZE% 字节)
echo [NeuroEvo] SHA-256 校验通过

:end
endlocal
