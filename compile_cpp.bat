@echo off
setlocal

REM --- Set your libtorch path ---
set TORCH_PATH=D:\libtorch-win-shared-with-deps-2.9.1+cpu\libtorch

REM --- Compile and link ---
cl.exe /std:c++17 /EHsc /MD ^
  /I"%TORCH_PATH%\include" ^
  /I"%TORCH_PATH%\include\torch\csrc\api\include" ^
  inference.cpp ^
  /link ^
  /LIBPATH:"%TORCH_PATH%\lib" ^
  torch.lib torch_cpu.lib c10.lib

endlocal
