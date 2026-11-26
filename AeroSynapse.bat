@echo off
rem AeroSynapse Studio: design, workflow (basic design to export), optimisation, simulation.
rem   AeroSynapse.bat                      open Studio
rem   AeroSynapse.bat design.json          open a design
rem   AeroSynapse.bat --project falcon     open Studio with the Design workflow of projects\falcon
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
set "PYTHONPATH=%~dp0src"
if exist "bin\AeroSynapseStudio.exe" goto run
echo Building the geometry kernel, the flow solver and AeroSynapse Studio - this needs g++ on PATH ...
"%PY%" -c "from aerosynapse.cad import kernel; [print(kernel.build(n)) for n in ('aerokernel', 'aeroflow', 'AeroSynapseStudio')]"
if not exist "bin\AeroSynapseStudio.exe" (
  echo Could not build AeroSynapse Studio. See the messages above.
  pause
  exit /b 1
)
:run
start "" "bin\AeroSynapseStudio.exe" %*
