@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

set "STREAMLIT_EXE=streamlit"
if exist ".venv\Scripts\streamlit.exe" set "STREAMLIT_EXE=.venv\Scripts\streamlit.exe"

rem Default environment values
if "%SKYCLEAR_NUM_SAMPLES%"=="" set "SKYCLEAR_NUM_SAMPLES=24"
if "%SKYCLEAR_PATCH_SIZE%"=="" set "SKYCLEAR_PATCH_SIZE=256"
if "%SKYCLEAR_EPOCHS%"=="" set "SKYCLEAR_EPOCHS=10"
if "%SKYCLEAR_BATCH_SIZE%"=="" set "SKYCLEAR_BATCH_SIZE=2"
if "%SKYCLEAR_BASE_CHANNELS%"=="" set "SKYCLEAR_BASE_CHANNELS=32"
if "%SKYCLEAR_CHECKPOINT_EVERY_STEPS%"=="" set "SKYCLEAR_CHECKPOINT_EVERY_STEPS=100"
if "%SKYCLEAR_BASELINE_BACKEND%"=="" set "SKYCLEAR_BASELINE_BACKEND=auto"
if "%SKYCLEAR_LAUNCH_APP%"=="" set "SKYCLEAR_LAUNCH_APP=1"

rem Bounding box and date for real data acquisition
if "%SKYCLEAR_BBOX%"=="" set "SKYCLEAR_BBOX=11.5 48.1 11.6 48.2"
if "%SKYCLEAR_DATE%"=="" set "SKYCLEAR_DATE=2023-06-01/2023-08-31"

if /I "%MODE%"=="help" goto :help
if /I "%MODE%"=="setup" goto :setup
if /I "%MODE%"=="test" goto :test
if /I "%MODE%"=="download" goto :download
if /I "%MODE%"=="smoke" goto :smoke
if /I "%MODE%"=="full" goto :full
if /I "%MODE%"=="real" goto :real
if /I "%MODE%"=="app" goto :app

echo [ERROR] Unknown mode: %MODE%
echo.
goto :help

:setup
echo ===================================================
echo [SkyClearAI] Setting up environment and dependencies
echo ===================================================
"%PYTHON_EXE%" -m pip install -e ".[dev]"
if errorlevel 1 goto :fail
goto :done

:test
echo ===================================================
echo [SkyClearAI] Running automated smoke tests
echo ===================================================
"%PYTHON_EXE%" -m pytest -q --basetemp "%TEMP%\skyclear_pytest_tmp" -o cache_dir="%TEMP%\skyclear_pytest_cache"
if errorlevel 1 goto :fail
goto :done

:download
echo ===================================================
echo [SkyClearAI] Querying and acquiring real Sentinel imagery
echo ===================================================
echo Target Bbox: %SKYCLEAR_BBOX%
echo Target Date: %SKYCLEAR_DATE%
"%PYTHON_EXE%" -m src.acquire_data --bbox %SKYCLEAR_BBOX% --date %SKYCLEAR_DATE% --output-dir data/raw
if errorlevel 1 goto :fail
goto :done

:smoke
echo ===================================================
echo [SkyClearAI] Running fast synthetic CPU smoke pipeline
echo ===================================================
set "SMOKE_PROCESSED=data\processed_smoke"
set "SMOKE_INFERENCE=outputs\smoke_inference"
set "SMOKE_EVALUATION=outputs\smoke_evaluation"
echo [1/3] Generating synthetic dataset...
"%PYTHON_EXE%" -m src.data_pipeline --force-synthetic --num-samples 6 --patch-size 32 --output-dir "%SMOKE_PROCESSED%"
if errorlevel 1 goto :fail
echo [2/3] Running standalone inference...
"%PYTHON_EXE%" -m src.infer --processed-dir "%SMOKE_PROCESSED%" --split test --output-dir "%SMOKE_INFERENCE%" --allow-random-weights --base-channels 4 --limit 1 --baseline-backend simple
if errorlevel 1 goto :fail
echo [3/3] Running evaluation and reporting...
"%PYTHON_EXE%" -m src.evaluate --inference-dir "%SMOKE_INFERENCE%" --output-dir "%SMOKE_EVALUATION%"
if errorlevel 1 goto :fail
echo.
echo [SUCCESS] Smoke outputs written to: %SMOKE_EVALUATION%
goto :done

:full
echo ===================================================
echo [SkyClearAI] Running full synthetic pipeline
echo ===================================================
echo Samples: %SKYCLEAR_NUM_SAMPLES%
echo Patch size: %SKYCLEAR_PATCH_SIZE%
echo Epochs: %SKYCLEAR_EPOCHS%
echo Batch size: %SKYCLEAR_BATCH_SIZE%
echo [1/4] Processing synthetic dataset...
"%PYTHON_EXE%" -m src.data_pipeline --force-synthetic --num-samples %SKYCLEAR_NUM_SAMPLES% --patch-size %SKYCLEAR_PATCH_SIZE% --output-dir data\processed
if errorlevel 1 goto :fail
echo [2/4] Training Model 1 generator...
"%PYTHON_EXE%" -m src.train --processed-dir data\processed --epochs %SKYCLEAR_EPOCHS% --batch-size %SKYCLEAR_BATCH_SIZE% --base-channels %SKYCLEAR_BASE_CHANNELS% --checkpoint-every-steps %SKYCLEAR_CHECKPOINT_EVERY_STEPS% --checkpoint-dir checkpoints
if errorlevel 1 goto :fail
echo [3/4] Running inference...
"%PYTHON_EXE%" -m src.infer --processed-dir data\processed --split test --checkpoint checkpoints\model1_latest.pt --output-dir outputs\inference\test --base-channels %SKYCLEAR_BASE_CHANNELS% --baseline-backend %SKYCLEAR_BASELINE_BACKEND%
if errorlevel 1 goto :fail
echo [4/4] Evaluating results...
"%PYTHON_EXE%" -m src.evaluate --inference-dir outputs\inference\test --output-dir outputs\evaluation
if errorlevel 1 goto :fail
echo.
echo [SUCCESS] Full evaluation reports generated in outputs\evaluation
if "%SKYCLEAR_LAUNCH_APP%"=="1" goto :app
goto :done

:real
echo ===================================================
echo [SkyClearAI] Running pipeline using REAL Sentinel data
echo ===================================================
if not exist "data\raw\sentinel2_clear.tif" (
    echo [INFO] Real Sentinel data not found in data\raw. Acquiring data first...
    call :download
    if errorlevel 1 goto :fail
)
echo [1/4] Processing, aligning and tiling real Sentinel data...
"%PYTHON_EXE%" -m src.data_pipeline --clear-dir data\raw --patch-size %SKYCLEAR_PATCH_SIZE% --output-dir data\processed
if errorlevel 1 goto :fail
echo [2/4] Training Model 1 generator...
"%PYTHON_EXE%" -m src.train --processed-dir data\processed --epochs %SKYCLEAR_EPOCHS% --batch-size %SKYCLEAR_BATCH_SIZE% --base-channels %SKYCLEAR_BASE_CHANNELS% --checkpoint-every-steps %SKYCLEAR_CHECKPOINT_EVERY_STEPS% --checkpoint-dir checkpoints
if errorlevel 1 goto :fail
echo [3/4] Running inference...
"%PYTHON_EXE%" -m src.infer --processed-dir data\processed --split test --checkpoint checkpoints\model1_latest.pt --output-dir outputs\inference\test --base-channels %SKYCLEAR_BASE_CHANNELS% --baseline-backend %SKYCLEAR_BASELINE_BACKEND%
if errorlevel 1 goto :fail
echo [4/4] Evaluating results...
"%PYTHON_EXE%" -m src.evaluate --inference-dir outputs\inference\test --output-dir outputs\evaluation
if errorlevel 1 goto :fail
echo.
echo [SUCCESS] Real data evaluation reports generated in outputs\evaluation
if "%SKYCLEAR_LAUNCH_APP%"=="1" goto :app
goto :done

:app
echo ===================================================
echo [SkyClearAI] Launching Streamlit Dashboard
echo ===================================================
"%STREAMLIT_EXE%" run app\streamlit_app.py
if errorlevel 1 goto :fail
goto :done

:help
echo SkyClearAI Command Runner
echo.
echo Usage:
echo   run_skyclear.bat ^<mode^>
echo.
echo Modes:
echo   full       Run synthetic data pipeline, training, inference, evaluation, and launch app.
echo   real       Run pipeline using real Sentinel-1/2 data (downloads automatically if missing).
echo   download   Query STAC and crop Sentinel-1 GRD and Sentinel-2 clear/cloudy scenes.
echo   smoke      Run a fast synthetic CPU wiring check (no trained checkpoint required).
echo   test       Run automated unit and integration tests.
echo   setup      Install required Python packages and dependencies in editable mode.
echo   app        Launch the interactive Streamlit dashboard directly.
echo.
echo Configuration overrides (set as env variables):
echo   SKYCLEAR_BBOX               Target WGS84 bounding box (default: 11.5 48.1 11.6 48.2)
echo   SKYCLEAR_DATE               Target date range (default: 2023-06-01/2023-08-31)
echo   SKYCLEAR_NUM_SAMPLES        Number of synthetic samples (default: 24)
echo   SKYCLEAR_PATCH_SIZE         Tiled patch dimension in pixels (default: 256)
echo   SKYCLEAR_EPOCHS             Number of training epochs (default: 10)
echo   SKYCLEAR_BATCH_SIZE         Training batch size (default: 2)
echo   SKYCLEAR_BASE_CHANNELS      Generator/Discriminator width (default: 32)
echo   SKYCLEAR_LAUNCH_APP=0       Disables launching Streamlit after pipeline runs.
exit /b 0

:fail
echo.
echo [ERROR] SkyClearAI task failed.
exit /b 1

:done
echo.
echo [SUCCESS] SkyClearAI task completed successfully.
exit /b 0
