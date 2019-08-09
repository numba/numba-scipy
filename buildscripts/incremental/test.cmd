
call activate %CONDA_ENV%

@rem Run system info tool
numba -s

@rem switch off color messages
set NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING=1
@rem switch on developer mode
set NUMBA_DEVELOPER_MODE=1
@rem enable the faulthandler
set PYTHONFAULTHANDLER=1

python -m pytest

if %errorlevel% neq 0 exit /b %errorlevel%
