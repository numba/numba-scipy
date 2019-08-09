
call activate %CONDA_ENV%

python -m pip install -e .[dev]

if %errorlevel% neq 0 exit /b %errorlevel%
