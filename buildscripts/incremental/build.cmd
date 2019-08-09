
call activate %CONDA_ENV%

python -m pip install -e .

if %errorlevel% neq 0 exit /b %errorlevel%
