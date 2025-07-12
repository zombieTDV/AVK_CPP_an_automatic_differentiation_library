@echo off
setlocal enabledelayedexpansion
title CMake Build System

:menu
cls
echo ===========================================
echo         CMake Build System v1.0
echo ===========================================
echo 1. Configure Project
echo 2. Build Project
echo 3. Clean Build
echo 4. Run Program
echo 5. Show Build Status
echo 6. Exit
echo ===========================================
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto configure
if "%choice%"=="2" goto build
if "%choice%"=="3" goto clean
if "%choice%"=="4" goto run
if "%choice%"=="5" goto status
if "%choice%"=="6" goto end

echo Invalid choice! Please try again.
timeout /t 2 >nul
goto menu

:configure
echo.
echo Configuring project...
if exist build (
    echo Removing old build directory...
    rmdir /s /q build
)
mkdir build
cd build
cmake -G "Ninja" ..
if %errorlevel% equ 0 (
    echo.
    echo Configuration successful!
) else (
    echo.
    echo Configuration failed! Please check the errors above.
)
cd ..
pause
goto menu

:build
echo.
if not exist build (
    echo Build directory not found! Please configure first.
    pause
    goto menu
)
cd build
echo Building project...
cmake --build .
if %errorlevel% equ 0 (
    echo.
    echo Build successful!
) else (
    echo.
    echo Build failed! Please check the errors above.
)
cd ..
pause
goto menu

:clean
echo.
echo Cleaning build files...
if exist build (
    cd build
    cmake --build . --target clean
    cd ..
    rmdir /s /q build
    echo Clean complete!
) else (
    echo No build directory found.
)
pause
goto menu

:run
echo.
if exist build\program.exe (
    echo Running program...
    echo ===========================================
    build\program.exe
    echo ===========================================
) else (
    echo Program not found! Please build first.
)
pause
goto menu

:status
echo.
echo Build Status:
echo ===========================================
if exist build (
    echo Build directory: Found
    if exist build\program.exe (
        echo Executable: Found
        echo Last modified: %~t0 build\program.exe
    ) else (
        echo Executable: Not found
    )
    if exist build\CMakeCache.txt (
        echo CMake cache: Found
    ) else (
        echo CMake cache: Not found
    )
) else (
    echo Build directory: Not found
    echo Status: Project not configured
)
echo ===========================================
pause
goto menu

:end
echo.
echo Thank you for using CMake Build System!
timeout /t 1 >nul
exit 