cd frontend
call flutter build web --base-href /chat/ --web-renderer canvaskit

cd ..
IF EXIST "app\web" (
    rmdir /s /q "app\web"
)
IF NOT EXIST "app\web" (
    mkdir "app\web"
)
xcopy /s /y "frontend\build\web\*" "app\web\"
