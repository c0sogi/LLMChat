cd ./repositories/llama_cpp/vendor/llama.cpp
rmdir /s /q build
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DLLAMA_CUBLAS=ON
cmake --build . --config Release
cd ../../../../..
copy repositories\llama_cpp\vendor\llama.cpp\build\bin\Release\llama.dll repositories\llama_cpp\llama_cpp


