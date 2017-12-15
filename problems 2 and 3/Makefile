
default:
	g++ -o app -c bfs.cpp -I"C:\Program Files (x86)\AMD APP SDK\2.9\include"
	g++ -o app.exe app -lopencl -L"C:\Program Files (x86)\AMD APP SDK\2.9\lib\x86_64"
	
blas:
	g++ -o app -c test.cpp -I"package\include" -I"C:\Program Files (x86)\AMD APP SDK\2.9\include"
	g++ -o app.exe app -lopencl -lclBLAS -L"package\bin" -L"C:\Program Files (x86)\AMD APP SDK\2.9\lib\x86_64"

