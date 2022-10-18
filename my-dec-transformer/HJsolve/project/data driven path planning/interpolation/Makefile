CC=g++


# libexample.so:example.o
# 	$(CC) -shared -Wl,-soname,libexample.so -fPIC -g3 -std=c++11 -shared -fopenmp -o libexample.so.1 example.o
# 	ln -s libexample.so.1 libexample.so 
libinter.so:interp3d.o
	$(CC) -shared -Wl,-soname,libinter.so -fPIC -O3 -g3 -std=c++11 -shared -fopenmp -o libinter.so.1 interp3d.o
	ln -s libinter.so.1 libinter.so && rm *.o


interp3d.o:interp3d.cpp interp3d.h
	$(CC) -c -O3 interp3d.cpp -fPIC -g3 -std=c++11 -shared
# example.o:example.cpp example.h
# 	$(CC) -c example.cpp -fPIC -g3 -std=c++11 -shared  -fopenmp

clean:
	rm *.so*