CC=mpic++

DIRECTORY=./include/grid2dio
HEADER_DIR=$(DIRECTORY)/headers
LIB=./include/lib
HEADER_LIB=$(LIB)/headers
MATPLT_DIR=$(LIB)/matplotlib
STD=-std=c++11 -I/usr/include/python2.7 -lpython2.7 -I $(DIRECTORY) -I $(HEADER_DIR) -I $(LIB) -I  $(HEADER_LIB) -I $(MATPLT_DIR)

executable:test2.o grid.o narrowband.o scheme.o contouring.o common.o alloc.o advection.o levelset.o
	$(CC) test2.o grid.o narrowband.o scheme.o contouring.o common.o alloc.o advection.o levelset.o $(STD)
	rm -r *.o
test2.o:test2.cpp $(HEADER_DIR)/grid.h
	$(CC) -c test2.cpp $(STD) 
grid.o:$(DIRECTORY)/grid.cpp $(HEADER_DIR)/grid.h $(HEADER_DIR)/narrowband.h $(HEADER_DIR)/scheme.h $(HEADER_LIB)/contouring.h $(MATPLT_DIR)/matplotlibcpp.h
	$(CC) -c $(DIRECTORY)/grid.cpp  $(STD)
narrowband.o:$(DIRECTORY)/narrowband.cpp $(HEADER_DIR)/narrowband.h $(HEADER_LIB)/common.h
	$(CC) -c $(DIRECTORY)/narrowband.cpp $(STD)
scheme.o:$(DIRECTORY)/scheme.cpp $(HEADER_DIR)/scheme.h $(HEADER_LIB)/common.h $(HEADER_LIB)/levelset.h $(HEADER_LIB)/advection.h
	$(CC) -c $(DIRECTORY)/scheme.cpp $(STD)
contouring.o:$(LIB)/contouring.cpp $(HEADER_LIB)/point.h
	$(CC) -c $(LIB)/contouring.cpp $(STD)
common.o:$(LIB)/common.cpp $(HEADER_LIB)/common.h $(HEADER_LIB)/alloc.h
	$(CC) -c $(LIB)/common.cpp $(STD)
alloc.o:$(LIB)/alloc.cpp $(HEADER_LIB)/alloc.h
	$(CC) -c $(LIB)/alloc.cpp $(STD)
advection.o:$(LIB)/advection.cpp $(HEADER_LIB)/advection.h
	$(CC) -c $(LIB)/advection.cpp $(STD)
levelset.o:$(LIB)/levelset.cpp $(HEADER_LIB)/levelset.h
	$(CC) -c $(LIB)/levelset.cpp $(STD)
# %.o :(HEADER_LIB)/%.h
# 	$(CC) -c $(LIB)/%.cpp $(STD)
clean:
	rm -rf *.o *.out