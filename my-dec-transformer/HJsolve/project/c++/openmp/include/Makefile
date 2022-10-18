CC=g++
PCC=python3
DIR=/home/revanth/Mtech/projct/c++/openmp/include

DIRECTORY=$(DIR)/grid
HEADER_DIR=$(DIRECTORY)/headers
BACKTRACK_DIR= $(DIR)/backtracking
LIB=$(DIR)/lib
HEADER_LIB=$(LIB)/headers
MATPLT_DIR=$(LIB)/matplotlib
NUMPY_DIR=/usr/local/lib/python3.6/dist-packages/numpy/core/include/
OPENMP_DIR=-fopenmp
READ_FILE_DIR=$(DIR)/read_file
Py_Inc=$(shell  echo `python3-config --includes`)
Py_lflags=$(shell echo `python3-config --ldflags`)
STD=-fPIC -g3 -std=c++11 -shared  -I $(DIRECTORY)  -I $(READ_FILE_DIR) -I $(HEADER_DIR) -I $(LIB) -I  $(HEADER_LIB)  -I $(BACKTRACK_DIR) -I $(DIR) $(OPENMP_DIR)# $(Py_Inc)   $(Py_lflags) -DWITHOUT_NUMPY -I $(MATPLT_DIR) -I $(NUMPY_DIR)

#$(shell $(PCC) $(DIR)/setup.py build_ext --inplace)
#$(shell cython -3 --cplus $(READ_FILE_DIR)/pyreadfile.pyx )

# executable:data_prep.o grid.o narrowband.o scheme.o contouring.o common.o alloc.o advection.o levelset.o backtrack.o read_files.o pyreadfile.o

# 	$(CC) -o test data_prep.o grid.o narrowband.o scheme.o contouring.o common.o alloc.o advection.o levelset.o backtrack.o read_files.o pyreadfile.o $(STD)

# libgrid.a:data_prep.o grid.o narrowband.o scheme.o contouring.o common.o alloc.o advection.o levelset.o backtrack.o #pyreadfile.o
# 	ar rcs $@ $^ && rm -rf *.o
libgrid.so:data_prep.o grid.o narrowband.o scheme.o contouring.o common.o alloc.o advection.o levelset.o backtrack.o
	$(CC) -shared -Wl,-soname,libgrid.so  $(STD) -o libgrid.so.1  data_prep.o grid.o narrowband.o scheme.o contouring.o common.o alloc.o advection.o levelset.o backtrack.o 
	ln -s libgrid.so.1 libgrid.so && rm *.o

data_prep.o:$(DIR)/data_prep.cpp $(DIR)/data_prep.h $(HEADER_DIR)/grid.h $(BACKTRACK_DIR)/backtrack.h 
	$(CC) -c $(DIR)/data_prep.cpp $(STD) 


grid.o:$(DIRECTORY)/grid.cpp $(HEADER_DIR)/grid.h $(HEADER_DIR)/narrowband.h $(HEADER_DIR)/scheme.h $(HEADER_LIB)/contouring.h #$(MATPLT_DIR)/matplotlibcpp.h 
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


# read_files.o:$(READ_FILE_DIR)/read_files.cpp $(READ_FILE_DIR)/read_files.h $(READ_FILE_DIR)/pyreadfile.h
# 	$(CC)  -c $(READ_FILE_DIR)/read_files.cpp  $(STD)

# pyreadfile.o:$(READ_FILE_DIR)/pyreadfile.cpp $(READ_FILE_DIR)/pyreadfile.h
# 	$(CC)  -c $(READ_FILE_DIR)/pyreadfile.cpp  $(STD)



backtrack.o:$(BACKTRACK_DIR)/backtrack.cpp $(BACKTRACK_DIR)/backtrack.h $(HEADER_LIB)/point.h # $(MATPLT_DIR)/matplotlibcpp.h
	$(CC) -c $(BACKTRACK_DIR)/backtrack.cpp $(STD)
# %.o :(HEADER_LIB)/%.h
# 	$(CC) -c $(LIB)/%.cpp $(STD)
clean:
	rm -rf *.o *.out  *.a *.so.* *.so $(READ_FILE_DIR)/pyreadfile.cpp $(READ_FILE_DIR)/pyreadfile.h test
	rm -rf $(DIR)/figures/*.png
	rm -rf $(DIR)/testlogs/phi/*.txt
	rm -rf $(DIR)/testlogs/zcs/*.txt