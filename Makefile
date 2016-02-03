NVCC=nvcc
CC=gcc

VPATH=cu
LIBNAME=lcuda
LIBTARGET=lib$(LIBNAME).so

TESTTARGET=testmorph

INCS=-I/usr/local/cuda/include -I$(PWD)

CFLAGS+=$(INCS) -fPIC
CXXFLAGS+=$(INCS)

C_OBJS=lcudafloat.o lcudamorph.o lcudatypes.o
CU_OBJS=lcudamorphfloat.o morphology.o

TESTLIBS=`pkg-config --libs opencv` -L. -L/usr/local/cuda/lib \
				 -l$(LIBNAME) -lnpps

all: $(LIBTARGET) $(TESTTARGET)

%.o: %.cu
	$(NVCC) -c $(INCS) -Xcompiler -fPIC --use_fast_math -arch sm_52 -o $@ $<

$(LIBTARGET): $(C_OBJS) $(CU_OBJS)
	$(NVCC) -shared -o $(LIBTARGET) $(C_OBJS) $(CU_OBJS) -L/usr/local/cuda/lib -lnppi

$(TESTTARGET): testmorph.o
	$(CXX) -o $@  testmorph.o $(TESTLIBS)

clean:
	@rm -f *.o
	@rm -f $(LIBTARGET)
	@rm -f *.bmp
	@rm -f $(TESTTARGET)
