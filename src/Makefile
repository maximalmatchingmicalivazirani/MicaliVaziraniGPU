# Compiler and flags
NVCC = nvcc
CXXFLAGS = -O3 -lineinfo
NVCCFLAGS = -O3 -Xptxas -O3 -Xcompiler -O3 --extended-lambda --extra-device-vectorization --use_fast_math

# Source files
SRCS = matching.cu matchgpu.cu mmiof.cu 

# Object files
OBJS = $(SRCS:.cu=.o)

# Executable name
TARGET = matching

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
