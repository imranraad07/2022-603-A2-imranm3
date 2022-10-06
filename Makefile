todo: cuda cuda_2
cuda: cuda_basic.cu
	nvcc --ptxas-options=-v -O3 cuda_basic.cu -o cuda -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
cuda_2: cuda_basic_2d.cu
	nvcc --ptxas-options=-v -O3 cuda_basic_2d.cu -o cuda_2 -Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
clean:
	rm cuda cuda_2
