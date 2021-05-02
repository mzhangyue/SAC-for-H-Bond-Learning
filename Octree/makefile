# Directory containing the octree libraries
LIB_DIR = FromBU/minilibmol

# Default make the octree code
default: octree

# Call the setup.py
octree: setup.py octree.pyx $(LIB_DIR)/libmol.0.0.6.a
	python3 setup.py build_ext --inplace && rm -f octree.c && rm -Rf build

# Make the octree archive file
$(LIB_DIR)/libmol.0.0.6.a:
	make -C $(LIB_DIR)

# Make clean
clean:
	rm *.so