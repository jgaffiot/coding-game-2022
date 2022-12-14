SHELL := /usr/bin/bash
ROOT_DIR    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
RELEASE_DIR := $(ROOT_DIR)/build/release/
DEBUG_DIR   := $(ROOT_DIR)/build/debug/
#CONAN_DIR   := $(ROOT_DIR)/build/conan/
INSTALL_DIR := $(ROOT_DIR)/install

CMAKE := cmake $(ROOT_DIR)/source # -DCONAN_DIR=$(CONAN_DIR) -DCONAN_CMAKE_SILENT_OUTPUT=1
FLAG := -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Enable the debug of the build or the Makefile itself
# Use: "make DEBUG_BUILD=1"
ifdef DEBUG_BUILD
	CMAKE += --trace --debug-output
	MAKE += --trace -d
	VERBOSE := 1
else
	MAKE += --no-print-directory
endif

.SUFFIXES:

# By default, suppress display of executed commands.
# Use: "make VERBOSE=1"
$(VERBOSE).SILENT:

########################################################################################
# Main targets
.PHONY: all
all: install

.PHONY: force
force: generate install

.PHONY: debug
debug: install_debug

########################################################################################
# Release build
# .PHONY: install_dependencies.%
# install_dependencies.%:
# 	echo "Installing the dependencies with conan in $* mode"
# 	( conan install $(ROOT_DIR) -if $(CONAN_DIR) --build missing -s build_type=$* )
#
# $(CONAN_DIR)/conanbuildinfo.cmake: conanfile.txt
# 	echo "Installing the dependencies with conan"
# 	( conan install $(ROOT_DIR) -if $(CONAN_DIR) --build missing )

.PHONY: generate
generate: #install_dependencies.Release
	echo "Creating directory $(RELEASE_DIR) and running cmake in Release mode"
	mkdir -p $(RELEASE_DIR);
	( cd $(RELEASE_DIR)$ ;  $(CMAKE) -DCMAKE_BUILD_TYPE=Release $(FLAG) ) || exit $$?;

# $(RELEASE_DIR)/CMakeCache.txt: $(CONAN_DIR)/conanbuildinfo.cmake
# 	echo "Creating directory $(RELEASE_DIR) and running cmake in Release mode"
# 	mkdir -p $(RELEASE_DIR);
# 	( cd $(RELEASE_DIR)$ ;  $(CMAKE) -DCMAKE_BUILD_TYPE=Release $(FLAG) ) || exit $$?;

.PHONY: build
build: generate # $(RELEASE_DIR)/CMakeCache.txt
	echo "Compiling in Release mode..."
	( $(MAKE) -C $(RELEASE_DIR) all ) || exit $$?;

.PHONY: install
install: build
	echo "Installing Release..."
	( $(MAKE) -C $(RELEASE_DIR) install ) || exit $$?;

########################################################################################
# Debug build
.PHONY: generate_debug
generate_debug: #| install_dependencies.Debug
	echo "Creating directory $(DEBUG_DIR) and running cmake in Debug mode"
	mkdir -p $(DEBUG_DIR);
	( cd $(DEBUG_DIR)$ ;  $(CMAKE) -DCMAKE_BUILD_TYPE=Debug $(FLAG)) || exit $$?;

$(DEBUG_DIR)/Makefile: generate_debug

.PHONY: build_debug
build_debug: $(DEBUG_DIR)/Makefile
	echo "Compiling in Debug mode..."
	( $(MAKE) -C $(DEBUG_DIR) all ) || exit $$?;

.PHONY: install_debug
install_debug: build_debug
	echo "Installing Debug..."
	( $(MAKE) -C $(DEBUG_DIR) install ) || exit $$?;


.PHONY: clean
clean: clean_install # clean_conan
	if [ -d  $(RELEASE_DIR) ]; \
	then \
		echo "Cleaning compiled files"; \
		( $(MAKE) -C $(RELEASE_DIR) clean ); \
		echo "Deleting the Release build directory $(RELEASE_DIR)"; \
		rm -rf $(RELEASE_DIR); \
	else \
		echo "The Release build directory does not exist"; \
	fi;

.PHONY: clean_debug
clean_debug: clean_install #clean_conan
	if [ -d  $(DEBUG_DIR) ]; \
	then \
		echo "Cleaning compiled files"; \
		( $(MAKE) -C $(DEBUG_DIR) clean ); \
		echo "Deleting the Debug build directory $(DEBUG_DIR)"; \
		rm -rf $(DEBUG_DIR); \
	else \
		echo "The Debug build directory does not exist"; \
	fi;

.PHONY: clean_install
clean_install:
	echo "Cleaning installed binaries and libraries"
	rm -rf $(INSTALL_DIR)/

# .PHONY: clean_conan
# clean_conan:
# 	echo "Cleaning Conan's dependency installation files"
# 	rm -rf $(CONAN_DIR)

########################################################################################
# Help
.PHONY : help
help:
	echo "The following are some of the valid targets for this Makefile:"
	echo ".. all (the default if no target is provided)"
	echo ".. force"
	echo ".. debug"
#	echo ".. install_dependencies.Release"
	echo ".. generate"
	echo ".. build"
	echo ".. install"
#	echo ".. install_dependencies.Debug"
	echo ".. generate_debug"
	echo ".. build_debug"
	echo ".. install_debug"
	echo ".. clean"
	echo ".. clean_debug"
	echo ".. clean_install"
#	echo ".. clean_conan"
	if [ -d  $(RELEASE_DIR) ]; \
	then \
		echo; echo "Targets of the CMake generated Makefile"; \
		( $(MAKE) -C $(RELEASE_DIR) help ) || exit $$?; \
	fi;

########################################################################################
# Black magic: redirect all unknown rule to the CMake generated Makefile
#conanfile.txt:;
Makefile:;
%/Makefile:;
.PHONY: %
%:: $(RELEASE_DIR)/Makefile
	echo "Passing target '$@' to generated Makefile in $(RELEASE_DIR)"
	( $(MAKE) -C $(RELEASE_DIR) $ ) || exit $$?;
