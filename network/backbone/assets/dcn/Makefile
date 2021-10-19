all:
    # install pycocotools locally
    	rm  *.so 
	python setup.py build_ext --inplace
	rm -rf build

install:
	# install pycocotools to the Python site-packages
	rm  *.so 
	python setup.py build_ext build develop
	rm -rf build
