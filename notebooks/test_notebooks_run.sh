#!/bin/bash


source ../venv/bin/activate
pytest --nbmake --nbmake-timeout=3000 -n=auto *.ipynb
