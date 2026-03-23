#!/bin/bash

# Obtener timestamp en el formato deseado
timestamp=$(date +"%Y_%b_%d_%H_%M")

# Compilar el archivo .tex
pdflatex -file-line-error -interaction=nonstopmode TC5033_Activity5_30.tex
bibtex TC5033_Activity5_30
pdflatex -file-line-error -interaction=nonstopmode TC5033_Activity5_30.tex
pdflatex -file-line-error -interaction=nonstopmode TC5033_Activity5_30.tex

# Renombrar el archivo PDF generado
# mv entregable.pdf "Actividad7_Seleccion_Tipo_Almacenamiento_Equipo4_${timestamp}.pdf"
