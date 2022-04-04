Buenas, les paso bien lo que habría que hacer como integrador.
Cargar los datos con objeto de clase Data (implementada por ustedes) 
con un método que cumpla esa función al pasarle la ruta.
 Hacer un split de los datos en train/test (usar 80/20)
Tratar los nans con al menos dos de las técnicas vistas en clase.
 (pasarían a tener dos datasets para comparar en lo que sigue)

Utilizar PCA para quedarse con las 3 CP. 
 (de cada uno del punto 2, idealmente usen su implementación, pero pueden usar las librerías)

Crear una clase métrica base y una clase MSE que herede es ella. (esto viene de ejercicios anteriores)

Crear una clase modelo base y clase regresión lineal que herede de ella.  (esto viene de ejercicios anteirores)
Entrenar la regresión lineal sobre train. 
Calcular MSE sobre validation. (para todas las variantes que hayan hecho en 2) y comparar.

# Paso 1 implementar clase Data
# Paso 1 cargo los datos con objeto de clase Data
# Paso 3 Hacer split de datos en 80/20
# Paso 4.1 Tratar nans con eliminacion
# Paso 4.2 Tratar nans con media
# Paso 5.1 Aplicar PCA para quedarse con 3 dimensiones a 4.1
# Paso 5.2 Aplicar PCA para quedarse con 3 dimensiones a 4.2
# Paso 6 Usar clas metrica, crear clase MSE que hereda de esta
# Paso 7 Calcular regresion lineal sobre train de 5.1 y 5.2
# Paso 8 calcular MSE sobre validacion de 5.1 y 5.2
