Ajouter sur l'IDE le plugin ignore qui prendra en charge le fichier .gitignore

Les fichiers cachés OSX/WINDOWS sont ingorés, ainsi que celui de l'IDE CLion.

itou pour les fichiers c++

Mon CMake :

```
cmake_minimum_required(VERSION 3.10)
project(cpp2PA)

set(CMAKE_CXX_STANDARD 11)

add_library(cpp2PA SHARED
        library.cpp
        library.h
        LinearPerceptron/linearPerceptron.cpp
        LinearPerceptron/linearPerceptron.h
        MultiLayerPerceptron/MultiLayerPerceptron.cpp
        MultiLayerPerceptron/MultiLayerPerceptron.h
        )
```

**Linéaire** :

* Classification : OK

* Rég Linéaire : OK

* Changement de dimension : OK


**MLP** :

* Classification : OK

* Rég : OK

**RBF Naif** :

* **NOK**

**RBF** : 

* **NOK**







