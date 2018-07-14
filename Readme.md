Ajouter sur l'IDE le plugin ignore qui prendra en charge le fichier .gitignore

Les fichiers cachés OSX/WINDOWS sont ingorés, ainsi que celui de l'IDE CLion.

itou pour les fichiers c++

Mon CMake :

```
cmake_minimum_required(VERSION 3.10)
project(cpp2PA)

set(CMAKE_CXX_STANDARD 11)

add_library(cpp2PA SHARED
        LinearPerceptron/linearPerceptron.cpp
        LinearPerceptron/h/linearPerceptron.h
        MultiLayerPerceptron/MultiLayerPerceptron.cpp
        MultiLayerPerceptron/h/MultiLayerPerceptron.h
        LinearPerceptron/interfaceWithCpp.cpp
        LinearPerceptron/h/interfaceWithCpp.h
        MultiLayerPerceptron/interfaceWithCpp.cpp
        MultiLayerPerceptron/h/interfaceWithCpp.h
        #RBF/InterfaceWithCpp.cpp
        #RBF/h/InterfaceWithCpp.h
        RBF/RBF.cpp
        RBF/h/RBF.h
        )
```

**Linéaire** :

* Classification : **OK**

* Rég Linéaire : **OK**

* Changement de dimension : **OK**


**MLP** :

* Classification : **OK**

* Régression : **OK**

**RBF Naif** :

* Classification : **OK**

* Régression : **OK**

**RBF** : 

* **NOK**

**Fonction de vérifications des arguments**

* **NOK**







