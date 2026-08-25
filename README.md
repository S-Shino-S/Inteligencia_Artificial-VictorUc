# Curso de Inteligencia Artificial

Presentaciones y laboratorios de un curso de pregrado (AIMA / Russell & Norvig),
desde agentes y búsqueda hasta redes, refuerzo, visión, modelos de lenguaje,
RAG y GANs.

Cada unidad suele tener:

| Carpeta | Contenido |
|---|---|
| `PPTXs/` | Diapositivas 16:9 (español) |
| `project/` | Programas numerados (`01_*.py`, …), YAML editable, algoritmo a mano (sin NumPy / sklearn / Gym) |
| `ejercicios/` o `Ejercicios/` | Tareas para el estudiante, cuando hay |
| `PDFs/`, `MDs/` | Lecturas o notas, cuando hay |

## Laboratorios

En cada `project/`:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python 01_*.py
```

Los detalles (qué imprime cada programa, qué editar en el YAML) están en el
`README.md` de esa unidad. La dependencia habitual es solo **PyYAML**.

## Unidades

| Unidad | Presentación | Laboratorio | Ejercicios |
|---|---|---|---|
| [Conceptos básicos de IA](Conceptos%20Básicos%20de%20IA) | `estado-del-arte.pptx` | — | [ejercicio 01](Conceptos%20Básicos%20de%20IA/ejercicios/ejercicio-01.md) |
| [Agentes](Agentes) | `wumpus-world.pptx` | Wumpus (AIMA cap. 2) | [ejercicio 01](Agentes/ejercicios/ejercicio-01.md) |
| [Búsqueda no informada](Búsqueda%20no%20informada) | — | Rumania, BFS / UCS / DFS / DLS / IDS | [ejercicio 01](Búsqueda%20no%20informada/Ejercicios/ejercicio-01.md) |
| [Búsqueda informada](Búsqueda%20informada) | — | Rumania, voraz y A* | — |
| [Razonamiento lógico](Razonamiento%20lógico) | `razonamiento-logico.pptx` | Encadenamiento hacia adelante / atrás | — |
| [Razonamiento probabilístico](Razonamiento%20probabilístico) | `redes-bayesianas.pptx` | Redes bayesianas | — |
| [Árboles de decisión](Árboles%20de%20decisión) | `arboles-de-decision.pptx` | ID3 | — |
| [Clustering K-medias](Clustering%20K-medias) | `clustering-k-medias.pptx` | Lloyd | — |
| [Perceptrón multicapa](Perceptrón%20multicapa) | `perceptron-multicapa.pptx` | MLP, XOR | — |
| [Cómputo evolutivo](Cómputo%20evolutivo) | `computo-evolutivo.pptx` | Algoritmo genético | — |
| [Q-learning](Q-learning) | `q-learning.pptx` | Q-learning tabular (pasillo A–B–G) | — |
| [Visión computacional](Visión%20computacional) | `vision-computacional.pptx` | Píxeles, convolución, Sobel | — |
| [GANs](GANs) | `gans.pptx` | Generador / discriminador 1D | — |
| [LLMs](LLMs) | `llms.pptx` | Siguiente token, softmax, atención | — |
| [RAG](RAG) | `RAG.pptx` | Embeddings, k-NN, citas | — |
| [AI Engineering](AI%20Engineering) | `the-ai-engineering-skills-map.pptx` | — | — |

Las presentaciones están en `Unidad/PPTXs/`. Los laboratorios, en `Unidad/project/`.

## Código

No hace falta GPU: todo cabe en una tabla o
en una matriz pequeña que se puede calcular a mano.

## RAG del repositorio usando DeepWiki

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/victoruccetina/inteligencia-artificial)
