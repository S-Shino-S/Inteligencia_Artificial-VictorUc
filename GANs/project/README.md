# GAN desde cero

Un GAN de juguete que puedes leer línea por línea (**sin NumPy, sin PyTorch, sin
GPU**): un generador y un discriminador, ambos MLP escritos a mano con
retropropagación manual, jugando el juego minimax de Goodfellow (2014).

Todo ocurre en **1 dimensión**: el generador aprende a convertir ruido en números
que siguen una distribución objetivo (por defecto, una gaussiana centrada en 4).
Como es 1D, puedes verificar el resultado a ojo con los histogramas ASCII.

La receta (generador vs. discriminador, pérdida no saturante, actualizaciones
alternas) es exactamente la misma que en un GAN de imágenes; aquí solo cambian el
tamaño de los datos y de las redes.

## Setup

```bash
cd "GANs/project"
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows (PowerShell):

```powershell
cd "GANs/project"
python3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Sesiones posteriores: activa el venv otra vez y ejecuta los programas.
Desactiva con `deactivate`. (La única dependencia es PyYAML, para leer la
configuración.)

## Programas

Cada programa ilustra una pieza del funcionamiento de un GAN.

| Archivo | Qué muestra |
|---|---|
| `01_data.py` | El problema: la distribución REAL a imitar y el RUIDO de entrada |
| `02_discriminator.py` | El discriminador **solo**: aprende a separar real de falso |
| `03_train.py` | El juego completo: G y D se entrenan a la vez y convergen |
| `04_generate.py` | Genera muestras nuevas con el generador ya entrenado |
| `05_2d.py` | Un objetivo en 2D (anillo o cúmulos) con **dispersión ASCII** |

```bash
python 01_data.py
python 02_discriminator.py
python 03_train.py
python 04_generate.py
python 05_2d.py
python 03_train.py --steps 4000 --seed 1
python 04_generate.py --samples 5000
python 05_2d.py --steps 5000
```

Banderas útiles: `--config PATH`, `--steps`, `--batch`, `--seed`, `--samples`,
`--model PATH`. `03_train.py` guarda el modelo en `model.json`; `04_generate.py`
lo carga (o entrena uno al vuelo si no existe).

## Cómo funciona

El juego minimax:

```
min_G  max_D   E_x[ log D(x) ]  +  E_z[ log(1 - D(G(z))) ]
```

- **Generador `G`** (`gan/gan.py`, `gan/mlp.py`) — toma un vector de ruido
  `z` y devuelve un número `G(z)`. Nunca ve datos reales: aprende solo del
  gradiente que baja desde `D`.
- **Discriminador `D`** — un clasificador binario que estima `D(·) ≈ P(real)`.
- **Entrenamiento** (`GAN.train`) — en cada paso actualiza `D` (separar real de
  `G(z)`) y luego `G` (que `D(G(z))` parezca real, con la pérdida **no
  saturante** `max log D(G(z))`).

Detalles de implementación pensados para leerse:

- `gan/mlp.py` — capas densas con LeakyReLU y salida lineal. `backward` devuelve
  los gradientes de los pesos **y** el gradiente respecto a la entrada; ese
  segundo gradiente es el que conecta a `G` con `D`. Incluye un optimizador Adam
  (SGD puro cuesta converger con un objetivo que se mueve).
- Las redes trabajan en un espacio **estandarizado** (media ~0, escala ~1) para
  que el sigmoide y los gradientes estén bien condicionados; solo al mostrar o
  muestrear se vuelve a las unidades reales.

## Qué esperar

**`01_data.py`** — dos histogramas: los datos reales (una campana en 4.0) y el
ruido de entrada (sin forma). La tarea del GAN es transformar el segundo en el
primero.

**`02_discriminator.py`** — entrenando **solo** a `D` contra un falso fijo, su
acierto sube hacia ~95% y la curva `D(x)` pasa de ~0 (zona del falso) a ~1 (zona
real). `D` no es más que un clasificador.

**`03_train.py`** — con el juego completo, el acierto de `D` **baja hacia ~50%**
(ya no distingue) y la media/desviación de lo generado se acerca a las reales.
El histograma final muestra el generado (▓) solapándose con el real (█). Verás
algo de vaivén a mitad de camino: es la inestabilidad típica de los GAN.

**`04_generate.py`** — cada número sale de un vector de ruido distinto; su
distribución imita a la real.

**`05_2d.py`** — ahora los datos son puntos `(x, y)`. Con un objetivo en forma
de **anillo**, el gráfico "antes" es ruido amontonado y el "después" traza el
círculo completo (`@` = punto generado sobre uno real). Reporta el radio medio y
la **cobertura angular**: si baja, el generador dejó huecos en el anillo
(colapso de modos). Con `target2d.kind: blobs` mide cuántos cúmulos cubrió.

## Experimentos sugeridos

Edita `data/gan.yaml` y reentrena con `03_train.py`:

- Cambia `target.mean` / `target.std` y observa cómo el generado sigue al
  objetivo.
- Pon `target.kind: mixture` con dos componentes (p. ej. en 2.5 y 5.5): es fácil
  que `G` cubra **solo una** de las dos jorobas — eso es el **colapso de modos**.
- Sube o baja `training.d_steps` y las tasas `lr_g` / `lr_d`: si `D` es demasiado
  fuerte, `G` deja de recibir gradiente; si es muy débil, `G` se dispersa.
- Reduce `generator.hidden` a `[4]` y mira cómo pierde capacidad para ajustar la
  forma.
- En 2D (`05_2d.py`), cambia `target2d.kind` entre `ring` y `blobs`, ajusta el
  `radius` o los `centers`, y observa la cobertura: es la forma más visual de ver
  el colapso de modos.

Un GAN de imágenes es este mismo bucle a gran escala: cambia los MLP por redes
convolucionales y el número 1D por una imagen, pero el juego es idéntico.
