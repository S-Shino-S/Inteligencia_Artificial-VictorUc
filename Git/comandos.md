# Comandos de Git

Los comandos más usados en el día a día. Ejecútalos en la carpeta del repositorio.

## Configuración

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
git config --list
```

Ver `instalación.md` para instalar Git.

## Crear o copiar un repositorio

```bash
git init                          # convierte la carpeta actual en un repo
git clone https://github.com/usuario/repo.git
git clone https://github.com/usuario/repo.git nombre-carpeta
```

## Ver el estado

```bash
git status                        # archivos modificados, staged y no rastreados
git diff                          # cambios aún no agregados
git diff --staged                 # cambios ya en el área de staging
git log                           # historial de commits
git log --oneline                 # historial compacto
git log --oneline --graph --all   # ramas e historial juntos
```

## Guardar cambios (add + commit)

```bash
git add archivo.txt               # un archivo
git add carpeta/                  # una carpeta
git add .                         # todo lo modificado en el directorio actual
git commit -m "Mensaje claro"     # crea el commit con lo que está en staging
```

Flujo habitual:

```bash
git status
git add .
git commit -m "Describe el cambio"
```

## Deshacer (con cuidado)

```bash
git restore archivo.txt           # descarta cambios locales de un archivo
git restore --staged archivo.txt  # saca el archivo del staging (sigue modificado)
git restore .                     # descarta todos los cambios no confirmados
```

`git restore` no borra commits. Evita `git reset --hard` salvo que sepas que vas a perder trabajo.

## Ramas

```bash
git branch                        # lista ramas locales
git branch nombre-rama            # crea una rama
git switch nombre-rama            # cambia a esa rama
git switch -c nombre-rama         # crea la rama y cambia a ella
git merge nombre-rama             # integra esa rama en la actual
git branch -d nombre-rama         # borra una rama ya fusionada
```

## Remoto (GitHub, GitLab, etc.)

```bash
git remote -v                     # muestra los remotos
git remote add origin URL         # asocia el remoto llamado origin
git fetch                         # baja novedades sin fusionar
git pull                          # fetch + merge en la rama actual
git push                          # sube la rama actual
git push -u origin nombre-rama    # primera vez: publica la rama y la asocia
```

## Stash (guardar cambios a medias)

```bash
git stash                         # guarda cambios locales y deja el working tree limpio
git stash list                    # lista stashes
git stash pop                     # recupera el último stash
git stash drop                    # descarta el último stash
```

## `.gitignore`

Crea un archivo `.gitignore` en la raíz del repo para no versionar archivos locales, por ejemplo:

```
venv/
__pycache__/
.DS_Store
```
