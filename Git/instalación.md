# Instalación de Git

Guía rápida para instalar Git en Windows, macOS y Linux.

## Windows (PowerShell)

### 1. Abre PowerShell como administrador

Desde el menú de inicio, busca `PowerShell`, haz clic derecho y selecciona **Ejecutar como administrador**.

### 2. Verifica si `winget` está disponible

```powershell
winget --version
```

### 3. Instala Git con `winget`

```powershell
winget install --id Git.Git -e --source winget
```

Espera a que termine la instalación y acepta los prompts si aparecen.

### 4. Cierra y vuelve a abrir PowerShell

Esto refresca las variables de entorno, como `PATH`.

### 5. Verifica que Git quedó instalado

```powershell
git --version
```

## macOS

### 1. Abre Terminal

En Spotlight (`Cmd + Espacio`) busca `Terminal` y ábrelo.

### 2. Comprueba si Git ya está instalado

```bash
git --version
```

Si ves un número de versión, ya lo tienes. En macOS reciente, este comando puede ofrecer instalar las **herramientas de línea de comandos de Xcode**; acepta e espera a que termine.

### 3. Instala Git (si hace falta)

**Opción A — herramientas de Xcode** (oficial de Apple):

```bash
xcode-select --install
```

**Opción B — Homebrew** (suele traer una versión más reciente):

```bash
brew install git
```

Si no tienes Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git
```

### 4. Verifica la instalación

```bash
git --version
```

## Linux

### 1. Abre una terminal

Usa la terminal de tu distribución (por ejemplo Terminal, Konsole o GNOME Terminal).

### 2. Instala Git con el gestor de paquetes

**Debian / Ubuntu / Linux Mint:**

```bash
sudo apt update
sudo apt install git
```

**Fedora:**

```bash
sudo dnf install git
```

**Arch Linux / Manjaro:**

```bash
sudo pacman -S git
```

### 3. Verifica la instalación

```bash
git --version
```

## Configura tu identidad global de Git

Este paso es el mismo en los tres sistemas. Es opcional, pero recomendado para registrar tu nombre y correo en los commits.

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
```

En Windows, puedes ejecutar los mismos comandos en PowerShell.

## Revisa la configuración

```bash
git config --list
```
