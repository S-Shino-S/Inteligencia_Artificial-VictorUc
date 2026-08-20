# Instalación de Git en Windows con PowerShell

Guía rápida para instalar Git en Windows usando PowerShell.

## 1. Abre PowerShell como administrador

Desde el menú de inicio, busca `PowerShell`, haz clic derecho y selecciona **Ejecutar como administrador**.

## 2. Verifica si `winget` está disponible

```powershell
winget --version
```

## 3. Instala Git con `winget`

```powershell
winget install --id Git.Git -e --source winget
```

Espera a que termine la instalación y acepta los prompts si aparecen.

## 4. Cierra y vuelve a abrir PowerShell

Esto refresca las variables de entorno, como `PATH`.

## 5. Verifica que Git quedó instalado

```powershell
git --version
```

## 6. Configura tu identidad global de Git

Este paso es opcional, pero recomendado para registrar tu nombre y correo en los commits.

```powershell
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
```

## 7. Revisa la configuración

```powershell
git config --list
```