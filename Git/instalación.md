Aquí tienes una guía rápida en PowerShell para instalar Git en Windows:

Abre PowerShell como administrador

Inicio → busca “PowerShell” → clic derecho → Ejecutar como administrador.
Verifica si ya tienes winget disponible

PowerShell
winget --version
Instala Git con winget

PowerShell
winget install --id Git.Git -e --source winget
Espera a que termine la instalación

Acepta prompts si aparecen.
Cierra y vuelve a abrir PowerShell

Para refrescar variables de entorno (PATH).
Verifica que Git quedó instalado

PowerShell
git --version
(Opcional) Configura tu identidad global de Git

PowerShell
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
(Opcional) Revisa la configuración

PowerShell
git config --list