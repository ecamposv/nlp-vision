# Tema-06: Chatbot con Rasa Open Source

Este tema incluye **dos versiones** del chatbot para poder compararlas en clase:

| Carpeta | Descripción | Costo | Internet |
|---|---|---|---|
| [01-chatbot-base/](01-chatbot-base/) | Chatbot clásico de Rasa (intents, stories, rules). Sin LLM. | Gratis | No |
| [02-chatbot-llm/](02-chatbot-llm/) | Mismo chatbot + acción personalizada que delega a un **LLM local con Ollama**. | Gratis | No (offline tras descargar el modelo) |

Ambas versiones comparten el mismo enfoque pedagógico; cambia únicamente cómo se generan las respuestas cuando no hay una regla específica que aplique.

---

## 1) Requisitos previos

- **Python 3.10** (recomendado para compatibilidad con Rasa 3.6)
- pip actualizado
- Solo para la versión con LLM: [Ollama](https://ollama.com) instalado localmente

Verifica Python:

```bash
python3 --version
```

## 2) Crear un environment nuevo de Python

Puedes elegir entre **venv** (Python estándar) o **Anaconda/Miniconda**. Solo necesitas uno.

> Recomendación: usa **un environment distinto por versión** (`rasa-tema06-base` y `rasa-tema06-llm`) para evitar mezclar dependencias.

### Opción A: venv

Entra al folder de la versión que quieras correr (`01-chatbot-base` o `02-chatbot-llm`) y ejecuta:

#### macOS / Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

#### Windows (PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### Opción B: Anaconda / Miniconda

Requiere tener instalado [Anaconda](https://www.anaconda.com/download) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda --version
```

Crear y activar el environment (cambia el nombre según la versión):

```bash
# Versión base
conda create -n rasa-tema06-base python=3.10 -y
conda activate rasa-tema06-base

# Versión con LLM
conda create -n rasa-tema06-llm python=3.10 -y
conda activate rasa-tema06-llm
```

Comandos útiles:

```bash
conda deactivate
conda remove -n rasa-tema06-base --all -y
conda remove -n rasa-tema06-llm --all -y
```

## 3) Instalar dependencias

Con el environment activado, dentro del folder de la versión:

```bash
pip install -r requirements.txt
```

---

## 4) Versión base (sin LLM) — [01-chatbot-base/](01-chatbot-base/)

```bash
cd 01-chatbot-base
rasa train
rasa shell
```

Mensajes de prueba: `hola`, `de qué trata este curso`, `eres un bot`, `adiós`.

Otras opciones:

```bash
rasa run --enable-api --cors "*"   # servidor HTTP
rasa test                          # pruebas de historias
```

## 5) Versión con LLM local — [02-chatbot-llm/](02-chatbot-llm/)

Esta versión usa la misma base, pero agrega una **acción personalizada** (`action_llm_responder`) que llama a un modelo servido por **Ollama**. Se dispara en dos casos:

- Cuando el usuario hace una pregunta libre (intent `preguntar_llm`).
- Como **fallback** cuando Rasa no entiende el mensaje con suficiente confianza.

### 5.1) Instalar Ollama y descargar un modelo

Tienes tres opciones para instalar Ollama. Solo necesitas una.

#### Opción A: Homebrew (macOS)

```bash
brew install ollama
```

> Si Homebrew te marca errores de kegs corruptos (`Error: ... is not a valid keg`), usa la Opción B o C.

#### Opción B: Script oficial (Linux y macOS)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Opción C: Descargar la app desde el sitio oficial (macOS y Windows)

1. Ve a [https://ollama.com/download](https://ollama.com/download) y descarga el instalador para tu sistema operativo.
2. En macOS: abre el `.dmg` y arrastra **Ollama** a la carpeta **Aplicaciones**.
3. **Abre la app una vez** desde Launchpad o Aplicaciones. macOS pedirá permiso porque viene de internet → **Abrir**. Este primer arranque instala el comando `ollama` en tu PATH (`/usr/local/bin/ollama`) y deja el servicio corriendo en segundo plano (verás un ícono de llama en la barra de menú).
4. **Cierra y reabre tu terminal** (o abre una pestaña nueva) para que el PATH se refresque.
5. Verifica:

   ```bash
   which ollama
   ollama --version
   ```

   Si el comando no aparece, crea el symlink manualmente:

   ```bash
   sudo ln -sf /Applications/Ollama.app/Contents/Resources/ollama /usr/local/bin/ollama
   ```

#### Levantar el servidor y descargar un modelo

Con la app de macOS (Opción C), el servidor ya queda corriendo automáticamente en `http://localhost:11434`. Con las Opciones A o B necesitas levantarlo manualmente:

```bash
ollama serve            # solo Opciones A y B; deja corriendo en http://localhost:11434
```

En otra terminal descarga el modelo (puedes elegir otro como `qwen2.5`, `mistral`, `phi3`):

```bash
ollama pull llama3.2
```

Verifica que responde:

```bash
ollama run llama3.2 "Hola, preséntate brevemente en español."
curl http://localhost:11434/api/tags
```

### 5.2) Entrenar el bot

```bash
cd 02-chatbot-llm
rasa train
```

### 5.3) Correr el chatbot

Necesitas que estén activos:

1. **Ollama** (servidor del LLM en el puerto 11434).
   - Si instalaste la app de macOS (Opción C), ya está corriendo automáticamente: verifica el ícono de llama en la barra de menú.
   - Si usaste Homebrew o el script (Opciones A/B), levántalo manualmente: `ollama serve`.
2. **Servidor de acciones de Rasa** (puerto 5055).
3. **Interfaz de chat** (`rasa shell`).

Abre las terminales que necesites según tu caso:

```bash
# Solo Opciones A/B — Terminal de Ollama
ollama serve

# Terminal del servidor de acciones
cd 02-chatbot-llm
rasa run actions

# Terminal de la interfaz de chat
cd 02-chatbot-llm
rasa shell
```

Mensajes para probar el LLM:
- `explícame qué es un transformer`
- `dame una definición de NLP`
- Cualquier frase que el bot no entienda (caerá al fallback con LLM).

### 5.4) Configuración opcional

La acción lee variables de entorno (ver [02-chatbot-llm/actions/actions.py](02-chatbot-llm/actions/actions.py)):

```bash
export OLLAMA_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="llama3.2"
```

---

## 6) Estructura del Tema-06

```text
Tema-06/
├── README.md                 ← este archivo (índice)
├── 01-chatbot-base/
│   ├── config.yml
│   ├── credentials.yml
│   ├── domain.yml
│   ├── endpoints.yml
│   ├── requirements.txt
│   ├── data/
│   │   ├── nlu.yml
│   │   ├── rules.yml
│   │   └── stories.yml
│   └── tests/
│       └── test_stories.yml
└── 02-chatbot-llm/
    ├── config.yml
    ├── credentials.yml
    ├── domain.yml
    ├── endpoints.yml          ← habilita action_endpoint
    ├── requirements.txt       ← + rasa-sdk + requests
    ├── actions/
    │   ├── __init__.py
    │   └── actions.py         ← acción que llama a Ollama
    ├── data/
    │   ├── nlu.yml            ← + intent preguntar_llm
    │   ├── rules.yml          ← + fallback con LLM
    │   └── stories.yml
    └── tests/
        └── test_stories.yml
```

## 7) Diferencias clave entre las dos versiones

| Aspecto | 01-chatbot-base | 02-chatbot-llm |
|---|---|---|
| Acciones personalizadas | No | Sí (`action_llm_responder`) |
| `endpoints.yml` con `action_endpoint` | Comentado | Activo |
| Dependencias extra | — | `rasa-sdk`, `requests` |
| Fallback ante mensajes desconocidos | `utter_no_entendi` | LLM responde |
| Requiere Ollama corriendo | No | Sí |
| Procesos al ejecutar | 1 (`rasa shell`) | 3 (`ollama serve`, `rasa run actions`, `rasa shell`) |
