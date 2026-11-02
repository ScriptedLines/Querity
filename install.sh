#!/usr/bin/env bash
set -e

# ==============================================================================
# Querity One-Line Installer
# ==============================================================================
# Installs Querity into an isolated environment and links the `querity` command
# to ~/.local/bin/querity so you can launch it from anywhere.
# ==============================================================================

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${CYAN}${BOLD}"
cat << "EOF"
   ___                  _ _         
  / _ \ _   _  ___ _ __(_) |_ _   _ 
 | | | | | | |/ _ \ '__| | __| | | |
 | |_| | |_| |  __/ |  | | |_| |_| |
  \__\_\\__,_|\___|_|  |_|\__|\__, |
                              |___/ 
EOF
echo -e "${RESET}${BOLD}Guarded Text-to-SQL Agent & Terminal UI for PostgreSQL${RESET}\n"

INSTALL_DIR="${HOME}/.local/share/querity"
BIN_DIR="${HOME}/.local/bin"
mkdir -p "${INSTALL_DIR}" "${BIN_DIR}"

# 1. Determine OS and Source
IS_WINDOWS=false
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=true ;;
esac

SOURCE=""
if [ -f "pyproject.toml" ] && grep -q 'name = "querity"' pyproject.toml 2>/dev/null; then
    SOURCE="$(pwd)"
    echo -e "${CYAN}→${RESET} Installing from local source: ${BOLD}${SOURCE}${RESET}"
else
    REPO_URL="${QUERITY_REPO_URL:-${QUERITY_REPO:-git+https://github.com/ScriptedLines/Querity.git}}"
    SOURCE="${REPO_URL}"
    echo -e "${CYAN}→${RESET} Installing from repository: ${BOLD}${SOURCE}${RESET}"
fi

# 2. Resolve or bootstrap modern runner (uv / pipx)
UV_BIN=""
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [ -x "${HOME}/.local/bin/uv" ]; then
    UV_BIN="${HOME}/.local/bin/uv"
    export PATH="${HOME}/.local/bin:${PATH}"
elif [ -x "${HOME}/.local/bin/uv.exe" ]; then
    UV_BIN="${HOME}/.local/bin/uv.exe"
    export PATH="${HOME}/.local/bin:${PATH}"
elif [ -x "${HOME}/.cargo/bin/uv" ]; then
    UV_BIN="${HOME}/.cargo/bin/uv"
    export PATH="${HOME}/.cargo/bin:${PATH}"
elif [ -x "${HOME}/.cargo/bin/uv.exe" ]; then
    UV_BIN="${HOME}/.cargo/bin/uv.exe"
    export PATH="${HOME}/.cargo/bin:${PATH}"
fi

# If neither uv nor pipx is found, attempt to bootstrap uv
if [ -z "${UV_BIN}" ] && ! command -v pipx >/dev/null 2>&1; then
    echo -e "${CYAN}→${RESET} Bootstrapping lightweight package runner (${BOLD}uv${RESET})..."
    if [ "${IS_WINDOWS}" = true ] && command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex" >/dev/null 2>&1 || true
    else
        curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1 || true
    fi

    if [ -x "${HOME}/.local/bin/uv" ]; then
        UV_BIN="${HOME}/.local/bin/uv"
        export PATH="${HOME}/.local/bin:${PATH}"
    elif [ -x "${HOME}/.local/bin/uv.exe" ]; then
        UV_BIN="${HOME}/.local/bin/uv.exe"
        export PATH="${HOME}/.local/bin:${PATH}"
    elif [ -x "${HOME}/.cargo/bin/uv" ]; then
        UV_BIN="${HOME}/.cargo/bin/uv"
        export PATH="${HOME}/.cargo/bin:${PATH}"
    fi
fi

if [ -n "${UV_BIN}" ]; then
    echo -e "${CYAN}→${RESET} Installing with ${BOLD}uv${RESET}..."
    "${UV_BIN}" tool install --force "${SOURCE}"
    
    # Ensure binary symlink or executable in ~/.local/bin
    if [ ! -f "${BIN_DIR}/querity" ] && [ ! -f "${BIN_DIR}/querity.exe" ]; then
        UV_TOOL_DIR="$("${UV_BIN}" tool dir 2>/dev/null || echo "${HOME}/.local/share/uv/tools")"
        if [ -f "${UV_TOOL_DIR}/querity/bin/querity" ]; then
            ln -sf "${UV_TOOL_DIR}/querity/bin/querity" "${BIN_DIR}/querity"
        elif [ -f "${UV_TOOL_DIR}/querity/Scripts/querity.exe" ]; then
            cp -f "${UV_TOOL_DIR}/querity/Scripts/querity.exe" "${BIN_DIR}/querity.exe" 2>/dev/null || true
        fi
    fi
elif command -v pipx >/dev/null 2>&1; then
    echo -e "${CYAN}→${RESET} Installing with ${BOLD}pipx${RESET}..."
    pipx install --force "${SOURCE}"
else
    # 3. Fallback: Standalone Python 3.12+ virtualenv
    PYTHON_BIN=""
    for cmd in python3.13 python3.12 python3 python py; do
        if command -v "$cmd" >/dev/null 2>&1; then
            if "$cmd" -c 'import sys; exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
                PYTHON_BIN="$(command -v "$cmd")"
                break
            fi
        fi
    done

    if [ -z "${PYTHON_BIN}" ]; then
        echo -e "${RED}Error: Python 3.12 or newer is required to install Querity.${RESET}"
        echo "Please install Python 3.12+ (or uv: https://docs.astral.sh/uv/) and re-run this script."
        exit 1
    fi

    echo -e "${CYAN}→${RESET} Using Python: ${BOLD}${PYTHON_BIN}${RESET}"
    VENV_DIR="${INSTALL_DIR}/venv"
    echo -e "${CYAN}→${RESET} Creating clean virtual environment at ${VENV_DIR}..."

    # Wipe any broken or incomplete virtualenv from previous runs
    rm -rf "${VENV_DIR}"

    # Create venv; if ensurepip is unavailable in host python, retry with --without-pip
    if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}" >/dev/null 2>&1; then
        echo -e "${CYAN}→${RESET} Creating virtual environment without bundled pip..."
        rm -rf "${VENV_DIR}"
        "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"
    fi

    # Determine internal venv bin/scripts layout
    if [ -d "${VENV_DIR}/Scripts" ]; then
        VENV_BIN="${VENV_DIR}/Scripts"
    else
        VENV_BIN="${VENV_DIR}/bin"
    fi

    if [ -x "${VENV_BIN}/python.exe" ]; then
        VENV_PYTHON="${VENV_BIN}/python.exe"
    elif [ -x "${VENV_BIN}/python" ]; then
        VENV_PYTHON="${VENV_BIN}/python"
    elif [ -x "${VENV_BIN}/python3.exe" ]; then
        VENV_PYTHON="${VENV_BIN}/python3.exe"
    else
        VENV_PYTHON="${VENV_BIN}/python3"
    fi

    if [ ! -x "${VENV_PYTHON}" ]; then
        echo -e "${RED}Error: Failed to create virtual environment.${RESET}"
        exit 1
    fi

    # Bootstrap pip inside the venv if missing
    if ! "${VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
        echo -e "${CYAN}→${RESET} Initializing pip inside virtual environment..."
        "${VENV_PYTHON}" -m ensurepip --default-pip >/dev/null 2>&1 || \
        (curl -fsSL https://bootstrap.pypa.io/get-pip.py 2>/dev/null | "${VENV_PYTHON}" - --quiet) || true
    fi

    if ! "${VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
        echo -e "${RED}Error: Could not initialize pip in ${VENV_DIR}.${RESET}"
        echo "Please install the python3-venv package or uv (https://docs.astral.sh/uv/)."
        exit 1
    fi

    echo -e "${CYAN}→${RESET} Upgrading packaging dependencies..."
    "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel --quiet

    echo -e "${CYAN}→${RESET} Installing Querity packages..."
    "${VENV_PYTHON}" -m pip install --upgrade "${SOURCE}"

    if [ -f "${VENV_BIN}/querity.exe" ]; then
        cp -f "${VENV_BIN}/querity.exe" "${BIN_DIR}/querity.exe" 2>/dev/null || ln -sf "${VENV_BIN}/querity.exe" "${BIN_DIR}/querity.exe"
    else
        ln -sf "${VENV_BIN}/querity" "${BIN_DIR}/querity"
    fi
fi

# Ensure executable wrappers for both Unix and Windows
if [ -f "${BIN_DIR}/querity.exe" ]; then
    if [ ! -f "${BIN_DIR}/querity" ]; then
        cat << 'EOF' > "${BIN_DIR}/querity"
#!/usr/bin/env sh
exec "$(dirname "$0")/querity.exe" "$@"
EOF
        chmod +x "${BIN_DIR}/querity"
    fi
    if [ ! -f "${BIN_DIR}/querity.cmd" ]; then
        cat << 'EOF' > "${BIN_DIR}/querity.cmd"
@echo off
"%~dp0querity.exe" %*
EOF
    fi
fi

# 4. System Integration & PATH Setup
if [ ! -f "${BIN_DIR}/querity" ] && [ ! -L "${BIN_DIR}/querity" ]; then
    echo -e "${RED}Error: Querity binary was not created at ${BIN_DIR}/querity.${RESET}"
    exit 1
fi

chmod +x "${BIN_DIR}/querity"

# Check if ~/.local/bin is already in PATH
PATH_IN_ENV=false
case ":${PATH}:" in
    *:"${BIN_DIR}":*) PATH_IN_ENV=true ;;
esac

# Try linking to /usr/local/bin if writable and not already linked
LINKED_GLOBAL=false
if [ -w "/usr/local/bin" ] && [ ! -e "/usr/local/bin/querity" ]; then
    if ln -sf "${BIN_DIR}/querity" "/usr/local/bin/querity" 2>/dev/null; then
        LINKED_GLOBAL=true
    fi
elif [ -w "/usr/local/bin/querity" ]; then
    ln -sf "${BIN_DIR}/querity" "/usr/local/bin/querity" 2>/dev/null && LINKED_GLOBAL=true
fi

# Detect shell and target profile for persistent PATH addition
CONFIG_FILE=""
SHELL_NAME="$(basename "${SHELL:-bash}")"
case "${SHELL_NAME}" in
    zsh)
        CONFIG_FILE="${HOME}/.zshrc"
        ;;
    bash)
        if [ -f "${HOME}/.bash_profile" ]; then
            CONFIG_FILE="${HOME}/.bash_profile"
        elif [ -f "${HOME}/.bashrc" ]; then
            CONFIG_FILE="${HOME}/.bashrc"
        else
            CONFIG_FILE="${HOME}/.profile"
        fi
        ;;
    fish)
        CONFIG_FILE="${HOME}/.config/fish/config.fish"
        ;;
    *)
        CONFIG_FILE="${HOME}/.profile"
        ;;
esac

ADDED_TO_PROFILE=false
if [ -n "${CONFIG_FILE}" ] && [ -f "${CONFIG_FILE}" ]; then
    if ! grep -q '\.local/bin' "${CONFIG_FILE}" 2>/dev/null; then
        echo "" >> "${CONFIG_FILE}"
        echo '# Added by Querity installer' >> "${CONFIG_FILE}"
        if [ "${SHELL_NAME}" = "fish" ]; then
            echo 'fish_add_path ~/.local/bin' >> "${CONFIG_FILE}"
        else
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${CONFIG_FILE}"
        fi
        ADDED_TO_PROFILE=true
    fi
elif [ -n "${CONFIG_FILE}" ] && [ ! -f "${CONFIG_FILE}" ] && [ "${SHELL_NAME}" = "zsh" ]; then
    echo '# Added by Querity installer' > "${CONFIG_FILE}"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${CONFIG_FILE}"
    ADDED_TO_PROFILE=true
fi

# If on Windows, also persist to Windows User environment variables
if [ "${IS_WINDOWS}" = true ] && command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -Command "
        \$bin = [System.IO.Path]::GetFullPath(\"$BIN_DIR\");
        \$p = [Environment]::GetEnvironmentVariable('Path', 'User');
        if (\$p -notlike \"*\$bin*\") {
            [Environment]::SetEnvironmentVariable('Path', \$p + ';' + \$bin, 'User');
        }
    " >/dev/null 2>&1 || true
fi

echo ""
echo -e "${GREEN}${BOLD}Querity installed successfully!${RESET}"

if [ "${PATH_IN_ENV}" = true ] || [ "${LINKED_GLOBAL}" = true ]; then
    echo -e "To launch the terminal UI anytime, simply type:"
    echo -e "\n  ${CYAN}${BOLD}querity${RESET}\n"
else
    echo -e "\n${YELLOW}${BOLD}Notice:${RESET} ${BIN_DIR} is not yet loaded in this active shell session."
    if [ "${ADDED_TO_PROFILE}" = true ]; then
        echo -e "We have added it to ${BOLD}${CONFIG_FILE}${RESET} for future sessions."
    fi
    echo -e "To use ${BOLD}querity${RESET} immediately in this window, run:"
    if [ "${SHELL_NAME}" = "fish" ]; then
        echo -e "\n  ${CYAN}${BOLD}fish_add_path ~/.local/bin && querity${RESET}\n"
    else
        echo -e "\n  ${CYAN}${BOLD}export PATH=\"\$HOME/.local/bin:\$PATH\" && querity${RESET}\n"
    fi
    echo "Or execute directly via:"
    echo -e "  ${BOLD}${BIN_DIR}/querity${RESET}\n"
fi


