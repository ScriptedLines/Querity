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

# 1. Determine Source (Local clone vs Remote Git repository)
SOURCE=""
if [ -f "pyproject.toml" ] && grep -q 'name = "querity"' pyproject.toml 2>/dev/null; then
    SOURCE="$(pwd)"
    echo -e "${CYAN}→${RESET} Installing from local source: ${BOLD}${SOURCE}${RESET}"
else
    REPO_URL="${QUERITY_REPO_URL:-https://github.com/$(git config --get remote.origin.url 2>/dev/null || echo '')}"
    if [ -z "${REPO_URL}" ] || [ "${REPO_URL}" = "https://github.com/" ]; then
        # Default fallback or user can specify QUERITY_REPO_URL
        REPO_URL="${QUERITY_REPO:-git+https://github.com/ScriptedLines/Querity.git}"
    fi
    SOURCE="${REPO_URL}"
    echo -e "${CYAN}→${RESET} Installing from repository: ${BOLD}${SOURCE}${RESET}"
fi

# 2. Check for uv (fastest & handles Python automatically)
if command -v uv >/dev/null 2>&1; then
    echo -e "${CYAN}→${RESET} Detected ${BOLD}uv${RESET}. Installing standalone tool..."
    uv tool install --force "${SOURCE}"
    INSTALLED_BIN="$(uv tool dir)/querity/bin/querity"
    if [ -f "${INSTALLED_BIN}" ]; then
        ln -sf "${INSTALLED_BIN}" "${BIN_DIR}/querity"
    fi
elif command -v pipx >/dev/null 2>&1; then
    echo -e "${CYAN}→${RESET} Detected ${BOLD}pipx${RESET}. Installing application..."
    pipx install --force "${SOURCE}"
else
    # 3. Fallback: Check Python 3.12+ or bootstrap uv
    PYTHON_BIN=""
    for cmd in python3.13 python3.12 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            if "$cmd" -c 'import sys; exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
                PYTHON_BIN="$(command -v "$cmd")"
                break
            fi
        fi
    done

    if [ -z "${PYTHON_BIN}" ]; then
        echo -e "${YELLOW}Python 3.12+ was not found on your system.${RESET}"
        echo -e "${CYAN}→${RESET} Bootstrapping lightweight ${BOLD}uv${RESET} to fetch Python 3.12 automatically..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
        if command -v uv >/dev/null 2>&1; then
            uv tool install --force "${SOURCE}"
        else
            echo -e "${RED}Error: Could not install uv or find Python >= 3.12.${RESET}"
            echo "Please install Python 3.12+ (or uv: https://docs.astral.sh/uv/) and re-run this script."
            exit 1
        fi
    else
        echo -e "${CYAN}→${RESET} Using Python: ${BOLD}${PYTHON_BIN}${RESET}"
        VENV_DIR="${INSTALL_DIR}/venv"
        echo -e "${CYAN}→${RESET} Creating isolated virtual environment at ${VENV_DIR}..."
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
        "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel --quiet
        echo -e "${CYAN}→${RESET} Installing Querity packages..."
        "${VENV_DIR}/bin/pip" install --upgrade "${SOURCE}" --quiet
        ln -sf "${VENV_DIR}/bin/querity" "${BIN_DIR}/querity"
    fi
fi

# 4. Check PATH
PATH_OK=false
case ":${PATH}:" in
    *:"${BIN_DIR}":*) PATH_OK=true ;;
esac

echo ""
echo -e "${GREEN}${BOLD}✓ Querity installed successfully!${RESET}"

if [ "${PATH_OK}" = false ]; then
    echo -e "\n${YELLOW}${BOLD}Notice:${RESET} ${BIN_DIR} is not in your PATH."
    echo "Add it to your shell configuration by running:"
    echo ""
    SHELL_NAME="$(basename "${SHELL:-bash}")"
    if [ "${SHELL_NAME}" = "zsh" ]; then
        echo -e "  ${BOLD}echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc && source ~/.zshrc${RESET}"
    elif [ "${SHELL_NAME}" = "fish" ]; then
        echo -e "  ${BOLD}fish_add_path ~/.local/bin${RESET}"
    else
        echo -e "  ${BOLD}echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc${RESET}"
    fi
    echo ""
    echo "Or launch directly using:"
    echo -e "  ${BOLD}${BIN_DIR}/querity${RESET}"
else
    echo -e "To launch the terminal UI anytime, simply type:"
    echo -e "\n  ${CYAN}${BOLD}querity${RESET}\n"
fi

