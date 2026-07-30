#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_PATH="$PROJ_ROOT/venv"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

check_environment() {
    print_header "Checking Environment"
    
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found at $VENV_PATH"
        print_warning "Create it with: python3 -m venv $VENV_PATH"
        exit 1
    fi
    
    print_success "Virtual environment found"
    
    source "$VENV_PATH/bin/activate"
    print_success "Virtual environment activated"
    
    python_version=$(python --version)
    print_success "Python: $python_version"
    
    cd "$SCRIPT_DIR"
    print_success "Working directory: $(pwd)"
}

install_dependencies() {
    print_header "Installing Dependencies"
    
    if [ -f "requirements.txt" ]; then
        pip install -q -r requirements.txt
        print_success "Dependencies installed"
    else
        print_warning "requirements.txt not found"
    fi
}

run_pipeline() {
    print_header "Running MIRACL Pipeline"
    
    local option=${1:-"full"}
    
    case $option in
        "full")
            print_warning "Full pipeline: download + preprocess + tasks"
            python miracl_pipeline.py --config config.json --verbose
            ;;
        "preprocess-only")
            print_warning "Preprocessing only (skipping download)"
            python miracl_pipeline.py --config config.json --skip-download --verbose
            ;;
        "tasks-only")
            print_warning "Tasks only (skipping download + preprocess)"
            python miracl_pipeline.py --config config.json --skip-download --skip-preprocess --verbose
            ;;
        *)
            print_error "Unknown option: $option"
            echo "Usage: $0 [full|preprocess-only|tasks-only]"
            exit 1
            ;;
    esac
}

show_results() {
    print_header "Pipeline Results"
    
    results_dir="../../results/miracl"
    
    if [ -f "$results_dir/pipeline_results.json" ]; then
        print_success "Pipeline results:"
        echo ""
        python -m json.tool "$results_dir/pipeline_results.json" | head -50
        echo ""
        print_warning "Full results: $results_dir/pipeline_results.json"
    fi
    
    if [ -d "$results_dir/logs" ]; then
        latest_log=$(ls -t "$results_dir/logs"/pipeline_*.log 2>/dev/null | head -1)
        if [ -f "$latest_log" ]; then
            print_warning "Latest log: $latest_log"
            echo ""
            tail -20 "$latest_log"
        fi
    fi
    
    echo ""
    print_success "Results directory: $results_dir"
    echo "  - processed/          (preprocessed data)"
    echo "  - splits/             (train/val/test splits)"
    echo "  - logs/               (execution logs)"
    echo "  - task*.jsonl         (pipeline outputs)"
}

main() {
    print_header "MIRACL Pipeline - RORetrieval"
    
    check_environment
    install_dependencies
    run_pipeline "${1:-full}"
    show_results
    
    print_success "Pipeline execution complete!"
}

main "$@"
