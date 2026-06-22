#!/bin/bash
mkdir -p "${HOME}/.streamlit"
printf '%s\n' '[browser]' 'gatherUsageStats = false' > "${HOME}/.streamlit/config.toml"