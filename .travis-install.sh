#!/usr/bin/env bash

if [[$TRAVIS_OS_NAME == 'osx']]; then
    brew update
    brew install pyenv-virtualenv
    case "${TOXENV}" in
        py37)
            pyenv install 3.7.2
            export PYENV_VERSION=3.7.2
            ;;
        py38)
            pyenv install 3.8.1
            export PYENV_VERSION=3.8.1
            ;;
        py39)
            pyenv install 3.9
            export PYENV_VERSION=3.9
            ;;
        py310)
            pyenv install 3.10
            export PYENV_VERSION=3.10
            ;;
        py311)
            pyenv install 3.11
            export PYENV_VERSION=3.11
            ;;
    esac
    export PATH="/Users/travis/.pyenv/shims:${PATH}"
    pyenv-virtualenv venv
    source venv/bin/activate
    python3 --version
fi
