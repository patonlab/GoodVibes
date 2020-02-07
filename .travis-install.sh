#!/usr/bin/env bash

if [[$TRAVIS_OS_NAME == 'osx']]; then
    brew update
    brew install pyenv-virtualenv
    case "${TOXENV}" in
        py35)
            pyenv install 3.5.2
            export PYENV_VERSION=3.5.2
            ;;
        py36)
            pyenv install 3.6.7
            export PYENV_VERSION=3.6.7
            ;;
        py37)
            pyenv install 3.7.2
            export PYENV_VERSION=3.7.2
            ;;
        py38)
            pyenv install 3.8.1
            export PYENV_VERSION=3.8.1
            ;;
    esac
    export PATH="/Users/travis/.pyenv/shims:${PATH}"
    pyenv-virtualenv venv
    source venv/bin/activate
    python3 --version
fi