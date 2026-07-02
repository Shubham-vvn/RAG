# Phase 1: Project Setup

## What was built?
Set up the base project structure, created a Python virtual environment, configured `.gitignore` and `.env` files, and installed the initial dependencies (`streamlit`, `pypdf2`, `langchain`, etc.).

## What problem does this solve?
Every software project needs a clean, reproducible, and isolated environment. Project setup prevents package conflicts between different projects on the same machine and manages sensitive credentials (like API keys) securely.

## Why is this phase required?
Without proper project setup, code would fail to run due to missing packages, dependency version mismatches, or exposed API keys. It establishes the foundations of codebase organization and reproducibility.

## Important concepts learned
- **Virtual Environment (`venv`)**: An isolated environment that allows you to install Python packages locally for a specific project without affecting the global Python installation.
- **Environment Variables (`.env`)**: A secure method to store configuration values and API keys separately from the codebase, avoiding security leaks when pushing to git.
- **Dependency Management**: Tracking package versions in `requirements.txt` to ensure consistent execution across different environments.

## Interview questions
1. **Why do we use a virtual environment instead of installing packages globally?**
   - *Answer*: Global installations can lead to dependency conflicts when different projects require different versions of the same package. Virtual environments isolate package spaces so each project has exactly the packages and versions it needs.
2. **Why is it critical to add `.env` to `.gitignore`?**
   - *Answer*: The `.env` file contains sensitive secrets like API keys. If pushed to a public repository (e.g., GitHub), anyone can steal the credentials, leading to security breaches, financial costs, and abuse of API services.
3. **What is the purpose of `requirements.txt`?**
   - *Answer*: It lists all packages and version constraints required to run the project. This allows anyone (or any deployment system) to reconstruct the exact environment by running `pip install -r requirements.txt`.

## Common mistakes
- **Committing `.env` to Git**: Accidentally committing secrets because `.gitignore` was not configured before the initial git commit.
- **Not activating the virtual environment**: Running `pip install` globally instead of inside the activated `venv`.
- **Hardcoding values**: Writing API keys directly into Python files instead of using `os.getenv` or `python-dotenv`.

## Key takeaway
A solid project foundation requires package isolation and secure configuration practices before writing any application logic.
