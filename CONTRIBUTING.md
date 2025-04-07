# Contributing to EvalScope

Thank you for considering contributing to EvalScope! We welcome contributions of all kinds, including bug fixes, new features, documentation, and more.

## Getting Started

1. **Fork the Repository**: Click the "Fork" button on the top right of this page to create a copy of this repository on your GitHub account.

2. **Clone Your Fork**: Clone your forked repository to your local machine using:
   ```bash
   git clone https://github.com/your-username/EvalScope.git
   ```
   Replace `your-username` with your GitHub username.

3. **Create a Branch**: Create a new branch for your changes. Use a descriptive name for your branch (e.g., `feature/new-feature`, `bugfix/issue-123`).
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Set Up Environment**: Follow the setup instructions in the `README.md` or `SETUP.md` to get the project up and running on your local machine.

## Making Changes

1. **Code Style**: Ensure your code follows the project's coding guidelines. If applicable, run the linter to check for any style issues.

2. **Pre-commit Hooks**: This project uses `pre-commit` hooks to maintain code quality. Make sure you have `pre-commit` installed and set up in your environment. Run the following commands to install the hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
   Before making a commit, you can manually run all pre-commit checks with:
   ```bash
   pre-commit run --all-files
   ```

3. **Testing**: Write tests to cover your changes. Run all tests to ensure nothing else is broken.

4. **Commit Changes**: Make sure your commit messages are clear and descriptive. Each commit should represent a single logical change.

5. **Push Changes**: Push your changes to your forked repository.
   ```bash
   git push origin feature/your-feature-name
   ```

## Creating a Pull Request

1. **Navigate to the Original Repository**: Go to the original repository where you want to submit your changes.

2. **Create a Pull Request**: Click the "New Pull Request" button. Ensure that you are comparing your branch from your fork against the correct branch in the original repository.

3. **Fill Out the Pull Request Template**: Provide a clear description of your changes, including any relevant issue numbers and a summary of the changes.

4. **Respond to Feedback**: Be ready to make adjustments as reviewers comment on your pull request. Engage in discussions to clarify any concerns.

Thank you for your contribution!
