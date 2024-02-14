# Setting Up the Project Environment

## Prerequisites

Ensure you have Conda installed on your system. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is sufficient for setting up and managing environments.

## Environment Setup

1. **Clone the Repository**: If you haven't already, clone the project repository to your local machine.
    TODO - fill in with the actual repository URL
    ```
    git clone https://github.com/your-username/project_name.git
    cd project_name
    ```

2. **Create the Conda Environment**: Use the `environment.yml` file to create a new Conda environment. This will install all necessary dependencies as specified in the file. This can take a few minutes to complete.

    ```
    conda env create -f environment.yml
    ```

3. **Activate the Environment**: Once the environment is successfully created, you can activate it using:

    ```
    conda activate gsl-bnn-mac-m2
    ```

4. **Verify the Environment**: To ensure everything is set up correctly, you can list all installed packages with:

    ```
    conda list
    ```
5. **VS Code Setup**: If you are using VS Code, you use Ctrl+Shift+P to select the Conda environment you just created.
6. **Adjust The Python Path**: Add the following to your `~/.bashrc`, `~\.zshrc`, or `~/.bash_profile` file to ensure that the project's code can be run from the command line.

    ```
    export PYTHONPATH=$PYTHONPATH:/path/to/project_name
    ```

    Replace `/path/to/project_name` with the actual path to the project directory, e.g. 'export PYTHONPATH=:/Users/your-name/projects/gsl-bnn'

You're now ready to run the project code. For specific instructions on running experiments or analyses, refer to the project's `README.md` or other documentation.
