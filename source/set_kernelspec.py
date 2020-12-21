import argparse
import json


def set_kernel_spec(notebook_filepath, display_name, kernel_name):
    '''
    Parameters:
    ----------
    notebook_filepath: str
        Path to jupyter notebook

    display_name: str
        Display name to change the notebook to

    kernel_name: str
        kernel name to change the notebook to
    '''
    with open(notebook_filepath, "r") as openfile:
        notebook = json.load(openfile)
    kernel_spec = {"display_name": display_name, "language": "python", "name": kernel_name}
    if "metadata" not in notebook:
        notebook["metadata"] = {}
    notebook["metadata"]["kernelspec"] = kernel_spec
    with open(notebook_filepath, "w") as openfile:
        json.dump(notebook, openfile, indent=4)


if __name__ == "__main__":
    '''
    Helper script to set the display and kernel name for jupyter notebooks.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook")
    parser.add_argument("--display-name")
    parser.add_argument("--kernel")
    args = parser.parse_args()
    set_kernel_spec(args.notebook, args.display_name, args.kernel)
