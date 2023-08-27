"""
This module defines the version of the package
"""

__version__ = '0.1.0'
VERSION = __version__
# TODO(desmondyuan): update 0.3.3
# * new function docstrings
# * remove the binder implementation
# * add basic tests

def get_msg() -> None:
    """Print the version history."""

    changelog = [
        """
        version 0.1.0
        -- Aug 24, 2023 --
        * First stable release of CellBox pytorch
        * Add pytest with a range of test cases
        """,
    ]
    print(
        "=" * 80 + '\n'
        "   _____     _ _ ____              \n"
        "  / ____|   | | |  _ \             \n"
        " | |     ___| | | |_) | _____  __  \n"
        " | |    / _ \ | |  _ < / _ \ \/ /  \n"
        " | |___|  __/ | | |_) | (_) >  <   \n"
        "  \_____\___|_|_|____/ \___/_/\_\  \n"
        "Running CellBox scripts developed in Sander lab\n"
        "Maintained by Bo Yuan, Judy Shen, and Augustin Luna; contributions by Phuc Nguyen"
    )

    print(changelog[-1])
    print(
        "Tutorials and documentations are available at https://github.com/Mustardburger/CellBox\n"
        "If you want to discuss the usage or to report a bug, please use the 'Issues' function at GitHub.\n"
        "If you find CellBox useful for your research, please consider citing the corresponding publication.\n"
        "For more information, please email us at boyuan@g.harvard.edu and c_shen@g.harvard.edu, "
        "augustin_luna@hms.harvard.edu\n",
        "-" * 80
    )
