import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Union

import joblib

from topicwizard.data import TopicData
from topicwizard.prepare.data import precompute_positions

IMPORTANT_PACKAGES = [
    "scikit-learn",
    "topic-wizard",
    "joblib",
    "turftopic",
    "gensim",
]

DOCKERFILE_TEMPLATE = """
FROM python:{python_version}-slim-bullseye

RUN apt update
RUN apt install -y build-essential

RUN pip install gunicorn==20.1.0
RUN pip install typing-extensions
{package_installs}

RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \\
	PATH=/home/user/.local/bin:$PATH

COPY --chown=user . $HOME/app


RUN mkdir /home/user/numba_cache
RUN chmod 777 /home/user/numba_cache

ENV NUMBA_CACHE_DIR=/home/user/numba_cache

# Set the working directory to the user's home directory
WORKDIR $HOME/app
EXPOSE {port}
CMD gunicorn --timeout 0 -b 0.0.0.0:{port} --workers=2 --threads=4 --worker-class=gthread main:server
"""

MAINFILE_TEMPLATE = """
import joblib
import topicwizard

topic_data = joblib.load("topic_data.joblib")

app = topicwizard.get_dash_app(topic_data)
server = app.server

if __name__ == "__main__":
    app.run_server(debug=False, port={port})
"""


def easy_deploy(
    topic_data: TopicData,
    dest_dir: Union[str, Path],
    port: int = 7860,
    precompute: bool = True,
):
    """Prepares topic data for easy deployment in a Docker container.

    Parameters
    ----------
    topic_data: TopicData
        Topic data to deploy.
    dest_dir: Path
        Directory to save the deployment to.
    port: int, default 7860
        Port to deploy the app to. Default is for HuggingFace Spaces.
    precomputed: bool, default True
        Determined whether to precompute positions for the deployment.
        If you want to stay backwards compatible you should set this to False.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(exist_ok=True, parents=True)
    print("Precomputing positions")
    topic_data = precompute_positions(topic_data)
    package_installs = []
    print("Collecting package information")
    for package in IMPORTANT_PACKAGES:
        try:
            package_version = version(package)
            package_installs.append(f"RUN pip install {package}=={package_version}")
        except PackageNotFoundError as e:
            if package in ["topic-wizard", "joblib", "scikit-learn"]:
                raise e
            warnings.warn(
                "Package {package} not found, won't be included in deployment."
            )
            continue
    print("Saving deployment")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    with dest_path.joinpath("main.py").open("w") as main_file:
        main_file.write(MAINFILE_TEMPLATE.format(port=port))
    with dest_path.joinpath("Dockerfile").open("w") as docker_file:
        docker_file.write(
            DOCKERFILE_TEMPLATE.format(
                python_version=python_version,
                package_installs="\n".join(package_installs),
                port=port,
            )
        )
    joblib.dump(topic_data, dest_path.joinpath("topic_data.joblib"))
