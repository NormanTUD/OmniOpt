#!/bin/bash

LOCAL_PORT=""

help_message() {
	echo "Usage: docker.sh [OPTIONS]"
	echo "Options:"
	echo "  --local-port       Local port to bind for the GUI"
	echo "  --help             Show this help message"
}

while [[ "$#" -gt 0 ]]; do
	case $1 in
		--local-port)
			LOCAL_PORT="$2"
			shift
			;;
		--help)
			help_message
			exit 0
			;;
		*)
			echo "Error: Unknown option '$1'. Use --help for usage."
			exit 1
			;;
	esac
	shift
done

if [[ -z $LOCAL_PORT ]]; then
	echo "Error: Missing required parameter --local-port. Use --help for usage."
	exit 1
fi


is_package_installed() {
	dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -c "ok installed"
}

UPDATED_PACKAGES=0

if ! command -v curl &>/dev/null; then
	if [[ $UPDATED_PACKAGES == 0 ]]; then
		sudo apt update || {
			echo "apt-get update failed. Are you online?"
			exit 3
		}

		UPDATED_PACKAGES=1
	fi

	sudo apt-get install -y curl || {
		echo "sudo apt install -y curl failed"
		exit 3
	}
fi

if ! command -v docker &>/dev/null; then
	echo "Docker not found. Installing Docker..."

	curl -fsSL https://get.docker.com | sudo bash
fi

export LOCAL_PORT

echo "#!/bin/bash" > .env
echo "LOCAL_PORT=$LOCAL_PORT" >> .env

function docker_compose {
	if id -nG "$USER" | grep -qw docker; then
		DOCKER_CMD=""
	else
		DOCKER_CMD="sudo"
	fi

	if command -v docker-compose >/dev/null 2>&1; then
		$DOCKER_CMD docker-compose "$@"
	else
		$DOCKER_CMD docker compose "$@"
	fi
}

docker_compose build || {
	echo "Failed to build container"
	exit 254
}

docker_compose up -d || {
	echo "Failed to build container"
	exit 255
}
