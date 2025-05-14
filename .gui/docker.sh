#!/bin/bash

LOCAL_PORT=""
SHARES_PATH=""

help_message() {
	echo "Usage: docker.sh [OPTIONS]"
	echo "Options:"
	echo "  --local-port       Local port to bind for the GUI (required)"
	echo "  --shares-path      Path to local directory for /var/www/html/shares (required)"
	echo "  --help             Show this help message"
}

while [[ "$#" -gt 0 ]]; do
	case $1 in
		--local-port)
			LOCAL_PORT="$2"
			shift
			;;
		--shares-path)
			SHARES_PATH="$2"
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

if [[ -z $SHARES_PATH ]]; then
	echo "Error: Missing required parameter --shares-path. Use --help for usage."
	exit 1
fi

if [[ ! -d "$SHARES_PATH" ]]; then
	echo "Info: Directory '$SHARES_PATH' does not exist. Creating it..."
	mkdir -p "$SHARES_PATH" || {
		echo "Failed to create directory '$SHARES_PATH'"
			exit 2
		}
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

cat > docker-compose.yml <<EOF
services:
  php-web:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "$LOCAL_PORT:80"
    volumes:
      - ./:/var/www/html:rw
      - ${SHARES_PATH}:/var/www/html/shares:rw
    restart: unless-stopped
    environment:
      - APACHE_RUN_USER=www-data
      - APACHE_RUN_GROUP=www-data
      - LOCAL_UID=${UID}
      - LOCAL_GID=${GID}
EOF

USER_NAME="$(whoami)"

if ! id -nG "$USER_NAME" | grep -qw docker; then
	if sudo usermod -aG docker "$USER_NAME"; then
		echo "User '$USER_NAME' added to 'docker' group."
		echo "restart for docker"
		exit 0
	else
		echo "Failed to add user '$USER_NAME' to 'docker' group." >&2
		exit 1
	fi
fi

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
	rm docker-compose.yml
	echo "Failed to build container"
	exit 254
}

docker_compose up -d || {
	rm docker-compose.yml
	echo "Failed to start container"
	exit 255
}

docker_compose exec php-web chown -R www-data:www-data /var/www/html || {
	echo "Failed to set ownership inside container"
	exit 256
}

docker_compose exec php-web find /var/www/html -type f -exec chmod 644 {} \; || {
	echo "Failed to chmod files inside container"
	exit 258
}

rm docker-compose.yml
